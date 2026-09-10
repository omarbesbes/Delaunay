"""How does each method's running time scale with the number of points?

    python scaling_study.py --ply data/*.ply --json results/scaling.json --plot scaling.png
    python scaling_study.py --plot-only results/scaling.json --plot scaling.png

Measures *time only* -- correctness is what test_delaunay_surfaces.py is for -- so no reference
triangulation is computed unless CGAL is one of the methods being timed.  For every
(point cloud, size, jitter, method) it does one untimed warm-up run and then `--repeats` timed
runs, and writes one record per measurement to the JSON after every size, so a job that is cut
short still leaves a usable curve.

Sizes above the cloud's own point count are built by **tiling**: the cloud is replicated on a
k x k x k lattice of translated copies (the point spacing, and therefore the local structure and
the degeneracies, are preserved; the extent grows) and the requested number of points is drawn
from that.  Records say how many tiles were used, and the plot marks tiled sizes, because they are
not the same input distribution as a subsample.

The plot is log-log, one panel per cloud, solid without jitter and dashed with it, and the legend
carries the fitted exponent alpha of t ~ N^alpha (least squares on log t vs log N over the
measured range).  alpha ~ 1 is linear scaling, alpha ~ 1.33 is the classic 3D Delaunay
worst case for surface-like input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

import numpy as np

import test_delaunay_surfaces as T

ALL_METHODS = [
    "paragram",
    "gdel3d",
    "gstar4d",
    "dewall",
    "geodel",
    "cgal_parallel",
    "cgal_sequential",
]
LABELS = {
    "paragram": "Paragram + conversion",
    "gdel3d": "gDel3D",
    "gstar4d": "gStar4D",
    "dewall": "Local DeWall",
    "geodel": "GeoDel (CPU, MT)",
    "cgal_parallel": "CGAL parallel",
    "cgal_sequential": "CGAL 1 thread",
}
COLORS = {
    "paragram": "#4477aa",
    "gdel3d": "#228833",
    "gstar4d": "#ccbb44",
    "dewall": "#aa3377",
    "geodel": "#66ccee",
    "cgal_parallel": "#ee6677",
    "cgal_sequential": "#bbbbbb",
}


# ----------------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------------


def parse_sizes(specs: list[str]) -> list[int]:
    """Accepts plain counts and `start:stop:step` ranges (stop included)."""
    out: set[int] = set()
    for spec in specs:
        if ":" in spec:
            a, b, c = (int(float(x)) for x in spec.split(":"))
            out.update(range(a, b + 1, c))
        else:
            out.add(int(float(spec)))
    return sorted(out)


def points_at(
    cloud: np.ndarray, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """n points from the cloud: a random subsample, or a tiling of it when n exceeds its size."""
    if n <= len(cloud):
        return cloud[rng.choice(len(cloud), n, replace=False)], 1
    k = math.ceil((n / len(cloud)) ** (1 / 3))
    extent = np.ptp(cloud, axis=0)
    gap = extent / np.cbrt(len(cloud))  # about one point spacing, so tiles do not touch
    copies = [
        cloud + np.array([i, j, m]) * (extent + gap)
        for i in range(k)
        for j in range(k)
        for m in range(k)
    ]
    big = np.concatenate(copies)
    return big[rng.choice(len(big), n, replace=False)], k**3


def jittered(points: np.ndarray, jitter: float, seed: int) -> np.ndarray:
    if jitter <= 0:
        return points
    rng = np.random.default_rng([seed, len(points), int(jitter * 1e12)])
    return points + rng.normal(
        scale=jitter * np.ptp(points, axis=0).max(), size=points.shape
    )


# ----------------------------------------------------------------------------------------
# one measurement
# ----------------------------------------------------------------------------------------


def run_once(method: str, pts: np.ndarray, args) -> dict:
    """Run one method once.  Returns {"total", "gpu", "cpu", "tets", ...} in seconds."""
    if method == "paragram":
        import torch

        from paragram_repair import repair_failed_cells
        from voronoi_to_delaunay import delaunay_from_adjacency

        dev = torch.device(args.device)
        p_dev = torch.from_numpy(np.ascontiguousarray(pts, np.float32)).to(dev)

        def sync():
            if dev.type == "cuda":
                torch.cuda.synchronize()

        t0 = time.perf_counter()
        adjacency, offsets, status, _ = T.voronoi_adjacency(
            p_dev,
            "paragram" if dev.type == "cuda" else "qhull",
            None,
            bbox_pad=args.paragram_bbox_pad,
        )
        sync()
        t_adj = time.perf_counter() - t0
        t_repair = 0.0
        if args.repair == "on" and status is not None:
            adjacency, offsets, st = repair_failed_cells(
                p_dev, adjacency, offsets, status, include_hull=True
            )
            t_repair = st["seconds"]
        t0 = time.perf_counter()
        tets, _ = delaunay_from_adjacency(
            p_dev, adjacency, offsets, status, return_circumcentres=False
        )
        sync()
        t_conv = time.perf_counter() - t0
        n_tets = int(tets.shape[0])
        del adjacency, offsets, status, tets, p_dev
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "total": t_adj + t_repair + t_conv,
            "gpu": t_adj + t_conv,
            "cpu": t_repair,
            "adjacency": t_adj,
            "repair": t_repair,
            "conversion": t_conv,
            "tets": n_tets,
        }
    if method == "gdel3d":
        tets, secs, info = T.run_gdel3d(pts)
        return {
            "total": secs,
            "gpu": info.get("gpu_seconds"),
            "cpu": info.get("cpu_seconds"),
            "tets": len(tets),
        }
    if method == "gstar4d":
        tets, secs, info = T.run_gstar4d(
            pts,
            args.gstar4d_bin,
            grid_size=args.gstar4d_grid,
            timeout=args.tool_timeout or None,
            verbose=True,
        )
        return {
            "total": secs,
            "gpu": info.get("gpu_seconds"),
            "cpu": 0.0,
            "io": info.get("io_seconds"),
            "tets": len(tets),
            "loops": info.get("consistency_loops"),
            "missing_points": info.get("missing_points"),
        }
    if method == "dewall":
        in_unit = bool(pts.min() >= 0.0 and pts.max() < 1.0)
        tets, secs, info = T.run_dewall(
            pts,
            args.dewall_bin,
            prenormalized=in_unit,
            timeout=args.tool_timeout or None,
        )
        return {
            "total": secs,
            "gpu": info.get("gpu_seconds"),
            "cpu": 0.0,
            "io": info.get("io_seconds"),
            "tets": len(tets),
        }
    if method == "geodel":
        tets, secs, info = T.run_geodel(pts, nb_threads=args.threads)
        return {"total": secs, "gpu": 0.0, "cpu": secs, "tets": len(tets)}
    if method in ("cgal_parallel", "cgal_sequential"):
        env_before = os.environ.get("CGAL_THREADS")
        if method == "cgal_sequential":
            os.environ["CGAL_THREADS"] = "1"
        elif args.threads:
            os.environ["CGAL_THREADS"] = str(args.threads)
        try:
            tets, _, info = T.reference_delaunay(pts)
        finally:
            if env_before is None:
                os.environ.pop("CGAL_THREADS", None)
            else:
                os.environ["CGAL_THREADS"] = env_before
        return {
            "total": info["seconds"],
            "gpu": 0.0,
            "cpu": info["seconds"],
            "tets": len(tets),
        }
    raise ValueError(f"unknown method {method}")


def measure_in_child(method: str, pts: np.ndarray, args) -> dict:
    """Same as measure(), but in a separate interpreter with a hard wall-clock limit.

    gDel3D has segfaulted, aborted and hung on individual inputs of these clouds; none of that can
    be caught in-process, and a hang blocks every method queued behind it.  The points go through
    a temporary .npy file and the result comes back as JSON."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        npy, out = os.path.join(d, "points.npy"), os.path.join(d, "result.json")
        np.save(npy, pts)
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--child",
            method,
            "--child-npy",
            npy,
            "--child-out",
            out,
            "--repeats",
            str(args.repeats),
            "--slow-threshold",
            str(args.slow_threshold),
            "--threads",
            str(args.threads),
            "--device",
            args.device,
            "--paragram-bbox-pad",
            str(args.paragram_bbox_pad),
            "--repair",
            args.repair,
            "--gstar4d-bin",
            args.gstar4d_bin or "",
            "--gstar4d-grid",
            str(args.gstar4d_grid),
            "--dewall-bin",
            args.dewall_bin or "",
            "--cgal-bin",
            args.cgal_bin or "",
            "--tool-timeout",
            str(args.tool_timeout),
        ]
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=args.measure_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{method} did not finish within --measure-timeout "
                f"{args.measure_timeout:.0f}s and was killed (it hung inside the library)"
            ) from exc
        if r.returncode != 0 or not os.path.exists(out):
            sig = {134: "SIGABRT", 139: "SIGSEGV", -6: "SIGABRT", -11: "SIGSEGV"}
            why = sig.get(r.returncode, f"exit {r.returncode}")
            raise RuntimeError(
                f"{method} crashed the interpreter ({why}): {(r.stdout + r.stderr)[-300:]}"
            )
        with open(out) as fh:
            return json.load(fh)


def measure(method: str, pts: np.ndarray, args) -> dict:
    """Warm-up, then `--repeats` timed runs (fewer if the first is slow)."""
    if method in (args.isolate or []) and not args.child:
        return measure_in_child(method, pts, args)
    run_once(method, pts, args)  # warm-up: JIT, clocks, first-touch page faults
    reps = args.repeats
    first = run_once(method, pts, args)
    runs = [first]
    if first["total"] < args.slow_threshold:
        runs += [run_once(method, pts, args) for _ in range(reps - 1)]
    out = {
        "runs": len(runs),
        "seconds": statistics.fmean(r["total"] for r in runs),
        "min": min(r["total"] for r in runs),
        "std": statistics.pstdev([r["total"] for r in runs]) if len(runs) > 1 else 0.0,
        "tets": first.get("tets"),
    }
    for key in (
        "gpu",
        "cpu",
        "io",
        "adjacency",
        "repair",
        "conversion",
        "loops",
        "missing_points",
    ):
        vals = [r[key] for r in runs if r.get(key) is not None]
        if vals:
            out[key] = (
                statistics.fmean(vals) if isinstance(vals[0], (int, float)) else vals[0]
            )
    return out


# ----------------------------------------------------------------------------------------
# the sweep
# ----------------------------------------------------------------------------------------


def record_key(rec: dict) -> tuple:
    return (rec["cloud"], rec["n"], rec["jitter"], rec["method"])


def load_previous(path: str) -> list[dict]:
    """Records of an earlier run of this sweep.  A record still marked "running" belongs to a
    measurement whose process died -- a segfault inside one of the libraries kills the whole
    interpreter -- so it becomes an error and is not attempted again."""
    if not path or not os.path.exists(path):
        return []
    with open(path) as fh:
        old = json.load(fh).get("runs", [])
    for rec in old:
        if rec.pop("status", None) == "running":
            rec["error"] = "the process died during this measurement (segfault or kill)"
    return old


def sweep(args) -> dict:
    clouds = {}
    for path in args.ply:
        name = os.path.splitext(os.path.basename(path))[0]
        clouds[name] = T.unit_cube(T.load_ply_vertices(path))
        print(f"{name}: {len(clouds[name])} points", flush=True)
    sizes = parse_sizes(args.sizes)
    records: list[dict] = load_previous(args.json) if args.resume else []
    done = {record_key(r) for r in records}
    if records:
        print(f"resuming: {len(records)} measurement(s) already recorded", flush=True)
    env = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sizes": sizes,
        "jitters": args.jitter,
        "methods": args.methods,
        "repeats": args.repeats,
        "threads": args.threads,
        "gstar4d_grid": args.gstar4d_grid,
        "tool_timeout": args.tool_timeout,
        "clouds": {k: len(v) for k, v in clouds.items()},
        "argv": sys.argv[1:],
    }
    if args.device == "cuda":
        try:
            import torch

            env["gpu"] = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001, S110 - the device name is informational only
            pass

    def save():
        if args.json:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            with open(args.json, "w") as fh:
                json.dump({"_env": env, "runs": records}, fh, indent=1)

    give_up: set[tuple] = (
        set()
    )  # (cloud, jitter, method) that grew too slow to keep measuring
    fails: dict[tuple, int] = {}  # consecutive failures per (cloud, jitter, method)
    for cloud_name, cloud in clouds.items():
        rng = np.random.default_rng(args.seed)
        for n in sizes:
            base, tiles = points_at(cloud, n, rng)
            for jit in args.jitter:
                pts = jittered(base, jit, args.seed)
                tag = f"{cloud_name} n={n} jitter={jit:g}" + (
                    f" ({tiles} tiles)" if tiles > 1 else ""
                )
                for m in args.methods:
                    key = (cloud_name, jit, m)
                    rec = {
                        "cloud": cloud_name,
                        "n": n,
                        "tiles": tiles,
                        "jitter": jit,
                        "method": m,
                    }
                    if record_key(rec) in done:
                        continue
                    if key in give_up:
                        rec["error"] = (
                            "skipped (this method exceeded --skip-above or failed "
                            f"{args.give_up_after} times in a row at smaller sizes)"
                        )
                        records.append(rec)
                        save()
                        continue
                    # Mark the measurement before starting it: if the process dies inside a
                    # library (gDel3D can segfault), --resume turns the marker into an error and
                    # skips it instead of crashing again on the same input.
                    rec["status"] = "running"
                    records.append(rec)
                    save()
                    print(
                        f"[{time.strftime('%H:%M:%S')}] {tag} {m}", end="", flush=True
                    )
                    try:
                        rec.update(measure(m, pts, args))
                        rec.pop("status", None)
                        print(
                            f" -> {rec['seconds']:.3f}s ({rec['runs']} run(s))",
                            flush=True,
                        )
                        fails[key] = 0
                        if args.skip_above and rec["seconds"] > args.skip_above:
                            give_up.add(key)
                            print(
                                f"   {m}: {rec['seconds']:.1f}s > --skip-above {args.skip_above:g}s,"
                                " not measuring it at larger sizes",
                                flush=True,
                            )
                    except Exception as exc:  # noqa: BLE001 - a failing method must not stop the sweep
                        rec.pop("status", None)
                        rec["error"] = str(exc)[:400]
                        # A crash or a timeout is input-specific -- gDel3D crashes on one 22k
                        # subsample and is fine at 32k -- so keep going, and give up only after
                        # several failures in a row.
                        fails[key] = fails.get(key, 0) + 1
                        if fails[key] >= args.give_up_after:
                            give_up.add(key)
                            print(
                                f" -> FAILED {fails[key]}x in a row, not measuring it at larger "
                                f"sizes: {rec['error'][:120]}",
                                flush=True,
                            )
                        else:
                            print(f" -> FAILED: {rec['error'][:160]}", flush=True)
                    save()
    save()
    return {"_env": env, "runs": records}


# ----------------------------------------------------------------------------------------
# plot
# ----------------------------------------------------------------------------------------


def _log_axis(ax) -> None:
    """Readable point counts on a log axis: 2k, 10k, 100k, 1M, and no minor labels."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _: (
                f"{v / 1e6:g}M"
                if v >= 1e6
                else (f"{v / 1e3:g}k" if v >= 1e3 else f"{v:g}")
            )
        )
    )
    ax.tick_params(labelsize=8)


def fit_exponent(ns, ts) -> float | None:
    """Least-squares slope of log t vs log N, i.e. the alpha of t ~ N^alpha."""
    pts = [(math.log(n), math.log(t)) for n, t in zip(ns, ts) if n > 0 and t > 0]
    if len(pts) < 3:
        return None
    xs, ys = zip(*pts)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


def plot(payload: dict, path: str, min_seconds: float = 2e-3) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [r for r in payload["runs"] if r.get("seconds")]
    clouds = sorted({r["cloud"] for r in runs})
    jitters = sorted({r["jitter"] for r in runs})
    if not clouds:
        print("nothing to plot")
        return []
    fig, axes = plt.subplots(
        1, len(clouds), figsize=(7.5 * len(clouds), 5.6), squeeze=False, sharey=True
    )
    for ax, cloud in zip(axes[0], clouds):
        for m in ALL_METHODS:
            for jit in jitters:
                sel = sorted(
                    (
                        r
                        for r in runs
                        if r["cloud"] == cloud
                        and r["method"] == m
                        and r["jitter"] == jit
                    ),
                    key=lambda r: r["n"],
                )
                if not sel:
                    continue
                ns = [r["n"] for r in sel]
                ts = [r["seconds"] for r in sel]
                # Fit the large-N end: below ~100k the times are dominated by fixed costs
                # (kernel launches, allocations, process spawn), which flattens the slope.
                big = [(n, t) for n, t in zip(ns, ts) if t >= min_seconds and n >= 1e5]
                small = [(n, t) for n, t in zip(ns, ts) if t >= min_seconds]
                a = fit_exponent(*zip(*big)) if len(big) >= 4 else None
                if a is None and len(small) >= 4:
                    a = fit_exponent(*zip(*small))
                label = f"{LABELS[m]}" + (" + jitter" if jit else "")
                if a is not None:
                    label += f"  ($\\alpha$={a:.2f})"
                ax.plot(
                    ns,
                    ts,
                    marker="o" if not jit else "^",
                    ms=3.5,
                    lw=1.5,
                    ls="-" if not jit else "--",
                    color=COLORS[m],
                    alpha=1.0 if not jit else 0.65,
                    label=label,
                )
        tiled = [r["n"] for r in runs if r["cloud"] == cloud and r.get("tiles", 1) > 1]
        if tiled:
            ax.axvline(min(tiled), color="k", lw=0.8, ls=":", alpha=0.6)
            ax.text(
                min(tiled),
                ax.get_ylim()[0],
                "  tiled copies of the cloud ->",
                fontsize=7,
                rotation=90,
                va="bottom",
                alpha=0.7,
            )
        _log_axis(ax)
        ax.set_yscale("log")
        ax.set_xlabel("points")
        ax.set_title(cloud)
        ax.grid(True, which="both", alpha=0.25)
    axes[0][0].set_ylabel("seconds (mean of the timed runs)")
    axes[0][-1].legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.suptitle(
        "3D Delaunay: time vs number of points"
        + (f"   [{payload['_env'].get('gpu')}]" if payload["_env"].get("gpu") else ""),
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return [path]


# What each method's time is made of.  Paragram reports its three phases separately, which is more
# informative than one GPU / CPU split; every other method reports GPU and CPU totals, and the two
# command-line tools additionally report the file exchange (measured, excluded from the total).
STACKS = {
    "paragram": [
        ("adjacency", "GPU: Voronoi adjacency", "#4477aa"),
        ("conversion", "GPU: 4-clique conversion", "#88ccee"),
        ("repair", "CPU: exact repair", "#cc6677"),
    ],
}
DEFAULT_STACK = [("gpu", "GPU", "#4477aa"), ("cpu", "CPU", "#cc6677")]


def plot_breakdown(payload: dict, stem: str) -> list[str]:
    """Per method: what the time is spent on as N grows, and how the CPU share moves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [r for r in payload["runs"] if r.get("seconds")]
    out = []
    for cloud in sorted({r["cloud"] for r in runs}):
        jitters = sorted({r["jitter"] for r in runs if r["cloud"] == cloud})
        methods = [
            m
            for m in ALL_METHODS
            if any(r["method"] == m and r["cloud"] == cloud for r in runs)
        ]
        if not methods or not jitters:
            continue
        ncol = 1 + len(methods)
        fig, axes = plt.subplots(
            len(jitters), ncol, figsize=(3.1 * ncol, 3.4 * len(jitters)), squeeze=False
        )
        for row, jit in enumerate(jitters):
            sel = [r for r in runs if r["cloud"] == cloud and r["jitter"] == jit]

            # left panel: the share of the total spent on the CPU
            ax = axes[row][0]
            for m in methods:
                pts = sorted(
                    (r for r in sel if r["method"] == m and r.get("seconds")),
                    key=lambda r: r["n"],
                )
                xy = [
                    (r["n"], (r.get("cpu") or 0.0) / r["seconds"])
                    for r in pts
                    if r["seconds"] > 0
                    and (r.get("cpu") is not None or r.get("gpu") is not None)
                ]
                if xy:
                    ax.plot(
                        *zip(*xy),
                        marker="o",
                        ms=3,
                        lw=1.4,
                        color=COLORS[m],
                        label=LABELS[m],
                    )
            _log_axis(ax)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel("fraction of the time on the CPU")
            ax.set_xlabel("points")
            ax.set_title(f"CPU share   (jitter {jit:g})", fontsize=9)
            ax.grid(True, which="both", alpha=0.25)
            if row == 0:
                ax.legend(fontsize=6, loc="center left")

            # one stacked panel per method
            for col, m in enumerate(methods, start=1):
                ax = axes[row][col]
                pts = sorted(
                    (r for r in sel if r["method"] == m and r.get("seconds")),
                    key=lambda r: r["n"],
                )
                ns = [r["n"] for r in pts]
                stack = STACKS.get(m, DEFAULT_STACK)
                parts = [
                    (lab, colour, [r.get(key) or 0.0 for r in pts])
                    for key, lab, colour in stack
                    if any(r.get(key) for r in pts)
                ]
                if parts and ns:
                    ax.stackplot(
                        ns,
                        *[v for _, _, v in parts],
                        labels=[lab for lab, _, _ in parts],
                        colors=[c for _, c, _ in parts],
                        alpha=0.85,
                    )
                if ns:
                    ax.plot(
                        ns,
                        [r["seconds"] for r in pts],
                        color="k",
                        lw=1.2,
                        ls="--",
                        label="total (mean)",
                    )
                    io = [r.get("io") or 0.0 for r in pts]
                    if any(io):
                        ax.plot(
                            ns,
                            io,
                            color="#999999",
                            lw=1.0,
                            ls=":",
                            label="excluded file I/O",
                        )
                _log_axis(ax)
                ax.set_xlabel("points")
                ax.set_title(f"{LABELS[m]}   (jitter {jit:g})", fontsize=9)
                ax.grid(True, which="both", alpha=0.25)
                ax.legend(fontsize=6, loc="upper left")
        fig.suptitle(
            f"Where the time goes: {cloud}"
            + (
                f"   [{payload['_env'].get('gpu')}]"
                if payload["_env"].get("gpu")
                else ""
            ),
            fontsize=11,
        )
        fig.tight_layout()
        path = f"{stem}_breakdown_{cloud}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"wrote {path}")
        out.append(path)
    return out


def write_csv(payload: dict, path: str) -> None:
    import csv

    cols = [
        "cloud",
        "n",
        "tiles",
        "jitter",
        "method",
        "runs",
        "seconds",
        "min",
        "std",
        "gpu",
        "cpu",
        "io",
        "adjacency",
        "repair",
        "conversion",
        "tets",
        "error",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in payload["runs"]:
            w.writerow(r)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ply", nargs="*", default=[], help="point clouds to scale up and down"
    )
    ap.add_argument(
        "--sizes",
        nargs="+",
        default=["2000:200000:10000", "200000", "300000:1000000:100000"],
        help="point counts: plain numbers and start:stop:step ranges, deduplicated and sorted "
        "(default: 2k..192k step 10k, 200k, then 300k..1M step 100k = 29 sizes)",
    )
    ap.add_argument(
        "--jitter",
        type=float,
        nargs="+",
        default=[0.0, 1e-6],
        help="relative jitters to measure, 0 = the cloud as it is (default: 0 1e-6)",
    )
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument(
        "--repeats", type=int, default=3, help="timed runs per point (default 3)"
    )
    ap.add_argument(
        "--slow-threshold",
        type=float,
        default=10.0,
        help="a method whose first timed run is slower than this is measured once (default 10 s)",
    )
    ap.add_argument(
        "--skip-above",
        type=float,
        default=60.0,
        help="once a method exceeds this many seconds, stop measuring it at larger sizes "
        "(default 60; 0 = never skip)",
    )
    ap.add_argument(
        "--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK") or 0)
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--paragram-bbox-pad", type=float, default=10.0)
    ap.add_argument("--repair", choices=["on", "off"], default="on")
    ap.add_argument(
        "--gstar4d-bin", default=os.environ.get("GSTAR4D_BIN", "bin/gstar4d")
    )
    ap.add_argument("--gstar4d-grid", type=int, default=512)
    ap.add_argument(
        "--dewall-bin", default=os.environ.get("LOCAL_DEWALL_BIN", "bin/dewall")
    )
    ap.add_argument(
        "--cgal-bin", default=os.environ.get("CGAL_DELAUNAY_BIN", "bin/cgal_delaunay")
    )
    ap.add_argument("--tool-timeout", type=float, default=120.0)
    ap.add_argument(
        "--give-up-after",
        type=int,
        default=3,
        help="stop measuring a method at larger sizes after this many consecutive failures "
        "(default 3; a single crash or timeout is input-specific, not a size limit)",
    )
    ap.add_argument(
        "--isolate",
        nargs="*",
        default=["gdel3d"],
        help="run these methods in a separate interpreter with a wall-clock limit, so that a "
        "crash or a hang inside the library costs one measurement instead of the sweep "
        "(default: gdel3d, which has segfaulted, aborted and hung on single inputs)",
    )
    ap.add_argument(
        "--measure-timeout",
        type=float,
        default=300.0,
        help="seconds a single isolated measurement may take before it is killed (default 300)",
    )
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--child-npy", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--child-out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="results/scaling.json")
    ap.add_argument("--csv", default=None, help="also write the records as CSV")
    ap.add_argument("--plot", default=None, help="write the log-log diagram here")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="keep the measurements already in --json and run only the missing ones; a "
        "measurement whose process died is recorded as such and not retried",
    )
    ap.add_argument(
        "--plot-only",
        nargs="+",
        default=None,
        help="skip the sweep and plot these JSON files (several are merged, e.g. one per method)",
    )
    args = ap.parse_args()

    if args.child:  # one isolated measurement, called by measure_in_child()
        if args.cgal_bin and os.path.exists(args.cgal_bin):
            os.environ["CGAL_DELAUNAY_BIN"] = args.cgal_bin
        res = measure(args.child, np.load(args.child_npy), args)
        with open(args.child_out, "w") as fh:
            json.dump(res, fh)
        return 0

    if args.plot_only:
        runs, envs = [], {}
        for path in args.plot_only:
            with open(path) as fh:
                one = json.load(fh)
            runs += one.get("runs", [])
            envs.update(one.get("_env", {}))
        payload = {"_env": envs, "runs": runs}
        print(f"{len(runs)} measurement(s) from {len(args.plot_only)} file(s)")
    else:
        if not args.ply:
            print("nothing to do: pass --ply data/*.ply")
            return 2
        if args.cgal_bin and os.path.exists(args.cgal_bin):
            os.environ["CGAL_DELAUNAY_BIN"] = args.cgal_bin
        elif any(m.startswith("cgal") for m in args.methods):
            print(
                f"CGAL tool not found ({args.cgal_bin}); the cgal_* methods will use whatever "
                "reference_delaunay finds (python bindings, else scipy/Qhull)"
            )
        for m, path in (("gstar4d", args.gstar4d_bin), ("dewall", args.dewall_bin)):
            if m in args.methods and not (path and os.path.exists(path)):
                print(f"{m}: binary not found ({path}); dropping it from the sweep")
                args.methods = [x for x in args.methods if x != m]
        payload = sweep(args)
    if args.csv:
        write_csv(payload, args.csv)
    if args.plot:
        plot(payload, args.plot)
        plot_breakdown(payload, os.path.splitext(args.plot)[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
