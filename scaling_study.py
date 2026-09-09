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


def measure(method: str, pts: np.ndarray, args) -> dict:
    """Warm-up, then `--repeats` timed runs (fewer if the first is slow)."""
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


def sweep(args) -> dict:
    clouds = {}
    for path in args.ply:
        name = os.path.splitext(os.path.basename(path))[0]
        clouds[name] = T.unit_cube(T.load_ply_vertices(path))
        print(f"{name}: {len(clouds[name])} points", flush=True)
    sizes = parse_sizes(args.sizes)
    records: list[dict] = []
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

    give_up: set[tuple] = set()  # (cloud, jitter, method) that failed or grew too slow
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
                    if key in give_up:
                        rec["error"] = (
                            "skipped (this method already failed or exceeded --skip-above)"
                        )
                        records.append(rec)
                        continue
                    print(
                        f"[{time.strftime('%H:%M:%S')}] {tag} {m}", end="", flush=True
                    )
                    try:
                        rec.update(measure(m, pts, args))
                        print(
                            f" -> {rec['seconds']:.3f}s ({rec['runs']} run(s))",
                            flush=True,
                        )
                        if args.skip_above and rec["seconds"] > args.skip_above:
                            give_up.add(key)
                            print(
                                f"   {m}: {rec['seconds']:.1f}s > --skip-above {args.skip_above:g}s,"
                                " not measuring it at larger sizes",
                                flush=True,
                            )
                    except Exception as exc:  # noqa: BLE001 - a failing method must not stop the sweep
                        rec["error"] = str(exc)[:400]
                        give_up.add(key)
                        print(f" -> FAILED: {rec['error'][:160]}", flush=True)
                    records.append(rec)
                    save()
    save()
    return {"_env": env, "runs": records}


# ----------------------------------------------------------------------------------------
# plot
# ----------------------------------------------------------------------------------------


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
                a = fit_exponent(
                    [n for n, t in zip(ns, ts) if t >= min_seconds],
                    [t for t in ts if t >= min_seconds],
                )
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
        ax.set_xscale("log")
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="results/scaling.json")
    ap.add_argument("--csv", default=None, help="also write the records as CSV")
    ap.add_argument("--plot", default=None, help="write the log-log diagram here")
    ap.add_argument("--plot-only", default=None, help="skip the sweep, plot this JSON")
    args = ap.parse_args()

    if args.plot_only:
        with open(args.plot_only) as fh:
            payload = json.load(fh)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
