"""How much does a jitter deform the cloud, and what does it buy each method?

    python jitter_study.py --ply data/*.ply --json results/jitter.json --plot jitter.png

The benchmark adds a tiny random perturbation to the LiDAR clouds because their points sit on a
sensor grid, which makes large groups of them exactly co-spherical: the Delaunay triangulation is
then not unique and several of the methods return overlapping tetrahedra or fail outright.  This
sweeps the size of that perturbation and records, for every (cloud, jitter):

**How much the cloud was deformed** -- the displacement in absolute terms and relative to the local
point spacing, the change in convex-hull volume, how many points had their nearest neighbour
changed, how many exact duplicates disappeared, and what happened to the reference triangulation
itself (tetrahedra, slivers, flat tetrahedra, smallest volume).

**What each method costs** -- mean seconds over the repetitions, and its tetrahedra compared
against a CGAL reference computed on the same perturbed points.

**How often the predicates fall back to exact arithmetic.**  A robust implementation evaluates its
in-sphere test in floating point first, with an error bound; only when the bound says the sign is
not trustworthy does it redo the test in exact arithmetic, which is one to two orders of magnitude
slower.  Degeneracy is exactly what makes the filter fail, so this is the number that says whether
a jitter has actually removed the degeneracy or merely hidden it.  Three of the methods can report
it:

    gDel3D    patch_pygdel3d.py counts doInSphereFast against doInSphereSoS on the GPU
    CGAL      -DCGAL_PROFILE counts each filtered predicate's calls and its filter failures
              (bin/cgal_delaunay_profile, run untimed and single-threaded so the counts are
              reproducible; built by build_tools.sh)
    Paragram  voronoi_to_delaunay.py counts the in-sphere determinants that land inside its
              float64 rounding-error bound -- it has no exact fallback, so those are tests it
              cannot decide at all rather than ones it repairs

GeoDel (Geogram) has the same counters behind a PCK_STATS build of Geogram, which the GeoDel wheel
is not built with; Local DeWall and gStar4D have no exact fallback to count.  Those methods report
no counters rather than zeros.

The jitter is the benchmark's: an independent Gaussian per coordinate with
sigma = jitter x (largest extent of the bounding box), from a fixed seed, so a given
(cloud, jitter) is always the same point set.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

import test_delaunay_surfaces as T

METHODS = [
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
    "cgal_sequential": "#777777",
}


# ----------------------------------------------------------------------------------------
# the perturbation and how much it deforms the cloud
# ----------------------------------------------------------------------------------------


def jittered(points: np.ndarray, jitter: float, seed: int) -> np.ndarray:
    """The benchmark's perturbation: N(0, jitter x largest extent) added to every coordinate."""
    if jitter <= 0:
        return points.copy()
    rng = np.random.default_rng([seed, len(points), int(jitter * 1e12)])
    return points + rng.normal(scale=jitter * np.ptp(points, axis=0).max(), size=points.shape)


def deformation(base: np.ndarray, pts: np.ndarray, jitter: float) -> dict:
    """How far the perturbed cloud sits from the original, in units a reader can judge.

    The displacement of a point is the length of a 3-vector whose components are each N(0, sigma),
    so its median is 1.538 sigma, not sigma.  What matters is not that displacement on its own but
    its ratio to the distance between neighbouring points: a shift of 0.05 % of the point spacing
    is invisible in any downstream use of the cloud, and still enough to break a tie."""
    disp = np.linalg.norm(pts - base, axis=1)
    d0, nn0 = cKDTree(base).query(base, k=2)
    d1, nn1 = cKDTree(pts).query(pts, k=2)
    spacing = float(np.median(d0[:, 1]))
    out = {
        "sigma": jitter * float(np.ptp(base, axis=0).max()),
        "displacement_median": float(np.median(disp)),
        "displacement_max": float(disp.max()),
        "spacing_median": spacing,
        "displacement_over_spacing": (float(np.median(disp) / spacing) if spacing else None),
        # a displacement far below the spacing can still reorder a neighbourhood, and the
        # neighbourhood is what the algorithms actually see
        "nearest_neighbour_changed": float((nn0[:, 1] != nn1[:, 1]).mean()),
        "duplicates_before": int((d0[:, 1] == 0).sum()),
        "duplicates_after": int((d1[:, 1] == 0).sum()),
        "min_spacing_after": float(d1[:, 1].min()),
    }
    try:
        h0, h1 = ConvexHull(base), ConvexHull(pts)
        out["hull_volume"] = float(h1.volume)
        out["hull_volume_rel_change"] = float(abs(h1.volume - h0.volume) / h0.volume)
    except Exception:  # noqa: BLE001 - a degenerate hull must not stop the sweep
        pass
    return out


def reference_metrics(pts: np.ndarray, ref: np.ndarray, backend: str, seconds: float) -> dict:
    """What the perturbation did to the triangulation itself, not just to the points.

    Sliver and volume statistics over ~700k tetrahedra are not cheap, so this is computed once per
    (cloud, jitter) and stored; the per-method processes reuse it through --resume."""
    m = T.analyze(pts, ref, seconds, ConvexHull(pts))
    return {
        "backend": backend,
        "seconds": seconds,
        "tets": int(m.tets),
        "slivers": int(m.slivers),
        "degenerate_tets": int(m.degenerate_tets),
        "vol_min": float(m.vol_min),
        "radius_ratio_min": float(m.radius_ratio_min),
    }


# ----------------------------------------------------------------------------------------
# predicate counters (untimed: CGAL's profiling build is markedly slower than the plain one)
# ----------------------------------------------------------------------------------------


def cgal_predicate_counts(pts: np.ndarray, profile_bin: str) -> dict | None:
    """Run the -DCGAL_PROFILE build once and read its predicate counters.

    Single-threaded on purpose: the parallel insertion visits the points in an order that depends
    on the thread schedule, so its counts wobble between runs and would not be comparable across
    jitters."""
    if not (profile_bin and os.path.exists(profile_bin)):
        return None
    keep_bin, keep_thr = os.environ.get("CGAL_DELAUNAY_BIN"), os.environ.get("CGAL_THREADS")
    os.environ["CGAL_DELAUNAY_BIN"] = os.path.abspath(profile_bin)
    os.environ["CGAL_THREADS"] = "1"
    try:
        _, _, info = T.reference_delaunay(pts)
    finally:
        for k, v in (("CGAL_DELAUNAY_BIN", keep_bin), ("CGAL_THREADS", keep_thr)):
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    if "insphere_calls" not in info:
        why = info.get("profile_unparsed")
        print(
            "\n   CGAL predicate counters unavailable: "
            + (
                f"the profiler printed a shape this does not parse:\n{why}"
                if why
                else f"{profile_bin} printed no [CGAL::Profile_*] lines "
                "(is it really a -DCGAL_PROFILE build?)"
            ),
            flush=True,
        )
        return None
    return {
        "source": "CGAL_PROFILE (1 thread, untimed)",
        # CGAL's "calls to" already includes the ones its filter failed on, so that is the total
        "total": int(info["insphere_calls"]),
        "exact": int(info.get("insphere_failures", 0)),
        "orientation_total": int(info.get("orientation_calls", 0)),
        "orientation_exact": int(info.get("orientation_failures", 0)),
        "all_predicates_total": int(info.get("predicate_calls", 0)),
        "all_predicates_exact": int(info.get("predicate_failures", 0)),
    }


# ----------------------------------------------------------------------------------------
# one method on one point set
# ----------------------------------------------------------------------------------------


def run_method(method: str, pts: np.ndarray, args) -> tuple[np.ndarray, dict]:
    """(tetrahedra, info) where info carries the seconds and, where available, the counters."""
    if method == "paragram":
        import torch

        from paragram_repair import repair_failed_cells
        from voronoi_to_delaunay import delaunay_from_adjacency, last_insphere_stats

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
        t_repair, repaired = 0.0, None
        if args.repair == "on" and status is not None:
            adjacency, offsets, st = repair_failed_cells(
                p_dev, adjacency, offsets, status, include_hull=True
            )
            t_repair, repaired = st["seconds"], st.get("repaired_fraction")
        t0 = time.perf_counter()
        tets, _ = delaunay_from_adjacency(
            p_dev, adjacency, offsets, status, return_circumcentres=False
        )
        sync()
        t_conv = time.perf_counter() - t0
        info = {
            "seconds": t_adj + t_repair + t_conv,
            "adjacency": t_adj,
            "repair": t_repair,
            "conversion": t_conv,
            "repaired_fraction": repaired,
        }
        # The converter's analogue of a filtered predicate: in-sphere determinants smaller than
        # the float64 rounding-error bound cannot be signed in double precision.  There is no
        # exact fallback, so these are tests it cannot decide at all.
        st = last_insphere_stats()
        if st.get("tests"):
            info["predicates"] = {
                "source": "float64 in-sphere against its rounding-error bound (no exact fallback)",
                "total": int(st["tests"]),
                "exact": int(st["uncertain"]),
                "cliques": int(st["cliques"]),
                "cospherical_tets": int(st["cospherical_tets"]),
            }
        out = tets.cpu().numpy() if hasattr(tets, "cpu") else np.asarray(tets)
        del adjacency, offsets, status, tets, p_dev
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return out, info

    if method == "gdel3d":
        tets, secs, info = T.run_gdel3d(pts)
        st = info.get("stats_ms") or {}
        out = {
            "seconds": secs,
            "gpu": info.get("gpu_seconds"),
            "cpu": info.get("cpu_seconds"),
            "flips": st.get("totalFlipNum"),
        }
        if info.get("predicate_total"):
            out["predicates"] = {
                "source": "doInSphereFast against doInSphereSoS (patch_pygdel3d.py)",
                "total": int(info["predicate_total"]),
                "exact": int(info["predicate_exact"]),
                "exact_tets": int(info.get("predicate_exact_tets", 0)),
            }
        return tets, out

    if method == "gstar4d":
        tets, secs, info = T.run_gstar4d(
            pts,
            args.gstar4d_bin,
            grid_size=args.gstar4d_grid,
            timeout=args.tool_timeout or None,
            verbose=True,
        )
        return tets, {
            "seconds": secs,
            "loops": info.get("consistency_loops"),
            "missing_points": info.get("missing_points"),
            "_points": info.get("_points"),
        }

    if method == "dewall":
        in_unit = bool(pts.min() >= 0.0 and pts.max() < 1.0)
        tets, secs, info = T.run_dewall(
            pts, args.dewall_bin, prenormalized=in_unit, timeout=args.tool_timeout or None
        )
        return tets, {
            "seconds": secs,
            "status": info.get("status"),
            "truncated": info.get("truncated"),
            "_points": info.get("_points"),
        }

    if method == "geodel":
        tets, secs, info = T.run_geodel(pts, nb_threads=args.threads)
        return tets, {"seconds": secs, "threads": info.get("threads_requested")}

    if method in ("cgal_parallel", "cgal_sequential"):
        # Pin the thread count in both cases: left unset, reference_delaunay also times a
        # single-threaded build for comparison, which would double the work here.
        keep = os.environ.get("CGAL_THREADS")
        os.environ["CGAL_THREADS"] = (
            "1" if method == "cgal_sequential" else str(args.threads or os.cpu_count() or 1)
        )
        try:
            tets, _, info = T.reference_delaunay(pts)
        finally:
            os.environ.pop("CGAL_THREADS", None)
            if keep is not None:
                os.environ["CGAL_THREADS"] = keep
        return tets, {"seconds": info["seconds"], "threads": info.get("threads")}

    raise ValueError(f"unknown method {method}")


def measure(method: str, pts: np.ndarray, args) -> tuple[np.ndarray, dict]:
    """Warm-up, then --repeats timed runs.  Counters come from the first timed run."""
    run_method(method, pts, args)  # warm-up: JIT, clocks, first-touch page faults
    tets, first = run_method(method, pts, args)
    times = [first["seconds"]]
    if first["seconds"] < args.slow_threshold:
        for _ in range(max(0, args.repeats - 1)):
            times.append(run_method(method, pts, args)[1]["seconds"])
    out = dict(first)
    out.update(
        runs=len(times),
        seconds=statistics.fmean(times),
        min=min(times),
        std=statistics.pstdev(times) if len(times) > 1 else 0.0,
    )
    return tets, out


# ----------------------------------------------------------------------------------------
# the sweep
# ----------------------------------------------------------------------------------------


def sweep(args) -> dict:
    clouds = {}
    for path in args.ply:
        name = os.path.splitext(os.path.basename(path))[0]
        clouds[name] = T.unit_cube(T.load_ply_vertices(path))
        print(f"{name}: {len(clouds[name])} points", flush=True)

    env = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "jitters": args.jitters,
        "methods": args.methods,
        "repeats": args.repeats,
        "threads": args.threads,
        "seed": args.seed,
        "clouds": {k: len(v) for k, v in clouds.items()},
        "argv": sys.argv[1:],
    }
    try:
        import torch

        env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - the device name is informational only
        pass

    deformations: list[dict] = []
    predicates: list[dict] = []
    runs: list[dict] = []
    done: set[tuple] = set()
    defo_done: set[tuple] = set()
    pred_done: set[tuple] = set()

    if args.resume and args.json and os.path.exists(args.json):
        with open(args.json) as fh:
            old = json.load(fh)
        deformations = old.get("deformations", [])
        predicates = old.get("predicates", [])
        for rec in old.get("runs", []):
            if rec.get("status") == "running":
                # the previous process died on this one: record that and never retry it
                rec["status"] = "crashed"
                rec["error"] = "the process died during this measurement"
            runs.append(rec)
            done.add((rec["cloud"], rec["jitter"], rec["method"]))
        defo_done = {(d["cloud"], d["jitter"]) for d in deformations}
        pred_done = {(p["cloud"], p["jitter"], p["method"]) for p in predicates}
        print(f"resuming: {len(done)} measurement(s) already recorded", flush=True)

    def save():
        if not args.json:
            return
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "_env": env,
                    "deformations": deformations,
                    "predicates": predicates,
                    "runs": runs,
                },
                fh,
                indent=1,
            )

    for cloud_name, base in clouds.items():
        for jit in args.jitters:
            todo = [m for m in args.methods if (cloud_name, jit, m) not in done]
            need_defo = (cloud_name, jit) not in defo_done
            need_cgal = args.predicates and (cloud_name, jit, "cgal_sequential") not in pred_done
            if not todo and not need_defo and not need_cgal:
                continue
            pts = jittered(base, jit, args.seed)
            ref, ref_backend, ref_info = T.reference_delaunay(pts)

            if need_defo:
                defo = deformation(base, pts, jit)
                ref_m = reference_metrics(pts, ref, ref_backend, ref_info["seconds"])
                defo.update(cloud=cloud_name, jitter=jit, reference=ref_m)
                deformations.append(defo)
                defo_done.add((cloud_name, jit))
                print(
                    f"\n[{time.strftime('%H:%M:%S')}] {cloud_name} jitter={jit:g}"
                    f"  sigma={defo['sigma']:.3e}"
                    f"  displacement {100 * (defo['displacement_over_spacing'] or 0):.3f}% of the"
                    f" point spacing, hull volume"
                    f" {100 * defo.get('hull_volume_rel_change', 0):+.4f}%,"
                    f" nearest neighbour changed for"
                    f" {100 * defo['nearest_neighbour_changed']:.2f}% of the points"
                    f"\n{' ' * 11}reference: {ref_m['tets']} tets, {ref_m['slivers']} slivers,"
                    f" {ref_m['degenerate_tets']} flat, duplicates"
                    f" {defo['duplicates_before']} -> {defo['duplicates_after']}",
                    flush=True,
                )
                save()
            else:
                print(f"\n[{time.strftime('%H:%M:%S')}] {cloud_name} jitter={jit:g}", flush=True)

            if need_cgal:
                c = cgal_predicate_counts(pts, args.cgal_profile_bin)
                if c:
                    predicates.append(
                        dict(c, cloud=cloud_name, jitter=jit, method="cgal_sequential")
                    )
                    pred_done.add((cloud_name, jit, "cgal_sequential"))
                    print(
                        f"   {'cgal predicates':16s} in-sphere {c['exact']}/{c['total']} exact"
                        f" ({100 * c['exact'] / max(1, c['total']):.4f}%)",
                        flush=True,
                    )
                    save()

            for m in todo:
                rec = {"cloud": cloud_name, "jitter": jit, "method": m, "status": "running"}
                runs.append(rec)
                save()  # marker: if the library kills the interpreter, --resume sees it
                print(f"   {m:16s}", end="", flush=True)
                try:
                    tets, info = measure(m, pts, args)
                    pts_m = info.pop("_points", None)
                    preds = info.pop("predicates", None)
                    ref_this = ref if pts_m is None else T.reference_delaunay(pts_m)[0]
                    cmp = T.compare_sets(tets, ref_this, len(pts))
                    rec.update(info, status="ok", tets=int(len(tets)), compare=cmp)
                    extra = ""
                    if preds:
                        predicates.append(dict(preds, cloud=cloud_name, jitter=jit, method=m))
                        pred_done.add((cloud_name, jit, m))
                        extra = (
                            f"  in-sphere {preds['exact']}/{preds['total']} exact"
                            f" ({100 * preds['exact'] / max(1, preds['total']):.4f}%)"
                        )
                    print(
                        f" {info['seconds']:8.4f}s  tets {len(tets):8d}"
                        f"  vs ref: -{cmp['ref_only']} +{cmp['method_only']}{extra}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - a failing method must not stop the sweep
                    rec.update(status="failed", error=str(exc)[:300])
                    print(f" FAILED: {rec['error'][:120]}", flush=True)
                done.add((cloud_name, jit, m))
                save()

    save()
    return {"_env": env, "deformations": deformations, "predicates": predicates, "runs": runs}


# ----------------------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------------------


def plot(payload: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    runs = payload["runs"]
    defos = {(d["cloud"], d["jitter"]): d for d in payload["deformations"]}
    preds = {(p["cloud"], p["jitter"], p["method"]): p for p in payload["predicates"]}
    clouds = sorted({r["cloud"] for r in runs} | {c for c, _ in defos})
    if not clouds:
        print("nothing to plot")
        return
    linthresh = min((j for _, j in defos if j > 0), default=1e-9)

    fig, axes = plt.subplots(3, len(clouds), figsize=(6.8 * len(clouds), 12.5), squeeze=False)
    seen: dict[str, object] = {}       # method curves, rows 1 and 2
    seen_defo: dict[str, object] = {}  # deformation curves, row 3

    def jit_axis(ax):
        # symlog so that jitter = 0 (the cloud as it is) has a place on a logarithmic axis
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.set_xlabel("jitter, relative to the cloud's largest extent")
        ax.grid(True, which="both", alpha=0.25)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:g}"))

    for col, cloud in enumerate(clouds):
        sel = [r for r in runs if r["cloud"] == cloud]
        jitters = sorted({j for c, j in defos if c == cloud})

        # --- 1. time -------------------------------------------------------------------
        ax = axes[0][col]
        for m in METHODS:
            xy = sorted(
                (r["jitter"], r["seconds"])
                for r in sel
                if r["method"] == m and r.get("status") == "ok" and r.get("seconds")
            )
            if xy:
                (ln,) = ax.plot(*zip(*xy), marker="o", ms=4, lw=1.6, color=COLORS[m])
                seen.setdefault(LABELS[m], ln)
        jit_axis(ax)
        ax.set_yscale("log")
        # a method that failed at some jitter leaves a gap in its curve; mark where
        for m in METHODS:
            bad = [r["jitter"] for r in sel if r["method"] == m and r.get("status") != "ok"]
            if bad:
                ax.plot(bad, [ax.get_ylim()[0]] * len(bad), "x", ms=7, color=COLORS[m])
        ax.set_ylabel("seconds")
        ax.set_title(f"{cloud}: time  (x on the axis = the method failed)")

        # --- 2. exact-arithmetic fallbacks ----------------------------------------------
        ax = axes[1][col]
        for m in METHODS:
            xy = sorted(
                (j, 100.0 * preds[(cloud, j, m)]["exact"] / max(1, preds[(cloud, j, m)]["total"]))
                for j in jitters
                if (cloud, j, m) in preds
            )
            if xy:
                (ln,) = ax.plot(*zip(*xy), marker="o", ms=4, lw=1.6, color=COLORS[m])
                seen.setdefault(LABELS[m], ln)
        jit_axis(ax)
        ax.set_yscale("symlog", linthresh=1e-5)
        ax.set_ylabel("% of in-sphere tests needing exact arithmetic")
        ax.set_title(f"{cloud}: exact-predicate fallbacks")

        # --- 3. deformation ---------------------------------------------------------------
        ax = axes[2][col]
        js = [j for j in jitters if j > 0]
        base_tets = defos[(cloud, jitters[0])]["reference"]["tets"] if jitters else 0
        # Black with four line styles, not colours: this panel shares a figure with the method
        # curves above, and a shared palette would read as "Paragram" rather than "displacement".
        series = [
            (
                "displacement / point spacing",
                [100 * (defos[(cloud, j)]["displacement_over_spacing"] or 0) for j in js],
                "-", "o",
            ),
            (
                "convex-hull volume change",
                [100 * defos[(cloud, j)].get("hull_volume_rel_change", 0) for j in js],
                "--", "s",
            ),
            (
                "points whose nearest neighbour changed",
                [100 * defos[(cloud, j)]["nearest_neighbour_changed"] for j in js],
                "-.", "^",
            ),
            (
                "tetrahedra added to the reference",
                [
                    100 * (defos[(cloud, j)]["reference"]["tets"] - base_tets) / max(1, base_tets)
                    for j in js
                ],
                ":", "D",
            ),
        ]
        for lab, ys, style, marker in series:
            # a log axis has no room for an exact zero; drop those points instead of spiking
            xy = [(x, y) for x, y in zip(js, ys) if y > 0]
            if xy:
                (ln,) = ax.plot(
                    *zip(*xy), linestyle=style, marker=marker, ms=4, lw=1.5, color="#222222"
                )
                seen_defo.setdefault(lab, ln)
        jit_axis(ax)
        ax.set_yscale("log")
        ax.set_ylabel("% change from the original cloud")
        ax.set_title(f"{cloud}: deformation")

    fig.suptitle(
        "Jitter: what it deforms, what it costs, and how often the predicates need exact arithmetic"
        + (f"   [{payload['_env'].get('gpu')}]" if payload["_env"].get("gpu") else ""),
        fontsize=13,
    )
    # Two legends for the whole figure, below the panels, so nothing is hidden behind a curve:
    # the methods (rows 1 and 2), then the deformation measures (row 3).
    lg = fig.legend(
        list(seen.values()),
        list(seen.keys()),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.035),
    )
    fig.add_artist(lg)
    fig.legend(
        list(seen_defo.values()),
        list(seen_defo.keys()),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        title="deformation panel",
        title_fontsize=9,
        bbox_to_anchor=(0.5, -0.004),
    )
    fig.tight_layout(rect=(0, 0.105, 1, 0.965))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def write_csv(payload: dict, path: str) -> None:
    import csv

    defos = {(d["cloud"], d["jitter"]): d for d in payload["deformations"]}
    preds = {(p["cloud"], p["jitter"], p["method"]): p for p in payload["predicates"]}
    cols = [
        "cloud", "jitter", "method", "status", "runs", "seconds", "min", "std",
        "tets", "ref_only", "method_only",
        "insphere_total", "insphere_exact", "insphere_exact_pct",
        "sigma", "displacement_median", "displacement_over_spacing", "spacing_median",
        "hull_volume_rel_change", "nearest_neighbour_changed",
        "duplicates_before", "duplicates_after",
        "reference_tets", "reference_slivers", "reference_flat", "error",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in payload["runs"]:
            row = dict(r)
            d = defos.get((r["cloud"], r["jitter"]), {})
            row.update(
                {
                    k: d.get(k)
                    for k in (
                        "sigma", "displacement_median", "displacement_over_spacing",
                        "spacing_median", "hull_volume_rel_change", "nearest_neighbour_changed",
                        "duplicates_before", "duplicates_after",
                    )
                }
            )
            ref = d.get("reference") or {}
            row["reference_tets"], row["reference_slivers"] = ref.get("tets"), ref.get("slivers")
            row["reference_flat"] = ref.get("degenerate_tets")
            row.update({k: (r.get("compare") or {}).get(k) for k in ("ref_only", "method_only")})
            p = preds.get((r["cloud"], r["jitter"], r["method"]))
            if p:
                row["insphere_total"], row["insphere_exact"] = p["total"], p["exact"]
                row["insphere_exact_pct"] = 100.0 * p["exact"] / max(1, p["total"])
            w.writerow(row)
    print(f"wrote {path}")


def write_markdown(payload: dict, path: str) -> None:
    """A report-ready summary: one deformation table and three per-method tables per cloud."""
    defos = {(d["cloud"], d["jitter"]): d for d in payload["deformations"]}
    preds = {(p["cloud"], p["jitter"], p["method"]): p for p in payload["predicates"]}
    runs = {(r["cloud"], r["jitter"], r["method"]): r for r in payload["runs"]}
    clouds = sorted({c for c, _ in defos})
    env = payload["_env"]
    L = [
        "# Jitter sweep on the two point clouds",
        "",
        str(env.get("date", ""))
        + (f" -- {env['gpu']}" if env.get("gpu") else "")
        + (f", {env['threads']} CPU cores" if env.get("threads") else ""),
        "",
        "The jitter adds an independent Gaussian to every coordinate, with",
        "`sigma = jitter x (largest extent of the bounding box)`, from a fixed seed.",
        "Jitter `0` is the cloud exactly as it came off the sensor.",
        "",
    ]

    def table(header: str, rows: list[tuple[str, list[str]]], js: list[float]) -> list[str]:
        out = [header, "", "| method | " + " | ".join(f"{j:g}" for j in js) + " |",
               "|---" * (len(js) + 1) + "|"]
        for label, cells in rows:
            if set(cells) != {"-"}:
                out.append(f"| {label} | " + " | ".join(cells) + " |")
        return out + [""]

    for cloud in clouds:
        js = sorted(j for c, j in defos if c == cloud)
        n = env.get("clouds", {}).get(cloud)
        L += [f"## {cloud}" + (f" ({n} points)" if n else ""), ""]

        L += [
            "### How much the jitter deforms the cloud",
            "",
            "| jitter | sigma | median shift | shift / point spacing | hull volume | "
            "nearest neighbour changed | exact duplicates | reference tets | slivers | flat |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        base = defos[(cloud, js[0])]["reference"]["tets"] if js else 0
        for j in js:
            d = defos[(cloud, j)]
            r = d["reference"]
            L.append(
                f"| {j:g} | {d['sigma']:.2e} | {d['displacement_median']:.2e} | "
                f"{100 * (d['displacement_over_spacing'] or 0):.3f} % | "
                f"{100 * d.get('hull_volume_rel_change', 0):+.4f} % | "
                f"{100 * d['nearest_neighbour_changed']:.2f} % | "
                f"{d['duplicates_before']} -> {d['duplicates_after']} | "
                f"{r['tets']} ({100 * (r['tets'] - base) / max(1, base):+.2f} %) | "
                f"{r['slivers']} | {r['degenerate_tets']} |"
            )
        L.append("")

        rows = []
        for m in METHODS:
            cells = []
            for j in js:
                r = runs.get((cloud, j, m))
                if r is None:
                    cells.append("-")
                elif r.get("status") != "ok":
                    cells.append("**" + str(r.get("status") or "failed") + "**")
                else:
                    cells.append(f"{r['seconds']:.3f}")
            rows.append((LABELS[m], cells))
        L += table("### Time (seconds, mean of the repetitions)", rows, js)

        rows = []
        for m in METHODS:
            cells = []
            for j in js:
                r = runs.get((cloud, j, m))
                if r is None or r.get("status") != "ok":
                    cells.append("-" if r is None else "**" + str(r.get("status") or "failed") + "**")
                    continue
                c = r.get("compare") or {}
                miss, extra = c.get("ref_only", 0), c.get("method_only", 0)
                cells.append("identical" if not miss and not extra else f"-{miss} / +{extra}")
            rows.append((LABELS[m], cells))
        L += table("### Tetrahedra against the CGAL reference on the same points", rows, js)

        have = [m for m in METHODS if any((cloud, j, m) in preds for j in js)]
        if have:
            rows = []
            for m in have:
                cells = []
                for j in js:
                    p = preds.get((cloud, j, m))
                    cells.append(
                        "-"
                        if p is None
                        else f"{100 * p['exact'] / max(1, p['total']):.4f} %<br>"
                        f"<sub>{p['exact']} / {p['total']}</sub>"
                    )
                rows.append((LABELS[m], cells))
            L += table(
                "### In-sphere tests that needed exact arithmetic\n\n"
                "Share of the in-sphere evaluations whose floating-point filter could not decide\n"
                "the sign, so the test had to be redone in exact arithmetic (for Paragram: could\n"
                "not be decided at all -- it has no exact fallback).",
                rows,
                js,
            )
            L.append("Counter sources:")
            L.append("")
            for m in have:
                src = next(preds[(cloud, j, m)]["source"] for j in js if (cloud, j, m) in preds)
                L.append(f"* {LABELS[m]}: {src}")
            L.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ply", nargs="*", default=[], help="the point clouds to study")
    ap.add_argument(
        "--jitters",
        type=float,
        nargs="+",
        default=[0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
        help="relative jitters to try (0 = the cloud as it is)",
    )
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument(
        "--slow-threshold",
        type=float,
        default=10.0,
        help="a method slower than this on its first timed run is measured once (default 10 s)",
    )
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK") or 0))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--paragram-bbox-pad", type=float, default=10.0)
    ap.add_argument("--repair", choices=["on", "off"], default="on")
    ap.add_argument("--gstar4d-bin", default=os.environ.get("GSTAR4D_BIN", "bin/gstar4d"))
    ap.add_argument("--gstar4d-grid", type=int, default=512)
    ap.add_argument("--dewall-bin", default=os.environ.get("LOCAL_DEWALL_BIN", "bin/dewall"))
    ap.add_argument("--cgal-bin", default=os.environ.get("CGAL_DELAUNAY_BIN", "bin/cgal_delaunay"))
    ap.add_argument("--cgal-profile-bin", default="bin/cgal_delaunay_profile")
    ap.add_argument(
        "--no-predicates",
        dest="predicates",
        action="store_false",
        help="skip the untimed CGAL profiling pass",
    )
    ap.add_argument("--tool-timeout", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="keep what --json already holds")
    ap.add_argument("--json", default="results/jitter.json")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--plot-only", nargs="+", default=None, help="merge these JSON files instead")
    args = ap.parse_args()

    if args.plot_only:
        payload = {"_env": {}, "deformations": [], "predicates": [], "runs": []}
        seen_defo, seen_pred = set(), set()
        for p in args.plot_only:
            with open(p) as fh:
                one = json.load(fh)
            payload["_env"].update(one.get("_env", {}))
            payload["runs"] += one.get("runs", [])
            for d in one.get("deformations", []):
                if (d["cloud"], d["jitter"]) not in seen_defo:
                    seen_defo.add((d["cloud"], d["jitter"]))
                    payload["deformations"].append(d)
            for q in one.get("predicates", []):
                k = (q["cloud"], q["jitter"], q["method"])
                if k not in seen_pred:
                    seen_pred.add(k)
                    payload["predicates"].append(q)
        print(
            f"{len(payload['runs'])} measurement(s), {len(payload['predicates'])} counter set(s) "
            f"from {len(args.plot_only)} file(s)"
        )
    else:
        if not args.ply:
            print("nothing to do: pass --ply data/*.ply")
            return 2
        if args.cgal_bin and os.path.exists(args.cgal_bin):
            os.environ["CGAL_DELAUNAY_BIN"] = os.path.abspath(args.cgal_bin)
        for m, path in (("gstar4d", args.gstar4d_bin), ("dewall", args.dewall_bin)):
            if m in args.methods and not (path and os.path.exists(path)):
                print(f"{m}: binary not found ({path}); dropping it")
                args.methods = [x for x in args.methods if x != m]
        if args.predicates and not os.path.exists(args.cgal_profile_bin):
            print(
                f"no CGAL profiling build at {args.cgal_profile_bin}; CGAL's predicate counters "
                "will be missing (build it with `bash build_tools.sh`)"
            )
        payload = sweep(args)

    if args.csv:
        write_csv(payload, args.csv)
    if args.markdown:
        write_markdown(payload, args.markdown)
    if args.plot:
        plot(payload, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
