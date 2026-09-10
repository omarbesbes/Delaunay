"""Turn the JSON written by `test_delaunay_surfaces.py --json` into a Markdown report with charts.

    python make_report.py results.json -o report.md
    python make_report.py results.json --baseline results_baseline.json --jitter results_jitter.json -o report.md

Produces `report.md` plus PNG charts next to it (`<stem>_jaccard.png`, `<stem>_timing.png`,
`<stem>_quality.png`).  Methods: Paragram + conversion (primary), and optionally gDel3D and
Local DeWall; every method is compared against the same reference (CGAL).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

DEGENERATE_HINTS = ("not unique", "co-circular", "co-spherical")
PRIMARY = "paragram"
EXTRA_METHODS = [
    ("gdel3d", "gDel3D"),
    ("gstar4d", "gStar4D"),
    ("dewall", "Local DeWall"),
    ("geodel", "GeoDel"),
]
COLORS = {
    "paragram": "#4477aa",
    "gdel3d": "#228833",
    "gstar4d": "#ccbb44",
    "dewall": "#aa3377",
    "geodel": "#66ccee",
    "ref": "#ee6677",
}


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------


def load(path):
    with open(path) as f:
        data = json.load(f)
    env = data.pop("_env", {})
    for k in [k for k in data if k.startswith("_")]:
        del data[k]  # bookkeeping keys such as _in_progress, written by --resume
    for (
        r
    ) in data.values():  # accept files written before the rename "ours" -> "paragram"
        if PRIMARY not in r and "ours" in r:
            r[PRIMARY] = r["ours"]
        c = r.get("compare", {})
        if "method_only" not in c and "ours_only" in c:
            c["method_only"] = c["ours_only"]
        for m, _ in EXTRA_METHODS:
            c = r.get(f"compare_{m}", {})
            if c and "method_only" not in c and "ours_only" in c:
                c["method_only"] = c["ours_only"]
    return env, data


def primary(r):
    return r[PRIMARY]


def methods_present(data) -> list[tuple[str, str]]:
    # A method that only ever failed still gets a column, so a crash or a timeout is visible
    # rather than silently absent.
    return [
        (m, lab)
        for m, lab in EXTRA_METHODS
        if any(m in r or f"{m}_error" in r for r in data.values())
    ]


def ref_name(data) -> str:
    return next(iter(data.values()))["reference"].split()[0] if data else "reference"


def failed_cells(row) -> int:
    h = row.get("paragram_status")
    return (row["n"] - h.get("success", 0)) if h else 0


def _diff_classes(row, key="difference"):
    d = row.get(key) or {}
    return d.get("ref_only", {}), d.get("method_only", {})


def verdict(row) -> str:
    """Paragram + conversion vs the reference, named after what actually differs."""
    c, o = row["compare"], primary(row)
    rp = row.get("repair") or {}
    frac = rp.get("repaired_fraction", 0.0)
    cpu = f" [{100 * frac:.0f}% of cells recomputed on the CPU]" if frac >= 0.05 else ""
    if c["method_only"] == 0 and c["ref_only"] == 0:
        return "IDENTICAL" + cpu
    ref_only, mine = _diff_classes(row)
    missing, extra = c["ref_only"], c["method_only"]
    unrepaired = failed_cells(row) if not row.get("repair") else 0
    if o["delaunay_violations"] > 0 or unrepaired > 0:
        return (
            f"ADJACENCY ERRORS ({o['delaunay_violations']} violations, {failed_cells(row)} failed cells)"
            + cpu
        )
    # a difference dominated by co-spherical ties is legitimate, a deficit of "clean" tets is not
    if missing and ref_only.get("clean", 0) >= max(1, 0.5 * missing):
        pct = 100 * ref_only.get("volume_frac", 0.0)
        return f"INCOMPLETE ({missing} tets missing, {pct:.1f}% of hull volume)" + cpu
    if mine.get("tie", 0) or ref_only.get("tie", 0):
        return f"TIE-BREAK ({missing} ref-only / {extra} extra, co-spherical)" + cpu
    if o["nonmanifold_faces"] > 0:
        return f"OVERLAPPING ({o['nonmanifold_faces']} non-manifold faces)" + cpu
    return f"DIFFERS ({missing} missing / {extra} extra)" + cpu


def method_verdict(row, m: str) -> str:
    """An extra method vs the reference: zero-volume tets first, then ties, then real errors."""
    if m not in row:
        return "FAILED" if f"{m}_error" in row else "-"
    c, g = row[f"compare_{m}"], row[m]
    if c["method_only"] == 0 and c["ref_only"] == 0:
        return "IDENTICAL"
    ref_only, mine = _diff_classes(row, f"difference_{m}")
    flats, ties = mine.get("flat", 0), mine.get("tie", 0)
    if (
        g["delaunay_violations"] == 0
        and flats
        and flats >= 0.9 * c["method_only"]
        and c["ref_only"] == 0
    ):
        return f"VALID + {flats} zero-volume tets"
    # A tie difference does not change the mesh: the volume still matches the convex hull.  A large
    # volume error with "ties" among the missing tets means tetrahedra are genuinely absent (Local
    # DeWall truncates its output at 140001 tetrahedra), so check that before believing the ties.
    if (
        g["delaunay_violations"] == 0
        and (ties or ref_only.get("tie", 0))
        and g["volume_rel_err"] < 1e-6
    ):
        return f"TIE-BREAK ({c['ref_only']} ref-only / {c['method_only']} extra)"
    if c["ref_only"] and g["volume_rel_err"] >= 1e-6:
        pct = (
            100 * (1.0 - g["total_volume"] / g["hull_volume"])
            if g["hull_volume"]
            else 0.0
        )
        return (
            f"INCOMPLETE ({c['ref_only']} tets missing, {pct:.1f}% of the hull volume, "
            f"Euler {g['euler']})"
        )
    if (
        g["delaunay_violations"] == 0
        and g["nonmanifold_faces"] == 0
        and g["volume_rel_err"] < 1e-9
    ):
        return f"VALID, differs ({c['ref_only']} / {c['method_only']})"
    return f"MISMATCH ({g['delaunay_violations']} violations)"


def f(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if v == 0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e6:
            return f"{v:.2e}"
        return f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return str(v)


def method_seconds(r, m):
    """Mean total (GPU + CPU) per run.  primary(r)["seconds"] is already the total, so the
    adjacency phase must not be added to it again."""
    if m == PRIMARY:
        t = (r.get("timing") or {}).get("paragram") or {}
        return t.get("mean", primary(r)["seconds"])
    return r[m]["seconds"] if m in r else float("nan")


# ----------------------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------------------


def summary_table(data) -> str:
    ex = methods_present(data)
    hdr = (
        "| dataset | N | tets ref | tets Paragram | Paragram-only | ref-only | Jaccard | viol Paragram | viol ref "
        "| vol err Paragram | vol err ref | Euler Paragram/ref | non-manifold faces | failed cells | cells repaired on CPU | verdict |"
    )
    for _, lab in ex:
        hdr += f" tets {lab} | {lab}-only | ref-only | viol {lab} | vol err {lab} | verdict {lab} |"
    rows = [hdr, "|" + "---|" * (16 + 6 * len(ex))]
    for name, r in data.items():
        o, rf, c = primary(r), r["ref"], r["compare"]
        line = (
            f"| {name} | {r['n']} | {rf['tets']} | {o['tets']} | {c['method_only']} | {c['ref_only']} | "
            f"{c['jaccard']:.4f} | {o['delaunay_violations']} | {rf['delaunay_violations']} | "
            f"{f(o['volume_rel_err'])} | {f(rf['volume_rel_err'])} | {o['euler']}/{rf['euler']} | "
            f"{o['nonmanifold_faces']} | {failed_cells(r)} | "
            f"{(r.get('repair') or {}).get('repaired_cells', 0)} "
            f"({100 * (r.get('repair') or {}).get('repaired_fraction', 0):.1f}%) | {verdict(r)} |"
        )
        for m, _ in ex:
            if m in r:
                g, cg = r[m], r[f"compare_{m}"]
                line += (
                    f" {g['tets']} | {cg['method_only']} | {cg['ref_only']} | {g['delaunay_violations']} | "
                    f"{f(g['volume_rel_err'])} | {method_verdict(r, m)} |"
                )
            else:
                line += f" - | - | - | - | - | {method_verdict(r, m)} |"
        rows.append(line)
    return "\n".join(rows)


def timing_table(data) -> str:
    ex = methods_present(data)
    seq = any(
        "sequential_seconds" in (r.get("reference_info") or {}) for r in data.values()
    )
    hdr = (
        "| dataset | N | adjacency (GPU, s) | repair (CPU, s) | voronoi→delaunay (GPU, s) | Paragram total (s) "
        "| reference (s) | speed-up vs ref |"
    )
    if seq:
        hdr += " reference 1 thread (s) |"
    for _, lab in ex:
        hdr += f" {lab} (s) | Paragram / {lab} |"
    rows = [hdr, "|" + "---|" * (8 + (1 if seq else 0) + 2 * len(ex))]
    for name, r in data.items():
        t = (r.get("timing") or {}).get("paragram") or {}
        adj = t.get("adjacency_gpu_mean", r.get("adjacency_seconds", float("nan")))
        repair = t.get("repair_cpu_mean", 0.0)
        conv = t.get("conversion_gpu_mean", primary(r)["seconds"])
        ref = r["ref"]["seconds"]
        tot = t.get("mean", adj + repair + conv)
        line = (
            f"| {name} | {r['n']} | {f(adj)} | {f(repair)} | {f(conv)} | {f(tot)} | {f(ref)} | "
            f"{f(ref / tot if tot else float('nan'), 2)}x |"
        )
        if seq:
            s = (r.get("reference_info") or {}).get("sequential_seconds")
            line += f" {f(s) if s is not None else '-'} |"
        for m, _ in ex:
            if m in r:
                tm = r[m]["seconds"]
                line += f" {f(tm)} | {f(tot / tm if tm else float('nan'), 2)}x |"
            else:
                line += " - | - |"
        rows.append(line)
    return "\n".join(rows)


TIMING_LABELS = {
    "paragram": "Paragram + conversion",
    "gdel3d": "gDel3D",
    "gstar4d": "gStar4D",
    "dewall": "Local DeWall",
    "geodel": "GeoDel (Geogram)",
    "cgal_parallel": "CGAL parallel (TBB)",
    "cgal_sequential": "CGAL sequential",
}


def repeated_timing_table(data) -> str:
    """Per dataset and method: runs, mean +/- std, min, GPU and CPU shares (from entry['timing'])."""
    if not any(r.get("timing") for r in data.values()):
        return ""
    rows = [
        "| dataset | method | runs | mean (s) | std (s) | min (s) | GPU (s) | CPU (s) | excluded I/O (s) | breakdown |",
        "|" + "---|" * 10,
    ]
    for name, r in data.items():
        for m, t in (r.get("timing") or {}).items():
            gpu, cpu = t.get("gpu_mean"), t.get("cpu_mean")
            extra = []
            if m == "paragram":
                extra = [
                    f"adjacency (GPU) {f(t.get('adjacency_gpu_mean'))}",
                    f"repair (CPU) {f(t.get('repair_cpu_mean', 0.0))}",
                    f"conversion (GPU) {f(t.get('conversion_gpu_mean'))}",
                ]
            elif m == "gdel3d":
                info = r.get("gdel3d_info") or {}
                ph = info.get("phases_seconds") or {}
                if ph:
                    extra = [f"{k} {v:.3f}" for k, v in ph.items()]
                    if info.get("stats_note"):
                        extra.append(f"({info['stats_note']})")
                    else:
                        extra.append(
                            "(GPU: init/split/flip/relocate/sort; CPU: splaying + copy-back)"
                        )
                st = {}
                if st:
                    extra = [
                        f"{k.replace('Time', '')} {st[k] / 1000:.3f}"
                        for k in (
                            "initTime",
                            "splitTime",
                            "flipTime",
                            "relocateTime",
                            "sortTime",
                            "splayingTime",
                            "outTime",
                        )
                        if k in st
                    ]
                    extra.append(
                        "(GPU: init/split/flip/relocate/sort; CPU: splaying + device->host copy)"
                    )
            elif m == "gstar4d":
                info = r.get("gstar4d_info") or {}
                ph = info.get("phases_seconds") or {}
                extra = [f"{k} {v:.3f}" for k, v in ph.items()] + [
                    (
                        f"(all phases GPU; grid {info.get('grid_size', '?')}^3; "
                        "excluded I/O = text write, PLY parse, process spawn)"
                    )
                ]
            elif m == "geodel":
                info = r.get("geodel_info") or {}
                extra = [
                    (
                        f"CPU only, {info.get('threads_requested', '?')} thread(s) of "
                        f"{info.get('max_threads', '?')} (Geogram ParallelDelaunay3d)"
                    )
                ]
            elif m == "dewall":
                ph = (r.get("dewall_info") or {}).get("phases_seconds") or {}
                extra = [f"{k} {v:.3f}" for k, v in ph.items()] + [
                    "(all phases GPU; CPU = file parsing/normalisation)"
                ]
            elif m.startswith("cgal"):
                extra = ["CPU only"]
            rows.append(
                f"| {name} | {TIMING_LABELS.get(m, m)} | {t['runs']} | {f(t['mean'])} | {f(t['std'])} | {f(t['min'])} | "
                f"{f(gpu) if gpu is not None else '-'} | {f(cpu) if cpu is not None else '-'} | "
                f"{f(t.get('io_mean')) if t.get('io_mean') is not None else '-'} | {'; '.join(extra)} |"
            )
    return "\n".join(rows)


def jitter_timing_table(data, jdata) -> str:
    """Mean time per method without and with the jitter: how much the exact degeneracies cost."""
    methods = [
        m
        for m in TIMING_LABELS
        if any(m in (r.get("timing") or {}) for r in data.values())
        or any(m in (r.get("timing") or {}) for r in jdata.values())
    ]
    rows = [
        "| dataset | " + " | ".join(TIMING_LABELS[m] for m in methods) + " |",
        "|" + "---|" * (len(methods) + 1),
    ]
    for name, r in data.items():
        if name not in jdata:
            continue
        cells = []
        for m in methods:
            a = ((r.get("timing") or {}).get(m) or {}).get("mean")
            b = ((jdata[name].get("timing") or {}).get(m) or {}).get("mean")
            if a is None or b is None:
                cells.append(f({"mean": a}.get("mean")) if a is not None else "-")
            else:
                cells.append(
                    f"{f(a)} -> {f(b)}"
                    + (f" (**{a / b:.1f}x**)" if b > 0 and a / b >= 1.5 else "")
                )
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


# ----------------------------------------------------------------------------------------
# the readable part of the report: scoreboard, one line per dataset, takeaways
# ----------------------------------------------------------------------------------------

SPEED_ORDER = [
    "paragram",
    "gdel3d",
    "gstar4d",
    "dewall",
    "geodel",
    "cgal_parallel",
    "cgal_sequential",
]
SHORT_LABELS = {
    "paragram": "Paragram",
    "gdel3d": "gDel3D",
    "gstar4d": "gStar4D",
    "dewall": "DeWall",
    "geodel": "GeoDel",
    "cgal_parallel": "CGAL par",
    "cgal_sequential": "CGAL seq",
}
REFERENCE_KEYS = ("cgal_parallel", "cgal_sequential")


def method_time(r, m):
    """Mean seconds of one method on one dataset, or None."""
    t = (r.get("timing") or {}).get(m) or {}
    if t.get("mean") is not None:
        return t["mean"]
    if m == PRIMARY:
        return primary(r)["seconds"]
    return r[m]["seconds"] if m in r else None


def timed_methods(data):
    return [
        m
        for m in SPEED_ORDER
        if any(method_time(r, m) is not None for r in data.values())
    ]


def output_methods(data):
    """Methods whose tetrahedra are compared (everything but the reference itself)."""
    return [(PRIMARY, "Paragram")] + list(methods_present(data))


def short_status(r, m) -> str:
    """One cell: how this method's tetrahedra compare with the reference on this dataset."""
    note = ""
    if m == PRIMARY:
        if m not in r and "paragram_error" in r:
            return "failed"
        v, c = verdict(r), r["compare"]
        # Paragram's repair recomputes failed and hull cells with exact CPU arithmetic, and falls
        # back to one global CGAL triangulation when it cannot certify local patches.  When that
        # covers most of the points, "identical to CGAL" says nothing about Paragram, so the share
        # travels with every status.
        frac = (r.get("repair") or {}).get("repaired_fraction", 0.0)
        if frac >= 0.5:
            note = f" (**{100 * frac:.0f}% from the CPU fallback**)"
        elif frac >= 0.05:
            note = f" ({100 * frac:.0f}% CPU)"
    else:
        if m not in r:
            if f"{m}_error" not in r:
                return "-"
            err = r[f"{m}_error"]
            if "did not finish within" in err:
                return "timeout"
            return (
                "**crashed**" if ("assert" in err or "exit code" in err) else "failed"
            )
        v, c = method_verdict(r, m), r[f"compare_{m}"]
    if v.startswith("IDENTICAL"):
        return "**identical**" + note
    flat = (r.get(f"difference_{m}") if m != PRIMARY else r.get("difference")) or {}
    flats = (flat.get("method_only") or {}).get("flat", 0)
    if v.startswith("VALID +") and flats:
        return f"+{flats} flat" + note
    if v.startswith("INCOMPLETE"):
        return f"-{c['ref_only']} missing" + note
    if v.startswith("TIE-BREAK"):
        return f"ties ({c['ref_only']}/{c['method_only']})" + note
    if v.startswith(("ADJACENCY", "MISMATCH")):
        return f"**wrong** ({c['ref_only']}/{c['method_only']})" + note
    return f"differs ({c['ref_only']}/{c['method_only']})" + note


def scoreboard(data) -> str:
    """One row per method: how often it matches the reference, how fast it is, what differs."""
    outs = dict(output_methods(data))
    rows = [
        "| method | runs on | matches CGAL exactly | median time (s) | vs CGAL parallel | what differs |",
        "|" + "---|" * 6,
    ]
    for m in timed_methods(data):
        times = [t for t in (method_time(r, m) for r in data.values()) if t is not None]
        med = statistics.median(times) if times else None
        refs = [
            method_time(r, "cgal_parallel")
            for r in data.values()
            if method_time(r, m) is not None
        ]
        ratios = [
            a / b for a, b in zip(refs, times) if a is not None and b not in (None, 0)
        ]
        speed = (
            f"**{statistics.median(ratios):.1f}x faster**"
            if ratios and statistics.median(ratios) >= 1.05
            else f"{1 / statistics.median(ratios):.1f}x slower"
            if ratios and statistics.median(ratios) > 0
            else "-"
        )
        if m in REFERENCE_KEYS:
            match, why = (
                ("reference", "-")
                if m == "cgal_parallel"
                else ("same as CGAL parallel", "-")
            )
            rows.append(
                f"| {SHORT_LABELS[m]} | {len(times)} / {len(data)} | {match} | {f(med)} | "
                f"{speed if m != 'cgal_parallel' else '-'} | {why} |"
            )
            continue
        stats = {
            n: short_status(r, m)
            for n, r in data.items()
            if m in r or f"{m}_error" in r
        }
        n_id = sum(x == "**identical**" for x in stats.values())
        # A tie difference is not a defect: count only the datasets where tetrahedra are really
        # missing or really wrong, and mention ties separately.
        holed = [
            n for n, x in stats.items() if x.startswith(("-", "**wrong**", "differs"))
        ]
        missing = sum(
            data[n][f"compare_{m}" if m != PRIMARY else "compare"]["ref_only"]
            for n in holed
        )
        tied = [n for n, x in stats.items() if x.startswith("ties")]
        failed = [n for n, x in stats.items() if x.startswith(("timeout", "failed"))]
        flats = sum(
            ((r.get(f"difference_{m}") if m != PRIMARY else r.get("difference")) or {})
            .get("method_only", {})
            .get("flat", 0)
            for r in data.values()
            if m in r
        )
        viol = sum(r[m]["delaunay_violations"] for r in data.values() if m in r)
        why = []
        if missing:
            why.append(f"**{missing} tets missing** on {len(holed)} dataset(s)")
        if flats:
            why.append(f"{flats} zero-volume tets")
        if tied:
            why.append(f"co-spherical ties on {len(tied)} dataset(s)")
        if failed:
            why.append(f"timed out on {len(failed)} dataset(s)")
        if viol:
            why.append(f"**{viol} empty-sphere violations**")
        if m == PRIMARY:
            fracs = [
                (r.get("repair") or {}).get("repaired_fraction", 0.0)
                for r in data.values()
            ]
            if fracs and statistics.median(fracs) >= 0.05:
                why.append(
                    f"median {100 * statistics.median(fracs):.0f}% of cells recomputed exactly "
                    "on the CPU (up to "
                    f"{100 * max(fracs):.0f}%)"
                )
        rows.append(
            f"| {outs.get(m, SHORT_LABELS[m])} | {len(times)} / {len(data)} | "
            f"**{n_id} / {len(data)}** | {f(med)} | {speed} | "
            f"{', '.join(why) if why else 'nothing'} |"
        )
    return "\n".join(rows)


def correctness_overview(data) -> str:
    """One row per dataset, one column per method: how its tetrahedra compare with CGAL's."""
    ms = output_methods(data)
    rows = [
        "| dataset | N | CGAL tets | " + " | ".join(lab for _, lab in ms) + " |",
        "|" + "---|" * (3 + len(ms)),
    ]
    for name, r in data.items():
        cells = [short_status(r, m) for m, _ in ms]
        rows.append(
            f"| {name} | {r['n']} | {r['ref']['tets']} | " + " | ".join(cells) + " |"
        )
    return "\n".join(rows)


def speed_overview(data) -> str:
    """One row per dataset, one column per method: mean seconds, fastest in bold."""
    ms = timed_methods(data)
    rows = [
        "| dataset | N | " + " | ".join(SHORT_LABELS[m] for m in ms) + " |",
        "|" + "---|" * (2 + len(ms)),
    ]
    for name, r in data.items():
        ts = {m: method_time(r, m) for m in ms}
        best = min((v for v in ts.values() if v is not None), default=None)
        cells = [
            "-" if ts[m] is None else (f"**{f(ts[m])}**" if ts[m] == best else f(ts[m]))
            for m in ms
        ]
        rows.append(f"| {name} | {r['n']} | " + " | ".join(cells) + " |")
    med = []
    for m in ms:
        ts = [t for t in (method_time(r, m) for r in data.values()) if t is not None]
        med.append(f(statistics.median(ts)) if ts else "-")
    rows.append("| **median** | | " + " | ".join(med) + " |")
    return "\n".join(rows)


def takeaways(data, jdata=None) -> list[str]:
    """A handful of sentences that state what the tables show."""
    out = []
    ms = [m for m, _ in output_methods(data) if m not in REFERENCE_KEYS]
    exact = [
        m
        for m in ms
        if all(short_status(r, m) == "**identical**" for r in data.values() if m in r)
        and any(m in r for r in data.values())
    ]
    labels = dict(output_methods(data))
    if exact:
        out.append(
            "* **Reproduces CGAL exactly on every dataset:** "
            + ", ".join(labels.get(m, m) for m in exact)
            + "."
        )
    times = {
        m: statistics.median(
            [t for t in (method_time(r, m) for r in data.values()) if t is not None]
            or [float("inf")]
        )
        for m in timed_methods(data)
    }
    if times:
        order = sorted(times, key=times.get)
        out.append(
            "* **Fastest (median over the datasets):** "
            + ", then ".join(f"{SHORT_LABELS[m]} {f(times[m])} s" for m in order[:3])
            + "."
        )
    if PRIMARY in ms:
        # only the datasets where tetrahedra are genuinely absent, not co-spherical ties
        holed = [
            n
            for n, r in data.items()
            if PRIMARY in r
            and short_status(r, PRIMARY).startswith(("-", "**wrong**", "differs"))
        ]
        miss = sum(data[n]["compare"]["ref_only"] for n in holed)
        bad = len(holed)
        if miss:
            out.append(
                f"* **Paragram is incomplete:** {miss} reference tetrahedra missing across "
                f"{bad} of {len(data)} datasets, and its repair falls back to a full CGAL "
                "triangulation when the local patches cannot be certified, so its time includes "
                "one (see the per-dataset notes)."
            )
    flats = {
        m: sum(
            ((r.get(f"difference_{m}") if m != PRIMARY else r.get("difference")) or {})
            .get("method_only", {})
            .get("flat", 0)
            for r in data.values()
            if m in r
        )
        for m in ms
    }
    noisy = [m for m, n in flats.items() if n]
    if noisy:
        out.append(
            "* **Zero-volume tetrahedra:** "
            + ", ".join(f"{labels.get(m, m)} {flats[m]}" for m in noisy)
            + " -- valid output of symbolic perturbation on co-planar points, but they inflate "
            "the tetrahedron count and the sliver statistics."
        )
    if jdata:
        gains = []
        for m in timed_methods(data):
            a = [t for t in (method_time(r, m) for r in data.values()) if t is not None]
            b = [
                t
                for t in (method_time(r, m) for n, r in jdata.items() if n in data)
                if t is not None
            ]
            if a and b and statistics.median(b) > 0:
                g = statistics.median(a) / statistics.median(b)
                if g >= 1.5:
                    gains.append((g, m))
        if gains:
            gains.sort(reverse=True)
            out.append(
                "* **Degeneracy is expensive:** a 1e-6 jitter, which removes the co-spherical "
                "groups, speeds up "
                + ", ".join(f"{SHORT_LABELS[m]} {g:.1f}x" for g, m in gains)
                + " (see the jitter section)."
            )
    return out


def _sides(r, data):
    sides = [("Paragram", primary(r))]
    for m, lab in methods_present(data):
        if m in r:
            sides.append((lab, r[m]))
    sides.append((r["reference"].split()[0], r["ref"]))
    return sides


def quality_table(data) -> str:
    rows = [
        (
            "| dataset | method | tets | vol min | vol max | vol mean | vol median | total vol | hull vol "
            "| radius ratio min | radius ratio mean | slivers (rr<0.05) | min dihedral (deg) | dihedral<5deg |"
        ),
        "|" + "---|" * 14,
    ]
    for name, r in data.items():
        for side, m in _sides(r, data):
            rows.append(
                f"| {name} | {side} | {m['tets']} | {f(m['vol_min'])} | {f(m['vol_max'])} | {f(m['vol_mean'])} | "
                f"{f(m['vol_median'])} | {f(m['total_volume'], 6)} | {f(m['hull_volume'], 6)} | "
                f"{f(m['radius_ratio_min'])} | {f(m['radius_ratio_mean'])} | {m['slivers']} | "
                f"{f(m['dihedral_min_deg'])} | {m['dihedral_below_5deg']} |"
            )
    return "\n".join(rows)


def topology_table(data) -> str:
    rows = [
        (
            "| dataset | method | vertices used / N | edges | faces | tets | Euler V-E+F-T | boundary faces "
            "| hull triangles | degenerate (zero-volume) tets |"
        ),
        "|" + "---|" * 10,
    ]
    for name, r in data.items():
        for side, m in _sides(r, data):
            rows.append(
                f"| {name} | {side} | {m['vertices_used']} / {r['n']} | {m['edges']} | {m['faces']} | {m['tets']} | "
                f"{m['euler']} | {m['boundary_faces']} | {m['hull_triangles']} | {m['degenerate_tets']} |"
            )
    return "\n".join(rows)


def difference_tables(data) -> str:
    out = []
    ref = ref_name(data)
    rows = [
        (
            f"| dataset | {ref}-only tets | vol % | flat | tie | violation | clean | missing edge: failed cell | "
            "missing edge: box-clipped | missing edge: other | all edges present, rejected | Paragram-only tets | vol % | "
            "flat | tie | violation | clean | with spurious edge | adjacency edges missing (failed / clipped / other; "
            "hull) | spurious edges |"
        ),
        "|" + "---|" * 20,
    ]
    for name, r in data.items():
        d = r.get("difference")
        if not d:
            continue
        ro, mo, e = d["ref_only"], d["method_only"], d.get("edges", {})
        rows.append(
            f"| {name} | {ro['count']} | {100 * ro['volume_frac']:.2f} | {ro['flat']} | {ro['tie']} | "
            f"{ro['violation']} | {ro['clean']} | {ro.get('missing_edge_failed_cell', '-')} | "
            f"{ro.get('missing_edge_box_clipped', '-')} | {ro.get('missing_edge_other', '-')} | "
            f"{ro.get('all_edges_present_rejected', '-')} | {mo['count']} | {100 * mo['volume_frac']:.2f} | "
            f"{mo['flat']} | {mo['tie']} | {mo['violation']} | {mo['clean']} | {mo.get('with_spurious_edge', '-')} | "
            + (
                f"{e['missing']} ({e['missing_failed_cell']} / {e['missing_box_clipped']} / {e['missing_other']}; "
                f"{e.get('missing_hull', '-')} hull) | {e['spurious']} |"
                if e
                else "- | - |"
            )
        )
    out += [
        f"### Paragram + conversion vs {ref}\n",
        (
            "Every reference tet that Paragram misses is attributed to its first missing adjacency edge: **failed "
            "cell** (an endpoint has a non-zero Paragram status; not applicable after repair), **box-clipped** (the "
            "dual Voronoi face lies entirely outside Paragram's clipping box, including unbounded faces of hull edges "
            "whose rays point away), or **other** (float32/fast-math effects in an otherwise successful cell). "
            "*All edges present, rejected* would be the only column attributable to the conversion itself. "
            "Paragram-only tets are either **ties** (co-spherical groups, both triangulations valid) or "
            "**violations** caused by a missing neighbour.\n"
        ),
        "\n".join(rows),
    ]
    for m, lab in methods_present(data):
        rows = [
            (
                f"| dataset | {ref}-only tets | vol % | flat | tie | violation | clean | {lab}-only tets | vol % "
                "| flat | tie | violation | clean |"
            ),
            "|" + "---|" * 13,
        ]
        for name, r in data.items():
            d = r.get(f"difference_{m}")
            if not d:
                continue
            ro, mo = d["ref_only"], d["method_only"]
            rows.append(
                f"| {name} | {ro['count']} | {100 * ro['volume_frac']:.2f} | {ro['flat']} | {ro['tie']} | "
                f"{ro['violation']} | {ro['clean']} | {mo['count']} | {100 * mo['volume_frac']:.2f} | {mo['flat']} | "
                f"{mo['tie']} | {mo['violation']} | {mo['clean']} |"
            )
        out += [
            f"\n### {lab} vs {ref}\n",
            (
                "Both are exact, so a difference can only be a **tie** (5+ co-spherical points, both choices valid), "
                "a **flat** tet (zero volume, produced by symbolic perturbation on co-planar points) or a genuine "
                "error (**violation** / **clean** with no tie).\n"
            ),
            "\n".join(rows),
        ]
    return "\n".join(out)


def before_after_table(base: dict, data: dict) -> str:
    rows = [
        (
            "| dataset | N | ref tets | Paragram before | missing before | extra before | failed cells before "
            "| vol err before | Paragram after | missing after | extra after | failed cells (pre-repair) "
            "| cells repaired | vol err after | time before (s) | time after (s) |"
        ),
        "|" + "---|" * 16,
    ]
    for name, r in data.items():
        b = base.get(name)
        if not b:
            continue
        rep_ = r.get("repair") or {}
        rows.append(
            f"| {name} | {r['n']} | {r['ref']['tets']} | {primary(b)['tets']} | {b['compare']['ref_only']} | "
            f"{b['compare']['method_only']} | {failed_cells(b)} | {f(primary(b)['volume_rel_err'])} | "
            f"{primary(r)['tets']} | {r['compare']['ref_only']} | {r['compare']['method_only']} | {failed_cells(r)} | "
            f"{rep_.get('repaired_cells', 0) if rep_ else 0} | {f(primary(r)['volume_rel_err'])} | "
            f"{f(method_seconds(b, PRIMARY))} | {f(method_seconds(r, PRIMARY))} |"
        )
    return "\n".join(rows)


# ----------------------------------------------------------------------------------------
# charts
# ----------------------------------------------------------------------------------------


def _grouped_bars(ax, x, series, log=False):
    k = len(series)
    w = 0.8 / k
    for j, (lab, vals, col) in enumerate(series):
        ax.bar([i - 0.4 + w / 2 + j * w for i in x], vals, w, label=lab, color=col)
    if log:
        ax.set_yscale("log")
    ax.legend(fontsize=7)


def charts(data, stem: str) -> list[str]:
    names = list(data)
    ex = methods_present(data)
    out = []
    x = list(range(len(names)))
    ref = ref_name(data)

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(names)), 3.4))
    series = [
        ("Paragram", [data[n]["compare"]["jaccard"] for n in names], COLORS[PRIMARY])
    ]
    for m, lab in ex:
        series.append(
            (
                lab,
                [
                    data[n][f"compare_{m}"]["jaccard"] if m in data[n] else 0
                    for n in names
                ],
                COLORS[m],
            )
        )
    _grouped_bars(ax, x, series)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"Jaccard(method, {ref})")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.axhline(1, color="k", lw=0.5)
    fig.tight_layout()
    p = f"{stem}_jaccard.png"
    fig.savefig(p, dpi=130)
    out.append(p)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(names)), 3.4))
    series = [
        (
            "Paragram (adjacency+repair+conversion)",
            [method_seconds(data[n], PRIMARY) for n in names],
            COLORS[PRIMARY],
        )
    ]
    for m, lab in ex:
        series.append((lab, [method_seconds(data[n], m) for n in names], COLORS[m]))
    series.append(
        (
            f"reference ({ref})",
            [data[n]["ref"]["seconds"] for n in names],
            COLORS["ref"],
        )
    )
    _grouped_bars(ax, x, series, log=True)
    ax.set_ylabel("seconds (log)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    fig.tight_layout()
    p = f"{stem}_timing.png"
    fig.savefig(p, dpi=130)
    out.append(p)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(max(8, 1.1 * len(names)), 3.4))
    sides = [("Paragram", PRIMARY, COLORS[PRIMARY])] + [
        (lab, m, COLORS[m]) for m, lab in ex
    ]
    sides.append((ref, "ref", COLORS["ref"]))
    for ax, key, label in (
        (axes[0], "radius_ratio_mean", "mean radius ratio (1 = regular)"),
        (axes[1], "dihedral_min_deg", "min dihedral angle (deg, log)"),
    ):
        series = [
            (
                lab,
                [data[n][m][key] if m in data[n] else float("nan") for n in names],
                col,
            )
            for lab, m, col in sides
        ]
        _grouped_bars(ax, x, series, log=(key == "dihedral_min_deg"))
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
    fig.tight_layout()
    p = f"{stem}_quality.png"
    fig.savefig(p, dpi=130)
    out.append(p)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("results")
    ap.add_argument(
        "--jitter", default=None, help="optional results JSON from a --jitter run"
    )
    ap.add_argument(
        "--baseline",
        default=None,
        help="optional results JSON of the uncorrected Paragram run",
    )
    ap.add_argument("-o", "--output", default="report.md")
    args = ap.parse_args()

    env, data = load(args.results)
    jenv, jdata = load(args.jitter) if args.jitter else (None, None)
    ref = ref_name(data)  # noqa: F841 - used by the appendix sections below
    ex = methods_present(data)
    stem = os.path.splitext(args.output)[0]
    imgs = charts(data, stem)
    rel = [os.path.basename(p) for p in imgs]

    n_ident = sum(verdict(r).startswith("IDENTICAL") for r in data.values())
    pad = env.get("paragram_bbox_pad")

    title = "# GPU Delaunay methods vs CGAL: Paragram + Voronoi→Delaunay conversion"
    if ex:
        title += ", " + ", ".join(lab for _, lab in ex)
    settings = ""
    if "paragram_bbox_pad" in env:
        settings = (
            f" Paragram settings: clipping pad = {pad}x extent"
            + (
                " (legacy +1.0 absolute)"
                if (pad if pad is not None else -1) < 0
                else ""
            )
            + f", failed-cell repair {env.get('repair')}"
            + (
                f" (hull cells {env.get('repair_hull')})"
                if env.get("repair_hull")
                else ""
            )
            + (
                f", cell budget {env.get('paragram_max_planes')} planes / {env.get('paragram_max_verts')} vertices."
                if env.get("paragram_max_planes")
                else ", cell budget 64/64 (default)."
            )
        )
    md = [
        title + "\n",
        f"Generated {env.get('date', '')}. Device: `{env.get('device')}`"
        + (f" ({env.get('gpu')})" if env.get("gpu") else "")
        + f", torch {env.get('torch')}, python {env.get('python')}."
        + (f" Jitter: {env.get('jitter')}." if env.get("jitter") else "")
        + f" Samples per analytic surface: {env.get('samples_per_surface')}."
        + settings
        + "\n",
        "## At a glance\n",
        scoreboard(data),
        (
            "\n*matches CGAL exactly* = the two tetrahedron sets are equal, tetrahedron for "
            "tetrahedron. *median time* is the mean of the repeated runs, taken across datasets. "
            "Times are pure compute: the file exchange of the command-line tools is measured "
            "separately and excluded (see the appendix).\n"
        ),
        *takeaways(data, jdata),
        "\n## Does each method get the right answer?\n",
        correctness_overview(data),
        (
            "\nCells read: **identical** = same tetrahedra as CGAL; *-N missing* = N of CGAL's "
            "tetrahedra are absent (holes in the mesh); *+N flat* = N extra tetrahedra of zero "
            "volume, the legitimate result of symbolic perturbation on co-planar points; *ties* = "
            "the two triangulations differ only on co-spherical point groups, where the Delaunay "
            "triangulation is not unique and both answers are correct; **wrong** = tetrahedra "
            "whose circumsphere contains another point, which no Delaunay triangulation may have.\n"
        ),
        f"\n![Jaccard]({rel[0]})\n",
        "## How fast is each method?\n",
        (
            f"Mean seconds over {env.get('repeats', 1)} repeated runs, fastest per dataset in bold "
            f"(a method slower than {env.get('slow_threshold', '-')} s on its first run is repeated "
            f"{env.get('slow_repeats', '-')} times instead).\n"
        ),
        speed_overview(data),
        f"\n![Timing]({rel[1]})\n",
        "\n---\n",
        "# Appendix: all metrics\n",
        (
            "*viol* = tetrahedra whose circumsphere strictly contains another input point (must be "
            "0); *vol err* = |sum of tet volumes - convex-hull volume| / hull volume (0 for a valid "
            "triangulation of the hull; > 0 means holes, overlaps or zero-volume tets); *Euler* = "
            "V - E + F - T (1 for a triangulated ball); *non-manifold faces* = faces shared by more "
            "than two tetrahedra.\n"
        ),
        "## Per-dataset metrics\n",
        summary_table(data),
        "\n## Timing, per phase\n",
        timing_table(data),
        "\n### CPU / GPU split and phase breakdown\n",
        (
            "mean = GPU + CPU. *Excluded I/O* is measured but deliberately not part of the mean: it "
            "is the file exchange of the command-line tools (gStar4D and Local DeWall read a "
            "100k-line text file, CGAL exchanges binary arrays), which a library integration would "
            "not pay. GPU = time in GPU phases as reported by the method itself (Paragram: "
            "adjacency + conversion, synchronised; gDel3D: its init/split/flip/relocate/sort "
            "timers; gStar4D: its init/PBA/initstar/consistency/staroutput timers; Local DeWall: "
            "its phase timers; GeoDel and CGAL are CPU-only). CPU = host work (Paragram: exact "
            "repair of failed and hull cells; gDel3D: star splaying + copy-back; CGAL: "
            "everything).\n"
        ),
        repeated_timing_table(data),
        "\n## Topology\n",
        topology_table(data),
        "\n## Tetrahedron quality\n",
        (
            "Radius ratio = 3·inradius / circumradius (1 for a regular tetrahedron, → 0 for slivers). Surface samples "
            "are notoriously sliver-heavy, which all exact methods reproduce identically on generic inputs.\n"
        ),
        quality_table(data),
        f"\n![Quality]({rel[2]})\n",
        "## Why the tetrahedron sets differ\n",
        (
            "*tie* = another input point lies exactly on the tet's circumsphere (insphere determinant against the 8 "
            "points nearest to the circumcentre), i.e. the Delaunay triangulation is not unique there; *flat* = zero "
            "volume; *violation* = a point strictly inside the circumsphere; *clean* = none of these. *vol %* = volume "
            "of those tets relative to the convex hull.\n"
        ),
        difference_tables(data),
        "\n## Per-dataset notes\n",
    ]
    for name, r in data.items():
        h = r.get("paragram_status") or {}
        detail = ", ".join(f"{k}={v}" for k, v in h.items() if v and k != "success")
        line = (
            f"* **{name}** (N={r['n']}, {r['note']}; adjacency: {r['adjacency']}, reference: {r['reference']}): "
            f"Paragram {verdict(r)}, common={r['compare']['common']}, Paragram-only={r['compare']['method_only']}, "
            f"ref-only={r['compare']['ref_only']}."
        )
        if detail:
            line += f" Paragram status: {detail}."
        if r.get("repair"):
            rp = r["repair"]
            line += (
                f" Repair of {rp['repaired_cells']} cells ({rp['failed_cells']} failed + {rp.get('hull_cells', 0)} on "
                f"the hull): {rp['local_certified']} from local patches, {rp['resolved_globally']} from a global exact "
                f"triangulation ({rp['global_backend']}), {rp['seconds']:.2f}s."
            )
        for m, lab in ex:
            if m in r:
                info = r.get(f"{m}_info", {})
                extra = ""
                if m == "gdel3d":
                    extra = f", self-check={info.get('self_check')}, dead tets removed={info.get('dead_tets', 'n/a')}"
                if m == "gstar4d":
                    extra = (
                        f", gpu {info.get('gpu_seconds', float('nan')):.3f}s, "
                        f"grid {info.get('grid_size', '?')}^3, duplicate points dropped "
                        f"{info.get('dropped_duplicate_points', 0)}"
                        f"+{info.get('dropped_after_scaling', 0)}"
                    )
                    co = info.get("compare_original_coordinates")
                    if co:
                        extra += (
                            "; compared on its float32 grid-scaled point set (reference there: "
                            f"{info.get('reference_tets_on_own_coordinates')} tets); vs the reference on the "
                            f"original coordinates it differs by {co['method_only']} / {co['ref_only']} tets"
                        )
                if m == "geodel":
                    extra = (
                        f", {info.get('threads_requested', '?')} thread(s) of "
                        f"{info.get('max_threads', '?')}, all CPU"
                    )
                if m == "dewall":
                    st = ", ".join(
                        f"{k}={v}" for k, v in (info.get("status") or {}).items()
                    )
                    extra = f", gpu {info.get('gpu_seconds', float('nan')):.3f}s" + (
                        f", status {st}" if st else ""
                    )
                    co = info.get("compare_original_coordinates")
                    if co:
                        extra += (
                            f"; compared on its float32-renormalised point set (reference there: "
                            f"{info.get('reference_tets_on_own_coordinates')} tets); vs the reference on the original "
                            f"coordinates it differs by {co['method_only']} / {co['ref_only']} tets"
                        )
                line += (
                    f" {lab}: {method_verdict(r, m)}, {r[f'compare_{m}']['method_only']} {lab}-only, "
                    f"{r[f'compare_{m}']['ref_only']} ref-only{extra}."
                )
            elif f"{m}_error" in r:
                line += f" {lab} failed: {r[f'{m}_error'][:200]}."
        md.append(line)

    if args.baseline:
        _, bdata = load(args.baseline)
        n_id_b = sum(verdict(r).startswith("IDENTICAL") for r in bdata.values())
        md += [
            "\n## Paragram corrections: before / after\n",
            (
                "*Before* = upstream behaviour (clipping box = point bounding box + 1.0 absolute, failed cells left "
                "truncated). *After* = relative clipping pad, larger cell budget if set, and the exact CPU repair of "
                "failed and hull cells (`paragram_repair.py`). *missing* = reference tets Paragram lacks, *extra* = "
                "Paragram tets not in the reference (co-spherical ties on symmetric inputs), *time* = adjacency "
                "(incl. repair) + conversion.\n"
            ),
            f"Datasets identical to the reference: **{n_id_b} / {len(bdata)}** before, **{n_ident} / {len(data)}** after.\n",
            before_after_table(bdata, data),
        ]

    if args.jitter:
        jenv, jdata = load(args.jitter)
        md += [
            f"\n## Same datasets with jitter {jenv.get('jitter')} (relative to bounding box)\n",
            "A generic perturbation removes co-spherical ties, so degenerate datasets are expected to become identical.\n",
            summary_table(jdata),
            (
                f"\nParagram identical after jitter: **{sum(verdict(r).startswith('IDENTICAL') for r in jdata.values())} / "
                f"{len(jdata)}**."
            ),
            (
                "\nMean seconds per method, without jitter -> with jitter.  The exact predicates of "
                "the degenerate cloud are what the speed-ups pay for: every co-spherical group "
                "forces the flipping / star-splaying methods into their exact-arithmetic path, and "
                "a generic perturbation removes them.\n"
            ),
            jitter_timing_table(data, jdata),
        ]

    md.append(
        "\n## Methods\n"
        "**Paragram + conversion.** Paragram (GPU, float32) returns the Voronoi face adjacency, i.e. the edge graph of "
        "the Delaunay triangulation. Cells that failed (non-zero status) and cells of points on the convex hull are "
        "recomputed exactly on the CPU (`paragram_repair.py`). Tetrahedra are then recovered as 4-cliques of the graph "
        "whose circumsphere contains no Delaunay neighbour of its four vertices (insphere determinant in float64, "
        "tolerance scaled by the rounding-error bound); exact for inputs in general position, all valid cliques "
        "(overlapping) for co-spherical groups.\n\n"
        "**gDel3D** (Cao, Nanjappa, Gao, Tan; I3D 2014, pyGDel3D bindings): GPU insertion + bistellar flipping, CPU "
        "star splaying; double precision, exact predicates, Simulation of Simplicity. Dead tetrahedra of the repair "
        "step and tets incident to the point at infinity are removed before comparison.\n\n"
        "**gStar4D** (Nanjappa, MSc thesis 2012 / Gao et al.): fully GPU star splaying, seeded by a "
        "discrete Voronoi diagram computed with the Parallel Banding Algorithm on a `g^3` voxel grid "
        "(`--gstar4d-grid`, default 512); float32 coordinates with Shewchuk's exact predicates on the "
        "GPU (compiled with `-fmad=false`, without which the error-free transformations break). The "
        "tool scales the input into its grid in float32 and Morton-sorts it, so it is compared "
        "against a reference on that scaled set (mapped back to the original frame for the metrics), "
        "and the difference to the reference on the original coordinates is noted. Ported to CUDA 12 "
        "by `patch_gstar4d.py`; at a grid of 256 it loses too many points to voxel collisions on a "
        "surface cloud and stalls, see the notes in README.md.\n\n"
        "**GeoDel** (Geogram's `ParallelDelaunay3d`, Levy; python binding by Anttwo): CPU only, "
        "multithreaded (given the cores of the job, the same as CGAL parallel); float64 with exact "
        "predicates and symbolic perturbation. Takes the points as they are and returns indices into "
        "them, so no coordinate remapping is needed.\n\n"
        "**Local DeWall** (Gao & Chen, CAD 2026): GPU Delaunay wall construction with ordered local point lists; "
        "float32 coordinates with exact predicates; the tool normalises the input to the unit cube in float32, which "
        "moves points by up to one ulp, so it is compared against a reference on that renormalised set (mapped back "
        "to the original frame for the metrics) and the difference to the reference on the original coordinates is "
        "noted; its sorted output points are matched back to the input indices; tets incident to the points at "
        "infinity are removed.\n\n"
        f"**Reference:** {next(iter(data.values()))['reference'] if data else 'CGAL'}."
    )

    with open(args.output, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"wrote {args.output} and {', '.join(rel)}")


if __name__ == "__main__":
    main()
