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

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

DEGENERATE_HINTS = ("not unique", "co-circular", "co-spherical")
PRIMARY = "paragram"
EXTRA_METHODS = [("gdel3d", "gDel3D"), ("dewall", "Local DeWall")]
COLORS = {
    "paragram": "#4477aa",
    "gdel3d": "#228833",
    "dewall": "#aa3377",
    "ref": "#ee6677",
}


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------


def load(path):
    with open(path) as f:
        data = json.load(f)
    env = data.pop("_env", {})
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
    return [(m, lab) for m, lab in EXTRA_METHODS if any(m in r for r in data.values())]


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
    if g["delaunay_violations"] == 0 and (ties or ref_only.get("tie", 0)):
        return f"TIE-BREAK ({c['ref_only']} ref-only / {c['method_only']} extra)"
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
    if m == PRIMARY:
        return r.get("adjacency_seconds", 0.0) + primary(r)["seconds"]
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
    "dewall": "Local DeWall",
    "cgal_parallel": "CGAL parallel (TBB)",
    "cgal_sequential": "CGAL sequential",
}


def repeated_timing_table(data) -> str:
    """Per dataset and method: runs, mean +/- std, min, GPU and CPU shares (from entry['timing'])."""
    if not any(r.get("timing") for r in data.values()):
        return ""
    rows = [
        "| dataset | method | runs | mean (s) | std (s) | min (s) | GPU (s) | CPU (s) | breakdown |",
        "|" + "---|" * 9,
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
            elif m == "dewall":
                ph = (r.get("dewall_info") or {}).get("phases_seconds") or {}
                extra = [f"{k} {v:.3f}" for k, v in ph.items()] + [
                    "(all phases GPU; CPU = file parsing/normalisation)"
                ]
            elif m.startswith("cgal"):
                extra = ["CPU only"]
            rows.append(
                f"| {name} | {TIMING_LABELS.get(m, m)} | {t['runs']} | {f(t['mean'])} | {f(t['std'])} | {f(t['min'])} | "
                f"{f(gpu) if gpu is not None else '-'} | {f(cpu) if cpu is not None else '-'} | {'; '.join(extra)} |"
            )
    return "\n".join(rows)


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
    ref = ref_name(data)
    ex = methods_present(data)
    stem = os.path.splitext(args.output)[0]
    imgs = charts(data, stem)
    rel = [os.path.basename(p) for p in imgs]

    n_ident = sum(verdict(r).startswith("IDENTICAL") for r in data.values())
    n_degen = sum(
        verdict(r).startswith(("TIE-BREAK", "OVERLAPPING")) for r in data.values()
    )
    n_adj = sum(verdict(r).startswith("ADJACENCY") for r in data.values())
    n_incomplete = sum(verdict(r).startswith("INCOMPLETE") for r in data.values())
    n_bad = sum(verdict(r).startswith("DIFFERS") for r in data.values())
    tot_failed = sum(failed_cells(r) for r in data.values())
    tot_viol = sum(primary(r)["delaunay_violations"] for r in data.values())
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
        "## Verdict\n",
        (
            f"* **Paragram + conversion**: **{n_ident} / {len(data)}** datasets identical to {ref}; "
            f"{n_incomplete} incomplete (tetrahedra missing because edges are absent from Paragram's adjacency); "
            f"{n_degen} differing only by co-spherical tie-breaks or overlaps; {n_adj} with adjacency errors "
            f"({tot_failed} failed cells in total, {tot_viol} empty-sphere violations in the output); "
            f"{n_bad} otherwise different."
        ),
    ]
    for m, lab in ex:
        vs = [method_verdict(r, m) for r in data.values()]
        n_id = sum(v == "IDENTICAL" for v in vs)
        n_ok = sum(v.startswith(("VALID", "TIE-BREAK")) for v in vs)
        n_fail = sum(v == "FAILED" for v in vs)
        n_mis = sum(v.startswith("MISMATCH") for v in vs)
        flats = sum(
            (r.get(f"difference_{m}") or {}).get("method_only", {}).get("flat", 0)
            for r in data.values()
            if m in r
        )
        md.append(
            f"* **{lab}**: **{n_id} / {len(data)}** identical to {ref}; {n_ok} valid but differing "
            f"({flats} zero-volume tetrahedra in total, produced by symbolic perturbation on co-planar points, "
            f"plus co-spherical tie-breaks); {n_mis} mismatches"
            + (f"; {n_fail} crashed." if n_fail else ".")
        )
    common = [r for r in data.values() if all(m in r for m, _ in ex)]
    if common:
        parts = [
            f"Paragram + conversion {sum(method_seconds(r, PRIMARY) for r in common):.2f} s"
        ]
        parts += [
            f"{lab} {sum(r[m]['seconds'] for r in common):.2f} s" for m, lab in ex
        ]
        parts.append(f"{ref} {sum(r['ref']['seconds'] for r in common):.2f} s")
        md.append(
            f"* Wall time over the {len(common)} datasets all methods ran on: "
            + ", ".join(parts)
            + ". Paragram's time includes the CPU repair; gDel3D's includes host→GPU transfer and its CPU star-splaying; "
            "Local DeWall's is the tool's own GPU total (file I/O excluded); the reference is CGAL's insertion + cell "
            "extraction."
        )
    md += [
        (
            "\nHow to read the columns: *viol* = tetrahedra whose circumsphere strictly contains another input point "
            "(must be 0 for a Delaunay triangulation); *vol err* = |Σ tet volumes − convex-hull volume| / hull volume "
            "(0 for a valid triangulation of the hull; > 0 means gaps, overlapping tets, or zero-volume tets); "
            "*Euler* = V − E + F − T (1 for a triangulated ball); *non-manifold faces* = faces shared by more than two "
            "tetrahedra.\n"
        ),
        "## Summary\n",
        summary_table(data),
        f"\n![Jaccard]({rel[0]})\n",
        "## Timing\n",
        (
            f"Timings are means over the repeated runs (repeats: {env.get('repeats', 1)}; methods slower than "
            f"{env.get('slow_threshold', '-')} s on their first run are repeated {env.get('slow_repeats', '-')} times).\n"
            if env.get("repeats")
            else ""
        ),
        timing_table(data),
        f"\n![Timing]({rel[1]})\n",
        "### Repeated timings and CPU / GPU breakdown\n",
        (
            "GPU = time spent in GPU phases as reported by the method itself (Paragram: adjacency + conversion, "
            "synchronised; gDel3D: its init/split/flip/relocate/sort timers; Local DeWall: its phase timers); "
            "CPU = host work (Paragram: exact repair of failed/hull cells; gDel3D: star splaying + copy-back; "
            "Local DeWall: file parsing and normalisation; CGAL: everything).\n"
        ),
        repeated_timing_table(data),
        "## Topology\n",
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
