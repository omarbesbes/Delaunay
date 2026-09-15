"""Fix gDel3D's skipped last flipping round, keeping upstream's behaviour available for comparison.

    python script/patch_pygdel3d_final_flip.py third_party/pyGDel3D             # patch the source
    python script/patch_pygdel3d_final_flip.py third_party/pyGDel3D --rebuild   # ... and reinstall
    python script/patch_pygdel3d_final_flip.py --verify                         # on a GPU node

Once installed, one build serves both behaviours:

    sbatch script/run_benchmark.sbatch                     # corrected gDel3D (default)
    GDEL3D_ORIGINAL=1 sbatch script/run_benchmark.sbatch   # gDel3D as published, bug included

and gDel3D's statistics carry `finalRoundSkippedNum` (1 when the upstream behaviour skipped the
round, 0 when it ran), so a result says which of the two produced it.

Symptom.  On the 8 corners of a cube gDel3D returned 5 tetrahedra where CGAL, GeoDel and Local
DeWall return 6; its own checker reported a failure, and the 5 tetrahedra held 5/6 of the cube's
volume, so the output was not a triangulation of the hull at all.  The jittered cube -- a generic
point set whose Delaunay triangulation has 10 tetrahedra -- gave the same 5.  Both runs reported
17 raw tetrahedra, which is exactly the count after the insertions and before any flip (5 initial
tetrahedra, then 3 more per inserted point): gDel3D had not flipped once.

Cause, in GpuDelaunay.cu.  doFlipping() defers small amounts of work to a later pass:

    // Too little work, leave it for the last round of flipping
    if ( actNum < PredThreadsPerBlock && _doFlipping )
        return false;

and splitAndFlip() runs that last round only once _doFlipping has been switched off, which
splitTetra() does when an insertion round inserts fewer than 10% of the points:

    if ( vertNum - _insNum < _insNum && _insNum < 0.1 * _pointNum )
        _doFlipping = false;

That describes the tail of a large input.  With 8 points (9 with the point at infinity) 10% is 0.9
point, so the rule can never fire: every flip is deferred to a round that never comes and the raw
insertion result is returned.  Any input whose final insertion round inserts at least 10% of the
points while the exact pass sees fewer than PredThreadsPerBlock (64) active tetrahedra is affected,
i.e. small inputs in general.  The benchmark's other datasets, from 505 points up, were not: their
tail rounds are small, the flag is cleared and the last round runs (gDel3D's tetrahedron counts
there match CGAL's, up to the flat tetrahedra its symbolic perturbation adds on degenerate input).

Fix.  Run the last round unconditionally, with _doFlipping cleared so that doFlipping() no longer
defers.  When nothing was deferred both loops find no active tetrahedron and return immediately,
after one compaction of the status bytes each -- measured on a 100 000-point cloud: same
tetrahedra, same time.  With GDEL3D_ORIGINAL set (to anything but "0" or empty) the block is
upstream's, verbatim.

This lives apart from patch_pygdel3d.py (counters, bindings) on purpose: that patch is validated,
and this one can be applied, or not, on its own.  Both are idempotent and independent of the order
in which they are applied; this one also upgrades its own earlier version (which had no switch).
The extension must be rebuilt afterwards (--rebuild, or the pip line of first_install.sh); --verify
then runs the installed build on small inputs in both modes and checks each result with an
independent empty-sphere and hull-coverage test.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKER = "patch_pygdel3d_final_flip.py"
SWITCH = "GDEL3D_ORIGINAL"
STAT = "finalRoundSkippedNum"

CORE = os.path.join("src", "gdel3d", "gDel3D")
GPU_DELAUNAY_CU = os.path.join(CORE, "GpuDelaunay.cu")
COMMON_TYPES_H = os.path.join(CORE, "CommonTypes.h")
BINDINGS_CPP = os.path.join("src", "gdel3d", "bindings.cpp")

# ---------------------------------------------------------------------------------------
# GpuDelaunay.cu: the block at the end of GpuDel::splitAndFlip()
# ---------------------------------------------------------------------------------------
# Whitespace-tolerant because the upstream sources have inconsistent trailing whitespace; line
# endings are normalised by _read.
UPSTREAM_BLOCK = re.compile(
    r"\n([ \t]*)if \( !_doFlipping \)[ \t]*\n"
    r"[ \t]*\{[ \t]*\n"
    r"[ \t]*doFlippingLoop\( SphereFastOrientFast \);[ \t]*\n"
    r"[ \t]*\n"
    r"[ \t]*markSpecialTets\(\);[ \t]*\n"
    r"[ \t]*doFlippingLoop\( SphereExactOrientSoS \);[ \t]*\n"
    r"[ \t]*\}[ \t]*\n"
)

# The first version of this patch (no switch, no counter): recognised and upgraded.
V1_BLOCK = re.compile(
    r"\n([ \t]*)// patch_pygdel3d_final_flip\.py: the last round of flipping, unconditionally\.[ \t]*\n"
    r"(?:[ \t]*//.*\n)*"
    r"[ \t]*_doFlipping = false;[ \t]*\n"
    r"[ \t]*\n"
    r"[ \t]*doFlippingLoop\( SphereFastOrientFast \);[ \t]*\n"
    r"[ \t]*\n"
    r"[ \t]*markSpecialTets\(\);[ \t]*\n"
    r"[ \t]*doFlippingLoop\( SphereExactOrientSoS \);[ \t]*\n"
)

BLOCK = """
{i}// {marker}: the last round of flipping.
{i}//
{i}// Upstream ran it only once splitTetra() had cleared _doFlipping, which it does when an
{i}// insertion round inserts fewer than 10% of the points -- the tail of a large input.
{i}// doFlipping() counts on that: while _doFlipping is set, its exact pass skips any active set
{i}// smaller than PredThreadsPerBlock ("leave it for the last round").  For a small input the
{i}// 10% rule never fires (8 points: 10% is 0.9 point), the deferred tetrahedra were never
{i}// flipped, and the raw insertion result came out -- 5 tetrahedra for the corners of a cube
{i}// instead of 6, neither Delaunay nor covering the hull.
{i}//
{i}// Corrected behaviour (default): clear the flag, so the round always runs; when nothing was
{i}// deferred both loops find no active tetrahedron and return at once.  {switch}=1 in
{i}// the environment keeps upstream's behaviour, for comparison.  {stat} records
{i}// whether the round was skipped, so a result says which of the two produced it.
{i}const char* gdel3dOriginal = getenv( "{switch}" );
{i}const bool  keepUpstream   = ( gdel3dOriginal != NULL && gdel3dOriginal[0] != '\\0' && gdel3dOriginal[0] != '0' );

{i}if ( !keepUpstream )
{i}    _doFlipping = false;

{i}if ( _doFlipping )
{i}    ++_output->stats.{stat};
{i}else
{i}{{
{i}    doFlippingLoop( SphereFastOrientFast );

{i}    markSpecialTets();
{i}    doFlippingLoop( SphereExactOrientSoS );
{i}}}
"""

INCLUDE_ANCHOR = re.compile(r'(?m)^#include "GpuDelaunay\.h"[ \t]*\n')
INCLUDE = f"#include <cstdlib>   // {MARKER}: getenv\n"


def apply_gpu_delaunay(text: str) -> tuple[str, str]:
    """Patch the LF-normalised text of GpuDelaunay.cu.

    Returns (new text, status): "patched" from upstream, "upgraded" from this patch's first
    version, "already", or "missing" when the end of splitAndFlip() is not the upstream one.
    """
    if SWITCH in text:
        return text, "already"
    m = V1_BLOCK.search(text)
    status = "upgraded"
    if m is None:
        m = UPSTREAM_BLOCK.search(text)
        status = "patched"
    if m is None:
        return text, "missing"
    block = BLOCK.format(i=m.group(1), marker=MARKER, switch=SWITCH, stat=STAT)
    out = text[: m.start()] + block + text[m.end() :]
    if INCLUDE not in out:
        out, n = INCLUDE_ANCHOR.subn(lambda m: m.group(0) + INCLUDE, out, count=1)
        if n != 1:
            return text, "missing"
    return out, status


# ---------------------------------------------------------------------------------------
# CommonTypes.h: the Statistics field
# ---------------------------------------------------------------------------------------
# Anchored on the totalFlipNum lines, like patch_pygdel3d.py's own fields: whichever patch runs
# second still finds its anchor, so the order does not matter.
COMMON_TYPES_EDITS = [
    (
        r"(\n([ \t]*)int totalFlipNum;[ \t]*\n)",
        "\\1"
        "\n"
        f"\\2// {MARKER}: 1 when {SWITCH} kept upstream's behaviour and the\n"
        "\\2// last flipping round was skipped, 0 when it ran (the corrected behaviour, or an input\n"
        "\\2// large enough for upstream's rule to fire).\n"
        f"\\2int {STAT};\n",
    ),
    (r"(\n([ \t]*)totalFlipNum[ \t]*= 0;[ \t]*\n)", f"\\1\\2{STAT} = 0;\n"),
    (
        r"(\n([ \t]*)totalFlipNum[ \t]*\+= s\.totalFlipNum;[ \t]*\n)",
        f"\\1\\2{STAT} += s.{STAT};\n",
    ),
    (r"(\n([ \t]*)totalFlipNum[ \t]*/= div;[ \t]*\n)", f"\\1\\2{STAT} /= div;\n"),
]


def apply_common_types(text: str) -> tuple[str, str]:
    if STAT in text:
        return text, "already"
    out = text
    for pattern, repl in COMMON_TYPES_EDITS:
        out, n = re.subn(pattern, repl, out, count=1)
        if n != 1:
            return text, "missing"
    return out, "patched"


# ---------------------------------------------------------------------------------------
# bindings.cpp: expose the field through get_stats(), which patch_pygdel3d.py adds
# ---------------------------------------------------------------------------------------
STATS_ANCHOR = re.compile(
    r'(\n([ \t]*)d\["finalStarNum"\] = output\.stats\.finalStarNum;[ \t]*\n)'
)


def apply_bindings(text: str) -> tuple[str, str]:
    """Returns "skipped" while get_stats() is absent (run patch_pygdel3d.py, then this again)."""
    if STAT in text:
        return text, "already"
    out, n = STATS_ANCHOR.subn(
        f'\\1\\2d["{STAT}"] = output.stats.{STAT};  // {MARKER}\n', text, count=1
    )
    return (out, "patched") if n == 1 else (text, "skipped")


# ---------------------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------------------


def _read(path: str) -> tuple[str, str]:
    """(text with LF endings, the file's own line ending).

    The gDel3D sources are CRLF; rewriting them with LF would turn a twenty-line patch into whole-
    file diffs.
    """
    with open(path, newline="") as f:
        raw = f.read()
    eol = "\r\n" if raw.count("\r\n") * 2 > raw.count("\n") else "\n"
    return raw.replace("\r\n", "\n"), eol


def _write(path: str, text: str, eol: str) -> None:
    with open(path, "w", newline="") as f:
        f.write(text.replace("\n", eol) if eol != "\n" else text)


def patch(repo: str) -> bool:
    steps = [
        (GPU_DELAUNAY_CU, apply_gpu_delaunay, "final flipping round + GDEL3D_ORIGINAL switch"),
        (COMMON_TYPES_H, apply_common_types, f"Statistics.{STAT}"),
        (BINDINGS_CPP, apply_bindings, f"{STAT} in get_stats()"),
    ]
    ok = True
    for rel, fn, what in steps:
        path = os.path.join(repo, rel)
        if not os.path.isfile(path):
            print(f"{path}: not found (is {repo!r} a pyGDel3D checkout?)")
            return False
        text, eol = _read(path)
        new, status = fn(text)
        if status == "missing":
            print(f"{path}: anchor not found -- the source differs from upstream ({what})")
            ok = False
        elif status == "skipped":
            print(
                f"{path}: get_stats() not present, {STAT} not exposed -- run patch_pygdel3d.py,"
                " then this script again"
            )
        elif status == "already":
            print(f"{path}: already patched ({what})")
        else:
            _write(path, new, eol)
            print(f"{path}: {status} ({what})")
    return ok


def rebuild(repo: str) -> bool:
    # first_install.sh's line, forced: pip otherwise sees the package as installed and keeps the
    # old extension.  The patched setup.py reads the architectures from TORCH_CUDA_ARCH_LIST,
    # which is set the way first_install.sh sets it: from GPU_ARCHS (default V100 + A100), never
    # inherited -- activating the environment exports a 13-architecture list that makes the
    # build take ten times longer.
    env = dict(os.environ)
    env["TORCH_CUDA_ARCH_LIST"] = env.get("GPU_ARCHS", "7.0;8.0")
    cmd = [
        sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
        "--no-build-isolation", "--no-deps", repo,
    ]  # fmt: skip
    print(f"-- TORCH_CUDA_ARCH_LIST={env['TORCH_CUDA_ARCH_LIST']}", " ".join(cmd), flush=True)
    return subprocess.call(cmd, env=env) == 0


# ---------------------------------------------------------------------------------------
# --verify: an independent check of what the installed build returns on small inputs
# ---------------------------------------------------------------------------------------


def check_triangulation(points, tets) -> tuple[list[str], dict]:
    """Is `tets` a Delaunay triangulation of the convex hull of `points`?

    Returns (problems, facts): `problems` is empty when every circumsphere is empty and the
    tetrahedra tile the convex hull without gap or overlap; `facts` holds the numbers behind it.
    Flat tetrahedra (gDel3D's symbolic perturbation emits some on co-planar input) have no
    circumsphere and are skipped by the empty-sphere test; they take part in the coverage test.
    """
    import numpy as np

    P = np.asarray(points, dtype=np.float64)
    T = np.asarray(tets, dtype=np.int64).reshape(-1, 4)
    extent = float(np.ptp(P, axis=0).max())
    problems: list[str] = []
    facts: dict = {"tets": len(T), "flat": 0, "violations": 0, "hull_ok": False}
    if len(T) == 0:
        return ["no tetrahedra"], facts

    a, b, c, d = (P[T[:, k]] for k in range(4))
    vol6 = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)  # 6 x signed volume
    flat = np.abs(vol6) <= 1e-12 * extent**3
    facts["flat"] = int(flat.sum())

    # --- empty circumspheres --------------------------------------------------------------
    # circumcentre x solves 2 (v - a) . x = |v|^2 - |a|^2 for v in {b, c, d}
    idx = np.flatnonzero(~flat)
    if len(idx):
        M = 2.0 * np.stack([b - a, c - a, d - a], axis=1)[idx]
        rhs = np.stack([(v * v).sum(1) - (a * a).sum(1) for v in (b, c, d)], axis=1)[idx]
        centre = np.linalg.solve(M, rhs[..., None])[..., 0]
        r2 = ((centre - a[idx]) ** 2).sum(1)
        d2 = ((P[None, :, :] - centre[:, None, :]) ** 2).sum(2)  # (tets, points)
        inside = d2 < r2[:, None] - 1e-9 * extent**2
        inside[np.arange(len(idx))[:, None], T[idx]] = False  # a tetrahedron's own vertices
        facts["violations"] = int(inside.any(1).sum())
    if facts["violations"]:
        problems.append(f"{facts['violations']} tetrahedra with a point inside the circumsphere")

    # --- coverage: the once-used faces must close a convex surface holding the summed volume --
    faces = T[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
    key = np.sort(faces, axis=1)
    _, first, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    if (counts > 2).any():
        problems.append(f"{int((counts > 2).sum())} faces shared by more than two tetrahedra")
    bf = faces[first[counts == 1]]
    facts["boundary_faces"] = len(bf)
    v = P[bf]  # (faces, 3 vertices, xyz)
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    # outward = away from the centroid, which is interior to any convex body
    flip = np.einsum("ij,ij->i", n, v[:, 0] - P.mean(0)) < 0
    n[flip] *= -1.0
    side = np.einsum("fj,fpj->fp", n, P[None, :, :] - v[:, None, 0, :])
    nonconvex = int((side > 1e-9 * extent**3).any(1).sum())
    if nonconvex:
        problems.append(
            f"{nonconvex} boundary faces have points beyond them (not the convex hull)"
        )
    edges = np.sort(bf[:, [[0, 1], [1, 2], [0, 2]]].reshape(-1, 2), axis=1)
    _, ecount = np.unique(edges, axis=0, return_counts=True)
    if (ecount != 2).any():
        problems.append(
            f"{int((ecount != 2).sum())} boundary edges not on exactly two faces (holes)"
        )
    enclosed = float(np.einsum("ij,ij->i", v[:, 0], n).sum() / 6.0)
    summed = float(np.abs(vol6).sum() / 6.0)
    facts["volume"], facts["hull_volume"] = summed, enclosed
    if abs(summed - enclosed) > 1e-9 * max(abs(enclosed), extent**3):
        problems.append(f"tetrahedra volume {summed:.6g} != enclosed volume {enclosed:.6g}")
    facts["hull_ok"] = not nonconvex and bool((ecount == 2).all())
    return problems, facts


VERIFY_TIMEOUT = 120.0  # seconds per case and mode
VERIFY_SIZES = (9, 10, 12, 16, 24, 32, 48, 64, 100, 200, 500)


def verify_cases(seed: int = 0) -> list:
    """The inputs the bug hit, preprocessed the way the benchmark does it: (name, points)."""
    import numpy as np

    sys.path.insert(0, os.path.join(ROOT, "src"))
    import benchmark as T

    rng = np.random.default_rng(seed)
    cases = [
        ("cube8", T.cube8()),
        ("cube8+jitter", T.cube8() + rng.normal(scale=1e-3, size=(8, 3))),
    ]
    cases += [(f"random-{n}", rng.random((n, 3))) for n in VERIFY_SIZES]
    return [(name, T.unit_cube(raw).astype(np.float32).astype(np.float64)) for name, raw in cases]


def run_one(idx: int, mode: str, seed: int) -> dict:
    """One case in one mode, in this process -- what the child started by verify() does."""
    if mode == "original":
        os.environ[SWITCH] = "1"
    else:
        os.environ.pop(SWITCH, None)
    name, pts = verify_cases(seed)[idx]
    import benchmark as T

    tets, seconds, info = T.run_gdel3d(pts)
    problems, facts = check_triangulation(pts, tets)
    self_check = info.get("self_check")
    if self_check is not True:
        problems.append(f"gDel3D's own checker says {self_check}")
    return {
        "case": name,
        "mode": mode,
        "n": int(len(pts)),
        "seconds": float(seconds),
        "self_check": self_check,
        "skipped": info.get("stats_ms", {}).get(STAT, "?"),
        "problems": problems,
        **facts,
    }


def run_child(idx: int, mode: str, seed: int) -> dict:
    """run_one() in a child process, so that a build that hangs -- upstream gDel3D did, on 32
    random points -- costs one timed-out row instead of the session.

    SIGKILL after the timeout; the driver reclaims the CUDA context.
    """
    import json

    env = dict(os.environ)
    if mode == "original":
        env[SWITCH] = "1"
    else:
        env.pop(SWITCH, None)
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--case", str(idx), "--mode", mode, "--seed", str(seed),
    ]  # fmt: skip
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=VERIFY_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return {"verdict": f"hung: no result after {VERIFY_TIMEOUT:.0f} s, killed"}
    line = next((ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("{")), None)
    if p.returncode != 0 or line is None:
        tail = (p.stderr.strip().splitlines() or ["no output"])[-1]
        return {"verdict": f"crashed: exit {p.returncode} ({tail[:90]})"}
    return json.loads(line)


def verify(seed: int = 0) -> bool:
    """Run the installed gDel3D, through the benchmark's own wrapper, on the inputs the bug hit.

    Every case runs twice, each in its own child process with a timeout: with GDEL3D_ORIGINAL=1
    (upstream's behaviour, shown for the record -- wrong results and, on some inputs, a hang) and
    without (the corrected one, which decides the exit status).  The `skipped` column is
    finalRoundSkippedNum as reported by gDel3D itself.
    """
    import torch

    if not torch.cuda.is_available():
        rel = os.path.relpath(os.path.abspath(__file__), ROOT)
        print(
            "no CUDA device here: run this inside a GPU allocation, e.g.\n"
            f"  srun --partition=gpua100 --gres=gpu:1 --time=00:10:00 --pty python {rel} --verify"
        )
        return False

    cases = verify_cases(seed)
    print(
        f"{'case':<14}{'mode':<10}{'N':>5}{'tets':>6}{'flat':>5}{'self-chk':>9}{'viol':>5}"
        f"{'hull':>5}{'vol/hull':>9}{'skipped':>8}{'s':>7}  verdict",
        flush=True,
    )
    all_ok = True
    for idx, (name, pts) in enumerate(cases):
        for mode in ("original", "corrected"):
            r = run_child(idx, mode, seed)
            prefix = "as upstream: " if mode == "original" else "FAIL: "
            if "verdict" in r:  # hung or crashed: nothing to measure
                ok = False
                dash = "".join(f"{'-':>{w}}" for w in (6, 5, 9, 5, 5, 9, 8, 7))
                print(
                    f"{name:<14}{mode:<10}{len(pts):>5}{dash}  {prefix}{r['verdict']}", flush=True
                )
            else:
                ok = not r["problems"]
                ratio = r["volume"] / r["hull_volume"] if r.get("hull_volume") else float("nan")
                verdict = "ok" if ok else prefix + "; ".join(r["problems"])
                print(
                    f"{name:<14}{mode:<10}{r['n']:>5}{r['tets']:>6}{r['flat']:>5}"
                    f"{str(r['self_check']):>9}{r['violations']:>5}{'ok' if r['hull_ok'] else 'NO':>5}"
                    f"{ratio:>9.4f}{str(r['skipped']):>8}{r['seconds']:>7.3f}  {verdict}",
                    flush=True,
                )
            if mode == "corrected":
                all_ok &= ok
    if not all_ok:
        print(
            "\nA corrected row failing with 5 tetrahedra and self-check False means the installed"
            " extension predates this patch: rebuild it (--rebuild) and run --verify again."
        )
    return all_ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fix gDel3D's skipped last flipping round; GDEL3D_ORIGINAL=1 keeps upstream's behaviour."
    )
    ap.add_argument("repo", nargs="?", help="pyGDel3D checkout (default third_party/pyGDel3D)")
    ap.add_argument(
        "--rebuild", action="store_true", help="reinstall the patched extension with pip"
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="run the installed build on small inputs (needs a GPU)",
    )
    # verify()'s child processes: one case, one mode, a JSON line on stdout
    ap.add_argument("--case", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--mode", choices=("original", "corrected"), help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.case is not None:
        import json

        print(
            json.dumps(run_one(args.case, args.mode or "corrected", args.seed), default=str),
            flush=True,
        )
        return 0

    ok = True
    if args.repo is not None or args.rebuild or not args.verify:
        ok = patch(args.repo or "third_party/pyGDel3D")
        if ok and args.rebuild:
            ok = rebuild(args.repo or "third_party/pyGDel3D")
    if ok and args.verify:
        ok = verify()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
