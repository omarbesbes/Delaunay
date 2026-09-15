"""Make gDel3D's last round of flipping unconditional, so that small inputs come out Delaunay.

    python script/patch_pygdel3d_final_flip.py third_party/pyGDel3D             # patch the source
    python script/patch_pygdel3d_final_flip.py third_party/pyGDel3D --rebuild   # ... and reinstall
    python script/patch_pygdel3d_final_flip.py --verify                         # on a GPU node

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
after one compaction of the status bytes each, so large inputs are not measurably affected.

This lives apart from patch_pygdel3d.py (counters, bindings) on purpose: that patch is validated,
and this one can be applied, or not, on its own.  Both are idempotent and independent of the order
in which they are applied.  The extension must be rebuilt afterwards (--rebuild, or the pip line of
first_install.sh); --verify then runs the installed build on small inputs and checks the result
with an independent empty-sphere and hull-coverage test.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKER = "patch_pygdel3d_final_flip.py"
GPU_DELAUNAY_CU = os.path.join("src", "gdel3d", "gDel3D", "GpuDelaunay.cu")

# The block at the end of GpuDel::splitAndFlip().  Whitespace-tolerant because the upstream
# sources have inconsistent trailing whitespace; line endings are normalised by _read.
ANCHOR = re.compile(
    r"\n([ \t]*)if \( !_doFlipping \)[ \t]*\n"
    r"[ \t]*\{[ \t]*\n"
    r"[ \t]*doFlippingLoop\( SphereFastOrientFast \);[ \t]*\n"
    r"[ \t]*\n"
    r"[ \t]*markSpecialTets\(\);[ \t]*\n"
    r"[ \t]*doFlippingLoop\( SphereExactOrientSoS \);[ \t]*\n"
    r"[ \t]*\}[ \t]*\n"
)

REPLACEMENT = """
{i}// {marker}: the last round of flipping, unconditionally.
{i}//
{i}// Upstream ran this block only once splitTetra() had cleared _doFlipping, which it does when
{i}// an insertion round inserts fewer than 10% of the points -- the tail of a large input.
{i}// doFlipping() counts on that: while _doFlipping is set, its exact pass skips any active set
{i}// smaller than PredThreadsPerBlock ("leave it for the last round").  For a small input the
{i}// 10% rule never fires (8 points: 10% is 0.9 point), the deferred tetrahedra were never
{i}// flipped, and the raw insertion result came out -- 5 tetrahedra for the corners of a cube
{i}// instead of 6, neither Delaunay nor covering the hull.  When nothing was deferred both
{i}// loops find no active tetrahedron and return at once.
{i}_doFlipping = false;

{i}doFlippingLoop( SphereFastOrientFast );

{i}markSpecialTets();
{i}doFlippingLoop( SphereExactOrientSoS );
"""


def apply(text: str) -> tuple[str, str]:
    """Patch the LF-normalised text of GpuDelaunay.cu.

    Returns (new text, status), status being "patched", "already" or "missing".
    """
    if MARKER in text:
        return text, "already"
    m = ANCHOR.search(text)
    if m is None:
        return text, "missing"
    new = REPLACEMENT.format(i=m.group(1), marker=MARKER)
    return text[: m.start()] + new + text[m.end() :], "patched"


def _read(path: str) -> tuple[str, str]:
    """(text with LF endings, the file's own line ending).

    GpuDelaunay.cu is CRLF; rewriting it with LF would turn a ten-line patch into a whole-file
    diff.
    """
    with open(path, newline="") as f:
        raw = f.read()
    eol = "\r\n" if raw.count("\r\n") * 2 > raw.count("\n") else "\n"
    return raw.replace("\r\n", "\n"), eol


def _write(path: str, text: str, eol: str) -> None:
    with open(path, "w", newline="") as f:
        f.write(text.replace("\n", eol) if eol != "\n" else text)


def patch(repo: str) -> bool:
    path = os.path.join(repo, GPU_DELAUNAY_CU)
    if not os.path.isfile(path):
        print(f"{path}: not found (is {repo!r} a pyGDel3D checkout?)")
        return False
    text, eol = _read(path)
    new, status = apply(text)
    if status == "missing":
        print(
            f"{path}: anchor not found -- the end of GpuDel::splitAndFlip() differs from upstream"
        )
        return False
    if status == "already":
        print(f"{path}: already patched (final flipping round)")
        return True
    _write(path, new, eol)
    print(f"{path}: patched (the final flipping round is now unconditional)")
    return True


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


def verify(seed: int = 0) -> bool:
    """Run the installed gDel3D, through the benchmark's own wrapper, on the inputs the bug hit."""
    import numpy as np

    sys.path.insert(0, os.path.join(ROOT, "src"))
    import torch

    if not torch.cuda.is_available():
        rel = os.path.relpath(os.path.abspath(__file__), ROOT)
        print(
            "no CUDA device here: run this inside a GPU allocation, e.g.\n"
            f"  srun --partition=gpua100 --gres=gpu:1 --time=00:10:00 --pty python {rel} --verify"
        )
        return False
    import benchmark as T

    rng = np.random.default_rng(seed)
    cases = [
        ("cube8", T.cube8()),
        ("cube8+jitter", T.cube8() + rng.normal(scale=1e-3, size=(8, 3))),
    ]
    cases += [
        (f"random-{n}", rng.random((n, 3))) for n in (9, 10, 12, 16, 24, 32, 48, 64, 100, 200, 500)
    ]

    print(
        f"{'case':<14}{'N':>5}{'tets':>6}{'flat':>5}{'self-chk':>9}{'viol':>5}{'hull':>5}"
        f"{'vol/hull':>9}{'s':>7}  verdict"
    )
    all_ok = True
    for name, raw in cases:
        pts = (
            T.unit_cube(raw).astype(np.float32).astype(np.float64)
        )  # the benchmark's preprocessing
        tets, seconds, info = T.run_gdel3d(pts)
        problems, f = check_triangulation(pts, tets)
        if info.get("self_check") is not True:
            problems.append(f"gDel3D's own checker says {info.get('self_check')}")
        ok = not problems
        all_ok &= ok
        ratio = f["volume"] / f["hull_volume"] if f.get("hull_volume") else float("nan")
        print(
            f"{name:<14}{len(pts):>5}{f['tets']:>6}{f['flat']:>5}{str(info.get('self_check')):>9}"
            f"{f['violations']:>5}{'ok' if f['hull_ok'] else 'NO':>5}{ratio:>9.4f}{seconds:>7.3f}"
            f"  {'ok' if ok else 'FAIL: ' + '; '.join(problems)}"
        )
    if not all_ok:
        print(
            "\nIf cube8 gives 5 tetrahedra with self-check False, the installed extension predates"
            " this patch: rebuild it (--rebuild) and run --verify again."
        )
    return all_ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Make gDel3D's final flipping round unconditional (small inputs came out unflipped)."
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
    args = ap.parse_args(argv)

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
