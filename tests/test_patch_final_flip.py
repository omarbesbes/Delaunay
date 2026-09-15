"""patch_pygdel3d_final_flip.py must find its anchor at the end of gDel3D's splitAndFlip(), apply
exactly once, keep the file's CRLF endings, and stay compatible with patch_pygdel3d.py whichever is
applied first.

The upstream block is reproduced here verbatim (trailing spaces included) so the anchor is tested
without a checkout; when third_party/pyGDel3D exists the real file is tested too.
"""

import importlib.util
import os
import shutil

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, "third_party", "pyGDel3D", "src", "gdel3d", "gDel3D", "GpuDelaunay.cu")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "script", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


final_flip = _load("patch_pygdel3d_final_flip")

# GpuDel::splitAndFlip() as in half-potato/pyGDel3D (commit 06347b0), LF here; the file is CRLF.
UPSTREAM = (
    "void GpuDel::splitAndFlip()\n"
    "{\n"
    "    int insLoop = 0;\n"
    "\n"
    "    _doFlipping = !_params.insertAll; \n"
    "\n"
    "    //////////////////\n"
    "    while ( _vertVec.size() > 0 )\n"
    "    //////////////////\n"
    "    {\n"
    "        ////////////////////////\n"
    "        splitTetra();\n"
    "        ////////////////////////\n"
    "\n"
    "        if ( _doFlipping ) \n"
    "        {\n"
    "            doFlippingLoop( SphereFastOrientFast ); \n"
    "\n"
    "            markSpecialTets(); \n"
    "            doFlippingLoop( SphereExactOrientSoS ); \n"
    "\n"
    "            relocateAll(); \n"
    "            //////////////////////////\n"
    "        }\n"
    "\n"
    "        ++insLoop;\n"
    "    }\n"
    "\n"
    "    //////////////////////////////\n"
    "    if ( !_doFlipping ) \n"
    "    {\n"
    "        doFlippingLoop( SphereFastOrientFast ); \n"
    "\n"
    "        markSpecialTets(); \n"
    "        doFlippingLoop( SphereExactOrientSoS ); \n"
    "    }\n"
    "\n"
    "    /////////////////////////////\n"
    "\n"
    "    if ( _params.verbose ) \n"
    '        std::cout << "\\nInsert loops: " << insLoop << std::endl;\n'
    "\n"
    "    return;\n"
    "}\n"
)


def test_applies_to_upstream_block():
    out, status = final_flip.apply(UPSTREAM)
    assert status == "patched"
    assert final_flip.MARKER in out
    assert "if ( !_doFlipping )" not in out, "the condition must be gone"
    tail = out[out.index(final_flip.MARKER) :]
    # the three calls follow, in upstream's order, after clearing the flag
    order = [
        tail.index("_doFlipping = false;"),
        tail.index("doFlippingLoop( SphereFastOrientFast );"),
        tail.index("markSpecialTets();"),
        tail.index("doFlippingLoop( SphereExactOrientSoS );"),
    ]
    assert order == sorted(order)
    # the per-round block inside the while loop is untouched: still exactly one conditional
    assert out.count("if ( _doFlipping )") == 1
    assert out.count("relocateAll();") == 1
    # nothing outside the block moved (the separator after the block has 29 slashes, the one
    # before it 30 -- a plain index() would stop inside the latter, hence rindex)
    assert out.startswith(UPSTREAM[: UPSTREAM.index("    if ( !_doFlipping )")])
    assert out.endswith(UPSTREAM[UPSTREAM.rindex("    /////////////////////////////\n") :])


def test_indentation_follows_the_file():
    out, _ = final_flip.apply(UPSTREAM.replace("\n    ", "\n\t"))
    assert "\n\t_doFlipping = false;" in out
    assert "\n\tdoFlippingLoop( SphereExactOrientSoS );" in out


def test_idempotent():
    once, _ = final_flip.apply(UPSTREAM)
    twice, status = final_flip.apply(once)
    assert status == "already"
    assert twice == once


def test_missing_anchor_is_reported_not_guessed():
    text = UPSTREAM.replace(
        "markSpecialTets(); \n        doFlippingLoop( SphereExactOrientSoS )", "x()"
    )
    out, status = final_flip.apply(text)
    assert status == "missing"
    assert out == text


def test_file_round_trip_keeps_crlf(tmp_path):
    repo = tmp_path / "pyGDel3D"
    path = repo / final_flip.GPU_DELAUNAY_CU
    path.parent.mkdir(parents=True)
    path.write_bytes(UPSTREAM.replace("\n", "\r\n").encode())
    assert final_flip.patch(str(repo)) is True
    raw = path.read_bytes()
    assert b"\r\n" in raw
    assert raw.count(b"\n") == raw.count(b"\r\n"), "every line must still end in CRLF"
    assert final_flip.MARKER.encode() in raw
    assert final_flip.patch(str(repo)) is True  # second run: already patched, unchanged
    assert path.read_bytes() == raw


def test_missing_checkout_fails_cleanly(tmp_path, capsys):
    assert final_flip.patch(str(tmp_path / "nowhere")) is False
    assert "not found" in capsys.readouterr().out


@pytest.mark.skipif(not os.path.isfile(REAL), reason="no pyGDel3D checkout in third_party/")
@pytest.mark.parametrize("first", ["final_flip", "counters"])
def test_real_checkout_both_patches_either_order(tmp_path, first):
    """Both patches must apply to the real source, in either order (they touch different
    functions)."""
    counters = _load("patch_pygdel3d")
    src = os.path.join(ROOT, "third_party", "pyGDel3D")
    repo = str(tmp_path / "pyGDel3D")
    shutil.copytree(
        src, repo, ignore=shutil.ignore_patterns(".git", "build", "*.so", "*.egg-info")
    )
    # the checkout is used as it is: on an installed tree "already patched" is the expected outcome
    steps = {
        "final_flip": lambda: final_flip.patch(repo),
        "counters": lambda: counters.patch_core(repo),
    }
    order = [first] + [k for k in steps if k != first]
    for k in order:
        assert steps[k]() is True, k
    text = final_flip._read(os.path.join(repo, final_flip.GPU_DELAUNAY_CU))[0]
    assert final_flip.MARKER in text
    assert "CounterPred" in text or "predCheckNum" in text


# ---------------------------------------------------------------------------------------
# the independent checker behind --verify: it must accept a Delaunay triangulation of the hull,
# tolerate flat tetrahedra, and reject precisely what the bug produced
# ---------------------------------------------------------------------------------------
np = pytest.importorskip("numpy")

# corner index = 4x + 2y + z
CUBE = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])


def kuhn():
    """The six tetrahedra around the diagonal 0-7: chains 0 -> e_a -> e_a + e_b -> 7."""
    import itertools

    return np.array([[0, a, a + b, 7] for a, b, _ in itertools.permutations((4, 2, 1))])


def test_checker_accepts_a_triangulation_of_the_cube():
    problems, f = final_flip.check_triangulation(CUBE, kuhn())
    assert problems == []
    assert (f["tets"], f["flat"], f["violations"], f["hull_ok"]) == (6, 0, 0, True)
    assert f["volume"] == pytest.approx(1.0) and f["hull_volume"] == pytest.approx(1.0)


def test_checker_rejects_the_unflipped_output():
    """Five of the six Kuhn tetrahedra: what gDel3D returned. All spheres are empty (the corners
    are co-spherical), so only the coverage test can catch it -- and it must."""
    problems, f = final_flip.check_triangulation(CUBE, kuhn()[:5])
    assert f["violations"] == 0
    assert f["volume"] == pytest.approx(5 / 6)
    assert not f["hull_ok"]
    assert any("convex hull" in p for p in problems), problems


def test_checker_tolerates_flat_tetrahedra():
    """GDel3D's symbolic perturbation can add a zero-volume tetrahedron on a co-planar face; it has
    no circumsphere and must not break the coverage test either."""
    tets = np.vstack([kuhn(), [[0, 2, 6, 4]]])  # the four corners of the face z = 0
    problems, f = final_flip.check_triangulation(CUBE, tets)
    assert problems == []
    assert (f["tets"], f["flat"]) == (7, 1)


def test_checker_flags_a_point_inside_a_circumsphere():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.25, 0.25, 0.25]], float)
    problems, f = final_flip.check_triangulation(pts, [[0, 1, 2, 3]])
    assert f["violations"] == 1
    assert len(problems) == 1 and "circumsphere" in problems[0]
