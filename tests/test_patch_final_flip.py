"""patch_pygdel3d_final_flip.py must find its anchors in gDel3D's sources, apply exactly once,
upgrade its own first version, keep the files' CRLF endings, and stay compatible with
patch_pygdel3d.py whichever is applied first.

The upstream text is reproduced here verbatim (trailing spaces included) so the anchors are tested
without a checkout; when third_party/pyGDel3D exists the real files are tested too.
"""

import importlib.util
import os
import re
import shutil

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, "third_party", "pyGDel3D", "src", "gdel3d", "gDel3D", "GpuDelaunay.cu")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "script", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ff = _load("patch_pygdel3d_final_flip")

# GpuDel::splitAndFlip() as in half-potato/pyGDel3D (commit 06347b0), LF here; the file is CRLF.
SPLIT_AND_FLIP = (
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
UPSTREAM_CU = (
    '#include "GpuDelaunay.h"\n\n#include <iomanip>\n#include <iostream>\n\n' + SPLIT_AND_FLIP
)

# what the first version of the patch left in place of the upstream block
V1_BLOCK = (
    "    // patch_pygdel3d_final_flip.py: the last round of flipping, unconditionally.\n"
    "    //\n"
    "    // Upstream ran this block only once splitTetra() had cleared _doFlipping, which it does when\n"
    "    // loops find no active tetrahedron and return at once.\n"
    "    _doFlipping = false;\n"
    "\n"
    "    doFlippingLoop( SphereFastOrientFast );\n"
    "\n"
    "    markSpecialTets();\n"
    "    doFlippingLoop( SphereExactOrientSoS );\n"
)
UPSTREAM_IF_BLOCK = SPLIT_AND_FLIP[
    SPLIT_AND_FLIP.index("    if ( !_doFlipping )") : SPLIT_AND_FLIP.index(
        "\n    /////////////////////////////\n"
    )
]
# the anchor consumed "    }\n" and the replacement ended in "\n", so the blank line before the
# separator survived; the fixture must keep it too
V1_CU = UPSTREAM_CU.replace(UPSTREAM_IF_BLOCK, V1_BLOCK)

# the Statistics struct of CommonTypes.h, upstream
STATISTICS = (
    "struct Statistics\n"
    "{\n"
    "    double initTime;\n"
    "    double totalTime;\n"
    "\n"
    "    int failVertNum; \n"
    "    int finalStarNum;\n"
    "    int totalFlipNum; \n"
    "\n"
    "    Statistics() \n"
    "    {\n"
    "        reset(); \n"
    "    }\n"
    "\n"
    "    void reset()\n"
    "    {\n"
    "        initTime        = .0;\n"
    "        failVertNum     = 0; \n"
    "        finalStarNum    = 0;\n"
    "        totalFlipNum    = 0; \n"
    "    }\n"
    "\n"
    "    void accumulate( Statistics s )\n"
    "    {\n"
    "        finalStarNum    += s.finalStarNum;\n"
    "        totalFlipNum    += s.totalFlipNum;\n"
    "    }\n"
    "\n"
    "    void average( int div )\n"
    "    {\n"
    "        finalStarNum    /= div;\n"
    "        totalFlipNum    /= div;\n"
    "    }\n"
    "};\n"
)

# get_stats() as patch_pygdel3d.py writes it (abridged)
GET_STATS = (
    "py::dict PyGDelOutput::getStats() {\n"
    "    py::dict d;\n"
    '    d["totalFlipNum"] = output.stats.totalFlipNum;\n'
    '    d["failVertNum"] = output.stats.failVertNum;\n'
    '    d["finalStarNum"] = output.stats.finalStarNum;\n'
    "    return d;\n"
    "}\n"
)


# ---------------------------------------------------------------------------------------
# GpuDelaunay.cu
# ---------------------------------------------------------------------------------------
def test_applies_to_upstream():
    out, status = ff.apply_gpu_delaunay(UPSTREAM_CU)
    assert status == "patched"
    assert "if ( !_doFlipping )" not in out, "the upstream condition must be gone"
    assert "#include <cstdlib>" in out and out.index("<cstdlib>") < out.index("<iomanip>")
    tail = out[out.index(ff.MARKER + ": the last round") :]
    order = [
        tail.index('getenv( "GDEL3D_ORIGINAL" )'),
        tail.index("if ( !keepUpstream )"),
        tail.index("_doFlipping = false;"),
        tail.index("++_output->stats.finalRoundSkippedNum;"),
        tail.index("doFlippingLoop( SphereFastOrientFast );"),
        tail.index("markSpecialTets();"),
        tail.index("doFlippingLoop( SphereExactOrientSoS );"),
    ]
    assert order == sorted(order)
    # the per-round block inside the while loop is untouched
    assert out.count("relocateAll();") == 1
    assert "        if ( _doFlipping ) \n        {\n            doFlippingLoop" in out
    # nothing outside the block moved (the separator after it has 29 slashes, the one before
    # it 30 -- a plain index() would stop inside the latter, hence rindex)
    assert out.endswith(
        SPLIT_AND_FLIP[SPLIT_AND_FLIP.rindex("    /////////////////////////////\n") :]
    )
    assert out.count("//////////////////////////////\n") == 1


def test_original_mode_is_upstream_verbatim():
    """With the switch set, the code path is upstream's: the round runs iff the flag is clear."""
    out, _ = ff.apply_gpu_delaunay(UPSTREAM_CU)
    body = out[out.index("if ( _doFlipping )\n        ++_output") :]
    assert body.startswith(
        "if ( _doFlipping )\n"
        "        ++_output->stats.finalRoundSkippedNum;\n"
        "    else\n"
        "    {\n"
        "        doFlippingLoop( SphereFastOrientFast );\n"
        "\n"
        "        markSpecialTets();\n"
        "        doFlippingLoop( SphereExactOrientSoS );\n"
        "    }\n"
    )


def test_indentation_follows_the_file():
    out, _ = ff.apply_gpu_delaunay(UPSTREAM_CU.replace("\n    ", "\n\t"))
    assert "\n\tif ( !keepUpstream )\n\t    _doFlipping = false;" in out


def test_upgrades_its_first_version():
    assert "unconditionally" in V1_CU and "if ( !_doFlipping )" not in V1_CU
    out, status = ff.apply_gpu_delaunay(V1_CU)
    assert status == "upgraded"
    assert out == ff.apply_gpu_delaunay(UPSTREAM_CU)[0], (
        "same result as patching upstream directly"
    )


def test_idempotent():
    once, _ = ff.apply_gpu_delaunay(UPSTREAM_CU)
    twice, status = ff.apply_gpu_delaunay(once)
    assert status == "already" and twice == once
    assert once.count("<cstdlib>") == 1


def test_missing_anchor_is_reported_not_guessed():
    text = UPSTREAM_CU.replace(
        "markSpecialTets(); \n        doFlippingLoop( SphereExactOrientSoS )", "x()"
    )
    assert ff.apply_gpu_delaunay(text) == (text, "missing")
    # the include anchor is required too
    text = UPSTREAM_CU.replace('#include "GpuDelaunay.h"', "")
    assert ff.apply_gpu_delaunay(text) == (text, "missing")


# ---------------------------------------------------------------------------------------
# CommonTypes.h and bindings.cpp
# ---------------------------------------------------------------------------------------
def test_statistics_field_in_all_four_places():
    out, status = ff.apply_common_types(STATISTICS)
    assert status == "patched"
    assert "    int finalRoundSkippedNum;\n" in out
    assert "        finalRoundSkippedNum = 0;\n" in out
    assert "        finalRoundSkippedNum += s.finalRoundSkippedNum;\n" in out
    assert "        finalRoundSkippedNum /= div;\n" in out
    assert ff.apply_common_types(out) == (out, "already")
    assert ff.apply_common_types(STATISTICS.replace("totalFlipNum    /= div;", "")) == (
        STATISTICS.replace("totalFlipNum    /= div;", ""),
        "missing",
    )


@pytest.mark.parametrize("first", ["final_flip", "counters"])
def test_statistics_edits_compatible_with_the_counters_patch(first):
    """patch_pygdel3d.py anchors its own fields on the same totalFlipNum lines; both must apply in
    either order, and every line of both must survive."""
    counters = _load("patch_pygdel3d")
    edits = [(p, r) for rel, p, r in counters.CORE_EDITS if rel == "CommonTypes.h"]
    assert len(edits) == 4

    def apply_counters(text):
        for pattern, repl in edits:
            text, n = re.subn(pattern, repl, text, count=1)
            assert n == 1, pattern
        return text

    text = STATISTICS
    for step in [first] + [s for s in ("final_flip", "counters") if s != first]:
        if step == "counters":
            text = apply_counters(text)
        else:
            text, status = ff.apply_common_types(text)
            assert status == "patched"
    for field in ("finalRoundSkippedNum", "predCheckNum", "exactCheckNum", "exactTetNum"):
        assert re.search(rf"\b(int|long long) {field};", text), f"{field}: declaration"
        assert re.search(rf"\b{field}\s*= 0;", text), f"{field}: reset"
        assert re.search(rf"\b{field}\s*\+= s\.{field};", text), f"{field}: accumulate"
        assert re.search(rf"\b{field}\s*/= div;", text), f"{field}: average"
    # the upstream fields are still declared exactly once
    assert len(re.findall(r"\bint totalFlipNum;", text)) == 1
    assert len(re.findall(r"\bint finalStarNum;", text)) == 1


def test_bindings_exposes_the_field_when_get_stats_exists():
    out, status = ff.apply_bindings(GET_STATS)
    assert status == "patched"
    assert 'd["finalRoundSkippedNum"] = output.stats.finalRoundSkippedNum;' in out
    assert (
        out.index('d["finalStarNum"]')
        < out.index('d["finalRoundSkippedNum"]')
        < out.index("return d;")
    )
    assert ff.apply_bindings(out) == (out, "already")


def test_bindings_skipped_without_get_stats():
    upstream = "bool PyGDelOutput::checkCorrectness(py::array_t<RealType> points) {\n"
    assert ff.apply_bindings(upstream) == (upstream, "skipped")


# ---------------------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------------------
def _fake_repo(tmp_path, cu=UPSTREAM_CU, with_get_stats=True):
    repo = tmp_path / "pyGDel3D"
    for rel, text in (
        (ff.GPU_DELAUNAY_CU, cu),
        (ff.COMMON_TYPES_H, STATISTICS),
        (ff.BINDINGS_CPP, GET_STATS if with_get_stats else "int x;\n"),
    ):
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.replace("\n", "\r\n").encode())
    return repo


def test_file_round_trip_keeps_crlf(tmp_path, capsys):
    repo = _fake_repo(tmp_path)
    assert ff.patch(str(repo)) is True
    for rel in (ff.GPU_DELAUNAY_CU, ff.COMMON_TYPES_H, ff.BINDINGS_CPP):
        raw = (repo / rel).read_bytes()
        assert raw.count(b"\n") == raw.count(b"\r\n"), f"{rel}: every line must still end in CRLF"
        assert b"finalRoundSkippedNum" in raw
    before = {
        rel: (repo / rel).read_bytes()
        for rel in (ff.GPU_DELAUNAY_CU, ff.COMMON_TYPES_H, ff.BINDINGS_CPP)
    }
    assert ff.patch(str(repo)) is True  # second run: already patched, unchanged
    assert {rel: (repo / rel).read_bytes() for rel in before} == before
    assert "already patched" in capsys.readouterr().out


def test_file_upgrade_from_v1(tmp_path, capsys):
    repo = _fake_repo(tmp_path, cu=V1_CU)
    assert ff.patch(str(repo)) is True
    assert "upgraded" in capsys.readouterr().out
    text = ff._read(str(repo / ff.GPU_DELAUNAY_CU))[0]
    assert text == ff.apply_gpu_delaunay(UPSTREAM_CU)[0]


def test_file_without_get_stats_patches_the_rest(tmp_path, capsys):
    repo = _fake_repo(tmp_path, with_get_stats=False)
    assert ff.patch(str(repo)) is True
    out = capsys.readouterr().out
    assert "not exposed" in out and "patch_pygdel3d.py" in out


def test_missing_checkout_fails_cleanly(tmp_path, capsys):
    assert ff.patch(str(tmp_path / "nowhere")) is False
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
        "final_flip": lambda: ff.patch(repo),
        "counters": lambda: counters.patch(repo),
    }
    for k in [first] + [k for k in steps if k != first]:
        assert steps[k]() is True, k
    # run this one once more: with get_stats() now present, the field gets exposed
    assert ff.patch(repo) is True
    assert "GDEL3D_ORIGINAL" in ff._read(os.path.join(repo, ff.GPU_DELAUNAY_CU))[0]
    assert "finalRoundSkippedNum" in ff._read(os.path.join(repo, ff.BINDINGS_CPP))[0]


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
    problems, f = ff.check_triangulation(CUBE, kuhn())
    assert problems == []
    assert (f["tets"], f["flat"], f["violations"], f["hull_ok"]) == (6, 0, 0, True)
    assert f["volume"] == pytest.approx(1.0) and f["hull_volume"] == pytest.approx(1.0)


def test_checker_rejects_the_unflipped_output():
    """Five of the six Kuhn tetrahedra: what gDel3D returned. All spheres are empty (the corners
    are co-spherical), so only the coverage test can catch it -- and it must."""
    problems, f = ff.check_triangulation(CUBE, kuhn()[:5])
    assert f["violations"] == 0
    assert f["volume"] == pytest.approx(5 / 6)
    assert not f["hull_ok"]
    assert any("convex hull" in p for p in problems), problems


def test_checker_tolerates_flat_tetrahedra():
    """GDel3D's symbolic perturbation can add a zero-volume tetrahedron on a co-planar face; it has
    no circumsphere and must not break the coverage test either."""
    tets = np.vstack([kuhn(), [[0, 2, 6, 4]]])  # the four corners of the face z = 0
    problems, f = ff.check_triangulation(CUBE, tets)
    assert problems == []
    assert (f["tets"], f["flat"]) == (7, 1)


def test_checker_flags_a_point_inside_a_circumsphere():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.25, 0.25, 0.25]], float)
    problems, f = ff.check_triangulation(pts, [[0, 1, 2, 3]])
    assert f["violations"] == 1
    assert len(problems) == 1 and "circumsphere" in problems[0]


# ---------------------------------------------------------------------------------------
# --verify's case list (the GPU part itself runs only on a cluster)
# ---------------------------------------------------------------------------------------
def test_verify_cases_are_the_thirteen_documented_ones():
    pytest.importorskip("torch")  # benchmark.py imports it
    cases = ff.verify_cases(seed=0)
    names = [n for n, _ in cases]
    assert names[:2] == ["cube8", "cube8+jitter"]
    assert names[2:] == [f"random-{n}" for n in ff.VERIFY_SIZES]
    assert [len(p) for _, p in cases] == [8, 8, *ff.VERIFY_SIZES]
    for _, p in cases:  # preprocessed like the benchmark: unit cube, float32-representable
        assert p.min() >= 0.0 and p.max() <= 1.0
        assert np.array_equal(p, p.astype(np.float32).astype(np.float64))
    assert ff.verify_cases(seed=0)[3][1].tolist() == cases[3][1].tolist(), "deterministic"


def test_child_arguments_are_parsed_and_hidden():
    """--case/--mode/--seed exist for the child processes and stay out of --help."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit):
        ff.main(["--help"])
    assert "--case" not in buf.getvalue() and "--verify" in buf.getvalue()
