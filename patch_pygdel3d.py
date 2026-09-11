"""Patch pyGDel3D so the Python side can drop dead tetrahedra and read gDel3D's own counters.

    python patch_pygdel3d.py /path/to/pyGDel3D

Three things are added, all idempotent:

**`DelOutput.get_tet_info()`** -- gDel3D keeps a per-tetrahedron status byte (`tetInfoVec`, bit 0 =
alive).  Its CPU star-splaying repair marks replaced tetrahedra dead and appends new ones *after*
the GPU compaction, and gDel3D's own checker skips dead ones.  The upstream binding returns the
whole `tetVec` without the flags, so dead tetrahedra show up as overlapping, non-Delaunay tets.

**`DelOutput.get_stats()`** -- gDel3D's phase timers (GPU: init/split/flip/relocate/sort, CPU: star
splaying) and, with the patch below, its predicate counters.

**Predicate counters.**  gDel3D decides every in-sphere test with a filtered floating-point
predicate first (`doInSphereFast`); the tetrahedra whose test the filter cannot decide are collected
and redone by `kerCheckDelaunayExact_Exact` in exact arithmetic with symbolic perturbation
(`doInSphereSoS`).  Upstream keeps no usable total: `_counterVec[CounterExact]` is a per-iteration
append index that `kerMarkRejectedFlips` zeroes again at the end of every iteration, and it is read
only when `verbose` is on.  This adds three counters that nothing else resets, accumulates them into
`Statistics` at the end of each flipping loop, and exposes them through `get_stats()`:

    predCheckNum    in-sphere tests evaluated with the floating-point filter -- the total
    exactCheckNum   of those, the ones the filter could not decide, redone in exact arithmetic
    exactTetNum     tetrahedra that needed at least one such exact test

Every exact evaluation is preceded by the filtered one that gave up on it, so exactCheckNum is a
subset of predCheckNum, not something to add to it.

They count the in-sphere predicate only -- the one that decides Delaunayhood.  gDel3D also filters
the orientation predicate, which is not counted here.

The counting costs one register increment per evaluation plus one shared and one global atomic per
block, i.e. it does not scale with the work done; the device->host reads happen once per flipping
loop, after `stopTiming`, so no timed iteration is synchronised by them.

Also makes setup.py honour TORCH_CUDA_ARCH_LIST so the extension can be built on a machine without
a GPU (a cluster login node).
"""

from __future__ import annotations

import os
import re
import sys

# ---------------------------------------------------------------------------------------
# bindings.cpp: get_tet_info / get_stats
# ---------------------------------------------------------------------------------------

DECL = "        py::array_t<int> getBoundaryTets(py::array_t<RealType> points);\n"
DECL_NEW = DECL + "        py::array_t<int8_t> getTetInfo();\n        py::dict getStats();\n"

IMPL_ANCHOR = "bool PyGDelOutput::checkCorrectness(py::array_t<RealType> points) {"
IMPL_NEW = (
    """py::array_t<int8_t> PyGDelOutput::getTetInfo() {
    // Per-tetrahedron status byte of gDel3D (bit 0: alive), aligned with the rows returned by compute().
    py::array_t<int8_t> result(output.tetInfoVec.size());
    py::buffer_info buf = result.request();
    int8_t* ptr = static_cast<int8_t*>(buf.ptr);
    for (size_t i = 0; i < output.tetInfoVec.size(); ++i) {
        ptr[i] = static_cast<int8_t>(output.tetInfoVec[i]);
    }
    return result;
}

py::dict PyGDelOutput::getStats() {
    // gDel3D's own phase timers (milliseconds).  GPU phases: init, split, flip, relocate, sort;
    // CPU phase: splaying (star splaying repair); out = device->host copy.
    py::dict d;
    d["totalTime"] = output.stats.totalTime;
    d["initTime"] = output.stats.initTime;
    d["splitTime"] = output.stats.splitTime;
    d["flipTime"] = output.stats.flipTime;
    d["relocateTime"] = output.stats.relocateTime;
    d["sortTime"] = output.stats.sortTime;
    d["outTime"] = output.stats.outTime;
    d["splayingTime"] = output.stats.splayingTime;
    d["totalFlipNum"] = output.stats.totalFlipNum;
    d["failVertNum"] = output.stats.failVertNum;
    d["finalStarNum"] = output.stats.finalStarNum;
COUNTER_LINES
    return d;
}

"""
    + IMPL_ANCHOR
)

COUNTER_LINES = (
    "    // In-sphere tests decided by the floating-point filter, those redone in exact arithmetic\n"
    "    // because the filter could not decide, and the tetrahedra that needed one of the latter\n"
    "    // (see patch_pygdel3d.py).\n"
    '    d["predCheckNum"] = output.stats.predCheckNum;\n'
    '    d["exactCheckNum"] = output.stats.exactCheckNum;\n'
    '    d["exactTetNum"] = output.stats.exactTetNum;\n'
)

BIND = '        .def("get_boundary_tets", &PyGDelOutput::getBoundaryTets)'
BIND_NEW = (
    BIND
    + '\n        .def("get_tet_info", &PyGDelOutput::getTetInfo)'
    + '\n        .def("get_stats", &PyGDelOutput::getStats)'
)

INCLUDE = "#include <iomanip>\n"
INCLUDE_NEW = INCLUDE + "#include <cstdint>\n"

SETUP_OLD = "if torch.cuda.is_available():\n    try:"
SETUP_NEW = (
    'if os.environ.get("TORCH_CUDA_ARCH_LIST"):\n'
    '    # e.g. "7.0;8.0" -> build for V100 and A100 without a GPU present (cluster login node)\n'
    '    for a in os.environ["TORCH_CUDA_ARCH_LIST"].replace(",", ";").split(";"):\n'
    '        a = a.strip().replace(".", "").replace("+PTX", "")\n'
    "        if a:\n"
    '            nvcc_args.append(f"-gencode=arch=compute_{a},code=sm_{a}")\n'
    '    print("TORCH_CUDA_ARCH_LIST:", os.environ["TORCH_CUDA_ARCH_LIST"])\n'
    "elif torch.cuda.is_available():\n    try:"
)


# ---------------------------------------------------------------------------------------
# gDel3D core: the predicate counters
# ---------------------------------------------------------------------------------------

# The anchors are matched with `\s*$`-tolerant regexes because the upstream sources have
# inconsistent trailing whitespace.
CORE_EDITS = [
    (
        "GPU/GPUDecl.h",
        r"enum Counter \{\s*\n\s*CounterExact,\s*\n\s*CounterFlip,\s*\n(\s*)CounterNum",
        "enum Counter {\n"
        "    CounterExact,\n"
        "    CounterFlip,\n"
        "    // patch_pygdel3d.py.  Unlike CounterExact, which kerMarkRejectedFlips zeroes at the\n"
        "    // end of every iteration, nothing resets these three inside a flipping loop.\n"
        "    CounterPred,        // in-sphere tests evaluated with the floating-point filter\n"
        "    CounterExactPred,   // in-sphere tests redone in exact arithmetic (SoS)\n"
        "    CounterExactTet,    // tetrahedra that needed at least one of those\n"
        "\\1CounterNum",
    ),
    # --- the filtered tests, in checkDelaunayFast<> ---------------------------------------
    (
        "GPU/KerPredicates.cu",
        r"(\n    __shared__ int s_num, s_offset;[ \t]*\n)",
        "\\1"
        "\n"
        "    // patch_pygdel3d.py: count the in-sphere tests this block decides with the float\n"
        "    // filter.  Per thread in a register, then one shared and one global atomic per block.\n"
        "    __shared__ int s_predNum;\n"
        "    int predNum = 0;\n"
        "\n"
        "    if ( THREAD_IDX == 0 )\n"
        "        s_predNum = 0;\n"
        "\n"
        "    __syncthreads();\n",
    ),
    (
        "GPU/KerPredicates.cu",
        r"(\n(\s*)const Side side = dPredWrapper\.doInSphereFast\()",
        "\n\\2++predNum;   // patch_pygdel3d.py\\1",
    ),
    (
        "GPU/KerPredicates.cu",
        r"(\n    if \( blockIdx\.x == 0 && threadIdx\.x == 0 \)[ \t]*\n    \{\s*\n\s*counterArr\[ CounterFlip \])",
        "\n"
        "    // patch_pygdel3d.py: publish this block's filtered-evaluation count.\n"
        "    atomicAdd( &s_predNum, predNum );\n"
        "\n"
        "    __syncthreads();\n"
        "\n"
        "    if ( THREAD_IDX == 0 )\n"
        "        atomicAdd( &counterArr[ CounterPred ], s_predNum );\n"
        "\\1",
    ),
    # --- the exact tests, in kerCheckDelaunayExact_Exact ----------------------------------
    (
        "GPU/KerPredicates.cu",
        r"(\n    const int exactNum = counterArr\[ CounterExact \];[ \t]*\n)",
        "\\1"
        "\n"
        "    // patch_pygdel3d.py: same scheme for the exact (symbolically perturbed) tests, and\n"
        "    // for the tetrahedra the filter sent here -- one per iteration of the loop below.\n"
        "    __shared__ int s_exactPredNum, s_exactTetNum;\n"
        "    int exactPredNum = 0, exactTetNum = 0;\n"
        "\n"
        "    if ( THREAD_IDX == 0 )\n"
        "    {\n"
        "        s_exactPredNum = 0;\n"
        "        s_exactTetNum  = 0;\n"
        "    }\n"
        "\n"
        "    __syncthreads();\n",
    ),
    (
        "GPU/KerPredicates.cu",
        r"(\n(\s*)int2 val    = exactCheckVi\[ idx \];[ \t]*\n)",
        "\\1\\2++exactTetNum;   // patch_pygdel3d.py\n",
    ),
    (
        "GPU/KerPredicates.cu",
        r"(\n(\s*)const Side side = dPredWrapper\.doInSphereSoS\()",
        "\n\\2++exactPredNum;   // patch_pygdel3d.py\\1",
    ),
    (
        "GPU/KerPredicates.cu",
        r"(\n        storeOpp\( oppArr, botTi, botOpp \);[ \t]*\n    \}\n)(\n    return;\n\}\n)",
        "\\1"
        "\n"
        "    // patch_pygdel3d.py: publish this block's exact-evaluation counts.\n"
        "    atomicAdd( &s_exactPredNum, exactPredNum );\n"
        "    atomicAdd( &s_exactTetNum, exactTetNum );\n"
        "\n"
        "    __syncthreads();\n"
        "\n"
        "    if ( THREAD_IDX == 0 )\n"
        "    {\n"
        "        atomicAdd( &counterArr[ CounterExactPred ], s_exactPredNum );\n"
        "        atomicAdd( &counterArr[ CounterExactTet ], s_exactTetNum );\n"
        "    }\n"
        "\\2",
    ),
    # --- Statistics ----------------------------------------------------------------------
    (
        "CommonTypes.h",
        r"(\n    int totalFlipNum;[ \t]*\n)",
        "\\1"
        "\n"
        "    // patch_pygdel3d.py: in-sphere tests evaluated with the floating-point filter, the\n"
        "    // subset of those it could not decide and that were redone in exact arithmetic,\n"
        "    // and the tetrahedra that needed one of the latter.\n"
        "    long long predCheckNum;\n"
        "    long long exactCheckNum;\n"
        "    long long exactTetNum;\n",
    ),
    (
        "CommonTypes.h",
        r"(\n        totalFlipNum    = 0;[ \t]*\n)",
        "\\1"
        "\n        predCheckNum    = 0;"
        "\n        exactCheckNum   = 0;"
        "\n        exactTetNum     = 0;\n",
    ),
    (
        "CommonTypes.h",
        r"(\n        totalFlipNum    \+= s\.totalFlipNum;[ \t]*\n)",
        "\\1"
        "\n        predCheckNum    += s.predCheckNum;"
        "\n        exactCheckNum   += s.exactCheckNum;"
        "\n        exactTetNum     += s.exactTetNum;\n",
    ),
    (
        "CommonTypes.h",
        r"(\n        totalFlipNum    /= div;[ \t]*\n)",
        "\\1"
        "\n        predCheckNum    /= div;"
        "\n        exactCheckNum   /= div;"
        "\n        exactTetNum     /= div;\n",
    ),
    # --- read them once per flipping loop ------------------------------------------------
    (
        "GpuDelaunay.cu",
        r"(\n    \} [ \t]*\n\n    stopTiming\( _output->stats\.flipTime \);[ \t]*\n)",
        "\\1"
        "\n"
        "    // patch_pygdel3d.py: _counterVec is zeroed at the top of this function and nothing\n"
        "    // resets these three slots inside the loop, so they hold this loop's totals.  Read\n"
        "    // after stopTiming: three 4-byte device->host copies, outside every timed iteration.\n"
        "    _output->stats.predCheckNum  += _counterVec[ CounterPred ];\n"
        "    _output->stats.exactCheckNum += _counterVec[ CounterExactPred ];\n"
        "    _output->stats.exactTetNum   += _counterVec[ CounterExactTet ];\n",
    ),
]


def _read(path: str) -> tuple[str, str]:
    """(text with LF endings, the file's own line ending).  Half the gDel3D sources are CRLF and
    rewriting them with LF would turn a five-line patch into a whole-file diff."""
    with open(path, newline="") as f:
        raw = f.read()
    eol = "\r\n" if raw.count("\r\n") * 2 > raw.count("\n") else "\n"
    return raw.replace("\r\n", "\n"), eol


def _write(path: str, text: str, eol: str) -> None:
    with open(path, "w", newline="") as f:
        f.write(text.replace("\n", eol) if eol != "\n" else text)


def patch_core(repo: str) -> bool:
    """Add the CounterPred slot, the counting, and the Statistics fields."""
    root = os.path.join(repo, "src", "gdel3d", "gDel3D")
    if not os.path.isdir(root):
        print(f"{root}: not found (is this a pyGDel3D checkout?)")
        return False
    if "CounterPred" in _read(os.path.join(root, "GPU", "GPUDecl.h"))[0]:
        print(f"{root}: predicate counters already patched")
        return True
    edits: dict[str, tuple[str, str]] = {}
    for rel, pattern, repl in CORE_EDITS:
        path = os.path.join(root, rel)
        text, eol = edits.get(path) or _read(path)
        out, n = re.subn(pattern, repl, text, count=1)
        if n != 1:
            print(f"{path}: anchor not found, upstream source changed:\n  {pattern}")
            return False
        edits[path] = (out, eol)
    for path, (text, eol) in edits.items():
        _write(path, text, eol)
        print(f"{path}: patched (predicate counters)")
    return True


def patch_setup(repo: str) -> bool:
    path = os.path.join(repo, "setup.py")
    with open(path) as f:
        src = f.read()
    if "TORCH_CUDA_ARCH_LIST" in src:
        print(f"{path}: already patched")
        return True
    if SETUP_OLD not in src:
        print(f"{path}: anchor not found, upstream setup.py changed")
        return False
    with open(path, "w") as f:
        f.write(src.replace(SETUP_OLD, SETUP_NEW, 1))
    print(f"{path}: patched (TORCH_CUDA_ARCH_LIST support)")
    return True


def patch_bindings(repo: str) -> bool:
    path = os.path.join(repo, "src", "gdel3d", "bindings.cpp")
    with open(path) as f:
        src = f.read()
    if "exactTetNum" in src:
        print(f"{path}: already patched")
        return True
    if "get_stats" in src:
        # patched by an older version of this script: only the counter lines are missing
        src, n = re.subn(
            r'(\n    d\["finalStarNum"\] = output\.stats\.finalStarNum;\n)',
            "\\1" + COUNTER_LINES.replace("\\", "\\\\"),
            src,
            count=1,
        )
        if n != 1:
            print(f"{path}: get_stats present but its anchor moved; revert the file and re-run")
            return False
        with open(path, "w") as f:
            f.write(src)
        print(f"{path}: patched (predicate counters added to get_stats)")
        return True
    if "get_tet_info" in src:  # patched by a much older version: start over from upstream
        print(f"{path}: partially patched; run `git checkout -- src/gdel3d/bindings.cpp` and re-run")
        return False
    for old in (DECL, IMPL_ANCHOR, BIND, INCLUDE):
        if old not in src:
            print(f"{path}: anchor not found, upstream bindings changed:\n{old}")
            return False
    src = src.replace(INCLUDE, INCLUDE_NEW, 1)
    src = src.replace(DECL, DECL_NEW, 1)
    src = src.replace(IMPL_ANCHOR, IMPL_NEW.replace("COUNTER_LINES\n", COUNTER_LINES), 1)
    src = src.replace(BIND, BIND_NEW, 1)
    with open(path, "w") as f:
        f.write(src)
    print(f"{path}: patched (get_tet_info, get_stats, predicate counters)")
    return True


def patch(repo: str) -> bool:
    # The core comes first: bindings.cpp only compiles once Statistics has the new fields.
    return patch_setup(repo) and patch_core(repo) and patch_bindings(repo)


if __name__ == "__main__":
    sys.exit(0 if patch(sys.argv[1] if len(sys.argv) > 1 else "pyGDel3D") else 1)
