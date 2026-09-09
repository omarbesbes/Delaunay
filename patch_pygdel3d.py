"""Patch pyGDel3D's bindings so the Python side can drop dead tetrahedra.

    python patch_pygdel3d.py /path/to/pyGDel3D

gDel3D keeps a per-tetrahedron status byte (`tetInfoVec`, bit 0 = alive).  Its CPU star-splaying
repair step marks replaced tetrahedra dead and appends new ones *after* the GPU compaction, and
gDel3D's own checker skips dead ones.  The pyGDel3D binding returns the whole `tetVec` without the
flags, so dead tetrahedra show up as overlapping, non-Delaunay tets.  This adds
`DelOutput.get_tet_info()` returning the status bytes as an int8 array, `DelOutput.get_stats()` with
gDel3D's phase timers (GPU: init/split/flip/relocate/sort, CPU: star splaying), and makes setup.py honour
TORCH_CUDA_ARCH_LIST so the extension can be built on a machine without a GPU.  Idempotent.
"""

from __future__ import annotations

import os
import sys

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
    return d;
}

"""
    + IMPL_ANCHOR
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


def patch(repo: str) -> bool:
    if not patch_setup(repo):
        return False
    path = os.path.join(repo, "src", "gdel3d", "bindings.cpp")
    with open(path) as f:
        src = f.read()
    if "get_stats" in src:
        print(f"{path}: already patched")
        return True
    if "get_tet_info" in src:  # patched by an older version of this script: start over from upstream
        print(f"{path}: patched by an older version; run `git checkout -- src/gdel3d/bindings.cpp` and re-run")
        return False
    for old in (DECL, IMPL_ANCHOR, BIND, INCLUDE):
        if old not in src:
            print(f"{path}: anchor not found, upstream bindings changed:\n{old}")
            return False
    src = src.replace(INCLUDE, INCLUDE_NEW, 1)
    src = src.replace(DECL, DECL_NEW, 1)
    src = src.replace(IMPL_ANCHOR, IMPL_NEW, 1)
    src = src.replace(BIND, BIND_NEW, 1)
    with open(path, "w") as f:
        f.write(src)
    print(f"{path}: patched (added DelOutput.get_tet_info)")
    return True


if __name__ == "__main__":
    sys.exit(0 if patch(sys.argv[1] if len(sys.argv) > 1 else "pyGDel3D") else 1)
