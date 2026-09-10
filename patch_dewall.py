"""Make Local-DeWall (https://github.com/WuhengGao/Local-DeWall) build on Linux and print a build line.

    python patch_dewall.py /path/to/Local-DeWall          # patch sources (idempotent)
    python patch_dewall.py /path/to/Local-DeWall --build  # ... and print the nvcc command line

The upstream project ships a Visual Studio solution only.  On Linux with nvcc it needs:
  * `throw std::exception("...")` -> `throw std::runtime_error("...")` (MSVC extension);
  * `std::ios::in | (binary ? std::ios::binary : 0)` -> a proper openmode expression (MSVC extension);
  * a forward declaration of `plane_intersection`, used in a template before it is defined;
  * relocatable device code (`-rdc=true`, the .vcxproj sets GenerateRelocatableDeviceCode);
  * OpenMP for the host loops in spatial_hash.cu / sampler.cpp (`-Xcompiler -fopenmp -lgomp`);
  * sampler.cpp compiled by nvcc as CUDA (`-x cu`), as the .vcxproj does (it includes cuda_utils.cuh).
Usage of the resulting binary (see README):  dewall <output-prefix> <input.txt> [--no-normalize]
  input.txt: first line N, then one "x y z" per line; coordinates are normalised to (0,1)^3;
  outputs <prefix>x.bin (sorted normalised points, float32) and <prefix>t.bin (int32 tets, 4 per
  row, indices into x.bin; indices >= N are the points at infinity); both files start with two
  uint64 (rows, cols).
The patched build allocates DEWALL_TETS_PER_POINT (16) tetrahedra per point instead of 7; pass
-DDEWALL_TETS_PER_POINT=<n> to nvcc to change it, and set the same value in the environment so
that the benchmark can tell a truncated result from a complete one.
"""

from __future__ import annotations

import os
import sys

EDITS = [
    (
        "include/cuda_utils.cuh",
        [
            ("#include <cassert>\n", "#include <cassert>\n#include <stdexcept>\n"),
            (
                '\t\tthrow std::exception("cuda runtime error");',
                '\t\tthrow std::runtime_error("cuda runtime error");',
            ),
            # openmode | int is an MSVC extension; libstdc++ needs an openmode on both sides
            (
                "infile.open(fname, std::ios::in | (binary ? std::ios::binary : 0));",
                "infile.open(fname, binary ? (std::ios::in | std::ios::binary) : std::ios::in);",
            ),
            (
                "outfile.open(fname, std::ios::out | (binary ? std::ios::binary : 0));",
                "outfile.open(fname, binary ? (std::ios::out | std::ios::binary) : std::ios::out);",
            ),
        ],
    ),
    (
        "src/main_delaunay.cu",
        [
            # optional 3rd argument "--no-normalize": take the input as is (must already lie in [0,1)^3),
            # so that a benchmark can feed every method exactly the same float32 point set
            ("#include <iostream>\n", "#include <iostream>\n#include <string>\n"),
            (
                "        if (!Sampler::load_file(argv[2], pts, true)) {",
                (
                    "        const bool normalize = !(argc > 3 && std::string(argv[3]) == "
                    'std::string("--no-normalize"));\n'
                    "        if (!Sampler::load_file(argv[2], pts, normalize)) {"
                ),
            ),
        ],
    ),
    (
        "src/delaunay_solver.cu",
        [
            # The tetrahedron array is allocated as 7 per point with no bound check, so a
            # triangulation with more than that is silently truncated: on 20k-point tube surfaces
            # (8-12 tets per point) the tool returned exactly 7*nv+1 tetrahedra and a mesh full of
            # holes.  int4 is 16 bytes, so 16 per point costs 256 MB at a million points.
            (
                "    tet = cuVector<int4>(7 * nv);",
                (
                    "#ifndef DEWALL_TETS_PER_POINT\n"
                    "#define DEWALL_TETS_PER_POINT 16   // was a hard-coded 7 (patch_dewall.py)\n"
                    "#endif\n"
                    "    tet = cuVector<int4>((size_t) DEWALL_TETS_PER_POINT * nv);"
                ),
            ),
        ],
    ),
    (
        "src/delaunay_kernels.cu",
        [
            # used inside a template before its definition: MSVC accepts it, nvcc/GCC need a declaration
            (
                '#include "geometry.h"\n',
                '#include "geometry.h"\n__device__ void plane_intersection(REAL* plane, REAL* lowboundr, REAL* upboundr);\n',
            ),
        ],
    ),
]

SOURCES = [
    "src/delaunay_kernels.cu",
    "src/delaunay_solver.cu",
    "src/spatial_hash.cu",
    "src/main_delaunay.cu",
]


def build_command(root: str, arch: str = "75", out: str = "dewall") -> str:
    srcs = " ".join(os.path.join(root, s) for s in SOURCES)
    return (
        f"nvcc -O3 -std=c++17 -rdc=true -I{os.path.join(root, 'include')} "
        f"-gencode arch=compute_{arch},code=sm_{arch} -Xcompiler -fopenmp "
        f"{srcs} -x cu {os.path.join(root, 'src', 'sampler.cpp')} -o {out} -lgomp"
    )


def patch(root: str) -> bool:
    ok = True
    for rel, edits in EDITS:
        path = os.path.join(root, rel)
        with open(path) as f:
            src = f.read()
        changed = 0
        for old, new in edits:
            if new in src:
                continue
            if old not in src:
                print(f"{rel}: anchor not found (upstream changed?):\n{old}")
                ok = False
                continue
            src = src.replace(old, new, 1)
            changed += 1
        if changed:
            with open(path, "w") as f:
                f.write(src)
        print(
            f"{rel}: {changed} edit(s) applied"
            if changed
            else f"{rel}: already patched"
        )
    return ok


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = args[0] if args else "Local-DeWall"
    good = patch(root)
    if "--build" in sys.argv:
        arch = next(
            (a.split("=", 1)[1] for a in sys.argv if a.startswith("--arch=")), "75"
        )
        print(build_command(root, arch))
    sys.exit(0 if good else 1)
