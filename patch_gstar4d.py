"""Make gStar4D (https://github.com/ashwin/gStar4D) build with CUDA 12 and emit parsable output.

    python patch_gstar4d.py /path/to/gStar4D                              # patch sources (idempotent)
    python patch_gstar4d.py /path/to/gStar4D --build --arch=70,80          # ... print the build commands
    python patch_gstar4d.py /path/to/gStar4D --build --run --arch=70,80    # ... and run them

`--arch` takes a comma-separated list of GPU architectures, `--out=` names the binary and `--cc=`
the C compiler used for Shewchuk's predicates (default: $CC).  Prefer `--run` over piping the
printed commands into a shell: it runs them in order, in this process, and checks that each output
file was actually produced.

gStar4D is from 2013 and needs three things before it can be used in this benchmark:

1. **Texture references.**  Its PBA stage (the discrete Voronoi diagram that seeds the stars)
   declares `texture<int>` / `texture<short>` references, binds them with `cudaBindTexture` and
   reads them with `tex1Dfetch`.  That whole API was **removed in CUDA 12**, so the code does not
   compile at all.  The three references are read-only fetches from *linear* device memory, so
   they are replaced by plain `__device__` pointers set with `cudaMemcpyToSymbol`, and
   `tex1Dfetch(tex, i)` becomes `__ldg(&tex[i])` (a read-only cached load, which is what the
   texture path was there for).  Numerically identical, one pointer copy per bind instead of a
   texture bind.

2. **A parsable tetrahedron list.**  `-outFile` writes an ASCII PLY in which every tetrahedron is
   split into *three triangular faces* ({0,1,2}, {0,1,3}, {0,2,3}), and the vertex block is
   written with the default `ostream` precision of 6 significant digits.  Six digits is not enough
   to identify which input point a vertex is (gStar4D scales the input into its grid and
   Morton-sorts it, so the benchmark has to match coordinates back to input indices).  The writer
   is changed to emit one 4-index face per tetrahedron and 9 significant digits, which round-trips
   float32 exactly.

3. **An `sm_35` build.**  The CMakeLists hardcodes `-gencode arch=compute_35,code=sm_35` (dropped
   in CUDA 12) and uses `find_package(CUDA)`/`cuda_add_executable`, removed in CMake 4.  This
   script prints a plain nvcc command line instead (as patch_dewall.py does for Local DeWall).
   Every source has to be compiled as CUDA, including the four `.cpp` files: they include
   `<thrust/count.h>` through `Geometry.h`, and a modern Thrust pulls CUB's device code into the
   translation unit, so compiling them as plain host C++ fails with hundreds of errors about
   `threadIdx` and `__syncthreads` being undeclared.  That cannot be done with `nvcc -x cu`,
   because unlike `gcc -x`, nvcc's `-x` is global rather than positional and would also apply to
   the pre-compiled `predicates.o` on the command line (nvcc then feeds the object file to the
   host compiler as source, which reports thousands of "null character(s) ignored" warnings).  So
   this script creates a `.cu` symlink next to each of those four files and compiles those
   instead, letting the suffix pick the language, file by file.

`-fmad=false` in that command line is required, not cosmetic: `GDelShewchukDevice.h` implements
Shewchuk's exact predicates on the GPU with Two_Product/Split, whose error-free transformations
break if the compiler contracts `a*b + c` into an FMA.  Shewchuk's host `predicates.c` is compiled
as C, separately, with the same guarantee (`-ffp-contract=off -fno-fast-math`).

Note that gStar4D ships with `CUDA_CHECK_ERROR` enabled in `Common/CudaWrapper.h`, which puts a
`cudaDeviceSynchronize()` after every kernel launch.  That is left alone: it is how the published
timings were produced, and gStar4D's phases are sequential anyway.

Usage of the resulting binary:
    gstar4d -inFile points.txt -outFile out.ply -g 256 [-timing] [-stats] [-check]
    points.txt: whitespace-separated coordinates, 3 per point (no count header).
    The tool drops duplicate points, scales every coordinate into [1, gridSize-2] with a single
    global (uniform) affine map, Morton-sorts the points, and writes the *scaled and sorted*
    points as the PLY vertex block, so tetrahedron indices refer to that block.
    Phase timings go to stdout in milliseconds (Init / PBA / InitStar / Consistency / StarOutput /
    Total Time) and are appended to gStar4D-time.txt in the working directory.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys

TEX_DECLS = "texture<int> pbaTexColor; \ntexture<int> pbaTexLinks; \ntexture<short> pbaTexPointer; \n"

TEX_DECLS_NEW = """// CUDA 12 removed the texture reference API (texture<>, cudaBindTexture, tex1Dfetch).  These
// three references were read-only fetches from linear int/short device memory, so they become
// plain device pointers, set with cudaMemcpyToSymbol by PBA_BIND below and read with __ldg() in
// pba3DKernel.h.  See patch_gstar4d.py.
__device__ int*   pbaTexColor;
__device__ int*   pbaTexLinks;
__device__ short* pbaTexPointer;

#define PBA_BIND( sym, ptr )                                                     \\
    do {                                                                         \\
        void* _pbaPtr = ( void* ) ( ptr );                                       \\
        CudaSafeCall( cudaMemcpyToSymbol( sym, &_pbaPtr, sizeof( void* ) ) );    \\
    } while ( 0 )
"""

TEXFETCH_ANCHOR = "#define TOID(x, y, z, w)"
TEXFETCH_NEW = (
    '// The pbaTex* "textures" are plain __device__ pointers now (see patch_gstar4d.py); keep the\n'
    "// call sites unchanged by turning tex1Dfetch() into a read-only cached load.\n"
    "#define tex1Dfetch( tex, idx ) ( __ldg( &( tex )[ ( idx ) ] ) )\n\n"
    + TEXFETCH_ANCHOR
)

PLY_HEADER = '    outFile << "element face " << tetraNum * 3 << endl;'
PLY_HEADER_NEW = '    outFile << "element face " << tetraNum << endl;  // one 4-index face per tetrahedron'

PLY_PRECISION_ANCHOR = "    const int pointNum = _pointVec.size();"
PLY_PRECISION_NEW = (
    "    // 9 significant digits round-trip float32 exactly, so the benchmark can match these\n"
    "    // scaled, Morton-sorted vertices back to its own input points (patch_gstar4d.py).\n"
    "    outFile.precision( 9 );\n\n" + PLY_PRECISION_ANCHOR
)

PLY_FACES = """    const int Faces[3][3] = {
        { 0, 1, 2 },
        { 0, 1, 3 },
        { 0, 2, 3 } };

    for ( int ti = 0; ti < tetraNum; ++ti )
    {
        const Tetrahedron& tet = _tetraVec[ ti ];

        for ( int fi = 0; fi < 3; ++fi )
        {
            outFile << "3 ";

            for ( int vi = 0; vi < 3; ++vi )
            {
                outFile << tet._v[ Faces[ fi ][ vi ] ] << " ";
            }

            outFile << endl;
        }
"""

PLY_FACES_NEW = """    // One line per tetrahedron ("4 v0 v1 v2 v3") instead of three triangular faces, so that the
    // tetrahedra can be read back without guessing which triples belong together (patch_gstar4d.py).
    for ( int ti = 0; ti < tetraNum; ++ti )
    {
        const Tetrahedron& tet = _tetraVec[ ti ];

        outFile << "4";

        for ( int vi = 0; vi < 4; ++vi )
        {
            outFile << " " << tet._v[ vi ];
        }

        outFile << endl;
"""

# These four are C++ files that must nevertheless be compiled as CUDA (see the note above), which
# is arranged by compiling them through a sibling symlink with a .cu suffix, created by patch().
CPP_AS_CU = [
    "GDelaunay/Common/DtRandom.cpp",
    "GDelaunay/GDelaunay/GDelaunay.cpp",
    "GDelaunay/Main/Application.cpp",
    "GDelaunay/Main/Main.cpp",
]
CU_SOURCES = [
    "GDelaunay/Common/Geometry.cu",
    "GDelaunay/GDelaunay/GDelCommon.cu",
    "GDelaunay/GDelaunay/GDelData.cu",
    "GDelaunay/GDelaunay/GDelHost.cu",
    "GDelaunay/GDelaunay/GDelKernels.cu",
    "GDelaunay/GDelaunay/GDelPredKernels.cu",
    "GDelaunay/PBA/Pba.cu",
    "GDelaunay/PBA/pba3DHost.cu",
] + [c[: -len(".cpp")] + ".cu" for c in CPP_AS_CU]
INCLUDE_DIRS = [
    "GDelaunay/Common",
    "GDelaunay/GDelaunay",
    "GDelaunay/Main",
    "GDelaunay/PBA",
]


def _split_top_level(text: str) -> list[str]:
    """Split `a, b[c, d]` on commas that are not inside brackets or parentheses."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts]


def _rewrite_binds(src: str) -> tuple[str, int]:
    """cudaBindTexture(0, tex, ptr[, size]) -> PBA_BIND(tex, ptr); cudaUnbindTexture -> nothing."""
    out, changed = [], 0
    for line in src.splitlines(keepends=True):
        m = re.match(
            r"^(\s*)CudaSafeCall\(\s*cudaBindTexture\(\s*0\s*,(.*)\)\s*\)\s*;\s*$", line
        )
        if m:
            args = _split_top_level(m.group(2))
            out.append(f"{m.group(1)}PBA_BIND( {args[0]}, {args[1]} );\n")
            changed += 1
            continue
        if re.match(r"^\s*CudaSafeCall\(\s*cudaUnbindTexture\(", line):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(
                f"{indent}// cudaUnbindTexture removed: plain device pointers (patch_gstar4d.py)\n"
            )
            changed += 1
            continue
        out.append(line)
    return "".join(out), changed


def build_commands(
    root: str, arch: str = "80", out: str = "gstar4d", cc: str = "cc"
) -> list[str]:
    """The two commands that build the binary: Shewchuk's predicates as C, then everything else."""
    root = os.path.abspath(root)
    obj = os.path.join(
        os.path.dirname(os.path.abspath(out)) or ".", "gstar4d_predicates.o"
    )
    predicates = os.path.join(root, "GDelaunay", "Common", "predicates.c")
    includes = " ".join(f"-I{os.path.join(root, d)}" for d in INCLUDE_DIRS)
    srcs = " ".join(os.path.join(root, s) for s in CU_SOURCES)
    # arch may be a comma-separated list ("70,80") to build one binary for several GPUs
    gencode = " ".join(
        f"-gencode arch=compute_{a.strip()},code=sm_{a.strip()}"
        for a in arch.split(",")
        if a.strip()
    )
    return [
        # exact predicates: no FMA contraction, no fast math, and never -ffast-math from CFLAGS
        f"{cc} -O2 -fno-fast-math -ffp-contract=off -c {predicates} -o {obj}",
        (
            f"nvcc -O3 -std=c++17 -DNDEBUG -fmad=false -Wno-deprecated-gpu-targets {gencode} "
            f"{includes} -Xcompiler -Wno-unknown-pragmas {srcs} {obj} -o {out}"
        ),
    ]


def run_build(commands: list[str]) -> bool:
    """Run the build commands in order, echoing each one, and stop at the first failure."""
    for cmd in commands:
        print(f"\n+ {cmd}", flush=True)
        argv = shlex.split(cmd)
        rc = subprocess.run(argv, check=False).returncode
        if rc != 0:
            print(f"FAILED (exit {rc}): {argv[0]}")
            return False
        target = argv[argv.index("-o") + 1] if "-o" in argv else None
        if target and not os.path.exists(target):
            print(f"FAILED: {argv[0]} exited 0 but did not produce {target}")
            return False
    return True


def patch(root: str) -> bool:
    ok = True

    host = os.path.join(root, "GDelaunay", "PBA", "pba3DHost.cu")
    with open(host) as f:
        src = f.read()
    if "PBA_BIND" in src:
        print("GDelaunay/PBA/pba3DHost.cu: already patched")
    elif TEX_DECLS not in src:
        print(
            f"GDelaunay/PBA/pba3DHost.cu: texture declarations not found:\n{TEX_DECLS}"
        )
        ok = False
    else:
        src = src.replace(TEX_DECLS, TEX_DECLS_NEW, 1)
        src, n = _rewrite_binds(src)
        with open(host, "w") as f:
            f.write(src)
        print(
            f"GDelaunay/PBA/pba3DHost.cu: textures -> device pointers, {n} bind/unbind site(s)"
        )
        leftover = [
            ln
            for ln in src.splitlines()
            if re.search(r"cuda(?:Un)?BindTexture\s*\(", ln)
            and not ln.lstrip().startswith("//")
        ]
        if leftover:
            print(
                "GDelaunay/PBA/pba3DHost.cu: WARNING some texture binds were not rewritten"
            )
            ok = False

    kern = os.path.join(root, "GDelaunay", "PBA", "pba3DKernel.h")
    with open(kern) as f:
        src = f.read()
    if "define tex1Dfetch" in src:
        print("GDelaunay/PBA/pba3DKernel.h: already patched")
    elif TEXFETCH_ANCHOR not in src:
        print("GDelaunay/PBA/pba3DKernel.h: anchor not found (upstream changed?)")
        ok = False
    else:
        with open(kern, "w") as f:
            f.write(src.replace(TEXFETCH_ANCHOR, TEXFETCH_NEW, 1))
        print("GDelaunay/PBA/pba3DKernel.h: tex1Dfetch -> __ldg")

    made = 0
    for rel in CPP_AS_CU:
        link = os.path.join(root, rel[: -len(".cpp")] + ".cu")
        target = os.path.basename(rel)
        if os.path.lexists(link):
            continue
        try:
            os.symlink(target, link)
        except OSError:  # no symlinks on this filesystem: a copy works as well
            with open(os.path.join(root, rel), "rb") as f_in, open(link, "wb") as f_out:
                f_out.write(f_in.read())
        made += 1
    print(
        f"{len(CPP_AS_CU)} .cpp file(s) reachable as .cu ({made} created): compiled as CUDA, "
        "as their Thrust includes require"
    )

    geom = os.path.join(root, "GDelaunay", "Common", "Geometry.cu")
    with open(geom) as f:
        src = f.read()
    if "one 4-index face per tetrahedron" in src:
        print("GDelaunay/Common/Geometry.cu: already patched")
    else:
        for old, new in (
            (PLY_PRECISION_ANCHOR, PLY_PRECISION_NEW),
            (PLY_HEADER, PLY_HEADER_NEW),
            (PLY_FACES, PLY_FACES_NEW),
        ):
            if old not in src:
                print(f"GDelaunay/Common/Geometry.cu: anchor not found:\n{old}")
                ok = False
                break
            src = src.replace(old, new, 1)
        else:
            with open(geom, "w") as f:
                f.write(src)
            print(
                "GDelaunay/Common/Geometry.cu: PLY writer emits 4-index tetrahedra at 9 digits"
            )

    return ok


if __name__ == "__main__":
    argv = sys.argv[1:]
    root = next((a for a in argv if not a.startswith("--")), "gStar4D")
    good = patch(root)
    if "--build" in argv:
        arch = next((a.split("=", 1)[1] for a in argv if a.startswith("--arch=")), "80")
        out = next(
            (a.split("=", 1)[1] for a in argv if a.startswith("--out=")), "gstar4d"
        )
        cc = next(
            (a.split("=", 1)[1] for a in argv if a.startswith("--cc=")),
            os.environ.get("CC", "cc"),
        )
        commands = build_commands(root, arch, out, cc)
        if "--run" in argv:
            good = good and run_build(commands)
        else:
            print("\n".join(commands))
    sys.exit(0 if good else 1)
