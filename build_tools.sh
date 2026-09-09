#!/bin/bash
# Build the standalone tools if they are missing (idempotent): the parallel CGAL tool, Local DeWall
# and gStar4D.  Requires an activated environment (conda env with nvcc, CGAL headers, TBB).
# Usage: bash build_tools.sh [--force]
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin third_party
FORCE=${1:-}
ARCHS=${TORCH_CUDA_ARCH_LIST:-"7.0;8.0"}
CXX=${CXX:-x86_64-conda-linux-gnu-g++}
PY=${PYTHON:-$(command -v python || command -v python3 || true)}

# Every tool here is built with the conda environment's nvcc / CGAL headers / TBB, so the
# environment has to be active.  Say that plainly instead of failing on a missing command.
missing=""
[ -n "$PY" ] || missing="$missing python"
command -v nvcc >/dev/null 2>&1 || missing="$missing nvcc"
[ -n "${CONDA_PREFIX:-}" ] || missing="$missing \$CONDA_PREFIX"
if [ -n "$missing" ]; then
  echo "missing:$missing -- activate the environment first:"
  echo "  module load anaconda3/2023.09-0/none-none"
  echo "  source activate \${DELAUNAY_ENV:-\$WORKDIR/envs/delaunay}"
  echo "  export CUDA_HOME=\$CONDA_PREFIX CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++"
  echo "(or run 'bash setup_ruche.sh', which activates it and calls this script)"
  exit 1
fi

if [ "$FORCE" = "--force" ] || [ ! -x bin/cgal_delaunay ]; then
  echo "-- building bin/cgal_delaunay (CGAL parallel, TBB)"
  $CXX -O3 -std=c++17 -pthread -DCGAL_LINKED_WITH_TBB -I"$CONDA_PREFIX/include" cgal_delaunay.cpp \
    -o bin/cgal_delaunay -L"$CONDA_PREFIX/lib" -Wl,-rpath,"$CONDA_PREFIX/lib" \
    -ltbb -ltbbmalloc -lgmp -lmpfr -lpthread
else
  echo "-- bin/cgal_delaunay present"
fi

if [ "$FORCE" = "--force" ] || [ ! -x bin/dewall ]; then
  echo "-- building bin/dewall (Local DeWall, archs $ARCHS)"
  [ -d third_party/Local-DeWall ] || git clone -q https://github.com/WuhengGao/Local-DeWall.git third_party/Local-DeWall
  $PY patch_dewall.py third_party/Local-DeWall
  GENCODE=""
  for a in ${ARCHS//;/ }; do a=${a/./}; GENCODE="$GENCODE -gencode arch=compute_$a,code=sm_$a"; done
  D=third_party/Local-DeWall
  nvcc -O3 -std=c++17 -rdc=true -I$D/include $GENCODE -Xcompiler -fopenmp \
    -diag-suppress 20054,68 \
    $D/src/delaunay_kernels.cu $D/src/delaunay_solver.cu $D/src/spatial_hash.cu $D/src/main_delaunay.cu \
    -x cu $D/src/sampler.cpp -o bin/dewall -lgomp
else
  echo "-- bin/dewall present"
fi

if [ "$FORCE" = "--force" ] || [ ! -x bin/gstar4d ]; then
  echo "-- building bin/gstar4d (gStar4D, archs $ARCHS)"
  [ -d third_party/gStar4D ] || git clone -q https://github.com/ashwin/gStar4D.git third_party/gStar4D
  GS_ARCHS=$(echo "$ARCHS" | tr ';' ',' | tr -d '.')
  # patch_gstar4d.py ports the PBA stage off the texture-reference API (removed in CUDA 12) and
  # makes the PLY writer emit one 4-index tetrahedron per line at 9 significant digits; then
  # --build --run compiles predicates.c as C, then everything else with nvcc, echoing both
  # commands and checking each output file (no shell in between: the login node's BASH_ENV
  # sources the module system, which does not survive a piped-and-traced script)
  $PY patch_gstar4d.py third_party/gStar4D --build --run --arch="$GS_ARCHS" \
    --out=bin/gstar4d --cc="${CC:-cc}"
else
  echo "-- bin/gstar4d present"
fi

echo "-- tools:"
for b in bin/cgal_delaunay bin/dewall bin/gstar4d; do
  printf '   %-20s %s\n' "$b" "$([ -x "$b" ] && echo OK || echo MISSING)"
done
