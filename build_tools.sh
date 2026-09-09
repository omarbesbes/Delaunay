#!/bin/bash
# Build the two standalone tools if they are missing (idempotent).  Requires an activated
# environment (conda env with nvcc, CGAL headers, TBB).  Usage: bash build_tools.sh [--force]
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin third_party
FORCE=${1:-}
ARCHS=${TORCH_CUDA_ARCH_LIST:-"7.0;8.0"}
CXX=${CXX:-x86_64-conda-linux-gnu-g++}

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
  python patch_dewall.py third_party/Local-DeWall
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

echo "-- tools:"
for b in bin/cgal_delaunay bin/dewall; do
  printf '   %-20s %s\n' "$b" "$([ -x "$b" ] && echo OK || echo MISSING)"
done
