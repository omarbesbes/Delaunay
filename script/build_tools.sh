#!/bin/bash
# Build the standalone tools if they are missing (idempotent): the parallel CGAL tool, Local DeWall
# and gStar4D.  Requires an activated environment (conda env with nvcc, CGAL headers, TBB).
# Usage: bash script/build_tools.sh [--force]
set -euo pipefail
# the script lives in script/ but every path below is relative to the repository root
cd "$(dirname "$0")/.."
mkdir -p bin third_party
FORCE=${1:-}
# GPU architectures for the standalone CUDA tools, e.g. GPU_ARCHS="7.0;8.0" (V100 and A100).
# Deliberately NOT TORCH_CUDA_ARCH_LIST: that variable belongs to torch's extension builds and
# often holds torch's full default list, with "+PTX" entries and architectures this nvcc rejects.
ARCHS=${GPU_ARCHS:-"7.0;8.0"}
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
  echo "(or run 'bash script/first_install.sh', which activates it and calls this script)"
  exit 1
fi

# Keep only real X.Y entries (dropping any "+PTX" suffix) that this nvcc actually supports.
SUPPORTED=$(nvcc --list-gpu-arch 2>/dev/null | sed 's/compute_//' | paste -sd, -)
SM=""
for a in $(echo "$ARCHS" | tr ';,' '  '); do
  a=${a%+PTX}
  case "$a" in *.*) ;; *) echo "   ignoring arch '$a' (not X.Y)"; continue ;; esac
  n=${a//./}
  if [ -n "$SUPPORTED" ] && ! echo ",$SUPPORTED," | grep -q ",$n,"; then
    echo "   ignoring sm_$n (this nvcc supports: $SUPPORTED)"
    continue
  fi
  SM="$SM $n"
done
SM=$(echo $SM | tr ' ' '\n' | awk '!seen[$0]++' | paste -sd' ' -)
[ -n "$SM" ] || { echo "no usable GPU architecture in GPU_ARCHS='$ARCHS'"; exit 1; }
GENCODE=""
for n in $SM; do GENCODE="$GENCODE -gencode arch=compute_$n,code=sm_$n"; done
GS_ARCHS=$(echo $SM | tr ' ' ',')
echo "-- GPU architectures: $(echo $SM | tr ' ' ',')"

# The compute nodes have no route to the internet, so a source that is not already in
# third_party/ cannot be fetched from inside a job.  Say that, instead of letting git fail with
# "Could not resolve host".
clone_or_explain() {  # clone_or_explain <url> <dir>
  local url=$1 dir=$2
  [ -d "$dir" ] && return 0
  if git clone -q "$url" "$dir" 2>/dev/null; then
    return 0
  fi
  cat >&2 <<MSG
-- cannot fetch $url

   $dir is missing and cloning failed.  On Ruche this almost always means the
   command is running on a compute node, which has no internet access.

   Run this once on the LOGIN node, then resubmit the job:

     module load anaconda3/2023.09-0/none-none
     source activate \$WORKDIR/envs/delaunay
     export CUDA_HOME=\$CONDA_PREFIX CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++
     cd $(pwd) && bash script/build_tools.sh
MSG
  return 1
}

if [ "$FORCE" = "--force" ] || [ ! -x bin/cgal_delaunay ]; then
  echo "-- building bin/cgal_delaunay (CGAL parallel, TBB)"
  $CXX -O3 -std=c++17 -pthread -DCGAL_LINKED_WITH_TBB -I"$CONDA_PREFIX/include" src/cgal_delaunay.cpp \
    -o bin/cgal_delaunay -L"$CONDA_PREFIX/lib" -Wl,-rpath,"$CONDA_PREFIX/lib" \
    -ltbb -ltbbmalloc -lgmp -lmpfr -lpthread
else
  echo "-- bin/cgal_delaunay present"
fi

# Same tool with CGAL's own predicate profiling switched on.  CGAL_PROFILE makes every filtered
# predicate count its calls and the calls its floating-point filter could not decide (the ones that
# fall back to exact arithmetic), and dump the totals to stderr at exit.  That costs time, so it is
# a separate binary: timings come from bin/cgal_delaunay, counts from this one.
if [ "$FORCE" = "--force" ] || [ ! -x bin/cgal_delaunay_profile ]; then
  echo "-- building bin/cgal_delaunay_profile (CGAL parallel, TBB, CGAL_PROFILE)"
  $CXX -O3 -std=c++17 -pthread -DCGAL_LINKED_WITH_TBB -DCGAL_PROFILE -I"$CONDA_PREFIX/include" src/cgal_delaunay.cpp \
    -o bin/cgal_delaunay_profile -L"$CONDA_PREFIX/lib" -Wl,-rpath,"$CONDA_PREFIX/lib" \
    -ltbb -ltbbmalloc -lgmp -lmpfr -lpthread
else
  echo "-- bin/cgal_delaunay_profile present"
fi

if [ "$FORCE" = "--force" ] || [ ! -x bin/dewall ]; then
  echo "-- building bin/dewall (Local DeWall, archs $GS_ARCHS)"
  clone_or_explain https://github.com/WuhengGao/Local-DeWall.git third_party/Local-DeWall
  $PY script/patch_dewall.py third_party/Local-DeWall
  D=third_party/Local-DeWall
  nvcc -O3 -std=c++17 -rdc=true -I$D/include $GENCODE -Xcompiler -fopenmp \
    -diag-suppress 20054,68 \
    $D/src/delaunay_kernels.cu $D/src/delaunay_solver.cu $D/src/spatial_hash.cu $D/src/main_delaunay.cu \
    -x cu $D/src/sampler.cpp -o bin/dewall -lgomp
else
  echo "-- bin/dewall present"
fi

if [ "$FORCE" = "--force" ] || [ ! -x bin/gstar4d ]; then
  echo "-- building bin/gstar4d (gStar4D, archs $GS_ARCHS)"
  clone_or_explain https://github.com/ashwin/gStar4D.git third_party/gStar4D
  # patch_gstar4d.py ports the PBA stage off the texture-reference API (removed in CUDA 12) and
  # makes the PLY writer emit one 4-index tetrahedron per line at 9 significant digits; then
  # --build --run compiles predicates.c as C, then everything else with nvcc, echoing both
  # commands and checking each output file (no shell in between: the login node's BASH_ENV
  # sources the module system, which does not survive a piped-and-traced script)
  $PY script/patch_gstar4d.py third_party/gStar4D --build --run --arch="$GS_ARCHS" \
    --out=bin/gstar4d --cc="${CC:-cc}"
else
  echo "-- bin/gstar4d present"
fi

echo "-- tools:"
for b in bin/cgal_delaunay bin/cgal_delaunay_profile bin/dewall bin/gstar4d; do
  printf '   %-20s %s\n' "$b" "$([ -x "$b" ] && echo OK || echo MISSING)"
done
