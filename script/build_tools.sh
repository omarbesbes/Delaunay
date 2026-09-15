   Run this once on the LOGIN node, then resubmit the job:

     cd $(pwd) && bash script/build_tools.sh
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
# The tools are built with the environment's nvcc, CGAL headers and TBB, so it has to be active.
# When it is not, activate it here rather than telling the user to do it by hand.
if [ -z "${CONDA_PREFIX:-}" ] || ! command -v nvcc >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  . "$(dirname "$0")/env.sh" || exit 1
  PY=${PYTHON:-$(command -v python || command -v python3 || true)}
  CXX=${CXX:-x86_64-conda-linux-gnu-g++}
fi
missing=""
[ -n "$PY" ] || missing="$missing python"
command -v nvcc >/dev/null 2>&1 || missing="$missing nvcc"
[ -n "${CONDA_PREFIX:-}" ] || missing="$missing \$CONDA_PREFIX"
if [ -n "$missing" ]; then
  echo "missing:$missing -- the project environment could not be activated."
  echo "Run 'bash script/first_install.sh' first, or set DELAUNAY_ENV to an existing environment."
  exit 1
fi

# nvcc writes large intermediates to $TMPDIR.  On a shared login node /tmp is small and shared,
# and a full /tmp kills nvcc with a bus error (SIGBUS on a memory-mapped temp file) rather than a
# message.  Use the scratch space when the cluster provides it, the repository otherwise.
export TMPDIR=${TMPDIR:-${WORKDIR:-$PWD}/.cache/nvcc-tmp}
mkdir -p "$TMPDIR"

# Runs a CUDA build and, if the compiler itself dies (bus error, killed), says what that means on
# this cluster instead of leaving a bare "core dumped" line.
cuda_build() {
  "$@" && return 0
  local rc=$?
  if [ "$rc" -ge 128 ]; then
    cat >&2 <<MSG
-- the compiler was killed (exit $rc: signal $((rc - 128)))

   On a shared login node this is almost always the node, not the code: /tmp full, or the
   per-process memory limit hit by a large CUDA translation unit built for several
   architectures.  The same command has built fine on this cluster.  Try, in order:

     bash script/build_tools.sh                          # simply again (transient)
     GPU_ARCHS=8.0 bash script/build_tools.sh            # one architecture: half the memory
     TMPDIR=\$WORKDIR/tmp bash script/build_tools.sh     # if /tmp is the problem

   The A100 partition (gpua100) only needs sm_80, so GPU_ARCHS=8.0 costs nothing there.
MSG
  fi
  return "$rc"
}

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

   $dir is missing and cloning failed.  This almost always means the command is
   running on a compute node, which has no route to the internet.

   Run this once on the LOGIN node, then resubmit the job:

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
  cuda_build nvcc -O3 -std=c++17 -rdc=true -I$D/include $GENCODE -Xcompiler -fopenmp \
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
  cuda_build $PY script/patch_gstar4d.py third_party/gStar4D --build --run --arch="$GS_ARCHS" \
    --out=bin/gstar4d --cc="${CC:-cc}"
else
  echo "-- bin/gstar4d present"
fi

echo "-- tools:"
for b in bin/cgal_delaunay bin/cgal_delaunay_profile bin/dewall bin/gstar4d; do
  printf '   %-20s %s\n' "$b" "$([ -x "$b" ] && echo OK || echo MISSING)"
done
