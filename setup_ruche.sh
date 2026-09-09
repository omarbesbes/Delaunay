#!/bin/bash
# One-time environment setup on the Ruche LOGIN node (needs internet; the compute nodes may not have it).
# Creates a self-contained conda env in $WORKDIR with Python 3.12, CUDA 12.8 (nvcc), GCC 13, torch cu128,
# CGAL headers + TBB, then builds: the parallel CGAL tool, Paragram (patched), pyGDel3D (patched),
# Local DeWall (patched), gStar4D (patched) and GeoDel.
# Usage:  bash setup_ruche.sh        (takes ~15-30 min)
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
mkdir -p bin third_party results    # git-ignored, absent in a fresh clone
: "${WORKDIR:?WORKDIR is not set (are you on Ruche?)}"
ENV=${DELAUNAY_ENV:-$WORKDIR/envs/delaunay}
ARCHS=${TORCH_CUDA_ARCH_LIST:-"7.0;8.0"}       # V100 (gpu, gpu_test) and A100 (gpua100)

echo "== [1/7] conda environment: $ENV"
module purge
set +u; module load anaconda3/2023.09-0/none-none; set -u
export CONDA_PKGS_DIRS=$WORKDIR/.conda/pkgs      # keep the 50 GB home quota free
mkdir -p "$CONDA_PKGS_DIRS"
if [ ! -d "$ENV" ]; then
  conda create -y -p "$ENV" -c conda-forge \
    python=3.12 "cuda-toolkit=12.8" "cuda-nvcc=12.8" gxx_linux-64=13 gcc_linux-64=13 \
    cmake ninja tbb tbb-devel gmp mpfr cgal-cpp boost-cpp git
fi
# conda activation scripts (cuda-nvcc) read variables that may be unset: relax "set -u" around them
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
set +u
source activate "$ENV"
set -u
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST=$ARCHS

# conda-forge keeps the CUDA headers and libraries under $CONDA_PREFIX/targets/x86_64-linux, which
# torch's JIT build does not add when compiling *host* code (ext.cpp): make them visible here.
CUDA_TARGET=$CONDA_PREFIX/targets/x86_64-linux
[ -d "$CUDA_TARGET/include" ] && export CPATH="$CUDA_TARGET/include${CPATH:+:$CPATH}"
[ -d "$CUDA_TARGET/lib" ] && export LIBRARY_PATH="$CUDA_TARGET/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
# the driver stub is a last resort for -lcuda; the real libcuda.so of the GPU node wins if present
[ -d "$CUDA_TARGET/lib/stubs" ] && export LIBRARY_PATH="${LIBRARY_PATH:+$LIBRARY_PATH:}$CUDA_TARGET/lib/stubs"
echo "cuda_runtime_api.h: $(ls "$CUDA_TARGET/include/cuda_runtime_api.h" 2>/dev/null || ls "$CONDA_PREFIX/include/cuda_runtime_api.h" 2>/dev/null || echo NOT FOUND)"
export CC=${CC:-x86_64-conda-linux-gnu-gcc}
export CXX=${CXX:-x86_64-conda-linux-gnu-g++}
echo "python: $(python -V) | nvcc: $(nvcc --version | tail -1) | host compiler: $($CXX --version | head -1)"

echo "== [2/7] python packages (torch cu128, numpy, scipy, matplotlib, cgal bindings)"
# cu128 wheels have no Volta (sm_70) kernels: for the V100 partitions (gpu, gpu_test) install
# the cu126 build instead with  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash setup_ruche.sh
PIP_FORCE=""
[ "${REBUILD:-0}" = "1" ] && PIP_FORCE="--force-reinstall"
pip install -q $PIP_FORCE torch --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
pip install -q numpy scipy matplotlib certifi cgal ninja packaging rich
python - <<'PY'
import torch

import CGAL.CGAL_Triangulation_3  # noqa: F401

archs = torch.cuda.get_arch_list()
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| CGAL bindings OK")
print("torch has kernels for:", archs)
if "sm_70" not in archs:
    print(
        "NOTE: no sm_70 kernels -> the V100 partitions (gpu, gpu_test) cannot run Paragram or gDel3D\n"
        "      with this build. Use --partition=gpua100 (A100, sm_80), or reinstall with\n"
        "        TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 REBUILD=1 bash setup_ruche.sh"
    )
PY

echo "== [3/7] standalone tools: parallel CGAL + Local DeWall + gStar4D"
bash build_tools.sh ${REBUILD:+--force}
python - <<'PY'
import subprocess

import numpy as np

np.random.default_rng(0).random((20000, 3)).astype("<f8").tofile("/tmp/_pts.f64")
out = subprocess.run(
    ["bin/cgal_delaunay", "/tmp/_pts.f64", "/tmp/_tets.i32"], capture_output=True, text=True, check=True
).stdout
print("cgal_delaunay:", out.strip())
PY

echo "== [4/7] Paragram (patched: relative clipping pad, cell budget)"
[ -d third_party/paragram ] || git clone -q --recursive https://github.com/zenseact/paragram.git third_party/paragram
python patch_paragram.py third_party/paragram
pip install -q $PIP_FORCE --no-deps third_party/paragram
python -c "import paragram, inspect; print('paragram import OK; bbox_pad:', 'bbox_pad' in inspect.signature(paragram.voronoi_diagram).parameters)"
echo "   (Paragram's CUDA extension is JIT-compiled at first use, inside the SLURM job on the GPU node)"

echo "== [5/7] pyGDel3D (patched: dead-tet flags, phase timers, TORCH_CUDA_ARCH_LIST)"
[ -d third_party/pyGDel3D ] || git clone -q https://github.com/half-potato/pyGDel3D.git third_party/pyGDel3D
python patch_pygdel3d.py third_party/pyGDel3D
pip install -q $PIP_FORCE --no-build-isolation --no-deps third_party/pyGDel3D
python -c "import gdel3d; print('pyGDel3D OK; get_stats:', hasattr(gdel3d.DelOutput, 'get_stats'))"

echo "== [6/7] GeoDel (Geogram ParallelDelaunay3d, CPU-parallel; compiles a 36k-line translation unit)"
pip install -q $PIP_FORCE "geodel @ git+https://github.com/Anttwo/GeoDel@v0.1.0"
python -c "import geodel; print('GeoDel', geodel.__version__, 'OK; max threads:', geodel.max_threads())"

echo "== [7/7] prefetch meshes for the full suite (optional; compute nodes may lack internet)"
python - <<'PY' || echo "   mesh download failed (only needed for the full suite)"
import test_delaunay_surfaces as T
for m in T.DEFAULT_MODELS:
    T.load_obj_vertices(m); print("  cached", m)
PY
echo "== setup done. Submit a job with:  sbatch run_ruche.sbatch"
