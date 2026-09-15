#!/bin/bash
# One-time environment setup, on a LOGIN node (it needs internet; compute nodes usually have none).
# Creates a self-contained conda environment with Python 3.12, CUDA 12.8 (nvcc), GCC 13, torch
# cu128, CGAL headers + TBB, then builds: the parallel CGAL tool, Paragram (patched), pyGDel3D
# (patched), Local DeWall (patched), gStar4D (patched) and GeoDel.
#
#   bash script/first_install.sh                       # ~15-30 min
#   DELAUNAY_ENV=~/envs/delaunay bash script/first_install.sh    # environment elsewhere
#   GPU_ARCHS=8.0 bash script/first_install.sh         # build for one GPU generation only
#
# Portable between Ruche (Mesocentre) and CentraleSupelec's DGX: script/env.sh finds conda either
# way and picks a default location for the environment.  See script/readme.md for the partitions.
set -euo pipefail
# the script lives in script/ but every path below is relative to the repository root
cd "$(dirname "$0")/.."
ROOT=$PWD
mkdir -p bin third_party results    # git-ignored, absent in a fresh clone
# CUDA architectures to build for.  Deliberately NOT read from TORCH_CUDA_ARCH_LIST: torch sets it
# to its own full default (13 architectures on CUDA 12.8), and each one multiplies the compile time
# of pyGDel3D's extension and of the two CUDA tools.  GPU_ARCHS=8.0 is enough for an A100-only
# cluster, "7.0;8.0" covers V100 and A100.
ARCHS=${GPU_ARCHS:-"7.0;8.0"}

echo "== [1/7] conda environment"
# The DGX ships no conda (it documents plain venv), and pip cannot provide nvcc, GCC 13, the CGAL
# headers or TBB.  INSTALL_CONDA=1 bootstraps miniconda into the user's home, once.
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniforge3}
if [ "${INSTALL_CONDA:-0}" = "1" ] && ! command -v conda >/dev/null 2>&1; then
  # Do not install a second one next to a conda that env.sh would have found anyway.
  for c in "$CONDA_ROOT" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -x "$c/bin/conda" ] && { CONDA_ROOT=$c; break; }
  done
  if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
    # Miniforge rather than Miniconda: it defaults to conda-forge, which is the only channel this
    # project uses, and carries none of the Anaconda terms-of-service prompts or the commercial-use
    # restrictions that apply to an institution.
    echo "-- installing miniforge into $CONDA_ROOT (a few hundred MB)"
    curl -fsSLo /tmp/miniforge-$$.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/miniforge-$$.sh -b -p "$CONDA_ROOT"
    rm -f /tmp/miniforge-$$.sh
  fi
fi
export CONDA_ROOT
CREATE=1 . script/env.sh
export TORCH_CUDA_ARCH_LIST=$ARCHS
echo "environment: $CONDA_PREFIX"
echo "cuda_runtime_api.h: $(ls "$CONDA_PREFIX/targets/x86_64-linux/include/cuda_runtime_api.h" 2>/dev/null || ls "$CONDA_PREFIX/include/cuda_runtime_api.h" 2>/dev/null || echo NOT FOUND)"
echo "python: $(python -V) | nvcc: $(nvcc --version | tail -1) | host compiler: $($CXX --version | head -1)"

echo "== [2/7] python packages (torch cu128, numpy, scipy, matplotlib, cgal bindings)"
# cu128 wheels have no Volta (sm_70) kernels: for the V100 partitions (gpu, gpu_test) install
# the cu126 build instead with  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash script/first_install.sh
PIP_FORCE=""
[ "${REBUILD:-0}" = "1" ] && PIP_FORCE="--force-reinstall"
pip install -q $PIP_FORCE torch --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
pip install -q numpy scipy matplotlib certifi cgal ninja packaging rich
python - <<'PY'
import torch

import CGAL.CGAL_Triangulation_3  # noqa: F401

archs = torch.cuda.get_arch_list()
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| CGAL bindings OK")
if not archs:
    # Without a driver (a login node) torch cannot list its kernels; the GPU job prints them.
    print("torch kernel list: not available here (no NVIDIA driver on this node); checked in the job")
else:
    print("torch has kernels for:", archs)
if archs and "sm_70" not in archs:
    print(
        "NOTE: no sm_70 kernels -> a V100 partition cannot run Paragram or gDel3D with this\n"
        "      build. Use an A100 partition (sm_80), or reinstall with\n"
        "        TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 REBUILD=1 bash script/first_install.sh"
    )
PY

N_ARCHS=$(echo "$ARCHS" | tr ';,' '  ' | wc -w | tr -d ' ')
if [ "$N_ARCHS" -gt 3 ]; then
  echo "NOTE: building for $N_ARCHS CUDA architectures ($ARCHS)."
  echo "      Each one is compiled separately, so this multiplies the build time of pyGDel3D and"
  echo "      of the CUDA tools.  Pass the one your cluster has -- GPU_ARCHS=8.0 for A100 only."
fi

echo "== [3/7] standalone tools: parallel CGAL + Local DeWall + gStar4D"
GPU_ARCHS="$ARCHS" bash script/build_tools.sh ${REBUILD:+--force}
# Smoke-test the tool just built.  Two guards, both learned on a DGX login node:
#   CGAL_THREADS caps TBB, which otherwise sizes its pool from the whole machine (hundreds of cores
#   on a DGX) while the cgroup grants a handful, and CGAL's retry-on-conflict loop then thrashes;
#   a timeout so that a stuck tool reports instead of hanging the install.  Inside a SLURM job
#   neither applies: the affinity mask already tells TBB the right number.
CGAL_THREADS=${CGAL_THREADS:-4} timeout 120 python - <<'PY' \
  || echo "   WARNING: bin/cgal_delaunay did not finish in 120 s -- it is built, but test it by hand"
import subprocess

import numpy as np

np.random.default_rng(0).random((20000, 3)).astype("<f8").tofile("/tmp/_pts.f64")
out = subprocess.run(
    ["bin/cgal_delaunay", "/tmp/_pts.f64", "/tmp/_tets.i32"], capture_output=True, text=True, check=True
).stdout
print("cgal_delaunay:", out.strip())
PY

echo "== [4/7] Paragram (patched: relative clipping pad, cell budget)"
if [ ! -d third_party/paragram ]; then
  git clone -q --recursive https://github.com/zenseact/paragram.git third_party/paragram \
    || { echo "cannot reach github.com -- run this on the LOGIN node, not a compute node" >&2; exit 1; }
fi
python script/patch_paragram.py third_party/paragram
pip install -q $PIP_FORCE --no-deps third_party/paragram
python -c "import paragram, inspect; print('paragram import OK; bbox_pad:', 'bbox_pad' in inspect.signature(paragram.voronoi_diagram).parameters)"
echo "   (Paragram's CUDA extension is JIT-compiled at first use, inside the SLURM job on the GPU node)"

echo "== [5/7] pyGDel3D (patched: dead-tet flags, phase timers, predicate counters, TORCH_CUDA_ARCH_LIST)"
if [ ! -d third_party/pyGDel3D ]; then
  git clone -q https://github.com/half-potato/pyGDel3D.git third_party/pyGDel3D \
    || { echo "cannot reach github.com -- run this on the LOGIN node, not a compute node" >&2; exit 1; }
fi
python script/patch_pygdel3d.py third_party/pyGDel3D
pip install -q $PIP_FORCE --no-build-isolation --no-deps third_party/pyGDel3D
python -c "import gdel3d; print('pyGDel3D OK; get_stats:', hasattr(gdel3d.DelOutput, 'get_stats'))"

echo "== [6/7] GeoDel (Geogram ParallelDelaunay3d, CPU-parallel; compiles a 36k-line translation unit)"
pip install -q $PIP_FORCE "geodel @ git+https://github.com/Anttwo/GeoDel@v0.1.0"
python -c "import geodel; print('GeoDel', geodel.__version__, 'OK; max threads:', geodel.max_threads())"

echo "== [7/7] prefetch meshes for the full suite (optional; compute nodes may lack internet)"
python - <<'PY' || echo "   mesh download failed (only needed for the full suite)"
import benchmark as T
for m in T.DEFAULT_MODELS:
    T.load_obj_vertices(m); print("  cached", m)
PY
echo "== setup done. Submit a job with:  sbatch script/run_benchmark.sbatch"
