#!/bin/bash
#SBATCH --job-name=delaunay-scaling
#SBATCH --output=results/%x.o%j
#SBATCH --error=results/%x.e%j
#SBATCH --partition=gpua100        # A100 40 GB (the cu128 torch wheels have no Volta kernels)
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # CGAL parallel, GeoDel and the CPU repair use these cores
#SBATCH --mem=128G                 # 1M points: the point sets and the tet arrays are large
#SBATCH --time=12:00:00
# Time vs number of points, for the two point clouds only, with and without jitter.
# Usage:  sbatch run_scaling.sbatch
#         SIZES="2000:200000:10000 200000" sbatch run_scaling.sbatch      (skip the tiled sizes)
#         CLOUD=data/voronoi_jax_068.ply sbatch run_scaling.sbatch        (one cloud, half the time)
#         REPEATS=1 sbatch run_scaling.sbatch                            (quick pass)
#         METHODS="geodel cgal_parallel" sbatch run_scaling.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set +u; module purge; module load anaconda3/2023.09-0/none-none; set -u
ENV=${DELAUNAY_ENV:-$WORKDIR/envs/delaunay}
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
set +u; source activate "$ENV"; set -u
export CUDA_HOME=$CONDA_PREFIX
export CC=${CC:-x86_64-conda-linux-gnu-gcc} CXX=${CXX:-x86_64-conda-linux-gnu-g++}
export PARAGRAM_MAX_PLANES=128 PARAGRAM_MAX_VERTS=128
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
CUDA_TARGET=$CONDA_PREFIX/targets/x86_64-linux
[ -d "$CUDA_TARGET/include" ] && export CPATH="$CUDA_TARGET/include${CPATH:+:$CPATH}"
[ -d "$CUDA_TARGET/lib" ] && export LIBRARY_PATH="$CUDA_TARGET/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
[ -d "$CUDA_TARGET/lib/stubs" ] && export LIBRARY_PATH="${LIBRARY_PATH:+$LIBRARY_PATH:}$CUDA_TARGET/lib/stubs"

# One JIT cache per architecture; read the capability from nvidia-smi (no CUDA context needed).
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
[ -n "$ARCH" ] || ARCH=8.0
export TORCH_CUDA_ARCH_LIST=$ARCH
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$WORKDIR/.cache/torch_extensions_sm${ARCH//./}}
mkdir -p "$TORCH_EXTENSIONS_DIR"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

OUT=results/scaling-$SLURM_JOB_ID
mkdir -p "$OUT"
SIZES=${SIZES:-"2000:200000:10000 200000 300000:1000000:100000"}
JITTERS=${JITTERS:-"0 1e-6"}
METHODS=${METHODS:-"paragram gdel3d gstar4d dewall geodel cgal_parallel cgal_sequential"}
REPEATS=${REPEATS:-3}
CLOUD=${CLOUD:-"data/voronoi_iarpa_001.ply data/voronoi_jax_068.ply"}

echo "== warm-up: JIT-compile Paragram's CUDA extension"
python -c "import torch, paragram; paragram.voronoi_diagram(torch.rand(10000, 3, device='cuda'), bbox_pad=10.0); print('paragram ready')"

echo "== scaling sweep: sizes [$SIZES], jitters [$JITTERS], methods [$METHODS]"
python scaling_study.py --ply $CLOUD --sizes $SIZES --jitter $JITTERS --methods $METHODS \
  --repeats "$REPEATS" --threads "$SLURM_CPUS_PER_TASK" \
  --cgal-bin bin/cgal_delaunay --dewall-bin bin/dewall --gstar4d-bin bin/gstar4d \
  --json "$OUT/scaling.json" --csv "$OUT/scaling.csv" --plot "$OUT/scaling.png" \
  2>&1 | tee "$OUT/scaling.log"
echo "== done: $OUT/scaling.png (and scaling.csv / scaling.json)"
