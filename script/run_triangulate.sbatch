#!/bin/bash
#SBATCH --job-name=triangulate
#SBATCH --output=results/%x.o%j
#SBATCH --error=results/%x.e%j
#SBATCH --partition=gpua100        # A100 40 GB (the cu128 torch wheels have no Volta kernels)
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
# Run one method on one point cloud on a GPU node and write the tetrahedra.
#
#   METHOD=gdel3d PLY=data/voronoi_jax_068.ply sbatch script/run_triangulate.sbatch
#   METHOD=gstar4d PLY=cloud.ply OUT=results/mine CHECK=true sbatch script/run_triangulate.sbatch
#   METHOD=paragram PLY=cloud.ply JITTER=1e-6 sbatch script/run_triangulate.sbatch
#
# The four GPU methods (paragram, gdel3d, gstar4d, dewall) need this; geodel, cgal_parallel and
# cgal_sequential also run directly on the login node:
#   python main.py experiment=triangulate method=geodel ply=cloud.ply
echo "run_triangulate: bash started on $(hostname -s) at $(date '+%F %T'), job ${SLURM_JOB_ID:-?}"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit directory not set: submit with sbatch, from the repository root}"
: "${METHOD:?set METHOD=<paragram|gdel3d|gstar4d|dewall|geodel|cgal_parallel|cgal_sequential>}"
: "${PLY:?set PLY=<path to the point cloud>}"
[ -f "$PLY" ] || { echo "PLY not found: $PLY (paths are relative to the repository root)" >&2; exit 2; }

# shellcheck disable=SC1091
. script/env.sh                     # conda env, compilers, CUDA paths, PYTHONPATH=src
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# One JIT cache per architecture; read the capability from nvidia-smi (no CUDA context needed).
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
[ -n "$ARCH" ] || ARCH=8.0
export TORCH_CUDA_ARCH_LIST=$ARCH
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${WORKDIR:-$HOME}/.cache/torch_extensions_sm${ARCH//./}}
mkdir -p "$TORCH_EXTENSIONS_DIR"
echo "node: $(hostname -s)   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=name,compute_cap,compute_mode,memory.total,memory.used --format=csv,noheader

# Some nodes accept the job and then refuse every CUDA context: wait, and if it stays broken,
# record the node and resubmit excluding it (see cuda_wait.sh).
echo "== CUDA readiness"
bash script/cuda_wait.sh script/run_triangulate.sbatch || exit 1

# The compiled tools must already exist: compute nodes cannot clone, so build on the login node
# (bash script/build_tools.sh) if this stops here.
for tool in bin/cgal_delaunay bin/dewall bin/gstar4d; do
  [ -x "$tool" ] || echo "   note: $tool missing -- run 'bash script/build_tools.sh' on the login node if you need it"
done

OUT=${OUT:-results/triangulate}
echo "== triangulate: method=$METHOD ply=$PLY out=$OUT jitter=${JITTER:-0} check=${CHECK:-false}"
python main.py experiment=triangulate method="$METHOD" ply="$PLY" out="$OUT" \
  jitter="${JITTER:-0}" check="${CHECK:-false}" threads="$SLURM_CPUS_PER_TASK"
echo "== done"
