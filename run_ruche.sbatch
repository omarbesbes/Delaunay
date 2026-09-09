#!/bin/bash
#SBATCH --job-name=delaunay-bench
#SBATCH --output=results/%x.o%j
#SBATCH --error=results/%x.e%j
#SBATCH --partition=gpua100        # A100 40 GB; alternatives: gpu (V100 32 GB, 24 h), gpu_test (V100, 1 h)
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # CGAL parallel (TBB) and the CPU repair use these cores
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# Usage:  sbatch run_ruche.sbatch                 (your two point clouds, 10 repeats)
#         FULL=1 sbatch run_ruche.sbatch          (also analytic surfaces + meshes, 20k samples each)
#         REPEATS=5 sbatch run_ruche.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set +u; module purge; module load anaconda3/2023.09-0/none-none; set -u
ENV=${DELAUNAY_ENV:-$WORKDIR/envs/delaunay}
# conda activation scripts (cuda-nvcc) read variables that may be unset: relax "set -u" around them
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
set +u
source activate "$ENV"
set -u
export CUDA_HOME=$CONDA_PREFIX
export CC=${CC:-x86_64-conda-linux-gnu-gcc}
export CXX=${CXX:-x86_64-conda-linux-gnu-g++}
export PARAGRAM_MAX_PLANES=128 PARAGRAM_MAX_VERTS=128     # Paragram build variant (JIT, first import)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Paragram's CUDA extension is JIT-compiled in this job: pin the architecture to this node's GPU
# (torch's auto-detection on cu128 builds can fail with "Unknown CUDA arch") and keep one build
# cache per architecture so V100 and A100 jobs do not reuse each other's binary.
ARCH=$(python -c "import torch; c = torch.cuda.get_device_capability(); print(f'{c[0]}.{c[1]}')")
export TORCH_CUDA_ARCH_LIST=$ARCH
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$WORKDIR/.cache/torch_extensions_sm${ARCH//./}}
mkdir -p "$TORCH_EXTENSIONS_DIR"
echo "GPU compute capability: $ARCH | torch extension cache: $TORCH_EXTENSIONS_DIR"

REPEATS=${REPEATS:-10}
OUT=results/$SLURM_JOB_ID
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$OUT/gpu.txt"

DATASETS="--no-analytic --models"
[ "${FULL:-0}" = "1" ] && DATASETS="--n 20000"
COMMON="--unit-cube --repeats $REPEATS --slow-repeats 2 --slow-threshold 100 --verbose \
  --cgal-bin bin/cgal_delaunay --dewall-bin bin/dewall --ply data/*.ply $DATASETS"

echo "== warm-up: JIT-compile Paragram's CUDA extension"
python -c "import torch, paragram; d = paragram.voronoi_diagram(torch.rand(10000, 3, device='cuda'), bbox_pad=10.0); print('paragram ready, status ok:', int((d.status == 0).sum()), '/ 10000')"

echo "== corrected Paragram (pad 10x, CPU repair) + gDel3D + Local DeWall + CGAL parallel/sequential"
python test_delaunay_surfaces.py $COMMON --paragram-bbox-pad 10 --repair on --json "$OUT/results.json" 2>&1 | tee "$OUT/run.log"

echo "== baseline Paragram (upstream box, no repair), 3 repeats"
python test_delaunay_surfaces.py $COMMON --repeats 3 --paragram-bbox-pad -1 --repair off --gdel3d off --json "$OUT/results_baseline.json" 2>&1 | tee "$OUT/run_baseline.log"

echo "== jittered (1e-6) run, 3 repeats"
python test_delaunay_surfaces.py $COMMON --repeats 3 --jitter 1e-6 --paragram-bbox-pad 10 --repair on --json "$OUT/results_jitter.json" 2>&1 | tee "$OUT/run_jitter.log"

python make_report.py "$OUT/results.json" --baseline "$OUT/results_baseline.json" --jitter "$OUT/results_jitter.json" -o "$OUT/report.md"
echo "== done: $OUT/report.md"
