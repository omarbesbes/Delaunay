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
echo "node: $(hostname -s)   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=name,compute_cap,compute_mode,memory.total,memory.used --format=csv,noheader

OUT=results/scaling-$SLURM_JOB_ID
mkdir -p "$OUT"

# Some nodes accept the job and then refuse every CUDA context: wait, and if it stays broken,
# record the node and resubmit excluding it (see cuda_wait.sh).
echo "== CUDA readiness"
bash cuda_wait.sh run_scaling.sbatch || exit 1
SIZES=${SIZES:-"2000:200000:10000 200000 300000:1000000:100000"}
JITTERS=${JITTERS:-"0 1e-6"}
METHODS=${METHODS:-"paragram gdel3d gstar4d dewall geodel cgal_parallel cgal_sequential"}
REPEATS=${REPEATS:-3}
CLOUD=${CLOUD:-"data/voronoi_iarpa_001.ply data/voronoi_jax_068.ply"}

echo "== warm-up: JIT-compile Paragram's CUDA extension"
python -c "import torch, paragram; paragram.voronoi_diagram(torch.rand(10000, 3, device='cuda'), bbox_pad=10.0); print('paragram ready')"

echo "== scaling sweep: sizes [$SIZES], jitters [$JITTERS], methods [$METHODS]"
# One process per method, so a library that crashes the interpreter (gDel3D has been seen to
# segfault on some inputs) only ends its own sweep.  Each process is restarted up to ATTEMPTS
# times with --resume, which keeps the measurements already written and marks the input that
# killed the previous process as failed instead of running into it again.
ATTEMPTS=${ATTEMPTS:-4}
ATTEMPT_TIMEOUT=${ATTEMPT_TIMEOUT:-5400}     # per attempt, seconds: a hang cannot eat the job
MEASURE_TIMEOUT=${MEASURE_TIMEOUT:-300}      # per isolated measurement (gDel3D hung for 13 min)
for m in $METHODS; do
  for attempt in $(seq 1 "$ATTEMPTS"); do
    echo "-- $m (attempt $attempt/$ATTEMPTS)"
    rc=0
    # `timeout` is a backstop: --measure-timeout already bounds the isolated methods, this
    # catches a hang anywhere else (124 = the attempt was killed by it).
    timeout "$ATTEMPT_TIMEOUT" python scaling_study.py --ply $CLOUD --sizes $SIZES \
      --jitter $JITTERS --methods "$m" --repeats "$REPEATS" --threads "$SLURM_CPUS_PER_TASK" \
      --resume --measure-timeout "$MEASURE_TIMEOUT" \
      --cgal-bin bin/cgal_delaunay --dewall-bin bin/dewall --gstar4d-bin bin/gstar4d \
      --json "$OUT/scaling-$m.json" 2>&1 | tee -a "$OUT/scaling-$m.log" || rc=$?
    [ "$rc" = 0 ] && break
    # 134 = SIGABRT, 139 = SIGSEGV: the library killed the interpreter, not an exception
    echo "   $m: the process died (exit $rc); resuming past that measurement"
  done
done

echo "== merging and plotting"
python scaling_study.py --plot-only "$OUT"/scaling-*.json \
  --csv "$OUT/scaling.csv" --plot "$OUT/scaling.png" 2>&1 | tee "$OUT/scaling.log"
echo "== done: $OUT/scaling.png (and scaling.csv, scaling-<method>.json)"
