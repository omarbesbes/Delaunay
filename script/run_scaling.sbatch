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
# Usage:  sbatch script/run_scaling.sbatch
#         SIZES="2000:200000:10000 200000" sbatch script/run_scaling.sbatch      (skip the tiled sizes)
#         CLOUD=data/voronoi_jax_068.ply sbatch script/run_scaling.sbatch        (one cloud, half the time)
#         REPEATS=1 sbatch script/run_scaling.sbatch                            (quick pass)
#         METHODS="geodel cgal_parallel" sbatch script/run_scaling.sbatch
#         UPSAMPLE=densify sbatch script/run_scaling.sbatch   (sizes above 100k by densifying the same
#                                                       volume instead of tiling copies of it)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
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

OUT=results/scaling-$SLURM_JOB_ID
mkdir -p "$OUT"

# Some nodes accept the job and then refuse every CUDA context: wait, and if it stays broken,
# record the node and resubmit excluding it (see cuda_wait.sh).
echo "== CUDA readiness"
bash script/cuda_wait.sh script/run_scaling.sbatch || exit 1
SIZES=${SIZES:-"2000:200000:10000 200000 300000:1000000:100000"}
JITTERS=${JITTERS:-"0 1e-6"}
METHODS=${METHODS:-"paragram gdel3d gstar4d dewall geodel cgal_parallel cgal_sequential"}
REPEATS=${REPEATS:-3}
CLOUD=${CLOUD:-"data/voronoi_iarpa_001.ply data/voronoi_jax_068.ply"}
# How to reach sizes above the cloud's own ~100k points: tile (more area, same resolution) or
# densify (same area, more resolution -- new points interpolated inside local tetrahedra).
UPSAMPLE=${UPSAMPLE:-tile}

echo "== warm-up: JIT-compile Paragram's CUDA extension"
python -c "import torch, paragram; paragram.voronoi_diagram(torch.rand(10000, 3, device='cuda'), bbox_pad=10.0); print('paragram ready')"

echo "== scaling sweep: sizes [$SIZES], jitters [$JITTERS], methods [$METHODS], upsample $UPSAMPLE"
# One process per method, so a library that crashes the interpreter (gDel3D has been seen to
# segfault on some inputs) only ends its own sweep.  Each process is restarted up to ATTEMPTS
# times with --resume, which keeps the measurements already written and marks the input that
# killed the previous process as failed instead of running into it again.
ATTEMPTS=${ATTEMPTS:-4}
ATTEMPT_TIMEOUT=${ATTEMPT_TIMEOUT:-5400}     # per attempt, seconds: a hang cannot eat the job
MEASURE_TIMEOUT=${MEASURE_TIMEOUT:-300}      # per isolated measurement (gDel3D hung for 13 min)

# Overrides are `key=value`, which replaces the value from the experiment config; raw `--flags`
# would append a second copy and leave whatever is not overridden pointing at the config's paths.
# Lists are comma-separated in that syntax.
EXPERIMENT=$([ "$UPSAMPLE" = "densify" ] && echo scaling_densify || echo scaling_tile)
PLY_LIST="[$(echo $CLOUD | tr ' ' ',')]"
SIZE_LIST="[$(echo $SIZES | tr ' ' ',')]"
JITTER_LIST="[$(echo $JITTERS | tr ' ' ',')]"
for m in $METHODS; do
  for attempt in $(seq 1 "$ATTEMPTS"); do
    echo "-- $m (attempt $attempt/$ATTEMPTS)"
    rc=0
    # `timeout` is a backstop: --measure-timeout already bounds the isolated methods, this
    # catches a hang anywhere else (124 = the attempt was killed by it).
    timeout "$ATTEMPT_TIMEOUT" python main.py experiment="$EXPERIMENT" \
      ply="$PLY_LIST" sizes="$SIZE_LIST" jitter="$JITTER_LIST" methods="$m" \
      repeats="$REPEATS" threads="$SLURM_CPUS_PER_TASK" resume=true \
      measure_timeout="$MEASURE_TIMEOUT" upsample="$UPSAMPLE" \
      json="$OUT/scaling-$m.json" csv="$OUT/scaling.csv" plot="$OUT/scaling.png" \
      2>&1 | tee -a "$OUT/scaling-$m.log" || rc=$?
    [ "$rc" = 0 ] && break
    # 134 = SIGABRT, 139 = SIGSEGV: the library killed the interpreter, not an exception
    echo "   $m: the process died (exit $rc); resuming past that measurement"
  done
done

echo "== merging and plotting"
# Rendering runs no experiment, so it calls the study directly rather than naming one.
python src/scaling_study.py --plot-only "$OUT"/scaling-*.json \
  --csv "$OUT/scaling.csv" --plot "$OUT/scaling.png" 2>&1 | tee "$OUT/scaling.log"
echo "== done: $OUT/scaling.png (and scaling.csv, scaling-<method>.json)"
