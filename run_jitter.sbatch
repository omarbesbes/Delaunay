#!/bin/bash
#SBATCH --job-name=delaunay-jitter
#SBATCH --output=results/%x.o%j
#SBATCH --error=results/%x.e%j
#SBATCH --partition=gpua100        # A100 40 GB (the cu128 torch wheels have no Volta kernels)
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # CGAL parallel, GeoDel and the CPU repair use these cores
#SBATCH --mem=64G
#SBATCH --time=08:00:00
# Sweep the size of the jitter on the two point clouds only, and record for every value:
#   how much it deforms the cloud, what each method costs, and how often each method's in-sphere
#   predicate has to fall back to exact arithmetic.
# Usage:  sbatch run_jitter.sbatch
#         JITTERS="0 1e-8 1e-6 1e-4" sbatch run_jitter.sbatch
#         CLOUD=data/voronoi_jax_068.ply sbatch run_jitter.sbatch       (one cloud, half the time)
#         METHODS="gdel3d cgal_sequential" sbatch run_jitter.sbatch
#         REPEATS=1 sbatch run_jitter.sbatch                            (quick pass)
#
# The predicate counters need the patched builds:  bash setup_ruche.sh  (pyGDel3D, for gDel3D's
# doInSphereFast/doInSphereSoS counts) and  bash build_tools.sh  (bin/cgal_delaunay_profile).
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

OUT=results/jitter-$SLURM_JOB_ID
mkdir -p "$OUT"

# Some nodes accept the job and then refuse every CUDA context: wait, and if it stays broken,
# record the node and resubmit excluding it (see cuda_wait.sh).
echo "== CUDA readiness"
bash cuda_wait.sh run_jitter.sbatch || exit 1

JITTERS=${JITTERS:-"0 1e-9 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3"}
CLOUD=${CLOUD:-"data/voronoi_iarpa_001.ply data/voronoi_jax_068.ply"}
METHODS=${METHODS:-"paragram gdel3d gstar4d dewall geodel cgal_parallel cgal_sequential"}
REPEATS=${REPEATS:-5}
TOOL_TIMEOUT=${TOOL_TIMEOUT:-120}
GSTAR4D_GRID=${GSTAR4D_GRID:-512}

# bin/cgal_delaunay_profile is a -DCGAL_PROFILE build of the same tool; without it CGAL reports
# no predicate counters.  Cheap to build, so do it here if it is missing.
[ -x bin/cgal_delaunay_profile ] || bash build_tools.sh

echo "== predicate counters available:"
python - <<'PY'
import os

import gdel3d
import voronoi_to_delaunay as v

# gDel3D's counters live in its Statistics struct, so a build patched before patch_pygdel3d.py
# grew them has get_stats() but no predCheckNum key; that only shows up at run time.
print("   gDel3D  :", "get_stats" if hasattr(gdel3d.DelOutput, "get_stats") else "NO (re-run setup_ruche.sh)")
print("   CGAL    :", "yes" if os.path.exists("bin/cgal_delaunay_profile") else "NO (run build_tools.sh)")
print("   Paragram:", "yes" if hasattr(v, "last_insphere_stats") else "NO")
PY

echo "== warm-up: JIT-compile Paragram's CUDA extension"
python -c "import torch, paragram; paragram.voronoi_diagram(torch.rand(10000, 3, device='cuda'), bbox_pad=10.0); print('paragram ready')"

echo "== jitter sweep: jitters [$JITTERS], methods [$METHODS], clouds [$CLOUD]"
# One process per method, so a library that crashes the interpreter only ends its own sweep.  All
# of them share one JSON: with --resume the first process computes the deformation metrics and the
# CGAL predicate counts, and the later ones read them back instead of recomputing them.
ATTEMPTS=${ATTEMPTS:-4}
ATTEMPT_TIMEOUT=${ATTEMPT_TIMEOUT:-3600}
JSON="$OUT/jitter.json"
for m in $METHODS; do
  for attempt in $(seq 1 "$ATTEMPTS"); do
    echo "-- $m (attempt $attempt/$ATTEMPTS)"
    rc=0
    timeout "$ATTEMPT_TIMEOUT" python jitter_study.py --ply $CLOUD --jitters $JITTERS \
      --methods "$m" --repeats "$REPEATS" --threads "$SLURM_CPUS_PER_TASK" --resume \
      --tool-timeout "$TOOL_TIMEOUT" --gstar4d-grid "$GSTAR4D_GRID" \
      --cgal-bin bin/cgal_delaunay --dewall-bin bin/dewall --gstar4d-bin bin/gstar4d \
      --json "$JSON" 2>&1 | tee -a "$OUT/jitter-$m.log" || rc=$?
    [ "$rc" = 0 ] && break
    # 134 = SIGABRT, 139 = SIGSEGV: the library killed the interpreter, not an exception
    echo "   $m: the process died (exit $rc); resuming past that measurement"
  done
done

echo "== tables and plot"
python jitter_study.py --plot-only "$JSON" \
  --csv "$OUT/jitter.csv" --markdown "$OUT/jitter.md" --plot "$OUT/jitter.png" \
  2>&1 | tee "$OUT/jitter-report.log"
echo "== done: $OUT/jitter.md, jitter.png, jitter.csv, jitter.json"
