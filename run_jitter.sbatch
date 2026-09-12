#!/bin/bash
#SBATCH --job-name=delaunay-jitter
#SBATCH --output=results/%x.o%j
#SBATCH --error=results/%x.e%j
#SBATCH --partition=gpua100        # A100 40 GB (the cu128 torch wheels have no Volta kernels)
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # CGAL parallel, GeoDel and the CPU repair use these cores
#SBATCH --mem=128G                 # gDel3D's CPU star-splaying repair on the *unjittered*
                                   # clouds was OOM-killed at 64G (job 1886808)
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
# The predicate counters need the patched builds.  The job checks for them and repairs what it can:
# it rebuilds bin/cgal_delaunay_profile if missing, and re-patches and reinstalls pyGDel3D if its
# gDel3D counters are absent (AUTOFIX=0 to only report).
# First executable line, before anything that can fail: if this does not reach the .o file, the
# script never ran and the cause is at the SLURM level (prolog, GPU allocation, node failure),
# not in here.  `sacct -j <jobid>` has the state and the reason in that case.
echo "run_jitter: bash started on $(hostname -s) at $(date '+%F %T'), job ${SLURM_JOB_ID:-?}"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit directory not set: submit with sbatch, from the repository root}"
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

# gDel3D's counters live in its Statistics struct, so a pyGDel3D built before patch_pygdel3d.py
# grew them still has get_stats() -- the missing keys only show up at run time.  Check by running
# it, and rebuild rather than discover after an eight-hour sweep that half the study is empty.
echo "== predicate counters"
probe() {
  python - 2>/dev/null <<'PY' | grep -c "GDEL3D_COUNTERS yes" || true
import numpy as np
import test_delaunay_surfaces as T
_, _, info = T.run_gdel3d(np.random.default_rng(0).random((1000, 3)))
print("GDEL3D_COUNTERS", "yes" if info.get("predicate_total") else "no")
PY
}
if [ "$(probe)" = "1" ]; then
  echo "   gDel3D  : yes"
elif [ "${AUTOFIX:-1}" = "1" ]; then
  echo "   gDel3D  : missing -- re-patching and rebuilding pyGDel3D (a few minutes)"
  python patch_pygdel3d.py third_party/pyGDel3D
  pip install -q --force-reinstall --no-build-isolation --no-deps third_party/pyGDel3D
  echo "   gDel3D  : $([ "$(probe)" = "1" ] && echo yes || echo "STILL MISSING -- continuing without them")"
else
  echo "   gDel3D  : missing (AUTOFIX=0; run 'python patch_pygdel3d.py third_party/pyGDel3D' and reinstall it)"
fi
python - <<'PY'
import os

import voronoi_to_delaunay as v

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
