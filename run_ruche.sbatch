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
#         TOOL_TIMEOUT=300 sbatch run_ruche.sbatch   (kill a hanging external tool sooner)
# Methods: Paragram + conversion (GPU), gDel3D (GPU), gStar4D (GPU), Local DeWall (GPU),
#          GeoDel / Geogram (CPU-parallel), CGAL parallel (TBB) and CGAL sequential.
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

# conda-forge keeps the CUDA headers and libraries under $CONDA_PREFIX/targets/x86_64-linux, which
# torch's JIT build does not add when compiling *host* code (ext.cpp): make them visible here.
CUDA_TARGET=$CONDA_PREFIX/targets/x86_64-linux
[ -d "$CUDA_TARGET/include" ] && export CPATH="$CUDA_TARGET/include${CPATH:+:$CPATH}"
[ -d "$CUDA_TARGET/lib" ] && export LIBRARY_PATH="$CUDA_TARGET/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
# the driver stub is a last resort for -lcuda; the real libcuda.so of the GPU node wins if present
[ -d "$CUDA_TARGET/lib/stubs" ] && export LIBRARY_PATH="${LIBRARY_PATH:+$LIBRARY_PATH:}$CUDA_TARGET/lib/stubs"
echo "cuda_runtime_api.h: $(ls "$CUDA_TARGET/include/cuda_runtime_api.h" 2>/dev/null || ls "$CONDA_PREFIX/include/cuda_runtime_api.h" 2>/dev/null || echo NOT FOUND)"
REPEATS=${REPEATS:-10}
# A run that takes longer than this is not of interest: the external tool (gStar4D, Local DeWall)
# is killed and reported as failed for that dataset.  gStar4D's star-consistency loop has no
# iteration cap and does not converge on every input; Local DeWall needs minutes on
# near-co-spherical input.  The warm-up run hits the limit first, so a method that cannot do a
# dataset costs one timeout, not one per repetition.
TOOL_TIMEOUT=${TOOL_TIMEOUT:-120}
OUT=results/$SLURM_JOB_ID
mkdir -p "$OUT"

# ---- GPU diagnostics -----------------------------------------------------------------------
{
  echo "node: $(hostname)   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,compute_cap,compute_mode,memory.total,memory.used,driver_version --format=csv
  echo "processes holding a GPU:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
} 2>&1 | tee "$OUT/gpu.txt"

# Compute capability read from nvidia-smi, which needs no CUDA context: torch's own auto-detection
# can fail on cu128 builds ("Unknown CUDA arch").  One JIT cache per architecture, so that V100 and
# A100 jobs never reuse each other's Paragram binary.
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
[ -n "$ARCH" ] || { echo "could not read compute_cap from nvidia-smi; falling back to 8.0"; ARCH=8.0; }
export TORCH_CUDA_ARCH_LIST=$ARCH
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$WORKDIR/.cache/torch_extensions_sm${ARCH//./}}
mkdir -p "$TORCH_EXTENSIONS_DIR"
echo "GPU compute capability: $ARCH | torch extension cache: $TORCH_EXTENSIONS_DIR"

# Probe CUDA: report the real error, retry only if the device is merely busy, and stop immediately
# if this torch build has no kernels for the GPU (the cu128 wheels contain no Volta/sm_70).
cuda_probe() {
  python - <<'PY'
import sys

import torch

try:
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
except Exception as exc:  # noqa: BLE001
    msg = str(exc)
    print(f"CUDA probe failed: {msg}", file=sys.stderr)
    try:
        cap = torch.cuda.get_device_capability(0)
    except Exception:  # noqa: BLE001
        cap = None
    print(
        f"torch {torch.__version__} (built for CUDA {torch.version.cuda}) has kernels for "
        f"{torch.cuda.get_arch_list()}; this GPU is compute capability {cap}",
        file=sys.stderr,
    )
    sys.exit(2 if ("no kernel image" in msg or "not supported" in msg) else 1)
print(
    f"CUDA OK: {torch.cuda.get_device_name(0)} sm_{''.join(map(str, torch.cuda.get_device_capability(0)))} "
    f"| torch {torch.__version__} archs: {torch.cuda.get_arch_list()}"
)
PY
}

echo "== CUDA readiness"
for attempt in $(seq 1 6); do
  set +e
  probe=$(cuda_probe 2>&1)
  rc=$?
  set -e
  echo "$probe" | sed 's/^/   /'
  [ "$rc" = 0 ] && break
  if [ "$attempt" = 1 ]; then
    echo "== GPU diagnostics (first probe failure)"
    bash diagnose_gpu.sh 2>&1 | tee "$OUT/diagnose_gpu.txt" || true
  fi
  if [ "$rc" = 2 ]; then
    echo "   FAILED: this torch build has no kernels for this GPU architecture."
    echo "   Either submit to the A100 partition (#SBATCH --partition=gpua100, sm_80), or reinstall torch"
    echo "   with Volta support for the V100 partitions:"
    echo "     source activate \$DELAUNAY_ENV && pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126"
    echo "   (Paragram and gDel3D are torch extensions and need a torch that runs on this GPU;"
    echo "    Local DeWall and CGAL are standalone and unaffected.)"
    exit 1
  fi
  if [ "$attempt" = 6 ]; then
    echo "   FAILED: no usable CUDA context after 1 minute on ${SLURMD_NODENAME:-this node}."
    echo "   See $OUT/diagnose_gpu.txt for the driver / runtime diagnostics."
    # Remember the node and resubmit excluding every node that has refused so far, up to RETRY_MAX times.
    RETRY=${RETRY:-0}
    RETRY_MAX=${RETRY_MAX:-4}
    echo "${SLURMD_NODENAME:-unknown}" >> results/unusable_nodes.txt
    EXCLUDE=$(sort -u results/unusable_nodes.txt | grep -v '^unknown$' | paste -sd, -)
    if [ "$RETRY" -lt "$RETRY_MAX" ] && [ -n "$EXCLUDE" ]; then
      echo "   resubmitting with --exclude=$EXCLUDE (retry $((RETRY + 1))/$RETRY_MAX)"
      sbatch --exclude="$EXCLUDE" --export=ALL,RETRY=$((RETRY + 1)),RETRY_MAX="$RETRY_MAX" run_ruche.sbatch
    else
      echo "   giving up after $RETRY retries; inspect $OUT/diagnose_gpu.txt"
    fi
    exit 1
  fi
  echo "   attempt $attempt/6: device unavailable, waiting 10 s"
  sleep 10
done

# ---- preflight: build any missing tool, then report what the benchmark will actually run ----
echo "== preflight: tools and methods"
GPU_ARCHS="$ARCH" bash build_tools.sh || echo "   WARNING: a tool failed to build; the benchmark will skip it"
python - <<'PY'
import importlib.util
import os

print("   paragram :", "importable" if importlib.util.find_spec("paragram") else "MISSING")
print("   gdel3d   :", "importable" if importlib.util.find_spec("gdel3d") else "MISSING (pyGDel3D not installed)")
print("   geodel   :", "importable" if importlib.util.find_spec("geodel") else "MISSING (GeoDel not installed)")
for tool, what in (
    ("bin/cgal_delaunay", "CGAL parallel/sequential"),
    ("bin/dewall", "Local DeWall"),
    ("bin/gstar4d", "gStar4D"),
):
    print(f"   {tool:17s}:", f"OK ({what})" if os.access(tool, os.X_OK) else f"MISSING -> {what} will be skipped")
PY

DATASETS="--no-analytic --models"
[ "${FULL:-0}" = "1" ] && DATASETS="--n 20000"
COMMON="--unit-cube --repeats $REPEATS --slow-repeats 2 --slow-threshold 100 --verbose \
  --cgal-bin bin/cgal_delaunay --dewall-bin bin/dewall --gstar4d-bin bin/gstar4d \
  --geodel auto --geodel-threads $SLURM_CPUS_PER_TASK --tool-timeout $TOOL_TIMEOUT \
  --ply data/*.ply $DATASETS"

echo "== warm-up: JIT-compile Paragram's CUDA extension"
python -c "import torch, paragram; d = paragram.voronoi_diagram(torch.rand(10000, 3, device='cuda'), bbox_pad=10.0); print('paragram ready, status ok:', int((d.status == 0).sum()), '/ 10000')"

echo "== corrected Paragram (pad 10x, CPU repair) + gDel3D + gStar4D + Local DeWall + GeoDel + CGAL parallel/sequential"
python test_delaunay_surfaces.py $COMMON --paragram-bbox-pad 10 --repair on --json "$OUT/results.json" 2>&1 | tee "$OUT/run.log"

echo "== baseline Paragram (upstream box, no repair), 3 repeats -- Paragram only, the others are unaffected"
python test_delaunay_surfaces.py $COMMON --repeats 3 --paragram-bbox-pad -1 --repair off \
  --gdel3d off --geodel off --gstar4d-bin "" --dewall-bin "" --json "$OUT/results_baseline.json" 2>&1 | tee "$OUT/run_baseline.log"

echo "== jittered (1e-6) run, 3 repeats"
python test_delaunay_surfaces.py $COMMON --repeats 3 --jitter 1e-6 --paragram-bbox-pad 10 --repair on --json "$OUT/results_jitter.json" 2>&1 | tee "$OUT/run_jitter.log"

python make_report.py "$OUT/results.json" --baseline "$OUT/results_baseline.json" --jitter "$OUT/results_jitter.json" -o "$OUT/report.md"
echo "== done: $OUT/report.md"
