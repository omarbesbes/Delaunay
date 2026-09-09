#!/bin/bash
# Wait until this node can actually create a CUDA context, and deal with the two ways it cannot.
#
#   bash cuda_wait.sh [script-to-resubmit] [attempts]
#
# Some Ruche GPU nodes accept a job and then refuse every CUDA context with
# "CUDA-capable device(s) is/are busy or unavailable" while nvidia-smi shows the GPU idle in
# Default mode.  Retrying on the same node does not help, so after `attempts` tries this records
# the node in results/unusable_nodes.txt and, if a script is given, resubmits it with
# --exclude=<every node that has refused so far> (up to RETRY_MAX=4 times).
#
# The other failure is a torch build with no kernels for this GPU (the cu128 wheels have no Volta):
# that is not transient, so it exits immediately with instructions instead of retrying.
set -uo pipefail
SCRIPT=${1:-}
ATTEMPTS=${2:-6}

probe() {
  python - <<'PY'
import sys

import torch

try:
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
except Exception as exc:  # noqa: BLE001
    msg = str(exc)
    print(f"CUDA probe failed: {msg.splitlines()[0]}", file=sys.stderr)
    try:
        cap = torch.cuda.get_device_capability(0)
    except Exception:  # noqa: BLE001
        cap = None
    print(
        f"torch {torch.__version__} (CUDA {torch.version.cuda}) has kernels for "
        f"{torch.cuda.get_arch_list()}; this GPU is compute capability {cap}",
        file=sys.stderr,
    )
    sys.exit(2 if ("no kernel image" in msg or "not supported" in msg) else 1)
print(
    f"CUDA OK: {torch.cuda.get_device_name(0)} "
    f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))} | torch {torch.__version__}"
)
PY
}

for attempt in $(seq 1 "$ATTEMPTS"); do
  out=$(probe 2>&1)
  rc=$?
  echo "$out" | sed 's/^/   /'
  [ "$rc" = 0 ] && exit 0
  if [ "$rc" = 2 ]; then
    echo "   FAILED: this torch build has no kernels for this GPU architecture."
    echo "   Submit to the A100 partition (#SBATCH --partition=gpua100), or reinstall torch with"
    echo "   Volta support:  pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126"
    exit 1
  fi
  [ "$attempt" = "$ATTEMPTS" ] && break
  echo "   attempt $attempt/$ATTEMPTS: device unavailable, waiting 10 s"
  sleep 10
done

NODE=${SLURMD_NODENAME:-$(hostname -s)}
echo "   FAILED: no usable CUDA context on $NODE after $((ATTEMPTS * 10)) s."
mkdir -p results
echo "$NODE" >> results/unusable_nodes.txt
bash diagnose_gpu.sh > "results/diagnose_gpu-${SLURM_JOB_ID:-manual}.txt" 2>&1 || true
RETRY=${RETRY:-0}
RETRY_MAX=${RETRY_MAX:-4}
EXCLUDE=$(sort -u results/unusable_nodes.txt | grep -v '^unknown$' | paste -sd, -)
if [ -n "$SCRIPT" ] && [ "$RETRY" -lt "$RETRY_MAX" ] && [ -n "$EXCLUDE" ]; then
  echo "   resubmitting $SCRIPT with --exclude=$EXCLUDE (retry $((RETRY + 1))/$RETRY_MAX)"
  sbatch --exclude="$EXCLUDE" --export=ALL,RETRY=$((RETRY + 1)),RETRY_MAX="$RETRY_MAX" "$SCRIPT"
else
  echo "   giving up; see results/diagnose_gpu-${SLURM_JOB_ID:-manual}.txt"
fi
exit 1
