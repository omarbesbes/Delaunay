#!/bin/bash
# GPU / CUDA diagnostics for "CUDA-capable device(s) is/are busy or unavailable" (cudaErrorDevicesUnavailable).
# Run inside a job (or an interactive srun) that has a GPU allocated:  bash diagnose_gpu.sh
# It never fails; it only prints. Everything below is read-only.
echo "=================== node / allocation ==================="
echo "host: $(hostname)   job: ${SLURM_JOB_ID:-<none>}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}  GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-<unset>}"

echo "=================== device files ==================="
ls -l /dev/nvidia* 2>&1 | sed 's/^/  /'
echo "  nvidia-uvm module: $(lsmod 2>/dev/null | grep -c nvidia_uvm) match(es) in lsmod"

echo "=================== nvidia-smi ==================="
nvidia-smi -L 2>&1 | sed 's/^/  /'
nvidia-smi --query-gpu=index,name,compute_cap,compute_mode,persistence_mode,memory.used,ecc.errors.uncorrected.volatile.total,driver_version \
           --format=csv 2>&1 | sed 's/^/  /'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | sed 's/^/  /'
echo "  Xid / RmInit errors in dmesg (may be unreadable as a user):"
dmesg 2>/dev/null | grep -iE "nvrm|xid" | tail -5 | sed 's/^/    /' || echo "    (dmesg not readable)"

echo "=================== driver library resolution ==================="
echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
for f in "$CONDA_PREFIX/lib/libcuda.so.1" "$CONDA_PREFIX/lib/libcuda.so" \
         "$CONDA_PREFIX/targets/x86_64-linux/lib/libcuda.so.1" \
         "$CONDA_PREFIX/targets/x86_64-linux/lib/stubs/libcuda.so" \
         /usr/lib64/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1; do
  [ -e "$f" ] && echo "  present: $(ls -l "$f" | sed 's/^.* -> /-> /;s/  */ /g')  [$f]"
done
echo "  a conda libcuda.so.1 (cuda-compat) would shadow the real driver: that is a known cause of this error"

echo "=================== driver API (raw error codes) ==================="
python - <<'PY' 2>&1 | sed 's/^/  /'
import ctypes, ctypes.util

for name in ("libcuda.so.1", "libcuda.so"):
    try:
        lib = ctypes.CDLL(name)
    except OSError as exc:
        print(f"{name}: cannot load ({exc})")
        continue
    print(f"{name}: loaded from {ctypes.util.find_library('cuda') or 'the dynamic loader'}")
    rc = lib.cuInit(0)
    print(f"  cuInit(0) -> {rc}   (0 = success, 100 = no device, 101 = invalid device, 46 = devices unavailable, 3 = not initialized, 999 = unknown)")
    n = ctypes.c_int()
    print(f"  cuDeviceGetCount -> rc={lib.cuDeviceGetCount(ctypes.byref(n))} count={n.value}")
    ver = ctypes.c_int()
    lib.cuDriverGetVersion(ctypes.byref(ver))
    print(f"  cuDriverGetVersion -> {ver.value}")
    break
PY
echo "  libcuda actually mapped into the process:"
python -c "
import ctypes, pathlib
ctypes.CDLL('libcuda.so.1')
print('\n'.join({l.split()[-1] for l in pathlib.Path('/proc/self/maps').read_text().splitlines() if 'libcuda' in l}))
" 2>&1 | sed 's/^/    /'

echo "=================== torch ==================="
python - <<'PY' 2>&1 | sed 's/^/  /'
import torch

print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
print("kernels for", torch.cuda.get_arch_list())
print("device_count:", torch.cuda.device_count())
try:
    print("capability:", torch.cuda.get_device_capability(0), "| name:", torch.cuda.get_device_name(0))
except Exception as exc:  # noqa: BLE001
    print("get_device_capability failed:", exc)
try:
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    print("allocation on cuda: OK")
except Exception as exc:  # noqa: BLE001
    print("allocation on cuda FAILED:", exc)
PY
echo "=================== end ==================="
