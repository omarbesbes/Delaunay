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

echo "=================== CUDA runtime (cudart) ==================="
# The driver API can be fine while the *runtime* refuses to create the primary context.
python - <<'PY' 2>&1 | sed 's/^/  /'
import ctypes
import glob
import os

import torch

libs = sorted(glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart*.so*")))
print("torch's cudart:", libs or "not bundled (uses the system one)")
for path in libs[:1] or ["libcudart.so"]:
    try:
        rt = ctypes.CDLL(path)
    except OSError as exc:
        print(f"cannot load {path}: {exc}")
        break
    ver = ctypes.c_int()
    rt.cudaRuntimeGetVersion(ctypes.byref(ver))
    n = ctypes.c_int()
    rc_count = rt.cudaGetDeviceCount(ctypes.byref(n))
    rc_set = rt.cudaSetDevice(0)
    ptr = ctypes.c_void_p()
    rc_malloc = rt.cudaMalloc(ctypes.byref(ptr), 1024)
    print(f"cudaRuntimeGetVersion = {ver.value}")
    print(f"cudaGetDeviceCount -> rc={rc_count} count={n.value}")
    print(f"cudaSetDevice(0)   -> rc={rc_set}")
    print(f"cudaMalloc(1 KiB)  -> rc={rc_malloc}   (0 = success, 46 = cudaErrorDevicesUnavailable,")
    print("                                        209 = no kernel image, 100 = no device, 803 = system driver mismatch)")
    if rc_malloc == 0:
        rt.cudaFree(ptr)
PY

echo "=================== tiny CUDA program built with this nvcc ==================="
if command -v nvcc >/dev/null 2>&1; then
  tmp=$(mktemp -d)
  cat > "$tmp/t.cu" <<'CU'
#include <cstdio>
int main() {
  int n = 0;
  cudaError_t e = cudaGetDeviceCount(&n);
  printf("cudaGetDeviceCount -> %d (%s), count=%d
", (int)e, cudaGetErrorString(e), n);
  void *p = nullptr;
  e = cudaMalloc(&p, 1024);
  printf("cudaMalloc         -> %d (%s)
", (int)e, cudaGetErrorString(e));
  if (e == cudaSuccess) cudaFree(p);
  int rt = 0, dr = 0;
  cudaRuntimeGetVersion(&rt);
  cudaDriverGetVersion(&dr);
  printf("runtime %d, driver %d
", rt, dr);
  return 0;
}
CU
  arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
  if nvcc -O0 -o "$tmp/t" "$tmp/t.cu" -gencode "arch=compute_${arch},code=sm_${arch}" 2>"$tmp/err"; then
    "$tmp/t" 2>&1 | sed 's/^/  /'
  else
    echo "  nvcc failed to build the test:"; sed 's/^/    /' "$tmp/err"
  fi
  rm -rf "$tmp"
else
  echo "  nvcc not on PATH (activate the environment first)"
fi

echo "=================== CUDA runtime (cudart) ==================="
# The driver API can be fine while the *runtime* refuses to create the primary context.
python - <<'PY' 2>&1 | sed 's/^/  /'
import ctypes
import glob
import os

import torch

libs = sorted(glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart*.so*")))
print("torch's cudart:", libs or "not bundled (uses the system one)")
for path in libs[:1] or ["libcudart.so"]:
    try:
        rt = ctypes.CDLL(path)
    except OSError as exc:
        print(f"cannot load {path}: {exc}")
        break
    ver = ctypes.c_int()
    rt.cudaRuntimeGetVersion(ctypes.byref(ver))
    n = ctypes.c_int()
    rc_count = rt.cudaGetDeviceCount(ctypes.byref(n))
    rc_set = rt.cudaSetDevice(0)
    ptr = ctypes.c_void_p()
    rc_malloc = rt.cudaMalloc(ctypes.byref(ptr), 1024)
    print(f"cudaRuntimeGetVersion = {ver.value}")
    print(f"cudaGetDeviceCount -> rc={rc_count} count={n.value}")
    print(f"cudaSetDevice(0)   -> rc={rc_set}")
    print(f"cudaMalloc(1 KiB)  -> rc={rc_malloc}")
    print("   (0 = success, 46 = devices unavailable, 209 = no kernel image, 100 = no device, 803 = driver mismatch)")
    if rc_malloc == 0:
        rt.cudaFree(ptr)
PY

echo "=================== tiny CUDA program built with this nvcc ==================="
if command -v nvcc >/dev/null 2>&1; then
  tmp=$(mktemp -d)
  cat > "$tmp/t.cu" <<'CU'
#include <cstdio>
int main() {
  int n = 0;
  cudaError_t e = cudaGetDeviceCount(&n);
  printf("cudaGetDeviceCount -> %d (%s), count=%d\n", (int)e, cudaGetErrorString(e), n);
  void *p = nullptr;
  e = cudaMalloc(&p, 1024);
  printf("cudaMalloc         -> %d (%s)\n", (int)e, cudaGetErrorString(e));
  if (e == cudaSuccess) cudaFree(p);
  int rt = 0, dr = 0;
  cudaRuntimeGetVersion(&rt);
  cudaDriverGetVersion(&dr);
  printf("runtime %d, driver %d\n", rt, dr);
  return 0;
}
CU
  arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
  if nvcc -O0 -o "$tmp/t" "$tmp/t.cu" -gencode "arch=compute_${arch},code=sm_${arch}" 2>"$tmp/err"; then
    "$tmp/t" 2>&1 | sed 's/^/  /'
  else
    echo "  nvcc failed to build the test:"; sed 's/^/    /' "$tmp/err"
  fi
  rm -rf "$tmp"
else
  echo "  nvcc not on PATH (activate the environment first)"
fi

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
