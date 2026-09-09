# Delaunay benchmark: GPU methods vs CGAL on surface point clouds

Benchmarks 3D Delaunay tetrahedralization methods on the same float32 point sets and compares every
result against an exact CGAL reference, tetrahedron for tetrahedron:

| method | what runs | precision |
|---|---|---|
| **Paragram + conversion** | Paragram (GPU Voronoi adjacency, float32) → exact CPU repair of failed / hull cells (`paragram_repair.py`) → 4-clique + insphere conversion to tetrahedra (`voronoi_to_delaunay.py`) | float32 GPU, float64 conversion |
| **gDel3D** | GPU insertion + flipping, CPU star splaying ([pyGDel3D](https://github.com/half-potato/pyGDel3D), patched) | float64, exact predicates |
| **Local DeWall** | GPU Delaunay-wall construction ([Local-DeWall](https://github.com/WuhengGao/Local-DeWall), patched for Linux) | float32, exact predicates |
| **CGAL parallel** | `Delaunay_triangulation_3` with `Parallel_tag` + TBB (`cgal_delaunay.cpp`), the reference | exact |
| **CGAL sequential** | same tool with `CGAL_THREADS=1` | exact |

For each dataset the script reports, per method: tetrahedron counts and set differences vs the reference
(Jaccard), empty-circumsphere violations, volume vs convex-hull volume, Euler characteristic, manifoldness,
volume / radius-ratio / dihedral-angle statistics, a breakdown of *why* sets differ (ties on co-spherical
groups, zero-volume tets, missing/spurious adjacency edges), and timings averaged over repeated runs with a
GPU / CPU split. `make_report.py` turns the JSON into a Markdown report with charts.

Datasets: the two point clouds in `data/` (≈100k points each), and optionally analytic surfaces (cube,
hollow cube, spheres, tori, Klein bottle, Möbius strip, trefoil) and classic meshes (bunny, spot, teapot,
cow, Suzanne, armadillo). With `--unit-cube` every dataset is normalised into the unit cube in float32 first,
so all methods triangulate *exactly* the same points (Local DeWall would otherwise renormalise internally).

## Running on Ruche (Mésocentre, SLURM)

Everything is installed into a conda environment under `$WORKDIR` (Python 3.12, CUDA 12.8 toolkit, GCC 13,
torch cu128, CGAL headers + TBB), so nothing depends on the cluster's module versions except `anaconda3`.

1. Connect and clone (login node):
   ```bash
   ssh <user>@ruche.mesocentre.universite-paris-saclay.fr
   cd $WORKDIR && git clone https://github.com/omarbesbes/Delaunay.git && cd Delaunay
   ```
2. One-time setup (login node, has internet; 15–30 min, mostly downloads and CUDA compilation):
   ```bash
   bash setup_ruche.sh
   ```
   It creates `$WORKDIR/envs/delaunay`, installs the Python packages, builds `bin/cgal_delaunay`
   (parallel CGAL), installs the patched Paragram and pyGDel3D, builds `bin/dewall` (Local DeWall) for
   V100 and A100 (`TORCH_CUDA_ARCH_LIST="7.0;8.0"`), and pre-downloads the meshes. Each step prints a
   check line; if one fails, the message says which tool is missing and the benchmark still runs without it.
3. Submit the benchmark. **Use the A100 partition** (`gpua100`, the default in the script): the
   `cu128` torch wheels contain no Volta (sm_70) kernels, so Paragram and gDel3D cannot run on the
   V100 partitions with that build. To use `gpu` / `gpu_test` (V100) instead, reinstall torch with
   Volta support first:
   ```bash
   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash setup_ruche.sh
   ```
   (Local DeWall and CGAL are standalone binaries and run on any of the partitions.)
   ```bash
   sbatch run_ruche.sbatch                 # the two point clouds, 10 timed repeats per method
   FULL=1 sbatch run_ruche.sbatch          # + analytic surfaces and meshes (20k samples each)
   REPEATS=5 sbatch run_ruche.sbatch
   squeue -u $USER                         # job state
   tail -f results/delaunay-bench.o<jobid> # live progress ([HH:MM:SS] lines: dataset, method, run i/N)
   ```
   The job runs three passes: corrected Paragram (10x clipping pad, CPU repair) with gDel3D, Local DeWall
   and CGAL; the upstream Paragram baseline (legacy clipping box, no repair; 3 repeats); and a jittered
   pass (3 repeats). A method whose first run exceeds 100 s is repeated only twice.
4. Results, in `results/<jobid>/`: `report.md` (+ PNG charts), `results*.json` (all numbers; written after
   every dataset, so partial results survive a crash), `run*.log` (full console output).
   Copy them back with `scp -r <user>@ruche...:$WORKDIR/Delaunay/results/<jobid> .`

Quick interactive test on a GPU node (1 h partition):
```bash
srun --partition=gpu_test --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:30:00 --pty bash
module load anaconda3/2023.09-0/none-none && source activate $WORKDIR/envs/delaunay
export CUDA_HOME=$CONDA_PREFIX PARAGRAM_MAX_PLANES=128 PARAGRAM_MAX_VERTS=128
python test_delaunay_surfaces.py --no-analytic --models --ply data/*.ply --unit-cube --repeats 2 --verbose \
  --cgal-bin bin/cgal_delaunay --dewall-bin bin/dewall --json results/test.json
```

## Troubleshooting (observed on Ruche)

| symptom | cause | fix |
|---|---|---|
| `no kernel image is available for execution on the device` | the `cu128` torch wheel has kernels for sm_75/80/86/90/100/120 only, and a V100 is sm_70 | use `--partition=gpua100`, or `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 REBUILD=1 bash setup_ruche.sh` |
| `CUDA-capable device(s) is/are busy or unavailable` while `nvidia-smi` shows the GPU idle in Default mode | node-specific driver state | the job records the node in `results/unusable_nodes.txt` and resubmits itself with `--exclude=` (up to `RETRY_MAX=4`) |
| `NVCC_PREPEND_FLAGS: unbound variable` | conda's `cuda-nvcc` activation script under `set -u` | handled: the scripts relax `set -u` around activation |
| `fatal error: cuda_runtime_api.h` in torch's JIT build | conda keeps the CUDA headers in `targets/x86_64-linux/include`, which torch does not add for host code | handled: the scripts export `CPATH` / `LIBRARY_PATH` |
| `ValueError: Unknown CUDA arch (10.1)` | torch's arch auto-detection on cu128 builds | handled: the job pins `TORCH_CUDA_ARCH_LIST` from `nvidia-smi --query-gpu=compute_cap` |

`bash diagnose_gpu.sh` prints the full picture on any GPU node: device files, compute mode, which `libcuda`
is mapped, raw `cuInit` / `cudaMalloc` error codes, a tiny `nvcc`-built CUDA program, and torch's arch list.

## Options worth knowing

```
--unit-cube                   same float32 points for every method (recommended)
--repeats N                   timed repetitions per method (correctness taken from run 1)
--slow-repeats 2 --slow-threshold 100   repetitions for methods slower than the threshold on run 1
--verbose                     timestamped progress on stderr
--paragram-bbox-pad 10 | -1   relative clipping pad (patched Paragram) | upstream absolute 1.0
--repair on|off, --repair-hull on|off   exact CPU repair of failed / convex-hull cells (capped at 20 % of the points)
--gdel3d auto|on|off, --dewall-bin PATH, --cgal-bin PATH, --cgal-python PATH
--n 20000, --models ..., --ply ..., --jitter 1e-6, --adjacency auto|paragram|qhull|ref-edges
```
Build variants of Paragram: `PARAGRAM_MAX_PLANES` / `PARAGRAM_MAX_VERTS` (default 64; the job uses 128),
`NO_FAST_MATH=1`. Paragram's CUDA extension is compiled on first import (a few minutes) into torch's
extension cache; the job does a warm-up call before timing.

## Files

- `test_delaunay_surfaces.py` — the benchmark driver (datasets, methods, metrics, JSON).
- `make_report.py` — Markdown report + charts from one or several JSON files.
- `voronoi_to_delaunay.py` — Voronoi adjacency → Delaunay tetrahedra (torch, GPU or CPU).
- `paragram_repair.py` — exact CPU fallback for Paragram's failed / hull cells.
- `cgal_delaunay.cpp` — parallel CGAL Delaunay command-line tool.
- `patch_paragram.py`, `patch_pygdel3d.py`, `patch_dewall.py` — source patches applied to the upstream
  repositories at setup time (documented at the top of each file).
- `setup_ruche.sh`, `run_ruche.sbatch` — cluster setup and SLURM job.
- `data/` — the two point clouds. `third_party/`, `bin/`, `results/` are created by the setup / jobs.

## Notes on interpreting results

- Identical tets to CGAL is the correctness bar; "tie" differences on co-spherical groups (regular grids,
  symmetric meshes) are legitimate; zero-volume tets (gDel3D on co-planar points) are flagged separately.
- Paragram's known limits: float32 cell clipping fails on sliver-shaped cells (`inconsistent_boundary`),
  loses edges silently on unbounded (hull) cells and on near-co-spherical input; the CPU repair fixes what
  is flagged or on the hull, and the report states what fraction of cells was recomputed on the CPU.
- Local DeWall is exact but very slow on near-co-spherical input (hundreds of seconds for 20k sphere points).
