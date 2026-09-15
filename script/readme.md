# script/

Everything that is not the benchmark itself: installing, building, submitting, patching.

| file | what it does |
|---|---|
| `first_install.sh` | conda env, torch, the six libraries and the compiled tools -- **run on a login node** (compute nodes cannot reach github.com) |
| `build_tools.sh` | builds `bin/cgal_delaunay`, `bin/cgal_delaunay_profile`, `bin/dewall`, `bin/gstar4d`; idempotent, `--force` to rebuild |
| `run_benchmark.sbatch` | the correctness benchmark (`FULL=1` adds analytic surfaces and meshes) |
| `run_jitter.sbatch` | the jitter sweep |
| `run_scaling.sbatch` | time versus point count (`UPSAMPLE=densify` for the second axis) |
| `run_triangulate.sbatch` | one method on one PLY on a GPU node: `METHOD=gdel3d PLY=cloud.ply sbatch script/run_triangulate.sbatch` |
| `cuda_wait.sh` | waits for a usable CUDA context; resubmits excluding the node if none appears |
| `diagnose_gpu.sh` | what a GPU node can and cannot do, for a bug report |
| `check_gstar4d.py` | smoke-tests gStar4D alone, to tell a broken build from an input it cannot handle |
| `patch_*.py` | source patches applied to the upstream repositories at install time; each explains why at its top |
| `env.sh` | activates the conda environment (sourced by the install script and by every job); the single place where the environment is defined |

## Do not run the tools bare on a login node

`bin/cgal_delaunay` is built with TBB, which sizes its thread pool from the machine. On a login
node nothing restricts it, and on a 128-core DGX the parallel insertion of a small point set does
not merely slow down -- it stops making progress:

```
CGAL_THREADS=4   bin/cgal_delaunay /tmp/p.f64 /tmp/t.i32    # 0.30 s, 133 665 cells
CGAL_THREADS=128 bin/cgal_delaunay /tmp/p.f64 /tmp/t.i32    # never returns
```

20 000 points over 128 threads is 156 points each, and the threads spend their time retrying
against the lock grid. Set `CGAL_THREADS` when running the tool by hand outside a job.

Inside a job this does not arise: SLURM's affinity mask already tells TBB how many cores it has
(the Ruche logs read `parallel (8 threads)`), so the measurements are unaffected.

## Which cluster

The scripts run on both clusters we have used; `env.sh` finds conda either way.

| | Ruche (Mesocentre) | DGX (CentraleSupelec) |
|---|---|---|
| conda | `module load anaconda3/...`, done by `env.sh` | **none provided** -- install miniforge once with `INSTALL_CONDA=1` |
| environment | `$WORKDIR/envs/delaunay` | `$HOME/envs/delaunay` |
| GPU partition | `gpua100` (full A100 40 GB) | `prod10`, `prod20`, `prod40`, `prod80` (MIG slices) |

The DGX documents a plain Python `venv`, which is not enough here: nvcc, GCC 13, the CGAL headers
and TBB come from conda-forge and cannot be installed with pip. The first install there is
therefore

```bash
INSTALL_CONDA=1 bash script/first_install.sh     # installs miniforge in $HOME, then the rest
```

Miniforge rather than Miniconda: every package here comes from conda-forge, which Miniforge uses by
default. Anaconda's own channels would additionally require accepting their terms of service --
which stops `conda create` outright -- and restrict commercial use by large organisations. The
environment is created with `--override-channels -c conda-forge` for the same reason, so an
existing Miniconda works too.

`CONDA_ROOT` puts miniconda somewhere other than `$HOME/miniconda3`, and `DELAUNAY_ENV` moves the
environment itself -- together they need a few GB, which a home quota may not have.

The `#SBATCH` directives in the job files target Ruche. On the DGX, override them on the command
line -- `sbatch` flags win over the directives in the file:

```bash
# a full A100 (80 GB): the only DGX partition that allows 8 CPUs, which the CPU methods want
sbatch --partition=prod80 --gres=gpu:A100.80gb:1 --cpus-per-task=8 --mem=64G \
       script/run_jitter.sbatch

# a 40 GB slice, 4 CPUs
sbatch --partition=prod40 --gres=gpu:3g.40gb:1 --cpus-per-task=4 --mem=32G \
       script/run_benchmark.sbatch
```

On the DGX the number of CPUs is bounded by the MIG slice (4 per `1g.10gb`, 8 for a full
`A100.80gb`), so `--cpus-per-task` must be lowered together with the partition. The CPU methods
(GeoDel, CGAL parallel) then have fewer threads, which changes their timings -- worth stating if
results from the two clusters are compared.

Every job is submitted from the repository root: `sbatch script/run_jitter.sbatch`. Environment
variables at the top of each script (`REPEATS`, `METHODS`, `JITTERS`, ...) narrow a run without
editing it.
