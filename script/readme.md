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

The SLURM scripts are written for Ruche (Mesocentre Paris-Saclay, partition `gpua100`); adapt the
partition, `--gres` and walltime for another cluster. Every job is submitted from the repository
root: `sbatch script/run_jitter.sbatch`. Environment variables at the top of each script
(`REPEATS`, `METHODS`, `JITTERS`, ...) narrow a run without editing it.
