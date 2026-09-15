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

The `#SBATCH` directives in the job files target Ruche. On the DGX, override the partition *and*
switch the GPU request off -- `sbatch` flags win over the directives in the file:

```bash
METHOD=gdel3d PLY=data/voronoi_iarpa_001.ply JITTER=1e-6 CHECK=true \
  sbatch --partition=prod10 --gres=none --cpus-per-task=4 script/run_triangulate.sbatch

sbatch --partition=prod40 --gres=none script/run_jitter.sbatch
```

`--gres=none` is the part that is easy to get wrong. On the DGX a job does not ask for a GPU at
all: the partition *is* the GPU request, and the MIG slice is attached by the scheduler
(`~/slurm-prod10.sbatch`, the template the cluster installs in every home directory, has no
`--gres` line). Any explicit `--gres` is refused -- `gpu:1` by the submit plugin with *only
A100.80gb GRES in partition prod80*, and every specific name, including the `A100.80gb` that the
plugin's own message quotes, with *Requested node configuration is not available*. Since Ruche does
need `--gres=gpu:1`, the directive stays in the files and `--gres=none` cancels it here.

Each partition also caps the CPUs per GPU slice, and the files ask for 8: `prod10` allows 4, so it
needs `--cpus-per-task=4` as well. The submit plugin names the cap in its refusal (*too many CPUs
requested for partition prod10 (max. 4 for 1 requested GPU(s))*), so there is nothing to look up --
read it off the error and resubmit. The jobs derive their thread counts from
`$SLURM_CPUS_PER_TASK`, so a smaller allocation is measured correctly, just with fewer threads.

`--test-only` answers both questions in a second, without queueing anything:

```bash
sbatch --test-only --partition=prod10 --gres=none --cpus-per-task=4 script/run_triangulate.sbatch
```

`CGAL_THREADS` is worth passing on this cluster: the node has 128 cores, and if the affinity mask
does not narrow TBB to the allocated CPUs, `bin/cgal_delaunay` stops making progress (see above).

The partitions are MIG slices of one A100: `prod10`, `prod40`, `prod80` and `interactive10`,
11 slices in total. VRAM is rarely the constraint here -- gDel3D uses 0.78 GB for a 100 000-point
cloud -- so `prod10` is enough for a single triangulation and is the most available. Note that the
CPU methods (GeoDel, CGAL parallel) get whatever `--cpus-per-task` grants, so timings from the two
clusters are only comparable at equal thread counts.

Every job is submitted from the repository root: `sbatch script/run_jitter.sbatch`. Environment
variables at the top of each script (`REPEATS`, `METHODS`, `JITTERS`, ...) narrow a run without
editing it.

## gDel3D on small inputs: `patch_pygdel3d_final_flip.py`

gDel3D returned 5 tetrahedra for the 8 corners of a cube where every other method returns 6 --
and the same 5 for the jittered cube, whose Delaunay triangulation has 10. The output was the raw
insertion result without a single flip: it did not cover the hull, and gDel3D's own checker said
so (`self-check=False` in the logs). The cause is in `GpuDelaunay.cu`: the exact flipping pass
defers any active set smaller than 64 tetrahedra to "the last round", and that round only runs
once an insertion round has inserted fewer than 10 % of the points -- impossible below 11 points,
and unlikely for a few dozen. Inputs from 505 points up were never affected (their tetrahedron
counts match CGAL's), so the benchmark's numbers stand.

`script/patch_pygdel3d_final_flip.py` makes the last round run unconditionally, **and keeps the
upstream behaviour available**: one build serves both, selected by an environment variable that
every entry point inherits (`main.py`, the `sbatch` files, `--verify`):

```bash
sbatch script/run_benchmark.sbatch                     # corrected gDel3D (default)
GDEL3D_ORIGINAL=1 sbatch script/run_benchmark.sbatch   # gDel3D as published -- the reference run's behaviour
```

Which one produced a result is recorded, not remembered: gDel3D's statistics gain
`finalRoundSkippedNum`, 1 when the upstream behaviour skipped the round and 0 when it ran (the
corrected behaviour, or an input large enough for upstream's own rule to fire). It comes out in
`get_stats()` and hence in the benchmark's `stats_ms` block for every gDel3D measurement.

It is a separate patch, kept out of `first_install.sh` on purpose so the validated install path
stays as it is; apply it by hand, rebuild, and check the installed build on a GPU:

```bash
python script/patch_pygdel3d_final_flip.py third_party/pyGDel3D --rebuild   # GPU_ARCHS=8.0 for A100 only

# Ruche
srun --partition=gpua100 --gres=gpu:1 --time=00:10:00 --pty python script/patch_pygdel3d_final_flip.py --verify
# DGX: the prod* partitions refuse srun ("sbatch script only"); interactive10 takes it ...
srun --partition=interactive10 --gres=none --cpus-per-task=4 --time=00:10:00 --pty \
  python script/patch_pygdel3d_final_flip.py --verify
# ... or wrap it in a batch job on prod10 and read results/verify.o<jobid>
sbatch --partition=prod10 --gres=none --cpus-per-task=4 --time=00:10:00 --job-name=verify \
  --output=results/%x.o%j --wrap '. script/env.sh && python script/patch_pygdel3d_final_flip.py --verify'
```

`--verify` runs the cube, the jittered cube and random sets of 9 to 500 points through the
benchmark's own wrapper, **in both modes**, and checks each result independently: every
circumsphere empty, the boundary faces closing a convex surface, the summed volume equal to the
enclosed one. The `original` rows show the bug (and `skipped=1`), the `corrected` rows decide the
exit status. The script also upgrades an installation carrying its first version (which had no
switch); the patch is idempotent and independent of the order with `patch_pygdel3d.py`. To make it
part of a fresh install, add `python script/patch_pygdel3d_final_flip.py third_party/pyGDel3D`
after the `patch_pygdel3d.py` line of `first_install.sh`.

Verified on the DGX (job 8419, `interactive10`), first version: all 13 cases pass. The cube comes
out as 9 tetrahedra, 6 of them real and 3 flat -- zero-volume tetrahedra on the cube's co-planar
faces, the same artefact of gDel3D's symbolic perturbation as the 14 980 flat tetrahedra on
`voronoi_iarpa_001`; the volume covered is the cube's, and `degenerate_tets` in the benchmark
table will read 3 for it. The jittered cube gives CGAL's 10.

Large inputs, measured before and after on the same MIG slice (`script/run_triangulate.sbatch`,
`voronoi_iarpa_001`, `CHECK=true`): with `JITTER=1e-6`, 669 051 tetrahedra identical to CGAL's
both times, 0.141 s before and 0.128 s after (noise); on the raw cloud, 675 445 tetrahedra, of
which 14 980 flat ones CGAL does not have and none of CGAL's missing -- the reference run's
structure exactly.
