# Activates the project's conda environment. Sourced by script/first_install.sh, by
# script/build_tools.sh and by every SLURM job, so the environment is defined in exactly one place.
#
#   source script/env.sh              # activate, fail if the environment is missing
#   CREATE=1 source script/env.sh     # ... create it first if it does not exist
#
# Knobs:
#   DELAUNAY_ENV   where the environment lives.  Default: $WORKDIR/envs/delaunay where the cluster
#                  provides $WORKDIR (Ruche), $HOME/envs/delaunay otherwise (CentraleSupelec's DGX).
#   CONDA_MODULE   module that puts conda on the PATH, when conda is not already there (Ruche).
#   CONDA_ROOT     a conda installed in the home directory, when the cluster ships none (the
#                  DGX documents plain venv); default $HOME/miniconda3.
#   GPU_ARCHS      CUDA architectures to build for, "7.0;8.0" by default (V100 and A100).
#
# Everything below is written to work on both clusters: nothing here may assume $WORKDIR exists or
# that an environment-module system is present.

DELAUNAY_ENV=${DELAUNAY_ENV:-${WORKDIR:-$HOME}/envs/delaunay}
CONDA_MODULE=${CONDA_MODULE:-anaconda3/2023.09-0/none-none}

# --- put conda on the PATH ----------------------------------------------------------------
# Three ways a cluster provides conda, tried in order:
#   1. it is already on the PATH;
#   2. an environment module puts it there (Ruche);
#   3. it is installed somewhere in the user's home (the DGX documents plain venv and ships no
#      conda, so one has to be installed there).
# `module` is a shell function, which command -v finds.
if ! command -v conda >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  set +u
  module purge >/dev/null 2>&1 || true
  module load "$CONDA_MODULE" >/dev/null 2>&1 || true
  set -u
fi
if ! command -v conda >/dev/null 2>&1; then
  for _c in "${CONDA_ROOT:-}" "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" \
            /opt/conda /usr/local/miniconda3; do
    if [ -n "$_c" ] && [ -x "$_c/bin/conda" ]; then
      export PATH="$_c/bin:$PATH"
      break
    fi
  done
  unset _c
fi
if ! command -v conda >/dev/null 2>&1; then
  cat >&2 <<'MSG'
conda not found, and this project needs it: nvcc, GCC 13, the CGAL headers and TBB come from
conda-forge and cannot be installed with pip into a plain venv.

If your cluster provides conda through modules, set CONDA_MODULE to the module name. Otherwise
install miniconda once in your home directory (no administrator rights needed), either by
letting the install script do it:

    INSTALL_CONDA=1 bash script/first_install.sh

or by hand:

    curl -fsSLo /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3" && rm /tmp/miniconda.sh

Set CONDA_ROOT to install it somewhere other than $HOME/miniconda3 -- the environment and its CUDA
toolkit need a few GB, which a home quota may not have.
MSG
  return 1 2>/dev/null || exit 1
fi

# `conda activate` needs the shell hook; `source activate` alone does not work in a non-interactive
# shell that has never initialised conda.
CONDA_BASE=$(conda info --base)
set +u
# shellcheck disable=SC1091
. "$CONDA_BASE/etc/profile.d/conda.sh"
set -u

# --- create it if asked -------------------------------------------------------------------
if [ ! -d "$DELAUNAY_ENV" ] && [ "${CREATE:-0}" = "1" ]; then
  # Keep the package cache off the home quota when the cluster gives us somewhere better.
  if [ -n "${WORKDIR:-}" ]; then
    export CONDA_PKGS_DIRS=$WORKDIR/.conda/pkgs
    mkdir -p "$CONDA_PKGS_DIRS"
  fi
  echo "-- creating $DELAUNAY_ENV (python 3.12, CUDA 12.8, GCC 13, CGAL, TBB)"
  conda create -y -p "$DELAUNAY_ENV" -c conda-forge \
    python=3.12 "cuda-toolkit=12.8" "cuda-nvcc=12.8" gxx_linux-64=13 gcc_linux-64=13 \
    cmake ninja tbb tbb-devel gmp mpfr cgal-cpp boost-cpp git
fi
if [ ! -d "$DELAUNAY_ENV" ]; then
  echo "no environment at $DELAUNAY_ENV -- run 'bash script/first_install.sh' first," >&2
  echo "or set DELAUNAY_ENV to an existing one." >&2
  return 1 2>/dev/null || exit 1
fi

# --- activate and set the build variables -------------------------------------------------
# conda's activation scripts (cuda-nvcc) read variables that may be unset: relax "set -u" here.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
set +u
conda activate "$DELAUNAY_ENV"
set -u

export CUDA_HOME=$CONDA_PREFIX
# The compilers come from the environment itself (gcc_linux-64), so this name is the same on any
# cluster; fall back to the system compiler if the environment was built differently.
command -v x86_64-conda-linux-gnu-gcc >/dev/null 2>&1 \
  && export CC=${CC:-x86_64-conda-linux-gnu-gcc} CXX=${CXX:-x86_64-conda-linux-gnu-g++} \
  || export CC=${CC:-gcc} CXX=${CXX:-g++}

# conda-forge keeps the CUDA headers and libraries under $CONDA_PREFIX/targets/x86_64-linux, which
# torch's JIT build does not add when compiling *host* code: make them visible.
CUDA_TARGET=$CONDA_PREFIX/targets/x86_64-linux
[ -d "$CUDA_TARGET/include" ] && export CPATH="$CUDA_TARGET/include${CPATH:+:$CPATH}"
[ -d "$CUDA_TARGET/lib" ] && export LIBRARY_PATH="$CUDA_TARGET/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
# the driver stub is a last resort for -lcuda; a GPU node's real libcuda.so wins if present
[ -d "$CUDA_TARGET/lib/stubs" ] && export LIBRARY_PATH="${LIBRARY_PATH:+$LIBRARY_PATH:}$CUDA_TARGET/lib/stubs"

export PARAGRAM_MAX_PLANES=${PARAGRAM_MAX_PLANES:-128} PARAGRAM_MAX_VERTS=${PARAGRAM_MAX_VERTS:-128}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# the studies live in src/ and import one another by plain module name
[ -d "$PWD/src" ] && export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
