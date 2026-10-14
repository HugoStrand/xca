#!/bin/bash -l
#SBATCH --account=naiss2026-3-569-gpu
#SBATCH --partition=gpu
#SBATCH --job-name=xca-bs-gh200-python
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/nobackup/proj/disk/naiss2026-3-569/shared/qualification/xca-block-sparse-python-7b5d223/logs/%x-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2026-3-569/shared/qualification/xca-block-sparse-python-7b5d223/logs/%x-%j.err
set -euo pipefail
umask 002
cd "$SLURM_SUBMIT_DIR"
test "$(uname -m)" = aarch64
stack=/nobackup/proj/disk/naiss2026-3-569
# The minimal TRIQS stack avoids CoQui's separate nda/h5 Python overrides.
module --force purge
# Strip inherited project activations before loading the coherent Arm toolchain.
for key in PATH LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH PYTHONPATH CMAKE_PREFIX_PATH; do
  IFS=: read -r -a components <<< "${!key:-}"
  cleaned=''
  for component in "${components[@]}"; do
    case "$component" in
      "$stack/apps"|"$stack/apps/"*|"$stack/shared/apps-cuda"|"$stack/shared/apps-cuda/"*|"$stack/shared/dev/"*/cbuild*|'') continue ;;
    esac
    cleaned=${cleaned:+$cleaned:}$component
  done
  printf -v "$key" '%s' "$cleaned"
  export "$key"
done
module load GPU/buildenv-gcc/2025b-eb
module load GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
module load HDF5/1.14.6 Boost/1.88.0 CUDA/12.9.1 cuTENSOR/2.3.0.6-CUDA-12.9.1
unset TRIQS_ROOT TRIQS_DIR nda_ROOT nda_DIR h5_ROOT h5_DIR triqs_xca_ROOT
# Driver link stubs/compat libraries must not enter runtime search paths.
runtime_paths=''
IFS=: read -r -a candidates <<< "${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
for candidate in "${candidates[@]}"; do
  case "$candidate" in */stubs|*/stubs/*|*/compat|*/compat/*|'') continue ;; esac
  runtime_paths=${runtime_paths:+$runtime_paths:}$candidate
done
export LD_LIBRARY_PATH="$EBROOTOPENBLAS/lib:$EBROOTGCCCORE/lib64:$EBROOTOPENMPI/lib:$EBROOTHDF5/lib:$EBROOTCUDA/lib:$EBROOTCUTENSOR/lib${runtime_paths:+:$runtime_paths}"
export PYTHONNOUSERSITE=1
export OMPI_MCA_opal_cuda_support=true OMPI_MCA_accelerator=cuda
source "$stack/shared/apps-cuda/triqs/share/triqs/triqsvars.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
record="logs/python-$SLURM_JOB_ID"
mkdir -p "$record"
cp source-revisions.json "$record/"
trap 'status=$?; printf "%s\n" "$status" > "$record/exit-code.txt"' EXIT
scontrol show job "$SLURM_JOB_ID" -o > "$record/job.txt"
uname -a > "$record/architecture.txt"
nvidia-smi -q > "$record/gpu.txt"
module -t list > "$record/modules.txt" 2>&1
g++ --version > "$record/compiler.txt"
nvcc --version > "$record/nvcc.txt"
# Normalize source timestamps during staging, before occupying a GPU. The marker
# is supplied only after pinned source files have been normalized to this epoch.
test "$(cat source-mtime-epoch.txt)" = 946684800
cp source-mtime-epoch.txt "$record/"
cmake -S src -B cbuild_cuda \
  -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release \
  -DTRIQS_DIR="$stack/shared/apps-cuda/triqs/lib64/cmake/triqs" \
  -DCMAKE_INSTALL_PREFIX="$PWD/unused-install" \
  -DPythonSupport=ON -DBuild_Tests=ON -DBuild_Benchmarks=OFF \
  -DUpdate_Python_Bindings=OFF -DBuild_Documentation=OFF \
  -DXCA_CUDA_SMALL=OFF -DXCA_CUDA_BLOCK_SPARSE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 > "$record/configure.log" 2>&1
cp cbuild_cuda/CMakeCache.txt "$record/CMakeCache.txt"
# Bound compiler concurrency by the actual allocation; no CPU/memory overrides.
workers=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}
if ((workers > 6)); then workers=6; fi
timeout 420 cmake --build cbuild_cuda --parallel "$workers" --target \
  _cppdlr _triqs_xca _triqs_xca_cuda > "$record/build.log" 2>&1
# Inventory before running tests: CTest --show-only can overwrite LastTest.log.
ctest --test-dir cbuild_cuda --show-only=json-v1 > "$record/ctest.json"
export PYTHONPATH="$PWD/cbuild_cuda/python:$PWD/src/deps/adapol/src:$PWD/src/deps/pyed${PYTHONPATH:+:$PYTHONPATH}"
python3 - > "$record/imports.json" <<'PYIMPORT'
import json, platform
from pathlib import Path
import numpy, h5, nda_py, triqs
import triqs_xca, cppdlr
from triqs_xca import gpu, _backend_config as config, _triqs_xca, _triqs_xca_cuda
from cppdlr import _cppdlr
assert platform.machine() == 'aarch64'
assert config.CUDA_ENABLED and not config.CUDA_SMALL_ENABLED
assert gpu.block_sparse_ordered_available()
assert gpu.block_sparse_mapped_available()
assert gpu.block_sparse_real_nca_available()
build = Path('cbuild_cuda/python').resolve()
modules = [triqs_xca, cppdlr, _triqs_xca, _triqs_xca_cuda, _cppdlr]
for module in modules:
    assert Path(module.__file__).resolve().is_relative_to(build), module.__file__
print(json.dumps({'architecture': platform.machine(), 'numpy': numpy.__version__,
 'origins': {m.__name__: m.__file__ for m in [*modules, triqs, h5, nda_py]},
 'cuda': config.CUDA_ENABLED, 'cuda_small': config.CUDA_SMALL_ENABLED,
 'ordered': gpu.block_sparse_ordered_available(), 'mapped': gpu.block_sparse_mapped_available(),
 'real_nca': gpu.block_sparse_real_nca_available()}, indent=2))
PYIMPORT
sha256sum cbuild_cuda/python/triqs_xca/*.so cbuild_cuda/python/cppdlr/*.so > "$record/binaries.sha256"
timeout 180 ctest --test-dir cbuild_cuda --parallel 1 \
  -R '^(Py_cuda_backends|Py_block_sparse_cuda)$' \
  --output-on-failure > "$record/test.log" 2>&1
cp cbuild_cuda/Testing/Temporary/LastTest.log "$record/LastTest.log"
timeout 420 python3 src/experiments/block_sparse_cuda/gh200_qualification/run_python_mpi.py \
  --ctest-json "$record/ctest.json" --record "$record/mpi-command.json" \
  --launcher /software/sse2/generic/manual/ssetools/v2.1.0/bin/mpprun --launcher-arg=--no-modules --launcher-arg=--allow-runpath \
  --launcher-arg=-n --launcher-arg=2 > "$record/mpi.log" 2>&1
printf '%s\n' PASS > "$record/result.txt"
