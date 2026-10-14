#!/bin/bash -l
#SBATCH --account=naiss2026-3-569-gpu
#SBATCH --partition=gpu
#SBATCH --job-name=xca-bs-gh200-density
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/nobackup/proj/disk/naiss2026-3-569/shared/qualification/xca-block-sparse-6987c4d/logs/%x-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2026-3-569/shared/qualification/xca-block-sparse-6987c4d/logs/%x-%j.err
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
record="logs/density-$SLURM_JOB_ID"
mkdir -p "$record"
cp source-revisions.json "$record/"
uname -a > "$record/architecture.txt"
nvidia-smi -q > "$record/gpu.txt"
module -t list > "$record/modules.txt" 2>&1
g++ --version > "$record/compiler.txt"
nvcc --version > "$record/nvcc.txt"
# Archive commit times can be ahead of the cluster clock. Normalize only source
# metadata to a fixed past date, avoiding repeated Make rebuilds without edits.
python3 - <<'PY'
from pathlib import Path
import os
for path in Path('src').rglob('*'):
    if path.is_file() and not path.is_symlink():
        os.utime(path, (946684800, 946684800))
PY
cmake -S src -B cbuild_cuda \
  -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release \
  -DTRIQS_DIR="$stack/shared/apps-cuda/triqs/lib64/cmake/triqs" \
  -DCMAKE_INSTALL_PREFIX="$PWD/unused-install" \
  -DPythonSupport=OFF -DBuild_Tests=ON -DBuild_Benchmarks=ON \
  -DUpdate_Python_Bindings=OFF -DBuild_Documentation=OFF \
  -DXCA_CUDA_SMALL=OFF -DXCA_CUDA_BLOCK_SPARSE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 > "$record/configure.log" 2>&1
cp cbuild_cuda/CMakeCache.txt "$record/CMakeCache.txt"
# Bound compiler concurrency by the actual allocation; no CPU/memory overrides.
workers=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}
if ((workers > 6)); then workers=6; fi
cmake --build cbuild_cuda --parallel "$workers" --target \
  block_sparse_cpu_density_sets_benchmark block_sparse_cpu_batched_density_sets_benchmark \
  block_sparse_general_density_sets_benchmark block_sparse_scalar_density_sets_benchmark \
  block_sparse_scalar_sets_tests block_sparse_cuda_device_ops block_sparse_cuda_dlr_ops \
  block_sparse_cuda_diagram_evaluator block_sparse_cuda_admission > "$record/build.log" 2>&1
ctest --test-dir cbuild_cuda \
  -R '^(experiments/block_sparse_cuda/scalar_sets|block_sparse/cuda_(device_ops|dlr_ops|diagram_evaluator|admission))$' \
  --output-on-failure > "$record/test.log" 2>&1
cp cbuild_cuda/Testing/Temporary/LastTest.log "$record/LastTest.log"
python3 run_cohort.py --cohort density --build cbuild_cuda --output "$record/results" \
  --revisions source-revisions.json > "$record/campaign.log" 2>&1
printf '%s\n' PASS > "$record/result.txt"
