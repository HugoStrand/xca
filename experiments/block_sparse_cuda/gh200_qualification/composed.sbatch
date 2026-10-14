#!/bin/bash -l
#SBATCH --account=naiss2026-3-569-gpu
#SBATCH --partition=gpu
#SBATCH --job-name=xca-bs-gh200-composed
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/nobackup/proj/disk/naiss2026-3-569/shared/qualification/xca-block-sparse-268f255/logs/%x-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2026-3-569/shared/qualification/xca-block-sparse-268f255/logs/%x-%j.err
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
record="logs/composed-$SLURM_JOB_ID"
mkdir -p "$record"
cp source-revisions.json "$record/"
trap 'status=$?; printf "%s\n" "$status" > "$record/exit-code.txt"' EXIT
# Check the existing one-GPU memory entitlement; do not request extra memory.
case "${SLURM_MEM_PER_NODE:-}" in ''|*[!0-9]*) echo 'Missing numeric Slurm host limit' >&2; exit 2 ;; esac
if ((SLURM_MEM_PER_NODE < 102400)); then echo 'Host entitlement below declared probe cap' >&2; exit 2; fi
printf '%s\n' "$SLURM_MEM_PER_NODE" > "$record/slurm-host-mib.txt"
cat /proc/meminfo > "$record/host-memory.txt"
if test -r /sys/fs/cgroup/memory.max; then cat /sys/fs/cgroup/memory.max > "$record/cgroup-memory-max.txt"; fi
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
  -DPythonSupport=OFF -DBuild_Tests=ON -DBuild_Benchmarks=ON \
  -DUpdate_Python_Bindings=OFF -DBuild_Documentation=OFF \
  -DXCA_CUDA_SMALL=OFF -DXCA_CUDA_BLOCK_SPARSE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 > "$record/configure.log" 2>&1
cp cbuild_cuda/CMakeCache.txt "$record/CMakeCache.txt"
# Bound compiler concurrency by the actual allocation; no CPU/memory overrides.
workers=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}
if ((workers > 6)); then workers=6; fi
cmake --build cbuild_cuda --parallel "$workers" --target \
  block_sparse_capacity_evaluator > "$record/build.log" 2>&1
ctest --test-dir cbuild_cuda --parallel 1 \
  -R '^experiments/block_sparse_cuda/capacity_evaluator$' \
  --output-on-failure > "$record/test.log" 2>&1
cp cbuild_cuda/Testing/Temporary/LastTest.log "$record/LastTest.log"
probe=cbuild_cuda/benchmarks/experiments/block_sparse_cuda/capacity_evaluator/block_sparse_capacity_evaluator
sha256sum "$probe" > "$record/binary.sha256"
/usr/bin/time -v "$probe" --modes 10 --lambda 4000 --samples 1 \
  > "$record/reduced.csv" 2> "$record/reduced.log"
# Complete N14 sector payloads, one active hybridization flavour, NCA only.
# Bound runtime and retain admission failures; never relax caps automatically.
timeout 480 /usr/bin/time -v "$probe" --modes 14 --lambda 4000 --samples 1 \
  --device-budget-mib 92160 --host-budget-mib 102400 --allow-large \
  > "$record/large.csv" 2> "$record/large.log"
printf '%s\n' PASS > "$record/result.txt"
