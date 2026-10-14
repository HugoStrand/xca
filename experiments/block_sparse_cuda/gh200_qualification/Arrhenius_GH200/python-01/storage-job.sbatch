#!/bin/bash -l
#SBATCH --account=naiss2026-3-569-gpu
#SBATCH --partition=gpu
#SBATCH --job-name=xca-bs-gh200-storage
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=1
#SBATCH --time=00:10:00
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
record="logs/python-storage-$SLURM_JOB_ID"
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
# Reuse the verified build/serial results; fail if any extension changed.
previous=logs/python-3851249
sha256sum --check "$previous/binaries.sha256" > "$record/binary-check.txt"
cp "$previous/imports.json" "$record/qualified-imports.json"
cp "$previous/ctest.json" "$record/ctest.json"
printf '%s\n' "$previous" > "$record/reused-checkpoint.txt"
# Give the site helper explicit OpenMPI metadata on a private alias of the
# identical configured Python interpreter. Python itself does not link libmpi,
# so automatic mpprun detection otherwise falls back to plain srun.
qualified_python="$PWD/qualified-mpi-python"
if ! test -L "$qualified_python"; then ln -s "$(command -v python3)" "$qualified_python"; fi
python3 - "$qualified_python.hpcinfo" "$EBROOTOPENMPI" <<'PYMPI'
import json, sys
from pathlib import Path
# Reuse the site's PMIx/fabric environment and launch arguments verbatim.
profile = Path(sys.argv[2]) / 'lib/libmpi.so.hpcinfo'
entries = json.loads(profile.read_text())
selected = next(entry for entry in entries if entry['launcher'] == 'srun')
assert selected['env']['SLURM_MPI_TYPE'] == 'pmix'
assert 'pmix' in selected['launch_args']
Path(sys.argv[1]).write_text(json.dumps(selected, indent=2) + '\n')
PYMPI
cp "$qualified_python.hpcinfo" "$record/python.hpcinfo"
sha256sum run_python_mpi.py > "$record/mpi-runner.sha256"
timeout 420 python3 run_python_mpi.py \
  --verbose --ctest-json "$record/ctest.json" --record "$record/mpi-command.json" --python "$qualified_python" \
  --launcher /software/sse2/generic/manual/ssetools/v2.1.0/bin/mpprun \
  --launcher-arg=--no-modules --launcher-arg=--allow-runpath --launcher-arg=--launcher=srun --launcher-arg=--handler=hpcinfo \
  --launcher-arg=-n --launcher-arg=2 \
  BlockSparseCudaTests.test_storage_policies_all_observables \
  BlockSparseCudaTests.test_storage_preferences_serialization \
  BlockSparseCudaTests.test_topology_storage_admission_and_transitions > "$record/mpi.log" 2>&1
printf '%s\n' PASS > "$record/result.txt"
