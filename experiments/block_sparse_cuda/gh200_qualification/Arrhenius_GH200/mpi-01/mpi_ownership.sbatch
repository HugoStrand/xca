#!/bin/bash -l
#SBATCH --account=naiss2026-3-569-gpu
#SBATCH --partition=gpu
#SBATCH --job-name=xca-bs-gh200-mpi-phases
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=1
#SBATCH --time=00:05:00
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
record="logs/mpi-phases-$SLURM_JOB_ID"
mkdir -p "$record"
trap 'status=$?; printf "%s\n" "$status" > "$record/exit-code.txt"' EXIT
scontrol show job "$SLURM_JOB_ID" -o > "$record/job.txt"
uname -a > "$record/architecture.txt"
nvidia-smi -q > "$record/gpu.txt"
module -t list > "$record/modules.txt" 2>&1
previous=logs/python-3851249
sha256sum --check "$previous/binaries.sha256" > "$record/binary-check.txt"
cp "$previous/binaries.sha256" "$record/"
cp "$previous/imports.json" "$record/qualified-imports.json"
printf '%s\n' "$previous" > "$record/reused-checkpoint.txt"
bench="$PWD/bench-mpi/0a55b82"
cp "$bench/source-revisions.json" "$record/"
sha256sum "$bench"/*.py "$bench/mpi_ownership.sbatch" > "$record/benchmark.sha256"
python3 - "$previous/ctest.json" "$record/environment.json" <<'PYENV'
import json, sys
from pathlib import Path
case = next(t for t in json.loads(Path(sys.argv[1]).read_text())['tests'] if t['name']=='Py_block_sparse_cuda')
props = {p['name']:p['value'] for p in case['properties']}
paths=[]
for entry in props.get('ENVIRONMENT_MODIFICATION', []):
    key, value=entry.split('=',1)
    if key=='PYTHONPATH':
        op, path=value.split(':',1)
        assert op=='path_list_prepend'
        paths.insert(0,path)
assert len(paths)==3
Path(sys.argv[2]).write_text(json.dumps({'python':case['command'][0], 'python_paths':paths},indent=2)+'\n')
PYENV
qualified_python="$PWD/qualified-mpi-python"
if ! test -L "$qualified_python"; then ln -s "$(command -v python3)" "$qualified_python"; fi
python3 - "$qualified_python.hpcinfo" "$EBROOTOPENMPI" <<'PYMPI'
import json, sys
from pathlib import Path
entries=json.loads((Path(sys.argv[2])/'lib/libmpi.so.hpcinfo').read_text())
selected=next(e for e in entries if e['launcher']=='srun')
assert selected['env']['SLURM_MPI_TYPE']=='pmix' and 'pmix' in selected['launch_args']
Path(sys.argv[1]).write_text(json.dumps(selected,indent=2)+'\n')
PYMPI
cp "$qualified_python.hpcinfo" "$record/python.hpcinfo"
launcher=/software/sse2/generic/manual/ssetools/v2.1.0/bin/mpprun
run_case() {
  name=$1
  cap=$2
  shift 2
  python3 "$bench/run_mpi.py" --environment "$record/environment.json" \
    --python "$qualified_python" --launcher "$launcher" --ranks 2 --timeout "$cap" \
    --launcher-arg=--no-modules --launcher-arg=--allow-runpath \
    --launcher-arg=--launcher=srun --launcher-arg=--handler=hpcinfo \
    --record "$record/$name-command.json" -- --output "$PWD/$record/$name" "$@" \
    > "$record/$name.log" 2>&1
  python3 "$bench/summarize_mpi.py" "$record/$name" --output "$record/$name/summary.csv"
}
# Sequential, bounded, no rebuild and no expensive N10/N14 repetition.
run_case number 100 --backend cuda --operator-backend ordered --special-backend mapped --partition number --modes 3 --count 3 --samples 3
run_case mixed-retarded 75 --backend cuda --partition mixed --dynamic --modes 3 --count 1 --samples 2
run_case occupation-tail 75 --backend cuda --operator-backend ordered --partition occupation --modes 3 --first 7 --count 3 --samples 2
printf '%s\n' PASS > "$record/result.txt"
