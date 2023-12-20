#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=15
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks-per-gpu=2
#SBATCH -o slurm-%x-%j.out
#SBATCH --account jdeshmuk_696
#SBATCH --gres=gpu:1


usage() {
  echo "USAGE: ./bin/experiment.job <SPEC> <LOGDIR>"
  echo "NOTE: The script must be run from the root of the sym-rl-experiments repo"
}

if [[ $# -ne 2 ]]; then
  >&2 usage
  exit 1
fi

SPEC="$1"
LOGDIR="$2"
echo "Running env: $SPEC"
echo "Logging to: $LOGDIR"

if [[ ! -z "$SLURM_NNODES" ]]; then
  echo "Number of nodes = $SLURM_NNODES"
  module purge
  module load gcc/8.3.0
  module load mesa-glu/9.0.0
  module load anaconda3
  module load parallel

  source /home1/anandbal/.bashrc
  conda deactivate
  echo "Activating sym-rl"
  conda activate sym-rl

  NENVS=$(($(nproc --all)))

  srun="srun --overlap -n1"
  parallel="parallel --delay .2 -j $SLURM_NNODES"
else
  NJOBS=$(($(nproc)/2))
  echo "Number of parallel procs = $NJOBS"

  NENVS="1"

  srun=""
  parallel="parallel --halt soon,fail=1 --progress --delay .2 -j $NJOBS"
fi

which python3
python3 --version

echo "Current working dir = $PWD"

export PYOPENGL_PLATFORM=egl 

$parallel --res "$LOGDIR/$SPEC/{1}/{2}/" \
  $srun \
  python3 "./bin/run_dqn.py" \
  --log-dir $LOGDIR \
  --env $SPEC \
  --method {1} \
  --seed {2} \
  --n-procs $NENVS \
  ::: "sparse" "symbolic" "lavaei2020" \
  ::: 00000 00100 00200 00300 00400


