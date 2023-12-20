#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=25-50
#SBATCH --mem=32GB
#SBATCH -o slurm-%x-%j.out


usage() {
  echo "USAGE: ./bin/experiment.job <LOGDIR>"
  echo "NOTE: The script must be run from the root of the sym-rl-experiments repo"
}

if [[ $# -ne 1 ]]; then
  >&2 usage
  exit 1
fi

LOGDIR="$1"

NJOBS=$(nproc)

echo "Logging to: $LOGDIR"

SPECS=(
  "GridMap01UnboundedReach-v0"
  "GridMap01BoundedReach-v0"
  "GridMap02BoundedRecurrence2-v0"
  "GridMap03Sequential-v0"
  "GridMap03BoundedSequential-v0"
  "GridMap04BranchReach-v0"
)

if [[ ! -z "$SLURM_NNODES" ]]; then
  echo "Number of nodes = $SLURM_NNODES"
  module purge
  module load gcc/8.3.0
  module load anaconda3
  module load parallel

  source /home1/anandbal/.bashrc
  conda deactivate
  echo "Activating sym-rl"
  conda activate sym-rl

  srun="srun --exclusive -N1 -n1"
  parallel="parallel --delay .2 -j $SLURM_NNODES"
else
  srun=""
  parallel="parallel --progress --delay .2 -j $NJOBS"
fi

which python3
python3 --version

echo "Current working dir = $PWD"

$parallel --res "$LOGDIR/{1}/{2}/parallel-seed{3}" \
  $srun \
  python3 "./bin/run_qlearning.py" \
  --log-dir $LOGDIR \
  --env {1} \
  --method {2} \
  --seed {3} \
  --resume \
  ::: ${SPECS[@]} \
  ::: "sparse" "symbolic" "lavaei2020" \
  ::: 0 100 200 300 400

