# Model-free Reinforcement Learning for Spatiotemporal Tasks using Symbolic Automata

- ArXiV version: <https://arxiv.org/abs/2202.02404>

In this paper, we propose using formal specifications in the form of symbolic automata:
these serve as a generalization of both bounded-time temporal logic-based specifications
and finite-state automata.


## Setup

The environment for running the experiments can be created using `conda` and `pip`:

```sh
$ conda env create -f environment.yaml
$ conda activate sym-rl
$ pip install -e .
# or (to recreate the plots)
# pip install -e .[plots]
```

## Scripts

All relevant scripts to run the experiments and generate figures are in the `bin/` directory

```
bin/
├── create_plots.py
├── plot_symbolic_potential.py
└── run_qlearning.py
```

### `run_qlearning.sh`

Run's all the automata-based algorithms for each environment.

**NOTE:** the script is used to run SLURM jobs on USC's HPC _and_
on the local machine. If you want to use it with SLURM, please change the paths and
settings in there to correspond to your HPC configuration.

### `run_qlearning.py`:

```sh
$ python3 "./bin/run_qlearning.py" \
    --log-dir logs \
    --env $SPEC \
    --method $REWARD_METHOD \
    --seed $RL_SEED \
    --resume
```

- `$SPEC` can be one of `GridMap01UnboundedReach-v0`, `GridMap01BoundedReach-v0`,
  `GridMap02BoundedRecurrence2-v0`, `GridMap03Sequential-v0`,
  `GridMap03BoundedSequential-v0`, or `GridMap04BranchReach-v0`.
- `$REWARD_METHOD` can be one of `sparse`, `symbolic`, or `lavaei2020`.
- `$RL_SEED` should be an integer seed for the experiment.


### `generate_symaut.py`

This script is used to convert STL specifications into Python code that constructs the
corresponding symbolic automata.

### `create_plots.py`

Used to create plots from logged experiment results.
