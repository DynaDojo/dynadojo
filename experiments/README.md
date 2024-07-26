# Experiments CLI

Command line interface for running experiments.

```
python -m experiments --help
```

# Setup
- Please edit params.py with the appropriate params for your experiments.
    - She's a bit hacky. >_<
    - Algorithm keys can include suffixes separated by an underscore. So `lr_test` is a valid key for `lr`.
    - And suffixed algorithms inherit and override params from the algorithm dict. So `lr_test` gets all of `lr` params.
    - Try using the `make` CLI command and examining the config file if you want to check what's going on. 
- System and algorithm keys should be specified in keys.py

# Commands

## `make`
To create config file for experiment.

Arguments for make:
```
    --algo: which algo, short name, see params.py algo_dict
    --system: which system, short name, see params.py system_dict
    --challenge: which challenge, one of ["fc", "fts", "fe"]
    --output_dir: where to save config, default "experiments/outputs"
    --all: if True, make all params, default False
```

Usage:
    ```
    python -m experiments make --challenge <challenge_key> --system <system_key> --algo <algo_key> --output_dir <output_dir>
    python -m experiments make --challenge fc --system lds --algo lr_5
    ```

## `run`
To run experiment, given config file. Can split jobs over multiple compute nodes.

Arguments for run:
```
    --config_file: which config file to run
    --total_nodes: how many machines to run on (default 1, for running locally)
    --node: which node is being run, [1, total_nodes], default None which runs the whole challenge
    --output_dir: where to save results, default "experiments/outputs"
    --num_cpu_parallel: number of cpus to use for parallelization, default None which runs without parallelization
    --jobs: which jobs to run, comma separated list of integers, default None which runs all jobs
    --if_missing: if True, only run missing jobs, default False
```
Usage:
```
    python -m experiments \
        run \
        --config_file experiments/outputs/fc/lds/fc_lds_lr_l=10/config.json \
        --node 2 --total_nodes 10 \
        --num_cpu_parallel -2 \
        --if_missing

    python -m experiments run --num_cpu_parallel -2 --config_file experiments/outputs/fc/lds/fc_lds_lr_5_l=5/config.json 
```

## Chaining `make` and `run`
You can specifically chain `make` and `run` commands to make a config.json file and immediately run said file.

```
python -m experiments \
        make \
            <make args>
        run \
            <run args except --config_file>
```

## `plot`
To aggregate and plot results

Arguments for plot:
```
    --data_dir: where to load results from
    --output_dir: where to save plots, default "experiments/outputs"
```

Usage:
```
    python -m experiments plot --data_dir experiments/outputs/fc/lds/fc_lds_lr_l=10 --output_dir experiments/outputs
```

## `check`
To check which jobs were completed and which are missing. Prints a list which can be passed to `--jobs` argument of `run` command.

Arguments for check:
```
    --data_dir: where to load results from
```

Usage:
    ```
    python -m experiments check --data_dir experiments/outputs/fc/lds/fc_lds_lr_l=10
    ```

# Using the CLI with cluster compute

See code in the `dynadojo/slurm` directory. In particular, `dynadojo/slurm/scripts` which call on `dynadojo/slurm/jobscripts` which use the CLI within a Singularity container.
