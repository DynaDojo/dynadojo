"""
Functions for running experiments for the paper. See __main__.py for usage.
Parameters are specified in experiments/params.py
"""
import itertools
import os
import pandas as pd
import numpy as np
from dynadojo.challenges import  FixedError, FixedComplexity, FixedTrainSize, ScalingChallenge
from .params import _get_system, _get_algo, _get_params, _serialize_params, _deserialize_params
import json


def get_max_splits(s="lds", m="lr", challenge_cls:type[ScalingChallenge] = FixedComplexity,):
    params = _get_params(s, m, challenge_cls=challenge_cls)
    trials = params["trials"]
    L = params.get("L", [params.get("l", 0)])

    return trials * len(L)

def run_challenge(
        s ="lds",
        a = "lr",
        output_dir="experiments/outputs", 
        split=(1,1),
        challenge_cls:type[ScalingChallenge] = FixedComplexity,
        num_cpu_parallel=None
        ):
    """
    Run a fixed complexity challenge and save the results to a csv file.
    :param s: system short name, defined in params.py system_dict
    :param m: algo short name, defined in params.py algo_dict
    :param output_dir: directory to save results
    :param split: tuple (split_num, total_splits) to run a subset of the total number of system runs, as specified by L and trials the challenge parameters
    """
    print(f"Running {_get_base_filename(s, a, challenge_cls)} {split=}")

    system = _get_system(s)
    algo = _get_algo(a)
    
    # Get challenge parameters
    challenge_params = _get_params(s, a, challenge_cls=challenge_cls)
    
    # Get evaluate parameters and remove from challenge parameters
    evaluate_params = challenge_params.get("evaluate", {})
    del challenge_params["evaluate"]
    evaluate_params["algo_cls"] = algo

    # Override system class
    challenge_params["system_cls"] = system

    # Override num_cpu_parallel
    if num_cpu_parallel:
        evaluate_params['num_parallel_cpu'] = num_cpu_parallel

    # Get L and trials
    L = challenge_params.get("L", [challenge_params.get("l", 0)])
    trials = challenge_params['trials']
    
    # Split trials into total_splits and run split_num
    if split:
        assert isinstance(split, tuple), "split must be a tuple, (split_num, total_splits)"
        split_num, total_splits = split
        runs = _get_runs(L, trials, split_num, total_splits) # list[tuples(trial, l)]
        print(f"Running split {split_num} of {total_splits} with runs {runs} --- list[tuples(trial, l)]")
    else:
        runs = None

    challenge = challenge_cls(
        **challenge_params,
    )

    # Save data to csv, specifying split if necessary
    if challenge_cls == FixedComplexity:
        path = f"{output_dir}/fc/{s}"
    elif challenge_cls == FixedTrainSize:
        path = f"{output_dir}/fts/{s}"
    else: # challenge_cls == FixedError:
        path = f"{output_dir}/fe/{s}"

    filename = _get_base_filename(s, a, challenge_cls)
    path = f"{path}/{filename}" #add subdir for experiment
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if split is not None:
        filename += f"_{split_num}-of-{total_splits}"
    filename += ".csv"
    file = f"{path}/{filename}"

    # save params
    if split_num == 1 or not os.path.exists(f"{path}/params.json"):
        with open(f"{path}/params.json", "w") as f:
            json.dump(_serialize_params(challenge_params, evaluate_params), f, indent=4, sort_keys=True)

    for run in runs:
        # Evaluate one run at a time and save to csv immediately!
        data = challenge.evaluate(
            **evaluate_params, 
            id=_get_base_filename(s, a, challenge_cls), 
            # Which trials and l pairings to evaluate. If None, evaluate all trials on all L. 
            # This is calculated by the split argument!
            rep_l_filter = [run]
        )
        data.to_csv(file, mode='a', index=False, header=not os.path.exists(file))
        print(f"COMPLETED RUN {run} -- {split_num=} / {total_splits=}")
    print(f"COMPLETED SPLIT -- {split_num=} / {total_splits=}")
    
    

def make_plots(
        s ="lds",
        a = "lr",
        output_dir="experiments/outputs",
        challenge_cls:type[ScalingChallenge] = FixedComplexity,
        save=True
    ):
    if challenge_cls == FixedComplexity:
        path = f"{output_dir}/fc/{s}"

    if challenge_cls == FixedTrainSize:
        path = f"{output_dir}/fts/{s}"
    
    if challenge_cls == FixedError:
        path = f"{output_dir}/fe/{s}"

    filebase = _get_base_filename(s, a, challenge_cls)
    csv_path = f"{path}/{filebase}"
    csv_filename = filebase + ".csv"
    figure_filename = filebase + ".pdf"

    
    files = _find_matching_files(csv_path, csv_filename) 
    # if not (os.path.exists(csv_path) and os.path.isdir(csv_path)):
    #     print(f"No plot created: Path {csv_path} does not exist or is not a directory")
    #     return
    # TODO: provide proper error message if no files found
    if len(files) <= 0:
        print(f"No plot created: No files matching {csv_filename} found in {csv_path}")
        return 

    data = pd.DataFrame()
    # Handling split challenge runs
    # Concatenate all files into one dataframe and drop duplicates
    for file in files:
        try:
            df = pd.read_csv(file)
        except:
            continue
        data = pd.concat([data, df])
    data = data.drop_duplicates()
    kwargs = {}
    kwargs["show_stats"] = True
    g = challenge_cls.plot(data, show=False, **kwargs)
    # linear axes instead of log
    # g.set(xscale="linear", yscale="linear")
    if save:
        g.figure.savefig(f"{path}/{figure_filename}", bbox_inches='tight')
    print(f"Plot created with {len(files)} files in {path} and {len(data)} rows: {figure_filename} ")
    # for file in files:
    #     print(f"\t- {file}")
    return g, data


### Helper functions

def _get_base_filename(sys:str, algo:str, challenge_cls:type[ScalingChallenge]):
    if challenge_cls == FixedComplexity:
        params = _get_params(sys, algo, challenge_cls=challenge_cls)
        l = params["l"]
        filebase = f"fc_{sys}_{algo}_{l=}"

    elif challenge_cls == FixedTrainSize:
        params = _get_params(sys, algo, challenge_cls=challenge_cls)
        n = params["n"]
        filebase = f"fts_{sys}_{algo}_{n=}"
    
    elif challenge_cls == FixedError:
        params = _get_params(sys, algo, challenge_cls=challenge_cls)
        e = params["target_error"]
        ood = params["evaluate"]["ood"]
        filebase = f"fe_{sys}_{algo}_{e=}"
        if ood:
            filebase = f"fe_ood_{sys}_{algo}_{e=}"
    else:
        raise ValueError("challenge_cls must be FixedComplexity, FixedTrainSize, or FixedError")

    return filebase

def _find_matching_files(path, filename, extended=False):
    """
    Find all files in path that match filename.
    For example, if filename is "fc_lds_lr_l=5.csv", this will return all files in path that contain "fc_lds_lr_l=5" and end with ".csv"
    """
    file_base = ".".join(filename.split(".")[:-1])
    file_ext = filename.split(".")[1]
    
    
    if not (os.path.exists(path) and os.path.isdir(path)):
        print(f" Path {path} does not exist or is not a directory")
        return []
    file_base += "_"
    files = os.listdir(path)
    matching = [f"{path}/{f}" for f in files if file_base in f and file_ext in f]
    return matching

def _get_runs(L, trials, split_num, total_splits):
    """
    Get the runs for a given split_num and total_splits. 
    Each run is a tuple (trial, l), where trial is the trial number and l is the L value for the system.
    This is used to split the challenge into parallelizable chunks of runs.

    :param L: list of L values
    :param trials: number of trials
    :param split_num: which split to run
    :param total_splits: total number of splits
    :return: list of tuples (trial, l) to run; to be passed to FixedComplexity.evaluate as eval_rep_l argument
    """
    assert 1 <= total_splits <= trials * len(L), "cannot split into more trials x L than there are"
    assert 1 <= split_num <= total_splits, "split_num must be less than total_splits and greater than 0"
    runs = list(itertools.product(range(trials), L))
    k, mod = divmod(len(runs), total_splits)
    splits = list(runs[i*k+min(i, mod):(i+1)*k+min(i+1, mod)] for i in range(total_splits))
    return splits[split_num-1]


