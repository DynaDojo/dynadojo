"""
Functions for running experiments for the paper. See __main__.py for usage.
Parameters are specified in experiments/params.py
"""
import itertools
import os
import pandas as pd
import numpy as np
from dynadojo.challenges import  FixedError, FixedComplexity, FixedTrainSize
from dynadojo.abstractions import Challenge 
from .params import _get_system, _get_model, _get_params

def get_max_splits(s="lds", m="lr", challenge_cls:type[Challenge] = FixedComplexity,):
    params = _get_params(s, m, challenge_cls=challenge_cls)
    reps = params["reps"]
    L = params.get("L", [params.get("l", 0)])

    return reps * len(L)

def run_challenge(
        s ="lds",
        m = "lr",
        output_dir="experiments/outputs", 
        split=(1,1),
        challenge_cls:type[Challenge] = FixedComplexity,
        ):
    """
    Run a fixed complexity challenge and save the results to a csv file.
    :param s: system short name, defined in params.py system_dict
    :param m: model short name, defined in params.py model_dict
    :param output_dir: directory to save results
    :param split: tuple (split_num, total_splits) to run a subset of the total number of system runs, as specified by L and reps the challenge parameters
    """
    print(f"Running {_get_base_filename(s, m, challenge_cls)} {split=}")

    system = _get_system(s)
    model = _get_model(m)
    
    # Get challenge parameters
    challenge_params = _get_params(s, m, challenge_cls=challenge_cls)
    
    # Get evaluate parameters and remove from challenge parameters
    evaluate_params = challenge_params.get("evaluate", {})
    del challenge_params["evaluate"]
    evaluate_params["model_cls"] = model

    # Override system class
    challenge_params["system_cls"] = system

    # Get L and reps
    L = challenge_params.get("L", [challenge_params.get("l", 0)])
    reps = challenge_params['reps']
    
    # Split reps into total_splits and run split_num
    if split:
        assert isinstance(split, tuple), "split must be a tuple, (split_num, total_splits)"
        split_num, total_splits = split
        runs = _get_runs(L, reps, split_num, total_splits) # list[tuples(rep, l)]
        print(f"Running split {split_num} of {total_splits} with runs {runs} --- list[tuples(rep, l)]")
    else:
        runs = None

    challenge = challenge_cls(
        **challenge_params,
    )

    data = challenge.evaluate(
        **evaluate_params, 
        id=_get_base_filename(s, m, challenge_cls), 
        # Which reps and l pairings to evaluate. If None, evaluate all reps on all L. 
        # This is calculated by the split argument!
        rep_l_filter = runs
    )
    
    # Save data to csv, specifying split if necessary
    if challenge_cls == FixedComplexity:
        path = f"{output_dir}/fc/{s}"
    elif challenge_cls == FixedTrainSize:
        path = f"{output_dir}/fts/{s}"
    else: # challenge_cls == FixedError:
        path = f"{output_dir}/fe/{s}"

    if not os.path.exists(path):
        os.makedirs(path)
    filename = _get_base_filename(s, m, challenge_cls)
    if split is not None:
        filename += f"_{split_num}-of-{total_splits}"
    filename += ".csv"
    file = f"{path}/{filename}"
    data.to_csv(file, index=False)

def make_plots(
        s ="lds",
        m = "lr",
        output_dir="experiments/outputs",
        challenge_cls:type[Challenge] = FixedComplexity,
    ):
    if challenge_cls == FixedComplexity:
        path = f"{output_dir}/fc/{s}"

    if challenge_cls == FixedTrainSize:
        path = f"{output_dir}/fts/{s}"
    
    if challenge_cls == FixedError:
        path = f"{output_dir}/fe/{s}"

    filebase = _get_base_filename(s, m, challenge_cls)
    csv_filename = filebase + ".csv"
    figure_filename = filebase + ".pdf"

    if not (os.path.exists(path) and os.path.isdir(path)):
        print(f"No plot created: Path {path} does not exist or is not a directory")
        return
    files = _find_matching_files(path, csv_filename)
    if len(files) <= 0:
        print(f"No plot created: No files matching {csv_filename} found in {path}")
        return 

    data = pd.DataFrame()
    # Handling split challenge runs
    # Concatenate all files into one dataframe and drop duplicates
    for file in files:
        df = pd.read_csv(file)
        data = pd.concat([data, df])
    data = data.drop_duplicates()
    kwargs = {}
    if challenge_cls == FixedError:
        kwargs["target_error"] = data["target_error"].unique()[0]
    if challenge_cls == FixedTrainSize:
        kwargs["n"] = data["n"].unique()[0]
    g = challenge_cls.plot(data, show=False, **kwargs)
    g.figure.savefig(f"{path}/{figure_filename}", bbox_inches='tight')
    print(f"Plot created: {figure_filename} using")
    for file in files:
        print(f"\t- {file}")


### Helper functions

def _get_base_filename(s:str, m:str, challenge_cls:type[Challenge]):
    if challenge_cls == FixedComplexity:
        params = _get_params(s, m, challenge_cls=challenge_cls)
        l = params["l"]
        filebase = f"fc_{s}_{m}_{l=}"

    elif challenge_cls == FixedTrainSize:
        params = _get_params(s, m, challenge_cls=challenge_cls)
        n = params["n"]
        filebase = f"fts_{s}_{m}_{n=}"
    
    elif challenge_cls == FixedError:
        params = _get_params(s, m, challenge_cls=challenge_cls)
        e = params["target_error"]
        ood = params["evaluate"]["ood"]
        filebase = f"fe_{s}_{m}_{e=}"
        if ood:
            filebase = f"fe_ood_{s}_{m}_{e=}"
    else:
        raise ValueError("challenge_cls must be FixedComplexity, FixedTrainSize, or FixedError")

    return filebase

def _find_matching_files(path, filename):
    """
    Find all files in path that match filename.
    For example, if filename is "fc_lds_lr_l=5.csv", this will return all files in path that contain "fc_lds_lr_l=5" and end with ".csv"
    """
    files = os.listdir(path)
    file_base = filename.split(".")[0]+"_" #include underscore to avoid matching "l=5" with "l=50"
    file_ext = filename.split(".")[1]
    matching = [f"{path}/{f}" for f in files if file_base in f and file_ext in f]
    return matching

def _get_runs(L, reps, split_num, total_splits):
    """
    Get the runs for a given split_num and total_splits. 
    Each run is a tuple (rep, l), where rep is the rep number and l is the L value for the system.
    This is used to split the challenge into parallelizable chunks of runs.

    :param L: list of L values
    :param reps: number of reps
    :param split_num: which split to run
    :param total_splits: total number of splits
    :return: list of tuples (rep, l) to run; to be passed to FixedComplexity.evaluate as eval_rep_l argument
    """
    assert 1 <= total_splits <= reps * len(L), "cannot split into more reps x L than there are"
    assert 1 <= split_num <= total_splits, "split_num must be less than total_splits and greater than 0"
    runs = list(itertools.product(range(reps), L))
    k, m = divmod(len(runs), total_splits)
    splits = list(runs[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(total_splits))
    return splits[split_num-1]


