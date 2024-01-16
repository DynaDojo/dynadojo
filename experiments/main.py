"""
Functions for running experiments for the paper. See __main__.py for usage.
Parameters are specified in experiments/params.py
"""
import itertools
import logging
import os
import pandas as pd
import numpy as np
from dynadojo.challenges import  FixedError, FixedComplexity, FixedTrainSize, ScalingChallenge
from .params import _get_system, _get_algo, _get_params, save_to_json, load_from_json
import json


def save_params(
        s ="lds",
        a = "lr",
        challenge_cls:type[ScalingChallenge] = FixedComplexity,
        output_dir="experiments/outputs"
    ):
    experiment_params = _get_params(s, a, challenge_cls=challenge_cls)
    folder_path = experiment_params["folder_path"]
    save_to_json(experiment_params, os.path.join(output_dir, folder_path, "params.json"))
    return os.path.join(output_dir, folder_path, "params.json")

def get_max_splits(s="lds", m="lr", challenge_cls:type[ScalingChallenge] = FixedComplexity,):
    params = _get_params(s, m, challenge_cls=challenge_cls)
    return params["total_jobs"]

def run_challenge(
        params_file_path,
        output_dir="experiments/outputs",
        split=(1,1),
        num_cpu_parallel=None,
        jobs_filter=None
        ):
    """
    Run an experiment given a params file.

    Parameters
    ----------
    params_file_path : str
        path to params file
    output_dir : str, optional
        base path to save results, by default "experiments/outputs"
    split : tuple, optional
        (split_num, total_splits) to run, by default (1,1) by default (1,1)
    num_cpu_parallel : _type_, optional
        how many cpus to use for parallelization, by default None (no parallelization)
    jobs_filter : list[int], optional
        which jobs to run, by default None (run all jobs)
    """
    # Load params
    experiment_params = load_from_json(params_file_path)
    challenge_params = experiment_params["challenge"]
    evaluate_params = experiment_params["evaluate"]
    challenge_cls = experiment_params["challenge_cls"]

    # Override num_cpu_parallel
    if num_cpu_parallel:
        evaluate_params['num_parallel_cpu'] = num_cpu_parallel

    all_jobs = list(range(experiment_params["total_jobs"]))
    if jobs_filter:
        all_jobs = jobs_filter

    if split:
        assert isinstance(split, tuple), "split must be a tuple, (split_num, total_splits)"
        split_num, total_splits = split
        jobs = _get_jobs(all_jobs, split_num, total_splits) # list[tuples(trial, l)]
        prGreen(f"Running split {split_num} of {total_splits} with jobs {jobs}")
        if jobs == []:
            prGreen(f"Split {split_num} of {total_splits} has no jobs...skipping")
            return
    else:
        jobs = all_jobs

    challenge = challenge_cls(
        **challenge_params,
    )

    # Make output directory 
    folder_path = os.path.join(output_dir, experiment_params["folder_path"])
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    # Save params if not already saved (in the case we rerun an experiment from renamed folder)
    if not os.path.exists(f"{folder_path}/params.json"):
        save_to_json(experiment_params, f"{folder_path}/params.json")
    
    # Get csv file path, specifying split if necessary
    filename = experiment_params["experiment_name"]
    if split is not None:
        filename += f"_{split_num}-of-{total_splits}"
    filename += ".csv"
    file_path = f"{folder_path}/{filename}"

    data = challenge.evaluate(
        **evaluate_params, 
        id=os.path.basename(folder_path), 
        jobs_filter = jobs, 
        csv_output_path = file_path #will save to csv in parallel
    )
    if split:
        prGreen(f"COMPLETED SPLIT -- {split_num=} / {total_splits=}")
    else:
        prGreen(f"COMPLETED ALL JOBS")
    

def make_plots(
        data_path="experiments/outputs/fc/lds/fc_lds_dnn_5_l=5",
        output_dir="experiments/outputs",
        save=True
    ):
    
    experiment_params = load_from_json(f"{data_path}/params.json")
    challenge_cls = experiment_params["challenge_cls"]

    filebase = experiment_params["folder_path"].split('/')[-1]
    csv_filename = filebase + ".csv"
    figure_filename = filebase + ".pdf"

    
    files, data = load_data(data_path)
    kwargs = {}
    kwargs["show_stats"] = True
    g = challenge_cls.plot(data, show=False, **kwargs)
    # linear axes instead of log
    # g.set(xscale="linear", yscale="linear")
    if save:
        g.figure.savefig(f"{output_dir}/{figure_filename}", bbox_inches='tight')
    prGreen(f"Plot created with {len(files)} files in {data_path} and {len(data)} rows: {output_dir}/{figure_filename} ")
    # for file in files:
    #     print(f"\t- {file}")
    return g, data

def load_data(data_path):
    files = _find_all_csv(data_path) 
    if len(files) <= 0:
        # print(f"No plot created: No files matching {csv_filename} found in {data_path}")
        prGreen(f"No CSV files found in {data_path}")
        return [], None

    data = pd.DataFrame()
    # Handling split challenge runs
    # Concatenate all files into one dataframe and drop duplicates
    for file in files:
        try:
            print(file)
            df = pd.read_csv(file)
            prCyan(f"Loaded {len(df)} rows from {file}")
        except:
            continue
        data = pd.concat([data, df])
    data = data.drop_duplicates(subset=df.columns.difference(['id', 'duration']), ignore_index=True) #ignore id/duration columns and index
    return files, data

### Helper functions
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

def _find_all_csv(path):
    """
    Find all files in path that match filename.
    For example, if filename is "fc_lds_lr_l=5.csv", this will return all files in path that contain "fc_lds_lr_l=5" and end with ".csv"
    """
    if not (os.path.exists(path) and os.path.isdir(path)):
        print(f" Path {path} does not exist or is not a directory")
        return []
    files = os.listdir(path)
    matching = [f"{path}/{f}" for f in files if os.path.splitext(f)[1] == ".csv"]
    return matching

def _get_jobs(all_jobs:list[int], split_num:int, total_splits:int):
    """
    Get the jobs for a given split_num and total_splits.
    
    Parameters
    ----------
    all_jobs : list[int]
        list of all jobs to split
    split_num : int
        which split to execute
    total_splits : int
        total number of splits
    """
    total_jobs = len(all_jobs)
    if not (1 <= total_splits <= total_jobs):
        logging.warn(f"Cannot split {total_splits=} into more jobs than there are...setting to {total_jobs=}")
        total_splits = total_jobs
    if not (1 <= split_num <= total_splits):
        logging.warn(f"{split_num=} must be less than {total_splits=} and greater than 0")
        return []
    k, mod = divmod(total_jobs, total_splits) # number of jobs per split
    splits = [all_jobs[i*k+min(i, mod):(i+1)*k+min(i+1, mod)] for i in range(total_splits)]
    return splits[split_num-1]

def prGreen(skk): print("\033[92m{}\033[00m" .format(skk))

def prCyan(skk): print("\033[96m{}\033[00m" .format(skk))

def prPink(skk): print("\033[95m{}\033[00m" .format(skk))