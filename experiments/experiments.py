import itertools
import os
import pandas as pd
import numpy as np
from dynadojo.challenges import  FixedError, FixedComplexity, FixedTrainSize
from dynadojo.baselines import LinearRegression
from dynadojo.systems.lds import LDSystem
from dynadojo.abstractions import Challenge 

system_dict = {
    "lds" : LDSystem
}
model_dict = {
    "lr" : LinearRegression
}

fc_challenge_params_dict = {
    "default" : {   "l" : 5, 
                    "N" : [10, 100, 1000],
                    "t" : 50,
                    "reps" : 5,
                    "test_examples" : 50,
                    "test_timesteps" : 50,
                    "e" : None,
                    "max_control_cost_per_dim": 1,
                    "control_horizons": 0,
                    "system_kwargs": None,
                    "evaluate": {
                        "seed": 100,
                        "model_kwargs" : None,
                        "fit_kwargs" : None,
                        "act_kwargs" : None,
                        "num_parallel_cpu" : -1,
                        "noisy": True, 
                        "ood": True,
                    }
                },
    "lds" : {   
                "default" : {  
                    "l" : 5, 
                    "N" : [10, 100, 1000],
                    "t" : 50,
                    "reps" : 5,
                    },
                "lr" : { 

                    }
            }
}


fts_challenge_params_dict = {
}

fe_challenge_params_dict = {
}

def get_max_splits(s="lds", m="lr", challenge_cls:type[Challenge] = FixedComplexity,):
    params = _get_params(s, m, challenge_cls=challenge_cls)
    reps = params["reps"]
    L = params.get("L", params["l"])
    if isinstance(L, int):
        return reps
    return reps * len(L)

def run_fc(
        s ="lds",
        m = "lr",
        output_dir="experiments/outputs", 
        split=(1,1)
        ):
    """
    Run a fixed complexity challenge and save the results to a csv file.
    :param s: system short name, defined in system_dict
    :param m: model short name, defined in model_dict
    :param output_dir: directory to save results
    :param split: tuple (split_num, total_splits) to run a subset of the total number of system runs, as specified by L and reps the challenge parameters
    """

    system = system_dict[s]
    model = model_dict[m]

    # Get challenge parameters
    fc_challenge_params = _get_params(s, m, challenge_cls=FixedComplexity)
    
    # Get evaluate parameters and remove from challenge parameters
    evaluate_params = fc_challenge_params.get("evaluate", {})
    del fc_challenge_params["evaluate"]
    evaluate_params["model_cls"] = model

    # Override system class
    fc_challenge_params["system_cls"] = system

    # Get L and reps
    assert "l" in fc_challenge_params and "reps" in fc_challenge_params, "must specify l and reps in challenge parameters"
    l = fc_challenge_params["l"] 
    reps = fc_challenge_params['reps']
    
    # Split reps into total_splits and run split_num
    if split:
        assert isinstance(split, tuple), "split must be a tuple, (split_num, total_splits)"
        split_num, total_splits = split
        runs = _get_runs([l], reps, split_num, total_splits) # list[tuples(rep, l)]
        print(f"Running split {split_num} of {total_splits} with runs {runs}")
    else:
        runs = None

    challenge = FixedComplexity(
        **fc_challenge_params,
    )

    
    data = challenge.evaluate(
        **evaluate_params, 
        id=f"fc_{s}_{m}_{l=}", 
        # Which reps and l pairings to evaluate. If None, evaluate all reps on all L. 
        # This is calculated by the split argument!
        rep_l_filter = runs
    )
    
    # Save data to csv, specifying split if necessary
    path = f"{output_dir}/fc/{s}"
    if not os.path.exists(path):
        os.makedirs(path)
    filename = f"fc_{s}_{m}_{l=}"
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
        params = _get_params(s, m, challenge_cls=challenge_cls)
        l = params["l"]
        path = f"{output_dir}/fc/{s}"
        filebase = f"fc_{s}_{m}_{l=}"
        csv_filename = filebase + ".csv"
        figure_filename = filebase + ".pdf"
    
    files = _find_matching_files(path, csv_filename)
    data = pd.DataFrame()
    # Handling split challenge runs
    # Concatenate all files into one dataframe and drop duplicates
    for file in files:
        df = pd.read_csv(file)
        data = pd.concat([data, df])
    data = data.drop_duplicates()
    g = challenge_cls.plot(data, show=False)
    g.figure.savefig(f"{path}/{figure_filename}", bbox_inches='tight')



### Helper functions

def _find_matching_files(path, filename):
    """
    Find all files in path that match filename.
    For example, if filename is "fc_lds_lr_l=5.csv", this will return all files in path that contain "fc_lds_lr_l=5" and end with ".csv"
    """
    files = os.listdir(path)
    file_base = filename.split(".")[0]
    file_ext = filename.split(".")[1]
    matching = [f"{path}/{f}" for f in files if file_base in f and file_ext in f]
    return matching

def _get_params(s, m, challenge_cls: type[Challenge]=FixedComplexity):
    """
    Get challenge parameters for a given system, model, and challenge class, overriding defaults with system and model specific parameters.

    :param s: system short name, defined in system_dict
    :param m: model short name, defined in model_dict
    :param challenge_cls: challenge class, one of Challenge.__subclasses__()
    """
    assert s in system_dict, f"s must be one of {system_dict.keys()}"
    assert m in model_dict, f"m must be one of {model_dict.keys()}"
    if challenge_cls == FixedComplexity:
        challenge_params_dict = fc_challenge_params_dict
    elif challenge_cls == FixedTrainSize:
        challenge_params_dict = fts_challenge_params_dict
    elif challenge_cls == FixedError:
        challenge_params_dict = fe_challenge_params_dict
    else:
        raise ValueError(f"challenge_cls must be one of {Challenge.__subclasses__()}")\
    
    # Get challenge parameters, starting with defaults and then overriding with s and m specific params
    default_params = challenge_params_dict["default"]
    s_m_params = challenge_params_dict.get(s, {}).get(m, {})
    s_default_params = challenge_params_dict.get(s, {}).get("default", {})
    challenge_params = { **default_params, **s_default_params, **s_m_params }
    return challenge_params

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


