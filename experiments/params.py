"""
This file contains parameters for experiments, including system and algo parameters, and challenge parameters.
Also contains functions for getting system, algo, and challenge parameters.
"""
import importlib
import inspect
import json
import os
import sys
import numpy as np
from dynadojo.baselines.lr import LinearRegression
from dynadojo.baselines.dnn import DNN
from dynadojo.baselines.sindy import SINDy

from dynadojo.systems.lds import LDSystem
from dynadojo.systems.lorenz import LorenzSystem
from dynadojo.challenges import  FixedError, FixedComplexity, FixedTrainSize, ScalingChallenge

system_dict = {
    "lds" : LDSystem,
    "lorenz": LorenzSystem,
    
}
algo_dict = {
    "lr" : LinearRegression,
    "dnn" : DNN,
    "sindy": SINDy
}

fc_challenge_params_dict = {
    "default" : {   "l" : 10, 
                    "N" : [int(n) for n in np.logspace(1, 2, num=10, endpoint=True)],
                    "t" : 50,
                    "trials" : 100,
                    "test_examples" : 50,
                    "test_timesteps" : 50,
                    "e" : None,
                    "max_control_cost_per_dim": 1,
                    "control_horizons": 0,
                    "system_kwargs": None,
                    "evaluate": {
                        "seed": 100,
                        "algo_kwargs" : None,
                        "fit_kwargs" : None,
                        "act_kwargs" : None,
                        "num_parallel_cpu" : 0,
                        "noisy": True, 
                        "ood": True,
                    }
                },
    "lds" : {   
                "default" : {
                    "l" : 10,  
                    "N" : [int(n) for n in np.logspace(1, 3, num=10, endpoint=True)],
                    "t" : 50,
                },
                "lr" : {
                    "t" : 50,
                    "N" : [int(n) for n in np.logspace(1, 3, num=20, endpoint=True)],
                },
                "lr_5" : { "l" : 5 }, #DONE
                "lr_10" : { "l" : 10 }, 
                "lr_20" : { "l" : 20 },
                "lr_30" : { "l" : 30 },
                "lr_50" : { "l" : 50 },
                "lr_100" : { "l" : 100 },
                # DNN
                "dnn" : {
                    "N" : [int(n) for n in np.logspace(1, 4, num=20, endpoint=True)]
                },
                "dnn_5" : { "l" : 5 },
                "dnn_10" : { "l" : 10 },
                "dnn_20" : { "l" : 20 },
                "dnn_30" : { "l" : 30 },
                "dnn_50" : { "l" : 50 },
                "dnn_100" : { "l" : 100 },
                # SINDY
                "sindy_3" : { "l" : 3 },
                "sindy_5" : { "l" : 5 },
                "sindy_10" : { "l" : 10 },
                "sindy_20" : { "l" : 20 },
                "sindy_30" : { "l" : 30 },
                "sindy_50" : { "l" : 50 },
                "sindy_100" : { "l" : 100 },
            }
    ,
    "lorenz" : {
        "default" : {
            "l" : 9,  #MUST BE ODD > 3
            "N" : [int(n) for n in np.logspace(1, 3, num=10, endpoint=True)],
            "t" : 50,
        },
        "lr" : { #FAIL
            "N" : [int(n) for n in np.logspace(1, 3, num=20, endpoint=True)],
            "trials":5
        },
        "sindy" : { #FAIL
            "t": 50,
            "test_timesteps" : 50,
            "N" : [int(n) for n in np.logspace(1, 3, num=15, endpoint=True)],
        },
        "lr_3" : { #SUCCESS
            "t": 50,
            "test_timesteps" : 50,
            "l" : 3,
            "N" : [int(n) for n in np.logspace(1, 3, num=10, endpoint=True)],
        },
        "sindy_3" : { #SUCCESS
            "l" : 3,
            "t": 50,
            "test_timesteps" : 50,
            "N" : [int(n) for n in np.logspace(1, 3, num=10, endpoint=True)],
        }
        , "sindy_5" : { #FAIL
            "l" : 5,
        }
    }
}

fts_challenge_params_dict = {
    "default" : {   "L" : [5, 10, 20, 30, 50, 100], 
                    "n" : 100,
                    "t" : 20,
                    "trials" : 100,
                    "test_examples" : 50,
                    "test_timesteps" : 50,
                    "E" : None,
                    "max_control_cost_per_dim": 1,
                    "control_horizons": 0,
                    "system_kwargs": None,
                    "evaluate": {
                        "seed": 1027,
                        "algo_kwargs" : None,
                        "fit_kwargs" : None,
                        "act_kwargs" : None,
                        "num_parallel_cpu" : 0,
                        "noisy": True, 
                        "ood": True,
                    }
                },
    "lds" : { #DONE
                "default" : {
                    "L" : [int(n) for n in np.logspace(1, 3, num=10, endpoint=True)],
                    "t" : 50,
                    "n" : 1000,
                },
                "lr" : {
                    "L" : [int(n) for n in np.logspace(1, 3, num=20, endpoint=True)],
                    "n" : 100,
                },
                "dnn": {
                    "L" : [int(n) for n in np.logspace(1, 2, num=10, endpoint=True)],
                    "n": 10000,
                },
    },
    "lorenz" : {
            "default" : {
                "n": 1000,
            }
    }
}

fe_challenge_params_dict = {
    "default" : {   "L" : [5, 10, 20, 30, 50, 100],
                    "n_starts" :  [1000]*6, #same length as L 
                    "t" : 50,
                    "trials" : 100,
                    "target_error": 1e-5,
                    "E" : None, #same length as L
                    "test_examples" : 500,
                    "test_timesteps" : 50,
                    "max_control_cost_per_dim": 1,
                    "control_horizons": 0,
                    "n_precision" : .05,
                    "n_window" :  5,
                    "n_max" : 10000,
                    "n_window_density": 1.0,
                    "system_kwargs": None,
                    "evaluate": {
                        "seed": 1027,
                        "algo_kwargs" : None,
                        "fit_kwargs" : None,
                        "act_kwargs" : None,
                        "num_parallel_cpu" : 0,
                        "noisy": True, 
                        "ood": False,
                    }
                },
    "lds" : { 
                "default" : {
                    "L" : [int(n) for n in np.logspace(1, 3, num=10, endpoint=True)],
                    "n_starts" : [int(n) for n in np.logspace(1, 3, num=10, endpoint=True)], #same length as L
                    "t" : 50,
                },
                "lr" : { #DONE w/ mem fails
                    "L" : [int(n) for n in np.logspace(1, 3, num=20, endpoint=True)],
                    "n_starts" :  [1000]*20, #same length as L
                    "target_error": 1e-5,
                    "n_window": 5,
                    "n_max" : 10000,
                    # "n_precision": 5 #not a percentage....run previously before changing to percentage 
                },
                "lr_ood" : { #DONE w/ mem fails
                    "evaluate": {
                        "ood": True,
                    }
                },
                "dnn" : { 
                    "target_error": 5e0,
                    "n_max" : 20000,
                },
                "dnn_100" : { #FAILED, too many np.infs #Search, Precision as a number of samples not percentage
                    "L" : [int(n) for n in np.logspace(1, 1.7, num=10, endpoint=True)],
                    "n_starts" :  [int(n) for n in np.logspace(1, 4, num=10, endpoint=True)],
                    "target_error": 5e0,
                    "n_window": 5,
                    # "n_precision": 5 #not a percentage....run previously before changing to percentage  
                },
                "dnn_100_window" : { #FAILED #Search
                    "L" : [int(n) for n in np.logspace(1, 1.7, num=10, endpoint=True)],
                    "n_starts" :  [int(n) for n in np.logspace(2, 4, num=10, endpoint=True)],
                    "target_error": 5e0,
                    "n_window": 10,
                    "n_precision": .05,
                    "n_window_density": 0.5,
                    "n_min": 3,
                },
                "dnn_simple_2" : { #PROMISING BUT MEMORY FAIL #Search Simple #TODO: rename to dnn_simple
                    "L" : [int(n) for n in np.logspace(1, 2, num=10, endpoint=True)],
                    "n_starts" :  [int(n) for n in np.logspace(4, 4.7, num=10, endpoint=True)],
                    "target_error": 1e0,
                    "n_window": 5,
                    "n_precision": .05,
                    "n_window_density": 0.5,
                    "n_min": 3,
                    "n_max" : 1e5,
                    "trials": 100,
                },
                "dnn_simple_q" : { #PROMISING BUT NMAX TOO LOW #Search Simple #TODO: rename to dnn_simple
                    "L" : [int(n) for n in np.logspace(1, 2, num=10, endpoint=True)],
                    "n_starts" :  [int(n) for n in np.logspace(3, 4, num=10, endpoint=True)],
                    "target_error": 1e0,
                    "n_window": 5,
                    "n_precision": .05,
                    "n_window_density": 0.6,
                    "n_min": 3,
                    "n_max" : 1e4,
                    "trials": 100,
                },
                "dnn_simple_q2" : { #??? Running 4949151 #Search Simple #TODO: rename to dnn_simple
                    "L" : [int(n) for n in np.logspace(1, 2, num=10, endpoint=True)],
                    "n_starts" :  [int(n) for n in np.logspace(3, 4, num=10, endpoint=True)],
                    "target_error": 1e0,
                    "n_window": 5,
                    "n_precision": .05,
                    "n_window_density": 0.6,
                    "n_min": 3,
                    "n_max" : 1e5,
                    "trials": 100,
                },
                "dnn_test" : {
                    "L" : [int(n) for n in np.logspace(1, 1.7, num=10, endpoint=True)],
                    "n_starts" :  [int(n) for n in np.logspace(2, 4, num=10, endpoint=True)],
                    "target_error": 1e0,
                    "n_window": 10,
                    "n_precision": .2,
                    "n_window_density": 0.25,
                    "n_min": 3,
                    "trials":3,
                }
    }
}

def _get_params(s, a, challenge_cls: type[ScalingChallenge]=FixedComplexity):
    """
    Get challenge parameters for a given system, algo, and challenge class, overriding defaults with system and algo specific parameters.

    :param s: system short name, defined in system_dict
    :param m: algo short name, defined in algo_dict
    :param challenge_cls: challenge class, one of ScalingChallenge.__subclasses__()
    """
    assert s in system_dict, f"s must be one of {system_dict.keys()}"
    assert a.split("_")[0] in algo_dict, f"m must be one of {algo_dict.keys()}"

    system = system_dict[s]
    algo = algo_dict[a.split("_")[0]]

    if challenge_cls == FixedComplexity:
        challenge_params_dict = fc_challenge_params_dict
    elif challenge_cls == FixedTrainSize:
        challenge_params_dict = fts_challenge_params_dict
    elif challenge_cls == FixedError:
        challenge_params_dict = fe_challenge_params_dict
    else:
        raise ValueError(f"challenge_cls must be one of {ScalingChallenge.__subclasses__()}")\
    
    # Get challenge parameters, starting with defaults and then overriding with s and m specific params
    # Get default params
    default_params = challenge_params_dict["default"]
    default_eval_params = default_params["evaluate"]
    default_params.pop("evaluate", None)

    # Get system default params
    s_default_params = challenge_params_dict.get(s, {}).get("default", {})
    s_default_eval_params = s_default_params.get("evaluate", {})
    s_default_params.pop("evaluate", None)

    # Get system and algo default params
    s_a_base_params = challenge_params_dict.get(s, {}).get(a.split("_")[0], {})
    s_a_base_eval_params =  s_a_base_params.get("evaluate", {})
    s_a_base_params.pop("evaluate", None)

    # Get system and algo_with_suffix specific params
    s_a_params = challenge_params_dict.get(s, {}).get(a, {})
    s_a_eval_params = s_a_params.get("evaluate", {})
    s_a_params.pop("evaluate", None)

    # Combine all params
    challenge_params = { **default_params, **s_default_params, **s_a_base_params, **s_a_params }
    assert ("L" in challenge_params or "l" in challenge_params) and "trials" in challenge_params, "must specify L (or l) and trials in challenge parameters"
    eval_params = { **default_eval_params, **s_default_eval_params, **s_a_base_eval_params,  **s_a_eval_params }
    
    challenge_params["system_cls"] = system
    eval_params["algo_cls"] = algo
    experiment_params = {
        "challenge": challenge_params,
        "evaluate": eval_params
    }
    experiment_params["challenge_cls"] = challenge_cls

    if challenge_cls == FixedComplexity:
        l = challenge_params["l"]
        folder_name = f"fc_{s}_{a}_{l=}"
        folder_path = f"fc/{s}/{folder_name}"
    elif challenge_cls == FixedTrainSize:
        n = challenge_params["n"]
        folder_name = f"fts_{s}_{a}_{n=}"
        folder_path = f"fts/{s}/{folder_name}"
    elif challenge_cls == FixedError:
        e = challenge_params["target_error"]
        ood = challenge_params["evaluate"]["ood"]
        folder_name = f"fe_{s}_{a}_{e=}"
        if ood:
            folder_name = f"fe{s}_{a}_ood-{e=}"
        folder_path = f"fe/{s}/{folder_name}"

    experiment_params["folder_path"] = folder_path
    experiment_params["total_jobs"] = challenge_cls(**challenge_params).get_num_jobs(trials = challenge_params["trials"])

    return experiment_params

def _get_system(s:str):
    assert s in system_dict, f"s must be one of {system_dict.keys()}"
    return system_dict[s]

def _get_algo(a:str):
    assert a.split("_")[0] in algo_dict, f"m must be one of {algo_dict.keys()}"
    return algo_dict.get(a, algo_dict[a.split("_")[0]])

def _deserialize_class(serialized_class):   
    my_module = serialized_class["module_name"]
    my_class = serialized_class["class_name"]
    try:
        cls = getattr(importlib.import_module(my_module), my_class) 
    except Exception as e:
        ImportError(f"Could not find {my_class} in {my_module}")
    return cls

def _serialize_class(my_class):
    return {
        "type": "serialized_class",
        "module_name": my_class.__module__,
        "class_name": my_class.__name__
    }

def serialize_params(params):
    """
    Serialize a deep dictionary into JSON writable format. 
    Class objects within the dictionary are replaced with a specific dictionary notation.

    Parameters:
    params (dict): The dictionary to serialize.

    Returns:
    str: A JSON string representing the serialized dictionary.
    """
    if isinstance(params, dict):
        serialized_dict = {}
        for key, value in params.items():
            serialized_dict[key] = serialize_params(value)  # Recursively serialize
        return serialized_dict
    elif isinstance(params, list):
        return [serialize_params(item) for item in params]
    elif inspect.isclass(params):  # Check if the obj is a class object
        return _serialize_class(params)
    else:
        return params
    
def deserialize_params(serialized_params):
    """
    Deserialize a dictionary from its serialized format. Specific sub-dictionaries are replaced with class objects.

    Parameters:
    serialized_params (dict): The dictionary to deserialize.

    Returns:
    dict: The deserialized dictionary with class objects.
    """
    if isinstance(serialized_params, dict):
        if serialized_params.get("type", None) == "serialized_class":
            return _deserialize_class(serialized_params)
        else:
            return {key: deserialize_params(value) for key, value in serialized_params.items()}
    elif isinstance(serialized_params, list):
        return [deserialize_params(item) for item in serialized_params]
    else:
        return serialized_params
    
def save_to_json(obj, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'w') as file:
        json.dump(serialize_params(obj), file, indent=4)

def load_from_json(file_path: str):
    with open(file_path, 'r') as file:
        return deserialize_params(json.load(file))