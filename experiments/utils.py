"""
This file contains parameters for experiments, including system and algo parameters, and challenge parameters.
Also contains functions for getting system, algo, and challenge parameters.
"""
import copy
import importlib
import inspect
import json
import os
import sys
import numpy as np
from dynadojo.challenges import  FixedError, FixedComplexity, FixedTrainSize, ScalingChallenge

from .params import fc_challenge_params_dict, fts_challenge_params_dict, fe_challenge_params_dict

system_dict = {
    "lds" : ("dynadojo.systems.lds", "LDSystem"),
    "lorenz": ("dynadojo.systems.lorenz", "LorenzSystem"),
    
}
algo_dict = {
    "lr" : ("dynadojo.baselines.lr", "LinearRegression"),
    "dnn" : ("dynadojo.baselines.dnn", "DNN"),
    "sindy": ("dynadojo.baselines.sindy", "SINDy"),
}
challenge_dicts = {
    "fc" : (FixedComplexity, fc_challenge_params_dict),
    "fts" : (FixedTrainSize, fts_challenge_params_dict),
    "fe" : (FixedError, fe_challenge_params_dict),
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

    system = _get_system(s)
    algo = _get_algo(a)

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
    default_params = challenge_params_dict["default"].copy()
    default_eval_params = default_params["evaluate"]
    default_params.pop("evaluate", None)

    # Get system default params
    s_default_params = challenge_params_dict.get(s, {}).get("default", {}).copy()
    s_default_eval_params = s_default_params.get("evaluate", {})
    s_default_params.pop("evaluate", None)

    # Get system and algo default params
    s_a_base_params = challenge_params_dict.get(s, {}).get(a.split("_")[0], {}).copy()
    s_a_base_eval_params =  s_a_base_params.get("evaluate", {})
    s_a_base_params.pop("evaluate", None)

    # Get system and algo_with_suffix specific params
    s_a_params = challenge_params_dict.get(s, {}).get(a, {}).copy()
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
        ood = eval_params.get("ood", False)
        folder_name = f"fe_{s}_{a}_{e=}"
        if ood:
            folder_name = f"fe{s}_{a}_ood-{e=}"
        folder_path = f"fe/{s}/{folder_name}"

    experiment_params["experiment_name"] = folder_name
    experiment_params["folder_path"] = folder_path
    experiment_params["total_jobs"] = challenge_cls(**challenge_params).get_num_jobs(trials = challenge_params["trials"])

    return experiment_params

def _get_system(s:str):
    assert s in system_dict, f"s must be one of {system_dict.keys()}"
    serialized_system = {
        "type": "serialized_class",
        "module_name": system_dict[s][0],
        "class_name": system_dict[s][1]
    }
    return serialized_system

def _get_algo(a:str):
    assert a.split("_")[0] in algo_dict, f"m must be one of {algo_dict.keys()}"
    serialized_algo = {
        "type": "serialized_class",
        "module_name": algo_dict[a.split("_")[0]][0],
        "class_name": algo_dict[a.split("_")[0]][1]
    }
    return serialized_algo

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