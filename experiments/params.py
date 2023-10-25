"""
This file contains parameters for experiments, including system and model parameters, and challenge parameters.
Also contains functions for getting system, model, and challenge parameters.
"""
import numpy as np
from dynadojo.baselines import LinearRegression
from dynadojo.baselines.dnn import DNN
from dynadojo.systems.lds import LDSystem
from dynadojo.challenges import  FixedError, FixedComplexity, FixedTrainSize
from dynadojo.abstractions import Challenge 

system_dict = {
    "lds" : LDSystem,
    
}
model_dict = {
    "lr" : LinearRegression,
    "dnn" : DNN,
}

fc_challenge_params_dict = {
    "default" : {   "l" : 10, 
                    "N" : [int(n) for n in np.logspace(1, 2, num=10, endpoint=True)],
                    "t" : 50,
                    "reps" : 100,
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
                    "l" : 10,  
                    "N" : [int(n) for n in np.logspace(1, 3, num=10, endpoint=True)],
                    "t" : 50,
                },
                "lr" : {
                    "t" : 50,
                    "N" : [int(n) for n in np.logspace(1, 3, num=20, endpoint=True)],
                },
                "lr_5" : { "l" : 5 },
                "lr_10" : { "l" : 10 },
                "lr_20" : { "l" : 20 },
                "lr_30" : { "l" : 30 },
                "lr_50" : { "l" : 50 },
                "lr_100" : { "l" : 100 },
                "dnn" : {
                    "N" : [int(n) for n in np.logspace(1, 4, num=20, endpoint=True)]
                },
                "dnn_5" : { "l" : 5 },
                "dnn_10" : { "l" : 10 },
                "dnn_20" : { "l" : 20 },
                "dnn_30" : { "l" : 30 },
                "dnn_50" : { "l" : 50 },
                "dnn_100" : { "l" : 100 },
            }
}


fts_challenge_params_dict = {
    "default" : {   "L" : [5, 10, 20, 30, 50, 100], 
                    "n" : 100,
                    "t" : 20,
                    "reps" : 100,
                    "test_examples" : 50,
                    "test_timesteps" : 50,
                    "E" : None,
                    "max_control_cost_per_dim": 1,
                    "control_horizons": 0,
                    "system_kwargs": None,
                    "evaluate": {
                        "seed": 1027,
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
                    "L" : [int(n) for n in np.logspace(1, 3, num=10, endpoint=True)],
                    "t" : 50,
                },
                "lr" : {
                    "L" : [int(n) for n in np.logspace(1, 3, num=20, endpoint=True)],
                    "n" : 100,
                },
                "dnn": {
                    "n": 10000,
                }
    }
}

fe_challenge_params_dict = {
    "default" : {   "L" : [5, 10, 20, 30, 50, 100],
                    "n_starts" :  [1000]*6, #same length as L 
                    "t" : 20,
                    "reps" : 100,
                    "target_error": 1e-5,
                    "E" : None, #same length as L
                    "test_examples" : 50,
                    "test_timesteps" : 50,
                    "max_control_cost_per_dim": 1,
                    "control_horizons": 0,
                    "n_precision" : 5,
                    "n_window" :  5,
                    "n_max" : 10000,
                    "system_kwargs": None,
                    "evaluate": {
                        "seed": 1027,
                        "model_kwargs" : None,
                        "fit_kwargs" : None,
                        "act_kwargs" : None,
                        "num_parallel_cpu" : -1,
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
                "lr" : { 
                    "L" : [int(n) for n in np.logspace(1, 3, num=20, endpoint=True)],
                    "n_starts" :  [1000]*20, #same length as L
                    "target_error": 1e-5,
                },
                "lr_ood" : {
                    "evaluate": {
                        "ood": True,
                    }
                },
                "dnn" : {
                    "target_error": 5e0,
                },
                "dnn_large" : {
                    "target_error": 1e1,
                },
    }
}

def _get_params(s, m, challenge_cls: type[Challenge]=FixedComplexity):
    """
    Get challenge parameters for a given system, model, and challenge class, overriding defaults with system and model specific parameters.

    :param s: system short name, defined in system_dict
    :param m: model short name, defined in model_dict
    :param challenge_cls: challenge class, one of Challenge.__subclasses__()
    """
    assert s in system_dict, f"s must be one of {system_dict.keys()}"
    assert m.split("_")[0] in model_dict, f"m must be one of {model_dict.keys()}"
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
    default_eval_params = default_params["evaluate"]

    s_m_base_params = challenge_params_dict.get(s, {}).get(m.split("_")[0], {})
    s_m_base_eval_params =  s_m_base_params.get("evaluate", {})

    s_m_params = challenge_params_dict.get(s, {}).get(m, {})
    s_m_eval_params = s_m_params.get("evaluate", {})

    s_default_params = challenge_params_dict.get(s, {}).get("default", {})
    s_default_eval_params = s_default_params.get("evaluate", {})

    challenge_params = { **default_params, **s_default_params, **s_m_base_params, **s_m_params }
    eval_params = { **default_eval_params, **s_default_eval_params, **s_m_base_eval_params,  **s_m_eval_params }
    challenge_params["evaluate"] = eval_params
    assert ("L" in challenge_params or "l" in challenge_params) and "reps" in challenge_params, "must specify L (or l) and reps in challenge parameters"

    return challenge_params

def _get_system(s:str):
    assert s in system_dict, f"s must be one of {system_dict.keys()}"
    return system_dict[s]

def _get_model(m:str):
    assert m.split("_")[0] in model_dict, f"m must be one of {model_dict.keys()}"
    return model_dict.get(m, model_dict[m.split("_")[0]])