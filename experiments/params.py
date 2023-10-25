"""
This file contains parameters for experiments, including system and model parameters, and challenge parameters.
Also contains functions for getting system, model, and challenge parameters.
"""
from dynadojo.baselines import LinearRegression
from dynadojo.systems.lds import LDSystem
from dynadojo.challenges import  FixedError, FixedComplexity, FixedTrainSize
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
                    "reps" : 10,
                    },
                "lr" : { 

                    }
            }
}


fts_challenge_params_dict = {
    "default" : {   "L" : [5, 10, 20], 
                    "n" : 100,
                    "t" : 50,
                    "reps" : 5,
                    "test_examples" : 50,
                    "test_timesteps" : 50,
                    "E" : None,
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
                    "reps" : 6,
                },
                "lr" : {
                }
    }
}

fe_challenge_params_dict = {
}

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
    assert ("L" in challenge_params or "l" in challenge_params) and "reps" in challenge_params, "must specify L (or l) and reps in challenge parameters"

    return challenge_params

def _get_system(s:str):
    assert s in system_dict, f"s must be one of {system_dict.keys()}"
    return system_dict[s]

def _get_model(m:str):
    assert m in model_dict, f"m must be one of {model_dict.keys()}"
    return model_dict[m]