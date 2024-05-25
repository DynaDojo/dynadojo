
import numpy as np


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
    "lorenz" : {
        "default" : {
            "l" : 9,  #MUST BE ODD > 3
            "N" : [int(n) for n in np.logspace(1, 3, num=10, endpoint=True)],
            "t" : 50,
        },
        "gru_long_l19": {
            "l": 19,
            "trials": 50,
            "evaluate": {
                        "algo_kwargs": {
                            "num_layers": 5,
                            "hidden_size": 128,
                            "lr": 5e-3
                        },
                        "fit_kwargs": {
                            "epochs": 20000,
                            "early_stopping": True,
                            "patience": 10,
                            "min_delta": 10
                        }
                }
        },
        "dnn": {
            "trials": 50,
            "evaluate": {
                        "fit_kwargs": {
                            "lr": 1e-2,
                            "epochs": 10000,
                            "min_epochs": 1000,
                        }
                }
        },
        "dnn_l5": { "l": 5 },
        "dnn_l9": { "l": 9 },
        "dnn_l19": { "l": 19 },
    
    },
    "lv_p": {
        "default" : {
            "l" : 5, 
            "N" : [int(n) for n in np.logspace(1, 3, num=10, endpoint=True)],
            "t" : 50,
        },
        "dnn": {
            "trials": 50,
            "evaluate": {
                        "fit_kwargs": {
                            "lr": 1e-2,
                            "epochs": 10000,
                            "min_epochs": 1000,
                        }
                }
        },
        "dnn_l5": { "l": 5 },
        "dnn_l9": { "l": 9 },
        "dnn_l19": { "l": 19 },
    },
    "kura": {
        "default" : {
            "l" : 5, 
            "N" : [int(n) for n in np.logspace(1, 3, num=10, endpoint=True)],
            "t" : 100,
            "test_timesteps" : 100,
            "system_kwargs": {
                "COUPLE_range": (0, 1)
            }
        },
        "dnn": {
            "trials": 50,
            "evaluate": {
                        "fit_kwargs": {
                            "lr": 1e-2,
                            "epochs": 10000,
                            "min_epochs": 1000,
                        }
                }
        },
        "dnn_l5": { "l": 5 },
        "dnn_l9": { "l": 9 },
        "dnn_l19": { "l": 19 },
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
    "lorenz": {
            "default" : {
                    "L" : [3, 5, 7, 9, 11],
                    "trials" : 50,
                    "n" : 1000,
                },
            "dnn" : {
                    "L" : [3, 5, 7, 9, 11],
                    "trials" : 50,
                    "n" : 1000,
                    "evaluate": {
                         "fit_kwargs": {
                            "lr": 1e-2,
                            "epochs": 10000,
                            "min_epochs": 1000,
                        }
                    }
            }
    },
    "lv_p": { 
        "dnn" : {
                "L" : [2, 4, 8, 16, 32],
                "trials" : 50,
                "n" : 1000,
                "evaluate": {
                        "fit_kwargs": {
                            "lr": 1e-2,
                            "epochs": 10000,
                            "min_epochs": 1000,
                        }
                }
            }
    },
    "kura": {
        "dnn" : { 
                "L" : [4, 8, 16, 32],
                "trials" : 50,
                "t" : 20,
                "test_timesteps" : 20,
                "n" : 1000,
                "evaluate": {
                        "fit_kwargs": {
                            "lr": 1e-2,
                            "epochs": 10000,
                            "min_epochs": 1000,
                        }
                }
        },
    },

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