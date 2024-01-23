
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
    "lorenz": {
            "sindy" : {
                    "L" : [3, 5, 7, 9, 11],
                    "trials" : 50,
                    "n" : 1000,
                },
            "lr" : {
                    "L" : [3, 5, 7, 9, 11],
                    "reps" : 50,
                    "n" : 1000,
            }
    },
    "lv_p": { 
        "sindy" : { #done
                "L" : [2, 4, 8, 16, 32],
                "trials" : 50,
                "n" : 1000,
            },
        "lr" : { #done
                "L" : [2, 4, 8, 16, 32],
                "trials" : 50,
                "n" : 1000,
        },
        "lr_cross" : { #done
                "L" : [int(n) for n in np.logspace(1, 3, num=20, endpoint=True)],
                "trials" : 100,
                "n" : 1000,
        }
    },
    "epi_1": { #done, error going down down down with complexity. weird after merging tommy's edits because 0 error for L = 2, 4?
        "lr_test" : { 
                "L" : [2, 4, 8, 16, 32],
                "trials" : 20,
                "t" : 20,
                "test_timesteps" : 20,
                "n" : 1000,
        },
    },
    "nbody": {
        "lr_test" : { #sloowwwwww 
                "L" : [4, 8, 16, 32, 64],
                "trials" : 20,
                "t" : 20,
                "test_timesteps" : 20,
                "n" : 1000,
        },
        
    },
    "heat": { #waiting for fix https://github.com/DynaDojo/dynadojo/issues/23
        "lr_test" : {
                "L" : [4, 9, 16, 25],
                "trials" : 20,
                "t" : 20,
                "test_timesteps" : 20,
                "n" : 1000,
        },
        
    },
    "fbsnn_1": { #waiting for fix.  https://github.com/DynaDojo/dynadojo/issues/22
        "lr_test" : { 
                "L" : [4, 8, 16, 32, 64],
                "trials" : 20,
                "t" : 20,
                "test_timesteps" : 20,
                "n" : 1000,
        },
        
    },
    "fbsnn_2": {  #waiting for fix.  https://github.com/DynaDojo/dynadojo/issues/22
        "lr_test" : {
                "L" : [4, 8, 16, 32, 64],
                "trials" : 20,
                "t" : 20,
                "test_timesteps" : 20,
                "n" : 1000,
        },
        
    },
     "ctln": {
        "lr_test" : { #waiting for fix. seeding error. https://github.com/DynaDojo/dynadojo/issues/21 
                "L" : [4, 8, 16, 32, 64],
                "trials" : 20,
                "t" : 20,
                "test_timesteps" : 20,
                "n" : 1000,
        },
        
    },
    "kura": {
        "lr_test" : { #Times out at L=64 because >2.5 hours
                "L" : [4, 8, 16, 32, 64],
                "trials" : 20,
                "t" : 20,
                "test_timesteps" : 20,
                "n" : 1000,
        },
        "lr" : { #Running
                "L" : [4, 8, 16, 32],
                "trials" : 100,
                "t" : 20,
                "test_timesteps" : 20,
                "n" : 1000,
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