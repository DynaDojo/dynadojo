import sys
import inspect
import doctest

from ..abstractions import AbstractSystem
from ..challenges import FixedError, FixedComplexity, FixedTrainSize
from ..baselines import LinearRegression

import importlib


def get_test_system(module):
    try:
        importlib.import_module(module)
    except Exception as e:
        ImportError(f"Could not import {module}")

    for name, obj in inspect.getmembers(sys.modules[module]):
        if inspect.isclass(obj) and issubclass(obj, AbstractSystem) and obj is not AbstractSystem:
            return obj
    print(f"Couldn't find valid System to test in {module}")


def test_fixed_complexity(N: list[int], l: int, e: int, t: int, max_control_cost_per_dim: int, control_horizons: int,
                          system_cls: type[AbstractSystem], reps: int, test_examples: int, test_timesteps: int,
                          system_kwargs: dict = None, model_kwargs: dict = None, 
                          seed:int|None =None,  
                          eval_reps: list[int] | None = None,
                          eval_L: list[int] | None = None):
    print("Testing FixedComplexity")
    challenge = FixedComplexity(N=N, l=l, e=e, t=t, max_control_cost_per_dim=max_control_cost_per_dim,
                                control_horizons=control_horizons, system_cls=system_cls, reps=reps,
                                test_examples=test_examples, test_timesteps=test_timesteps, 
                                system_kwargs=system_kwargs)
    for in_dist, noisy in ([True, True], [True, False], [False, True], [False, False]):
        print(f"\nTesting {in_dist=} {noisy=}")
        challenge.evaluate(LinearRegression, seed=seed, in_dist=in_dist, noisy=noisy, 
                            eval_reps=eval_reps, 
                            eval_L = eval_L,
                            model_kwargs=model_kwargs)

def test_fixed_training(n: int, L: list[int], t: int, max_control_cost_per_dim: int, control_horizons: int,
                          system_cls: type[AbstractSystem], reps: int, test_examples: int, test_timesteps: int,
                          system_kwargs: dict = None, model_kwargs: dict = None, 
                          seed:int|None =None, 
                          eval_reps: list[int] | None = None,
                          eval_L: list[int] | None = None):
    print("Testing FixedTrainSize")
    challenge = FixedTrainSize(n=2, L=L, E=None, t=t, max_control_cost_per_dim=max_control_cost_per_dim,
                                control_horizons=control_horizons, system_cls=system_cls, reps=reps,
                                test_examples=test_examples, test_timesteps=test_timesteps, 
                                system_kwargs=system_kwargs)
    for in_dist, noisy in ([True, True], [True, False], [False, True], [False, False]):
        print(f"\nTesting {in_dist=} {noisy=}")
        challenge.evaluate(LinearRegression, seed=seed,
                            in_dist=in_dist, noisy=noisy, 
                            eval_reps=eval_reps, 
                            eval_L = eval_L,
                            model_kwargs=model_kwargs)


def test_system(system_module: str,
                n=[1, 10], 
                l=[4, 10],
                e=None, 
                t=2, 
                max_control_cost_per_dim=0, 
                control_horizons=0, 
                test_examples=2, 
                test_timesteps=10,
                seed = None,
                reps = 1,
                eval_reps = None,
                eval_L = None,
                system_kwargs: dict = None):
    system_cls = get_test_system(system_module)
    print(f"System: {system_cls}")
    print(f"Model: {LinearRegression}")
    test_fixed_complexity(N=n, l=l[0], e=e, t=t, max_control_cost_per_dim=max_control_cost_per_dim,
                          control_horizons=control_horizons, test_examples=test_examples,
                          reps=reps, test_timesteps=test_timesteps, system_kwargs=system_kwargs, system_cls=system_cls, 
                          seed=seed, eval_reps=eval_reps, eval_L=eval_L)

    test_fixed_training(n=n[0], L=l, t=t, max_control_cost_per_dim=max_control_cost_per_dim,
                          control_horizons=control_horizons, test_examples=test_examples,
                          reps=reps, test_timesteps=test_timesteps, system_kwargs=system_kwargs, system_cls=system_cls, 
                          seed=seed, eval_reps=eval_reps, eval_L=eval_L)
