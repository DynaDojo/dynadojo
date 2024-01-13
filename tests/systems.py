import importlib
import inspect
import sys

from ..abstractions import AbstractSystem
from ..baselines import LinearRegression
from ..challenges import FixedError, FixedComplexity, FixedTrainSize


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
                          system_cls: type[AbstractSystem], trials: int, test_examples: int, test_timesteps: int,
                          system_kwargs: dict = None, model_kwargs: dict = None, 
                          seed:int|None =None,  
                          trials_filter: list[int] | None = None,
                          L_filter: list[int] | None = None):
    print(f"\n----Testing FixedComplexity----")
    challenge = FixedComplexity(N=N, l=l, e=e, t=t, max_control_cost_per_dim=max_control_cost_per_dim,
                                control_horizons=control_horizons, system_cls=system_cls, trials=trials,
                                test_examples=test_examples, test_timesteps=test_timesteps, 
                                system_kwargs=system_kwargs)
    ood = True
    noisy = True
    # for  noisy in [True, False]:
    print(f"\nTesting {ood=} {noisy=}")
    df = challenge.evaluate(LinearRegression, seed=seed, ood=ood, noisy=noisy,
                            trials_filter=trials_filter,
                            L_filter = L_filter,
                            algo_kwargs=model_kwargs)
    return df

def test_fixed_training(n: int, L: list[int], t: int, max_control_cost_per_dim: int, control_horizons: int,
                          system_cls: type[AbstractSystem], trials: int, test_examples: int, test_timesteps: int,
                          system_kwargs: dict = None, model_kwargs: dict = None, 
                          seed:int|None =None, 
                          trials_filter: list[int] | None = None,
                          L_filter: list[int] | None = None):
    print("\n----Testing FixedTrainSize----")
    challenge = FixedTrainSize(n=2, L=L, E=None, t=t, max_control_cost_per_dim=max_control_cost_per_dim,
                                control_horizons=control_horizons, system_cls=system_cls, trials=trials,
                                test_examples=test_examples, test_timesteps=test_timesteps, 
                                system_kwargs=system_kwargs)
    # for in_dist, noisy in ([True, True]): #([True, True], [True, False], [False, True], [False, False]):
    ood = True
    noisy = True
    print(f"\nTesting {ood=} {noisy=}")
    df = challenge.evaluate(LinearRegression, seed=seed,
                            ood=ood, noisy=noisy,
                            trials_filter=trials_filter,
                            L_filter = L_filter,
                            algo_kwargs=model_kwargs)
    return df

def test_fixed_error(   target_error: float, L: list[int], t: int, max_control_cost_per_dim: int, control_horizons: int,
                          system_cls: type[AbstractSystem], trials: int, test_examples: int, test_timesteps: int,
                          system_kwargs: dict = None, model_kwargs: dict = None, 
                          n_starts: list[int] = None,
                          n_window: int = 1,
                          n_precision: int = 5,
                          seed:int|None =None, 
                          trials_filter: list[int] | None = None,
                          L_filter: list[int] | None = None):
    print(f"\n----Testing FixedError---- {target_error=}")
    challenge = FixedError( L=L, t=t, target_error=target_error, 
                                n_window=n_window, n_starts=n_starts, n_precision=n_precision,
                                max_control_cost_per_dim=max_control_cost_per_dim,
                                control_horizons=control_horizons, system_cls=system_cls, trials=trials,
                                test_examples=test_examples, test_timesteps=test_timesteps, 
                                system_kwargs=system_kwargs)
    # for ood, noisy in ([True, True]): #([True, True], [True, False], [False, True], [False, False]):
    ood = True
    noisy = True
    print(f"\nTesting {ood=} {noisy=}")
    df = challenge.evaluate(LinearRegression, seed=seed,
                            ood=ood, noisy=noisy,
                            trials_filter=trials_filter,
                            L_filter = L_filter,
                            algo_kwargs=model_kwargs)
    return df


def test_system(system_module: str,
                n=[1, 10, 20], 
                L=[5, 10, 20],
                n_starts=[10, 10, 10],
                e=None, 
                t=10, 
                max_control_cost_per_dim=0, 
                control_horizons=0, 
                test_examples=2, 
                test_timesteps=10,
                seed = None,
                trials = 1,
                trials_filter = None,
                L_filter = None,
                system_kwargs: dict = None,
                model_kwargs: dict = None,
                test_ids : list[int] = [0, 1, 2],
                plot: bool = False):
    system_cls = get_test_system(system_module)
    print(f"System: {system_cls}")
    print(f"Model: {LinearRegression}")
    if 0 in test_ids:
        data = test_fixed_complexity(N=n, l=L[0], e=e, t=t, max_control_cost_per_dim=max_control_cost_per_dim,
                            control_horizons=control_horizons, test_examples=test_examples, model_kwargs=model_kwargs,
                            trials=trials, test_timesteps=test_timesteps, system_kwargs=system_kwargs, system_cls=system_cls, 
                            seed=seed, trials_filter=trials_filter, L_filter=L_filter)
        if plot:
            FixedComplexity.plot(data, latent_dim=L[0])
    if 1 in test_ids:
        data = test_fixed_training(n=n[0], L=L, t=t, max_control_cost_per_dim=max_control_cost_per_dim,
                          control_horizons=control_horizons, test_examples=test_examples, model_kwargs=model_kwargs,
                          trials=trials, test_timesteps=test_timesteps, system_kwargs=system_kwargs, system_cls=system_cls, 
                          seed=seed, trials_filter=trials_filter, L_filter=L_filter)
        if plot:
            FixedTrainSize.plot(data, n=n[0])   

    if 2 in test_ids:
        target_error = 5
        data = test_fixed_error(target_error=target_error, L=L, t=t,
                    n_starts = n_starts,
                    n_window = 2,
                    n_precision = 0,
                    max_control_cost_per_dim=max_control_cost_per_dim,
                    control_horizons=control_horizons, test_examples=test_examples,
                    trials=trials, test_timesteps=test_timesteps, system_kwargs=system_kwargs, system_cls=system_cls, 
                    model_kwargs=model_kwargs,
                    seed=seed, trials_filter=trials_filter, L_filter=L_filter)
        if plot:
            FixedError.plot(data, target_error=target_error)
