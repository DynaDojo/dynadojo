from .abstractions import Task, Challenge
import pandas as pd
import numpy as np

from dynascale.utils.plotting import plot_target_error, plot_metric

# class TargetError(Task):
#     def __init__(self, N: list[int], L: list[int], E: list[int], T: list[int], control_horizons: int,
#                  challenge_cls: type[Challenge], trials: int, test_size: int, target_loss: float):
#         self._target_loss = target_loss
#         # TODO: Tommy transfer summer code by overloading evaluate if necessary
#         super().__init__(N, L, E, T, control_horizons, challenge_cls, trials, test_size)
#
#     # adaptive trying to fit previous samples and extrapolate to predict the best new value
#
#     # predict -- scipy regression function (# of points before me - 1)
#
#     # initial low = last radius's guess
#
#     # initial high = predicted - low + predicted
#
#     # if that initial high does not work, new high = 2x * (high - low) + high; new low = old high
#     def evaluate(self, model_cls: type[Model], in_dist=True, **kwargs):
#         results = {}
#
#         for l in L:
#             lower_bound = 1
#             upper_bound = 5
#             prediction = 3
#
#             if results:
#                 lower_bound = results[l - 1]
#
#                 # with enough data, fit polynomial function
#                 if len(list(results.keys())) > 1:
#                     p = np.polyfit(list(results.keys()), list(results.values()), deg=len(list(results.values())) - 1)
#                     predict = np.poly1d(p)
#                     prediction = round(predict(l))
#
#                 # with only one data point, assume a very small slope
#                 else:
#                     prediction = results[l - 1] + 1
#
#                 upper_bound = (2 * prediction) - lower_bound
#
#                 # ensure upper_bound works
#                 while fit(N, L, E, T, supepochs, factory_cls, trials, test_size)[0][0] < self._target_err:
#                     if(upper_bound == lower_bound):
#                         upper_bound += 1
#                     else:
#                         temp_upper_bound = round((2 * (upper_bound - lower_bound)) + upper_bound)
#                         lower_bound = upper_bound
#                         upper_bound = temp_upper_bound
#
#                 # perform binary search
#                 lowest_succ = upper_bound
#
#                 while (lower_bound != upper_bound):
#                     mid_bound = int((lower_bound + upper_bound) / 2)
#
#                     y_mid = fit(radiiToTest = [radius], numTrainSamples = midSamples, ruleIndicesToTest=ruleIndicesToTest)[0][0]
#
#                     if y_mid < minError:
#                         if lowSamples == midSamples:
#                             lowSamples = midSamples + 1
#                         else:
#                             lowSamples = midSamples
#                     else:
#                         upper_bound = mid_bound
#                         lowest_succ = mid_bound
#
#                 results[l] = lowest_succ
#
#     def plot(self, data: pd.DataFrame):
#         plot_target_error(data, "latent_dim", "n", target_error=self._target_loss)

class FixedComplexity(Task):
    def __init__(self, N: list[int], l: int, e: int, t: int, max_control_cost_per_dim: int, control_horizons: int,
                 challenge_cls: type[Challenge], reps: int, test_examples: int, test_timesteps: int, challenge_kwargs: dict = None):
        L = [l] * len(N)
        E = [e] * len(N)
        T = [t] * len(N)
        super().__init__(N, L, E, T, max_control_cost_per_dim, control_horizons, challenge_cls, reps, test_examples, test_timesteps, challenge_kwargs=challenge_kwargs)

    @staticmethod
    def plot(data):
        plot_metric(data, x_col="n", y_col="loss")

class FixedTrainSize(Task):
    def __init__(self, n: int, L: list[int], E: list[int], T: list[int], max_control_cost_per_dim: int, control_horizons: int,
                 challenge_cls: type[Challenge], reps: int, test_examples: int, test_timesteps: int, challenge_kwargs: dict = None):
        N = [n] * len(L)
        super().__init__(N, L, E, T, max_control_cost_per_dim, control_horizons, challenge_cls, reps, test_examples, test_timesteps, challenge_kwargs=challenge_kwargs)

    @staticmethod
    def plot(data: pd.DataFrame):
        plot_metric(data, "latent_dim", "loss")