import sys
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from dynascale.utils.plotting import plot_target_loss, plot_metric

from joblib import Parallel, delayed

from dynascale.abstractions import Task, System, Model

class TargetError(Task):
    def __init__(self, L: list[int], t: int, max_control_cost_per_dim: int, control_horizons: int,
                 system_cls: type[System], reps: int, test_examples: int, test_timesteps: int, target_loss: float, max_samples=10000, E: int | list[int] = None):

        if isinstance(E, list):
            assert (len(L) == len(E))
            for idx, l in enumerate(L):
                assert (E[idx] >= l)
        elif isinstance(E, int):
            assert (E >= np.max(L))

        self._target_loss = target_loss
        self.max_samples = max_samples

        self.X = None
        self.rep_id = None
        self.result = None
        self.samples_needed = {}

        super().__init__([1], L, E, t, max_control_cost_per_dim, control_horizons, system_cls, reps, test_examples, test_timesteps)

    def _evaluate_n(self, system, model, test, n, max_control_cost, noisy, fit_kwargs, act_kwargs):
        x = None
        if self.X is None:
            self.X = self._gen_trainset(system, n, self._T, noisy)
            x = self.X

        # if already generated n samples, pull from existing training set; else add more samples
        else:
            if (len(self.X) >= n):
                x = self.X[:int(n), : , :]
            else:
                to_add = self._gen_trainset(
                    system, n - len(self.X), self._T, noisy)
                self.X = np.concatenate((self.X, to_add), axis=0)
                x = self.X

        total_cost = self._fit_model(system, model, x, self._T,
                                     max_control_cost, fit_kwargs, act_kwargs, noisy)


        pred = model.predict_wrapper(test[:, 0], self._test_timesteps)

        loss = system.calc_loss_wrapper(pred, test)

        self._append_result(self.result, self.rep_id, n, system.latent_dim, system.embed_dim, self._T, loss, total_cost)

        return loss

    def _gen_bounds(self, system, model, test, l, max_control_cost, noisy, fit_kwargs, act_kwargs):

        lower = 1
        upper = 3
        predicted = 2

        if self.samples_needed:
            prev_largest_dim = np.max(list(self.samples_needed.keys()))
            lower = self.samples_needed[prev_largest_dim]

            # adaptively predict based on median of samples needed in lower dims
            if len(self.samples_needed.keys()) > 1:
                p = np.polyfit(list(self.samples_needed.keys()), list(self.samples_needed.values()),
                               deg=len(list(self.samples_needed.keys())) - 1)
                predict_func = np.poly1d(p)
                predicted = int(round(predict_func(l)))

            # with only one data dim seens, assume a very small slope
            else:
                predicted = self.samples_needed[prev_largest_dim] + 1

            if predicted < lower:
                predicted = lower+1
            upper = (2 * predicted) - lower

        # if lower meets target loss, decrease to be lower
        lower_error = self._evaluate_n(system, model, test, lower, max_control_cost, noisy, fit_kwargs, act_kwargs)
        while lower_error < self._target_loss:
            if lower == 1:
                break
            lower = lower // 2
            lower_error = self._evaluate_n(system, model, test, lower, max_control_cost, noisy, fit_kwargs, act_kwargs)

        # increase upper until it satisfies target loss or exceeds max_samples
        while upper <= self.max_samples:
            if upper == self.max_samples:
                max_samples_error = self._evaluate_n(
                    system, model, test, self.max_samples, max_control_cost, noisy, fit_kwargs, act_kwargs)
                if max_samples_error < self._target_loss:
                    break
                else:
                    sys.exit(
                        f'At {self.max_samples} samples, error is {max_samples_error}, which is larger than the target fixed error')

            else:
                upper_error = self._evaluate_n(
                    system, model, test, upper, max_control_cost, noisy, fit_kwargs, act_kwargs)
                
                if upper_error < self._target_loss:
                    break
                else:
                    if upper == lower:
                        temp_upper = lower+1
                    else:
                        temp_upper = round((2 * (upper - lower)) + upper)
                    lower = upper
                    upper = min(self.max_samples, temp_upper)
        

        return (lower, upper)

    def _search(self, system, model, test, lower, upper, max_control_cost, noisy, fit_kwargs, act_kwargs):
        lowest_needed = None
       
        while (lower != upper):
            mid = int((lower + upper) / 2)

            mid_error = self._evaluate_n(
                system, model, test, mid, max_control_cost, noisy, fit_kwargs, act_kwargs)


            if mid_error > self._target_loss:
                if lower == mid:
                    lower = mid + 1
                else:
                    lower = mid
            else:
                upper = mid
                lowest_needed = mid
        
        return lowest_needed
               

    def evaluate(self,
                 model_cls: type[Model],
                 model_kwargs: dict = None,
                 fit_kwargs: dict = None,
                 act_kwargs: dict = None,
                 in_dist=True,
                 noisy=False,
                 id=None,
                 ):

        model_kwargs = model_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        act_kwargs = act_kwargs or {}

        data = []

        def _do_rep(rep_id, latent_dim, embed_dim):

            self.rep_id = rep_id

            self.result = {k: [] for k in ["rep", "n", "latent_dim", "embed_dim", "timesteps", "loss", "total_cost"]}

            system = None
            system = self._set_system(system, latent_dim, embed_dim)
            max_control_cost = self._max_control_cost_per_dim * latent_dim
            model = model_cls(embed_dim, self._T, max_control_cost, **model_kwargs)
            test = self._gen_testset(system, in_dist)

            lower, upper = self._gen_bounds(system, model, test, latent_dim, max_control_cost, noisy, fit_kwargs, act_kwargs)
            self._search(system, model, test, lower, upper, max_control_cost, noisy, fit_kwargs, act_kwargs)

            return pd.DataFrame(self.result)

        for idx, latent_dim in enumerate(self._L):
            self.X = None
            if not self._E:
                embed_dim = latent_dim
            elif isinstance(self._E, int):
                embed_dim = self._E
            else:
                embed_dim = self._E[idx]

          

            temp_df = Parallel(n_jobs=1)(delayed(_do_rep)(rep_id, latent_dim, embed_dim)
                                                      for rep_id in range(self._reps))

  
            temp_df = pd.concat(temp_df, ignore_index=True)

            self.samples_needed[latent_dim] = int(round(np.median(temp_df["n"])))

            data.append(temp_df)

        data = pd.concat(data,  ignore_index=True)
        data["id"] = id or next(self._id)

        return data

    def plot(self, data):
        plot_target_loss(data, "latent_dim", "n", target_loss=self._target_loss, ylabel=r'$n$')


class FixedComplexity(Task):
    def __init__(self, N: list[int], l: int, e: int, t: int, max_control_cost_per_dim: int, control_horizons: int,
                 system_cls: type[System], reps: int, test_examples: int, test_timesteps: int, system_kwargs: dict = None):
        L = [l]
        E = e
        T = [t]
        super().__init__(N, L, E, T, max_control_cost_per_dim, control_horizons,
                         system_cls, reps, test_examples, test_timesteps, system_kwargs=system_kwargs)

    @staticmethod
    def plot(data):
        plot_metric(data, "n", "loss", xlabel=r'$n$')
        # g = sns.lmplot(
        #     data=data,
        #     x="n", y="loss", hue="id",
        #     height=5,
        #     ci=50,
        #     robust=True,
        #     facet_kws=dict(sharey=False),
        #     scatter=False,
        #     fit_reg=True
        # )
        # g.set_axis_labels(x_var="$n$", y_var="Loss")
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.show()


class FixedTrainSize(Task):
    def __init__(self, n: int, L: list[int], E: list[int] | int | None, T: list[int], max_control_cost_per_dim: int, control_horizons: int,
                 system_cls: type[System], reps: int, test_examples: int, test_timesteps: int, system_kwargs: dict = None):
        N = [n]
        super().__init__(N, L, E, T, max_control_cost_per_dim, control_horizons,
                         system_cls, reps, test_examples, test_timesteps, system_kwargs=system_kwargs)

    @staticmethod
    def plot(data: pd.DataFrame):
        g = sns.lmplot(
            data=data,
            x="latent_dim", y="loss", hue="id",
            height=5,
            ci=50,
            robust=True,
            facet_kws=dict(sharey=False),
            scatter=False,
            fit_reg=True
        )
        g.set_axis_labels(x_var="Latent Dimension", y_var="Loss")
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
