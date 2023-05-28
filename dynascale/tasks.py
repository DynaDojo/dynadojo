import sys
import itertools

import seaborn as sns
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from joblib import Parallel, delayed

from dynascale.utils.plotting import plot_target_loss, plot_metric
from abstractions import Task, System, Model


class TargetError(Task):
    def __init__(self, L: list[int], t: int, max_control_cost_per_dim: int, control_horizons: int,
                 system_cls: type[System], reps: int, test_examples: int, test_timesteps: int, target_loss: float, max_samples=10000, E: int | list[int] = None):

        if isinstance(E, list):
            assert (len(L) == len(E))
            for l, idx in enumerate(L):
                assert (E[idx] >= l)
        elif isinstance(E, int):
            assert (E >= np.max(L))

        self._target_loss = target_loss

        self.X = None
        self.max_samples = max_samples
        self.samples_needed = {}

        super().__init__(L, E, t, max_control_cost_per_dim, control_horizons,
                         system_cls, reps, test_examples, test_timesteps)

    def _evaluate_n(self, system, model, test, n, max_control_cost, noisy, fit_kwargs, act_kwargs):
        x = None
        if not self.X:
            X = self._gen_trainset(system, n, self._T, noisy)
            x = X

        # if already generated n samples, pull from existing training set; else add more samples
        else:
            if (len(X) >= n):
                x = X[:n]
            else:
                to_add = self._gen_trainset(
                    system, n - len(X), self._T, noisy)
                X += to_add
                x = X

        total_cost = self._fit_model(system, model, x, self._T,
                                     max_control_cost, fit_kwargs, act_kwargs, noisy)

        pred = model.predict(test[:, 0], self._test_timesteps)
        return system.calc_loss(pred, test), total_cost

    def _gen_bounds(self, system, model, test, l, results, max_control_cost, noisy, fit_kwargs, act_kwargs):

        lower = 1
        upper = 1
        predicted = 1

        if self.samples_needed:
            prev_largest_dim = np.max(self.samples_needed.keys())
            lower = self.samples_needed[prev_largest_dim]

            # adaptively predict based on median of samples needed in lower dims
            if len(self.samples_needed.keys()) > 1:
                p = np.polyfit(self.samples_needed.keys(), self.samples_needed.values(),
                               deg=len(self.samples_needed.keys()) - 1)
                predict_func = np.poly1d(p)
                predicted = round(predict_func(l))

            # with only one data dim seens, assume a very small slope
            else:
                predicted = self.samples_needed[prev_largest_dim] + 1

            upper = (2 * predicted) - lower

        # if lower meets target loss, decrease to be lower
        lower_error, _ = self._evaluate_n(
            system, model, test, lower, max_control_cost, noisy, fit_kwargs, act_kwargs)
        while lower_error < self._target_loss:
            lower = lower / 2

        # increase upper until it satisfies target loss or exceeds max_samples
        while upper < self.max_samples:
            if upper == self.max_samples:
                max_samples_error, _ = self._evaluate_n(
                    system, model, test, self.max_samples, max_control_cost, noisy, fit_kwargs, act_kwargs)
                if max_samples_error < self._target_loss:
                    break
                else:
                    sys.exit(
                        f'At {self.max_samples} samples error is {max_samples_error}, which is larger than the target fixed error')

            else:
                upper_error, _ = self._evaluate_n(
                    system, model, test, self.max_samples, max_control_cost, noisy, fit_kwargs, act_kwargs)
                if upper_error < self._target_loss:
                    break
                else:
                    temp_upper = round((2 * (upper - lower)) + upper)
                    lower = upper
                    upper = min(self.max_samples, temp_upper)

        return (lower, upper)

    def _search(self, system, model, test, lower, upper, max_control_cost, noisy, fit_kwargs, act_kwargs):
        lowest_needed = upper

        while (lower != upper):
            mid = int((lower + upper) / 2)

            mid_error, total_cost = self._evaluate_n(
                system, model, test, mid, max_control_cost, noisy, fit_kwargs, act_kwargs)

            if mid_error < self._target_loss:
                if lower == mid:
                    lower = mid + 1
                else:
                    lower = mid
            else:
                upper = mid
                lowest_needed = mid

        return lowest_needed, mid_error, total_cost

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

        def _do_rep():
            result = {k: [] for k in ["n", "latent_dim",
                                      "embed_dim", "timesteps", "loss", "total_cost"]}

            if not self._E:
                embed_dim = latent_dim
            elif isinstance(self._E, int):
                embed_dim = self._E
            else:
                embed_dim = self._E[idx]

            system = self._set_system(latent_dim, embed_dim)

            max_control_cost = self._max_control_cost_per_dim * latent_dim

            model = model_cls(embed_dim, self._T,
                              max_control_cost, **model_kwargs)

            test = self._gen_testset(system, in_dist)

            lower, upper = self._gen_bounds(system, model, test, latent_dim, results,
                                            max_control_cost, noisy, fit_kwargs, act_kwargs)

            lowest_needed, loss_achieved, total_cost = self._search(
                system, model, test, lower, upper, max_control_cost, noisy, fit_kwargs, act_kwargs)

            result['n'].append(lowest_needed)
            result['latent_dim'].append(latent_dim)
            result['embed_dim'].append(embed_dim)
            result['timesteps'].append(self._T)
            result['loss'].append(loss_achieved)
            result['total_cost'].append(total_cost)

            return pd.DataFrame(result)

        for idx, latent_dim in enumerate(self._L):
            temp_df = Parallel(n_jobs=4, timeout=1e5)(delayed(_do_rep)()
                                                      for _ in range(self._reps))

            self.samples_needed[latent_dim] = np.median(temp_df.n)

            data.append(temp_df)

        data = pd.concat(data)
        data["id"] = id or next(self._id)

        return data

    def plot(self, data):
        plot_target_loss(data, "latent_dim", "n",
                         target_error=self._target_err)


class FixedComplexity(Task):
    def __init__(self, N: list[int], l: int, e: int, t: int, max_control_cost_per_dim: int, control_horizons: int,
                 system_cls: type[System], reps: int, test_examples: int, test_timesteps: int, system_kwargs: dict = None):
        L = [l]
        E = [e]
        T = [t]
        super().__init__(N, L, E, T, max_control_cost_per_dim, control_horizons,
                         system_cls, reps, test_examples, test_timesteps, system_kwargs=system_kwargs)

    @staticmethod
    def plot(data):
        g = sns.lmplot(
            data=data,
            x="n", y="loss", hue="id",
            height=5,
            ci=50,
            robust=True,
            facet_kws=dict(sharey=False),
            scatter=False,
            fit_reg=True
        )
        g.set_axis_labels(x_var="$n$", y_var="Loss")
        plt.yscale('log')
        plt.xscale('log')
        plt.show()


class FixedTrainSize1(Task):
    def __init__(self, n: int, L: list[int], E: list[int], T: list[int], max_control_cost_per_dim: int, control_horizons: int,
                 system_cls: type[System], reps: int, test_examples: int, test_timesteps: int, system_kwargs: dict = None):
        N = [n]
        super().__init__(N, L, E, T, max_control_cost_per_dim, control_horizons,
                         system_cls, reps, test_examples, test_timesteps, system_kwargs=system_kwargs)

    @staticmethod
    def plot(data: pd.DataFrame):
        # TODO: specify l, e, t
        plot_metric(data, "latent_dim", "loss")

# TODO: constant, empty, list


class FixedTrainSize2(Task):
    def __init__(self,
                 n: int,
                 D: list[int],
                 T: list[int],
                 max_control_cost_per_dim: int,
                 control_horizons: int,
                 system_cls: type[System],
                 reps: int,
                 test_examples: int,
                 test_timesteps: int,
                 system_kwargs: dict = None
                 ):
        N = [n]
        self._D = D
        super().__init__(N, [], [], T, max_control_cost_per_dim, control_horizons, system_cls, reps, test_examples,
                         test_timesteps, system_kwargs=system_kwargs)

    @staticmethod
    def plot(data: pd.DataFrame):
        # TODO: specify l, e, t
        plot_metric(data, "latent_dim", "loss")

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

        def do_rep():
            result = {k: [] for k in ["n", "latent_dim",
                                      "embed_dim", "timesteps", "loss", "total_cost"]}
            system = None
            for n, dim, timesteps in itertools.product(self._N, self._D, self._E, self._T):
                if system is None:
                    system = self._system_cls(dim, dim, **self._system_kwargs)
                if dim != system.latent_dim:
                    system.latent_dim = dim
                if dim != system.embed_dim:
                    system.embed_dim = dim

                max_control_cost = self._max_control_cost_per_dim * dim
                print(f"{n=}, {dim=}, {timesteps=}")

                # Create and train model
                model = model_cls(
                    dim, timesteps, max_control_cost, **model_kwargs)
                train_init_conds = system.make_init_conds(n)

                total_cost = 0

                for j in range(self._control_horizons):
                    if j == 0:
                        x = system.make_data(
                            train_init_conds, timesteps=timesteps, noisy=noisy)
                    else:
                        control = model.act(x, **act_kwargs)
                        cost = system.calc_control_cost(control)
                        total_cost += cost
                        assert np.all(
                            cost <= max_control_cost), "Control cost exceeded!"
                        x = system.make_data(init_conds=x[:, 0], control=control, timesteps=timesteps,
                                             noisy=noisy)
                    model.fit(x, **fit_kwargs)

                # create test data
                test_init_conds = system.make_init_conds(
                    self._test_examples, in_dist)
                test = system.make_data(
                    test_init_conds, timesteps=self._test_timesteps)
                pred = model.predict(test[:, 0], self._test_timesteps)
                loss = system.calc_loss(pred, test)
                result['n'].append(n)
                result['latent_dim'].append(dim)
                result['embed_dim'].append(dim)
                result['timesteps'].append(timesteps)
                result['loss'].append(loss)
                result['total_cost'].append(total_cost)
            return pd.DataFrame(result)

        data = Parallel(n_jobs=4)(delayed(do_rep)() for _ in range(self._reps))

        data = pd.concat(data)
        data["id"] = id or next(self._id)
        return data
