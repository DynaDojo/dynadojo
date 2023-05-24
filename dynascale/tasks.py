import itertools

import seaborn as sns
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from dynascale.utils.plotting import plot_metric
from abstractions import Model, Task, System


class FixedComplexity(Task):
    def __init__(self, N: list[int], l: int, e: int, t: int, max_control_cost_per_dim: int, control_horizons: int,
                 system_cls: type[System], reps: int, test_examples: int, test_timesteps: int, system_kwargs: dict = None):
        L = [l]
        E = [e]
        T = [t]
        super().__init__(N, L, E, T, max_control_cost_per_dim, control_horizons, system_cls, reps, test_examples, test_timesteps, system_kwargs=system_kwargs)

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
        super().__init__(N, L, E, T, max_control_cost_per_dim, control_horizons, system_cls, reps, test_examples, test_timesteps, system_kwargs=system_kwargs)

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
            result = {k: [] for k in ["n", "latent_dim", "embed_dim", "timesteps", "loss", "total_cost"]}
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
                model = model_cls(dim, timesteps, max_control_cost, **model_kwargs)
                train_init_conds = system.make_init_conds(n)

                total_cost = 0

                for j in range(self._control_horizons):
                    if j == 0:
                        x = system.make_data(train_init_conds, timesteps=timesteps, noisy=noisy)
                    else:
                        control = model.act(x, **act_kwargs)
                        cost = system.calc_control_cost(control)
                        total_cost += cost
                        assert np.all(cost <= max_control_cost), "Control cost exceeded!"
                        x = system.make_data(init_conds=x[:, 0], control=control, timesteps=timesteps,
                                                noisy=noisy)
                    model.fit(x, **fit_kwargs)

                # create test data
                test_init_conds = system.make_init_conds(self._test_examples, in_dist)
                test = system.make_data(test_init_conds, timesteps=self._test_timesteps)
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
