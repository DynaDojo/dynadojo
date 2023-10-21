import warnings
import pandas as pd
import numpy as np

from .utils.plotting import plot_target_error, plot_metric

from joblib import Parallel, delayed

from .abstractions import Challenge, AbstractSystem, AbstractModel


class FixedError(Challenge):
    def __init__(self, L: list[int], t: int, max_control_cost_per_dim: int, control_horizons: int,
                 system_cls: type[AbstractSystem], reps: int, test_examples: int, test_timesteps: int, target_error: float,
                 system_kwargs: dict = None,
                 max_samples=10000, E: int | list[int] = None):

        if isinstance(E, list):
            assert (len(L) == len(E))
            for idx, l in enumerate(L):
                assert (E[idx] >= l)
        elif isinstance(E, int):
            assert (E >= np.max(L))

        # the dimensions have to be sorted, we need a guarantee that they are sorted
        self._target_error = target_error
        self.max_samples = max_samples

        self.X = None
        self.rep_id = None
        self.result = None
        self.samples_needed = {}

        super().__init__([1], L, E, t, max_control_cost_per_dim, control_horizons, system_cls, reps, test_examples, test_timesteps, system_kwargs=system_kwargs)

    def _evaluate_n(self, system, model_cls, test, n, max_control_cost, noisy, fit_kwargs, act_kwargs, model_kwargs):
        x = None

        df = pd.DataFrame(self.result)
        df = df.loc[(df['rep'] == self.rep_id) & (df['n'] == n) & df['latent_dim'] == system.latent_dim]
        if len(df) == 1:
            return df.iloc[0]['error']

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

        model = model_cls(system.embed_dim, self._T, max_control_cost, **model_kwargs)
        total_cost = self._fit_model(system, model, x, self._T,
                                     max_control_cost, fit_kwargs, act_kwargs, noisy)


        pred = model.predict_wrapper(test[:, 0], self._test_timesteps)

        error = system.calc_error_wrapper(pred, test)

        self._append_result(self.result, self.rep_id, n, system.latent_dim, system.embed_dim, self._T, error, total_cost)

        return error

    def _gen_bounds(self, system, model_cls, test, l, max_control_cost, noisy, fit_kwargs, act_kwargs, model_kwargs):
        lower = 1
        upper = 3
        predicted = 2

        if self.samples_needed:
            prev_largest_dim = np.max(list(self.samples_needed.keys()))
            lower = self.samples_needed[prev_largest_dim]

            # adaptively predict based on median of samples needed in lower dims
            if len(self.samples_needed.keys()) > 1:
                # p = np.polyfit(list(self.samples_needed.keys()), list(self.samples_needed.values()), deg=len(list(self.samples_needed.keys())) - 1)
                p = self.samples_needed[prev_largest_dim] * 2
                predict_func = np.poly1d(p)
                predicted = int(round(predict_func(l)))

            # with only one data dim seens, assume a very small slope
            else:
                predicted = self.samples_needed[prev_largest_dim] + 1

            if predicted < lower:
                predicted = lower+1
            upper = (2 * predicted) - lower

        # if lower meets target error, decrease to be lower
        lower_error = self._evaluate_n(system, model_cls, test, lower, max_control_cost, noisy, fit_kwargs, act_kwargs, model_kwargs)
        while lower_error <= self._target_error:
            if lower == 1:
                break
            lower = lower // 2
            lower_error = self._evaluate_n(system, model_cls, test, lower, max_control_cost, noisy, fit_kwargs, act_kwargs, model_kwargs)

        # increase upper until it satisfies target error or exceeds max_samples
        while upper <= self.max_samples:
            if upper == self.max_samples:
                max_samples_error = self._evaluate_n(
                    system, model_cls, test, self.max_samples, max_control_cost, noisy, fit_kwargs, act_kwargs, model_kwargs)
                if max_samples_error <= self._target_error:
                    break
                else:
                    warnings.warn(f'Rep #{self.rep_id} for l={system.latent_dim} and e={system.embed_dim} failed: Fixed error not achieved within max samples. At {self.max_samples} samples, error is still {max_samples_error}, which is above {self._target_error}')
                    return (lower, None) 

            else:
                upper_error = self._evaluate_n(
                    system, model_cls, test, upper, max_control_cost, noisy, fit_kwargs, act_kwargs, model_kwargs)
                
                if upper_error <= self._target_error:
                    break
                else:
                    if upper == lower:
                        temp_upper = lower+1
                    else:
                        temp_upper = round((2 * (upper - lower)) + upper)
                    lower = upper
                    upper = min(self.max_samples, temp_upper)
        

        return (lower, upper)

    def _search(self, system, model_cls, test, lower, upper, max_control_cost, noisy, fit_kwargs, act_kwargs, model_kwargs):
        lowest_needed = None
       
        while (lower != upper):
            mid = int((lower + upper) / 2)

            mid_error = self._evaluate_n(
                system, model_cls, test, mid, max_control_cost, noisy, fit_kwargs, act_kwargs, model_kwargs)


            if mid_error > self._target_error:
                if lower == mid:
                    lower = mid + 1
                else:
                    lower = mid
            else:
                upper = mid
                lowest_needed = mid
        
        return lowest_needed
               

    def evaluate(self,
                 model_cls: type[AbstractModel],
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

            self.result = {k: [] for k in ["rep", "n", "latent_dim", "embed_dim", "timesteps", "control_horizons", "error", "total_cost"]}
            print(f"{latent_dim=}, {embed_dim=}, timesteps={self._T}, control_horizons={self._control_horizons}, {rep_id=}, {id=}")

            system = None
            system = self._set_system(system, latent_dim, embed_dim)
            max_control_cost = self._max_control_cost_per_dim * latent_dim
            
            test = self._gen_testset(system, in_dist)

            lower, upper = self._gen_bounds(system, model_cls, test, latent_dim, max_control_cost, noisy, fit_kwargs, act_kwargs, model_kwargs)
            if upper is not None:
                self._search(system, model_cls, test, lower, upper, max_control_cost, noisy, fit_kwargs, act_kwargs, model_kwargs)

            self.X = None 
            return pd.DataFrame(self.result)

        for idx, latent_dim in enumerate(self._L):
            self.X = None
            if not self._E:
                embed_dim = latent_dim
            elif isinstance(self._E, int):
                embed_dim = self._E
            else:
                embed_dim = self._E[idx]

            temp_df = Parallel(n_jobs=1)(delayed(_do_rep)(rep_id, latent_dim, embed_dim) for rep_id in range(self._reps))
            temp_df = pd.concat(temp_df, ignore_index=True)

            self.samples_needed[latent_dim] = int(round(np.median(temp_df["n"])))

            data.append(temp_df)
       
        self.samples_needed = None
        self.X = None
        self.result = None

        data = pd.concat(data,  ignore_index=True)
        data["id"] = id or next(self._id)

        return data

    def plot(self, data):
        plot_target_error(data, "latent_dim", "n", target_error=self._target_error, ylabel=r'$n$')


class FixedComplexity(Challenge):
    def __init__(self, N: list[int], l: int, e: int, t: int, max_control_cost_per_dim: int, control_horizons: int,
                 system_cls: type[AbstractSystem], reps: int, test_examples: int, test_timesteps: int, system_kwargs: dict = None):
        L = [l]
        E = e
        super().__init__(N, L, E, t, max_control_cost_per_dim, control_horizons,
                         system_cls, reps, test_examples, test_timesteps, system_kwargs=system_kwargs)

    @staticmethod
    def plot(data):
        plot_metric(data, "n", "error", xlabel=r'$n$')

class FixedTrainSize(Challenge):
    def __init__(self, n: int, L: list[int], E: list[int] | int | None, t: int, 
                max_control_cost_per_dim: int, control_horizons: int,
                system_cls: type[AbstractSystem], reps: int, test_examples: int, test_timesteps: int, system_kwargs: dict = None):
        N = [n]
        super().__init__(N, L, E, t, max_control_cost_per_dim, control_horizons,
                         system_cls, reps, test_examples, test_timesteps, system_kwargs=system_kwargs)

    @staticmethod
    def plot(data: pd.DataFrame):
        plot_metric(data, "latent_dim", "error", xlabel=r'$L$', log=False)
