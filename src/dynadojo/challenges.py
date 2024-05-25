"""
Contains a `ScalingChallenge` class and several commonly used subclasses. The `ScalingChallenge` class is a stand-alone class that
can also be extended.
"""
import itertools
import logging
import math
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import functools

from .abstractions import AbstractSystem, AbstractAlgorithm, AbstractChallenge
from .wrappers import AlgorithmChecker, SystemChecker
from .utils.plotting import plot_target_error, plot_metric



class ScalingChallenge(AbstractChallenge):

    """Challenge class used to benchmark algorithms and systems."""
    def __init__(self,
                 N: list[int],
                 L: list[int],
                 E: list[int] | int | None,
                 t: int,
                 max_control_cost_per_dim: int,
                 control_horizons: int,
                 system_cls: type[AbstractSystem],
                 trials: int,
                 test_examples: int,
                 test_timesteps: int,
                 system_kwargs: dict | None = None
                 ):
        """
        Initialize the class.

        Parameters
        ----------
        N : list of int
            Train sizes, (# of trajectories).
        L : list of int
            Latent dimensions.
        E : int or list or None, optional
            Embedded dimensions.
            If list, then evaluate iterates across embedded dimensions. (e >= l)
            If int, then evaluate uses a fixed embedded dimension. (E >= max(L))
            If None, then evaluate sets the embedded dimension equal to the latent dimension. (e = l)
        t : int
            Timesteps (length of a trajectory).
        max_control_cost_per_dim : int
            Max control cost per control trajectory.
        control_horizons : int
            Number of times to generate training data with control.
        system_cls : type[AbstractSystem]
            Class constructor (NOT instance) for a concrete system.
        trials : int
            Number of trials for each run.
        test_examples : int
            Test size.
        test_timesteps : int
            Test timesteps.
        system_kwargs : dict or None, optional
            Additional keyword arguments for the system constructor.
        """
        assert control_horizons >= 0

        self._id = itertools.count()
        self._N = N
        self._L = L
        self._E = E
        self._t = t
        self._max_control_cost_per_dim = max_control_cost_per_dim
        self._system_cls = system_cls
        self._system_kwargs = system_kwargs or {}
        self._control_horizons = control_horizons
        self._trials = trials
        self._test_examples = test_examples
        self._test_timesteps = test_timesteps

        # Sets embedded dim array when self._E is None, constant, or an array
        if self._E is None:
            E = self._L
        elif isinstance(self._E, int):
            assert self._E >= max(self._L), "E must be greater than or equal to max(L)."
            E = [self._E] * len(self._L)
        else:
            assert isinstance(self._E, list), "E must of type List[int], int, or None."
            assert len(self._E) != len(self._L), "E (type List[int]) and L must be of the same length."
            E = self._E

        super().__init__(
            sweep_params = {
                "latent_dim": L,
                "embed_dim": E,
            }
        )

    def evaluate(self,
                    algo_cls: type[AbstractAlgorithm],
                    algo_kwargs: dict | None = None,
                    fit_kwargs: dict | None = None,
                    act_kwargs: dict | None = None,
                    ood=False,
                    noisy=False,
                    id=None, 
                    num_parallel_cpu=-1,
                    seed=None,
                    jobs_filter: list[int] | None = None,
                    csv_output_path: str | None = None,
                    
                 ) -> pd.DataFrame:
        """
        Evaluates an algorithm class (NOT an instance) on a dynamical system over a set of experimental parameters.

        Parameters
        ----------
        algo_cls : type[AbstractAlgorithm]
            Algorithm class to be evaluated.
        algo_kwargs : dict or None, optional
            Keyword arguments to be passed to algo_cls.
        fit_kwargs : dict or None, optional
            Keyword arguments to be passed to algo_cls.fit.
        act_kwargs : dict or None, optional
            Keyword arguments to be passed to algo_cls.act.
        ood : bool, optional
            If True, also test on out-distribution initial conditions for the test set.
            (For FixedError, search is performed on ood_error if ood=True.) Defaults to False.
            If False, generate out-of-distribution initial conditions for the test set.
        noisy : bool, optional
            If True, add noise to the train set. Defaults to False. If False, no noise is added.
        id : int or None, optional
            Algorithm ID associated with evaluation results in the returned DataFrame.
        num_parallel_cpu : int, optional
            Number of CPUs to use in parallel. Defaults to -1, which uses all available CPUs.
        seed : int or None, optional
            Seed to seed the random number generator for seeding systems and algorithms. Defaults to None.
            Is overridden by seeds in system_kwargs or algo_kwargs.
        jobs_filter : list[int] | None, optional
            Specifies which jobs to evaluate as specified by job_id. Defaults to None, which evaluates all jobs.
        csv_output_path : str | None, optional
            Path to save results. Will add to file if exists. Defaults to None, which does not save results.

        Returns
        -------
        pandas.DataFrame
            A DataFrame where each row is the result of an algorithm trained and evaluated on a single system.
        """
        algo_kwargs = algo_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        act_kwargs = act_kwargs or {}

        if not id:
            id = algo_cls.__name__

        return super().evaluate(
            seed = seed,
            trials = self._trials,
            num_parallel_cpu = num_parallel_cpu,
            jobs_filter = jobs_filter,
            csv_output_path = csv_output_path,
            algo_cls = algo_cls,        #to pass to execute_job
            algo_kwargs = algo_kwargs,  #to pass to execute_job
            fit_kwargs = fit_kwargs,    #to pass to execute_job
            act_kwargs = act_kwargs,    #to pass to execute_job
            noisy = noisy,              #to pass to execute_job
            ood = ood,                  #to pass to execute_job
            id = id                     #to pass to execute_job
        )

    def _gen_trainset(self, system, n: int, timesteps: int, noisy=False):
        train_init_conds = system.make_init_conds(n)
        return system.make_data(train_init_conds, timesteps=timesteps, noisy=noisy)

    def _gen_testset(self, system, in_dist=True, noisy=False):
        test_init_conds = system.make_init_conds(self._test_examples, in_dist)
        return system.make_data(test_init_conds, timesteps=self._test_timesteps, noisy=noisy)

    def _fit_algo(self, system, algo, x: np.ndarray, timesteps: int, max_control_cost: int, fit_kwargs: dict = None,
                  act_kwargs: dict = None, noisy=False) -> int:
        total_cost = 0
        algo.fit(x)

        for _ in range(self._control_horizons):
            control = algo.act(x, **act_kwargs)
            cost = system.calc_control_cost(control)
            total_cost += cost
            assert np.all(cost <= max_control_cost), "Control cost exceeded!"
            x = system.make_data(
                init_conds=x[:, -1], control=control, timesteps=timesteps, noisy=noisy)
            algo.fit(x, **fit_kwargs)

        return total_cost

    @staticmethod
    def _append_result(job_id, result, trial, n, latent_dim, embed_dim, timesteps, total_cost, error, ood_error=None,
                       duration=None):
        result['job_id'].append(job_id)
        result['trial'].append(trial)
        result['n'].append(n)
        result['latent_dim'].append(latent_dim)
        result['embed_dim'].append(embed_dim)
        result['timesteps'].append(timesteps)
        result['error'].append(error)
        result['total_cost'].append(total_cost)
        result['ood_error'].append(ood_error)
        result['duration'].append(duration)

    @staticmethod
    def _init_result_dict():
        result = {k: [] for k in [
            "job_id",
            "trial", 
            "latent_dim",
            "embed_dim",
            "timesteps",
            "n",
            "error",
            "ood_error",
            "total_cost",
            "duration"]}
        return result

    def execute_job(
                    self,
                    job_id : int,
                    trial : int = None,
                    system_seed : int = 0, 
                    algo_seed : int = 0,
                   latent_dim : int = None,      #from sweep_params
                   embed_dim : int = None,       #from sweep_params
                   algo_cls: type[AbstractAlgorithm] = None,    #from evaluate
                   algo_kwargs: dict = None,                    #from evaluate
                   fit_kwargs: dict = None,                     #from evaluate
                   act_kwargs: dict = None,                     #from evaluate
                   noisy: bool = False,                         #from evaluate
                   ood: bool = False,                           #from evaluate
                   id: str = None,                              #from evaluate
                   **kwargs                      #extra params from evaluate
                   ):
        """
        For a given system latent dimension and embedding dimension, instantiates system and for a specific N,
        evaluates the algorithm on the system. Across runs, the algorithm is re-initialized with the same seed.

        Note
        -----
        The seed in algo_kwargs and system_seed in system_kwargs takes precedence over the seed passed to this function.
        """
        result = ScalingChallenge._init_result_dict()

        if not embed_dim or not latent_dim or embed_dim < latent_dim:
            return

        # Seed in system_kwargs takes precedence over the seed passed to this function.
        system = SystemChecker(self._system_cls(latent_dim, embed_dim, **{"seed": system_seed, **self._system_kwargs}))

        # Create all data
        test_set = self._gen_testset(system, in_dist=True, noisy=noisy)
        ood_test_set = self._gen_testset(system, in_dist=False, noisy=noisy)
        largest_N = max(self._N)
        training_set = self._gen_trainset(system, largest_N, self._t, noisy)

        max_control_cost = self._max_control_cost_per_dim * latent_dim

        # Define algo_run helper function
        def algo_run(n):
            """
            For a given number of trajectories n, instantiates algo, trains, and evaluates on test set.
            """
            start = time.time()
            # Create algo. Seed in algo_kwargs takes precedence over the seed passed to this function.
            algo = algo_cls(embed_dim, self._t, max_control_cost, **{"seed": algo_seed, **algo_kwargs})
            algo = AlgorithmChecker(algo)
            training_set_n = training_set[:n]  # train on subset of training set
            total_cost = self._fit_algo(system, algo, training_set_n, self._t, max_control_cost, fit_kwargs,
                                        act_kwargs, noisy)
            pred = algo.predict(test_set[:, 0], self._test_timesteps)
            error = system.calc_error(pred, test_set)

            # check if system has save_plotted_trajectories method 
            #TODO HACKY INTERMEDIATE PLOTTING should fix
            if hasattr(system._system, 'save_plotted_trajectories') and kwargs.get('intermediate_plots_dir', False):
                system._system.save_plotted_trajectories(
                    test_set, 
                    pred, 
                    os.path.join(kwargs['intermediate_plots_dir'], f"n={n}_l={latent_dim}_trial={trial}_e={error:.3e}.pdf"),
                    tag = f"n={n}_trial={trial}_e={error:.3f}")

            ood_error = None
            if ood: 
                ood_pred = algo.predict(ood_test_set[:, 0], self._test_timesteps)
                ood_error = system.calc_error(ood_pred, ood_test_set)
            end = time.time()
            duration = end - start
            ood_error_str = f"{ood_error=:0.3}" if ood else "ood_error=NA"
            logging.debug(
                f"{job_id=} | {trial=}, {latent_dim=}, {embed_dim=}, {n=}, t={self._t}, control_h={self._control_horizons}, {total_cost=}, {error=:0.3}, {ood_error_str},algo_seed={algo.seed}, sys_seed={system.seed}")
            ScalingChallenge._append_result(
                job_id, result, trial, n, latent_dim, embed_dim, self._t, total_cost, error, ood_error=ood_error, duration=duration)

        # On each subset of the training set, we retrain the algo from scratch (initialized with the same random seed).
        # If you don't, then # of training epochs will scale with N. This would confound the effect of training set size with training time.
        for n in self._N:
            algo_run(n)

        data = pd.DataFrame(result)
        data['system_seed'] = system_seed
        data['algo_seed'] = algo_seed
        data['id'] = id
        return data


class FixedComplexity(ScalingChallenge):
    """Challenge where complexity is fixed, training set size is varied, and error is measured."""
    def __init__(self,
                 l: int,
                 t: int,
                 N: list[int],
                 system_cls: type[AbstractSystem],
                 trials: int,
                 test_examples: int,
                 test_timesteps: int,
                 e: int = None,
                 max_control_cost_per_dim: int = 1,
                 control_horizons: int = 0,
                 system_kwargs: dict = None):
        """
        Initialize the class.

        Parameters
        ----------
        l : int
            Latent dimension.
        t : int
            Number of timesteps.
        N : list[int]
            List of training set sizes.
        system_cls : type[AbstractSystem]
            System class.
        trials : int
            Number of trials for each run/training set size.
        test_examples : int
            Number of test examples.
        test_timesteps : int
            Number of test timesteps.
        e : int, optional
            Embedding dimension. Defaults to None.
        max_control_cost_per_dim : int, optional
            Maximum control cost per dimension. Defaults to 1.
        control_horizons : int, optional
            Number of control horizons. Defaults to 0.
        system_kwargs : dict, optional
            System kwargs. Defaults to None.
        """
        L = [l]
        E = e
        super().__init__(N, L, E, t, max_control_cost_per_dim, control_horizons,
                         system_cls, trials, test_examples, test_timesteps, system_kwargs=system_kwargs)

    @staticmethod
    def plot(data, latent_dim: int = None, embedding_dim: int = None, show: bool = True, show_stats: bool = False, showLegend=True):
        """
        Returns
        --------
        matplotlib.axes.Axes
            A plot of the data.
        """
        if show_stats: #TODO add column names...
            # get the n_target for each latent_dim
            stats = data[['trial', 'latent_dim', 'embed_dim', 'n', 'error', 'ood_error']]
            stats = stats.drop_duplicates()
            total = stats.groupby(['latent_dim', 'embed_dim', 'n'])['trial'].count().reset_index(name="total")
            total.reset_index()
            stats = stats.dropna(subset=['error', 'ood_error'])
            stats = stats.groupby(['latent_dim', 'embed_dim', 'n'])['trial'].count().reset_index(name="plotted")
            stats.reset_index()
            stats = pd.merge(total, stats, how="outer")
            print(stats)
        # data = data.dropna(subset=['error', 'ood_error'])
        if not latent_dim:
            latent_dim = data["latent_dim"].unique()[0]
        if not data['ood_error'].isnull().any():
            assert not np.isnan(data[['error','ood_error']]).values.any(), "data[['error','ood_error']] contains np.nan"
            assert not np.isinf(data[['error','ood_error']]).values.any(), "data[['error','ood_error']] contains np.inf"
            ax = plot_metric(data, "n", ["error", "ood_error"], xlabel=r'$n$', ylabel=r'$\mathcal{E}$',
                             errorbar=("pi", 50))
            if showLegend:
                ax.legend(title='Distribution')
            else:
                ax.get_legend().remove()
        else:
            assert not np.isnan(data['error']).any(), "data['error'] contains np.nan"
            assert not np.isinf(data['error']).any(), "data['error'] contains np.inf"
            ax = plot_metric(data, "n", "error", xlabel=r'$n$', ylabel=r'$\mathcal{E}$', errorbar=("pi", 50))
            if showLegend:
                ax.legend()
            else:
                ax.get_legend().remove()
        title = "Fixed Complexity"
        if latent_dim:
            title += f", latent={latent_dim}"
        if embedding_dim:
            title += f", embedding={embedding_dim}"
        ax.set_title(title)
        ax.set_xlabel(r'$n$')
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return ax

class FixedTrainSize(ScalingChallenge):
    """Challenge where the size of the training set is fixed, the complexity of the system is varied, and the error is measured."""
    def __init__(self, n: int, L: list[int], E: list[int] | int | None, t: int,
                 system_cls: type[AbstractSystem], trials: int, test_examples: int, test_timesteps: int,
                 max_control_cost_per_dim: int = 0, control_horizons: int = 0,
                 system_kwargs: dict = None):

        """
        Initialize the class.

        Parameters
        ----------
        n : int
            The size of the training set.
        L : int
            The complexities of the system.
        E : int
            The embedding dimensions of the system.
        t : int
            The number of timesteps to simulate.
        max_control_cost_per_dim : int
            The maximum control cost per dimension.
        control_horizons : int
            The number of control horizons to consider.
        system_cls : type
            The system class to use.
        trials : int
            Number of trials for each run/latent dimension.
        test_examples : int
            The number of test examples to use.
        test_timesteps : int
            The number of timesteps to simulate for the test examples.
        system_kwargs : dict
            The keyword arguments to pass to the system class.
        """
        N = [n]
        super().__init__(N, L, E, t, max_control_cost_per_dim, control_horizons,
                 system_cls, trials, test_examples, test_timesteps, system_kwargs=system_kwargs)

    @staticmethod
    def plot(data: pd.DataFrame, n: int = None, show: bool = True, show_stats: bool = False, plot_ood=True, ax=None, showLegend=True):
        """
        Returns
        --------
        matplotlib.axes.Axes
            A plot of the data.
        """
        if not n:
            n = data["n"].unique()[0]
        if plot_ood and not data['ood_error'].isnull().any():
            assert not np.isnan(data[['error','ood_error']]).values.any(), "data[['error','ood_error']] contains np.nan"
            assert not np.isinf(data[['error','ood_error']]).values.any(), "data[['error','ood_error']] contains np.inf"
            ax = plot_metric(data, "latent_dim", ["error", "ood_error"], xlabel=r'$L$', log=True,
                             ylabel=r'$\mathcal{E}$', errorbar=("pi", 50))
            if not showLegend:
                ax.get_legend().remove()
            else:
                ax.legend(title='Distribution')
        else:
            assert not np.isnan(data['error']).any(), "data['error'] contains np.nan"
            assert not np.isinf(data['error']).any(), "data['error'] contains np.inf"
            ax = plot_metric(data, "latent_dim", "error", xlabel=r'$L$', log=True, ylabel=r'$\mathcal{E}$',
                             errorbar=("pi", 50))
            if showLegend:
                ax.legend()
            else:
                ax.get_legend().remove()
        title = "Fixed Train Size"
        if n:
            title += f", n={n}"
        ax.set_title(title)

        if show_stats: #TODO add column names...
            # get the n_target for each latent_dim
            stats = data[['trial', 'latent_dim', 'embed_dim']]
            stats = stats.drop_duplicates()
            stats = stats.groupby(['latent_dim', 'embed_dim'])['trial'].count().reset_index(name="total")
            stats.reset_index()
            print(stats)

        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return ax



class FixedError(ScalingChallenge):
    """
    Challenge where the target error is fixed and the latent dimensionality is varied and the number of training samples to achieve the error is measured.
    Performs a binary search over the number of training samples to find the minimum number of samples needed to achieve the target error rate.
    """
    def __init__(self,
                 L: list[int],
                 t: int,
                 max_control_cost_per_dim: int,
                 control_horizons: int,
                 system_cls: type[AbstractSystem],
                 trials: int,
                 test_examples: int,
                 test_timesteps: int,
                 target_error: float,
                 E: int | list[int] = None,
                 system_kwargs: dict = None,
                 n_precision: float = 0,  # 0 = precision, 0.05 = 5% precision
                 n_starts: list[int] = None,
                 n_window: int = 0,
                 n_max=10000,
                 n_min=1,
                 n_window_density: float = 1.0,  # 1 = include all points
                 ):
        """
        Parameters
        ----------
        L : list
            List of latent dimensions to test.
        t : int
            Number of timesteps of each training trajectory.
        max_control_cost_per_dim : int
            Maximum control cost per dimension.
        control_horizons : int
            Number of control horizons to test.
        system_cls : type
            System class to use.
        trials : int
            Number of trials to repeat for each run/latent dimension.
        test_examples : int
            Number of test examples to use.
        test_timesteps : int
            Number of timesteps of each test trajectory.
        target_error : float
            Target error to test.
        E : list, optional
            List of embedding dimensions to test. If None, defaults to L.
        system_kwargs : dict
            Keyword arguments to pass to the system class.
        n_precision : float
            Uncertainty interval around number of training samples to achieve target error. If 0, we find the exact minimum number of samples needed to achieve the target error rate.
        n_window : int
            Number of n to smooth (median) over when calculating the error rate during search. If 0, no filtering is done. Rounds up to the nearest odd number. 
        n_starts : list, optional
            List of starting points for the binary search over the number of training samples for each latent dim in L. len must equal len(L). If None, defaults to 1.
        n_max : int
            Maximum number of training samples to use.
        n_min : int
            Minimum number of training samples to use.
        n_window_density : float
            Density of n to test during smoothing. 1 = test every n. 0.5 = test every other n. 0.25 = test every fourth n. etc.
        """
        assert (1 > n_precision >= 0), "Precision must be between [0 and 1)"
        assert (0 < n_window_density <= 1), "Window density must be between (0 and 1]"
        assert (n_max > n_min), "n_max must be greater than n_min"
        assert (n_min > 0), "n_min must be greater than 0"
        assert (0 <= n_window), "Window size must be non-negative"
        # ToDo: sort L and E if not sorted instead of throwing error.

        self.n_starts = {l: 1 for l in L}
        if n_starts:
            assert (len(L) == len(n_starts)), "if supplied, must provide a starting point for each latent dimension"
            self.n_starts = {l: n for l, n in zip(L, n_starts)}

        self.n_precision = n_precision
        self.n_half_window = n_window//2 #this rounds the window up to nearest odd number. 
        self.n_max = n_max
        self.n_min = n_min
        self._target_error = target_error
        self.n_window_density = n_window_density

        self.rep_id = None
        self.result = None
        self.n_needed = {}

        super().__init__([1], L, E, t, max_control_cost_per_dim, control_horizons, system_cls, trials, test_examples,
                         test_timesteps, system_kwargs=system_kwargs)

    def evaluate(self,
                 algo_cls: type[AbstractAlgorithm],
                 algo_kwargs: dict | None = None,
                 fit_kwargs: dict | None = None,
                 act_kwargs: dict | None = None,
                 ood=False,
                 noisy=False,
                 id=None,
                 num_parallel_cpu=-1,
                 seed=None,
                 jobs_filter: list[int] | None = None,
                 csv_output_path: str | None = None,
                 ) -> pd.DataFrame:
        """
        Evaluates an algorithm class (NOT an instance) on a dynamical system over a set of experimental parameters.

        Parameters
        ----------
        algo_cls : type
            algo class to be evaluated.
        algo_kwargs : dict
            Keyword arguments to be passed to algo_cls.
        fit_kwargs : dict
            Keyword arguments to be passed to algo_cls.fit.
        act_kwargs : dict
            Keyword arguments to be passed to algo_cls.act.
        ood : bool, optional
            If True, also test on out-distribution initial conditions for the test set. (For FixedError, search is performed on ood_error if ood=True.) Defaults to False.
            If False, generate out-of-distribution initial conditions for the test set.
        noisy : bool, optional
            If True, add noise to the train set. Defaults to False. If False, no noise is added.
        id : str
            algo ID associated with evaluation results in the returned DataFrame.
        num_parallel_cpu : int, optional
            Number of CPUs to use in parallel. Defaults to -1, which uses all available CPUs.
        seed : int, optional
            Seed to initialize the random number generator for seeding systems and models. Defaults to None. Is overridden by seeds in system_kwargs or model_kwargs.
        jobs_filter : list[int] | None, optional
            Specifies which job ids to evaluate. Defaults to None, which evaluates all jobs.
        csv_output_path : str | None, optional
            Path to save results. Will add to file if exists. Defaults to None, which does not save results.
            
        Returns
        -------
        pandas.DataFrame
            A DataFrame where each row is a result from an algorithm trained and evaluated on a single system.
        """
        results = super().evaluate(algo_cls, algo_kwargs, fit_kwargs, act_kwargs, ood, noisy, id, num_parallel_cpu,
                                   seed, jobs_filter, csv_output_path)
        targets = results[['job_id','latent_dim', 'embed_dim', 'trial', 'n_target', 'algo_seed', 'system_seed']].drop_duplicates()

        for _, row in targets.iterrows():
            logging.info(
                f"ERROR TARGETED: job_id={row['job_id']} | n_target={row['n_target']}, latent={row['latent_dim']}, embed={row['embed_dim']}, trial={row['trial']}, seed={row['system_seed']},{row['algo_seed']}")

        return results

    

    def execute_job(self,
                    job_id : int,
                    trial : int = None,
                    system_seed : int = 0, 
                    algo_seed : int = 0,
                   latent_dim : int = None,      #from sweep_params
                   embed_dim : int = None,       #from sweep_params
                   algo_cls: type[AbstractAlgorithm] = None,    #from evaluate
                   algo_kwargs: dict = None,                    #from evaluate
                   fit_kwargs: dict = None,                     #from evaluate
                   act_kwargs: dict = None,                     #from evaluate
                   noisy: bool = False,                         #from evaluate
                   ood: bool = False,                           #from evaluate
                   id: str = None,                              #from evaluate
                   **kwargs                      #extra params from evaluate
                   ):
        """
        For a given system latent dimension and embedding dimension, instantiates system and evaluates a single trial of finding a
        number of training examples that achieves the target error. 

        Note
        -----
        The algo seed in model_kwargs and system_seed in system_kwargs takes precedence over the seed passed to this function.
        """
        result = ScalingChallenge._init_result_dict()

        if embed_dim < latent_dim:
            return

        # Seed in system_kwargs takes precedence over the seed passed to this function.
        system = SystemChecker(self._system_cls(latent_dim, embed_dim, **{"seed": system_seed, **self._system_kwargs}))

        # generate test set and max control cost
        test_set = self._gen_testset(system, in_dist=True, noisy=noisy)
        ood_test_set = self._gen_testset(system, in_dist=False, noisy=noisy)
        max_control_cost = self._max_control_cost_per_dim * latent_dim

        # generating master training set which is used to generate training sets of different sizes
        existing_training_set = None
        def get_training_set(required_size):
            """ 
            Helper function to get training set of size required size. 
            If we have already generated enough trajectories, simply return a subset of the training set. 
            If not, generate more trajectories and add to training set.
            """
            nonlocal existing_training_set
            current_size = len(existing_training_set) if existing_training_set is not None else 0
            if current_size < required_size:
                additional_data = self._gen_trainset(system, required_size - current_size, self._t, noisy)
                existing_training_set = np.concatenate((existing_training_set, additional_data), axis=0) if existing_training_set is not None else additional_data
            return existing_training_set[:required_size]

        # Given training set size, instantiate algorithm, train, and test once. Records result.
        @functools.cache   #memoized
        def algorithm_test(n):
            start = time.time()
            algo = AlgorithmChecker(algo_cls(embed_dim, self._t, max_control_cost, **{"seed": algo_seed, **algo_kwargs}))
            training_set = get_training_set(n)
            total_cost = self._fit_algo(system, algo, training_set, self._t, max_control_cost, fit_kwargs, act_kwargs, noisy)
            pred = algo.predict(test_set[:, 0], self._test_timesteps)
            error = system.calc_error(pred, test_set)

            ood_error = None
            if ood:
                ood_pred = algo.predict(ood_test_set[:, 0], self._test_timesteps)
                ood_error = system.calc_error(ood_pred, ood_test_set)
            end = time.time()
            duration = end - start

            ScalingChallenge._append_result(job_id, result, trial, n, latent_dim, embed_dim, self._t, total_cost, error, ood_error=ood_error, duration=duration)

            if ood:
                return ood_error, total_cost
            else:
                return error, total_cost
                        
        # Given training set size (ts) and window, figure out how many ts to sample. Instantiate algorithm, train, and test once per ts in window. Return median result from window.
        @functools.cache    #memoized
        def windowed_algorithm_test(n, half_window=0):
            """
            Helper function to instantiate and test an algorithm once for a given n. If window > 0, then we take the moving average of the error over the last window runs.

            param n: number of trajectories to train on
            param half_window: number of train sizes to the left of and right of n to take the moving average over
            """
            n = int(n // 1)  # make sure n is an integer
            half_window = int(half_window // 1)  # make sure window is an integer
            window_size = (half_window * 2 + 1)
            
            if half_window > 0:  # moving average/median window
                #window_range = list(range(int(max(self.n_min, n - window)), int(min(n + window + 1, self.n_max)))) #clip if beyond bounds
                
                #half-sample symmetric / reflect: when the windowed range is beyond bounds, extend the range by reflecting about the edge.
                #inspired by scipy.ndimage.median_filter 'reflect' mode
                window_range = list(range(n - half_window, n + half_window+1))
                while True:
                    reflected_range = [ self.n_min + (self.n_min-nn) if nn < self.n_min  else nn for nn in window_range  ]
                    reflected_range = [ self.n_max - (nn-self.n_max) if nn > self.n_max else nn for nn in window_range  ]
                    if reflected_range == window_range:
                        break
                    else:
                        window_range = reflected_range
                
                window_len = int(window_size * self.n_window_density)
                if len(window_range) > window_len:
                    step = int(len(window_range) // window_len)
                    window_range = window_range[::step]
                    if n not in window_range:
                        window_range.append(n)
                results_window += [algorithm_test(nn) for nn in window_range]
                error = np.median([r[0] for r in results_window])
                total_cost = np.median([r[1] for r in results_window])
            else:
                error, total_cost = algorithm_test(n)
            
            logging.debug(
                f"{job_id=} | {trial=}, {latent_dim=}, {embed_dim=}, {n=}, {window_size=}, t={self._t}, control_h={self._control_horizons}, {total_cost=}, {error=:0.3}, incl {ood=}, algo_seed={algo_seed}, sys_seed={system_seed}")
            return error, total_cost

        def search():
            # Search with exponential search until we find an n_left that is above target and an n_right that is below target
            # Then do binary search between n_left and n_right to find the left most n_curr that is below target

            # Always track our best answer
            best_answer = np.inf

            # Our starting point
            n = self.n_starts.get(latent_dim, self.n_min)

            # Check if this is our n_left or n_right
            error, _ = windowed_algorithm_test(n, half_window=self.n_half_window)
            n_left = n  # n_left is the largest n that is above target
            n_right = n  # n_right is the smallest n that is below target
            if error > self._target_error:
                # double n until we find an n that is below target
                while (n < self.n_max):
                    n = min(self.n_max, int(n * 2))
                    error, _ = windowed_algorithm_test(n, half_window=self.n_half_window)
                    if error > self._target_error:
                        # found a larger n that is below target
                        n_left = n
                    else:  # found an n that is below target
                        n_right = n
                        break  # do binary search
            else:
                # Below target so look for left boundary! halve n until we find an n that is above target
                while (n > self.n_min):
                    n = max(self.n_min, int(n // 2))
                    error, _ = windowed_algorithm_test(n, half_window=self.n_half_window)
                    if error > self._target_error:
                        n_left = n
                        break  # do binary search if n_left is above target
                    else:  # found a smaller n that is below target
                        n_right = n

            # could not find an n_left that is above target, so return best answer
            if n_right == self.n_min:
                return n_right

            # could not find an n_right that is below target, so return best answer
            if n_left == self.n_max:
                return np.inf

            # Binary Search Time!
            while True:
                n = math.ceil((n_left + n_right) / 2)
                # stop if n_upper and n_lower are within precision
                # n_precision is a percentage of n_curr so if 
                # n_precision is 0, we must converge to a specific n.
                if n_right - n_left <= max(self.n_precision * n, 1):
                    return n
                
                error, _ = windowed_algorithm_test(n, half_window=self.n_half_window)
                if error > self._target_error:
                    n_left = n
                else:
                    n_right = n

        # Do search
        n_target = search()

        #NOTE: result is appended to as a side effect of calling `algorithm_test()` inside of `windowed_algorithm_test()`inside of `search()` ;  
        data = pd.DataFrame(result)
        data['n_target'] = n_target
        data['target_error'] = self._target_error
        data['n_start'] = self.n_starts.get(latent_dim, 1)
        data['n_window'] = self.n_half_window * 2 + 1
        data['n_precision'] = self.n_precision
        data['n_max'] = self.n_max
        data['system_seed'] = system_seed
        data['algo_seed'] = algo_seed
        data['id'] = id

        return data

    def _generate_training_set(self, system, n, noisy, existing_set=None):
        if existing_set is None or n > len(existing_set):
            additional_set = self._gen_trainset(system, n - len(existing_set), self._t, noisy)
            return np.concatenate((existing_set, additional_set), axis=0) if existing_set is not None else additional_set
        return existing_set[:n]

    def _update_or_create_training_set(self, system, required_size, existing_set=None, noisy=False):
        current_size = len(existing_set) if existing_set is not None else 0
        if current_size < required_size:
            additional_data = self._gen_trainset(system, required_size - current_size, self._t, noisy)
            existing_set = np.concatenate((existing_set, additional_data), axis=0) if existing_set is not None else additional_data
        return existing_set

    @staticmethod
    def plot(data, target_error: float = None, show: bool = True, show_stats: bool = False):
        if not target_error:
            target_error = data["target_error"].unique()[0]

        title = "Fixed Error"
        if not data['ood_error'].isnull().any():
            title += ", OOD"
            ax = plot_target_error(data, "latent_dim", "n_target", ylabel=r'$n$', xlabel=r'$L$', error_col='ood_error')
        else:
            ax = plot_target_error(data, "latent_dim", "n_target", ylabel=r'$n$', xlabel=r'$L$')
        if target_error:
            title += f", target error={target_error}"

        if show_stats: #TODO add column names...
            # get the n_target for each latent_dim
            stats = data[['trial', 'latent_dim', 'embed_dim', 'n_target']]
            stats = stats.drop_duplicates()
            total = stats.groupby(['latent_dim', 'embed_dim'])['n_target'].count().reset_index(name="total")
            total.reset_index()
            stats = stats[stats['n_target'] > 0]
            stats = stats[stats['n_target'] != np.inf]
            stats = stats.groupby(['latent_dim', 'embed_dim'])['n_target'].count().reset_index(name="plotted")
            stats.reset_index()
            stats = pd.merge(total, stats, how="outer")
            print(stats)

        ax.set_title(title)
        ax.get_legend().remove()

        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return ax
