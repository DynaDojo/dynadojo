"""
Challenges
===========
Contains a `Challenge` class and several commonly used subclasses. The `Challenge` class is a stand-alone class that
can also be extended.
"""
import itertools
import logging
import math
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .abstractions import AbstractSystem, AbstractAlgorithm
from .wrappers import AlgorithmChecker, SystemChecker
from .utils.plotting import plot_target_error, plot_metric


class Challenge:
    def __init__(self,
                 N: list[int],
                 L: list[int],
                 E: list[int] | int | None,
                 t: int,
                 max_control_cost_per_dim: int,
                 control_horizons: int,
                 system_cls: type[AbstractSystem],
                 reps: int,
                 test_examples: int,
                 test_timesteps: int,
                 system_kwargs: dict | None = None,
                 verbose: bool = True,
                 #  save_class: bool = False,
                 ):
        """
        :param N: train sizes, (# of trajectories)
        :param L: latent dimensions
        :param E: embedded dimensions. Optional.
            If list, then evaluate iterates across embedded dimensions. (e >= l)
            If int, then evaluate uses a fixed embedded dimension. (E >= max(L))
            If None, then evaluate sets the embedded dimension equal to the latent dimension. (e = l)
        :param t: timesteps (length of a trajectory)
        :param max_control_cost_per_dim: max control cost per control trajectory
        :param control_horizons: number of times to generate training data with control
        :param system_cls: class constructor (NOT instance) for a concrete system
        :param reps: number of times to repeat each experiment
        :param test_examples: test size
        :param test_timesteps: test timesteps
        :param system_kwargs:
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
        self._reps = reps
        self._test_examples = test_examples
        self._test_timesteps = test_timesteps
        self._verbose = verbose

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
                 # Filters which reps and L to evaluate. If None, no filtering is performed.
                 # We recommend using these filters to parallelize evaluation across multiple machines, while retaining reproducibility.
                 reps_filter: list[int] = None,
                 L_filter: list[int] | None = None,
                 rep_l_filter: list[tuple[int, int]] | None = None,
                 ) -> pd.DataFrame:
        """
        Evaluates an algorithm class (NOT an instance) on a dynamical system over a set of experimental parameters.

        :param algo_cls: algo class to be evaluated
        :param algo_kwargs: kwargs to be passed to algo_cls
        :param fit_kwargs: kwargs to be passed to algo_cls.fit
        :param act_kwargs: kwargs to be passed to algo_cls.act
        :param ood: Boolean. If True, also test on out-distribution initial conditions for the test set. (For FixedError, search is performed on ood_error if ood=True.) Defaults to False.
        If False, generate out-of-distribution initial conditions for the test set.
        :param noisy: Boolean. If True, add noise to train set. Defaults to False. If False, no noise is added.
        :param id: algo ID associated with evaluation results in returned DataFrame
        :param num_parallel_cpu: number of cpus to use in parallel. Defaults to -1, which uses all available cpu.
        :param seed: to seed random number generator for seeding systems and algos. Defaults to None. Is overriden by seeds in system_kwargs or algo_kwargs.
        :param reps_filter: if provided, will only evaluate system_runs with the given rep_ids. Defaults to None, which evaluates all repetitions.
        :param L_filter: if provided, will only evaluate system_runs with the given latent dimensions. Defaults to None, which evaluates all latent dimensions.
        :param rep_l_filter: if provided, will only evaluate system_runs with the given (rep_id, latent_dim) pairs. Defaults to None, which evaluates all (rep_id, latent_dim) pairs.
        :
        return: a pandas DataFrame where each row is a algo_run result -- a algo trained and evaluated on a single system. (See algo_run() for more details.)
        """

        algo_kwargs = algo_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        act_kwargs = act_kwargs or {}

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

        # Handling which reps to evaluate.
        # First, making seeds for all reps
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        # We have to have the algo seeded so that it is the same for each iteration of training in system_run (which may vary/search through N)
        system_seeds = rng.integers(0, 2 ** 32, size=self._reps * len(self._L))
        algo_seeds = rng.integers(0, 2 ** 32, size=self._reps * len(self._L))

        # Second creating all system_run arguments (rep_id, latent_dim, embed_dim, system_seed, algo_seed)
        system_run_args = zip(itertools.product(range(self._reps), zip(self._L, E)), system_seeds, algo_seeds)
        # flatten system_args to a list of tuples
        system_run_args = [(r, l, e, system_seed, algo_seed) for (r, (l, e)), system_seed, algo_seed in
                           system_run_args]
        # Third, figuring out which reps to run based on specified subset of reps
        if reps_filter is not None and len(reps_filter) > 0:
            system_run_args = [args for args in system_run_args if args[0] in reps_filter]
        # Fourth, figuring out which systems to run based on specified subset of L
        if L_filter is not None and len(L_filter) > 0:
            system_run_args = [args for args in system_run_args if args[1] in L_filter]
        # Fifth, figuring out which systems to run based on specified subset of (rep_id, L)
        if rep_l_filter is not None and len(rep_l_filter) > 0:
            system_run_args = [args for args in system_run_args if (args[0], args[1]) in rep_l_filter]

        fixed_run_args = {
            # **kwargs, #ToDo: consider adding extra kwargs to pass to system_run
            "algo_cls": algo_cls,
            "algo_kwargs": algo_kwargs,
            "fit_kwargs": fit_kwargs,
            "act_kwargs": act_kwargs,
            "noisy": noisy,
            "test_ood": ood
        }

        if num_parallel_cpu == 0:
            logging.info(f"Running systems sequentially. {num_parallel_cpu=}")
            data = []
            for rep_id, l, e, system_seed, algo_seed in system_run_args:
                data.append(
                    self.system_run(rep_id, l, e, **fixed_run_args, system_seed=system_seed, algo_seed=algo_seed))

        else:
            logging.warning(f"Running systems in parallel. {num_parallel_cpu=}")
            # Run systems in parallel
            data = Parallel(n_jobs=num_parallel_cpu, timeout=1e6)(
                delayed(self.system_run)(rep_id, l, e, **fixed_run_args, system_seed=system_seed, algo_seed=algo_seed)
                for rep_id, l, e, system_seed, algo_seed in system_run_args)

        if data:
            data = pd.concat(data)
            data["id"] = id or next(self._id)
            data["control_horizon"] = self._control_horizons
        return data

    def _gen_trainset(self, system, n: int, timesteps: int, noisy=False):
        train_init_conds = system.make_init_conds(n)
        return system.make_data(train_init_conds, timesteps=timesteps, noisy=noisy)

    def _gen_testset(self, system, in_dist=True):
        test_init_conds = system.make_init_conds(self._test_examples, in_dist)
        return system.make_data(test_init_conds, timesteps=self._test_timesteps)

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
    def _append_result(result, rep_id, n, latent_dim, embed_dim, timesteps, total_cost, error, ood_error=None,
                       duration=None):
        result['rep'].append(rep_id)
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
            "rep",
            "latent_dim",
            "embed_dim",
            "timesteps",
            "n",
            "error",
            "ood_error",
            "total_cost",
            "duration"]}
        return result

    def system_run(self,
                   rep_id,
                   latent_dim,
                   embed_dim,
                   algo_cls: type[AbstractAlgorithm],
                   algo_kwargs: dict = None,
                   fit_kwargs: dict = None,
                   act_kwargs: dict = None,
                   noisy: bool = False,
                   test_ood: bool = False,
                   system_seed=None,
                   algo_seed=None,
                   **kwargs
                   ):
        """
        For a given system latent dimension and embedding dimension, instantiates system and for a specific N, evaluates the algo on the system (a algo_run).
        Across algo_runs, the algo is re-initialized with the same seed.
        Note that algo seed in algo_kwargs and system_seed in system_kwargs takes precedence over the seed passed to this function.
        """
        result = Challenge._init_result_dict()

        if embed_dim < latent_dim:
            return

        # Seed in system_kwargs takes precedence over the seed passed to this function.
        system = self._system_cls(latent_dim, embed_dim, **{"seed": system_seed, **self._system_kwargs})

        # Create all data
        test_set = self._gen_testset(system, in_dist=True)
        ood_test_set = self._gen_testset(system, in_dist=False)
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
            ood_error = None
            if test_ood:
                ood_pred = algo.predict(ood_test_set[:, 0], self._test_timesteps)
                ood_error = system.calc_error(ood_pred, ood_test_set)
            end = time.time()
            duration = end - start
            # TODO: fix logging? Should we use a logger?
            ood_error_str = f"{ood_error=:0.3}" if test_ood else "ood_error=NA"
            logging.debug(
                f"{rep_id=}, {latent_dim=}, {embed_dim=}, {n=}, t={self._t}, control_h={self._control_horizons}, {total_cost=}, {error=:0.3}, {ood_error_str},algo_seed={algo._seed}, sys_seed={system._seed}")
            Challenge._append_result(result, rep_id, n, latent_dim, embed_dim, self._t, total_cost, error,
                                     ood_error=ood_error, duration=duration)

        # On each subset of the training set, we retrain the algo from scratch (initialized with the same random seed).
        # If you don't, then # of training epochs will scale with N. This would confound the effect of training set size with training time.
        for n in self._N:
            algo_run(n)

        data = pd.DataFrame(result)
        data['system_seed'] = system_seed
        data['algo_seed'] = algo_seed
        return data


class FixedComplexity(Challenge):
    def __init__(self,
                 l: int,
                 t: int,
                 N: list[int],
                 system_cls: type[AbstractSystem],
                 reps: int,
                 test_examples: int,
                 test_timesteps: int,
                 e: int = None,
                 max_control_cost_per_dim: int = 1,
                 control_horizons: int = 0,
                 system_kwargs: dict = None,
                 verbose: bool = True):
        """
        Challenge where complexity is fixed, training set size is varied, and error is measured.

        :param l (int): latent dimension
        :param t (int): number of timesteps
        :param N (list[int]): list of training set sizes
        :param system_cls (type[AbstractSystem]): system class
        :param reps (int): number of repetitions
        :param test_examples (int): number of test examples
        :param test_timesteps (int): number of test timesteps
        :param e (int, optional): embedding dimension. Defaults to None.
        :param max_control_cost_per_dim (int, optional): maximum control cost per dimension. Defaults to 1.
        :param control_horizons (int, optional): number of control horizons. Defaults to 0.
        :param system_kwargs (dict, optional): system kwargs. Defaults to None.
        """
        L = [l]
        E = e
        super().__init__(N, L, E, t, max_control_cost_per_dim, control_horizons,
                         system_cls, reps, test_examples, test_timesteps, system_kwargs=system_kwargs, verbose=verbose)

    @staticmethod
    def plot(data, latent_dim: int = None, embedding_dim: int = None, show: bool = True, show_stats: bool = False):
        """
        Returns: matplotlib.axes.Axes object
        """
        if show_stats:
            # get the n_target for each latent_dim
            stats = data[['rep', 'latent_dim', 'embed_dim', 'n', 'error', 'ood_error']]
            stats = stats.drop_duplicates()
            total = stats.groupby(['latent_dim', 'embed_dim', 'n'])['rep'].count().reset_index(name="total")
            total.reset_index()
            stats = stats.dropna(subset=['error', 'ood_error'])
            stats = stats.groupby(['latent_dim', 'embed_dim', 'n'])['rep'].count().reset_index(name="plotted")
            stats.reset_index()
            stats = pd.merge(total, stats, how="outer")
            print(stats)
        data = data.dropna(subset=['error', 'ood_error'])
        if not latent_dim:
            latent_dim = data["latent_dim"].unique()[0]
        if not data['ood_error'].isnull().any():
            ax = plot_metric(data, "n", ["error", "ood_error"], xlabel=r'$n$', ylabel=r'$\mathcal{E}$',
                             errorbar=("pi", 50))
            ax.legend(title='Distribution')
        else:
            ax = plot_metric(data, "n", "error", xlabel=r'$n$', ylabel=r'$\mathcal{E}$', errorbar=("pi", 50))
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


class FixedTrainSize(Challenge):
    def __init__(self, n: int, L: list[int], E: list[int] | int | None, t: int,
                 max_control_cost_per_dim: int, control_horizons: int,
                 system_cls: type[AbstractSystem], reps: int, test_examples: int, test_timesteps: int,
                 system_kwargs: dict = None,
                 verbose: bool = True):
        """
        Challenge where the size of the training set is fixed, the complexity of the system is varied, and the error is measured.

        :param n: The size of the training set.
        :param L: The complexities of the system.
        :param E: The embedding dimensions of the system.
        :param t: The number of timesteps to simulate.
        :param max_control_cost_per_dim: The maximum control cost per dimension.
        :param control_horizons: The number of control horizons to consider.
        :param system_cls: The system class to use.
        :param reps: The number of repetitions to run.
        :param test_examples: The number of test examples to use.
        :param test_timesteps: The number of timesteps to simulate for the test examples.
        :param system_kwargs: The keyword arguments to pass to the system class.

        """
        N = [n]
        super().__init__(N, L, E, t, max_control_cost_per_dim, control_horizons,
                         system_cls, reps, test_examples, test_timesteps, system_kwargs=system_kwargs,
                         verbose=verbose)

    @staticmethod
    def plot(data: pd.DataFrame, n: int = None, show: bool = True, show_stats: bool = False, plot_ood=True, ax=None):
        """
        Returns: matplotlib.axes.Axes object
        """
        if not n:
            n = data["n"].unique()[0]
        if plot_ood and not data['ood_error'].isnull().any():
            ax = plot_metric(data, "latent_dim", ["error", "ood_error"], xlabel=r'$L$', log=True,
                             ylabel=r'$\mathcal{E}$', errorbar=("pi", 50))
            ax.legend(title='Distribution')
        else:
            ax = plot_metric(data, "latent_dim", "error", xlabel=r'$L$', log=True, ylabel=r'$\mathcal{E}$',
                             errorbar=("pi", 50))
            ax.get_legend().remove()
        title = "Fixed Train Size"
        if n:
            title += f", n={n}"
        ax.set_title(title)

        if show_stats:
            # get the n_target for each latent_dim
            stats = data[['rep', 'latent_dim', 'embed_dim']]
            stats = stats.drop_duplicates()
            stats = stats.groupby(['latent_dim', 'embed_dim'])['rep'].count().reset_index(name="total")
            stats.reset_index()
            print(stats)

        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return ax


class FixedError(Challenge):
    def __init__(self,
                 L: list[int],
                 t: int,
                 max_control_cost_per_dim: int,
                 control_horizons: int,
                 system_cls: type[AbstractSystem],
                 reps: int,
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
                 verbose: bool = True
                 ):
        """
        Challenge where the target error is fixed and the latent dimensionality is varied and the number of training samples to achieve the error is measured.  
        Performs a binary search over the number of training samples to find the minimum number of samples needed to achieve the target error rate.

        :param L: List of latent dimensions to test
        :param t: Number of timesteps of each training trajectory
        :param max_control_cost_per_dim: Maximum control cost per dimension
        :param control_horizons: Number of control horizons to test
        :param system_cls: System class to use
        :param reps: Number of repetitions to run for each latent dimension
        :param test_examples: Number of test examples to use
        :param test_timesteps: Number of timesteps of each test trajectory
        :param target_error: Target error to test
        :param E: List of embedding dimensions to test. If None, defaults to L.
        :param system_kwargs: Keyword arguments to pass to the system class
        :param n_precision: Uncertainty interval around number of training samples to achieve target error. (i.e. the true number of samples needed for the target error rate will be in [samples_needed - sample_precision, samples_needed + sample_precision]). If 0, we find the exact minimum number of samples needed to achieve the target error rate.
        :param n_window: Number of n to smooth over on left and right when calculating the error rate during search. If 0, no averaging is done.
        :param n_starts: List of starting points for the binary search over the number of training samples for each latent dim in L. len must equal len(L). If None, defaults to 1.
        :param n_max: Maximum number of training samples to use
        :param n_min: Minimum number of training samples to use
        :param n_window_density: Density of n to test during smoothing. 1 = test every n. 0.5 = test every other n. 0.25 = test every fourth n. etc.
        """
        assert (1 > n_precision >= 0), "Precision must be between [0 and 1)"
        assert (0 <= n_window), "Window size must be non-negative"
        assert (0 < n_window_density <= 1), "Window density must be between (0 and 1]"
        assert (n_max > n_min), "n_max must be greater than n_min"
        assert (n_min > 0), "n_min must be greater than 0"
        # ToDo: sort L and E if not sorted instead of throwing error.

        self.n_starts = {l: 1 for l in L}
        if n_starts:
            assert (len(L) == len(n_starts)), "if supplied, must provide a starting point for each latent dimension"
            self.n_starts = {l: n for l, n in zip(L, n_starts)}

        self.n_precision = n_precision
        self.n_window = n_window
        self.n_max = n_max
        self.n_min = n_min
        self._target_error = target_error
        self.n_window_density = n_window_density

        self.rep_id = None
        self.result = None
        self.n_needed = {}

        super().__init__([1], L, E, t, max_control_cost_per_dim, control_horizons, system_cls, reps, test_examples,
                         test_timesteps, system_kwargs=system_kwargs, verbose=verbose)

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
                 # Filters which reps and L to evaluate. If None, no filtering is performed.
                 # We recommend using these filters to parallelize evaluation across multiple machines, while retaining reproducibility.
                 reps_filter: list[int] = None,
                 L_filter: list[int] | None = None,
                 rep_l_filter: list[tuple[int, int]] | None = None,
                 ) -> pd.DataFrame:
        """
        Evaluates a model class (NOT an instance) on a dynamical system over a set of experimental parameters.

        :param algo_cls: model class to be evaluated
        :param algo_kwargs: kwargs to be passed to model_cls
        :param fit_kwargs: kwargs to be passed to model_cls.fit
        :param act_kwargs: kwargs to be passed to model_cls.act
        :param ood: Boolean. If True, also test on out-distribution initial conditions for the test set. (For FixedError, search is performed on ood_error if ood=True.) Defaults to False.
        If False, generate out-of-distribution initial conditions for the test set.
        :param noisy: Boolean. If True, add noise to train set. Defaults to False. If False, no noise is added.
        :param id: model ID associated with evaluation results in returned DataFrame
        :param num_parallel_cpu: number of cpus to use in parallel. Defaults to -1, which uses all available cpu.
        :param seed: to seed random number generator for seeding systems and models. Defaults to None. Is overriden by seeds in system_kwargs or model_kwargs.
        :param reps_filter: if provided, will only evaluate system_runs with the given rep_ids. Defaults to None, which evaluates all repetitions.
        :param L_filter: if provided, will only evaluate system_runs with the given latent dimensions. Defaults to None, which evaluates all latent dimensions.
        :param rep_l_filter: if provided, will only evaluate system_runs with the given (rep_id, latent_dim) pairs. Defaults to None, which evaluates all (rep_id, latent_dim) pairs.
        :
        return: a pandas DataFrame where each row is a model_run result -- a model trained and evaluated on a single system. (See model_run() for more details.)
        """

        results = super().evaluate(algo_cls, algo_kwargs, fit_kwargs, act_kwargs, ood, noisy, id, num_parallel_cpu,
                                   seed, reps_filter, L_filter, rep_l_filter)
        targets = results[['latent_dim', 'embed_dim', 'rep', 'n_target', 'model_seed', 'system_seed']].drop_duplicates()

        # TODO: use logger instead of print
        if self._verbose:
            for _, row in targets.iterrows():
                print(
                    f"!!! rep_id={row['rep']}, latent_dim={row['latent_dim']}, n_target={row['n_target']}, embed_dim={row['embed_dim']}, seed={row['system_seed']},{row['model_seed']} ")

        return results

    def system_run(self,
                   rep_id,
                   latent_dim,
                   embed_dim,
                   algo_cls: type[AbstractAlgorithm],
                   algo_kwargs: dict = None,
                   fit_kwargs: dict = None,
                   act_kwargs: dict = None,
                   noisy: bool = False,
                   test_ood: bool = False,
                   system_seed=None,
                   model_seed=None
                   ):
        """
        For a given system latent dimension and embedding dimension, instantiates system and evaluates a single trial of finding an 
        number of training examples that achieves the target error. 

        Note that model seed in model_kwargs and system_seed in system_kwargs takes precedence over the seed passed to this function.
        """
        result = Challenge._init_result_dict()

        if embed_dim < latent_dim:
            return

        # Seed in system_kwargs takes precedence over the seed passed to this function.
        system = self._system_cls(latent_dim, embed_dim, **{"seed": system_seed, **self._system_kwargs})
        system = SystemChecker(system)

        # generate test set and max control cost
        test_set = self._gen_testset(system, in_dist=True)
        ood_test_set = self._gen_testset(system, in_dist=False)
        max_control_cost = self._max_control_cost_per_dim * latent_dim

        # generating master training set which is used to generate training sets of different sizes
        master_training_set = None

        def get_training_set(n):
            """ 
            Helper function to get training set of size n. 
            If we have already generated enough trajectories, simply return a subset of the training set. 
            If not, generate more trajectories and add to training set.
            """
            nonlocal master_training_set
            if master_training_set is None:  # if we have not generated any trajectories yet
                master_training_set = self._gen_trainset(system, n, self._t, noisy)
            if n <= len(master_training_set):  # if we have already generated enough trajectories
                return master_training_set[:n]  # train on subset of training set
            else:  # add more trajectories to training set
                to_add = self._gen_trainset(system, n - len(master_training_set), self._t, noisy)
                master_training_set = np.concatenate((master_training_set, to_add), axis=0)
                return master_training_set[:n]

        # memoized run function which stores results in result dictionary and memoizes whether or not we have already run the model for a given n 
        memo = {}

        def model_run(n, window=0):
            """
            Helper function to instantiate and run a model once for a given n. If window > 0, then we take the moving average of the error over the last window runs. Results are stored in result dictionary and memoized.

            param n: number of trajectories to train on
            param window: number of runs to the left of and right of the current run to take the moving average over; ToDo: make this the absolute window size...lazy
            """
            n = int(n // 1)  # make sure n is an integer
            window = int(window // 1)  # make sure window is an integer

            def run_helper(n):  # for a given n, fit model and return error
                # check memo
                nonlocal memo
                if n in memo:
                    return memo[n]
                start = time.time()
                model = algo_cls(embed_dim, self._t, max_control_cost, **{"seed": model_seed, **algo_kwargs})
                training_set = get_training_set(n)
                total_cost = self._fit_model(system, model, training_set, self._t, max_control_cost, fit_kwargs,
                                             act_kwargs, noisy)
                pred = model.predict(test_set[:, 0], self._test_timesteps)
                error = system.calc_error(pred, test_set)
                ood_error = None
                if test_ood:
                    ood_pred = model.predict(ood_test_set[:, 0], self._test_timesteps)
                    ood_error = system.calc_error(ood_pred, ood_test_set)
                end = time.time()
                duration = end - start
                # append/memoize result
                Challenge._append_result(result, rep_id, n, latent_dim, embed_dim, self._t, total_cost, error,
                                         ood_error=ood_error, duration=duration)
                if test_ood:
                    memo[n] = (ood_error, total_cost)
                    return ood_error, total_cost
                else:
                    memo[n] = (error, total_cost)
                    return error, total_cost

            if window > 0:  # moving average/median
                window_range = list(range(int(max(self.n_min, n - window)), int(min(n + window + 1, self.n_max))))
                window_len = int((window * 2 + 1) * self.n_window_density)
                if len(window_range) > window_len:
                    step = int(len(window_range) // window_len)
                    window_range = window_range[::step]
                    if n not in window_range:
                        window_range.append(n)
                results_window = [run_helper(nn) for nn in window_range]
                error = np.median([r[0] for r in results_window])  # TODO: allow for choosing density of median points
                total_cost = np.median([r[1] for r in results_window])
            else:
                error, total_cost = run_helper(n)

            if self._verbose:
                # TODO: fix logging? Should we use a logger?
                print(
                    f"{rep_id=}, {latent_dim=}, {embed_dim=}, {n=}, {window=}, t={self._t}, control_h={self._control_horizons}, {total_cost=}, {error=:0.3}, {test_ood=}, model_seed={model_seed}, sys_seed={system._seed}")
            return error, total_cost

        # run model for different values of the number of trajectories n, searching for n needed to achieve the target error            
        def search():
            """
            TODO: Replaced this with search_simple. Remove this function?
            Helper function to search for the number of trajectories N needed to achieve the target error.
            Returns -1 if target error is never reached within n_max trajectories, np.inf if target error is stuck in a local minimum, and the number of trajectories N needed to achieve the target error otherwise.
            Assumption that the test error is a convex function of N. Does an exponential search to narrow down the range of N to search over, then does a binary search to find the number of trajectories N needed to achieve the target error.
            In the exponential search, we start with n_curr = 1 or n_starts[latent_dim] if it exists. We then double n_curr until we have found a range of n where the target error is achieved. We handle the case where we have overshot the N needed to achieve the target error by backing up (either by two steps or by halving n_curr) and restarting the exponential search with a smaller increment. We do this until we have found a range of n where the target error is achieved.
            """
            # Doing an exponential search.
            # Narrow down the range of N to search over, giving a left and right bound where the target error is achieved. 
            # If we have overshot the N needed to achieve the target error, back up and restart the exponential search with a smaller increment.
            # If we exceed the maximum number of trajectories, return -1. If we converge to a local minimum, return np.inf.
            # Assumption: the test error is a convex function of N.
            n_curr = self.n_starts.get(latent_dim,
                                       self.n_min)  # start with n_curr = 1 or n_starts[latent_dim] if it exists
            current_path = []
            error_prev = None
            increment = 1
            increment_max = self.n_max // 10  # TODO: what is a good value for this?
            multiplicative_factor = 2
            history = set()  # keep track of (n_curr, increment) pairs we have already tried to avoid infinite loops
            best_answer = np.inf
            while True:

                # Above n_max, we have exceeded the maximum number of trajectories, so we return -1
                if n_curr > self.n_max:
                    return -1  # target error is never reached within n_max trajectories

                # Below n_min, we do not have enough data to fit a model, so we return the best answer we have found so far
                if n_curr < self.n_min:
                    return best_answer

                # Cycle detection
                if (n_curr, increment) in history:  # cycle detected, return best answer
                    if self._verbose:
                        print(f"CYCLEEEEEEE {n_curr=}, {increment=}, {error_curr=}, {error_prev=}, {current_path=}")
                    return best_answer

                # Track paths we have already tried
                history.add((n_curr, increment))

                # Run model on n_curr, update our best answer if we have found a new best answer
                error_curr, total_cost = model_run(n_curr, window=self.n_window)
                if error_curr <= self._target_error and n_curr < best_answer:
                    best_answer = n_curr

                if len(current_path) > 0:  # If not start of a path
                    if error_curr > error_prev:  # Error is increasing
                        # Move backwards, because of convexity assumption
                        if len(current_path) > 1:  # Can move back twice
                            current_path.pop()
                            n_prev_prev = current_path.pop()
                            if n_curr - n_prev_prev <= max(self.n_precision * n_curr, 1):
                                # target error is never reached, minimum error is above threshold under parabolic assumption
                                # This ignores the possibility that the target error is reached within the precision (only possible if high curvature)
                                return np.inf
                            n_curr = n_prev_prev
                        else:  # or halve n_prev
                            n_curr = current_path.pop()
                            n_curr = max(1, n_curr // multiplicative_factor)
                        current_path = []
                        increment = 1
                        error_prev = None  # reset error_prev
                        continue
                    else:  # error is decreasing
                        if error_curr < self._target_error:  # found n_curr below target
                            break  # do binary search if n_curr is below target
                        else:
                            current_path.append(n_curr)
                            error_prev = error_curr
                            n_curr = n_curr + max(increment, n_curr * self.n_precision)
                            # scale increment, but not below 1 nor above increment_max
                            increment = min(max(1, math.ceil(increment * multiplicative_factor)), increment_max)
                            continue
                else:  # no n_prevs
                    if error_curr > self._target_error:
                        # move forward
                        current_path.append(n_curr)
                        error_prev = error_curr
                        # Increment n_curr by at least n_precision * n_curr
                        n_curr = n_curr + max(increment, n_curr * self.n_precision)
                        # scale increment, but not below 1 nor above increment_max
                        increment = min(max(1, math.ceil(increment * multiplicative_factor)), increment_max)
                        continue
                    else:  # error_curr < target
                        # move backward
                        n_curr = max(1, n_curr // multiplicative_factor)
                        continue

            # do binary search, assuming error is convex and that error(n_lower) > target and error(n_upper) < target. So we can always check the left side of the interval if target is between error_lower and error_curr, and the right side otherwise.
            # n_prevs[-1] is the last n_curr that is above target
            # n_curr is the first n_curr that is below target
            # binary search between n_prevs[-1] and n_curr
            n_lower = current_path[-1]
            error_lower, _ = model_run(n_lower, window=self.n_window)
            n_upper = n_curr
            error_upper, _ = model_run(n_upper, window=self.n_window)
            while True:
                n_curr = (n_lower + n_upper) // 2
                # stop if n_upper and n_lower are within precision
                if n_upper - n_lower <= max(self.n_precision, 1):
                    return math.ceil((n_lower + n_upper) / 2)

                error_curr, _ = model_run(n_curr, window=self.n_window)
                # find which side of the interval to keep
                # if target is between error_lower and error_curr, keep left side
                # if target is between error_curr and error_upper, keep right side
                if error_lower > self._target_error > error_curr:
                    n_upper = n_curr
                    error_upper = error_curr
                else:
                    n_lower = n_curr
                    error_lower = error_curr

        def search_simple():
            # Search with exponential search until we find an n_left that is above target and an n_right that is below target
            # Then do binary search between n_left and n_right to find the left most n_curr that is below target

            # Always track our best answer
            best_answer = np.inf

            # Our starting point
            n = self.n_starts.get(latent_dim, self.n_min)

            # Check if this is our n_left or n_right
            error, _ = model_run(n, window=self.n_window)
            n_left = n  # n_left is the largest n that is above target
            n_right = n  # n_right is the smallest n that is below target
            if error > self._target_error:
                # double n until we find an n that is below target
                while (n < self.n_max):
                    n = min(self.n_max, int(n * 2))
                    error, _ = model_run(n, window=self.n_window)
                    if error > self._target_error:
                        # found a larger n that is below target
                        n_left = n
                    else:  # found an n that is below target
                        n_right = n
                        break  # do binary search
            else:
                # Below target so look for left boundary!
                # halve n until we find an n that is above target
                while (n > self.n_min):
                    n = max(self.n_min, int(n // 2))
                    error, _ = model_run(n, window=self.n_window)
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
                n = math.ceil((n_left + n_right) // 2)
                # stop if n_upper and n_lower are within precision
                # n_precision is a percentage of n_curr so if 
                # n_precision is 0, we must converge to a specific n.
                if n_right - n_left <= self.n_precision * n:
                    return n

                error, _ = model_run(n, window=self.n_window)
                if error > self._target_error:
                    n_left = n
                else:
                    n_right = n

        # Run search
        n_target = search_simple()

        data = pd.DataFrame(result)
        data['n_target'] = n_target
        data['target_error'] = self._target_error
        data['n_start'] = self.n_starts.get(latent_dim, 1)
        data['n_window'] = self.n_window
        data['n_precision'] = self.n_precision
        data['n_max'] = self.n_max
        data['system_seed'] = system_seed
        data['model_seed'] = model_seed

        # note: result is appended to as a side effect of calling run() inside of search(); 
        # TODO: refactor to avoid side effects, make pure functions. 
        return data

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

        if show_stats:
            # get the n_target for each latent_dim
            stats = data[['rep', 'latent_dim', 'embed_dim', 'n_target']]
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
