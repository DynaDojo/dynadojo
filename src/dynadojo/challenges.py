import math
import time
import pandas as pd
import numpy as np

from .utils.plotting import plot_target_error, plot_metric
from .abstractions import Challenge, AbstractSystem, AbstractModel

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
    def plot(data, latent_dim:int=None, embedding_dim:int=None, show: bool=True):
        """
        Returns: matplotlib.axes.Axes object
        """
        if not latent_dim:
            latent_dim = data["latent_dim"].unique()[0]
        if not data['ood_error'].isnull().any():
            ax = plot_metric(data, "n", ["error", "ood_error"], xlabel=r'$n$', ylabel=r'$\mathcal{E}$', errorbar=("pi", 50))
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
                system_cls: type[AbstractSystem], reps: int, test_examples: int, test_timesteps: int, system_kwargs: dict = None,
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
    def plot(data: pd.DataFrame, n: int = None, show : bool =True):
        """
        Returns: matplotlib.axes.Axes object
        """
        if not n:
            n = data["n"].unique()[0]
        if not data['ood_error'].isnull().any():
            ax = plot_metric(data, "latent_dim", ["error", "ood_error"], xlabel=r'$L$', log=True, ylabel=r'$\mathcal{E}$', errorbar=("pi", 50))
            ax.legend(title='Distribution')
        else:
            ax = plot_metric(data, "latent_dim", "error", xlabel=r'$L$', log=True, ylabel=r'$\mathcal{E}$', errorbar=("pi", 50))
            ax.get_legend().remove()
        title = "Fixed Train Size"
        if n:
            title += f", n={n}"
        ax.set_title(title)
        
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
                n_precision: int = 5,
                n_starts: list[int] = None,
                n_window: int = 0,
                n_max=10000,
                n_window_density: float = 1.0, # 1 = include all points
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
        """
        assert (n_precision >= 0), "Precision must be non-negative"
        assert (L == sorted(L)), "Latent dimensions have to be sorted" #ToDo: is this necessary?
        #ToDo: sort L and E if not sorted instead of throwing error.

        self.n_starts = {l: 1 for l in L}
        if n_starts:
            assert (len(L) == len(n_starts)), "if supplied, must provide a starting point for each latent dimension"
            self.n_starts = {l: n for l, n in zip(L, n_starts)}

        self.n_precision = n_precision
        self.n_window = n_window
        self.n_max = n_max
        self._target_error = target_error
        self.n_window_density = n_window_density

        self.rep_id = None
        self.result = None
        self.n_needed = {}
        
        super().__init__([1], L, E, t, max_control_cost_per_dim, control_horizons, system_cls, reps, test_examples, test_timesteps, system_kwargs=system_kwargs, verbose=verbose)

    def evaluate(self, 
                model_cls: type[AbstractModel],
                model_kwargs: dict | None = None,
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

        :param model_cls: model class to be evaluated
        :param model_kwargs: kwargs to be passed to model_cls
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
        
        results = super().evaluate(model_cls, model_kwargs, fit_kwargs, act_kwargs, ood, noisy, id, num_parallel_cpu, seed, reps_filter, L_filter, rep_l_filter)
        targets = results[['latent_dim', 'embed_dim', 'rep',  'n_target', 'model_seed', 'system_seed']].drop_duplicates()

        #TODO: use logger instead of print
        if self._verbose:
            for _, row in targets.iterrows():
                print(f"!!! rep_id={row['rep']}, latent_dim={row['latent_dim']}, n_target={row['n_target']}, embed_dim={row['embed_dim']}, seed={row['system_seed']},{row['model_seed']} ")
            
        return results

    def system_run(self, 
                    rep_id, 
                    latent_dim, 
                    embed_dim, 
                    model_cls : type[AbstractModel],
                    model_kwargs : dict = None,
                    fit_kwargs : dict = None,
                    act_kwargs : dict = None,
                    noisy : bool = False,
                    test_ood : bool = False,
                    system_seed=None, 
                    model_seed=None
                    ):
        """
        For a given system latent dimension and embedding dimension, instantiates system and evaluates reps of
        iterating over the number of trajectories N.

        Note that model seed in model_kwargs and system_seed in system_kwargs takes precedence over the seed passed to this function.
        """
        result = Challenge._init_result_dict()

        if embed_dim < latent_dim:
            return

        # Seed in system_kwargs takes precedence over the seed passed to this function.
        system = self._system_cls(latent_dim, embed_dim, **{"seed":system_seed, **self._system_kwargs})
        
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
            if master_training_set is None: #if we have not generated any trajectories yet
                master_training_set = self._gen_trainset(system, n, self._t, noisy)
            if n <= len(master_training_set): # if we have already generated enough trajectories
                return master_training_set[:n] #train on subset of training set
            else: #add more trajectories to training set 
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
            n = int(n // 1) #make sure n is an integer
            window = int(window // 1) #make sure window is an integer

            def run_helper(n):  #for a given n, fit model and return error
                #check memo
                nonlocal memo
                if n in memo:
                    return memo[n]
                start = time.time()
                model = model_cls(embed_dim, self._t, max_control_cost, **{"seed": model_seed, **model_kwargs})
                training_set = get_training_set(n)
                total_cost = self._fit_model(system, model, training_set, self._t, max_control_cost, fit_kwargs, act_kwargs, noisy)
                pred = model.predict_wrapper(test_set[:, 0], self._test_timesteps)
                error = system.calc_error_wrapper(pred, test_set)
                ood_error = None
                if test_ood:
                    ood_pred = model.predict_wrapper(ood_test_set[:, 0], self._test_timesteps)
                    ood_error = system.calc_error_wrapper(ood_pred, ood_test_set)
                end = time.time()
                duration = end - start
                #append/memoize result
                Challenge._append_result(result, rep_id, n, latent_dim, embed_dim, self._t, total_cost, error, ood_error=ood_error, duration=duration)
                if test_ood:
                    memo[n] = (ood_error, total_cost)
                    return ood_error, total_cost
                else:
                    memo[n] = (error, total_cost)
                    return error, total_cost
            if window > 0: #moving average/median 
                window_range = list(range(max(1,n - window), min(n+window+1, self.n_max)))
                window_len = int((window*2 + 1) * self.n_window_density)
                if len(window_range) > window_len:
                    step = int(len(window_range) // window_len)
                    window_range = window_range[::step]
                    if n not in window_range:
                        window_range.append(n)
                results_window = [run_helper(nn) for nn in window_range]
                error = np.median([r[0] for r in results_window]) #TODO: allow for choosing density of median points
                total_cost = np.median([r[1] for r in results_window])
            else:
                error, total_cost = run_helper(n)
            
            if self._verbose:
                #TODO: fix logging? Should we use a logger?
                print(f"{rep_id=}, {latent_dim=}, {embed_dim=}, {n=}, {window=}, t={self._t}, control_h={self._control_horizons}, {total_cost=}, {error=:0.3}, {test_ood=}, model_seed={model_seed}, sys_seed={system._seed}")
            return error, total_cost
            
        # run model for different values of the number of trajectories n, searching for n needed to achieve the target error            
        def search():
            """
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
            n_curr = self.n_starts.get(latent_dim, 1) # start with n_curr = 1 or n_starts[latent_dim] if it exists
            n_prevs = []
            error_prev = None
            increment = 1
            increment_max = self.n_max//10 #TODO: what is a good value for this?
            multiplicative_factor = 2
            history = set() #keep track of (n_curr, increment) pairs we have already tried to avoid infinite loops
            best_answer = np.inf
            while True:
                if n_curr > self.n_max:
                    return -1 # target error is never reached within n_max trajectories
                
                if (n_curr, increment) in history: #cycle detected, return best answer
                    if self._verbose:
                        print(f"CYCLEEEEEEE {n_curr=}, {increment=}, {error_curr=}, {error_prev=}, {n_prevs=}")
                    return best_answer

                history.add((n_curr, increment))

                # run model on n_curr, update our best answer if we have found a new best answer
                error_curr, total_cost = model_run(n_curr, window=self.n_window)
                if error_curr <= self._target_error and n_curr < best_answer:
                    best_answer = n_curr

                if len(n_prevs) > 0:
                    if error_curr > error_prev: # error is increasing
                        # Move backwards
                        if len(n_prevs) > 1: # Back twice
                            n_prevs.pop()
                            n_prev_prev = n_prevs.pop()
                            if n_curr - n_prev_prev <= max(self.n_precision * n_curr, 1):
                                # target error is never reached, minimum error is above threshold under parabolic assumption
                                # This ignores the possibility that the target error is reached within the precision (only possible if high curvature)
                                return np.inf 
                            n_curr = n_prev_prev
                        else: # or halve n_prev
                            n_curr =  n_prevs.pop()

                            # CYCLES will handle this case
                            # if n_curr < self.n_precision: #can't move backwards anymore
                            #     if error_curr > self._target_error:
                            #         return np.inf
                            #     return n_curr #no more backing up can be done
                            
                            n_curr = max(1, n_curr//multiplicative_factor)
                        n_prevs = []
                        increment = 1
                        error_prev = None # reset error_prev
                    else: # error is decreasing
                        if error_curr < self._target_error: # found n_curr below target
                            break # do binary search if n_curr is below target
                        else:
                            n_prevs.append(n_curr)
                            error_prev = error_curr
                            n_curr = n_curr + max(increment, n_curr * self.n_precision)
                            # scale increment, but not below 1 nor above increment_max
                            increment = min(max(1, math.ceil(increment * multiplicative_factor)), increment_max)
                else: #no n_prevs
                    if error_curr > self._target_error:
                        #move forward
                        n_prevs.append(n_curr)
                        error_prev = error_curr
                        n_curr = n_curr +  max(increment, n_curr * self.n_precision) 
                        increment = min(max(1, math.ceil(increment * multiplicative_factor)), increment_max)
                    else: # error_curr < target
                        
                        # CYCLES will handle this case
                        # if n_curr < self.n_precision + 1:
                        #     return n_curr #no more backing up can be done
                        
                        #move backward
                        n_curr = max(1, n_curr//multiplicative_factor)


            # do binary search, assuming error is convex and that error(n_lower) > target and error(n_upper) < target. So we can always check the left side of the interval if target is between error_lower and error_curr, and the right side otherwise.
            # n_prevs[-1] is the last n_curr that is above target
            # n_curr is the first n_curr that is below target
            # binary search between n_prevs[-1] and n_curr
            n_lower = n_prevs[-1] 
            error_lower, _ = model_run(n_lower, window=self.n_window)
            n_upper = n_curr
            error_upper, _ = model_run(n_upper, window=self.n_window)
            while True:
                n_curr = (n_lower + n_upper)//2
                # stop if n_upper and n_lower are within precision
                if n_upper - n_lower <= max(self.n_precision, 1):
                    return math.ceil((n_lower + n_upper)/2)

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

        # Run search    
        n_target = search()
        
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
    def plot(data, target_error:float=None, show:bool=True, show_stats:bool=False):
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
            stats = data[['rep', 'latent_dim', 'embed_dim','n_target']]
            stats = stats.drop_duplicates()
            total = stats.groupby(['latent_dim', 'embed_dim'])['n_target'].count().reset_index(name="total")
            total.reset_index()
            stats = stats[stats['n_target'] > 0 ]
            stats = stats[stats['n_target'] != np.inf]
            stats = stats.groupby(['latent_dim', 'embed_dim'])['n_target'].count().reset_index(name="plotted")
            stats.reset_index()
            stats = pd.merge(total, stats,how="outer")
            print(stats)

        ax.set_title(title)
        ax.get_legend().remove()

        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return ax


