import itertools
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


class AbstractModel(ABC):
    def __init__(self, embed_dim: int, timesteps: int, max_control_cost: float, **kwargs):
        """
        Base class for all models. Your models should subclass this class.

        :param embed_dim: embedded dimension of the dynamics
        :param timesteps: timesteps per training trajectory (per action horizon)
        :param max_control_cost: maximum control cost per trajectory (per action horizon)
        :param kwargs:
        """
        self._embed_dim = embed_dim
        # NOTE: this is the timesteps of the training data; NOT the predicted trajectories
        self._timesteps = timesteps
        self._max_control_cost = max_control_cost

    @abstractmethod
    def fit(self, x: np.ndarray, **kwargs) -> None:
        """
        Trains the model. Your models must implement this method.

        :param x: (n, timesteps, embed_dim) trajectories tensor
        :param kwargs:
        :return: None
        """
        raise NotImplementedError

    def act(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Determines the control for each action horizon. Your models should override this method if they use
        non-trivial control.

        :param x: (n, timesteps, embed_dim) trajectories tensor
        :param kwargs:
        :return: (n, timesteps, embed_dim) controls tensor
        """
        return np.zeros_like(x)

    def act_wrapper(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Wrapper for act() called in evaluate(). Verifies control tensor is the right shape. You should NOT override this.

        :param x: (n, timesteps, embed_dim) trajectories tensor
        :param kwargs:
        :return: (n, timesteps, embed_dim) controls tensor
        """
        control = self.act(x, **kwargs)
        assert control.shape == x.shape
        return control

    @abstractmethod
    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        """
        Predict how initial conditions matrix evolves over a given number of timesteps. Your models must implement this method.

        NOTE: The timesteps argument can differ from the ._timesteps attribute. This allows model to train on data
        that is shorter/longer than trajectories in the test set.

        NOTE: The first coordinate of each trajectory should match the initial condition.

        :param x0: (n, embed_dim) initial conditions matrix
        :param timesteps: timesteps per predicted trajectory
        :param kwargs:
        :return: (n, timesteps, embed_dim) trajectories tensor
        """
        raise NotImplementedError

    def predict_wrapper(self, x0: np.ndarray, timesteps, **kwargs) -> np.ndarray:
        """
        Wrapper for predict() called in evaluate(). Verifies predicted trajectories tensor has the right shape.
        You should NOT override this.

        NOTE: Does not enforce that the first coordinate of each trajectory is the same as the initial condition. This
        allows DynaDojo to handle models that completely mispredict trajectory evolution.

        :param x0: (n, embed_dim) initial conditions matrix
        :param timesteps: timesteps per predicted trajectory
        :param kwargs:
        :return: (n, timesteps, embed_dim) trajectories tensor
        """
        pred = self.predict(x0, timesteps, **kwargs)
        n = x0.shape[0]
        assert pred.shape == (n, timesteps, self._embed_dim)
        return pred


class AbstractSystem(ABC):
    def __init__(self, latent_dim, embed_dim):
        """
        Base class for all systems. Your systems should subclass this class.

        NOTE: The reason we use properties and setter methods for ._latent_dim and ._embed_dim is to allow
        systems to maintain information through parameter shifts. See LDSSystem in './systems/lds.py' for a principled
        usage example of the setter methods.
        """
        self._latent_dim = latent_dim
        self._embed_dim = embed_dim

    @property
    def latent_dim(self):
        return self._latent_dim

    @property
    def embed_dim(self):
        return self._embed_dim

    @latent_dim.setter
    def latent_dim(self, value):
        self._latent_dim = value

    @embed_dim.setter
    def embed_dim(self, value):
        self._embed_dim = value

    @abstractmethod
    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        """
        Abstract method to generate initial conditions. Your system should override this method.

        NOTE: Systems developers determine what counts as in vs out-of-distribution. DynaDojo doesn't provide
        any verification that this distinction makes sense or even exists.

        :param n: number of initial conditions
        :param in_dist: Boolean. If True, generate in-distribution initial conditions. Defaults to True. If False,
        generate out-of-distribution initial conditions.
        :return: (n, embed_dim) initial conditions matrix
        """
        raise NotImplementedError

    def make_init_conds_wrapper(self, n: int, in_dist=True):
        """
        Wrapper for make_init_conds() called in evaluate(). Verifies initial condition matrix is the right shape.
        You should NOT override this.

        :param n: number of initial conditions
        :param in_dist: Boolean. If True, generate in-distribution initial conditions. Defaults to True. If False,
        generate out-of-distribution initial conditions.
        :return: (n, embed_dim) initial conditions matrix
        """
        init_conds = self.make_init_conds(n, in_dist)
        assert init_conds.shape == (n, self.embed_dim)
        return init_conds

    @abstractmethod
    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        """
        Abstract method that makes trajectories from initial conditions. Your system must override this method.

        :param init_conds: (n, embed_dim) initial conditions matrix
        :param control: (n, timesteps, embed_dim) controls tensor
        :param timesteps: timesteps per training trajectory (per action horizon)
        :param noisy: Boolean. If True, add noise to trajectories. Defaults to False. If False, no noise is added.
        :return: (n, timesteps, embed_dim) trajectories tensor
        """
        raise NotImplementedError

    def make_data_wrapper(self, init_conds: np.ndarray, control: np.ndarray = None, timesteps: int = 1,
                          noisy=False) -> np.ndarray:
        """
        Wraps make_data(). Checks that trajectories tensor has the proper shape. You should NOT override this.

        :param init_conds: (n, embed_dim) initial conditions matrix
        :param control: (n, timesteps, embed_dim) controls tensor
        :param timesteps: timesteps per training trajectory (per action horizon)
        :param noisy: Boolean. If True, add noise to trajectories. Defaults to False. If False, no noise is added.
        :return: (n, timesteps, embed_dim) trajectories tensor
        """
        assert timesteps > 0
        assert init_conds.ndim == 2 and init_conds.shape[1] == self.embed_dim
        n = init_conds.shape[0]
        if control is None:
            control = np.zeros((n, timesteps, self.embed_dim))
        assert control.shape == (n, timesteps, self.embed_dim)
        data = self.make_data(init_conds=init_conds,
                              control=control, timesteps=timesteps, noisy=noisy)
        assert data.shape == (n, timesteps, self.embed_dim)
        return data

    @abstractmethod
    def calc_error(self, x, y) -> float:
        """
        Calculates the error between two tensors of trajectories. Your systems must implement this.

        :param x: (n, timesteps, embed_dim) trajectories tensor
        :param y: (n, timesteps, embed_dim) trajectories tensor
        :return: Float. The error between x and y.
        """
        raise NotImplementedError

    def calc_error_wrapper(self, x, y) -> float:
        """
        Wraps calc_error. Checks that calc_error is called with properly-shaped x and y.
        Your systems should NOT override this.

        :param x: (n, timesteps, embed_dim) trajectories tensor
        :param y: (n, timesteps, embed_dim) trajectories tensor
        :return: Float. The error between x and y.
        """
        assert x.shape == y.shape
        return self.calc_error(x, y)

    @abstractmethod
    def calc_control_cost(self, control: np.ndarray) -> np.ndarray:
        """
        Calculate the control cost for each control TRAJECTORY (i.e., calculates the costs for every
        control matrix, not for the whole tensor). Your systems must implement this.

        :param control: (n, timesteps, embed_dim) controls tensor
        :return: (n,) control costs vector
        """
        raise NotImplementedError

    def calc_control_cost_wrapper(self, control: np.ndarray) -> np.ndarray:
        """
        Wraps calc_control_cost(). Your systems should NOT override this.

        :param control: (n, timesteps, embed_dim) controls tensor
        :return: (n,) control costs vector
        """
        assert control.shape[2] == self.embed_dim and control.ndim == 3
        cost = self.calc_control_cost(control)
        assert cost.shape == (len(control),)
        return cost


class Challenge:
    def __init__(self,
                 N: list[int],
                 L: list[int],
                 E: list[int] | int | None,
                 T: list[int],
                 max_control_cost_per_dim: int,
                 control_horizons: int,
                 system_cls: type[AbstractSystem],
                 reps: int,
                 test_examples: int,
                 test_timesteps: int,
                 system_kwargs: dict | None = None,
                 ):
        """
        :param N: train sizes
        :param L: latent dimensions
        :param E: embedded dimensions. Optional. If list, then evaluate iterates across embedded dimensions.
        If int, then evaluate uses a fixed embedded dimension. If None, then evaluate sets the embedded dimension
        equal to the latent dimension.
        :param T: timesteps
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
        self._T = T
        self._max_control_cost_per_dim = max_control_cost_per_dim
        self._system_cls = system_cls
        self._system_kwargs = system_kwargs or {}
        self._control_horizons = control_horizons
        self._reps = reps
        self._test_examples = test_examples
        self._test_timesteps = test_timesteps

    def evaluate(self,
                 model_cls: type[AbstractModel],
                 model_kwargs: dict | None = None,
                 fit_kwargs: dict | None = None,
                 act_kwargs: dict | None = None,
                 in_dist=True,
                 noisy=False,
                 id=None,
                 ) -> pd.DataFrame:
        """
        Evaluates a model class (NOT an instance) on a dynamical system over a set of experimental parameters.

        :param model_cls: model class to be evaluated
        :param model_kwargs:
        :param fit_kwargs:
        :param act_kwargs:
        :param in_dist: Boolean. If True, generate in-distribution initial conditions for the test set. Defaults to True.
        If False, generate out-of-distribution initial conditions for the test set.
        :param noisy: Boolean. If True, add noise to train set. Defaults to False. If False, no noise is added.
        :param id: model ID associated with evaluation results in returned DataFrame
        :return: a pandas DataFrame with experimental results
        """

        model_kwargs = model_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        act_kwargs = act_kwargs or {}

        def do_rep1(rep_id):
            """
            Handles case when E is an array
            """
            result = {k: [] for k in ["rep", "n", "latent_dim",
                                      "embed_dim", "timesteps", "control_horizons", "error", "total_cost"]}
            system = None
            for n, d, timesteps in itertools.product(self._N, zip(self._L, self._E), self._T):
                latent_dim, embed_dim = d
                if embed_dim < latent_dim:
                    return
                system = self._set_system(system, latent_dim, embed_dim)
                self._do_rep(rep_id, id, result, system, n, latent_dim, embed_dim, timesteps, model_cls, model_kwargs, fit_kwargs,
                             act_kwargs, in_dist, noisy)
            return pd.DataFrame(result)

        def do_rep2(rep_id):
            """
            Handles case when E is a None
            """
            result = {k: [] for k in ["rep", "n", "latent_dim",
                                      "embed_dim", "timesteps", "control_horizons", "error", "total_cost"]}
            system = None
            for n, latent_dim, timesteps in itertools.product(self._N, self._L, self._T):
                embed_dim = latent_dim
                system = self._set_system(system, latent_dim, embed_dim)
                self._do_rep(rep_id, id, result, system, n, latent_dim, embed_dim, timesteps, model_cls, model_kwargs, fit_kwargs,
                             act_kwargs, in_dist, noisy)
            return pd.DataFrame(result)

        def do_rep3(rep_id):
            """
            Handles case when E is a constant
            """
            result = {k: [] for k in ["rep", "n", "latent_dim",
                                      "embed_dim", "timesteps", "control_horizons", "error", "total_cost"]}
            system = None
            for n, latent_dim, timesteps in itertools.product(self._N, self._L, self._T):
                embed_dim = self._E
                if embed_dim < latent_dim:
                    return
                system = self._set_system(system, latent_dim, embed_dim)
                self._do_rep(rep_id, id, result, system, n, latent_dim, embed_dim, timesteps, model_cls, model_kwargs, fit_kwargs,
                             act_kwargs, in_dist, noisy)
            return pd.DataFrame(result)

        if isinstance(self._E, list):
            do_rep = do_rep1
        elif self._E is None:
            do_rep = do_rep2
        elif isinstance(self._E, int):
            do_rep = do_rep3
        else:
            raise TypeError("E must of type List[int], int, or None.")

        data = Parallel(n_jobs=4, timeout=1e6)(delayed(do_rep)(rep_id)
                                               for rep_id in range(self._reps))
        data = pd.concat(data)
        data["id"] = id or next(self._id)
        return data

    def _set_system(self, system, latent_dim, embed_dim):
        if system is None:
            system = self._system_cls(
                latent_dim, embed_dim, **self._system_kwargs)
        if latent_dim != system.latent_dim:
            system.latent_dim = latent_dim
        if embed_dim != system.embed_dim:
            system.embed_dim = embed_dim
        return system

    def _gen_trainset(self, system, n: int, timesteps: int, noisy=False):
        train_init_conds = system.make_init_conds_wrapper(n)
        return system.make_data_wrapper(train_init_conds, timesteps=timesteps, noisy=noisy)

    def _gen_testset(self, system, in_dist=True):
        test_init_conds = system.make_init_conds_wrapper(self._test_examples, in_dist)
        return system.make_data_wrapper(test_init_conds, timesteps=self._test_timesteps)

    def _fit_model(self, system, model, x: np.ndarray, timesteps: int,  max_control_cost: int,  fit_kwargs: dict = None,
                   act_kwargs: dict = None, noisy=False) -> int:
        total_cost = 0
        model.fit(x, **fit_kwargs)

        for _ in range(self._control_horizons):
            control = model.act_wrapper(x, **act_kwargs)
            cost = system.calc_control_cost_wrapper(control)
            total_cost += cost
            assert np.all(cost <= max_control_cost), "Control cost exceeded!"
            x = system.make_data_wrapper(
                init_conds=x[:, -1], control=control, timesteps=timesteps, noisy=noisy)
            model.fit(x, **fit_kwargs)

        return total_cost

    def _append_result(self, result, rep_id, n, latent_dim, embed_dim, timesteps, error, total_cost):
        result['rep'].append(rep_id)
        result['n'].append(n)
        result['latent_dim'].append(latent_dim)
        result['embed_dim'].append(embed_dim)
        result['timesteps'].append(timesteps)
        result['control_horizons'].append(self._control_horizons)
        result['error'].append(error)
        result['total_cost'].append(total_cost)

    def _do_rep(self,
                rep_id: int,
                id: str | int,
                result: dict,
                system: AbstractSystem,
                n: int,
                latent_dim: int,
                embed_dim: int,
                timesteps: int,
                model_cls: type[AbstractModel],
                model_kwargs: dict = None,
                fit_kwargs: dict = None,
                act_kwargs: dict = None,
                in_dist=True,
                noisy=False,
                ):
        max_control_cost = self._max_control_cost_per_dim * latent_dim
        print(f"{n=}, {latent_dim=}, {embed_dim=}, {timesteps=}, control_horizons={self._control_horizons}, { rep_id=}, {id=}")

        # Create model and data
        model = model_cls(embed_dim, timesteps, max_control_cost, **model_kwargs)
        x = self._gen_trainset(system, n, timesteps, noisy)
        total_cost = self._fit_model(system, model, x, timesteps, max_control_cost, fit_kwargs, act_kwargs, noisy)
        test = self._gen_testset(system, in_dist)
        pred = model.predict_wrapper(test[:, 0], self._test_timesteps)
        error = system.calc_error_wrapper(pred, test)
        self._append_result(result, rep_id, n, latent_dim, embed_dim, timesteps, error, total_cost)
