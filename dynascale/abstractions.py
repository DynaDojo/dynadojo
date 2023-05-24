import itertools
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


class Model(ABC):
    def __init__(self, embed_dim: int, timesteps: int, max_control_cost: float, **kwargs):
        """
        Base class for all models. Your models should subclass this class.

        :param embed_dim: embedded dimension of the dynamics
        :param timesteps: timesteps per training trajectory (per action horizon)
        :param max_control_cost: maximum control cost per trajectory (per action horizon)
        :param kwargs:
        """
        self._embed_dim = embed_dim
        self._timesteps = timesteps  # NOTE: this is the timesteps of the training data; NOT the predicted trajectories
        self._max_control_cost = max_control_cost

    @abstractmethod
    def fit(self, x: np.ndarray, **kwargs):
        """
        Trains the model. Your models must implement this method.

        :param x: (n, timesteps, embed_dim) a tensor of trajectories
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def _act(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Driver for act(). Your models should overload this method if they use non-trivial control.

        :param x: (n, timesteps, embed_dim) a tensor of trajectories
        :param kwargs:
        :return: (n, timesteps, embed_dim) a tensor of controls
        """
        return np.zeros_like(x)

    def act(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        # TODO: fill out

        :param x: (n, timesteps, embed_dim) a tensor of trajectories
        :param kwargs:
        :return: (n, timesteps, embed_dim) a tensor of controls
        """
        control = self._act(x, **kwargs)
        assert control.shape == x.shape
        return control

    @abstractmethod
    def _predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        """
        Driver for predict(). Your models must implement this method.

        The timesteps argument can differ from the ._timesteps attribute. Allows models to predict trajectories
        with more/less timesteps than their training data.

        :param x0: (n, embed_dim) a matrix of initial conditions
        :param timesteps: timesteps per predicted trajectory
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def predict(self, x0: np.ndarray, timesteps, **kwargs) -> np.ndarray:
        """

        :param x0: (n, embed_dim) a matrix of initial conditions
        :param timesteps: timesteps per predicted trajectory
        :param kwargs:
        :return:
        """
        pred = self._predict(x0, timesteps, **kwargs)
        n = x0.shape[0]
        assert pred.shape == (n, timesteps, self._embed_dim)
        return pred


class System(ABC):
    def __init__(self, latent_dim, embed_dim):
        """
        Base class for all systems. Your systems should subclass this class.

        :param latent_dim: # TODO
        :param embed_dim:  # TODO
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
    def _make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        raise NotImplementedError

    def make_init_conds(self, n: int, in_dist=True):
        init_conds = self._make_init_conds(n, in_dist)
        assert init_conds.shape == (n, self.embed_dim)
        return init_conds

    @abstractmethod
    def _make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        raise NotImplementedError

    def make_data(self, init_conds: np.ndarray, control: np.ndarray = None, timesteps: int = 1,
                  noisy=False) -> np.ndarray:
        assert timesteps > 0
        assert init_conds.ndim == 2 and init_conds.shape[1] == self.embed_dim
        n = init_conds.shape[0]
        if control is None:
            control = np.zeros((n, timesteps, self.embed_dim))
        assert control.shape == (n, timesteps, self.embed_dim)
        data = self._make_data(init_conds=init_conds, control=control, timesteps=timesteps, noisy=noisy)
        assert data.shape == (n, timesteps, self.embed_dim)
        return data

    @abstractmethod
    def _calc_loss(self, x, y) -> float:
        raise NotImplementedError

    def calc_loss(self, x, y) -> float:
        assert x.shape == y.shape
        return self._calc_loss(x, y)

    @abstractmethod
    def _calc_control_cost(self, control: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def calc_control_cost(self, control: np.ndarray) -> np.ndarray:
        assert control.shape[2] == self.embed_dim and control.ndim == 3
        cost = self._calc_control_cost(control)
        assert cost.shape == (len(control),)
        return cost


class Task:
    def __init__(self,
                 N: list[int],
                 L: list[int],
                 E: list[int],
                 T: list[int],
                 max_control_cost_per_dim: int,
                 control_horizons: int,
                 system_cls: type[System],
                 reps: int,
                 test_examples: int,
                 test_timesteps: int,
                 system_kwargs: dict = None,
                 ):
        assert control_horizons > 0

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
            for n, latent_dim, embed_dim, timesteps in itertools.product(self._N, self._L, self._E, self._T):
                if embed_dim < latent_dim:
                    continue
                if system is None:
                    system = self._system_cls(latent_dim, embed_dim, **self._system_kwargs)
                if latent_dim != system.latent_dim:
                    system.latent_dim = latent_dim
                if embed_dim != system.embed_dim:
                    system.embed_dim = embed_dim

                max_control_cost = self._max_control_cost_per_dim * latent_dim
                print(f"{n=}, {latent_dim=}, {embed_dim=}, {timesteps=}")

                # Create model and data
                model = model_cls(embed_dim, timesteps, max_control_cost, **model_kwargs)
                train_init_conds = system.make_init_conds(n)

                x = system.make_data(train_init_conds, timesteps=timesteps, noisy=noisy)
                total_cost = 0
                model.fit(x, **fit_kwargs)

                # change to generate init conds out of for loop
                for _ in range(self._control_horizons):
                    control = model.act(x, **act_kwargs)
                    cost = system.calc_control_cost(control)
                    total_cost += cost
                    assert np.all(cost <= max_control_cost), "Control cost exceeded!"
                    x = system.make_data(init_conds=x[:, 0], control=control, timesteps=timesteps, noisy=noisy)
                    model.fit(x, **fit_kwargs)

                # create test data
                test_init_conds = system.make_init_conds(self._test_examples, in_dist)
                test = system.make_data(test_init_conds, timesteps=self._test_timesteps)
                pred = model.predict(test[:, 0], self._test_timesteps)
                loss = system.calc_loss(pred, test)
                result['n'].append(n)
                result['latent_dim'].append(latent_dim)
                result['embed_dim'].append(embed_dim)
                result['timesteps'].append(timesteps)
                result['loss'].append(loss)
                result['total_cost'].append(total_cost)
            return pd.DataFrame(result)

        data = Parallel(n_jobs=4)(delayed(do_rep)() for _ in range(self._reps))

        data = pd.concat(data)
        data["id"] = id or next(self._id)
        return data
