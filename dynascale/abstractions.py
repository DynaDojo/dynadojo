import itertools
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class Model(ABC):
    def __init__(self, latent_dim, embed_dim, timesteps, control_constraint, **kwargs):
        """
        An abstract base class for custom models that will be evaluated by Task.

        :param latent_dim: the dimensionality of the underlying dynamical system
        :param embed_dim: the dimensionality of the data
        :param timesteps: the number of timesteps in each trajectory
        :param kwargs:
        """
        # TODO: obscure latent dimension

        self._latent_dim = latent_dim
        self._embed_dim = embed_dim
        self._timesteps = timesteps
        self._control_constraint = control_constraint

    @property
    def latent_dim(self):
        return self._latent_dim

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def timesteps(self):
        return self._timesteps
    
    @property
    def control_constraint(self):
        return self._control_constraint

    @latent_dim.setter
    def latent_dim(self, value):
        self._latent_dim = value

    @embed_dim.setter
    def embed_dim(self, value):
        self._embed_dim = value

    @abstractmethod
    def fit(self, x: np.ndarray, *args, **kwargs):
        raise NotImplementedError

    def _act(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        return np.zeros_like(x)

    def act(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        control = self._act(x, *args, **kwargs)
        assert control.shape == x.shape
        return control

    @abstractmethod
    def _predict(self, x0: np.ndarray, timesteps: int, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def predict(self, x0: np.ndarray, timesteps, *args, **kwargs) -> np.ndarray:
        """
        wrapper method for _predict(...).

        :param x0: initial conditions
        :param timesteps: number of timesteps in the predicted trajectory
        :param args:
        :param kwargs:
        :return: predicted trajectories
        """
        pred = self._predict(x0, timesteps, *args, **kwargs)
        n = x0.shape[0]
        assert pred.shape == (n, self.timesteps, self.embed_dim)
        return pred


class Challenge(ABC):
    def __init__(self, latent_dim, embed_dim):
        """
        An abstract base class for implementing challenges.

        :param latent_dim:
        :param embed_dim:
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
        """
        Calculates the loss between two datasets x and y.

        :param x: a dataset
        :param y: another dataset
        :return: a scalar loss
        """
        assert x.shape == y.shape
        return self._calc_loss(x, y)

    def _visualize(self, x, *args, **kwargs):
        raise NotImplementedError


class Task(object):
    def __init__(self,
                 N: list[int],
                 L: list[int],
                 E: list[int],
                 T: list[int],
                 control_horizons: int,
                 challenge_cls: type[Challenge],
                 trials: int,
                 test_size: int,
                 control_constraint: int,
                 challenge_kwargs: dict = None,
                 ):
        assert control_horizons > 0

        self._id = itertools.count()
        self._N = N
        self._L = L
        self._E = E
        self._T = T
        self._challenge_cls = challenge_cls
        self._challenge_kwargs = challenge_kwargs
        self._control_horizons = control_horizons
        self._trials = trials
        self._test_size = test_size
        self._control_constraint = control_constraint
        self._ord = "fro"  # TODO: add noorm to challenge and constraint to challenge by Monday


def evaluate(self, model_cls: type[Model], model_kwargs: dict = None, in_dist=True, noisy=False):
    data = {"n": [], "latent_dim": [], "embed_dim": [], "timesteps": [], "loss": []}
    total = len(self._N) * len(self._L) * len(self._E) * len(self._T) * self._trials
    with tqdm(total=total, position=0, leave=False) as pbar:
        for i in range(self._trials):
            challenge = None
            for n, latent_dim, embed_dim, timesteps in itertools.product(self._N, self._L, self._E, self._T):
                pbar.set_description(f"Trial {i}/{self._trials}: {n=}, {latent_dim=}, {embed_dim=}, {timesteps=}")
                if challenge is None:
                    challenge = self._challenge_cls(latent_dim, embed_dim, self._challenge_kwargs)
                if latent_dim != challenge.latent_dim:
                    challenge.latent_dim = latent_dim
                if embed_dim < latent_dim:
                    continue
                if embed_dim != challenge.embed_dim:
                    challenge.embed_dim = embed_dim

                # Create and train model
                model = model_cls(latent_dim, embed_dim, timesteps, self._control_constraint)  # TODO: does timesteps need to be here?
                train_init_conds = challenge.make_init_conds(n)
                for j in range(self._control_horizons):
                    if j == 0:
                        x = challenge.make_data(train_init_conds, timesteps=timesteps, noisy=noisy)
                    else:
                        control = model.act(x)
                        # TODO: use challenge.norm
                        assert np.all(np.norm(control, axis=0) / timesteps <= self._control_constraint), "control constraint violated"
                        x = challenge.make_data(init_conds=x[:, 0], control=control, timesteps=timesteps,
                                                noisy=noisy)
                    model.fit(x, **model_kwargs)

                # create test data
                test_init_conds = challenge.make_init_conds(self._test_size, in_dist)
                test = challenge.make_data(test_init_conds, timesteps=timesteps)
                pred = model.predict(test[:, 0], timesteps)
                loss = challenge.calc_loss(pred, test)
                data["n"].append(n)
                data["latent_dim"].append(latent_dim)
                data["embed_dim"].append(embed_dim)
                data["timesteps"].append(timesteps)
                data["loss"].append(loss)
                pbar.update()
            data["id"] = next(self._id)
    return pd.DataFrame(data)
