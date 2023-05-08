import itertools

from tqdm.auto import tqdm
import numpy as np
import pandas as pd


class Model(object):
    def __init__(self, latent_dim, embed_dim, timesteps, **kwargs):
        """
        An abstract base class for custom models that will be evaluated by Task.

        :param latent_dim: the dimensionality of the underlying dynamical system
        :param embed_dim: the dimensionality of the data
        :param timesteps: the number of timesteps in each trajectory
        :param kwargs:
        """
        self._latent_dim = latent_dim
        self._embed_dim = embed_dim
        self._timesteps = timesteps

    @property
    def latent_dim(self):
        return self._latent_dim

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def timesteps(self):
        return self._timesteps

    @latent_dim.setter
    def latent_dim(self, value):
        self._latent_dim = value

    @embed_dim.setter
    def embed_dim(self, value):
        self._embed_dim = value

    def fit(self, x: np.ndarray, *args, **kwargs):
        """
        fit only takes in a single dataset array (as opposed to a collection of features and labels), because
        both features and labels are inherently encoded in the dataset (e.g., if x_t is my feature, then
        x_{t+1} is my label)

        :param x: a NumPy array of trajectories with shape (n, timesteps, embed_dim)
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def act(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        return np.zeros_like(x)

    def _predict(self, x0: np.ndarray, timesteps: int, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def predict(self, x0: np.ndarray, timesteps: int, *args, **kwargs) -> np.ndarray:
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
        assert pred.shape == (n, timesteps, self.embed_dim)
        return pred


class Challenge(object):
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

    def _make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        raise NotImplementedError

    def make_init_conds(self, n: int, in_dist=True):
        init_conds = self._make_init_conds(n, in_dist)
        assert init_conds.shape == (n, self.embed_dim)
        return init_conds

    def _make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int) -> np.ndarray:
        raise NotImplementedError


    def make_data(self, init_conds: np.ndarray, control: np.ndarray = None, timesteps: int = 1) -> np.ndarray:
        assert timesteps > 0
        assert init_conds.ndim == 2 and init_conds.shape[1] == self.embed_dim
        n = init_conds.shape[0]
        if control is None:
            control = np.zeros((n, timesteps, self.embed_dim))
        assert control.shape == (n, timesteps, self.embed_dim)
        data = self._make_data(init_conds, control, timesteps)
        assert data.shape == (n, timesteps, self.embed_dim)
        return data

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
    def __init__(self, N: list[int], L: list[int], E: list[int], T: list[int], control_horizons: int,
                 factory_cls: type[Challenge],
                 trials: int, test_size: int):
        assert control_horizons > 0
        self._id = itertools.count()
        self._N = N
        self._L = L
        self._E = E
        self._T = T
        self._factory_cls = factory_cls  # TODO: change this to accept an actual instance
        self._control_horizons = control_horizons
        self._trials = trials
        self._test_size = test_size

    def evaluate(self, model_cls: type[Model], in_dist=True, **kwargs):
        """
        :param model_cls: a class constructor for the model to test (must inherit from Model)
        :param in_dist: test data is in distribution if True, out of distribution otherwise
        :param kwargs:
        :return:
        """
        data = {"n": [], "latent_dim": [], "embed_dim": [], "timesteps": [], "loss": []}
        total = len(self._N) * len(self._L) * len(self._E) * len(self._T) * self._trials
        with tqdm(total=total, position=0, leave=False) as pbar:
            for i in range(self._trials):
                factory = None
                for n, latent_dim, embed_dim, timesteps in itertools.product(self._N, self._L, self._E, self._T):
                    pbar.set_description(f"Trial {i}/{self._trials}: {n=}, {latent_dim=}, {embed_dim=}, {timesteps=}")
                    if factory is None:
                        factory = self._factory_cls(latent_dim, embed_dim)
                    if latent_dim != factory.latent_dim:
                        factory.latent_dim = latent_dim
                    if embed_dim < latent_dim:
                        continue
                    if embed_dim != factory.embed_dim:
                        factory.embed_dim = embed_dim

                    # Create and train model
                    model = model_cls(latent_dim, embed_dim, timesteps)
                    train_init_conds = factory.make_init_conds(n)
                    for j in range(self._control_horizons):
                        if j == 0:
                            x = factory.make_data(train_init_conds, timesteps=timesteps)
                        else:
                            control = model.act(x)
                            x = factory.make_data(init_conds=x[:, 0], control=control, timesteps=timesteps)
                        model.fit(x, **kwargs)  # TODO: is there a more elegant way of passing in epochs? perhaps w/ partial?

                    # create test data
                    test_init_conds = factory.make_init_conds(self._test_size, in_dist)
                    test = factory.make_data(test_init_conds, timesteps=timesteps)
                    pred = model.predict(test[:, 0], timesteps)
                    loss = factory.calc_loss(pred, test)
                    data["n"].append(n)
                    data["latent_dim"].append(latent_dim)
                    data["embed_dim"].append(embed_dim)
                    data["timesteps"].append(timesteps)
                    data["loss"].append(loss)
                    pbar.update()
                data["id"] = next(self._id)
        return pd.DataFrame(data)
