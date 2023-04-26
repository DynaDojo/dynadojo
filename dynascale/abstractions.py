import itertools

import numpy as np

from . import metrics

from inspect import signature

class Model(object):
    def __init__(self, latent_dim, embed_dim, timesteps, **kwargs):
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

    def fit(self, x: np.ndarray, *args, **kwargs):
        raise NotImplementedError

    def act(self, x: np.ndarray) -> np.ndarray | None:
        return None

    def predict(self, x0: np.ndarray) -> np.ndarray:
        raise NotImplementedError




class Factory(object):
    def __init__(self, latent_dim, embed_dim):
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

    def _make_data(self, n: int, timesteps: int, control=None, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def make_data(self, n: int, timesteps, control=None, *args, **kwargs):
        assert n > 0
        assert timesteps > 1
        data = self._make_data(n, timesteps, control, *args, **kwargs)
        assert data.shape == (n, timesteps, self.embed_dim)
        return data


class Task(object):
    def __init__(self, N: list[int], L: list[int], E: list[int], T: list[int],
                 supepochs: int, factory: Factory, metric=metrics.mean_squared_error):
        self._N = N
        self._L = L
        self._E = E
        self._T = T
        self._factory = factory
        self._metric = metric
        self._supepochs = supepochs

    def evaluate(self, model_cls: type[Model], trials: int = 1, **kwargs):
        scores = {}
        for _ in range(trials):
            for n, latent_dim, embed_dim, timesteps in itertools.product(self._N, self._L, self._E, self._T):
                if latent_dim != self._factory.latent_dim:
                    self._factory.latent_dim(latent_dim)
                if embed_dim < latent_dim:
                    continue
                if embed_dim != self._factory.embed_dim:
                    self._factory.embed_dim(embed_dim)

                # Create and train model
                model = model_cls(latent_dim, embed_dim, timesteps, **kwargs)
                test = self._factory.make_data(n, timesteps)  # TODO: figure out how to size the test dataset
                control = None
                for i in range(self._supepochs):
                    train = self._factory.make_data(n, timesteps, control)
                    model.fit(train, **kwargs)
                    model.act(train)
                    pred = model.predict(test[:, 0])  # TODO: add evaluation method
                    score = self._metric(pred, test)
                    # TODO: add row to pandas DataFrame
                    # scores[]
                    # TODO: construct dataFrame column here
                # TODO: figure out how to modify control size (ratio of n? fixed argument to method?)
                # TODO: add OOD
        return scores



class Contract(object):
    def __init__(self, factory: Factory, metric):
        pass