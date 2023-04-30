import itertools

import numpy as np
import pandas as pd

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

    @latent_dim.setter
    def latent_dim(self, value):
        self._latent_dim = value

    @embed_dim.setter
    def embed_dim(self, value):
        self._embed_dim = value

    def fit(self, x: np.ndarray, *args, **kwargs):
        raise NotImplementedError

    def act(self, x: np.ndarray, *args, **kwargs) -> np.ndarray | None:
        return None

    def predict(self, x0: np.ndarray, timesteps: int, *args, **kwargs) -> np.ndarray:
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

    def _make_data(self, n: int, timesteps: int, control=None, in_dist=True, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def make_data(self, n: int, timesteps, control=None, in_dist=True, *args, **kwargs):
        assert n > 0
        assert timesteps > 1
        data = self._make_data(n, timesteps, control, in_dist, *args, **kwargs)
        assert data.shape == (n, timesteps, self.embed_dim)
        return data


class Task(object):
    def __init__(self, N: list[int], L: list[int], E: list[int], T: list[int],
                 supepochs: int, factory_cls: type[Factory], metric=metrics.mean_squared_error):
        self._N = N
        self._L = L
        self._E = E
        self._T = T
        self._factory_cls = factory_cls
        self._metric = metric
        self._supepochs = supepochs

    def evaluate(self, model_cls: type[Model], trials: int = 1, in_dist=True, **kwargs):
        data = {"n": [], "latent_dim": [], "embed_dim": [], "timesteps": [], "supepoch": [], "score": []}
        for _ in range(trials):
            factory = None
            for n, latent_dim, embed_dim, timesteps in itertools.product(self._N, self._L, self._E, self._T):
                if factory is None:
                    factory = self._factory_cls(latent_dim, embed_dim)
                if latent_dim != factory.latent_dim:
                    factory.latent_dim = latent_dim  # TODO: how to use setter method?
                if embed_dim < latent_dim:
                    continue
                if embed_dim != factory.embed_dim:
                    factory.embed_dim = embed_dim

                # Create and train model
                model = model_cls(latent_dim, embed_dim, timesteps, **kwargs)
                test = factory.make_data(n, timesteps)  # TODO: figure out how to size the test dataset
                control = None
                for i in range(self._supepochs):
                    train = factory.make_data(n, timesteps, control)
                    model.fit(train, **kwargs)  # TODO: add closed loop control
                    model.act(train)
                    pred = model.predict(test[:, 0], timesteps)  # TODO: add evaluation method
                    score = self._metric(pred, test)
                    data["n"].append(n)
                    data["latent_dim"].append(latent_dim)
                    data["embed_dim"].append(embed_dim)
                    data["timesteps"].append(timesteps)
                    data["supepoch"].append(i)
                    data["score"].append(score)
                # TODO: figure out how to modify control size (ratio of n? fixed argument to method?)
                # TODO: add OOD
        return pd.DataFrame(data)