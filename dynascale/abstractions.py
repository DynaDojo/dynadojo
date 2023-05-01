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

    def _make_data(self, init_conds: np.ndarray, timesteps: int, control=None) -> np.ndarray:
        raise NotImplementedError

    def _make_init_conds(self, n: int, in_dist=True) -> (np.ndarray, np.ndarray):
        raise NotImplementedError

    def make_data(self, n: int, timesteps, control=None, in_dist=True, init_conds: np.ndarray = None):
        assert n > 0
        assert timesteps > 1
        init_conds = self._make_init_conds(n, in_dist) if init_conds is None else init_conds
        assert init_conds.shape == (n, self.latent_dim)
        data, final_conds = self._make_data(init_conds, timesteps, control)
        assert final_conds.shape == (n, self.latent_dim)  # NOTE: the final condition should be in LATENT SPACE
        assert data.shape == (n, timesteps, self.embed_dim)
        return data, init_conds


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

    def evaluate(self, model_cls: type[Model], trials: int = 1, test_size: int = 1, in_dist=True, **kwargs):
        data = {"n": [], "latent_dim": [], "embed_dim": [], "timesteps": [], "supepoch": [], "score": []}
        for _ in range(trials):
            factory = None
            for n, latent_dim, embed_dim, timesteps in itertools.product(self._N, self._L, self._E, self._T):
                if factory is None:
                    factory = self._factory_cls(latent_dim, embed_dim)
                if latent_dim != factory.latent_dim:
                    factory.latent_dim = latent_dim
                if embed_dim < latent_dim:
                    continue
                if embed_dim != factory.embed_dim:
                    factory.embed_dim = embed_dim

                # Create and train model
                model = model_cls(latent_dim, embed_dim, timesteps, **kwargs)
                test, _ = factory.make_data(test_size, timesteps, in_dist=in_dist)
                control = None
                init_conds = None
                for i in range(self._supepochs):
                    x, init_conds = factory.make_data(n, timesteps, control=control, init_conds=init_conds)
                    model.fit(x)
                    model.act(x)
                    pred = model.predict(test[:, 0], timesteps)
                    score = self._metric(pred, test)
                    data["n"].append(n)
                    data["latent_dim"].append(latent_dim)
                    data["embed_dim"].append(embed_dim)
                    data["timesteps"].append(timesteps)
                    data["supepoch"].append(i)
                    data["score"].append(score)
        return pd.DataFrame(data)