import itertools

from tqdm.auto import tqdm
import numpy as np
import pandas as pd


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




class Challenge(object):
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

    def _calc_error(self, x, y) -> float:
        raise NotImplementedError

    def calc_error(self, x, y):
        assert x.shape == y.shape
        return self._calc_error(x, y)



class Task(object):
    def __init__(self, N: list[int], L: list[int], E: list[int], T: list[int], supepochs: int, factory_cls: type[Challenge],
                 trials: int, test_size: int):
        self._id = itertools.count()
        self._N = N
        self._L = L
        self._E = E
        self._T = T
        self._factory_cls = factory_cls
        self._supepochs = supepochs
        self._trials = trials
        self._test_size = test_size

    def evaluate(self, model_cls: type[Model], in_dist=True, **kwargs):
        data = {"n": [], "latent_dim": [], "embed_dim": [], "timesteps": [], "error": []}
        for _ in tqdm(range(self._trials)):
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
                test, _ = factory.make_data(self._test_size, timesteps, in_dist=in_dist)
                control = None
                init_conds = None
                pred = None
                for i in range(self._supepochs):
                    x, init_conds = factory.make_data(n, timesteps, control=control, init_conds=init_conds)
                    model.fit(x)
                    model.act(x)
                    pred = model.predict(test[:, 0], timesteps)
                err = factory.calc_error(pred, test)
                data["n"].append(n)
                data["latent_dim"].append(latent_dim)
                data["embed_dim"].append(embed_dim)
                data["timesteps"].append(timesteps)
                data["error"].append(err)  # TODO: change this to "loss"
        data["id"] = next(self._id)
        return pd.DataFrame(data)

    def plot(self, frames: list[pd.DataFrame], labels: list[str]):
        raise NotImplementedError