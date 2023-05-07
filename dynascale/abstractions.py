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

    def _predict(self, x0: np.ndarray, timesteps: int, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def predict(self, x0: np.ndarray, timesteps: int, *args, **kwargs) -> np.ndarray:
        pred = self._predict(x0, timesteps, *args, **kwargs)
        n = x0.shape[0]
        assert pred.shape == (n, timesteps, self.embed_dim)
        return pred


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

    def _make_data(self, timesteps, n: int = None, init_conds: np.ndarray = None, control=None, in_dist: bool = True) -> np.ndarray:
        raise NotImplementedError

    def make_data(self, timesteps, n: int = None, init_conds: np.ndarray = None, control=None, in_dist: bool = True) -> np.ndarray:
        if init_conds is not None:
            assert init_conds.ndim == 2
            assert init_conds.shape[1] == self.embed_dim
        n = n or init_conds.shape[0]
        assert n > 0
        assert timesteps > 1
        data = self._make_data(timesteps, n, init_conds, control=control, in_dist=in_dist)
        assert data.shape == (n, timesteps, self.embed_dim)
        return data

    def _calc_error(self, x, y) -> float:
        raise NotImplementedError

    def calc_error(self, x, y) -> float:
        assert x.shape == y.shape
        return self._calc_error(x, y)

    def _visualize(self, x, *args, **kwargs):
        raise NotImplementedError

    def visualize(self, x, *args, **kwargs):
        self._visualize(x, *args, **kwargs)


class Task(object):
    def __init__(self, N: list[int], L: list[int], E: list[int], T: list[int], supepochs: int,
                 factory_cls: type[Challenge],
                 trials: int, test_size: int):
        assert supepochs > 0
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
                    model = model_cls(latent_dim, embed_dim, timesteps, **kwargs)
                    test = factory.make_data(timesteps, n=self._test_size, in_dist=in_dist)
                    control = None
                    for j in range(self._supepochs):
                        if j == 0:
                            x = factory.make_data(timesteps, n=n, control=control)
                        else:
                            x = factory.make_data(timesteps, init_conds=x[:, 0], control=control)
                        model.fit(x)
                        model.act(x)
                        pred = model.predict(test[:, 0], timesteps)
                    err = factory.calc_error(pred, test)
                    data["n"].append(n)
                    data["latent_dim"].append(latent_dim)
                    data["embed_dim"].append(embed_dim)
                    data["timesteps"].append(timesteps)
                    data["error"].append(err)
                    pbar.update()
                data["id"] = next(self._id)
        return pd.DataFrame(data)
