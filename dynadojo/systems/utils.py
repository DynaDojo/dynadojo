import numpy as np
from scipy.stats import ortho_group
from scipy.integrate import solve_ivp
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep

from ..abstractions import AbstractSystem


class SimpleSystem(AbstractSystem):
    def __init__(self, latent_dim, embed_dim,
                 # singular values are non-neg by convention; >0 since we don't want a nontrivial null space
                 embedder_sv_range=(0.1, 1),
                 controller_sv_range=(0.1, 1),
                 IND_range=(0, 10),
                 OOD_range=(-10, 0),
                 noise_scale=0.01,
                 t_range=(0, 1),
                 ):
        super().__init__(latent_dim, embed_dim)

        self._t_range = t_range

        self.IND_range = IND_range
        self.OOD_range = OOD_range

        self._noise_scale = noise_scale
        self._rng = np.random.default_rng()

        self._embedder_sv_range = embedder_sv_range
        self._controller_sv_range = controller_sv_range
        self._embedder = None
        self._controller = None
        self._update_embedder_and_controller()

    @property
    def embedder(self):
        return self._embedder

    @property
    def controller(self):
        return self._controller

    def _update_embedder_and_controller(self):
        self._embedder = self._sv_to_matrix(self.latent_dim, self.embed_dim, self._embedder_sv_range)
        self._controller = self._sv_to_matrix(self.latent_dim, self.embed_dim, self._controller_sv_range)

    @AbstractSystem.embed_dim.setter
    def embed_dim(self, value):
        self._embed_dim = value
        self._update_embedder_and_controller()

    @AbstractSystem.latent_dim.setter
    def latent_dim(self, value):
        self._latent_dim = value
        self._update_embedder_and_controller()

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        """Uniformly samples embedded-dimensional points from an inside or outside distribution"""
        init_cond_range = self.IND_range if in_dist else self.OOD_range
        return self._rng.uniform(*init_cond_range, (n, self.embed_dim))

    def calc_error(self, x, y) -> float:
        """Returns MSE"""
        error = x - y
        return np.mean(error ** 2) / self.embed_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        """Calculates the L2 norm / dimension of every vector in the control"""
        return np.linalg.norm(control, axis=(1, 2), ord=2) / self.embed_dim

    def calc_dynamics(self, t, x):
        raise NotImplementedError

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []
        init_conds = init_conds @ np.linalg.pinv(self.embedder)
        time = np.linspace(self._t_range[0], self._t_range[1], num=timesteps)

        def dynamics(t, x, u):
            i = np.argmin(np.abs(t - time))
            dx = self.calc_dynamics(t, x) + self.controller @ u[i]
            if noisy:
                dx += self._rng.normal(scale=self._noise_scale, size=self.latent_dim)
            return dx

        for x0, u in zip(init_conds, control):
            sol = solve_ivp(dynamics, t_span=[self._t_range[0], self._t_range[1]], y0=x0, t_eval=time, dense_output=True, args=(u,))
            data.append(sol.y)

        data = np.transpose(np.array(data), axes=(0, 2, 1)) @ self.embedder
        return data

    def _sv_to_matrix(self, m, n, sv_range):
        U = ortho_group.rvs(m)
        sigma = np.eye(m, n) * self._rng.uniform(*sv_range, size=n)
        V = ortho_group.rvs(n)
        M = U @ sigma @ V
        return M

class OpinionSystem(AbstractSystem):
    def __init__(self, latent_dim, embed_dim,
                 noise_scale,
                 IND_range, 
                 OOD_range,
                 seed=None):

        super().__init__(latent_dim, embed_dim)

        assert embed_dim == latent_dim
        self._rng = np.random.default_rng(seed)

        self.noise_scale = noise_scale
        self.IND_range = IND_range
        self.OOD_range = OOD_range

    def create_model(self, x0):
        return

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        x0 = []
        for _ in range(n):
            if in_dist:
                x0.append({node: np.random.uniform(
                    self.IND_range[0], self.IND_range[1]) for node in range(self.latent_dim)})
            else:
                x0.append({node: np.random.uniform(
                    self.OOD_range[0], self.OOD_range[1]) for node in range(self.latent_dim)})
        return x0 

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []

        if noisy:
            noise = np.random.normal(
                0, self.noise_scale, (self.latent_dim))
        else:
            noise = np.zeros((self.latent_dim))

        def dynamics(x0):
            self.create_model(x0)

            iterations = self.model.iteration_bunch(timesteps)
            dX = []
            for iteration in iterations:
                step = []
                for idx in range(self.latent_dim):
                    if (idx in iteration["status"]):
                        step.append(iteration["status"][idx])
                    else:
                        step.append(dX[-1][idx])
                dX.append(step + noise)
            return dX

        if control:
            for x0, u in zip(init_conds, control):
                sol = dynamics(x0)
                data.append(sol)

        else:
            for x0 in init_conds:
                sol = dynamics(x0)
                data.append(sol)

        data = np.transpose(data, axes=(0, 2, 1))
        return data

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2) / self.latent_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2)

class EpidemicSystem(AbstractSystem):
    def __init__(self, latent_dim, embed_dim,
                 noise_scale,
                 IND_range, 
                 OOD_range,
                 group_status,
                 seed=None):

        super().__init__(latent_dim, embed_dim)

        if not group_status:
            assert embed_dim == latent_dim
        self._rng = np.random.default_rng(seed)

        self.noise_scale = noise_scale
        self.IND_range = IND_range
        self.OOD_range = OOD_range
        self.group_status = group_status

    def create_model(self, x0):
        return

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        x0 = []
        for _ in range(n):
            if in_dist:
                x0.append({node: int(np.random.uniform(
                    self.IND_range[0], self.IND_range[1])) for node in range(self.latent_dim)})
            else:
                x0.append({node: int(np.random.uniform(
                    self.OOD_range[0], self.OOD_range[1])) for node in range(self.latent_dim)})

        return x0 

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []

        if noisy:
            noise = np.random.normal(
                0, self.noise_scale, (self.latent_dim))
        else:
            noise = np.zeros((self.latent_dim))

        def dynamics(x0):
            self.create_model(x0)

            iterations = self.model.iteration_bunch(timesteps)
            dX = []
            for iteration in iterations:
                if(self.group_status):
                    step = [val for _, val in iteration['node_count'].items()]
                    dX.append(step)
                else:
                    step = []
                    for idx in range(self.latent_dim):
                        if (idx in iteration["status"]):
                            step.append(iteration["status"][idx])
                        else:
                            step.append(dX[-1][idx])
                    dX.append([int(x) for x in (step + noise)])
            return dX

        if control:
            for x0, u in zip(init_conds, control):
                sol = dynamics(x0)
                data.append(sol)

        else:
            for x0 in init_conds:
                sol = dynamics(x0)
                data.append(sol)

        data = np.transpose(data, axes=(0, 2, 1))
        return data

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2) / self.latent_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2)
