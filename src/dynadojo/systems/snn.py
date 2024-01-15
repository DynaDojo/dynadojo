import numpy as np
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize, NonlinearConstraint

from .lds import LDSystem


class SNNSystem(LDSystem):

    def __init__(self, latent_dim=2, embed_dim=4,
                 IND_range=(0, 1),
                 OOD_range=(-1, 0),
                 seed=None,
                 **kwargs,
                 ):
        assert embed_dim >= 2 * latent_dim, "REQUIRED: embed_dim ≥ 2 * latent_dim"
        super().__init__(latent_dim, embed_dim,
                         IND_range=IND_range,
                         OOD_range=OOD_range,
                         seed=seed,
                         **kwargs,
                         )
        c = lambda t: np.zeros(latent_dim)
        self.LDS = LinearDynamicalSystem(self.A, self._controller, c)
        self.SNN = SpikingNeuralNetwork(self.LDS,
                                        N=embed_dim,
                                        seed=seed,
                                        max_error=1 / 2)  # hyperparam: max_error

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        return super().make_init_conds(n, in_dist)

    def make_data(self, init_conds: np.ndarray,
                  control: np.ndarray,
                  timesteps: int,
                  noisy=False) -> np.ndarray:
        time = np.linspace(0, 1, num=timesteps)  # hyperparam: [t0, tf]

        def get_trajectory(x0, u):
            u_interp = interp1d(time, u.T, kind=3)  # TODO: only works if u[-1] = u(t_f)
            self.SNN.set_control(u_interp)
            results = self.SNN.simulate(x0, time)
            t_snn = np.hstack(results["t"])
            x_snn = self.SNN.Phi @ (np.hstack(results["rv"])[:self.SNN.N, :])
            v_snn = np.hstack(results["rv"])[self.SNN.N:, :]
            return v_snn

        data = Parallel(n_jobs=4)(delayed(get_trajectory)(x0, u) for x0, u in zip(init_conds, control))
        # PARALLELIZE: This is slower than serial for some reason?
        # https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop
        # https://colab.research.google.com/github/QuantEcon/lecture-python-programming.notebooks/blob/master/parallelization.ipynb

        # trajs = []
        # for x0, u in tqdm(zip(init_conds, control), total=len(init_conds)):
        #     trajs.append(get_trajectory(x0, u))

        data = np.array(data)
        data = np.transpose(data, axes=(0, 2, 1))
        return data


class LinearDynamicalSystem:

    def __init__(self, A, B, control):
        # dx/dt = Ax + Bc
        self.dim = A.shape[0]
        self.A = A
        self.B = B
        self.c = control
        self.x_dot = self._construct_LDS()

    def _construct_LDS(self):
        return lambda t, x: self.A @ x + self.B @ self.c(t)

    def simulate(self, x_0, t_eval):
        t_0, t_f = t_eval[[0, -1]]
        max_step = (t_eval[1] - t_eval[0]) / 10
        results = solve_ivp(self.x_dot,
                            [t_0, t_f],
                            x_0,
                            method=['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'][0],
                            t_eval=t_eval,
                            max_step=max_step,
                            # rtol = 1e-3,
                            # atol= 1e-6,
                            vectorized=False
                            )
        return results


#

class SpikingNeuralNetwork:

    def __init__(self, LDS, N=None, max_error=1 / np.sqrt(2), seed=None):
        self._rng = np.random.default_rng(seed=seed)
        self.LDS = LDS
        self.N = N if N else 2 * LDS.dim
        self.threshold = (2 * max_error) ** 2 / 2.0
        self.Phi = self._generate_decoders()
        self.control = lambda t: np.zeros(self.N)
        self.rv_dot = self._construct_SNN()

    def _generate_decoders(self):
        d = self.LDS.dim
        magnitude = np.sqrt(2 * self.threshold)  # |d| = 2*max_error , T = |d|^2 / 2

        def min_distance(x_flat):
            # objective: maximize the minimum pairwise distance
            from sklearn.metrics import pairwise_distances as pdist
            X = np.reshape(x_flat, (d, self.N))
            dist = pdist(X.T)
            return -np.min(dist[np.tril_indices_from(dist, k=-1)])

        def threshold(x_flat):
            X = np.reshape(x_flat, (d, self.N))
            return np.linalg.norm(X, axis=0)

        def generate_decoder(magnitude):
            X = self._rng.normal(size=(d, self.N))
            X *= magnitude / np.linalg.norm(X, axis=0)
            result = minimize(min_distance,
                              X.flatten(),
                              constraints=NonlinearConstraint(threshold,
                                                              magnitude,
                                                              magnitude),
                              tol=1e-10,
                              )
            return result.x.reshape(d, self.N)

        D = generate_decoder(magnitude)
        while ~np.all((np.max(D, axis=1) > 0) & (np.min(D, axis=1) < 0)):
            D = generate_decoder(magnitude)

        return D

    def _construct_SNN(self):
        d = self.LDS.dim
        VV = self.Phi.T @ self.LDS.A @ np.linalg.pinv(self.Phi.T)
        VR = self.Phi.T @ (self.LDS.A + np.eye(d)) @ self.Phi
        VD = -self.Phi.T @ self.Phi
        VC = self.Phi.T @ self.LDS.B
        RR = -np.eye(self.N)
        RD = np.eye(self.N)
        RVRV = np.vstack([np.hstack([RR, np.zeros((self.N, self.N))]),
                          np.hstack([VR, VV])])
        RVD = np.vstack([RD, VD])
        self.RVD = RVD
        RVC = np.vstack([np.zeros(VC.shape), VC])
        return lambda t, rv: RVRV @ rv + np.hstack([np.zeros(self.N), self.control(t)])  # RVC@self.LDS.c(t)

    def set_control(self, control):
        self.control = control

    def simulate(self, x_0, t_eval):

        def event_list_generator():
            event_list = []
            for i in range(self.N):
                e = (lambda i: lambda t, rv: rv[self.N + i] - self.threshold)(i)
                e.terminal = True
                e.direction = 1
                event_list += [e]
            return event_list

        t_0, t_f = t_eval[[0, -1]]
        max_step = (t_eval[1] - t_eval[0]) / 10

        rv_0 = np.hstack([x_0, np.zeros(self.N)])
        events = event_list_generator()

        all_results = {
            "t": [],
            "rv": [],
            'spikes': []
        }

        while t_0 < t_f:

            results = solve_ivp(self.rv_dot,
                                [t_0, t_f],
                                rv_0,
                                method=['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'][0],
                                events=events,
                                t_eval=t_eval[t_0 <= t_eval],
                                max_step=max_step,
                                # rtol = 1e-3,
                                # atol= 1e-6,
                                vectorized=False)

            if np.any(results.t_events):
                t_spike = np.hstack(results.t_events)[0]
                n_idx = np.nonzero(np.array(results.t_events, dtype=object))[0][0]
                # print('Neuron {}: {} s'.format(n_idx, t_spike))
                all_results['spikes'].append((n_idx, t_spike))

                # spike update
                t_0 = t_spike
                delta = np.zeros(self.N)
                delta[n_idx] = 1
                rv_0 = results.y_events[n_idx][0] + self.RVD @ delta

            if np.any(results.t):
                all_results["t"] += [results.t]
                all_results["rv"] += [results.y]
                if results.t[-1] == t_f:
                    break

        return all_results
