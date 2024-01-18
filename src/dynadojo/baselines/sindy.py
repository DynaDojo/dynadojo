"""
Sparse Identification of Nonlinear Dynamical systems (SINDy)
===============================================================
"""
import numpy as np
import pysindy as ps

from ..abstractions import AbstractAlgorithm


class SINDy(AbstractAlgorithm):
    """
    Sparse Identification of Nonlinear Dynamical systems (SINDy). Wrapper for
    ``pysindy``[1]_.

    References
    -----------
    .. [1] https://pysindy.readthedocs.io/en/latest/

    Note
    -----
    For an example of how to use ``SINDy`` with a challenge, see ``LorenzSystem``.

    Example
    --------
    .. include:: ../sindy_example.rst
    """
    def __init__(self,
                 embed_dim: int,
                 timesteps: int,
                 max_control_cost: float = 0,
                 differentiation_method=None,
                 integrator="solve_ivp",
                 **kwargs):
        """
        Initialize the class.

        Parameters
        -------------
        embed_dim : int
            The embedded dimension of the system. Recommended to keep embed dimension small (e.g., <10).
        timesteps : int
            The timesteps of the training trajectories. Must be greater than 2.
        differentiation_method : str, optional
            The differentiation used in SINDy. See PySINDy documentation for more details.
        max_control_cost : float, optional
            Ignores control, so defaults to 0.
        integrator : str, optional
            The integrator used in SINDy. See PySINDy documentation for more details. Only `odeint` and `solve_ivp` is supported.
        """
        assert timesteps > 2, "timesteps must be greater than 2. "
        if differentiation_method == 'smoothed_fd':
            differentiation_method = ps.SmoothedFiniteDifference(
                smoother_kws={
                    'window_length': np.log2(timesteps).astype(int),
                    'polyorder': np.log10(timesteps).astype(int)
                })
        if differentiation_method == "fd":
            differentiation_method = ps.FiniteDifference()

        super().__init__(embed_dim, timesteps, max_control_cost, **kwargs)
        np.random.seed(self._seed)
        THRESHOLD = 0.025
        MAX_ITERATIONS = 10
        optimizer = ps.STLSQ(threshold=THRESHOLD, max_iter=MAX_ITERATIONS)
        poly_order = max(2, int(np.log2(embed_dim) // 1))
        self.integrator = integrator
        self._model = ps.SINDy(
            differentiation_method=differentiation_method,
            optimizer=optimizer,
            feature_library=ps.PolynomialLibrary(degree=poly_order)
        )

    def fit(self, x: np.ndarray, **kwargs) -> None:
        X = [*x]
        t = [np.linspace(0, 1, self._timesteps) for _ in range(len(x))]
        self._model.fit(X, t=t, multiple_trajectories=True, quiet=True, ensemble=True, n_models=5, unbias=True)

    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        results = [self._model.simulate(point, np.linspace(0, 1, timesteps), integrator=self.integrator) for point in x0]
        results = np.array(results)
        return results
