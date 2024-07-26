"""

Functions that act on DynSys or DynMap objects

"""

import numpy as np
import warnings

def find_lyapunov_exponents(
    model, traj_length, pts_per_period=500, tol=1e-8, min_tpts=10, **kwargs
):
    """
    Given a dynamical system, compute its spectrum of Lyapunov exponents.
    Args:
        model (callable): the right hand side of a differential equation, in format 
            func(X, t)
        traj_length (int): the length of each trajectory used to calulate Lyapunov
            exponents
        pts_per_period (int): the sampling density of the trajectory
        kwargs: additional keyword arguments to pass to the model's make_trajectory 
            method

    Returns:
        final_lyap (ndarray): A list of computed Lyapunov exponents

    References:
        Christiansen & Rugh (1997). Computing Lyapunov spectra with continuous
            Gram-Schmidt orthonormalization

    Example:
        >>> import dysts
        >>> model = dysts.Lorenz()
        >>> lyap = dysts.find_lyapunov_exponents(model, 1000, pts_per_period=1000)
        >>> print(lyap)

    """
    d = np.asarray(model.ic).shape[-1]
    tpts, traj = model.make_trajectory(
        traj_length, pts_per_period=pts_per_period, resample=True, return_times=True,
        postprocessing=False,
        **kwargs
    )
    dt = np.median(np.diff(tpts))
    # traj has shape (traj_length, d), where d is the dimension of the system
    # tpts has shape (traj_length,)
    # dt is the dimension of the system

    u = np.identity(d)
    all_lyap = list()
    # for i in range(traj_length):
    for i, (t, X) in enumerate(zip(tpts, traj)):
        X = traj[i]

        if model.jac(model.ic, 0) is None:
            rhsy = lambda x: np.array(model.rhs(x, t))
            jacval = jac_fd(rhsy, X)
        else:
            jacval = np.array(model.jac(X, t))

        # If postprocessing is applied to a trajectory, transform the jacobian into the
        # new coordinates.
        if hasattr(model, "_postprocessing"):
            X0 = np.copy(X)
            y2h = lambda y: model._postprocessing(*y)
            dhdy = jac_fd(y2h, X0)
            dydh = np.linalg.inv(dhdy)  # dy/dh
            ## Alternate version if good second-order fd is ever available
            # dydh = jac_fd(y2h, X0, m=2, eps=1e-2) @ rhsy(X0) + jac_fd(y2h, y0) @ jac_fd(rhsy, X0))
            jacval = dhdy @ jacval @ dydh

        ## Forward Euler update
        # u_n = np.matmul(np.eye(d) + jacval * dt, u)
        
        ## Backward Euler update
        if i < 1: continue
        u_n = np.matmul(np.linalg.inv(np.eye(d) - jacval * dt), u)
        
        q, r = np.linalg.qr(u_n)
        lyap_estimate = np.log(abs(r.diagonal()))
        all_lyap.append(lyap_estimate)
        u = q  # post-iteration update axes

        ## early stopping if middle exponents are close to zero, a requirement for
        ## continuous-time dynamical systems
        if (np.min(np.abs(lyap_estimate)) < tol) and (i > min_tpts):
            traj_length = i

    all_lyap = np.array(all_lyap)
    final_lyap = np.sum(all_lyap, axis=0) / (dt * traj_length)
    return np.sort(final_lyap)[::-1]