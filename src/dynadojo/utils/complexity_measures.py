import numpy as np
import warnings
from dysts.utils import jac_fd
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from dysts.utils import standardize_ts
import pandas as pd

try:
    import neurokit2 # Used for computing multiscale entropy
    has_neurokit = True
except:
    warnings.warn("Neurokit2 must be installed before computing multiscale entropy")
    has_neurokit = False

def estimate_powerlaw(data0):
    """
    Given a 1D array of continuous-valued data, estimate the power law exponent using the 
    maximum likelihood estimator proposed by Clauset, Shalizi, Newman (2009).
    
    Args:
        data0 (np.ndarray): An array of continuous-valued data

    Returns:
        float: The estimated power law exponent
    """
    data = np.sort(data0, axis=0).copy()
    xmin = np.min(data, axis=0)
    n = data.shape[0]
    ahat = 1 + n / np.sum(np.log(data / xmin), axis=0)
    return ahat

## Correlation Dimension
def gp_dim(data, y_data=None, rvals=None, nmax=100):
    """
    Estimate the Grassberger-Procaccia dimension for a numpy array using the 
    empirical correlation integral.

    Args:
        data (np.array): T x D, where T is the number of datapoints/timepoints, and D
            is the number of features/dimensions
        y_data (np.array, Optional): A second dataset of shape T2 x D, for 
            computing cross-correlation.
        rvals (np.array): A list of radii
        nmax (int): The number of points at which to evaluate the correlation integral

    Returns:
        rvals (np.array): The discrete bins at which the correlation integral is 
            estimated
        corr_sum (np.array): The estimates of the correlation integral at each bin

    """

    data = np.asarray(data)
    # data = embed(data)

    ## For self-correlation
    if y_data is None:
        y_data = data.copy()

    if rvals is None:
        std = np.std(data)
        rvals = np.logspace(np.log10(0.1 * std), np.log10(0.5 * std), nmax)

    n = len(data)
    
    # dists = cdist(data, y_data)
    # corr_sum = []
    # for r in rvals:
    #     corr_sum.append(np.sum(dists < r))
    # corr_sum = np.array(corr_sum) / (n * (n - 1))

    # dists = cdist(data, y_data)
    # hist, _ = np.histogram(dists, bins=np.hstack([0, rvals])) # can we skip this and direct fit?
    # corr_sum = np.cumsum(hist).astype(float)
    # corr_sum /= n * (n - 1)
    
    dists = cdist(data, y_data)
    rvals = dists.ravel()

    ## Truncate the distance distribution to the linear scaling range
    std = np.std(data)
    rvals = rvals[rvals > 0]
    rvals = rvals[rvals > np.percentile(rvals, 5)]
    rvals = rvals[rvals < np.percentile(rvals, 50)]
    
    return estimate_powerlaw(rvals)

    # dists = cdist(data, y_data)
    # rvals = np.sort(dists.ravel())
    # corr_sum = np.arange(len(rvals)).astype(float)
    # corr_sum /= n * (n - 1)
    # std = np.std(data)
    # sel_inds = rvals > 0.1 * std
    # rvals = rvals[sel_inds]
    # corr_sum = corr_sum[sel_inds]
    # sel_inds = rvals < 0.5 * std
    # rvals = rvals[sel_inds]
    # corr_sum = corr_sum[sel_inds]

    # ## Drop zeros before regression
    # sel_inds = corr_sum > 0
    # rvals = rvals[sel_inds]
    # corr_sum = corr_sum[sel_inds]
    
    # # poly = np.polyfit(np.log(rvals), np.log(corr_sum), 1)
    # # return poly[0]

    # power_law = lambda x, a, b: a * (x ** b)
    # fit_vals = curve_fit(power_law, rvals, corr_sum)
    # return fit_vals[0][1]

## Multiscale Entropy
def mse_mv(traj, return_info=False):
    """
    Generate an estimate of the multivariate multiscale entropy. The current version 
    computes the entropy separately for each channel and then averages. It therefore 
    represents an upper-bound on the true multivariate multiscale entropy

    Args:
        traj (ndarray): a trajectory of shape (n_timesteps, n_channels)

    Returns:
        mmse (float): the multivariate multiscale entropy

    TODO:
        Implement algorithm from Ahmed and Mandic PRE 2011
    """

    if not has_neurokit:
        raise Exception("NeuroKit not installed; multiscale entropy cannot be computed.")

    #mmse_opts = {"composite": True, "refined": False, "fuzzy": True}
    mmse_opts = {"composite": True, "fuzzy": True}
    if len(traj.shape) == 1:
        mmse, info = neurokit2.entropy_multiscale(traj, dimension=2, **mmse_opts)
        return mmse, info

    traj = standardize_ts(traj) # traj is TxD
    all_mse = list()
    all_info = []

    # now D by T, where sol_coord is one dimension across all time
    for sol_coord in traj.T:
        all_mse.append(
            neurokit2.entropy_multiscale(sol_coord, dimension=2, **mmse_opts)[0]
        )
        all_info.append(
            neurokit2.entropy_multiscale(sol_coord, dimension=2, **mmse_opts)[1]["Value"]
        )
    
    if return_info == True:
        return np.sum(all_mse), pd.DataFrame(all_info)
    
    return np.sum(all_mse)

## Principal Component Analysis
def pca(data, threshold=0.80):
    # Perform PCA
    pca = PCA()
    pca.fit(data)

    # Calculate cumulative explained variance
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the number of components that explain at least 80% of the variance
    n_components = np.argmax(cumulative_explained_variance >= threshold) + 1

    return n_components

"""
change params:
model -> trajectory data array, timepoints data array, rhs equations
see if we can bypass tpts and rhs, by working only with trajectory data.

tpts needed for: 
 1. calculating an average timestep dt for Euler
 2. accessing the RHS/Jacobian at a trajectory point
 3. normalizing the sum of each dimension's lyapunov exponent over time to find an average

 rhs needed for:
 1. 
 New Args:
 trajectory (ndarray): (n, timesteps, embed_dim) Trajectories tensor.
"""
#def find_lyapunov_exponents(
#    model, traj_length, pts_per_period=500, tol=1e-8, min_tpts=10, **kwargs
#):
def find_lyapunov_exponents(
    trajectory, tpts, traj_length, model, pts_per_period=500, tol=1e-8, min_tpts=10, **kwargs
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
#/    d = np.asarray(model.seed).shape[-1]

    # get the dimensionality
    d = np.asarray(trajectory).shape[-1]

#/    tpts, traj = model.make_trajectory(
#/        traj_length, pts_per_period=pts_per_period, resample=True, return_times=True,
#/        postprocessing=False,
#/        **kwargs
#/    )

    #dt is actually the average timestep of the system (Gilpin Typo) used for Backward Euler
    dt = np.median(np.diff(tpts))
#/    # traj has shape (traj_length, d), where d is the dimension of the system
#/    # tpts has shape (traj_length,)
#/    # dt is the dimension of the system

    # make an identity matrix of dimension d
    u = np.identity(d)

    # Contains all full lyapunov spectrums for every point along trajectory
    # a list/vector of subvectors; subvectors contain lyapunov spectrum of a point in time
    all_lyap = list()  

    # iterates over each time point t and corresponding state X.
    # If the model does not provide a Jacobian (model.jac), 
    # it computes the Jacobian numerically using jac_fd
    # using the righthand side (Gilpin's systems have an RHS property)
        
    for i in range(traj_length): # Compute the Jacobian numerically at time t and state x
        t = tpts[0][i] # for some reason, gilpin's makedata returns a single array of
        # float timepoints, which is nested in another array (hence first indexing 0)
        x = trajectory[i]
        rhsy = lambda a: np.array(model.rhs(a, t)) # define a function 'rhsy', using the model's right hand side diffeq
        jacval = jac_fd(rhsy, x) # 'a' is a dummy variable that jac_fd plugs values into when it calls rhys.
        
#/    # for i in range(traj_length):
#/    for i, (t, X) in enumerate(zip(tpts, traj)):
#/        X = traj[i]
#/
#/        if model.jac(model.ic, 0) is None: 
#/            rhsy = lambda x: np.array(model.rhs(x, t)) 
#/            jacval = jac_fd(rhsy, X)
#/        else: # If the model provides a Jacobian method, use it directly
#/            jacval = np.array(model.jac(X, t))

        # If postprocessing is applied to a trajectory, transform the jacobian into the
        # new coordinates.
        if hasattr(model, "_postprocessing"):
            x0 = np.copy(x)
            y2h = lambda y: model._postprocessing(*y)
            dhdy = jac_fd(y2h, x0)
            dydh = np.linalg.inv(dhdy)  # dy/dh
            ## Alternate version if good second-order fd is ever available
            # dydh = jac_fd(y2h, X0, m=2, eps=1e-2) @ rhsy(X0) + jac_fd(y2h, y0) @ jac_fd(rhsy, X0))

#/       # If postprocessing is applied to a trajectory, transform the jacobian into the
#/       # new coordinates.
#/       if hasattr(model, "_postprocessing"):
#/           X0 = np.copy(X)
#/           y2h = lambda y: model._postprocessing(*y)
#/           dhdy = jac_fd(y2h, X0)
#/           dydh = np.linalg.inv(dhdy)  # dy/dh
#/           ## Alternate version if good second-order fd is ever available
#/           # dydh = jac_fd(y2h, X0, m=2, eps=1e-2) @ rhsy(X0) + jac_fd(y2h, y0) @ jac_fd(rhsy, X0))
#/           jacval = dhdy @ jacval @ dydh

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

    all_lyap = np.array(all_lyap) # convert list of lists into nparray
    # average each dimension's lyapunov over time (axis = 0)
    # by summing lyapunov at each point/time for a given dimension, and then normalizing over total time elapsed
    # results in a single vector that is the spectrum of average lyapunov exponents
    final_lyap = np.sum(all_lyap, axis=0) / (dt * traj_length)
    return np.sort(final_lyap)[::-1] # return the average exponents in descending order

def find_lyapunov_exponents_og(
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

## Maximum Lyapunov Exponent
def find_max_lyapunov(spectrum0):
    max_exp = spectrum0[0]
    for exp in spectrum0[1:]:
        if abs(exp) > abs(max_exp):
            max_exp = exp
    return max_exp

## Kaplan Yorke Dimension
def kaplan_yorke_dimension(spectrum0):
    """Calculate the Kaplan-Yorke dimension, given a list of
    Lyapunov exponents"""

    if np.all(spectrum0 < 0):
        return 0

    spectrum = np.sort(spectrum0)[::-1]
    d = len(spectrum)
    cspec = np.cumsum(spectrum)
    j = np.max(np.where(cspec >= 0))
    if j > d - 2:
        j = d - 2
        warnings.warn(
            "Cumulative sum of Lyapunov exponents never crosses zero. System may be ill-posed or undersampled."
        )
    dky = 1 + j + cspec[j] / np.abs(spectrum[j + 1])

    return dky

## Pesin Entropy
def pesin(lyapval):
    all_estimates_lyap = list()
    all_estimates_lyap.append(lyapval)
    
    all_estimates_pesin = list()
    all_estimates_pesin.append(np.sum(np.array(lyapval)[np.array(lyapval) > 0]))

    pesin_entropy = np.median(all_estimates_pesin)

    return pesin_entropy