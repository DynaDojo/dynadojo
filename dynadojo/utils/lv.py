import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import PCA

MAX_LINES = 30


def _plot2d(trajs_grid, gridlabels: list = None):
    fig = plt.figure()
    ax = fig.add_subplot()
    for i, trajs in enumerate(trajs_grid):
        if gridlabels is None:
            line_collection = LineCollection(trajs, color=f"C{i}")
        else:
            line_collection = LineCollection(
                trajs, color=f"C{i}", label=gridlabels[i])
        ax.add_collection(line_collection)
        ax.scatter(trajs[:, 0, 0], trajs[:, 0, 1], color=f"C{i}", marker="o")
    minima = trajs_grid.min(axis=(0, 1, 2))
    maxima = trajs_grid.max(axis=(0, 1, 2))
    ax.set_xlim(minima[0], maxima[0])
    ax.set_ylim(minima[1], maxima[1])
    if gridlabels is not None:
        ax.legend()
    return fig, ax


def _plot3d(grid, fig, gridlabels: list = None):
    ax = fig.add_subplot(projection='3d')
    for i, trajs in enumerate(grid):
        if gridlabels is None:
            line_collection = Line3DCollection(trajs, color=f"C{i}")
        else:
            line_collection = Line3DCollection(
                trajs, color=f"C{i}", label=gridlabels[i])
        ax.add_collection(line_collection)
        ax.scatter(trajs[:, 0, 0], trajs[:, 0, 1],
                   trajs[:, 0, 2], color=f"C{i}", marker="o")
    minima = grid.min(axis=(0, 1, 2))
    maxima = grid.max(axis=(0, 1, 2))
    ax.set_xlim(minima[0], maxima[0])
    ax.set_ylim(minima[1], maxima[1])
    ax.set_zlim(minima[2], maxima[2])
    if gridlabels is not None:
        ax.legend()
    plt.show()

    return fig, ax


def _plot4d(trajs_grid, pca: PCA, gridlabels: list = None):
    assert pca.n_components == 2 or pca.n_components == 3
    trajs_grid = _apply_pca_to_grid(trajs_grid, pca)
    if pca.n_components == 2:
        return _plot2d(trajs_grid, gridlabels)
    else:
        return _plot3d(trajs_grid, gridlabels)


def make_pca(trajs: np.ndarray, n_components=3):
    dim = trajs.shape[-1]
    X = trajs.reshape((-1, dim))
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca


def _apply_pca_to_grid(trajs_grid, pca) -> np.ndarray:
    dim = trajs_grid.shape[-1]
    X = trajs_grid.reshape((-1, dim))
    result = pca.transform(X)
    target_shape = list(trajs_grid.shape)
    target_shape[-1] = pca.n_components
    result = result.reshape(target_shape)
    return result


def plot(grid: list[np.ndarray], target_dim: int = 3, max_lines=MAX_LINES, specieslabels: list[str] = None, gridlabels: list[str] = None):
    grid = np.array([x[:max_lines] for x in grid])
    dim = grid.shape[-1]
    fig = plt.figure(figsize=(16, 4))
    posidx = 1
    for dataset in grid:
        pos = 100 + len(grid) * 10 + posidx
        ax = fig.add_subplot(1, len(grid), posidx)

        for i in range(len(dataset[0][0])):
            ydata = dataset[0][:, i]
            if specieslabels:
                ax.plot(ydata, label=specieslabels[i])
                ax.legend()
            else:
                ax.plot(ydata)
        posidx += 1

    assert target_dim <= dim
    assert target_dim <= 3
    if dim == 2:
        return _plot2d(grid, gridlabels)
    elif dim == 3 and target_dim == 3:
        return _plot3d(grid, gridlabels)
    else:
        pca = make_pca(grid, n_components=target_dim)
        return _plot4d(grid,  pca, gridlabels)
