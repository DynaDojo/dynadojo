import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import PCA

MAX_LINES = 30


def _plot2d(trajs_grid, labels: list = None):
    fig = plt.figure()
    ax = fig.add_subplot()
    for i, trajs in enumerate(trajs_grid):
        if labels is None:
            line_collection = LineCollection(trajs, color=f"C{i}")
        else:
            line_collection = LineCollection(trajs, color=f"C{i}", label=labels[i])
        ax.add_collection(line_collection)
        ax.scatter(trajs[:, 0, 0], trajs[:, 0, 1], color=f"C{i}", marker="o")
    minima = trajs_grid.min(axis=(0, 1, 2))
    maxima = trajs_grid.max(axis=(0, 1, 2))
    ax.set_xlim(minima[0], maxima[0])
    ax.set_ylim(minima[1], maxima[1])
    if labels is not None:
        ax.legend()
    return fig, ax


def _plot3d(grid, labels: list = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i, trajs in enumerate(grid):
        if labels is None:
            line_collection = Line3DCollection(trajs, color=f"C{i}")
        else:
            line_collection = Line3DCollection(trajs, color=f"C{i}", label=labels[i])
        ax.add_collection(line_collection)
        ax.scatter(trajs[:, 0, 0], trajs[:, 0, 1], trajs[:, 0, 2], color=f"C{i}", marker="o")
    minima = grid.min(axis=(0, 1, 2))
    maxima = grid.max(axis=(0, 1, 2))
    ax.set_xlim(minima[0], maxima[0])
    ax.set_ylim(minima[1], maxima[1])
    ax.set_zlim(minima[2], maxima[2])
    if labels is not None:
        ax.legend()
    return fig, ax


def _plot4d(trajs_grid, pca: PCA, labels: list = None):
    assert pca.n_components == 2 or pca.n_components == 3
    trajs_grid = _apply_pca_to_grid(trajs_grid, pca)
    if pca.n_components == 2:
        return _plot2d(trajs_grid, labels)
    else:
        return _plot3d(trajs_grid, labels)


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


def plot(grid: list[np.ndarray], target_dim: int = 3, max_lines=MAX_LINES, labels: list[str] = None):
    grid = np.array([x[:max_lines] for x in grid])
    dim = grid.shape[-1]
    assert target_dim <= dim
    assert target_dim <= 3
    if dim == 2:
        return _plot2d(grid, labels)
    elif dim == 3 and target_dim == 3:
        return _plot3d(grid, labels)
    else:
        pca = make_pca(grid, n_components=target_dim)
        return _plot4d(grid, pca, labels)
