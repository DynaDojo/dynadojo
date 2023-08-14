import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import PCA

MAX_LINES = 30


def plot(grid: list[np.ndarray], target_dim: int = 3, max_lines=MAX_LINES, specieslabels: list[str] = None, gridlabels: list[str] = None):
    grid = np.array([x[:max_lines] for x in grid])
    fig = plt.figure(figsize=(16, 4))

    for dataset in grid:
        ax = fig.add_subplot()

        for i in range(len(dataset[0])):
            ydata = dataset[0][i]
            ax.plot(ydata)

    plt.show()
