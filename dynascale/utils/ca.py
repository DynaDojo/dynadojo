import cellpylib as cpl
import numpy as np
from matplotlib.colors import ListedColormap


def plot(grid: list[np.ndarray], labels: list[str]):
    grid = [x[0] for x in grid]
    cpl.plot_multiple(grid, labels, vmin=0)
