import matplotlib.pyplot as plt
import numpy as np

MAX_LINES = 30


def plot(grid: list[np.ndarray], target_dim: int = 3, max_lines=MAX_LINES, datalabels: list[str] = None, gridlabels: list[str] = None):
    grid = np.array([x[:max_lines] for x in grid])
    fig = plt.figure(figsize=(16, 4))

    posidx = 1
    for idx, dataset in enumerate(grid):
        ax = fig.add_subplot(1, len(grid), posidx)

        ydata = dataset[0]
        if(datalabels):
            ax.plot(ydata, label=datalabels)
            ax.legend()
        else:
            ax.plot(ydata)
        ax.set_title(gridlabels[idx])
        posidx += 1

    plt.show()
