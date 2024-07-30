import matplotlib.pyplot as plt
import numpy as np

MAX_LINES = 30
def _plot1d(datasets, labels, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    for i, data in enumerate(datasets):
        time_steps = np.arange(data.shape[1])  # Assuming the second dimension is time steps
        if labels is None:
            ax.scatter(time_steps, data[0, :, 0], color=f"C{i}", marker="o", alpha=0.8)
        else:
            ax.scatter(time_steps, data[0, :, 0], color=f"C{i}", marker="o", alpha=0.8, label=labels[i])  
   

    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Value')
    

    if labels:
        ax.legend()
    if title:
        ax.set_title(title)
    return ax.figure, ax

def _plot2d(trajs_grid, labels: list = None, ax : plt.Axes = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    else:
        fig = ax.get_figure()
    for i, trajs in enumerate(trajs_grid):
        if labels is None:
            ax.scatter(trajs[0, :, 0], trajs[0, :, 1], color=f"C{i}", marker="o", alpha=0.8)
        else:
            ax.scatter(trajs[0, :, 0], trajs[0, :, 1], color=f"C{i}", marker="o", alpha=0.8, label=labels[i])

    minima = trajs_grid.min(axis=(0, 1, 2))
    maxima = trajs_grid.max(axis=(0, 1, 2))
    ax.set_xlim(minima[0], maxima[0])
    ax.set_ylim(minima[1], maxima[1])
    
    if labels is not None:
        ax.legend()
    return fig, ax


def plot(
        datasets: list[np.ndarray], 
        target_dim: int = 3, #ok will need to fix that 
        max_lines=MAX_LINES, 
        labels: list[str] = None,
        title = None
    ):
    fig = plt.figure()
    ax = fig.add_subplot()
    datasets = np.array([x[:max_lines] for x in datasets])
    dim = datasets.shape[-1] #what it is giving is 2
    if dim == 2:
         fig, ax = _plot2d(datasets, labels, ax=ax)
    if dim == 1:
         fig, ax = _plot1d(datasets, labels, ax=ax)
    if title:
        ax.set_title(title)
    return fig, ax