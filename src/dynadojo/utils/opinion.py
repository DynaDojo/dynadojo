import matplotlib.pyplot as plt
import numpy as np

MAX_LINES = 30


def plot(
        datasets: list[np.ndarray],
        target_dim: int = 3, 
        max_lines=MAX_LINES, 
        datalabels: list[str] = None, 
        labels: list[str] = None,
        title: str = None
    ):

    datasets = np.array([x[:max_lines] for x in datasets]) 
    fig = plt.figure(figsize=(16, 4))

    posidx = 1
    for idx, dataset in enumerate(datasets):
        ax = fig.add_subplot(1, len(datasets), posidx)

        ydata = dataset[0]
        if(datalabels):
            ax.plot(ydata, label=datalabels)
        else:
            ax.plot(ydata)
        ax.set_title(labels[idx])
        ax.legend()
        posidx += 1


    if title:
        fig.suptitle(title)
    return fig, ax
