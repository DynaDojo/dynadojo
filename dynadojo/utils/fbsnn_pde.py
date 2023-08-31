from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np

MAX_LINES = 30



    
def plot(grid: list[np.ndarray], timesteps, target_dim: int = 3, max_lines=MAX_LINES, labels: list[str] = None):
    grid = np.array([x[:max_lines] for x in grid])

    time = np.linspace(0, 1, timesteps+1)
    print(grid.shape)

    for idx, dataset in enumerate(grid):
        print(dataset.shape)
        print(idx)
        for i in range(len(dataset)):
            plt.plot(time, dataset[i,:,:], label=labels[idx], color=f'C{idx}')

    if labels:
        plt.legend()
        

    plt.show()
