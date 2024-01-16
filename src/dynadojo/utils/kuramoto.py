import numpy as np
import pylab as plt

MAX_LINES = 30


def plot(grid: list[np.ndarray], timesteps, max_lines=MAX_LINES, dt=0.02, gridlabels: list[str] = None):
    grid = np.array([x[:max_lines] for x in grid])
    
    for idx, dataset in enumerate(grid):
        plt.figure(figsize=(16, 8))

        time = np.linspace(0, timesteps, len(dataset[0])-1)

        for comp in range(len(dataset[0][0])):
            plt.subplot(len(dataset[0][0]),1,comp+1)
            plt.plot(time, np.diff(dataset[0][:,comp])/dt,'r')
            plt.ylabel(comp+1)

        plt.suptitle(gridlabels[idx])
    plt.show()
