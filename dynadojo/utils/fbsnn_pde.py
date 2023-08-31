from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np

MAX_LINES = 30


    
def plot(grid: list[np.ndarray], timesteps, target_dim: int = 3, max_lines=MAX_LINES, labels: list[str] = None):
    grid = np.array([x[:max_lines] for x in grid])

    time = np.linspace(0,5,timesteps+1)
    _, ax = plt.subplots()

    for idx, dataset in enumerate(grid):
        trajs = [np.column_stack([time, cond]) for cond in dataset]
        
        line_collection = LineCollection(trajs, color=f"C{idx}", label=labels[idx])
        ax.add_collection(line_collection)

        ax.scatter(np.zeros(len(dataset)),dataset[:,0], color=f"C{idx}", marker="o")
       
    minima = grid.min(axis=(0, 1, 2))
    maxima = grid.max(axis=(0, 1, 2))

    ax.set_ylim(minima-1, maxima+1)

    if labels:
        plt.legend()
        
    plt.show()
