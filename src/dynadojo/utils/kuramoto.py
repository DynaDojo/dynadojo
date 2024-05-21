import numpy as np
import pylab as plt
import seaborn as sns

MAX_LINES = 10
MAX_OSCILLATORS = 5

linestyle_tuple = [
     ('solid',                 (0, ())),
     ('dotted',                (0, (1, 1))),
     ('dashed',                (0, (5, 5))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('loosely dotted',        (0, (1, 10))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

def plot(
        datasets: list[np.ndarray],
        max_oscillators:int = MAX_OSCILLATORS, 
        max_lines:int = MAX_LINES,
        dt:float = 0.02, 
        labels: list[str] = None, 
        title: str = None, 
        phase_dynamics:bool =True
    ):
    """
    Plot the Kuramoto model.

    Parameters
    ----------
    datasets : list[np.ndarray]
        The datasets to plot. Datasets must have the same dimensions (number of datasets x [n, timesteps, embed_dim])
    max_oscillators : _type_, optional
        Maximum number of oscillators to plot, by default MAX_LINES=30
    dt : float, optional
        Time interval between timesteps, by default 0.02
    legend_labels : list[str], optional
        Labels for each dataset, by default None
    title : str, optional
        Title of the plot, by default None
    phase_dynamics : bool, optional
        Whether to plot the phase dynamics or the raw time series, by default True
    """
    # check that all datasets have the same dimensions
    assert all([dataset.shape == datasets[0].shape for dataset in datasets]), "All datasets must have the same dimensions"

    if labels:
        assert len(datasets) == len(labels), "labels must be the same length as the length of datasets"

    datasets = np.array(datasets) #convert datasets to a numpy array 

    # truncate number of oscillators to max_oscillators
    if datasets.shape[-1] > max_oscillators:
        datasets = datasets[..., :max_oscillators]

     # truncate number of lines to max_lines
    if datasets.shape[1] > max_lines:
        datasets = datasets[:, :max_lines, ...]
    print(datasets.shape)
    
    oscillators = datasets.shape[-1] #get the number of oscillators

    fig = plt.figure(figsize=(16, 1.8*oscillators))      #create a figure
    axes = fig.get_axes()                    #get the axes of the figure
    
    #create a time vector based on the number of timesteps
    timesteps = datasets.shape[2]
    time = np.arange(0, dt*(timesteps-1), dt)

    for osc in range(oscillators): #for each dimension
        ax = plt.subplot(oscillators,1,osc+1) #plot each dimension in a separate subplot
        for d, dataset in enumerate(datasets):
            with sns.color_palette("Spectral", n_colors=datasets.shape[0]):
                if phase_dynamics:
                    #compute the phase Dynamics of each trajectory
                    data = dataset[:, :, osc]
                    phaseDynamics = (np.diff(data)/dt).T
                    print(phaseDynamics.shape, time.shape)
                    ax.plot(time, phaseDynamics, linestyle=linestyle_tuple[d][1], alpha=0.8)
                else:
                    ax.plot(dataset[:, :, osc].T, linestyle=linestyle_tuple[d][1], alpha=0.8)
        ax.set_ylabel(osc+1) #label the y-axis with the oscillator number
    if labels:
        # plt.legend(legend_labels)
        fig.legend(labels, bbox_to_anchor=(1, 1), loc="upper right", ncols=len(labels))
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes
