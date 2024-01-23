import numpy as np
import pylab as plt

MAX_LINES = 30


def plot(datasets: list[np.ndarray], timesteps, max_oscillators=MAX_LINES, dt=0.02, legend_labels: list[str] = None, title = None, phase_dynamics=True):
    """
    Plot the Kuramoto model.

    Parameters
    ----------
    datasets : list[np.ndarray]
        The datasets to plot. Datasets must have the same dimensions. 
    timesteps : _type_
        Number of timesteps
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

    if legend_labels:
        assert len(datasets) == len(legend_labels), "legend_labels must be the same length as the length of datasets"

    datasets = np.array(datasets) #convert datasets to a numpy array 
    # dimensions are (number of datasets in each sublist aka lines, 1, number of timesteps, number of oscilators aka subplots)

    # truncate number of oscillators to max_oscillators
    if datasets.shape[-1] > max_oscillators:
        datasets = datasets[..., :max_oscillators]
    
    datasets = datasets.squeeze(1) #remove the second dimension of the array
    oscillators = datasets.shape[-1] #get the number of oscillators

    fig = plt.figure(figsize=(16, 1.8*oscillators))      #create a figure
    axes = fig.get_axes()                    #get the axes of the figure
        #create a time vector based on the number of timesteps
    time = np.linspace(0, timesteps, timesteps-1)
    for osc in range(oscillators): #for each dimension
        ax = plt.subplot(oscillators,1,osc+1) #plot each dimension in a separate subplot
        if phase_dynamics:
            #compute the phase Dynamics of each trajectory
            phaseDynamics = (np.diff(datasets[...,osc])/dt).T
            plt.plot(time, phaseDynamics)
        else:
            plt.plot(datasets[...,osc].T)
        plt.ylabel(osc+1) #label the y-axis with the oscillator number
    if legend_labels:
        # plt.legend(legend_labels)
        fig.legend(legend_labels, bbox_to_anchor=(1, 1), loc="upper right", ncols=len(legend_labels))
    if title:
        fig.suptitle(title)
    fig.tight_layout()                     #tighten the layout
    plt.show()
    return fig, axes
