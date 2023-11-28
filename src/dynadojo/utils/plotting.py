import numpy as np
import pandas as pd
import seaborn as sns

def _assign_labels(data, labels):
    assert len(data) == len(labels)
    for frameidx, frame in enumerate(data):
        data[frameidx] = frame.assign(id=labels[frameidx])


def plot_metric(data, xcol, ycol, idlabels=None, xlabel=None, ylabel=None, hue="id", log=True, estimator=np.median, errorbar=("pi", 100), **kwargs):
    if idlabels:
        _assign_labels(data, idlabels)

    sns.set_context("paper")
    sns.set_theme(style="ticks")
    sns.despine()

    if not isinstance(data, pd.DataFrame):
        data = pd.concat(data)
   
    # ax = sns.catplot(data=data, x=xcol, y=ycol, hue=hue, estimator=estimator, kind="point", errorbar=errorbar)
    if type(ycol) == list: 
        #if multiple ycols, melt them and plot as separate lines
        data = data[[xcol] + ycol]
        data = pd.melt(data, [xcol])
        ax = sns.lineplot(data=data, x=xcol, y='value', hue="variable", estimator=estimator, errorbar=errorbar, **kwargs)
    else:
        ax = sns.lineplot(data=data, x=xcol, y=ycol, hue=hue, estimator=estimator, errorbar=errorbar, **kwargs)

    ax.set(xticks=data[xcol].unique())
    if log:
        ax.set(xscale="log")
        ax.set(yscale="log")
        ax.get_yaxis().get_major_formatter().labelOnlyBase = False

    if xlabel:
       ax.set(xlabel=xlabel)
    if ylabel:
       ax.set(ylabel=ylabel)

    return ax


def plot_target_error(data, xcol, ycol, 
            idlabels=None, xlabel=None, ylabel=None, 
            hue="id", log=True, estimator=np.median, 
            errorbar=("pi", 50), error_col="error", 
            target_error=0.1, **kwargs
            ):

    target_error = data['target_error'].unique()[0] #first unique target error
    if idlabels:
        _assign_labels(data, idlabels)

    if not isinstance(data, pd.DataFrame):
        data = pd.concat(data)

    

    # remove all rows where n_target is -1 or np.inf
    filtered = data[data["n_target"] != -1]
    filtered = data[data["n_target"] != np.inf]
    filtered = data[data[error_col] <= target_error]
    # for each rep in the x dim, get the lowest y that was successful
    successes = filtered.loc[filtered.groupby(["id", "rep", xcol])[ycol].idxmin()].reset_index(drop=True)

    assert (len(successes) > 0)
    return plot_metric(successes, xcol, ycol, None, xlabel, ylabel, hue, log, estimator, errorbar, **kwargs)
