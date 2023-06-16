import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
TODO:
- DONE  argument to control log scale
- DONE latex
- DONE add rep_id
- DONE pi -- 
"""


def _assign_labels(data, labels):
    assert len(data) == len(labels)
    for frameidx, frame in enumerate(data):
        data[frameidx] = frame.assign(id=labels[frameidx])


def plot_metric(data, xcol, ycol, idlabels=None, xlabel=None, ylabel=None, hue="id", log=True, estimator=np.median, errorbar=("pi", 50)):
    if idlabels:
        _assign_labels(data, idlabels)

    sns.set_context("paper")
    sns.set_theme(style="ticks")
    sns.despine()

    if not isinstance(data, pd.DataFrame):
        data = pd.concat(data)

   
    ax = sns.catplot(data=data, x=xcol, y=ycol, hue=hue, estimator=estimator, kind="point", errorbar=errorbar)

    ax.set(xticks=range(len(data[xcol].unique())))
    if log:
        ax.set(yscale="log")

    if xlabel:
       ax.set(xlabel=xlabel)
    if ylabel:
       ax.set(ylabel=ylabel)

    plt.show()


def plot_target_loss(data, xcol, ycol, idlabels=None, xlabel=None, ylabel=None, hue="id", log=True, estimator=np.median, errorbar=("pi", 50), error_col="loss", target_loss=0.1):
    if idlabels:
        _assign_labels(data, idlabels)

    if not isinstance(data, pd.DataFrame):
        data = pd.concat(data)

    filtered = data[data[error_col] <= target_loss]
    # for each rep in the x dim, get the lowest y that was successful
    successes = filtered.loc[filtered.groupby(["id", "rep", xcol])[ycol].idxmin()].reset_index(drop=True)

    assert (len(successes) > 0)
    plot_metric(successes, xcol, ycol, None, xlabel, ylabel, hue, log, estimator, errorbar)
