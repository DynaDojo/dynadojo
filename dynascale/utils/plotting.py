import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _assign_labels(data, labels):
    assert len(data) == len(labels)
    for frameidx, frame in enumerate(data):
        data[frameidx] = frame.assign(id=labels[frameidx])


def plot_metric(data, x_col, y_col, labels=None, hue="id", estimator=np.mean, errorbar="sd"):
    if labels:
        _assign_labels(data, labels)

    sns.set_context("paper")
    sns.set_theme(style="ticks")
    sns.despine()

    if not isinstance(data, pd.DataFrame):
        data = pd.concat(data)

    ax = sns.catplot(data=data, x=x_col, y=y_col, hue=hue, estimator=estimator, kind="point", errorbar=errorbar)
    ax.set(xticks=range(len(data[x_col].unique())))

    plt.show()


def plot_target_error(data, x_col, y_col, labels=None, hue="id", estimator=np.mean, errorbar="sd", error_col="error",
                      target_error=0.1):
    if labels:
        _assign_labels(data, labels)

    if not isinstance(data, pd.DataFrame):
        data = pd.concat(data)

    grouped = data.groupby(["id", x_col, y_col])[error_col].mean().reset_index()
    grouped = grouped[grouped[error_col] < target_error]

    assert (len(grouped) > 0)
    plot_metric(grouped, x_col, y_col, labels=None, hue=hue, estimator=estimator, errorbar=errorbar)
