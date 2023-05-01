import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plotMetric(df, xDim, yDim, hue=None, estimator=np.mean, errorbar="sd"):


  ax = sns.catplot(data=df, x=xDim, y=yDim, hue=hue, estimator=estimator, kind="point", errorbar=errorbar)
   
  sns.set_context("paper")
  sns.set_theme(style="ticks")
  sns.despine()

  ax.set(xticks=range(len(df[xDim].unique())))

  plt.show()


def plot_metric(*data, x_col, y_col):
    df = pd.concat([data])
    ax = sns.catplot(data=df, x=x_col, y=y_col, hue=None, estimator=np.mean, kind="point", errorbar="sd")

def plotFixedError(df, xDim, yDim, hue=None, estimator=np.mean, errorbar="sd", maxError=0.1):
    print("DURING FIXED ERROR")
    print(df.head(20))
    grouped = df.groupby(["modelID", "dim", "numSamples"])['accuracy'].mean().reset_index()
    grouped = grouped[grouped["accuracy"] < 100]
    print("GROUPED")

    print(grouped.head(20))
    # df=[df['accuracy'] < maxError]


    ax = sns.catplot(data=df, x=xDim, y=yDim, hue=hue, estimator=estimator, kind="point", errorbar=errorbar)

    sns.set_context("paper")
    sns.set_theme(style="ticks")
    sns.despine()

    ax.set(xticks=range(len(df[xDim].unique())))

    plt.show()