import numpy as np
import pandas as pd
from pathlib import Path
from .utils import read_dataframe
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def t_sne(config):
    """t-sne analysis of tele-operation data.

    Parameters
    ----------
    config : yaml
        The configuration file.

    Returns
    -------
    None

    """
    read_path = Path(__file__).parents[2] / config['processed_dataframe']
    df = read_dataframe(read_path)
    df = df[df['task_stage']!=1]

    x = df[['transition_ratio', 'glance_ratio', 'pupil_size',
    'mental_workload', 'high_engagement', 'low_engagement', 'distraction']].values
    y= df['task_stage'].values
    stages = ['stage 1', 'stage 2', 'stage 3', 'stage 4']

    x_embedded = TSNE(n_components=2).fit_transform(x)
    color = ['#edf8fb', '#b3cde3', '#8c96c6', '#8856a7', '#810f7c']
    fig, ax = plt.subplots(figsize=(8, 8))
    for item in [2,3,4,5]:
        ax.scatter(x_embedded [y==item, 0], x_embedded [y==item, 1], c=color[item-1], cmap=plt.cm.Spectral, label=stages[item-2])
    ax
    plt.xlabel('Embedding 1')
    plt.ylabel('Embedding 2')
    plt.legend()
    plt.show()

    return None
