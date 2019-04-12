import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import KernelDensity
from .utils import read_dataframe


def estimate_density(config):
    """Kernel density estimation of reaction time.

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
    x = df['reaction_time'].values
    x = x[:, np.newaxis]
    x_plot = np.linspace(0, 2, 1000)[:, np.newaxis]

    fig, ax = plt.subplots()
    for kernel in ['gaussian', 'tophat', 'epanechnikov']:
        print(kernel)
        kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(x)
        log_dens = kde.score_samples(x_plot)
        ax.plot(x_plot[:, 0], np.exp(log_dens), '-',
                label="kernel = '{0}'".format(kernel))

    plt.legend()
    plt.show()

    return None
