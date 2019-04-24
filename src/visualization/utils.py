from pathlib import Path
import pickle
import matplotlib


def read_dataframe(path):
    """Save the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataframe : dict
        dictionary of pandas dataframe to save


    """

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def read_model_log(read_path):
    """Read the model log.

    Parameters
    ----------
    read_path : str
        Path to read data from.

    Returns
    -------
    dict
        model log.

    """
    with open(read_path, 'rb') as handle:
        data = pickle.load(handle)

    return data


def figure_asthetics(ax):
    """Change the asthetics of the given figure (operators in place).

    Parameters
    ----------
    ax : matplotlib ax object

    """
    matplotlib.rcParams['font.family'] = "Arial"
    ax.set_axisbelow(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    return None
