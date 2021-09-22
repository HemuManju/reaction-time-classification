import pickle
# import deepdish as dd
import pandas as pd


def read_dataframe(path):
    """Save the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataframe : dict
        dictionary of pandas dataframe to save
    """

    data = pd.read_hdf(path)
    # with open(path, 'rb') as f:
    #     data = pickle.load(f)

    return data


def model_logging(config, info, model):
    return None


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