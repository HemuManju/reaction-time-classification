import pickle
import deepdish as dd


def read_dataframe(path):
    """Save the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataframe : dict
        dictionary of pandas dataframe to save


    """

    data = dd.io.load(path)

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
