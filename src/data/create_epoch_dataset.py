import pandas as pd
import numpy as np
import scipy.io as sio
from pathlib import Path


def read_matlab_file(config):
    """Reads the matlab file.

    Parameters
    ----------
    config : yaml
        Configuration file.

    Returns
    -------
    array
        Numpy N-D array.

    """

    path = Path(__file__).parents[2] / config['raw_data_path'] / 'matlab_data.mat'
    data = sio.loadmat(str(path))['local_data']

    return data


def create_secondary_dataset(config):
    """Create secondary dataset.

    Parameters
    ----------
    config : yaml
        The configuration data file.

    Returns
    -------
    pandas dataframe

    """

    path = Path(__file__).parents[2] / config['raw_data_path'] / 'secondary_data.xls'
    dataframe = pd.read_excel(str(path))

    return dataframe


def create_dataset(subjects, config):
    """Create dictionary dataset of subjects.

    Parameters
    ----------
    subjects : type
        Description of parameter `subjects`.
    config : type
        Description of parameter `config`.

    Returns
    -------
    type
        Description of returned object.

    """

    matlab_data = read_matlab_file(config)
    data = {}
    df = np.empty((0, matlab_data.shape[1]))
    for i, subject in enumerate(subjects):
        data[subject] = matlab_data[:,:,i]
        df = np.vstack((df, matlab_data[:,:,i]))

    dataframe = pd.DataFrame(df, columns = config['features'])
    secondary_dataframe = create_secondary_dataset(config)

    return data, dataframe, secondary_dataframe
