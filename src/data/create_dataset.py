import pandas as pd
import scipy.io as sio
from pathlib import Path
import feather
import deepdish as dd

from sklearn.model_selection import train_test_split


def read_dataframe(path):
    """
    Read the DataFrame.

    Parameters
    ----------
    path : str
        Path to DataFrame.

    Returns
    -------
    DataFrame
        Stored DataFrame in the path.

    """

    data = dd.io.load(path)

    return data


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

    path = Path(
        __file__).parents[2] / config['raw_data_path'] / 'matlab_data.mat'
    data = sio.loadmat(str(path))['local_data']

    return data


def create_secondary_dataframe(config):
    """Create secondary dataset.

    Parameters
    ----------
    config : yaml
        The configuration data file.

    Returns
    -------
    pandas dataframe

    """

    path = Path(
        __file__).parents[2] / config['raw_data_path'] / 'secondary_data.xls'
    dataframe = pd.read_excel(str(path))
    # Add the performance level
    experts_id = config['expert_id']
    performance = ['low_performer'] * len(config['subjects'])
    for i, _ in enumerate(performance):
        if i in experts_id:
            performance[i] = 'high_performer'
    dataframe['performance_level'] = performance

    return dataframe


def create_dataframe(subjects, config):
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
    dataframe = pd.DataFrame()
    for i, subject in enumerate(subjects):
        data[subject] = matlab_data[:, :, i]
        df_temp = pd.DataFrame(matlab_data[:, :, i],
                               columns=config['features'])
        df_temp['subject'] = subject
        # Append task type
        df_temp.loc[df_temp.task_stage <= 3, 'task_type'] = 0
        df_temp.loc[df_temp.task_stage >= 4, 'task_type'] = 1

        # Append task difficulty
        # Visual
        df_temp.loc[(df_temp.task_stage <= 2) & (df_temp.task_type == 0),
                    'task_difficulty'] = 1
        df_temp.loc[(df_temp.task_stage == 3) & (df_temp.task_type == 0),
                    'task_difficulty'] = 2
        # Motor
        df_temp.loc[(df_temp.task_stage == 4) & (df_temp.task_type == 1),
                    'task_difficulty'] = 1
        df_temp.loc[(df_temp.task_stage == 5) & (df_temp.task_type == 1),
                    'task_difficulty'] = 2

        if subject in config['expert']:
            df_temp['performance_level'] = 'high_performer'
        else:
            df_temp['performance_level'] = 'low_performer'
        dataframe = dataframe.append(df_temp, ignore_index=True)
    secondary_dataframe = create_secondary_dataframe(config)

    # Remove nan and zeros
    dataframe.dropna(inplace=True)
    dataframe.loc[~(dataframe == 0).all(axis=1)]
    dataframe = dataframe[dataframe['reaction_time'] != 0]
    dataframe = dataframe[dataframe['task_stage'] != 1]

    return data, dataframe, secondary_dataframe


def create_r_dataframe(config, save_as_csv=True):
    """Create a r dataframe.

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

    train_save_path = Path(__file__).parents[2] / config['r_dataframe_train']
    test_save_path = Path(__file__).parents[2] / config['r_dataframe_test']
    x_train, x_test = train_test_split(df,
                                       test_size=0.4,
                                       random_state=42,
                                       stratify=df['subject'])
    # Sort them to align subjects
    x_train.sort_index(inplace=True)
    x_test.sort_index(inplace=True)
    if save_as_csv:
        x_train.to_csv(train_save_path, index=False)
        x_test.to_csv(test_save_path, index=False)
    else:
        feather.write_dataframe(df, train_save_path)

    return None
