import pickle
from pathlib import Path
import numpy as np
import pandas as pd


def create_classification_data(config, features, predicted_variable):
    """Create a classification dataset with features.

    Parameters
    ----------
    config : yaml
        The configuration file.
    features : list
        A list of features from configuration file.
    predicted_variable : list
        A list of predicted variable (response time).

    Returns
    -------
    array
        Array of x and y with reaction time converted to classes.

    """

    read_path = Path(__file__).parents[2] / config['processed_dataframe']
    df = read_dataframe(read_path)

    #Initialise
    x = np.empty((0, len(features)))
    y = np.empty((0, len(predicted_variable)))

    for subject in config['subjects']:
        df_temp = df[df['subject']==subject]
        x_temp = df_temp[features]
        y_temp = np.log(df_temp[predicted_variable].values)
        dummy = y_temp
        percentile = np.percentile(y_temp, [0, 25, 75, 100])
        # Get percentile and divide into class
        for i in range(len(percentile)-1):
            temp = (y_temp >= percentile[i]) & (y_temp <= percentile[i+1])
            dummy[temp] = i

        x = np.vstack((x, x_temp))
        y = np.vstack((y, dummy))

    return x, y


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


def model_logging(config, info, model):

    return None
