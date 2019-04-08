import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .utils import read_dataframe, create_classification_data




def feature_selection(config):
    """Generate different combination of features from eye, pupil, and brain.

    Parameters
    ----------
    config : yaml
        The configuration file.

    Returns
    -------
    dict
        A dictionary with different training data.

    """

    eye_features = ['fixation_rate','transition_ratio', 'glance_ratio']
    pupil_size = ['pupil_size']
    brain_features = ['mental_workload', 'avg_mental_workload', 'high_engagement', 'low_engagement', 'distraction']
    predicted_variable = ['reaction_time']
    task_stage = ['']
    features = [pupil_size, eye_features, eye_features+pupil_size, brain_features, brain_features+pupil_size, brain_features+eye_features, eye_features+brain_features+pupil_size]

    x, y = {}, {}
    for i, feature in enumerate(features):
        x[i], y[i] = create_classification_data(config, feature, predicted_variable)

    return x, y


def reaction_time_classification(config):

    X, Y = feature_selection(config)
    clf = {}
    for key in X.keys():
        x_train, x_test, y_train, y_test = train_test_split(X[key], Y[key], test_size=config['test_size'])
        clf[key] = svm.SVC(gamma=config['gamma'], kernel=config['kernel'], decision_function_shape=config['decision_function_shape'])
        clf[key].fit(x_train, y_train.flatten())
        y_pred =  clf[key].predict(x_test)
        print(accuracy_score(y_test, y_pred))


    return None
