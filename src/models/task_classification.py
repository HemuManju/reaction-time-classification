import yaml
import numpy as np
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import svm
from scipy.stats import invgauss, norm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from .utils import read_dataframe


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
        x_temp = df_temp[features].values
        y_temp = df_temp[predicted_variable].values
        y_temp[y_temp<=3]=0
        y_temp[y_temp>3]=1
        # x_temp = stats.zscore(x_temp, axis=0)
        x = np.vstack((x, x_temp))
        y = np.vstack((y, y_temp))
    # Balance the dataset
    rus = RandomUnderSampler(random_state=0)
    x, y = rus.fit_resample(x, y)
    # print(sorted(Counter(y.flatten()).items()))

    return x, y


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

    eye_features = ['fixation_rate', 'transition_ratio', 'glance_ratio']
    pupil_size = ['pupil_size']
    brain_features = ['mental_workload','high_engagement', 'low_engagement', 'distraction']
    predicted_variable = ['task_stage']
    task_stage = ['']
    features = [pupil_size, eye_features, eye_features+pupil_size, brain_features, brain_features+pupil_size, brain_features+eye_features, eye_features+brain_features+pupil_size]

    x, y = {}, {}
    for i, feature in enumerate(features):
        x[i], y[i] = create_classification_data(config, feature, predicted_variable)

    return x, y


def task_type_classification(config):
    """Perform reaction time classification with different features.

    Parameters
    ----------
    config : yaml
        The configuration file.

    Returns
    -------
    list
        Accuracy of classification.

    """

    X, Y = feature_selection(config)

    clf = {}
    for key in X.keys():
        x_train, x_test, y_train, y_test = train_test_split(X[key], Y[key], test_size=config['test_size'])
        clf[key] = svm.SVC(gamma=config['gamma'], kernel=config['kernel'], decision_function_shape=config['decision_function_shape'])
        clf[key].fit(x_train, y_train.flatten())
        y_pred =  clf[key].predict(x_test)
        print(accuracy_score(y_test, y_pred))

    return None
