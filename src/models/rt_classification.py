import yaml
import numpy as np
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import svm
from scipy.stats import invgauss, norm
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from .utils import read_dataframe


def recinormal(rt, mu, sigma):
    """Recinormal distribution.

    Parameters
    ----------
    rt : array
        Reaction time vector.
    mu : numeric
        mean of the distribution.
    sigma : numeric
        Variance of the distribution.

    Returns
    -------
    array
        recinormal pdf of given rt.

    """

    if sigma==0:
        f = np.zeros_like(rt)
    else:
        f = 1/(rt**2*sigma*np.sqrt(2*np.pi))*np.exp(-(mu*rt-1)**2/(2*rt**2*sigma**2))
    return f

def inverse_gaussian_percentile(data, percentiles):
    """Fit recinormal distribution and get the values at given percentile values.

    Parameters
    ----------
    data : array
        Numpy array.
    percentile : list
        Percentile list.

    Returns
    -------
    array
        reaction time at given percentile.

    """
    result = invgauss.fit(data)
    value = []
    for percentile in percentiles:
        value.append(invgauss.ppf(percentile, mu=result[0], loc=result[1], scale=result[2]))

    return np.asarray(value)


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
        y_dummy = y_temp.copy()
        percentile = inverse_gaussian_percentile(y_temp, [0.0001, 0.10, 0.90, 0.9999])
        # Get percentile and divide into class
        for i in range(len(percentile)-1):
            temp = (y_temp >= percentile[i]) & (y_temp <= percentile[i+1])
            y_dummy[temp] = i
        # z-score of the features
        x_dummy = x_temp
        x = np.vstack((x, x_dummy))
        y = np.vstack((y, y_dummy))
    # Balance the dataset
    rus = RandomUnderSampler()
    x, y = rus.fit_resample(x, y)

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

    eye_features = ['fixation_rate','transition_ratio', 'glance_ratio', 'pupil_size']
    brain_features = ['mental_workload', 'distraction']
    predicted_variable = ['reaction_time']
    features = [eye_features, brain_features,  brain_features+eye_features]
    if config['include_task']:
        features = [item + ['task_stage'] for item in features]

    x, y = {}, {}
    for i, feature in enumerate(features):
        x[i], y[i] = create_classification_data(config, feature, predicted_variable)

    return x, y


def reaction_time_classification(config):
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
    results = []
    for key in X.keys():
        x_train, x_test, y_train, y_test = train_test_split(X[key], Y[key], test_size=config['test_size'])
        kfold = model_selection.KFold(n_splits=3, random_state=2)
        cart = DecisionTreeClassifier()
        num_trees = 200
        model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=2)
        output = model_selection.cross_val_score(model, x_train, y_train.ravel(), cv=kfold)
        results.append(output.mean())

    return results
