import collections
import numpy as np
from pathlib import Path

from scipy.stats import invgauss

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from imblearn.under_sampling import RandomUnderSampler

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

    if sigma == 0:
        f = np.zeros_like(rt)
    else:
        f = 1 / (rt**2 * sigma * np.sqrt(2 * np.pi)) * np.exp(
            -(mu * rt - 1)**2 / (2 * rt**2 * sigma**2))
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
        value.append(
            invgauss.ppf(percentile,
                         mu=result[0],
                         loc=result[1],
                         scale=result[2]))

    return np.asarray(value)


def create_classification_data(df, features, predicted_variable, config):
    """Create a classification dataset with features and convert
    continous value of reaction time to classes by fitting
    an inverse gaussian distribution.

    Parameters
    ----------
    df : dataframe
        Pandas dataframe.
    features : list
        A list of features from configuration file.
    predicted_variable : list
        A list of predicted variable (response time).

    Returns
    -------
    array
        Array of x and y with reaction time converted to classes.

    """
    # Initialise
    x = np.empty((0, len(features)))
    y = np.empty((0, len(predicted_variable)))

    for subject in df['subject'].unique():
        df_temp = df[df['subject'] == subject]
        x_temp = df_temp[features].values
        y_temp = df_temp[predicted_variable].values
        y_dummy = y_temp.copy()
        percentile = inverse_gaussian_percentile(y_temp,
                                                 [0.0001, 0.25, 0.75, 0.9999])
        # Get percentile and divide into class
        for i in range(len(percentile) - 1):
            temp = (y_temp >= percentile[i]) & (y_temp <= percentile[i + 1])
            y_dummy[temp] = i
        # z-score of the features
        x_dummy = x_temp
        x = np.vstack((x, x_dummy))
        y = np.vstack((y, y_dummy))
    # Balance the dataset
    rus = RandomUnderSampler()
    x, y = rus.fit_resample(x, y)

    return x, y


def create_feature_set(config):
    """Generate different combination of features
    with task type and without task type.

    Parameters
    ----------
    config : yaml
        The configuration file.

    Returns
    -------
    dict
        A dictionary with different training data.

    """

    predicted_variable = ['reaction_time']

    x, y = {}, {}

    if config['include_task_type']:
        features = config['classification_features'] + ['task_type']
    else:
        features = config['classification_features']

    for i, item in enumerate(config['performance_level']):
        read_path = Path(__file__).parents[2] / config['processed_dataframe']
        df = read_dataframe(read_path)
        if item != 'all_subjects':
            df = df[df['performance_level'] == item]
        x[item], y[item] = create_classification_data(df, features,
                                                      predicted_variable,
                                                      config)

    return x, y


def reaction_time_classification(config):
    """Perform reaction time classification with different
    features using Adaptive boosting of decision trees.

    Parameters
    ----------
    config : yaml
        The configuration file.

    Returns
    -------
    list
        Accuracy of classification.

    """

    X, Y = create_feature_set(config)
    output = collections.defaultdict(dict)
    for key in X.keys():
        accuracy, y_pred_array, y_true_array = [], [], []
        for i in range(20):
            x_train, x_test, y_train, y_test = train_test_split(
                X[key], Y[key], test_size=config['test_size'])
            # Classifier
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                                     algorithm="SAMME",
                                     n_estimators=200)
            clf.fit(x_train, y_train.ravel())
            # Predictions
            y_pred = clf.predict(x_test)
            y_pred_array.append(y_pred)
            y_true_array.append(y_test)
            accuracy.append(accuracy_score(y_test, y_pred))

        # Select top 10
        temp = -np.sort(-np.asarray(accuracy))[0:10]
        print(np.mean(temp), np.std(temp), key)  # Sanity check
        # Model to save
        output[key]['trained_model'] = clf
        output[key]['accuracy'] = np.asarray(accuracy)
        output[key]['prediction'] = np.asarray(y_pred_array)
        output[key]['true'] = np.asarray(y_true_array)
        output[key]['confusion_matrix'] = np.asarray(y_true_array)

    return output
