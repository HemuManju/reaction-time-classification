import numpy as np
from pathlib import Path
from scipy.stats import invgauss
from sklearn.preprocessing import normalize
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV

from imblearn.under_sampling import RandomUnderSampler
from .utils import read_dataframe


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
    # Initialise
    x = np.empty((0, len(features)))
    y = np.empty((0, len(predicted_variable)))

    for subject in config['subjects']:
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
        # x_dummy = zscore(x_temp[:,0:-1], axis=0)
        x_dummy = x_temp[:, 0:-1]
        # Add back the task type
        x_dummy = np.hstack((x_dummy, np.expand_dims(x_dummy[:, -1], axis=1)))
        x = np.vstack((x, x_dummy))
        y = np.vstack((y, y_dummy))
    # Balance the dataset
    rus = RandomUnderSampler()
    x, y = rus.fit_resample(x, y)
    x = normalize(x, axis=0)
    print(x.shape, y.shape)

    return x, y


def selected_features(config):
    """Selected features for the classification of reaction time.

    Parameters
    ----------
    config : yaml
        The configuration file.

    Returns
    -------
    list
        A list of selected features.

    """

    eye_features = [
        'fixation_rate', 'transition_ratio', 'glance_ratio', 'pupil_size'
    ]
    brain_features = [
        'mental_workload', 'high_engagement', 'low_engagement', 'distraction'
    ]
    predicted_variable = ['reaction_time']
    features = eye_features + brain_features

    if config['include_task_type']:
        features = features + ['task_type']

    # Dataset creation with all features
    x, y = create_classification_data(config, features, predicted_variable)

    # Estimator
    base_clf = DecisionTreeClassifier(max_depth=2)

    # num_trees = 200
    # clf = BaggingClassifier(base_estimator=base_clf,
    #                         n_estimators=num_trees,
    #                         random_state=2)

    cv = model_selection.StratifiedKFold(5)
    oz = RFECV(base_clf,
               cv=cv,
               scoring='accuracy',
               title='Recursive selection of features')
    oz.fit(x, y)
    for tick in oz.ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    for tick in oz.ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    oz.poof()

    return None
