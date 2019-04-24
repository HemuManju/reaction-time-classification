import pandas as pd
import seaborn as sb
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .utils import read_dataframe, figure_asthetics, read_model_log



def plot_classification_accuracy(config):
    """Plots the bar plot of classification accuracy

    Parameters
    ----------
    config : yaml
        The yaml configuration rate.

    Returns
    -------
    None

    """
    # import data from
    read_path = Path(__file__).parents[2] / config['save_path']
    fname = [str(f) for f in read_path.iterdir() if f.suffix == '.pkl']
    fname.sort()
    fig, ax = plt.subplots()
    ax.yaxis.grid(True)
    labels = ['Included', 'Not included']
    x_pos = np.arange(len(labels))

    # Form the dataframe
    for i, item in enumerate(fname):
        data = read_model_log(item)
        temp = np.sort(-data['accuracy'])[0:10]
        accuracy, std = -np.mean(temp), -np.std(temp)
        ax.bar(x_pos[i], accuracy, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)

    ax.axhline(y=0.33, xmin=0, xmax=1, linestyle='--')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Classification accuracy')
    ax.set_xlabel('Task type information')
    plt.show()

    return None


def plot_box_reaction_time(config):
    """Plots the reaction time box plot

    Parameters
    ----------
    config : yaml
        The yaml configuration rate.

    Returns
    -------
    None

    """
    # Using the data from MATLAB file run


    return None


def plot_detection_false_alarm(config):
    """Plots the detection rate and false alarm rate.

    Parameters
    ----------
    config : yaml
        The yaml configuration rate.

    Returns
    -------
    None

    """

    features = ['detection_percent_av','false_detection_av','performance_level']
    read_path = Path(__file__).parents[2] / config['secondary_dataframe']
    df = read_dataframe(str(read_path))

    temp_df = df[features]
    cols_to_norm = ['detection_percent_av','false_detection_av']
    df[cols_to_norm] = MinMaxScaler().fit_transform(df[cols_to_norm])
    data = df

    # Plotting
    fig, ax = plt.subplots()

    markers = {"low_performer": "o", "high_performer": "s"}
    sb.scatterplot(x = df[features[0]], y = df[features[1]], style=df[features[2]], markers=markers, hue=df[features[2]])
    ax.set_axisbelow(True)
    plt.grid()
    plt.ylabel('Normalised false alarm rate')
    plt.xlabel('Normalised detection rate')
    plt.show()

    return None


def plot_reaction_time(subject, config):
    """Plot the reaction time of a subject (from all stage of the mission).

    Parameters
    ----------
    subject : str
        Subject id eg.'8807'.
    config : yaml
        The yaml configuration rate.

    Returns
    -------
    None

    """

    read_path = Path(__file__).parents[2] / config['processed_dataframe']
    df = read_dataframe(str(read_path))
    subject_df = df[df['subject']==subject]

    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

    # Add a graph in each part
    sb.set()
    sb.boxplot(subject_df['reaction_time'], ax=ax_box)
    sb.distplot(subject_df['reaction_time'], ax=ax_hist)

    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    ax_hist.set_axisbelow(True)
    ax_hist.grid()
    plt.xlabel('Reaction time')
    plt.tight_layout()
    plt.show()

    return None
