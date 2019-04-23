import pandas as pd
import seaborn as sb
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .utils import read_dataframe, figure_asthetics


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
    data = np.array(
    [[0, 42.09, 3.54], [0.5, 50.33, 4.76], [0.1, 44.34, 3.69], [0.6, 51.63, 3.19], [0.2, 44.13, 1.60], [0.7, 49.71, 4.02]])

    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(0, 6, 2):
        plt.bar(data[i:i + 2, 0]/2, data[i:i + 2, 1], yerr=data[i:i + 2, 2], capsize=4, width=0.04)
    plt.legend(['All subjects', 'High performers', 'Low performers'])
    plt.xlabel('Task information')
    plt.xticks([0.05, 0.55], ['Not \n included', 'Included'])
    plt.ylabel('Reaction time classification accuracy')
    plt.subplots_adjust()
    plt.show()

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
