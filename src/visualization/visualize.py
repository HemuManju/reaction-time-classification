import pandas as pd
import seaborn as sb
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from .utils import *
sb.set()


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
    fname.sort(reverse=True)

    labels = ['Not included', 'Included']

    df = pd.DataFrame()
    # Form the dataframe
    for i, item in enumerate(fname):
        data = read_model_log(item)
        for j, performance_level in enumerate(config['performance_level']):
            temp_df = pd.DataFrame(columns=['accuracy', 'task_information', 'subject_information'])
            temp = np.sort(-data[performance_level]['accuracy'])[0:10]
            temp_df['accuracy'] = -temp
            temp_df['task_information'] = labels[i]
            temp_df['subject_information'] = performance_level
            df = df.append(temp_df, ignore_index=True)

    # perform statistical analysis
    p_value = [1,1,1]
    for j, performance_level in enumerate(config['performance_level']):
        temp = df[df['subject_information']==performance_level]
        dummy_1 = temp[temp['task_information']=='Included']
        dummy_2 = temp[temp['task_information']=='Not included']
        t, p_value[j] = ttest_ind(dummy_1['accuracy'].values, dummy_2['accuracy'].values)

    plt.figure(figsize=(5, 5), dpi=80)
    # Bar plot
    plt.rcParams['axes.labelweight'] = 'bold'
    color = ['darkgrey', 'lightgrey', 'whitesmoke']
    ax = sb.barplot(x='task_information', y='accuracy', hue='subject_information', data=df, capsize=0.05, linewidth=1, edgecolor=".2", palette=color)

    # Add hatches
    # add_hatches(ax)

    # Add annotations
    x_pos = [-0.25, 0, 0.25]
    y_pos = [0.60, 0.65, 0.70]
    for i, p in enumerate(p_value):
        x1, x2 = x_pos[i], x_pos[i]+1
        annotate_significance(x1, x2, y_pos[i], p)

    ax.axhline(y=0.33, xmin=0, xmax=1, linestyle='--', color='k', label='chance')
    ax.tick_params(labelsize=14)
    ax.set_ylim([0,0.75])
    ax.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=14)
    ax.set_ylabel('Classification accuracy', fontsize=14)
    ax.set_xlabel('Task type information', fontsize=14)
    plt.tight_layout()
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
    sb.boxplot(subject_df['reaction_time'].values-0.015, ax=ax_box)
    sb.distplot(subject_df['reaction_time'], ax=ax_hist, kde=False, norm_hist=True)

    # Fit the inverse gaussian distribution
    xt = plt.xticks()[0]
    xmin, xmax = min(xt), max(xt) + 0.1
    lnspc = np.linspace(xmin, xmax, 200)
    result = stats.invgauss.fit(subject_df['reaction_time'].values)
    pdf_invgauss = stats.invgauss.pdf(lnspc, mu=result[0], loc=result[1], scale=result[2])
    plt.plot(lnspc, pdf_invgauss, color='#465F95')

    # Append the 25, 75 percentile
    x_25 = stats.invgauss.ppf(0.25, mu=result[0], loc=result[1], scale=result[2])
    x_75 = stats.invgauss.ppf(0.75, mu=result[0], loc=result[1], scale=result[2])
    plt.axvline(x=x_25, color='#3C3D40')
    plt.axvline(x=x_75, color='#3C3D40')

    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    ax_hist.set_axisbelow(True)
    # ax_hist.grid()
    plt.xlim([0.15,1.2])
    plt.xlabel('Reaction time')
    plt.tight_layout()
    plt.show()

    return None
