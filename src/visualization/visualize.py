import pandas as pd
import seaborn as sb
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from .utils import read_dataframe, figure_asthetics


def plot_detection_false_alarm_rate(config):
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
