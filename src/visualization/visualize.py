import pandas as pd
import seaborn as sb
from pathlib import Path
import matplotlib.pyplot as plt
from .utils import read_dataframe, figure_asthetics


def plot_detection_false_alarm_rate(config):


    features = ['detection_percent_av', 'false_detection_av']
    read_path = Path(__file__).parents[2] / config['secondary_dataframe']
    df = read_dataframe(str(read_path))

    temp_df = df[features]
    data = (temp_df-temp_df.min())/(temp_df.max()-temp_df.min())
    data = data.values
    experts_id = config['expert_id']
    marker_style = ['novice']*len(config['subjects'])
    for i, style in enumerate(marker_style):
        if i in experts_id:
            marker_style[i] = 'expert'

    # Plotting
    fig, ax = plt.subplots()
    sb.scatterplot(x = data[:,0], y = data[:,1], style=marker_style)
    ax.set_axisbelow(True)
    plt.grid()
    plt.xlabel('Normalised false alarm rate')
    plt.ylabel('Normalised detection rate')
    plt.show()

    return None
