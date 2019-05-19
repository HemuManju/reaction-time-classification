import yaml
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from scipy.optimize import curve_fit
from scipy.stats import invgauss
from data.create_dataset import create_dataframe
from data.create_dataset import create_r_dataframe
from models.t_sne_analysis import t_sne
from features.features_selection import selected_features
from models.rt_classification import reaction_time_classification
from models.task_classification import task_type_classification
from models.density_estimation import estimate_density
from visualization.visualize import plot_classification_accuracy, plot_reaction_time

config = yaml.load(open('config.yml'), Loader=yaml.SafeLoader)

with skip_run('run', 'create_dataset') as check, check():
    data, dataframe, secondary_dataframe = create_dataframe(
        config['subjects'], config)
    save_path = Path(__file__).parents[1] / config['processed_dataframe']
    save_dataset(str(save_path), dataframe, save=True)

    save_path = Path(__file__).parents[1] / config['secondary_dataframe']
    save_dataset(str(save_path), secondary_dataframe, save=True)

    save_path = Path(__file__).parents[1] / config['processed_dataset']
    save_dataset(str(save_path), data, save=True)

with skip_run('skip', 'create_r_dataframe') as check, check():
    create_r_dataframe(config)

with skip_run('skip', 'box_plot_reaction_time') as check, check():
    plot_box_reaction_time(config)

with skip_run('skip', 't_sne_analysis') as check, check():
    t_sne(config)

with skip_run('skip', 'density_analysis') as check, check():
    estimate_density(config)

with skip_run('skip', 'features_selection') as check, check():
    selected_features(config)

with skip_run('skip', 'reaction_time_classification') as check, check():
    output = reaction_time_classification(config)
    # Append more information to model
    if config['include_task_type']:
        output['model_name'] = 'model_task_type_included'
    else:
        output['model_name'] = 'model_task_type_not_included'
    save_path = str(Path(__file__).parents[1] / config['save_path'])
    save_model_log(output, save_path)

with skip_run('skip', 'plot_classification_accuracy') as check, check():
    plot_classification_accuracy(config)

with skip_run('skip', 'task_type_classification') as check, check():
    task_type_classification(config)

with skip_run('skip', 'plot_reaction_time') as check, check():
    plot_reaction_time(config['subjects'][1], config)

with skip_run('skip', 'plot_detection_false_alarm') as check, check():
    plot_detection_false_alarm_rate(config)
