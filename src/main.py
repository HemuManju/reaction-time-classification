import yaml
from pathlib import Path
import matplotlib.pyplot as plt

from data.create_dataset import create_dataframe
from data.create_dataset import create_r_dataframe
from data.export_data import export_dataset

from features.features_selection import selected_features

from models.t_sne_analysis import t_sne
from models.rt_classification import reaction_time_classification
from models.task_classification import task_type_classification
from models.density_estimation import estimate_density
from models.statistical_test import friedman_test

from visualization.visualize import (plot_classification_accuracy,
                                     plot_reaction_time,
                                     plot_box_reaction_time,
                                     plot_detection_false_alarm)

from utils import (skip_run, save_dataset)

# The configuration file
config_path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('skip', 'create_dataset') as check, check():
    data, dataframe, secondary_dataframe = create_dataframe(
        config['subjects'], config)

    # Save
    save_path = Path(__file__).parents[1] / config['processed_dataframe']
    save_dataset(str(save_path), dataframe, save=True, use_pandas=True)

    save_path = Path(__file__).parents[1] / config['secondary_dataframe']
    save_dataset(str(save_path),
                 secondary_dataframe,
                 save=True,
                 use_pandas=True)

    # save_path = Path(__file__).parents[1] / config['processed_dataset']
    # save_dataset(str(save_path), data, save=True)

with skip_run('skip', 'export_data') as check, check():
    config['save_path'] = 'models/experiment_1/'
    export_dataset(config, name=None)

    config['save_path'] = 'models/experiment_2/'
    export_dataset(config, name=None)

    config['save_path'] = 'models/experiment_0/'
    export_dataset(config, name=None)

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
    config['classification_features'] = ['glance_ratio', 'mental_workload']
    config['include_task_type'] = True
    output = reaction_time_classification(config)
    # Append more information to model
    if config['include_task_type']:
        output['model_name'] = 'model_task_type_included'
    else:
        output['model_name'] = 'model_task_type_not_included'
    save_path = str(Path(__file__).parents[1] / config['save_path'])
    # save_model_log(output, save_path)

with skip_run('skip', 'plot_classification_accuracy') as check, check():
    plt.style.use('clean_box')
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 3.5), sharey=True)
    config['save_path'] = 'models/experiment_1/'
    plot_classification_accuracy(config, ax[0], name='Fig10')
    ax[0].set_ylabel('Classification accuracy', fontsize=14)
    ax[0].legend(
        ['Chance', 'All subjects', 'Low performers', 'High performers'])

    config['save_path'] = 'models/experiment_2/'
    plot_classification_accuracy(config, ax[1], name='Fig11')

    config['save_path'] = 'models/experiment_0/'
    plot_classification_accuracy(config, ax[2], name='Fig12')
    plt.tight_layout(pad=0.75)
    plt.show()

with skip_run('skip', 'statitical_analysis') as check, check():
    friedman_test(config)

with skip_run('skip', 'task_type_classification') as check, check():
    task_type_classification(config)

with skip_run('skip', 'plot_reaction_time') as check, check():
    plot_reaction_time(config['subjects'][4], config)

with skip_run('skip', 'plot_detection_false_alarm') as check, check():
    plot_detection_false_alarm(config)
