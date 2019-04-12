import yaml
from utils import *
from data.create_dataset import create_dataframe
from data.create_dataset import create_r_dataframe
from models.t_sne_analysis import t_sne
from visualization.visualize import plot_detection_false_alarm
from visualization.visualize import plot_reaction_time
from models.rt_classification import reaction_time_classification
from models.density_estimation import estimate_density


config = yaml.load(open('config.yml'))


with skip_run_code('skip', 'create_dataset') as check, check():
    data, dataframe, secondary_dataframe = create_dataframe(config['subjects'], config)
    save_path = Path(__file__).parents[1] / config['processed_dataframe']
    save_dataframe(str(save_path), dataframe, save=True)

    save_path = Path(__file__).parents[1] / config['secondary_dataframe']
    save_dataframe(str(save_path), secondary_dataframe, save=True)

    save_path = Path(__file__).parents[1] / config['processed_dataset']
    save_dataset(str(save_path), data, save=True)


with skip_run_code('run', 'create_r_dataframe') as check, check():
    create_r_dataframe(config)


with skip_run_code('skip', 't_sne_analysis') as check, check():
    t_sne(config)


with skip_run_code('skip', 'density_analysis') as check, check():
    estimate_density(config)


with skip_run_code('skip', 'reaction_time_classification') as check, check():
    reaction_time_classification(config)


with skip_run_code('skip', 'plot_reaction_time') as check, check():
    plot_reaction_time(config['subjects'][0], config)


with skip_run_code('skip', 'plot_detection_false_alarm') as check, check():
    plot_detection_false_alarm_rate(config)
