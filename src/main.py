import yaml
from utils import *
from data.create_epoch_dataset import create_dataset
from visualization.visualize import plot_detection_false_alarm_rate


config = yaml.load(open('config.yml'))


with skip_run_code('skip', 'create_dataset') as check, check():
    data, dataframe, secondary_dataframe = create_dataset(config['subjects'], config)
    save_path = Path(__file__).parents[1] / config['processed_dataframe']
    save_dataframe(str(save_path), dataframe, save=True)

    save_path = Path(__file__).parents[1] / config['secondary_dataframe']
    save_dataframe(str(save_path), secondary_dataframe, save=True)

    save_path = Path(__file__).parents[1] / config['processed_dataset']
    save_dataset(str(save_path), data, save=True)


with skip_run_code('run', 'plot_detection_false_alarm_rate') as check, check():
    plot_detection_false_alarm_rate(config)
