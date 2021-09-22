import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

from pathlib import Path
from .utils import read_model_log


def export_dataset(config, name):
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

    labels = ['not_included', 'included']

    # Form the dataframe
    for i, item in enumerate(fname):
        data = read_model_log(item)
        for j, performance_level in enumerate(config['performance_level']):
            df = pd.DataFrame()

            # Get the true label
            arg_ind = np.argsort(-data[performance_level]['accuracy'])[0:10]
            # Select the rows
            predictions = data[performance_level]['prediction'][
                arg_ind].flatten()
            true = data[performance_level]['true'][arg_ind].flatten()
            df['predicted'] = predictions
            df['true'] = true

            name = performance_level + '_task_' + labels[i]
            confusion_mat = confusion_matrix(true,
                                             predictions,
                                             normalize='pred')

            confuse = pd.DataFrame(confusion_mat)
            confuse.to_excel('data/external/confusion_matrix/' + name +
                             '.xlsx')

            # Save the data frame
            df.to_excel('data/external/raw/' + name + '.xlsx', index=False)
