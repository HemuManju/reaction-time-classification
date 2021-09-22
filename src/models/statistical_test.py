from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .utils import read_model_log


def friedman_test(config):
    # import data from
    read_path = Path(__file__).parents[2] / config['save_path']
    fname = [str(f) for f in read_path.iterdir() if f.suffix == '.pkl']
    fname.sort(reverse=True)

    # Form the dataframe
    for i, item in enumerate(fname):
        data = read_model_log(item)
        df_temp = pd.DataFrame()
        for j, performance_level in enumerate(config['performance_level']):
            temp_df = pd.DataFrame(columns=['accuracy', 'subject_information'])
            temp = np.sort(-data[performance_level]['accuracy'])[0:10]
            temp_df['accuracy'] = -temp * 100
            # temp_df['task_information'] = labels[i]
            temp_df['subject_information'] = performance_level
            df_temp = df_temp.append(temp_df, ignore_index=True)

        # Statistical test
        group_low_performer = df_temp.loc[df_temp['subject_information'] ==
                                          'low_performer']['accuracy']
        group_high_performer = df_temp.loc[df_temp['subject_information'] ==
                                           'high_performer']['accuracy']
        group_all_subjects = df_temp.loc[df_temp['subject_information'] ==
                                         'all_subjects']['accuracy']
        print(np.mean(group_all_subjects), np.mean(group_high_performer),
              np.mean(group_low_performer))
        results = stats.friedmanchisquare(group_low_performer,
                                          group_high_performer,
                                          group_all_subjects)

        print(results)

    return None
