import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from ExGUtils import pyexg


def read_dataframe(path):
    """Save the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataframe : dict
        dictionary of pandas dataframe to save


    """

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data
