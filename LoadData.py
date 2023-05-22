"""
This class contrals how the data is loaded and split into inputs and labels.
"""

import numpy as np


def load_data(csv: str):
    """Load data from csv file. First two columns are not needed (s. description.pdf).

    :csv: path to csv file
    :returns: numpy array containing the data

    """
    data = np.loadtxt(csv, delimiter=",",
                      skiprows=1, usecols=(range(2, 17)))
    return data


def split_data(data):
    """Split data into input and label by separating the last column of the input array.

    :data: numpy array containing the data
    :returns: two numpy arrays, first resembles inputs, second labels

    """
    splitted_data = np.hsplit(data, [-1])
    input_hour = splitted_data[0]
    # Remove one dimension, ie. (N, 1) -> (N,)
    label_hour = splitted_data[1][:, 0]
    return input_hour, label_hour
