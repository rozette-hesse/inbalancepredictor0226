# src/evaluation.py

import numpy as np


def mean_absolute_error(true_values, predicted_values):
    return np.mean(np.abs(np.array(true_values) - np.array(predicted_values)))
