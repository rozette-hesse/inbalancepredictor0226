# src/cycle_model.py

import numpy as np
from .config import LAMBDA_BLEND


def compute_population_stats(cycle_df):
    median = cycle_df["cycle_length"].median()
    mad = np.median(
        np.abs(cycle_df["cycle_length"] - median)
    )

    return {
        "median": median,
        "mad": mad
    }


def blend(pop_value, user_value, n_user, lam=LAMBDA_BLEND):
    w = n_user / (n_user + lam)
    return w * user_value + (1 - w) * pop_value


def predict_cycle_length(user_cycles, pop_stats):
    if len(user_cycles) == 0:
        return pop_stats["median"]

    user_median = np.median(user_cycles)
    pred = blend(pop_stats["median"], user_median, len(user_cycles))
    return round(pred)
