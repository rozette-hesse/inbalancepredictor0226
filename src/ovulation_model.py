# src/ovulation_model.py

import numpy as np


def compute_luteal_distribution(df):
    df = df.copy()
    df["luteal_length"] = df["cycle_length"] - df["ovulation_day"]

    median = df["luteal_length"].median()
    mad = np.median(
        np.abs(df["luteal_length"] - median)
    )

    return {
        "median": median,
        "mad": mad
    }


def predict_ovulation_day(predicted_cycle_length, luteal_stats):
    return predicted_cycle_length - round(luteal_stats["median"])
