# src/data_cleaning.py

import pandas as pd
from .config import MIN_CYCLE_LENGTH, MAX_CYCLE_LENGTH


def standardize_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def clean_cycle_lengths(df, cycle_col="cycle_length"):
    df = df.copy()
    df = df[df[cycle_col].notna()]
    df = df[(df[cycle_col] >= MIN_CYCLE_LENGTH) &
            (df[cycle_col] <= MAX_CYCLE_LENGTH)]
    return df


def build_cycle_master(datasets):
    master = []

    for df, source in datasets:
        df = standardize_columns(df)

        if "cycle_length" not in df.columns:
            continue

        df = clean_cycle_lengths(df, "cycle_length")

        df["source"] = source

        keep_cols = ["cycle_length", "age", "bmi", "cycle_start_date"]
        existing = [c for c in keep_cols if c in df.columns]

        master.append(df[existing + ["source"]])

    return pd.concat(master, ignore_index=True)
