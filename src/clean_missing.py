import pandas as pd
import numpy as np

def flag_and_fill_missing(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df = df.replace(["", "NA", "N/A", "null", "None", "NaN"], np.nan)
    df["has_missing"] = df.isna().any(axis=1)
    for col in df.columns.drop("has_missing"):
        df[col] = df[col].fillna(0 if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]) else "missing")
    return df

def fill_and_count_missing(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df = df.replace(["", "NA", "N/A", "null", "None", "NaN"], np.nan)
    df["num_missing"] = df.isna().sum(axis=1)
    for col in df.columns.drop("num_missing"):
        df[col] = df[col].fillna(0 if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]) else "missing")
    return df