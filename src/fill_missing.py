import pandas as pd
import numpy as np

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces missing values:
    - Numeric columns: fill with 0
    - String/object columns: fill with None

    Returns a new DataFrame with missing values handled.
    """
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(np.nan)
        elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].where(df[col].notna(), "None")

    return df
