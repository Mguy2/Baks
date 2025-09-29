import pandas as pd
import numpy as np

def describe_object_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary of all object/string features in a DataFrame.
    Columns include:
    - Number of unique values
    - Number of missing values
    - Average length of entries (ignores missing)
    """
    object_cols = df.select_dtypes(include=['object', 'string']).columns
    summary = []

    for col in object_cols:
        col_data = df[col].dropna().astype(str)
        n_unique = col_data.nunique()
        n_missing = df[col].isna().sum()
        avg_len = col_data.str.len().mean() if not col_data.empty else 0

        summary.append({
            'Feature': col,
            'Unique Values': n_unique,
            'Missing Values': n_missing,
            'Average Length': round(avg_len, 2)
        })

    return pd.DataFrame(summary).sort_values(by='Missing Values', ascending=False)
