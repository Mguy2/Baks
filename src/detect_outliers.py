import pandas as pd
import numpy as np

def detect_outliers(df: pd.DataFrame, strictness: float = 1.5) -> pd.DataFrame:
    """
    Detects outliers in numeric columns using IQR method.
    strictness controls how far from IQR the cut-off is (default 1.5).

    Returns the actual rows (with all columns) where at least one outlier is found.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_rows = set()

    for col in numeric_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - strictness * IQR, Q3 + strictness * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_rows.update(outliers.index)
    return df.loc[list(outlier_rows)]
