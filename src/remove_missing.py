import pandas as pd

def remove_missing_features(df: pd.DataFrame, strictness: float = 0.5) -> pd.DataFrame:
    """
    Removes columns where proportion of missing values exceeds strictness.
    strictness is a threshold between 0 and 1 (default 0.5).
    
    Returns a new DataFrame with selected features removed.
    """
    missing_fraction = df.isna().mean()
    keep_cols = missing_fraction[missing_fraction <= strictness].index
    return df[keep_cols]