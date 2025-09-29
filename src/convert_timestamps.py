import pandas as pd
import numpy as np

def timestamps_to_unix(timestamps):
    """
    Convert timestamp strings to seconds since 1970-01-01 (Unix epoch).
    """
    ts_series = pd.Series(timestamps)
    parsed = pd.to_datetime(ts_series, errors='coerce', dayfirst=True)
    unix_seconds = parsed.astype('int64') // 10**9  # nanoseconds → seconds
    unix_seconds = unix_seconds.replace({pd.NaT: np.nan})
    return unix_seconds

def convert_datetime_columns_to_unix(df: pd.DataFrame, drop_original=False) -> pd.DataFrame:
    """
    Convert all datetime-like columns (with or without timezone) in a DataFrame to Unix seconds.

    Parameters:
    - df: input DataFrame
    - drop_original: if True, drop the original datetime columns after conversion

    Returns:
    - df: DataFrame with new columns "<col>_unix" containing Unix timestamps
    """
    df = df.copy()
    
    # Select all datetime-like columns, including timezone-aware
    datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetimetz']).columns

    for col in datetime_cols:
        # Convert to UTC if timezone-aware, then to Unix seconds
        if pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col + "_unix"] = df[col].dt.tz_convert('UTC').astype('int64') // 10**9
        else:
            df[col + "_unix"] = df[col].astype('int64') // 10**9

        print(f"Column '{col}' converted to Unix time: '{col}_unix'")
        if drop_original:
            df.drop(columns=[col], inplace=True)

    return df