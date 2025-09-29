import pandas as pd

def add_unique_id(df, id_col='unique_id'):
    df[id_col] = range(1, len(df) + 1)
    return df