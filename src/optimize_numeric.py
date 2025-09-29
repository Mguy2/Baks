
def optimize_numeric_floats(df):
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.select_dtypes(include=[np.number]).columns:
        col_data = df[col]
        if np.issubdtype(col_data.dtype, np.integer):
            df[col] = pd.to_numeric(col_data, downcast='integer')
        elif np.issubdtype(col_data.dtype, np.floating):
            if col_data.min() >= np.finfo(np.float16).min and col_data.max() <= np.finfo(np.float16).max:
                df[col] = col_data.astype(np.float16)
            else:
                df[col] = col_data.astype(np.float32)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB")
    print(df.dtypes)
    return df