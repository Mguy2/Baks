import pandas as pd

def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prints the number and percentage of missing values per column.
    Returns a DataFrame with the results.
    """
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df)) * 100

    report = pd.DataFrame({
        'Missing Values': missing_count,
        'Percentage Missing': missing_pct.round(2)
    })

    # Filter only columns with missing values
    report = report[report['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False)

    print(report)
    return report