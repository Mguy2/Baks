import pandas as pd
import numpy as np

def high_corr_table(df, threshold=0.8):
    """
    Finds and prints highly correlated feature pairs above a given threshold.

    Parameters:
    - df: pandas DataFrame
    - threshold: float, minimum absolute correlation to include
    """
    # Compute absolute correlation matrix
    corr = df.corr().abs()
    
    # Select upper triangle to avoid duplicates
    upper = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
    
    # Find feature pairs above threshold
    high_corr_pairs = [(col1, col2, upper.loc[col1, col2]) 
                       for col1 in upper.columns 
                       for col2 in upper.columns 
                       if upper.loc[col1, col2] > threshold]
    
    # Sort by correlation descending
    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)
    
    # Create a DataFrame for nice display
    if high_corr_pairs:
        table = pd.DataFrame(high_corr_pairs, columns=["Feature 1", "Feature 2", "Correlation"])
        print(table.to_string(index=False))
    else:
        print(f"No feature pairs with correlation above {threshold}")
