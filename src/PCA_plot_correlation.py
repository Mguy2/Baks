import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_high_corr(df, threshold=0.8, figsize=(12,10), annot_size=8, cmap="coolwarm"):
    """
    Plots a heatmap of feature correlations above a given threshold.

    Parameters:
    - df: pandas DataFrame
    - threshold: float, minimum absolute correlation to show
    - figsize: tuple, size of the figure
    - annot_size: int, size of annotation text
    - cmap: str, colormap for heatmap
    """
    # Compute correlation matrix for numeric columns only
    corr = df.corr().abs()
    
    # Select upper triangle to avoid duplicate pairs and self-correlation
    upper = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
    
    # Find columns with any correlation above threshold
    high_corr_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    
    if len(high_corr_cols) < 2:
        print("No feature pairs above the threshold.")
        return
    
    # Filter correlation matrix for these columns
    high_corr_matrix = corr.loc[high_corr_cols, high_corr_cols]
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        high_corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        square=True,
        annot_kws={"size": annot_size},
        cbar=True
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title(f"Highly Correlated Features (Threshold > {threshold})", fontsize=14)
    plt.show()
    
    # Print the highly correlated feature pairs
    high_corr_pairs = [(col1, col2, upper.loc[col1, col2])
                       for col1 in upper.columns
                       for col2 in upper.columns
                       if upper.loc[col1, col2] > threshold]
    
    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)
    
    print("Highly correlated feature pairs:")
    for f1, f2, val in high_corr_pairs:
        print(f"{f1} and {f2}: {val:.2f}")