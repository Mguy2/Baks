import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def pca_transform(df, variance_threshold=0.95, plot=True):
    """
    Applies PCA to reduce multicollinearity for anomaly detection/clustering.

    Parameters:
    - df: pandas DataFrame (numeric features only)
    - variance_threshold: cumulative explained variance ratio to retain
    - plot: whether to plot explained variance

    Returns:
    - pca_df: DataFrame with PCA components
    - pca: fitted PCA object
    - scaler: fitted scaler
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Fit PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Select number of components to reach variance threshold
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cum_var >= variance_threshold) + 1

    print(f"Selected {n_components} components to explain {variance_threshold*100:.1f}% of variance.")

    # Reduce to selected components
    X_reduced = X_pca[:, :n_components]
    pca_df = pd.DataFrame(X_reduced, columns=[f"PC{i+1}" for i in range(n_components)], index=df.index)

    if plot:
        plt.figure(figsize=(8,5))
        plt.plot(range(1, len(cum_var)+1), cum_var, marker='o')
        plt.axhline(y=variance_threshold, color='r', linestyle='--')
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Explained Variance")
        plt.grid()
        plt.show()

    return pca_df, pca, scaler
