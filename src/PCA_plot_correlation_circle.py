import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca_correlation_circle(df, n_components=2, figsize=(8,8)):
    """
    Creates a PCA correlation circle plot for the given DataFrame.
    
    Parameters:
    - df: pandas DataFrame (numeric features only)
    - n_components: number of PCA components (default=2)
    - figsize: tuple, size of the figure
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    # PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)
    
    # Correlation circle
    plt.figure(figsize=figsize)
    for i, (x, y) in enumerate(zip(pca.components_[0], pca.components_[1])):
        plt.arrow(0, 0, x, y, color='r', alpha=0.5)
        plt.text(x*1.15, y*1.15, df.columns[i], color='g', ha='center', va='center', fontsize=8)
    
    # Add unit circle
    circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--')
    plt.gca().add_artist(circle)
    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Correlation Circle")
    plt.grid()
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.show()

# Example usage:
# pca_correlation_circle(data.select_dtypes(include=['float32','float64','int32','int64']))
