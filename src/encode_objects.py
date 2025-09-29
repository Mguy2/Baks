import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import numpy as np

def greedy_optimal_clusters(X, n_unique, random_state=42, n_init=5):
    """
    Greedy selection of optimal clusters based on number of unique values.
    Narrows search range for efficiency.
    """
    k_min = max(2, n_unique // 5)
    k_max = max(k_min + 1, min(n_unique, n_unique // 2 + 2))

    best_score = -1
    best_k = k_min
    
    if k_max > 200: k_max = 200
    if k_min > 200: k_min = 199

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = kmeans.fit_predict(X)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k

    return best_k

def encode_object_features(df: pd.DataFrame, drop_original: bool = True, 
                           random_state=42, n_init=5, binary_threshold=10,
                           fast_threshold=50) -> pd.DataFrame:
    """
    Encode object features for clustering.
    
    - Low-cardinality (≤ binary_threshold unique values): one-hot encoding
    - High-cardinality (> binary_threshold unique values): TF-IDF + KMeans
      * For very high cardinality or large datasets, uses MiniBatchKMeans for speed.
    
    Parameters:
    - df: input DataFrame
    - drop_original: whether to drop original object columns
    - random_state: random seed for KMeans
    - n_init: KMeans n_init parameter
    - binary_threshold: max unique values for one-hot encoding
    - fast_threshold: above this number of unique values, switch to MiniBatchKMeans
    
    Returns:
    - df: transformed DataFrame
    """
    df = df.copy()
    object_cols = df.select_dtypes(include=['object', 'string']).columns

    for col in object_cols:
        unique_vals = df[col].unique()
        n_unique = df[col].nunique(dropna=True)
        print(f"Encoding feature: \"{col}\", unique values: {unique_vals}")

        if n_unique <= binary_threshold:
            # One-hot encoding
            ohe = pd.get_dummies(df[col], prefix=col, dummy_na=True)
            df = pd.concat([df, ohe], axis=1)
            print(f"Binary encoding applied: {ohe.shape[1]} new features added.")
        else:
            print(f"Clustering feature \"{col}\" ({n_unique} unique values).")

            text_data = df[col].fillna("").astype(str)
            tfidf = TfidfVectorizer(max_features=500)
            X_tfidf = tfidf.fit_transform(text_data)

            # Use MiniBatchKMeans for high cardinality or large datasets
            if n_unique > fast_threshold or len(df) > 2000:
                best_k = min(max(2, n_unique // 10), 50)  # heuristic
                print(f"Using MiniBatchKMeans with {best_k} clusters for speed.")
                kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=random_state, batch_size=1000)
                cluster_labels = kmeans.fit_predict(X_tfidf)
            else:
                best_k = greedy_optimal_clusters(X_tfidf, n_unique, random_state=random_state, n_init=n_init)
                print(f"Found best k via greedy search: {best_k} clusters.")
                kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=n_init)
                cluster_labels = kmeans.fit_predict(X_tfidf)

            df[f"{col}_tfidf_cluster"] = cluster_labels
            print(f"Feature '{col}_tfidf_cluster' created.")

        if drop_original:
            df.drop(columns=[col], inplace=True)

    return df
