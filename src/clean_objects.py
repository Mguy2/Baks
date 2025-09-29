import pandas as pd
import numpy as np
import string
import unicodedata

def clean_object_features(df: pd.DataFrame, rare_threshold=5, drop_original=False) -> pd.DataFrame:
    """
    Clean all object/string features in a DataFrame for clustering.
    
    Cleaning steps:
    - Lowercase and strip spaces
    - Remove punctuation
    - Normalize accents
    - Fill missing values with 'unknown'
    - Replace rare categories with 'other'
    
    Parameters:
    - df: input DataFrame
    - rare_threshold: categories with fewer occurrences than this are replaced with 'other'
    - drop_original: if True, drop original columns after cleaning
    
    Returns:
    - df: cleaned DataFrame
    """
    df = df.copy()
    object_cols = df.select_dtypes(include=['object', 'string']).columns
    
    for col in object_cols:
        # Convert to string, lowercase, strip spaces
        df[col + "_clean"] = df[col].astype(str).str.lower().str.strip()
        
        # Normalize accents
        df[col + "_clean"] = df[col + "_clean"].apply(
            lambda x: unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode('utf-8')
        )
        
        # Remove punctuation
        df[col + "_clean"] = df[col + "_clean"].str.translate(str.maketrans('', '', string.punctuation))
        
        # Fill missing values (after converting to string, NaNs become 'nan')
        df[col + "_clean"] = df[col + "_clean"].replace('nan', 'unknown')
        
        # Group rare categories
        counts = df[col + "_clean"].value_counts()
        rare = counts[counts < rare_threshold].index
        df[col + "_clean"] = df[col + "_clean"].replace(rare, 'other')
        
        print(f"Feature '{col}' cleaned: {df[col + '_clean'].nunique()} unique values remain")
        
        if drop_original:
            df.drop(columns=[col], inplace=True)
            
    return df
