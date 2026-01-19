import pandas as pd
from typing import Tuple

def get_train_val_test_split(
    df: pd.DataFrame, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into training, validation, and test sets by time.
    
    Args:
        df: Input DataFrame.
        val_ratio: Ratio of validation set size.
        test_ratio: Ratio of test set size.
        
    Returns:
        Tuple of (train, val, test) DataFrames.
    """
    n = len(df)
    num_val = int(round(n * val_ratio))
    num_test = int(round(n * test_ratio))
    num_train = n - num_val - num_test
    
    train = df.iloc[:num_train].copy()
    val = df.iloc[num_train : num_train + num_val].copy()
    test = df.iloc[num_train + num_val:].copy()
    
    return train, val, test
