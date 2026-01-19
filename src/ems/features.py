import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclic time features (hour sine/cosine) to the DataFrame.
    Assumes the DataFrame has a DatetimeIndex.
    """
    df = df.copy()
    # Cyclic Features (Hour of Day)
    # Allows the model to understand that 23:00 and 00:00 are close.
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    return df

def add_lag_features(df: pd.DataFrame, col_name: str, lags: list[int]) -> pd.DataFrame:
    """
    Add lag features for a specific column.
    
    Args:
        df: Input DataFrame.
        col_name: Name of the column to create lags for.
        lags: List of integer lags (e.g. [24] for 24 hours ago).
    """
    df = df.copy()
    for lag in lags:
        df[f'{col_name}_lag_{lag}h'] = df[col_name].shift(lag)
    
    return df
