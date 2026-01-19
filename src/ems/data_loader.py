import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load energy data from a CSV file.

    Args:
        filepath: Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(filepath)

def convert_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the loaded data by setting the index as timestamps.

    Args:
        df: Raw dataframe.

    Returns:
        pd.DataFrame: Converted dataframe with DatetimeIndex.
    """
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df.set_index('utc_timestamp', inplace=True)
    return df

def filter_household(df: pd.DataFrame, household_id: str = 'DE_KN_residential3') -> pd.DataFrame:
    """
    Filter data for a specific household and rename columns.

    Args:
        df: The full dataframe.
        household_id: The household ID prefix to select. 
                      Defaults to 'DE_KN_residential3' (Constance, Germany).

    Returns:
        pd.DataFrame: Dataframe containing only consumption and pv generation for the household.
    """
    # Columns to keep mapping
    cols_to_keep = {
        f'{household_id}_grid_import': 'consumption_kWh',
        f'{household_id}_pv': 'pv_generation_kWh'
    }
    
    # Select and rename
    df_house = df[cols_to_keep.keys()].rename(columns=cols_to_keep)
    
    # Remove NaNs
    df_house = df_house.dropna()
    
    return df_house

def load_and_process_data(filepath: str, household_id: str = 'DE_KN_residential3') -> pd.DataFrame:
    """
    Load, convert, filter, and clean data for a specific household.
    Combines load_data, convert_data, filter_household, calculate_differences, and sanity_check.

    Args:
        filepath: Path to the CSV file.
        household_id: The household ID prefix to select.

    Returns:
        pd.DataFrame: Processed dataframe ready for analysis.
    """
    # Import here to avoid potential circular imports if structure changes
    from ems.analysis import calculate_differences, sanity_check

    df = load_data(filepath)
    df = convert_data(df)
    df_house = filter_household(df, household_id)
    df_house = calculate_differences(df_house)
    df_house = sanity_check(df_house)
    
    return df_house
