import pandas as pd

def calculate_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the difference between consecutive rows.
    Useful for converting cumulative meter readings to interval consumption.

    Args:
        df: Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with differences calculated.
    """
    df_diff = df.diff()
    return df_diff.dropna()

def sanity_check(df: pd.DataFrame, threshold: float = 20.0) -> pd.DataFrame:
    """
    Perform a sanity check on the data (physics check).
    Removes negative values and values exceeding a threshold.

    Args:
        df: Input dataframe.
        threshold: Maximum plausible value (e.g. 20 kWh).

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    return df[(df >= 0) & (df <= threshold)]

def calculate_metrics(results_df: pd.DataFrame) -> dict:
    """
    Calculate KPIs including autarky rate with and without battery.
    
    Args:
        results_df: DataFrame containing simulation results 
                    (consumption_kWh, grid_import_kWh, prediction).
    
    Returns:
        dict: Dictionary containing calculated metrics.
    """
    total_consumption = results_df['consumption_kWh'].sum()
    total_import = results_df['grid_import_kWh'].sum()
    
    # Autarky = (Consumption - Grid Import) / Consumption
    autarky_rate = (total_consumption - total_import) / total_consumption if total_consumption > 0 else 0.0
    
    # COMPARISON: How would it have been WITHOUT a battery?
    # Without battery, every deficit (Consumption > PV) must come from the grid.
    without_battery_import = (results_df['consumption_kWh'] - results_df['prediction']).clip(lower=0).sum()
    autarky_rate_no_bat = (total_consumption - without_battery_import) / total_consumption if total_consumption > 0 else 0.0
    
    return {
        'total_consumption': total_consumption,
        'total_import': total_import,
        'autarky_rate': autarky_rate,
        'without_battery_import': without_battery_import,
        'autarky_rate_no_bat': autarky_rate_no_bat,
        'improvement': autarky_rate - autarky_rate_no_bat
    }
