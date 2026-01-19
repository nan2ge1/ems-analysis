import pandas as pd
import requests
from datetime import date

def fetch_weather_data(start_date: date, end_date: date, latitude: float = 47.66, longitude: float = 9.17) -> dict:
    """
    Fetch historical weather data from Open-Meteo API.

    Args:
        start_date: Start date for the data.
        end_date: End date for the data.
        latitude: Latitude of the location (default: Constance).
        longitude: Longitude of the location (default: Constance).

    Returns:
        dict: JSON response from the API.
    """
    api_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,shortwave_radiation"
    response = requests.get(api_url)
    response.raise_for_status()
    return response.json()

def process_weather_data(weather_data: dict) -> pd.DataFrame:
    """
    Process raw weather JSON data into a DataFrame.

    Args:
        weather_data: JSON data from Open-Meteo API.

    Returns:
        pd.DataFrame: Processed weather data with DatetimeIndex (UTC).
    """
    df_weather = pd.DataFrame({
        'timestamp': pd.to_datetime(weather_data['hourly']['time']),
        'temperature': weather_data['hourly']['temperature_2m'],
        'radiation': weather_data['hourly']['shortwave_radiation'] # Global radiation in W/m²
    })
    
    # Open-Meteo archive defaults to GMT/UTC.
    df_weather['timestamp'] = df_weather['timestamp'].dt.tz_localize('UTC')
    df_weather.set_index('timestamp', inplace=True)
    return df_weather

def merge_weather_data(df_house: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    Merge household data with weather data.

    Args:
        df_house: Household DataFrame.
        df_weather: Weather DataFrame.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    return df_house.join(df_weather, how='inner')
