import pytest
import pandas as pd
from unittest.mock import patch, Mock
from datetime import date
from ems.weather import fetch_weather_data, process_weather_data, merge_weather_data

@pytest.fixture
def mock_weather_response():
    return {
        "hourly": {
            "time": ["2023-01-01T00:00", "2023-01-01T01:00"],
            "temperature_2m": [10.5, 11.2],
            "shortwave_radiation": [0.0, 150.0]
        }
    }

@patch('ems.weather.requests.get')
def test_fetch_weather_data(mock_get, mock_weather_response):
    mock_get.return_value.json.return_value = mock_weather_response
    mock_get.return_value.status_code = 200
    
    start = date(2023, 1, 1)
    end = date(2023, 1, 2)
    data = fetch_weather_data(start, end)
    
    assert data == mock_weather_response
    mock_get.assert_called_once()
    assert "latitude=47.66" in mock_get.call_args[0][0]

def test_process_weather_data(mock_weather_response):
    df = process_weather_data(mock_weather_response)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None # Should be UTC
    assert 'temperature' in df.columns
    assert 'radiation' in df.columns
    assert df.iloc[0]['temperature'] == 10.5

def test_merge_weather_data():
    # Setup household df
    idx = pd.date_range("2023-01-01 00:00", periods=2, freq="h", tz="UTC")
    df_house = pd.DataFrame({
        "consumption_kWh": [1.0, 1.5]
    }, index=idx)
    
    # Setup weather df
    df_weather = pd.DataFrame({
        "temperature": [10.0, 11.0]
    }, index=idx)
    
    merged = merge_weather_data(df_house, df_weather)
    
    assert len(merged) == 2
    assert "consumption_kWh" in merged.columns
    assert "temperature" in merged.columns
