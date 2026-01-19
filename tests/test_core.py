import pandas as pd
import pytest
import numpy as np
from ems.data_loader import filter_household, convert_data
from ems.analysis import calculate_differences, sanity_check

@pytest.fixture
def sample_data():
    date_range = pd.date_range(start='2020-01-01', periods=5, freq='h')
    df = pd.DataFrame({
        'utc_timestamp': date_range,
        'DE_KN_residential3_grid_import': [10, 12, 15, 13, 20],
        'DE_KN_residential3_pv': [0, 0, 5, 10, 2],
        'other_household_col': [1, 2, 3, 4, 5]
    })
    # Convert string dates to datetime if strictly mimicking raw load, but pd.date_range makes datetimes.
    # The convert_data function expects 'utc_timestamp' column if index is not set yet.
    # If we pass date_range directly it is datetime64ns.
    return df

def test_convert_data(sample_data):
    # Ensure timestamp conversion works (if it was string)
    sample_data['utc_timestamp'] = sample_data['utc_timestamp'].astype(str)
    cleaned = convert_data(sample_data.copy())
    assert isinstance(cleaned.index, pd.DatetimeIndex)
    assert 'utc_timestamp' not in cleaned.columns # Should be index now

def test_filter_household(sample_data):
    # Pre-clean or setup index if filter_household expects it? 
    # filter_household just renames columns, doesn't care about index essentially, 
    # but logically follows loading.
    
    filtered = filter_household(sample_data)
    assert 'consumption_kWh' in filtered.columns
    assert 'pv_generation_kWh' in filtered.columns
    assert 'other_household_col' not in filtered.columns
    assert len(filtered) == 5

def test_calculate_differences():
    df = pd.DataFrame({'a': [1, 2, 4, 7]})
    diff = calculate_differences(df)
    # First row dropped because diff is NaN
    assert len(diff) == 3
    expected = [1.0, 2.0, 3.0]
    np.testing.assert_array_equal(diff['a'].values, expected)

def test_sanity_check():
    df = pd.DataFrame({'a': [10, -5, 25, 15]})
    # Threshold 20
    # -5 is < 0 (invalid)
    # 25 is > 20 (invalid)
    # 10, 15 are valid
    
    # Note: check returns DataFrame with NaNs where False?
    # Our implementation: return df[(df >= 0) & (df <= threshold)]
    # This keeps structure but replaces invalid with NaN.
    
    checked = sanity_check(df, threshold=20.0)
    
    # 10 is kept
    assert checked.loc[0, 'a'] == 10.0
    # -5 becomes NaN
    assert pd.isna(checked.loc[1, 'a'])
    # 25 becomes NaN
    assert pd.isna(checked.loc[2, 'a'])
    # 15 is kept
    assert checked.loc[3, 'a'] == 15.0
