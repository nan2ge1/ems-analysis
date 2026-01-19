import pandas as pd
import numpy as np
import pytest
from ems.physics import calculate_theoretical_pv

def test_calculate_theoretical_pv_basic():
    """Test basic calculation with standard values."""
    radiation = pd.Series([1000.0, 0.0, 500.0])
    temperature = pd.Series([25.0, 25.0, 35.0]) # 25°C is STC (no loss)
    
    expected = pd.Series([1000.0, 0.0, 500.0 * (1 + (10 * -0.004))]) 
    # 500 * (1 - 0.04) = 500 * 0.96 = 480
    
    result = calculate_theoretical_pv(radiation, temperature)
    
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_calculate_theoretical_pv_temperature_effect():
    """Test that higher temperatures reduce output."""
    radiation = pd.Series([1000.0, 1000.0])
    temperature = pd.Series([25.0, 35.0])
    
    result = calculate_theoretical_pv(radiation, temperature)
    
    assert result[1] < result[0] # Hotter should be less efficient

def test_calculate_theoretical_pv_extreme_values():
    """Test clipping at 0 and handling of negative inputs."""
    radiation = pd.Series([-100.0, 100.0])
    temperature = pd.Series([25.0, 1000.0]) # Very high temp to force negative factor
    
    # 1000°C -> (975 * -0.004) = -3.9 -> factor = -2.9. Output would be negative without clip.
    
    result = calculate_theoretical_pv(radiation, temperature)
    
    assert (result >= 0).all()
    assert result[0] == 0 # Negative radiation clipped
    assert result[1] == 0 # Negative factor clipped
