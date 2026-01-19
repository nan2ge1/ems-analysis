import pandas as pd
import numpy as np

def calculate_theoretical_pv(
    radiation: pd.Series, 
    temperature: pd.Series, 
    temp_coeff: float = -0.004
) -> pd.Series:
    """
    Calculate a physically-informed feature for PV generation based on radiation and temperature.
    
    The formula used is:
        Output ~ Radiation * (1 + (Temperature - 25) * Temperature_Coefficient)
        
    Args:
        radiation: Series containing global radiation values (W/m²).
        temperature: Series containing ambient temperature values (°C).
        temp_coeff: Temperature coefficient for efficiency loss (default: -0.004 per °C).
        
    Returns:
        pd.Series: Calculated theoretical PV potential (kWh, scaled by capacity later).
    """
    # Calculate efficiency factor based on temperature deviation from STC (25°C)
    # If temp is 35°C, factor = 1 + (10 * -0.004) = 0.96 (4% loss)
    efficiency_factor = 1 + (temperature - 25) * temp_coeff
    
    # Calculate theoretical output
    theoretical_output = radiation * efficiency_factor
    
    # Ensure no negative values (physically impossible for generation)
    # Although radiation >= 0 usually ensures this, extremely high temps could flip the sign if unchecked (unlikely in reality)
    # or if input radiation is negative (bad data).
    theoretical_output = theoretical_output.clip(lower=0)
    
    return theoretical_output
