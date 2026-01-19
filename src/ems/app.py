import sys
import os

# Add src to path to allow imports without installation
# Assuming this file is at src/ems/app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from ems.data_loader import load_and_process_data
from ems.analysis import calculate_metrics
from ems.weather import fetch_weather_data, process_weather_data, merge_weather_data
from ems.physics import calculate_theoretical_pv
from ems.features import add_time_features, add_lag_features
from ems.model_selection import get_train_val_test_split
from ems.training import train_model, predict_and_clip
from ems.simulation import simulate_battery
from ems.dashboard import run_dashboard


# Step 1: Data Cleaning & Wrangling (Python/Pandas)

# 1. Load and process data
# Source: Open Power System Data
# Link: https://data.open-power-system-data.org/household_data/2020-04-15/household_data_60min_singleindex.csv
# This function loads, converts timestamps, filters for 'DE_KN_residential3',
# calculates differences (cumulative -> interval), and performs a physics-based sanity check.
url = "data/household_data_60min_singleindex.csv"
df_house = load_and_process_data(url)

print("Data processing complete.")
print(df_house.head())


# Step 2: Integrate weather data (The "Fusion")

# Coordinates for Constance
latitude = 47.66
longitude = 9.17

start_date = df_house.index[0].date()
end_date = df_house.index[-1].date()

# 1. Fetch data
weather_json = fetch_weather_data(start_date, end_date, latitude, longitude)

# 2. Process data
df_weather = process_weather_data(weather_json)

# 3. Merge
df_final = merge_weather_data(df_house, df_weather)

print("Weather data merged.")
print(df_final.head())


# Step 3: Feature Engineering for physics

# Assumptions for physics
# 1. We don't know exactly how big the system is. We estimate it via the maximum.
max_pv_output = df_final['pv_generation_kWh'].max() 

# 2. Temperature coefficient (typical for silicon modules)
temp_coeff = -0.004 # -0.4% pro Degree Celsius

# 3. Feature Engineering: "Theoretically possible power"
# Use vectorized function from physics module
df_final['phys_feature_pv'] = calculate_theoretical_pv(
    radiation=df_final['radiation'],
    temperature=df_final['temperature'],
    temp_coeff=temp_coeff
)

print("Physics feature calculated.")
print(df_final[['radiation', 'temperature', 'phys_feature_pv']].head())


# Step 4: Feature Engineering for time series

# 1. Create cyclic features (hour of day)
df_final = add_time_features(df_final)

# 2. Lag Features (Delay)
df_final = add_lag_features(df_final, col_name='pv_generation_kWh', lags=[24])

# Remove NaNs created by shifting
df_final = df_final.dropna()

print("Features engineered.")
print(df_final.head())


# Step 5: The Train-Validation-Test Split

# We split strictly by time.
# Training:Validation:Test = 70:15:15
train, val, test = get_train_val_test_split(df_final, val_ratio=0.15, test_ratio=0.15)

# Define features (Input)
features = ['radiation', 'temperature', 'phys_feature_pv', 
            'hour_sin', 'hour_cos', 'pv_generation_kWh_lag_24h']
target = 'pv_generation_kWh'

X_train, y_train = train[features], train[target]
X_val, y_val = val[features], val[target]
X_test, y_test = test[features], test[target]

print(f'Size: training:validation:test = {len(y_train)}:{len(y_val)}:{len(y_test)}')


# Step 6: Model Training (XGBoost) and Prediction

reg = train_model(X_train, y_train, X_val, y_val)

# Prediction on test data
test.loc[:, 'prediction'] = predict_and_clip(reg, X_test)


# Step 7: Simulate Battery

# Hardware parameters
BATTERY_CAPACITY_KWH = 10.0  # Storage size
MAX_POWER_KW = 5.0           # How fast can be charged/discharged? (e.g. Sunny Boy Storage)
EFFICIENCY = 0.95            # Charge/discharge loss (approx. 95%)
INITIAL_SOC = 0.0            # We start with empty battery

# Run simulation
results = simulate_battery(test, BATTERY_CAPACITY_KWH, MAX_POWER_KW, EFFICIENCY)

print("Battery simulation complete.")
print(results.head())


# Step 8: Validation & KPIs (Business Value)

metrics = calculate_metrics(results)

print(f"Autarky rate with Smart Battery: {metrics['autarky_rate']:.2%}")
print(f"Autarky rate WITHOUT Battery: {metrics['autarky_rate_no_bat']:.2%}")
print(f"Improvement through algorithm: {metrics['improvement']:.2%}")


# Step 9: Interactive Dashboard
if __name__ == "__main__":
    try:
        import streamlit as st
        # Only run dashboard if executed via 'streamlit run'
        # st.runtime.exists() returns True if running inside Streamlit
        if st.runtime.exists():
            run_dashboard(test)
        else:
            print("\nAnalysis complete.")
            print("To verify results or run the dashboard interactively, use:")
            print("  streamlit run src/ems/app.py")
            
    except ImportError:
        # If streamlit is not installed, just finish
        pass
