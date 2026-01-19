# EMS Analysis

A Python package for analyzing household energy consumption, PV generation, and battery storage simulation.

## Description

This project analyzes energy management system (EMS) data to understand household energy usage patterns, optimize PV utilization, and simulate battery storage systems. It provides a complete pipeline from raw data processing to interactive visualization.

### Key Features
- **Data Loading & Cleaning**: Robust handling of time-series data with timestamp conversion and error filtering.
- **Physics-Based Analysis**: Theoretical PV generation calculation based on radiation and temperature.
- **Machine Learning**: XGBoost integration for forecasting PV generation.
- **Battery Simulation**: Simulate battery storage behavior (charging/discharging) to estimate autarky and self-consumption improvements.
- **Interactive Dashboard**: Streamlit-based dashboard for exploring data and simulation results.
- **Weather Integration**: Merging of historical weather data with energy consumption profiles.

## Installation

```bash
pip install .
```

To run the Jupyter notebook `demo.ipynb`, ensure you have installed the package as above so that `ems` can be imported.


## Usage

### Running the Analysis
The core logic is demonstrated in `src/ems/app.py`. You can run the full analysis pipeline directly:

```bash
python src/ems/app.py
```

This will:
1. Load and process the household data.
2. Fetch and merge weather data.
3. Perform specific feature engineering (time-lag features, physics-based features).
4. Train an XGBoost model.
5. Simulate a battery storage system.
6. Print key performance indicators (KPIs) like Autarky Rate.

Alternatively, you can run the full analysis pipeline interactively using the Jupyter notebook: `demo.ipynb`.

### Running the Dashboard
To visualize the results interactively:

```bash
streamlit run src/ems/app.py
```

### Example Code
```python
from ems.data_loader import load_and_process_data

# Load and process data
# Note: The path depends on where you run the script from.
# Inside src/ems/app.py it uses "data/..." relative to project root.
df = load_and_process_data('data/household_data_60min_singleindex.csv')
print(df.head())
```

## Data Source
The default dataset is included in `data/` and is sourced from [Open Power System Data](https://data.open-power-system-data.org/household_data/2020-04-15/household_data_60min_singleindex.csv).


## License
MIT
