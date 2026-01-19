import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from ems.simulation import simulate_battery

def run_dashboard(test: pd.DataFrame):
    """
    Run the Streamlit dashboard.
    
    Args:
        test: DataFrame containing test data.
    """
    # --- 1. SETUP ---
    # Note: set_page_config must be the first Streamlit command. 
    # Calling this function wraps it, so ensure no other st commands run before this function in the main script.
    try:
        st.set_page_config(page_title="Vault-Tec Energy Optimizer", layout="wide")
    except st.errors.StreamlitAPIException:
        # Ignore if already set (e.g. if running in a way where config is already established)
        pass

    # --- 2. SIDEBAR (Configuration options) ---
    st.sidebar.header("System Configuration")
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Vault-Tec_Logo.svg/1280px-Vault-Tec_Logo.svg.png", width=100) # Vault-Tec Industries
    battery_capacity = st.sidebar.slider(
        "🔋 Battery Capacity (kWh)", 
        min_value=0.0, 
        max_value=25.0, 
        value=10.0, 
        step=0.5,
        help="0 kWh simulates the system without storage."
    )
    max_power = st.sidebar.slider(
        "⚡ Max. Power (kW)", 
        min_value=1.0, 
        max_value=10.0, 
        value=5.0, 
        step=0.1,
        help="Max. charging/discharging power of the inverter (e.g. Vault-Tec Sunny Boy Storage)."
    )
    efficiency = st.sidebar.slider(
        "📉 System Efficiency", 
        min_value=0.80, 
        max_value=1.00, 
        value=0.90, 
        step=0.01,
        format="%.2f",
        help="Ratio of discharged to charged energy (Round-Trip)."
    )
    st.sidebar.markdown("---")
    st.sidebar.info("Model: XGBoost Forecasting + Greedy Optimization")

    # Run simulation
    results = simulate_battery(test, battery_capacity, max_power, efficiency)

    # --- 3. MAIN DASHBOARD ---
    st.title("🔋 Smart Energy Management Simulation")

    # --- KPI Calculation (Business Value!) ---
    
    # 1. Baseline Simulation (Without Battery)
    # We simulate once with 0 kWh storage to see the "Baseline" effect.
    baseline_results = simulate_battery(test, capacity=0.0, max_power=max_power, efficiency=efficiency)
    
    # 2. Calculate Metrics
    def calculate_kpis(df):
        total_load = df['consumption_kWh'].sum()
        total_pv = df['pv_generation_kWh'].sum()
        total_import = df['grid_import_kWh'].sum()
        total_export = df['grid_export_kWh'].sum()
        
        # Autarky = (Consumption - Grid Import) / Consumption
        autarky = (total_load - total_import) / total_load if total_load > 0 else 0
        
        # Self-consumption = (PV - Grid Export) / PV
        self_consumption = (total_pv - total_export) / total_pv if total_pv > 0 else 0
        
        return {
            'load': total_load,
            'import': total_import, 
            'export': total_export,
            'autarky': autarky,
            'self_consumption': self_consumption
        }

    kpis_opt = calculate_kpis(results)
    kpis_base = calculate_kpis(baseline_results)

    # 3. Financial Comparison
    PRICE_IMPORT = 0.30 # €/kWh
    PRICE_EXPORT = 0.08 # €/kWh

    cost_opt = (kpis_opt['import'] * PRICE_IMPORT) - (kpis_opt['export'] * PRICE_EXPORT)
    cost_base = (kpis_base['import'] * PRICE_IMPORT) - (kpis_base['export'] * PRICE_EXPORT)
    savings = cost_base - cost_opt

    # 4. Battery Cycles
    # One full cycle = Discharged Energy / Capacity
    total_discharge = results['battery_discharge_kWh'].sum() if 'battery_discharge_kWh' in results.columns else 0 # Ensure column exists or derive
    # Note: simple simulation might not store discharge separately in 'battery_flow', need to check.
    # Actually battery_flow is positive for charge, negative for discharge.
    if 'battery_flow' in results.columns:
         # Discharge is negative flow
         total_discharge = results[results['battery_flow'] < 0]['battery_flow'].abs().sum()
    
    cycles = total_discharge / battery_capacity if battery_capacity > 0 else 0

    # --- Display Metrics ---
    
    st.subheader("📊 Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Autarky Rate", f"{kpis_opt['autarky']:.1%}", delta=f"{(kpis_opt['autarky'] - kpis_base['autarky']):.1%} vs. Without")
    col2.metric("Self-consumption Rate", f"{kpis_opt['self_consumption']:.1%}", delta=f"{(kpis_opt['self_consumption'] - kpis_base['self_consumption']):.1%} vs. Without")
    col3.metric("Electricity Costs (Sim)", f"{cost_opt:.2f} €", delta=f"{savings:.2f} € Saved", delta_color="inverse") # Green when costs decrease (inverse)
    col4.metric("Battery Cycles", f"{cycles:.1f}", help="Full cycles in simulation period")

    # --- 4. VISUALIZATION (Plotly) ---
    st.subheader("Energy Flow & Storage Level")

    # Create a graph with two axes (Left: Power, Right: SoC)
    fig = go.Figure()

    # Area for PV
    if 'pv_generation_kWh' in results.columns:
        fig.add_trace(go.Scatter(x=results.index, y=results['pv_generation_kWh'], fill='tozeroy', name='PV Generation', line=dict(color='#ffcc00'))) # Yellow-ish
    # Line for Consumption
    if 'consumption_kWh' in results.columns:
        fig.add_trace(go.Scatter(x=results.index, y=results['consumption_kWh'], name='Consumption', line=dict(color='#004e9e'))) # Blue-ish
    # Area for Battery SoC (Secondary Axis)
    if 'battery_soc_kWh' in results.columns:
        fig.add_trace(go.Scatter(x=results.index, y=results['battery_soc_kWh'], name='Battery Charge (kWh)', line=dict(color='#66cc00', width=2), yaxis='y2'))

    # Determine Max Power for Y-Axis
    max_y_val = 0
    cols_to_check = ['pv_generation_kWh', 'consumption_kWh']
    for c in cols_to_check:
        if c in results.columns:
            max_y_val = max(max_y_val, results[c].max())
            
    if max_y_val == 0: max_y_val = 5.0 # Fallback

    # Adjust Layout
    fig.update_layout(
        xaxis_title="Time",
        yaxis=dict(title="Power (kW)", range=[0, max_y_val * 1.1]),
        yaxis2=dict(title="Storage Level (kWh)", overlaying='y', side='right', range=[0, battery_capacity * 1.1]),
        legend=dict(x=0, y=1.1, orientation='h'),
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- 5. DATA TABLE (Transparency) ---
    with st.expander("View Detailed Raw Data"):
        st.dataframe(results)
