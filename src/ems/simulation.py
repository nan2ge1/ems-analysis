import pandas as pd

def simulate_battery(df: pd.DataFrame, capacity: float, max_power: float, efficiency: float) -> pd.DataFrame:
    """
    Simulate a battery storage system.
    
    Args:
        df: Input DataFrame with 'prediction' (PV) and 'consumption_kWh'.
        capacity: Battery capacity in kWh.
        max_power: Max charge/discharge power in kW.
        efficiency: Round-trip efficiency (applied at charging side for simplicity).
        
    Returns:
        DataFrame with added columns for battery SOC and grid flows.
    """
    # Lists for results
    battery_soc_list = []      # State of charge in kWh
    grid_import_list = []      # Grid import in kWh
    grid_export_list = []      # Grid export in kWh
    battery_flow_list = []     # + Charge, - Discharge
    
    current_soc = 0.0 # Initial value (empty)
    
    for index, row in df.iterrows():
        # 1. Calculate balance: Do we have too much or too little power?
        # We use the PREDICTION ('prediction') for PV here
        pv_generation = row['prediction'] 
        consumption = row['consumption_kWh']
        
        net_energy = pv_generation - consumption
        
        # Reset variables for this step
        flow = 0.0
        import_kwh = 0.0
        export_kwh = 0.0
        
        # --- CASE A: SURPLUS (We want to charge) ---
        if net_energy > 0:
            # How much CAN we physically charge?
            # Limited by: 
            # 1. Free space in battery
            # 2. Max charging power of the inverter
            max_charge_space = capacity - current_soc
            possible_charge = min(net_energy, max_power, max_charge_space)
            
            # Physics: We charge 'possible_charge', but less arrives in the battery (losses)
            # We simplify: We withdraw 'possible_charge' from the system, 
            # but the SoC only increases by efficiency * possible_charge
            effective_charge = possible_charge * efficiency
            
            current_soc += effective_charge
            flow = possible_charge # Positive = Charge
            
            # The rest that didn't fit in the battery goes to the grid
            export_kwh = net_energy - possible_charge
            
        # --- CASE B: DEFICIT (We want to discharge) ---
        else:
            deficit = abs(net_energy)
            
            # How much CAN we withdraw?
            # Limited by:
            # 1. Current content
            # 2. Max discharge power
            possible_discharge = min(deficit, max_power, current_soc)
            
            # Physics: Discharge losses (optional, we often apply efficiency at charging)
            # Here simple: SOC decreases by withdrawal
            current_soc -= possible_discharge
            flow = -possible_discharge # Negative = Discharge
            
            # The rest that the battery couldn't cover comes from the grid
            import_kwh = deficit - possible_discharge
            
        # Safety net (Correct floating point errors)
        current_soc = max(0.0, min(current_soc, capacity))
        
        # Store
        battery_soc_list.append(current_soc)
        grid_import_list.append(import_kwh)
        grid_export_list.append(export_kwh)
        battery_flow_list.append(flow)
        
    # Write results to DataFrame
    result_df = df.copy()
    result_df['battery_soc_kWh'] = battery_soc_list
    result_df['grid_import_kWh'] = grid_import_list
    result_df['grid_export_kWh'] = grid_export_list
    result_df['battery_flow'] = battery_flow_list
    
    return result_df
