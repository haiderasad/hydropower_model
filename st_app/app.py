import streamlit as st
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import json
import pandas as pd

class HydropowerCalculator:
    def __init__(self):
        self.g = 9.81  # Gravity
        self.secs_hr = 3600  # Seconds in an hour
        self.eff_t = 0.90  # Efficiency turbine mode
        self.eff_p = 0.91  # Efficiency pump turbine
        self.hl_rated_ini_percent = 0.05  # Initial rated headloss percent
        self.__load_volume_data("elevation_volume.json")

    def volume_lookup(self,elevation):
        """Lookup or interpolate volume based on elevation."""
        elevations = list(self.u_res_rang.keys()) + list(self.l_res_rang.keys())
        volumes = list(self.u_res_rang.values()) + list(self.l_res_rang.values())
        volume_interpolator = interp1d(elevations, volumes, kind='linear', bounds_error=False, fill_value="extrapolate")

        if isinstance(elevation, np.ndarray):
            return np.array([volume_interpolator(el) for el in elevation])
        else:
            return volume_interpolator(elevation)
        
    def __load_volume_data(self,file_path):
        # Load JSON data from the file
        #with open(file_path, 'r') as file:
        #    data = json.load(file)
        # Convert keys and values to floats
        self.u_res_rang={
        "360.0": "28601016.1",
        "355.0": "26074652.9",
        "350.0": "23688301.1",
        "345.0": "21440442.0",
        "340.0": "19348609.0",
        "335.0": "17386185.3",
        "330.0": "15547204.1",
        "325.0": "13905178.5",
        "320.0": "12365875.2",
        "315.0": "10899667.5",
        "310.0": "9502636.3",
        "305.0": "8259678.8",
        "300.0": "7086569.8",
        "295.0": "5958291.3",
        "290.0": "4963086.1",
        "285.0": "4079506.3",
        "280.0": "3277253.0",
        "275.0": "2626475.4",
        "270.0": "2046862.4",
        "265.0": "1575161.7",
        "260.0": "1173751.0",
        "255.0": "817220.2",
        "250.0": "565285.1",
        "245.0": "366519.2",
        "240.0": "198353.5",
        "235.0": "91400.0",
        "230.0": "6515.6"}    #{float(key): float(value) for key, value in data['u_reservoir_range'].items()}
        self.l_res_rang= {
        "155.0": "10000000.0",
        "150.0": "7000000.0",
        "145.0": "4977061.9",
        "140.0": "4039600.9",
        "135.0": "3255642.2",
        "130.0": "2527796.3",
        "125.0": "1879396.2",
        "120.0": "1282028.9",
        "115.0": "758076.6",
        "110.0": "281725.8",
        "105.0": "129246.0"
    }   #{float(key): float(value) for key, value in data['l_reservoir_range'].items()}

    def get_live_volume(self,fsl,mol):
        fsl_vol = self.volume_lookup(fsl)
        mol_vol = self.volume_lookup(mol)
        live_vol = fsl_vol - mol_vol
        return live_vol
    
    def calculate_reservoir_parameters(self,u_fsl, u_mol, l_fsl, l_mol):
        """Calculate reservoir parameters."""
        u_fsl_vol = self.volume_lookup(u_fsl)
        u_mol_vol = self.volume_lookup(u_mol)
        l_fsl_vol = self.volume_lookup(l_fsl)
        l_mol_vol = self.volume_lookup(l_mol)

        u_live_vol = u_fsl_vol - u_mol_vol
        l_live_vol = l_fsl_vol - l_mol_vol

        hg_max = u_fsl - l_mol
        hg_min = u_mol - l_fsl
        hg_avg = (hg_max + hg_min) / 2

        # Handle divide by zero in head range calculation
        if hg_min <= 0:
            hr_gross = float('inf')  # or an appropriate large number or error value
        else:
            hr_gross = hg_max / hg_min

        #add condition if HEAD_RANGE > 1.35
        if self.rating_point == 'hg_avg':
            hg_rated = hg_avg
        else:
            hg_rated = hg_min

        hl_rated_ini = hg_rated * self.hl_rated_ini_percent
        hn_rated_gen_ini = hg_rated - hl_rated_ini

        return {
            'u_live_vol': u_live_vol, 'l_live_vol': l_live_vol,
            'hg_max': hg_max, 'hg_min': hg_min, 'hg_avg': hg_avg,
            'hr_gross': hr_gross, 'hl_rated_ini': hl_rated_ini,
            'hn_rated_gen_ini': hn_rated_gen_ini
        }
        
    def calculate_power_output(self,hn_rated_gen_ini, live_vol_gen, cycle_hours_gen):
        """Calculate power output."""
        q_t_rated = live_vol_gen / (cycle_hours_gen * self.secs_hr)
        power_gen_rated = q_t_rated * hn_rated_gen_ini * self.g * self.eff_t / 1000  # in MW
        return power_gen_rated
    
    def set_rating_p(self, rp):
        self.rating_point=rp
        
    def case_1(self, u_fsl, u_mol, l_fsl, l_mol, cycle_hours_gen):
        reservoir_params = self.calculate_reservoir_parameters(u_fsl, u_mol, l_fsl, l_mol)
        live_vol_gen = min(reservoir_params['u_live_vol'], reservoir_params['l_live_vol'])
        power_output = self.calculate_power_output(reservoir_params['hn_rated_gen_ini'], live_vol_gen, cycle_hours_gen)
        return power_output, reservoir_params
    
    def balance_live_volume(self,prim_mol,prim_fsl,primary_live_vol, secondary_fixed_level, secondary_fixed_is_fsl, is_upper_primary, elevation_range):
        def objective(estimated_level):
            if is_upper_primary:
                # If the fixed level is FSL, then estimated_level is MOL, and vice versa.
                l_fsl = secondary_fixed_level if secondary_fixed_is_fsl else estimated_level
                l_mol = estimated_level if secondary_fixed_is_fsl else secondary_fixed_level
                u_mol=prim_mol
                u_fsl=prim_fsl

                secondary_live_vol = self.calculate_reservoir_parameters(u_fsl, u_mol, l_fsl, l_mol)['l_live_vol']
            else:
                u_fsl = secondary_fixed_level if secondary_fixed_is_fsl else estimated_level
                u_mol = estimated_level if secondary_fixed_is_fsl else secondary_fixed_level
                l_mol=prim_mol
                l_fsl=prim_fsl

                secondary_live_vol = self.calculate_reservoir_parameters(u_fsl, u_mol, l_fsl, l_mol)['u_live_vol']

            return abs(primary_live_vol - secondary_live_vol)

        result = minimize(objective, x0=np.mean(elevation_range), bounds=[(min(elevation_range), max(elevation_range))])
        return result.x[0] if result.success else None


    def solve_for_power(self,prim_fsl,prim_mol, desired_power, cycle_hours_gen, secondary_fixed_level,secondary_fixed_is_fsl, is_upper_primary, elevation_range):
        def objective(estimated_level):
            if is_upper_primary:
                u_mol=prim_mol
                u_fsl=prim_fsl
                # If the fixed level is FSL, then estimated_level is MOL, and vice versa.
                l_fsl = secondary_fixed_level if secondary_fixed_is_fsl else estimated_level
                l_mol = estimated_level if secondary_fixed_is_fsl else secondary_fixed_level
                reser_params= self.calculate_reservoir_parameters(u_fsl, u_mol, l_fsl, l_mol)
                hn_rated_gen_ini = reser_params['hn_rated_gen_ini']
                hn_rated_gen_ini = reser_params['hn_rated_gen_ini']
                live_vol_gen=min(reser_params['u_live_vol'],reser_params['l_live_vol'])
            else:
                u_fsl = secondary_fixed_level if secondary_fixed_is_fsl else estimated_level
                u_mol = estimated_level if secondary_fixed_is_fsl else secondary_fixed_level
                l_mol=prim_mol
                l_fsl=prim_fsl
                reser_params=self.calculate_reservoir_parameters(u_fsl, u_mol, l_fsl, l_mol)
                hn_rated_gen_ini = reser_params['hn_rated_gen_ini']
                live_vol_gen=min(reser_params['u_live_vol'],reser_params['l_live_vol'])

            power_output = self.calculate_power_output(hn_rated_gen_ini, live_vol_gen, cycle_hours_gen)
            return abs(desired_power - power_output)

        result = minimize(objective, x0=np.mean(elevation_range), bounds=[(min(elevation_range), max(elevation_range))])
        return result.x[0] if result.success else None
    def solve_for_power_two_wls(self,prim_fsl,prim_mol, desired_power, cycle_hours_gen, is_upper_primary, elevation_range):
        def objective(estimated_levels):
            # Unpack estimated levels based on which reservoir is primary
            if is_upper_primary:
                # If upper reservoir is primary, the levels for lower reservoir are being estimated
                l_fsl_param, l_mol_param = estimated_levels
                u_fsl_param, u_mol_param = prim_fsl, prim_mol
            else:
                # If lower reservoir is primary, the levels for upper reservoir are being estimated
                u_fsl_param, u_mol_param = estimated_levels
                l_fsl_param, l_mol_param = l_fsl, l_mol

            # Calculate reservoir parameters and power output
            res_params = self.calculate_reservoir_parameters(u_fsl_param, u_mol_param, l_fsl_param, l_mol_param)
            live_vol_gen = min(res_params['u_live_vol'], res_params['l_live_vol'])
            power_output = self.calculate_power_output(res_params['hn_rated_gen_ini'], live_vol_gen, cycle_hours_gen)
            
            return abs(desired_power - power_output)

        # Initialize estimated levels based on which reservoir is primary
        initial_estimates = [np.mean(elevation_range)] * 2 if is_upper_primary else [np.mean(elevation_range)] * 2

        result = minimize(objective, x0=initial_estimates, bounds=[(min(elevation_range), max(elevation_range))] * 2)
        return result.x if result.success else None
    
    def adjust_for_head_range(self,prim_fsl,prim_mol, desired_head_range, secondary_fixed_level,secondary_fixed_is_fsl, is_upper_primary, elevation_range):
        def objective(estimated_level):
            if is_upper_primary:
                u_mol=prim_mol
                u_fsl=prim_fsl
                l_fsl = secondary_fixed_level if secondary_fixed_is_fsl else estimated_level
                l_mol = estimated_level if secondary_fixed_is_fsl else secondary_fixed_level
                reser_params= self.calculate_reservoir_parameters(u_fsl, u_mol, l_fsl, l_mol)
                hg_max, hg_min = reser_params['hg_max'],reser_params['hg_min']
            else:
                u_fsl = secondary_fixed_level if secondary_fixed_is_fsl else estimated_level
                u_mol = estimated_level if secondary_fixed_is_fsl else secondary_fixed_level
                l_mol=prim_mol
                l_fsl=prim_fsl
                reser_params=self.calculate_reservoir_parameters(u_fsl, u_mol, l_fsl, l_mol)
                hg_max, hg_min = reser_params['hg_max'],reser_params['hg_min']
            
            current_head_range = hg_max / hg_min
            return abs(desired_head_range - current_head_range)

        result = minimize(objective, x0=np.mean(elevation_range), bounds=[(min(elevation_range), max(elevation_range))])
        return result.x[0] if result.success else None


    # ... (Add method case_3)

# Streamlit UI
st.title("Hydropower Plant Reservoir Calculator")

# Case Selection
case_number = st.selectbox("Select the Case Number", [1, 2, 3])

calculator = HydropowerCalculator()

# Display inputs based on selected case
if case_number == 1:
    rating_point = st.selectbox("Rating Point", ['hg_avg', 'hg_min'])
    u_fsl = st.slider("Upper Reservoir FSL (masl)", min_value=min(calculator.u_res_rang.keys()), max_value=max(calculator.u_res_rang.keys()))
    u_mol = st.slider("Upper Reservoir MOL (masl)", min_value=min(calculator.u_res_rang.keys()), max_value=u_fsl-1)
    l_fsl = st.slider("Lower Reservoir FSL (masl)", min_value=min(calculator.l_res_rang.keys()), max_value=max(calculator.l_res_rang.keys()))
    l_mol = st.slider("Lower Reservoir MOL (masl)", min_value=min(calculator.l_res_rang.keys()), max_value=l_fsl-1)
    cycle_hours_gen = st.slider("Cycle Time in Hours", min_value=1, max_value=24,value=8)
    
    if st.button("Calculate Values"):
        calculator.set_rating_p(rating_point)
        power_output, reservoir_params = calculator.case_1(u_fsl, u_mol, l_fsl, l_mol, cycle_hours_gen)
        results_df = pd.DataFrame(reservoir_params.items(), columns=['Parameter', 'Value'])
        power_output_df = pd.DataFrame([{'Parameter': 'Power Output (MW)', 'Value': power_output}])
        final_results_df = pd.concat([results_df, power_output_df], ignore_index=True)
        st.table(final_results_df)

elif case_number == 2:
    
    rating_point = st.selectbox("Rating Point", ['hg_avg', 'hg_min'])
    is_upper_primary = st.radio("Select Governing Reservoir", ('Upper', 'Lower')) == 'Upper'
    
    if is_upper_primary:
        u_fsl = st.slider("Upper Reservoir FSL (masl)", min_value=min(calculator.u_res_rang.keys()), max_value=max(calculator.u_res_rang.keys()))
        u_mol = st.slider("Upper Reservoir MOL (masl)", min_value=min(calculator.u_res_rang.keys()), max_value=u_fsl-1)
        l_fixed_level_type = st.selectbox("Is the Lower Reservoir Fixed Level FSL or MOL?", ['FSL', 'MOL'])
        l_fixed_level = st.slider("Lower Reservoir Fixed Level (masl)", min_value=min(calculator.l_res_rang.keys()), max_value=max(calculator.l_res_rang.keys()))

    else:
        l_fsl = st.slider("Lower Reservoir FSL (masl)", min_value=min(calculator.l_res_rang.keys()), max_value=max(calculator.l_res_rang.keys()))
        l_mol = st.slider("Lower Reservoir MOL (masl)", min_value=min(calculator.l_res_rang.keys()), max_value=l_fsl-1)
        u_fixed_level_type = st.selectbox("Is the Upper Reservoir Fixed Level FSL or MOL?", ['FSL', 'MOL'])
        u_fixed_level = st.slider("Upper Reservoir Fixed Level (masl)", min_value=min(calculator.u_res_rang.keys()), max_value=max(calculator.u_res_rang.keys()))
    
    
    # UI for selecting the sub-case
    sub_case = st.selectbox("Select Sub-Case for Case 2", ['Balance Live Volume', 'Solve for Desired Power', 'Adjust for Head Range'])

    # Inputs based on the sub-case
    if sub_case == 'Solve for Desired Power':
        desired_power = st.slider("Desired Power Output (MW)", min_value=0.0, value=10.0)
        cycle_hours_gen = st.slider("Enter the Cycle Time in Hours", min_value=1, max_value=24,value=8)
    elif sub_case == 'Adjust for Head Range':
        desired_head_range = st.slider("Desired Head Range", min_value=1.0, value=2.0)

    # Button to perform calculations
    if st.button("Calculate"):
        calculator.set_rating_p(rating_point)
        # Determine primary and secondary reservoir parameters
        if is_upper_primary:
            primary_live_vol = calculator.get_live_volume(u_fsl, u_mol)
            secondary_fixed_level = l_fixed_level
            secondary_fixed_is_fsl = (l_fixed_level_type == 'FSL')
            elevation_range =[min(calculator.l_res_rang.keys()),max(calculator.l_res_rang.keys())] # Elevation range for lower reservoir
            prim_fsl=u_fsl
            prim_mol=u_mol
        else:
            primary_live_vol = calculator.get_live_volume(l_fsl, l_mol)
            secondary_fixed_level = u_fixed_level
            secondary_fixed_is_fsl = (u_fixed_level_type == 'FSL')
            elevation_range = [min(calculator.u_res_rang.keys()),max(calculator.u_res_rang.keys())]  # Elevation range for upper reservoir
            prim_fsl=l_fsl
            prim_mol=l_mol

        if sub_case == 'Balance Live Volume':
            balanced_level = calculator.balance_live_volume(prim_mol,prim_fsl,primary_live_vol, secondary_fixed_level, secondary_fixed_is_fsl, is_upper_primary, elevation_range)
            st.write(f"Balanced Water Level for Secondary Reservoir : {round(balanced_level,2)} masl")

        elif sub_case == 'Solve for Desired Power':
            adjusted_level = calculator.solve_for_power(prim_fsl,prim_mol, desired_power, cycle_hours_gen, secondary_fixed_level, secondary_fixed_is_fsl, is_upper_primary, elevation_range)
            st.write(f"Adjusted Water Level for Desired Power in Secondary Reservoir: {adjusted_level} masl")

        elif sub_case == 'Adjust for Head Range':
            adjusted_level = calculator.adjust_for_head_range(prim_fsl,prim_mol, desired_head_range, secondary_fixed_level, secondary_fixed_is_fsl, is_upper_primary, elevation_range)
            st.write(f"Adjusted Water Level for Desired Head Range in Secondary Reservoir: {adjusted_level} masl")

elif case_number == 3:
    # Add inputs for Case 3
    rating_point = st.selectbox("Rating Point", ['hg_avg', 'hg_min'])
    is_upper_primary = st.radio("Select Independent Reservoir", ('Upper', 'Lower')) == 'Upper'

    if is_upper_primary:
        u_fsl = st.slider("Upper Reservoir FSL (masl)", min_value=min(calculator.u_res_rang.keys()), max_value=max(calculator.u_res_rang.keys()))
        u_mol = st.slider("Upper Reservoir MOL (masl)", min_value=min(calculator.u_res_rang.keys()), max_value=u_fsl-1)
    else:
        l_fsl = st.slider("Lower Reservoir FSL (masl)", min_value=min(calculator.l_res_rang.keys()), max_value=max(calculator.l_res_rang.keys()))
        l_mol = st.slider("Lower Reservoir MOL (masl)", min_value=min(calculator.l_res_rang.keys()), max_value=l_fsl-1)

    # UI for selecting the sub-case for the secondary reservoir
    sub_case_3 = st.selectbox("Select Objective for Secondary Reservoir", ['Solve for Desired Power', 'Adjust for Head Range'])

    # Inputs based on the selected sub-case
    if sub_case_3 == 'Solve for Desired Power':
        desired_power = st.slider("Desired Power Output (MW)", min_value=0.0, value=10.0)
        cycle_hours_gen = st.slider("Cycle Time in Hours", min_value=1, max_value=24, value=8)
    elif sub_case_3 == 'Adjust for Head Range':
        desired_head_range = st.slider("Desired Head Range", min_value=1.0, value=2.0)

    # Button to perform calculations for Case 3
    if st.button("Calculate"):
        calculator.set_rating_p(rating_point)  # Assuming rating_point is globally selected

        if is_upper_primary:
            primary_live_vol = calculator.get_live_volume(u_fsl, u_mol)
            elevation_range = [min(calculator.l_res_rang.keys()), max(calculator.l_res_rang.keys())]  # For lower reservoir
            prim_fsl, prim_mol = u_fsl, u_mol
        else:
            primary_live_vol = calculator.get_live_volume(l_fsl, l_mol)
            elevation_range= [min(calculator.u_res_rang.keys()), max(calculator.u_res_rang.keys())]  # For upper reservoir
            prim_fsl, prim_mol = l_fsl, l_mol

        # Implement logic based on selected sub-case
        if sub_case_3 == 'Solve for Desired Power':
            # Logic to solve for desired power
            adjusted_level = calculator.solve_for_power_two_wls(prim_fsl,prim_mol, desired_power, cycle_hours_gen, is_upper_primary, elevation_range)
            st.write(f"Adjusted Water Level for Desired Power in Secondary Reservoir: {adjusted_level} masl")
        elif sub_case_3 == 'Adjust for Head Range':
            # Logic to adjust for head range
            pass