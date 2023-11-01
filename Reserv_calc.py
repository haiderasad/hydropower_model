import numpy as np
from scipy.optimize import minimize

class Reservoir_Calc:
    def __init__(self, g=9.81, secs_hr=3600, eff_t=0.9, eff_p=0.91, hl_rated_ini_percent=0.05,cycle_hours_gen=8):
        self.g = g
        self.secs_hr = secs_hr
        self.cycle_hours_gen=cycle_hours_gen
        self.eff_t = eff_t
        self.eff_p = eff_p
        self.hl_rated_ini_percent = hl_rated_ini_percent

    def calculate_power(self, live_vol_gen, hg_rated):
        q_t_rated = live_vol_gen / (self.cycle_hours_gen * self.secs_hr)
        power_gen_rated = q_t_rated * (hg_rated - hg_rated * self.hl_rated_ini_percent) * self.g * self.eff_t / 1000  # in MW
        return power_gen_rated

    def objective_function(self, x):
        # Objective function for optimization (to minimize)
        if self.case == 2:
            # Case 2: Balance live volume
            l_fsl = x[0]
            u_fsl = self.fixed_level
            l_live_vol = u_fsl * self.l_l_ratio - l_fsl * self.u_l_ratio
            u_live_vol = l_fsl * self.u_l_ratio
            live_vol_gen = min(l_live_vol, u_live_vol)
            power_gen_rated = self.calculate_power(live_vol_gen, hg_rated=self.hg_rated)
            return abs(power_gen_rated - self.target_power)
        elif self.case == 3:
            # Case 3: Adjust head range
            u_fsl, l_fsl = x[0], x[1]
            hg_max = u_fsl - l_fsl
            hg_min = self.fixed_hg_min
            hg_rated = (hg_max + hg_min) / 2
            power_gen_rated = self.calculate_power(self.fixed_live_vol_gen, hg_rated=hg_rated)
            return abs(power_gen_rated - self.fixed_power)

    def solve_case(self, case, fixed_level=None, target_power=None, fixed_hg_min=None):
        self.case = case
        if case == 1:
            # Case 1: User supplies water levels for both reservoirs
            u_fsl, l_fsl = fixed_level[0], fixed_level[1]
            hg_max = u_fsl - l_fsl
            hg_min = l_fsl - l_fsl  # Assuming l_fsl is the minimum level for the secondary reservoir
            hg_rated = (hg_max + hg_min) / 2
            live_vol_gen = min(self.u_live_vol, self.l_live_vol)
            power_gen_rated = self.calculate_power(live_vol_gen, hg_rated)
            return hg_max, hg_min, hg_rated, power_gen_rated

        elif case == 2:
            # Case 2: User supplies one water level for the secondary reservoir
            self.fixed_level = fixed_level
            self.u_l_ratio = 1
            self.l_l_ratio = self.l_live_vol / (self.fixed_level * self.u_l_ratio)
            
            self.cycle_hours_gen = self.fixed_cycle_hours_gen
            self.target_power = target_power

            # Optimize to balance live volume
            result = minimize(self.objective_function, x0=fixed_level[0], bounds=[(self.u_mol, self.u_fsl)],
                              method='L-BFGS-B')
            u_fsl = self.u_fsl
            l_fsl = result.x[0]
            hg_max = u_fsl - l_fsl
            hg_min = self.fixed_level - l_fsl
            hg_rated = (hg_max + hg_min) / 2
            live_vol_gen = min(l_fsl * self.u_l_ratio, self.fixed_level * self.l_l_ratio)
            power_gen_rated = self.calculate_power(live_vol_gen, hg_rated)
            return hg_max, hg_min, hg_rated, power_gen_rated

        elif case == 3:
            # Case 3: User supplies independent reservoir and adjusts head range
            self.fixed_hg_min = fixed_hg_min
            self.fixed_power = self.calculate_power(self.fixed_live_vol_gen, hg_rated=fixed_hg_min)
            self.fixed_level = fixed_level
            self.cycle_hours_gen = self.fixed_cycle_hours_gen

            # Optimize to adjust head range
            result = minimize(self.objective_function, x0=[self.u_fsl, self.l_fsl],
                              bounds=[(self.fixed_level, self.u_fsl), (self.l_mol, self.fixed_level)],
                              method='L-BFGS-B')
            u_fsl, l_fsl = result.x
            hg_max = u_fsl - l_fsl
            hg_min = self.fixed_hg_min
            hg_rated = (hg_max + hg_min) / 2
            power_gen_rated = self.calculate_power(self.fixed_live_vol_gen, hg_rated)
            return hg_max, hg_min, hg_rated, power_gen_rated

    def set_parameters(self, u_fsl, u_mol, u_fsl_vol, u_mol_vol, l_fsl, l_mol, l_fsl_vol, l_mol_vol,
                       cycle_hours_gen, fixed_cycle_hours_gen):
        self.u_fsl = u_fsl
        self.u_mol = u_mol
        self.u_fsl_vol = u_fsl_vol
        self.u_mol_vol = u_mol_vol
        self.l_fsl = l_fsl
        self.l_mol = l_mol
        self.l_fsl_vol = l_fsl_vol
        self.l_mol_vol = l_mol_vol
        self.u_live_vol = u_fsl_vol - u_mol_vol
        self.l_live_vol = l_fsl_vol - l_mol_vol
        self.fixed_cycle_hours_gen = fixed_cycle_hours_gen
        self.fixed_live_vol_gen = min(self.u_live_vol, self.l_live_vol)


# Example usage
reservoir_calc = Reservoir_Calc()

# Set parameters
reservoir_calc.set_parameters(u_fsl=100, u_mol=90, u_fsl_vol=50000, u_mol_vol=20000,
l_fsl=20, l_mol=10, l_fsl_vol=30000, l_mol_vol=10000,
cycle_hours_gen=24, fixed_cycle_hours_gen=24)

# Case 1: User supplies water levels for both reservoirs
hg_max, hg_min, hg_rated, power_gen_rated = reservoir_calc.solve_case(case=2, fixed_level=(100, 10))
print("Case 2:")
print("Maximum Gross Head:", hg_max, "m")
print("Minimum Gross Head:", hg_min, "m")
print("Rated Gross Head:", hg_rated)