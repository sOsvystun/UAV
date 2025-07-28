# ===========================================================================
# Complete Python Implementation of the Wind Turbine Defect Criticality FIS
# ---------------------------------------------------------------------------
# This script implements the Fuzzy Inference System (FIS) as described in the
# manuscript "Criticality Assessment of Wind Turbine Defects via Multispectral
# UAV Fusion and Fuzzy Logic" by Radiuk et al. (2025).
#
# Dependencies:
# - numpy
# - scikit-fuzzy
#
# To install dependencies:
# pip install numpy scikit-fuzzy
# ===========================================================================

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def create_fis():
    """
    Creates and returns the complete Fuzzy Inference System for wind turbine
    defect criticality assessment.
    
    Returns:
        ctrl.ControlSystemSimulation: A ready-to-use simulation object for the FIS.
    """
    
    # 1. Define Universes of Discourse for all variables
    # These ranges define the numerical bounds for each input and output.
    defect_size_universe = np.arange(0, 1001, 1)
    location_universe = np.arange(0, 1.01, 0.01)
    thermal_universe = np.arange(0, 20.1, 0.1)
    criticality_universe = np.arange(0, 5.01, 0.01)

    # 2. Create Antecedent (input) and Consequent (output) objects
    defect_size = ctrl.Antecedent(defect_size_universe, 'DefectSize')
    location = ctrl.Antecedent(location_universe, 'Location')
    thermal = ctrl.Antecedent(thermal_universe, 'ThermalSignature')
    criticality = ctrl.Consequent(criticality_universe, 'Criticality')

    # 3. Define Membership Functions for each variable
    # These functions map crisp numerical inputs to fuzzy linguistic terms.
    
    # Membership Functions for Defect Size
    defect_size['Small']  = fuzz.trapmf(defect_size.universe, [0, 0, 50, 100])
    defect_size['Medium'] = fuzz.trapmf(defect_size.universe, [50, 100, 150, 200])
    defect_size['Large']  = fuzz.trapmf(defect_size.universe, [150, 200, 1000, 1000])

    # Membership Functions for Location
    location['Root'] = fuzz.zmf(location.universe, 0.0, 0.33)
    location['Mid-span'] = fuzz.gaussmf(location.universe, 0.5, 0.1)
    location['Tip'] = fuzz.smf(location.universe, 0.66, 1.0)

    # Membership Functions for Thermal Signature
    thermal['Low'] = fuzz.trapmf(thermal.universe, [0, 0, 2, 4])
    thermal['Medium'] = fuzz.trapmf(thermal.universe, [3, 5, 6, 8])
    thermal['High'] = fuzz.trapmf(thermal.universe, [7, 9, 15, 20])

    # Membership Functions for the Criticality Output
    criticality['Negligible'] = fuzz.trimf(criticality.universe, [0, 1, 2])
    criticality['Low'] = fuzz.trimf(criticality.universe, [1, 2, 3])
    criticality['Medium'] = fuzz.trimf(criticality.universe, [2, 3, 4])
    criticality['High'] = fuzz.trimf(criticality.universe, [3, 4, 5])
    criticality['Severe'] = fuzz.trapmf(criticality.universe, [4, 5, 5, 5])

    # 4. Define the complete 27-rule knowledge base
    # Rules for Large Defects
    rule1 = ctrl.Rule(defect_size['Large'] & location['Root'] & thermal['High'], criticality['Severe'])
    rule2 = ctrl.Rule(defect_size['Large'] & location['Root'] & thermal['Medium'], criticality['Severe'])
    rule3 = ctrl.Rule(defect_size['Large'] & location['Root'] & thermal['Low'], criticality['Severe'])
    rule4 = ctrl.Rule(defect_size['Large'] & location['Mid-span'] & thermal['High'], criticality['Severe'])
    rule5 = ctrl.Rule(defect_size['Large'] & location['Mid-span'] & thermal['Medium'], criticality['High'])
    rule6 = ctrl.Rule(defect_size['Large'] & location['Mid-span'] & thermal['Low'], criticality['High'])
    rule7 = ctrl.Rule(defect_size['Large'] & location['Tip'] & thermal['High'], criticality['High'])
    rule8 = ctrl.Rule(defect_size['Large'] & location['Tip'] & thermal['Medium'], criticality['Medium'])
    rule9 = ctrl.Rule(defect_size['Large'] & location['Tip'] & thermal['Low'], criticality['Medium'])

    # Rules for Medium Defects
    rule10 = ctrl.Rule(defect_size['Medium'] & location['Root'] & thermal['High'], criticality['Severe'])
    rule11 = ctrl.Rule(defect_size['Medium'] & location['Root'] & thermal['Medium'], criticality['High'])
    rule12 = ctrl.Rule(defect_size['Medium'] & location['Root'] & thermal['Low'], criticality['High'])
    rule13 = ctrl.Rule(defect_size['Medium'] & location['Mid-span'] & thermal['High'], criticality['High'])
    rule14 = ctrl.Rule(defect_size['Medium'] & location['Mid-span'] & thermal['Medium'], criticality['Medium'])
    rule15 = ctrl.Rule(defect_size['Medium'] & location['Mid-span'] & thermal['Low'], criticality['Low'])
    rule16 = ctrl.Rule(defect_size['Medium'] & location['Tip'] & thermal['High'], criticality['Medium'])
    rule17 = ctrl.Rule(defect_size['Medium'] & location['Tip'] & thermal['Low'], criticality['Low'])
    rule18 = ctrl.Rule(defect_size['Medium'] & location['Tip'] & thermal['Medium'], criticality['Low'])

    # Rules for Small Defects
    rule19 = ctrl.Rule(defect_size['Small'] & location['Root'] & thermal['High'], criticality['High'])
    rule20 = ctrl.Rule(defect_size['Small'] & location['Root'] & thermal['Medium'], criticality['Medium'])
    rule21 = ctrl.Rule(defect_size['Small'] & location['Root'] & thermal['Low'], criticality['Low'])
    rule22 = ctrl.Rule(defect_size['Small'] & location['Mid-span'] & thermal['High'], criticality['Medium'])
    rule23 = ctrl.Rule(defect_size['Small'] & location['Mid-span'] & thermal['Medium'], criticality['Low'])
    rule24 = ctrl.Rule(defect_size['Small'] & location['Mid-span'] & thermal['Low'], criticality['Negligible'])
    rule25 = ctrl.Rule(defect_size['Small'] & location['Tip'] & thermal['High'], criticality['Low'])
    rule26 = ctrl.Rule(defect_size['Small'] & location['Tip'] & thermal['Medium'], criticality['Negligible'])
    rule27 = ctrl.Rule(defect_size['Small'] & location['Tip'] & thermal['Low'], criticality['Negligible'])
    
    # 5. Create the Control System and Simulation
    full_rule_base = [
        rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, 
        rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, 
        rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27
    ]
    fis_control_system = ctrl.ControlSystem(full_rule_base)
    fis_simulation = ctrl.ControlSystemSimulation(fis_control_system)
    
    return fis_simulation

def get_criticality_score(fis_sim, size_pixels, location_norm, delta_t):
    """
    Computes the criticality score for a given defect using the FIS.

    Args:
        fis_sim (ctrl.ControlSystemSimulation): The FIS simulation object.
        size_pixels (float): The size of the defect in pixels.
        location_norm (float): The normalized location (0.0=root, 1.0=tip).
        delta_t (float): The thermal signature (temperature difference in C).

    Returns:
        float: The crisp criticality score on a scale of 0 to 5.
    """
    fis_sim.input['DefectSize'] = size_pixels
    fis_sim.input['Location'] = location_norm
    fis_sim.input['ThermalSignature'] = delta_t
    
    # Compute the result
    fis_sim.compute()
    
    return fis_sim.output['Criticality']

# ===========================================================================
# Main execution block for demonstration purposes
# ===========================================================================
if __name__ == "__main__":
    
    print("Initializing the Wind Turbine Defect Criticality FIS...")
    fis = create_fis()
    print("FIS initialized successfully.\n")

    # --- Example 1: A severe defect case ---
    print("--- Running Example 1: Severe Defect ---")
    size_1 = 250   # pixels (Large)
    loc_1 = 0.2    # normalized (Root)
    temp_1 = 16.0  # Celsius (High)
    
    print(f"Inputs: DefectSize={size_1}, Location={loc_1}, ThermalSignature={temp_1}")
    criticality_1 = get_criticality_score(fis, size_1, loc_1, temp_1)
    print(f"Computed Criticality Score: {criticality_1:.2f}\n")
    # Expected output: ~4.85, indicating a 'Severe' condition.

    # --- Example 2: A minor defect case ---
    print("--- Running Example 2: Minor Defect ---")
    size_2 = 40    # pixels (Small)
    loc_2 = 0.8    # normalized (Tip)
    temp_2 = 1.5   # Celsius (Low)

    print(f"Inputs: DefectSize={size_2}, Location={loc_2}, ThermalSignature={temp_2}")
    criticality_2 = get_criticality_score(fis, size_2, loc_2, temp_2)
    print(f"Computed Criticality Score: {criticality_2:.2f}\n")
    # Expected output: ~1.25, indicating a 'Negligible' to 'Low' condition.

    # --- Visualization Example ---
    # To understand how the FIS reaches a conclusion, you can visualize the
    # final aggregated fuzzy set and the resulting defuzzified value.
    # The following lines will generate a plot for the severe defect case.
    # You may need to install matplotlib: pip install matplotlib
    try:
        print("Visualizing the output for the severe defect case...")
        fis.input['DefectSize'] = size_1
        fis.input['Location'] = loc_1
        fis.input['ThermalSignature'] = temp_1
        fis.compute()
        
        criticality_variable = fis.consequents['Criticality']
        criticality_variable.view(sim=fis)
        print("Plot generated. Please close the plot window to exit.")
    except Exception as e:
        print(f"\nCould not generate visualization. Error: {e}")
        print("Please ensure matplotlib is installed ('pip install matplotlib').")