"""
Main script to process a list of wind turbine defects and calculate their
criticality scores using the Fuzzy Inference System.

This script reads defect data from a CSV file, computes the criticality for each,
and saves the results to a new CSV file.

Usage:
    python src/process_defects.py [--input INPUT_PATH] [--output OUTPUT_PATH]
"""
import pandas as pd
import argparse
from skfuzzy import control as ctrl
from fis_core import create_fis

def process_defects(input_path: str, output_path: str):
    """
    Loads defect data, computes criticality scores, and saves the results.
    """
    print(f"Loading defects from: {input_path}")
    try:
        defects_df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    # Initialize the FIS
    fis_control_system = create_fis()
    fis_simulation = ctrl.ControlSystemSimulation(fis_control_system)
    
    criticality_scores = []
    
    print("Processing defects...")
    for _, row in defects_df.iterrows():
        fis_simulation.input['DefectSize'] = row['size_pixels']
        fis_simulation.input['Location'] = row['location_normalized']
        fis_simulation.input['ThermalSignature'] = row['delta_t_celsius']
        
        fis_simulation.compute()
        criticality_scores.append(fis_simulation.output['Criticality'])
        
    defects_df['criticality_score'] = criticality_scores
    
    # Round to align with EPRI 1-5 scale for easier interpretation
    defects_df['epri_level_estimate'] = np.round(defects_df['criticality_score']).astype(int)
    
    print(f"Saving results to: {output_path}")
    defects_df.to_csv(output_path, index=False)
    print("Processing complete.")
    print("\n--- Results ---")
    print(defects_df.head())
    print("---------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process wind turbine defects to assess criticality.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/sample_defects.csv",
        help="Path to the input CSV file with defect data."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample_defects_with_criticality.csv",
        help="Path to save the output CSV file with criticality scores."
    )
    args = parser.parse_args()
    
    process_defects(args.input, args.output)```

### **File 8: `tests/test_fis.py`**

This file provides unit tests to validate the FIS logic, ensuring reproducibility and correctness.

```python
"""
Unit tests for the Fuzzy Inference System (FIS) defined in fis_core.py.

These tests validate that the FIS produces expected criticality scores for
a set of known defect scenarios, ensuring the system's logic is correct
and stable.
"""
import pytest
from skfuzzy import control as ctrl
from src.fis_core import create_fis

@pytest.fixture(scope="module")
def fis_simulation():
    """Pytest fixture to create the FIS simulation object once for all tests."""
    fis_control = create_fis()
    return ctrl.ControlSystemSimulation(fis_control)

def test_severe_defect_scenario(fis_simulation):
    """
    Tests a classic severe defect: a large crack at the blade root with a
    high thermal signature. Expects a very high criticality score.
    """
    fis_simulation.input['DefectSize'] = 250
    fis_simulation.input['Location'] = 0.2
    fis_simulation.input['ThermalSignature'] = 16.0
    
    fis_simulation.compute()
    score = fis_simulation.output['Criticality']
    
    assert score == pytest.approx(4.85, abs=0.1)

def test_minor_defect_scenario(fis_simulation):
    """
    Tests a classic minor defect: a small crack at the blade tip with no
    thermal signature. Expects a very low criticality score.
    """
    fis_simulation.input['DefectSize'] = 40
    fis_simulation.input['Location'] = 0.8
    fis_simulation.input['ThermalSignature'] = 1.5
    
    fis_simulation.compute()
    score = fis_simulation.output['Criticality']
    
    assert score == pytest.approx(1.25, abs=0.1)

def test_medium_defect_scenario(fis_simulation):
    """
    Tests a medium-range defect: a medium-sized flaw at the mid-span with
    a moderate thermal signature. Expects a mid-range criticality score.
    """
    fis_simulation.input['DefectSize'] = 125
    fis_simulation.input['Location'] = 0.5
    fis_simulation.input['ThermalSignature'] = 5.5
    
    fis_simulation.compute()
    score = fis_simulation.output['Criticality']
    
    assert 2.5 < score < 3.5