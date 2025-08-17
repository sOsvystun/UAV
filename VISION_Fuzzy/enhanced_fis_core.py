"""
Enhanced Fuzzy Inference System for Wind Turbine Defect Criticality Assessment
==============================================================================

This module implements the complete three-block framework described in:
"Criticality Assessment of Wind Turbine Defects via Multispectral UAV Fusion and Fuzzy Logic"

The system consists of:
- Block 1: Automated physical and thermal parameterization (integrated with VISION_Recognition)
- Block 2: Expert-driven criticality models
- Block 3: Fuzzy Inference System integration

Author: Based on research by Radiuk et al. (2025)
Dependencies: numpy, scikit-fuzzy, matplotlib
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DefectType(Enum):
    """Enumeration of supported defect types"""
    CRACK = "crack"
    EROSION = "erosion"
    HOTSPOT = "hotspot"

class ComponentType(Enum):
    """Wind turbine blade components with specific criticality weights"""
    BLADE_ROOT = "blade_root"
    BLADE_MID_SPAN = "blade_mid_span"
    BLADE_TIP = "blade_tip"
    SPAR_CAP = "spar_cap"
    TRAILING_EDGE = "trailing_edge"
    NACELLE_HOUSING = "nacelle_housing"
    TOWER_SECTION = "tower_section"
    GENERATOR_HOUSING = "generator_housing"

@dataclass
class DefectParameters:
    """Physical and thermal parameters extracted from image processing"""
    # Geometric parameters
    area_real_mm2: float  # Real-world area in mm²
    length_real_mm: float  # Real-world length in mm
    width_avg_mm: float  # Average width in mm
    perimeter_real_mm: float  # Perimeter in mm
    curvature_avg: float  # Average curvature
    
    # Location parameters
    location_normalized: float  # 0.0 = root, 1.0 = tip
    component_type: ComponentType
    
    # Thermal parameters
    delta_t_max: float  # Maximum temperature difference in °C
    thermal_gradient: float  # Temperature Laplacian
    thermal_contrast: float  # Thermal contrast measure
    
    # Detection metadata
    confidence: float  # Detection confidence
    defect_type: DefectType

@dataclass
class CriticalityWeights:
    """Component-specific weighting coefficients from IEC 61400-5 standard"""
    # Crack criticality weights (β_c)
    CRACK_WEIGHTS = {
        ComponentType.BLADE_ROOT: 1.00,
        ComponentType.BLADE_MID_SPAN: 0.65,
        ComponentType.BLADE_TIP: 0.40,
        ComponentType.SPAR_CAP: 0.80,
        ComponentType.TRAILING_EDGE: 0.50,
    }
    
    # Erosion criticality weights (γ_c)
    EROSION_WEIGHTS = {
        ComponentType.BLADE_ROOT: 0.90,
        ComponentType.BLADE_MID_SPAN: 0.50,
        ComponentType.BLADE_TIP: 0.30,
        ComponentType.NACELLE_HOUSING: 0.70,
        ComponentType.TOWER_SECTION: 0.60,
    }
    
    # Hotspot criticality weights (η_c)
    HOTSPOT_WEIGHTS = {
        ComponentType.GENERATOR_HOUSING: 1.00,
        ComponentType.SPAR_CAP: 0.80,
        ComponentType.BLADE_MID_SPAN: 0.50,
        ComponentType.TOWER_SECTION: 0.70,
        ComponentType.TRAILING_EDGE: 0.60,
    }

class ExpertCriticalityModels:
    """
    Block 2: Expert-driven criticality models based on physics and engineering principles
    
    These models provide initial severity estimates based on formal mathematical
    representations of how different defect types compromise component integrity.
    """
    
    @staticmethod
    def crack_criticality_model(params: DefectParameters) -> float:
        """
        Crack criticality model based on fracture mechanics principles.
        
        Formula: C_exp^crack = β_c * L * w_avg * (1 + κ_avg)
        
        Args:
            params: DefectParameters containing crack measurements
            
        Returns:
            Expert-driven criticality score for crack
        """
        beta_c = CriticalityWeights.CRACK_WEIGHTS.get(params.component_type, 0.5)
        
        # Simplified implementation of the integral model
        # C_exp = β_c * ∫[0,L] w_visible(s) * |r'(s)| * [1 + κ(s)] ds
        # Approximated as: β_c * L * w_avg * (1 + κ_avg)
        
        criticality = (beta_c * 
                      params.length_real_mm * 
                      params.width_avg_mm * 
                      (1 + params.curvature_avg))
        
        # Normalize to 0-5 scale
        return min(criticality / 1000.0, 5.0)
    
    @staticmethod
    def erosion_criticality_model(params: DefectParameters) -> float:
        """
        Erosion criticality model based on material degradation principles.
        
        Formula: C_exp^erosion = γ_c * A_real
        
        Args:
            params: DefectParameters containing erosion measurements
            
        Returns:
            Expert-driven criticality score for erosion
        """
        gamma_c = CriticalityWeights.EROSION_WEIGHTS.get(params.component_type, 0.5)
        
        # Simplified model: C_exp = γ_c * A_real
        criticality = gamma_c * params.area_real_mm2
        
        # Normalize to 0-5 scale
        return min(criticality / 500.0, 5.0)
    
    @staticmethod
    def hotspot_criticality_model(params: DefectParameters) -> float:
        """
        Hotspot criticality model based on thermal analysis principles.
        
        Formula: C_exp^hotspot = η_c * (ΔT_max)² * |∇²T|_avg
        
        Args:
            params: DefectParameters containing thermal measurements
            
        Returns:
            Expert-driven criticality score for hotspot
        """
        eta_c = CriticalityWeights.HOTSPOT_WEIGHTS.get(params.component_type, 0.5)
        
        # Model: C_exp = η_c * (ΔT_max)² * |∇²T|_avg
        criticality = (eta_c * 
                      (params.delta_t_max ** 2) * 
                      abs(params.thermal_gradient))
        
        # Normalize to 0-5 scale
        return min(criticality / 100.0, 5.0)
    
    @classmethod
    def compute_expert_criticality(cls, params: DefectParameters) -> float:
        """
        Compute expert-driven criticality score based on defect type.
        
        Args:
            params: DefectParameters containing all measurements
            
        Returns:
            Expert criticality score (C_exp)
        """
        if params.defect_type == DefectType.CRACK:
            return cls.crack_criticality_model(params)
        elif params.defect_type == DefectType.EROSION:
            return cls.erosion_criticality_model(params)
        elif params.defect_type == DefectType.HOTSPOT:
            return cls.hotspot_criticality_model(params)
        else:
            logger.warning(f"Unknown defect type: {params.defect_type}")
            return 2.5  # Default medium criticality

class EnhancedFuzzyInferenceSystem:
    """
    Block 3: Enhanced Fuzzy Inference System with complete 27-rule knowledge base
    
    This implements a Mamdani-type FIS that integrates data-driven measurements
    from Block 1 with knowledge-based estimates from Block 2.
    """
    
    def __init__(self):
        """Initialize the enhanced FIS with complete rule base"""
        self.fis_simulation = None
        self._create_enhanced_fis()
    
    def _create_enhanced_fis(self):
        """Create the complete enhanced FIS with all membership functions and rules"""
        
        # Define universes of discourse
        defect_size_universe = np.arange(0, 1001, 1)  # mm²
        location_universe = np.arange(0, 1.01, 0.01)  # normalized
        thermal_universe = np.arange(0, 25.1, 0.1)    # °C
        expert_score_universe = np.arange(0, 5.01, 0.01)  # expert score
        criticality_universe = np.arange(0, 5.01, 0.01)   # final score
        
        # Create antecedent and consequent objects
        defect_size = ctrl.Antecedent(defect_size_universe, 'DefectSize')
        location = ctrl.Antecedent(location_universe, 'Location')
        thermal = ctrl.Antecedent(thermal_universe, 'ThermalSignature')
        expert_score = ctrl.Antecedent(expert_score_universe, 'ExpertScore')
        criticality = ctrl.Consequent(criticality_universe, 'Criticality')
        
        # Enhanced membership functions based on research paper parameters
        
        # Defect Size membership functions (Table from supplementary material)
        defect_size['Small'] = fuzz.trapmf(defect_size.universe, [0, 0, 50, 100])
        defect_size['Medium'] = fuzz.trapmf(defect_size.universe, [50, 100, 400, 500])
        defect_size['Large'] = fuzz.trapmf(defect_size.universe, [400, 500, 1000, 1000])
        
        # Location membership functions
        location['Root'] = fuzz.zmf(location.universe, 0.0, 0.33)
        location['Mid-span'] = fuzz.gaussmf(location.universe, 0.5, 0.1)
        location['Tip'] = fuzz.smf(location.universe, 0.66, 1.0)
        
        # Thermal Signature membership functions (extended range)
        thermal['Low'] = fuzz.trapmf(thermal.universe, [0, 0, 2, 4])
        thermal['Medium'] = fuzz.trapmf(thermal.universe, [3, 5, 8, 10])
        thermal['High'] = fuzz.trapmf(thermal.universe, [9, 12, 25, 25])
        
        # Expert Score membership functions
        expert_score['Low'] = fuzz.trapmf(expert_score.universe, [0, 0, 1.5, 2.5])
        expert_score['Medium'] = fuzz.trapmf(expert_score.universe, [2, 2.5, 3.5, 4])
        expert_score['High'] = fuzz.trapmf(expert_score.universe, [3.5, 4, 5, 5])
        
        # Criticality output membership functions (EPRI-aligned)
        criticality['Negligible'] = fuzz.trimf(criticality.universe, [0, 1, 2])
        criticality['Low'] = fuzz.trimf(criticality.universe, [1, 2, 3])
        criticality['Medium'] = fuzz.trimf(criticality.universe, [2, 3, 4])
        criticality['High'] = fuzz.trimf(criticality.universe, [3, 4, 5])
        criticality['Severe'] = fuzz.trapmf(criticality.universe, [4, 5, 5, 5])
        
        # Enhanced 27-rule knowledge base with expert score integration
        rules = []
        
        # Rules for Large Defects
        rules.extend([
            ctrl.Rule(defect_size['Large'] & location['Root'] & thermal['High'] & expert_score['High'], criticality['Severe']),
            ctrl.Rule(defect_size['Large'] & location['Root'] & thermal['Medium'] & expert_score['High'], criticality['Severe']),
            ctrl.Rule(defect_size['Large'] & location['Root'] & thermal['Low'] & expert_score['Medium'], criticality['Severe']),
            ctrl.Rule(defect_size['Large'] & location['Mid-span'] & thermal['High'] & expert_score['High'], criticality['Severe']),
            ctrl.Rule(defect_size['Large'] & location['Mid-span'] & thermal['Medium'] & expert_score['Medium'], criticality['High']),
            ctrl.Rule(defect_size['Large'] & location['Mid-span'] & thermal['Low'] & expert_score['Medium'], criticality['High']),
            ctrl.Rule(defect_size['Large'] & location['Tip'] & thermal['High'] & expert_score['Medium'], criticality['High']),
            ctrl.Rule(defect_size['Large'] & location['Tip'] & thermal['Medium'] & expert_score['Low'], criticality['Medium']),
            ctrl.Rule(defect_size['Large'] & location['Tip'] & thermal['Low'] & expert_score['Low'], criticality['Medium']),
        ])
        
        # Rules for Medium Defects
        rules.extend([
            ctrl.Rule(defect_size['Medium'] & location['Root'] & thermal['High'] & expert_score['High'], criticality['Severe']),
            ctrl.Rule(defect_size['Medium'] & location['Root'] & thermal['Medium'] & expert_score['Medium'], criticality['High']),
            ctrl.Rule(defect_size['Medium'] & location['Root'] & thermal['Low'] & expert_score['Medium'], criticality['High']),
            ctrl.Rule(defect_size['Medium'] & location['Mid-span'] & thermal['High'] & expert_score['Medium'], criticality['High']),
            ctrl.Rule(defect_size['Medium'] & location['Mid-span'] & thermal['Medium'] & expert_score['Medium'], criticality['Medium']),
            ctrl.Rule(defect_size['Medium'] & location['Mid-span'] & thermal['Low'] & expert_score['Low'], criticality['Low']),
            ctrl.Rule(defect_size['Medium'] & location['Tip'] & thermal['High'] & expert_score['Low'], criticality['Medium']),
            ctrl.Rule(defect_size['Medium'] & location['Tip'] & thermal['Medium'] & expert_score['Low'], criticality['Low']),
            ctrl.Rule(defect_size['Medium'] & location['Tip'] & thermal['Low'] & expert_score['Low'], criticality['Low']),
        ])
        
        # Rules for Small Defects
        rules.extend([
            ctrl.Rule(defect_size['Small'] & location['Root'] & thermal['High'] & expert_score['Medium'], criticality['High']),
            ctrl.Rule(defect_size['Small'] & location['Root'] & thermal['Medium'] & expert_score['Low'], criticality['Medium']),
            ctrl.Rule(defect_size['Small'] & location['Root'] & thermal['Low'] & expert_score['Low'], criticality['Low']),
            ctrl.Rule(defect_size['Small'] & location['Mid-span'] & thermal['High'] & expert_score['Low'], criticality['Medium']),
            ctrl.Rule(defect_size['Small'] & location['Mid-span'] & thermal['Medium'] & expert_score['Low'], criticality['Low']),
            ctrl.Rule(defect_size['Small'] & location['Mid-span'] & thermal['Low'] & expert_score['Low'], criticality['Negligible']),
            ctrl.Rule(defect_size['Small'] & location['Tip'] & thermal['High'] & expert_score['Low'], criticality['Low']),
            ctrl.Rule(defect_size['Small'] & location['Tip'] & thermal['Medium'] & expert_score['Low'], criticality['Negligible']),
            ctrl.Rule(defect_size['Small'] & location['Tip'] & thermal['Low'] & expert_score['Low'], criticality['Negligible']),
        ])
        
        # Create control system and simulation
        fis_control_system = ctrl.ControlSystem(rules)
        self.fis_simulation = ctrl.ControlSystemSimulation(fis_control_system)
        
        logger.info(f"Enhanced FIS initialized with {len(rules)} rules")
    
    def compute_final_criticality(self, params: DefectParameters) -> Tuple[float, Dict]:
        """
        Compute final criticality score using the complete three-block framework.
        
        Args:
            params: DefectParameters from Block 1 (image processing)
            
        Returns:
            Tuple of (final_criticality_score, detailed_results)
        """
        # Block 2: Compute expert-driven criticality
        expert_criticality = ExpertCriticalityModels.compute_expert_criticality(params)
        
        # Block 3: Fuzzy inference integration
        self.fis_simulation.input['DefectSize'] = min(params.area_real_mm2, 1000)
        self.fis_simulation.input['Location'] = np.clip(params.location_normalized, 0, 1)
        self.fis_simulation.input['ThermalSignature'] = min(params.delta_t_max, 25)
        self.fis_simulation.input['ExpertScore'] = np.clip(expert_criticality, 0, 5)
        
        # Compute final result
        self.fis_simulation.compute()
        final_criticality = self.fis_simulation.output['Criticality']
        
        # Prepare detailed results
        results = {
            'final_criticality': final_criticality,
            'expert_criticality': expert_criticality,
            'defect_type': params.defect_type.value,
            'component_type': params.component_type.value,
            'physical_params': {
                'area_mm2': params.area_real_mm2,
                'length_mm': params.length_real_mm,
                'location_norm': params.location_normalized,
                'thermal_delta': params.delta_t_max
            },
            'epri_level': self._map_to_epri_level(final_criticality)
        }
        
        return final_criticality, results
    
    def _map_to_epri_level(self, criticality_score: float) -> str:
        """Map criticality score to EPRI damage taxonomy levels"""
        if criticality_score <= 1.5:
            return "Level 1: Monitor"
        elif criticality_score <= 2.5:
            return "Level 2: Repair at next opportunity"
        elif criticality_score <= 3.5:
            return "Level 3: Repair soon"
        elif criticality_score <= 4.5:
            return "Level 4: Urgent repair"
        else:
            return "Level 5: Immediate action required"
    
    def visualize_membership_functions(self, save_path: Optional[str] = None):
        """Visualize all membership functions"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Enhanced FIS Membership Functions', fontsize=16)
        
        # Get the antecedents and consequent
        defect_size = None
        location = None
        thermal = None
        expert_score = None
        criticality = None
        
        for antecedent in self.fis_simulation.ctrl.antecedents:
            if antecedent.label == 'DefectSize':
                defect_size = antecedent
            elif antecedent.label == 'Location':
                location = antecedent
            elif antecedent.label == 'ThermalSignature':
                thermal = antecedent
            elif antecedent.label == 'ExpertScore':
                expert_score = antecedent
        
        for consequent in self.fis_simulation.ctrl.consequents:
            if consequent.label == 'Criticality':
                criticality = consequent
        
        # Plot membership functions
        if defect_size:
            defect_size.view(ax=axes[0, 0])
            axes[0, 0].set_title('Defect Size (mm²)')
        
        if location:
            location.view(ax=axes[0, 1])
            axes[0, 1].set_title('Location (normalized)')
        
        if thermal:
            thermal.view(ax=axes[0, 2])
            axes[0, 2].set_title('Thermal Signature (°C)')
        
        if expert_score:
            expert_score.view(ax=axes[1, 0])
            axes[1, 0].set_title('Expert Score')
        
        if criticality:
            criticality.view(ax=axes[1, 1])
            axes[1, 1].set_title('Final Criticality')
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Membership functions saved to {save_path}")
        
        plt.show()

def create_sample_defect_parameters() -> List[DefectParameters]:
    """Create sample defect parameters for testing"""
    samples = [
        # Severe crack at blade root
        DefectParameters(
            area_real_mm2=450.0,
            length_real_mm=25.0,
            width_avg_mm=2.5,
            perimeter_real_mm=55.0,
            curvature_avg=0.15,
            location_normalized=0.15,
            component_type=ComponentType.BLADE_ROOT,
            delta_t_max=18.0,
            thermal_gradient=0.8,
            thermal_contrast=0.9,
            confidence=0.95,
            defect_type=DefectType.CRACK
        ),
        
        # Minor erosion at blade tip
        DefectParameters(
            area_real_mm2=75.0,
            length_real_mm=8.0,
            width_avg_mm=1.2,
            perimeter_real_mm=18.0,
            curvature_avg=0.05,
            location_normalized=0.85,
            component_type=ComponentType.BLADE_TIP,
            delta_t_max=3.5,
            thermal_gradient=0.2,
            thermal_contrast=0.3,
            confidence=0.88,
            defect_type=DefectType.EROSION
        ),
        
        # Critical hotspot on generator
        DefectParameters(
            area_real_mm2=120.0,
            length_real_mm=12.0,
            width_avg_mm=10.0,
            perimeter_real_mm=35.0,
            curvature_avg=0.02,
            location_normalized=0.0,  # Not applicable for generator
            component_type=ComponentType.GENERATOR_HOUSING,
            delta_t_max=22.0,
            thermal_gradient=1.5,
            thermal_contrast=0.95,
            confidence=0.92,
            defect_type=DefectType.HOTSPOT
        )
    ]
    
    return samples

# Example usage and testing
if __name__ == "__main__":
    print("Enhanced Wind Turbine Defect Criticality Assessment System")
    print("=" * 60)
    
    # Initialize the enhanced FIS
    enhanced_fis = EnhancedFuzzyInferenceSystem()
    
    # Create sample defect parameters
    sample_defects = create_sample_defect_parameters()
    
    # Process each sample defect
    for i, defect in enumerate(sample_defects, 1):
        print(f"\n--- Sample Defect {i}: {defect.defect_type.value.title()} ---")
        print(f"Component: {defect.component_type.value}")
        print(f"Area: {defect.area_real_mm2:.1f} mm²")
        print(f"Location: {defect.location_normalized:.2f}")
        print(f"Thermal ΔT: {defect.delta_t_max:.1f}°C")
        
        # Compute criticality
        final_score, results = enhanced_fis.compute_final_criticality(defect)
        
        print(f"Expert Score: {results['expert_criticality']:.2f}")
        print(f"Final Criticality: {final_score:.2f}")
        print(f"EPRI Level: {results['epri_level']}")
    
    # Visualize membership functions
    print("\nGenerating membership function visualization...")
    enhanced_fis.visualize_membership_functions("enhanced_fis_membership_functions.png")
    
    print("\nEnhanced FIS demonstration completed successfully!")