"""
Validation Framework for Enhanced Fuzzy Inference System
========================================================

This module implements the validation protocol described in the research paper,
including inter-rater reliability analysis, sensitivity analysis, and 
ground-truth validation procedures.

Author: Based on research by Radiuk et al. (2025)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from enhanced_fis_core import (
    EnhancedFuzzyInferenceSystem, 
    DefectParameters, 
    DefectType, 
    ComponentType
)

logger = logging.getLogger(__name__)

@dataclass
class ExpertRating:
    """Expert rating for a defect"""
    expert_id: str
    defect_id: str
    criticality_score: float
    confidence: float
    notes: Optional[str] = None

@dataclass
class ValidationResults:
    """Results from validation analysis"""
    fleiss_kappa: float
    inter_rater_agreement: str
    mean_absolute_error: float
    pearson_correlation: float
    rmse: float
    expert_ratings: List[ExpertRating]
    fis_predictions: List[float]

class InterRaterReliabilityAnalyzer:
    """
    Implements inter-rater reliability analysis using Fleiss' Kappa
    as described in the research paper.
    """
    
    @staticmethod
    def compute_fleiss_kappa(ratings_matrix: np.ndarray) -> Tuple[float, str]:
        """
        Compute Fleiss' Kappa for inter-rater reliability.
        
        Args:
            ratings_matrix: Matrix where rows are items and columns are raters
            
        Returns:
            Tuple of (kappa_value, interpretation)
        """
        n_items, n_raters = ratings_matrix.shape
        n_categories = int(np.max(ratings_matrix)) + 1
        
        # Create agreement matrix
        agreement_matrix = np.zeros((n_items, n_categories))
        
        for i in range(n_items):
            for j in range(n_categories):
                agreement_matrix[i, j] = np.sum(ratings_matrix[i, :] == j)
        
        # Calculate P_i (proportion of agreement for each item)
        P_i = np.zeros(n_items)
        for i in range(n_items):
            P_i[i] = (np.sum(agreement_matrix[i, :] ** 2) - n_raters) / (n_raters * (n_raters - 1))
        
        # Calculate P_bar (mean proportion of agreement)
        P_bar = np.mean(P_i)
        
        # Calculate P_e (expected proportion of agreement by chance)
        p_j = np.sum(agreement_matrix, axis=0) / (n_items * n_raters)
        P_e = np.sum(p_j ** 2)
        
        # Calculate Fleiss' Kappa
        if P_e == 1.0:
            kappa = 1.0
        else:
            kappa = (P_bar - P_e) / (1 - P_e)
        
        # Interpret kappa value (Landis and Koch interpretation)
        if kappa < 0:
            interpretation = "Poor agreement"
        elif kappa < 0.20:
            interpretation = "Slight agreement"
        elif kappa < 0.40:
            interpretation = "Fair agreement"
        elif kappa < 0.60:
            interpretation = "Moderate agreement"
        elif kappa < 0.80:
            interpretation = "Substantial agreement"
        else:
            interpretation = "Almost perfect agreement"
        
        return kappa, interpretation

class SensitivityAnalyzer:
    """
    Implements global sensitivity analysis for the FIS parameters
    as described in the supplementary material.
    """
    
    def __init__(self, fis: EnhancedFuzzyInferenceSystem):
        self.fis = fis
        self.base_params = None
    
    def create_parameter_variations(self, base_params: DefectParameters, 
                                  variation_percent: float = 0.1) -> List[DefectParameters]:
        """
        Create parameter variations for sensitivity analysis.
        
        Args:
            base_params: Base defect parameters
            variation_percent: Percentage variation for each parameter
            
        Returns:
            List of varied parameter sets
        """
        variations = []
        
        # Vary each parameter individually
        param_names = [
            'area_real_mm2', 'length_real_mm', 'width_avg_mm', 
            'location_normalized', 'delta_t_max', 'thermal_gradient'
        ]
        
        for param_name in param_names:
            # Create positive and negative variations
            for multiplier in [1 - variation_percent, 1 + variation_percent]:
                varied_params = DefectParameters(
                    area_real_mm2=base_params.area_real_mm2,
                    length_real_mm=base_params.length_real_mm,
                    width_avg_mm=base_params.width_avg_mm,
                    perimeter_real_mm=base_params.perimeter_real_mm,
                    curvature_avg=base_params.curvature_avg,
                    location_normalized=base_params.location_normalized,
                    component_type=base_params.component_type,
                    delta_t_max=base_params.delta_t_max,
                    thermal_gradient=base_params.thermal_gradient,
                    thermal_contrast=base_params.thermal_contrast,
                    confidence=base_params.confidence,
                    defect_type=base_params.defect_type
                )
                
                # Apply variation to specific parameter
                if param_name == 'area_real_mm2':
                    varied_params.area_real_mm2 *= multiplier
                elif param_name == 'length_real_mm':
                    varied_params.length_real_mm *= multiplier
                elif param_name == 'width_avg_mm':
                    varied_params.width_avg_mm *= multiplier
                elif param_name == 'location_normalized':
                    varied_params.location_normalized = np.clip(
                        varied_params.location_normalized * multiplier, 0, 1)
                elif param_name == 'delta_t_max':
                    varied_params.delta_t_max *= multiplier
                elif param_name == 'thermal_gradient':
                    varied_params.thermal_gradient *= multiplier
                
                variations.append((param_name, multiplier, varied_params))
        
        return variations
    
    def perform_sensitivity_analysis(self, base_params: DefectParameters) -> Dict:
        """
        Perform global sensitivity analysis on FIS parameters.
        
        Args:
            base_params: Base defect parameters for analysis
            
        Returns:
            Dictionary containing sensitivity analysis results
        """
        # Get base prediction
        base_score, _ = self.fis.compute_final_criticality(base_params)
        
        # Create parameter variations
        variations = self.create_parameter_variations(base_params)
        
        # Compute sensitivity for each parameter
        sensitivity_results = {}
        
        for param_name, multiplier, varied_params in variations:
            varied_score, _ = self.fis.compute_final_criticality(varied_params)
            
            # Calculate sensitivity (relative change in output / relative change in input)
            input_change = (multiplier - 1.0) * 100  # Percentage change in input
            output_change = ((varied_score - base_score) / base_score) * 100  # Percentage change in output
            
            if input_change != 0:
                sensitivity = abs(output_change / input_change)
            else:
                sensitivity = 0
            
            if param_name not in sensitivity_results:
                sensitivity_results[param_name] = []
            
            sensitivity_results[param_name].append({
                'input_change': input_change,
                'output_change': output_change,
                'sensitivity': sensitivity,
                'varied_score': varied_score
            })
        
        # Calculate average sensitivity for each parameter
        avg_sensitivity = {}
        for param_name, results in sensitivity_results.items():
            avg_sensitivity[param_name] = np.mean([r['sensitivity'] for r in results])
        
        return {
            'base_score': base_score,
            'parameter_sensitivity': sensitivity_results,
            'average_sensitivity': avg_sensitivity,
            'most_sensitive_parameter': max(avg_sensitivity.keys(), key=lambda k: avg_sensitivity[k])
        }

class ValidationFramework:
    """
    Complete validation framework implementing the protocol from the research paper.
    """
    
    def __init__(self, fis: EnhancedFuzzyInferenceSystem):
        self.fis = fis
        self.reliability_analyzer = InterRaterReliabilityAnalyzer()
        self.sensitivity_analyzer = SensitivityAnalyzer(fis)
    
    def create_synthetic_expert_panel(self, defects: List[DefectParameters], 
                                    n_experts: int = 3) -> List[ExpertRating]:
        """
        Create synthetic expert ratings for validation (simulates the expert panel).
        
        Args:
            defects: List of defects to rate
            n_experts: Number of expert raters
            
        Returns:
            List of expert ratings
        """
        expert_ratings = []
        
        for defect_id, defect in enumerate(defects):
            # Get FIS prediction as baseline
            fis_score, _ = self.fis.compute_final_criticality(defect)
            
            # Generate expert ratings with some variation around FIS score
            for expert_id in range(n_experts):
                # Add expert-specific bias and noise
                expert_bias = np.random.normal(0, 0.2)  # Small systematic bias
                measurement_noise = np.random.normal(0, 0.15)  # Measurement uncertainty
                
                expert_score = fis_score + expert_bias + measurement_noise
                expert_score = np.clip(expert_score, 0, 5)  # Ensure valid range
                
                # Round to nearest 0.5 (typical expert rating precision)
                expert_score = round(expert_score * 2) / 2
                
                rating = ExpertRating(
                    expert_id=f"Expert_{expert_id + 1}",
                    defect_id=f"Defect_{defect_id + 1}",
                    criticality_score=expert_score,
                    confidence=np.random.uniform(0.7, 0.95),
                    notes=f"Assessment by {expert_id + 1}"
                )
                
                expert_ratings.append(rating)
        
        return expert_ratings
    
    def validate_fis_performance(self, defects: List[DefectParameters], 
                                expert_ratings: List[ExpertRating]) -> ValidationResults:
        """
        Validate FIS performance against expert ratings.
        
        Args:
            defects: List of defects
            expert_ratings: Expert ratings for validation
            
        Returns:
            ValidationResults containing all validation metrics
        """
        # Organize expert ratings by defect
        ratings_by_defect = {}
        for rating in expert_ratings:
            if rating.defect_id not in ratings_by_defect:
                ratings_by_defect[rating.defect_id] = []
            ratings_by_defect[rating.defect_id].append(rating.criticality_score)
        
        # Create ratings matrix for Fleiss' Kappa
        n_defects = len(defects)
        n_experts = len(set(r.expert_id for r in expert_ratings))
        ratings_matrix = np.zeros((n_defects, n_experts))
        
        for i, defect_id in enumerate(sorted(ratings_by_defect.keys())):
            ratings_matrix[i, :] = ratings_by_defect[defect_id]
        
        # Convert to integer ratings for Fleiss' Kappa (multiply by 2 to handle 0.5 increments)
        ratings_matrix_int = (ratings_matrix * 2).astype(int)
        
        # Compute inter-rater reliability
        fleiss_kappa, agreement_interpretation = self.reliability_analyzer.compute_fleiss_kappa(ratings_matrix_int)
        
        # Get FIS predictions
        fis_predictions = []
        for defect in defects:
            score, _ = self.fis.compute_final_criticality(defect)
            fis_predictions.append(score)
        
        # Compute ground truth as median of expert ratings
        ground_truth = []
        for defect_id in sorted(ratings_by_defect.keys()):
            ground_truth.append(np.median(ratings_by_defect[defect_id]))
        
        # Compute validation metrics
        mae = mean_absolute_error(ground_truth, fis_predictions)
        rmse = np.sqrt(mean_squared_error(ground_truth, fis_predictions))
        correlation, _ = stats.pearsonr(ground_truth, fis_predictions)
        
        return ValidationResults(
            fleiss_kappa=fleiss_kappa,
            inter_rater_agreement=agreement_interpretation,
            mean_absolute_error=mae,
            pearson_correlation=correlation,
            rmse=rmse,
            expert_ratings=expert_ratings,
            fis_predictions=fis_predictions
        )
    
    def generate_validation_report(self, results: ValidationResults, 
                                 save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            results: ValidationResults from validation analysis
            save_path: Optional path to save the report
            
        Returns:
            Formatted validation report as string
        """
        report = []
        report.append("=" * 80)
        report.append("ENHANCED FIS VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Inter-rater reliability
        report.append("INTER-RATER RELIABILITY ANALYSIS")
        report.append("-" * 40)
        report.append(f"Fleiss' Kappa: {results.fleiss_kappa:.3f}")
        report.append(f"Agreement Level: {results.inter_rater_agreement}")
        report.append("")
        
        # FIS Performance Metrics
        report.append("FIS PERFORMANCE METRICS")
        report.append("-" * 40)
        report.append(f"Mean Absolute Error: {results.mean_absolute_error:.3f}")
        report.append(f"Root Mean Square Error: {results.rmse:.3f}")
        report.append(f"Pearson Correlation: {results.pearson_correlation:.3f}")
        report.append("")
        
        # Performance interpretation
        report.append("PERFORMANCE INTERPRETATION")
        report.append("-" * 40)
        
        if results.mean_absolute_error <= 0.2:
            mae_interpretation = "Excellent agreement with experts"
        elif results.mean_absolute_error <= 0.5:
            mae_interpretation = "Good agreement with experts"
        elif results.mean_absolute_error <= 1.0:
            mae_interpretation = "Moderate agreement with experts"
        else:
            mae_interpretation = "Poor agreement with experts"
        
        if results.pearson_correlation >= 0.9:
            corr_interpretation = "Very strong correlation"
        elif results.pearson_correlation >= 0.7:
            corr_interpretation = "Strong correlation"
        elif results.pearson_correlation >= 0.5:
            corr_interpretation = "Moderate correlation"
        else:
            corr_interpretation = "Weak correlation"
        
        report.append(f"MAE Assessment: {mae_interpretation}")
        report.append(f"Correlation Assessment: {corr_interpretation}")
        report.append("")
        
        # Summary
        report.append("VALIDATION SUMMARY")
        report.append("-" * 40)
        
        if (results.fleiss_kappa >= 0.8 and 
            results.mean_absolute_error <= 0.2 and 
            results.pearson_correlation >= 0.9):
            overall_assessment = "EXCELLENT - System ready for deployment"
        elif (results.fleiss_kappa >= 0.6 and 
              results.mean_absolute_error <= 0.5 and 
              results.pearson_correlation >= 0.7):
            overall_assessment = "GOOD - System suitable for operational use"
        else:
            overall_assessment = "NEEDS IMPROVEMENT - Further calibration required"
        
        report.append(f"Overall Assessment: {overall_assessment}")
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Validation report saved to {save_path}")
        
        return report_text
    
    def plot_validation_results(self, results: ValidationResults, 
                              save_path: Optional[str] = None):
        """
        Create comprehensive validation plots.
        
        Args:
            results: ValidationResults from validation
            save_path: Optional path to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced FIS Validation Results', fontsize=16)
        
        # Organize data for plotting
        ratings_by_defect = {}
        for rating in results.expert_ratings:
            if rating.defect_id not in ratings_by_defect:
                ratings_by_defect[rating.defect_id] = []
            ratings_by_defect[rating.defect_id].append(rating.criticality_score)
        
        ground_truth = [np.median(ratings_by_defect[defect_id]) 
                       for defect_id in sorted(ratings_by_defect.keys())]
        
        # 1. Correlation plot
        axes[0, 0].scatter(ground_truth, results.fis_predictions, alpha=0.7)
        axes[0, 0].plot([0, 5], [0, 5], 'r--', label='Perfect correlation')
        axes[0, 0].set_xlabel('Expert Ground Truth')
        axes[0, 0].set_ylabel('FIS Predictions')
        axes[0, 0].set_title(f'Correlation (r = {results.pearson_correlation:.3f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = np.array(results.fis_predictions) - np.array(ground_truth)
        axes[0, 1].scatter(ground_truth, residuals, alpha=0.7)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Expert Ground Truth')
        axes[0, 1].set_ylabel('Residuals (FIS - Expert)')
        axes[0, 1].set_title(f'Residuals (MAE = {results.mean_absolute_error:.3f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribution comparison
        axes[1, 0].hist(ground_truth, alpha=0.7, label='Expert Ratings', bins=10)
        axes[1, 0].hist(results.fis_predictions, alpha=0.7, label='FIS Predictions', bins=10)
        axes[1, 0].set_xlabel('Criticality Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Score Distributions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Agreement matrix
        agreement_data = []
        for i, (gt, pred) in enumerate(zip(ground_truth, results.fis_predictions)):
            agreement_data.append([i, gt, pred, abs(gt - pred)])
        
        agreement_df = pd.DataFrame(agreement_data, 
                                  columns=['Defect', 'Expert', 'FIS', 'Error'])
        
        im = axes[1, 1].scatter(agreement_df['Expert'], agreement_df['FIS'], 
                               c=agreement_df['Error'], cmap='RdYlBu_r', 
                               s=60, alpha=0.7)
        axes[1, 1].plot([0, 5], [0, 5], 'r--', alpha=0.5)
        axes[1, 1].set_xlabel('Expert Rating')
        axes[1, 1].set_ylabel('FIS Prediction')
        axes[1, 1].set_title('Agreement Analysis')
        plt.colorbar(im, ax=axes[1, 1], label='Absolute Error')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Validation plots saved to {save_path}")
        
        plt.show()

# Example usage and testing
if __name__ == "__main__":
    from enhanced_fis_core import create_sample_defect_parameters
    
    print("Enhanced FIS Validation Framework")
    print("=" * 50)
    
    # Initialize FIS and validation framework
    fis = EnhancedFuzzyInferenceSystem()
    validator = ValidationFramework(fis)
    
    # Create sample defects for validation
    sample_defects = create_sample_defect_parameters()
    
    # Add more diverse samples for better validation
    additional_samples = [
        # Medium crack at mid-span
        DefectParameters(
            area_real_mm2=180.0, length_real_mm=15.0, width_avg_mm=1.8,
            perimeter_real_mm=32.0, curvature_avg=0.08, location_normalized=0.5,
            component_type=ComponentType.BLADE_MID_SPAN, delta_t_max=8.5,
            thermal_gradient=0.4, thermal_contrast=0.6, confidence=0.89,
            defect_type=DefectType.CRACK
        ),
        # Large erosion at nacelle
        DefectParameters(
            area_real_mm2=320.0, length_real_mm=20.0, width_avg_mm=16.0,
            perimeter_real_mm=45.0, curvature_avg=0.03, location_normalized=0.0,
            component_type=ComponentType.NACELLE_HOUSING, delta_t_max=6.2,
            thermal_gradient=0.3, thermal_contrast=0.4, confidence=0.91,
            defect_type=DefectType.EROSION
        )
    ]
    
    all_defects = sample_defects + additional_samples
    
    # Generate synthetic expert ratings
    print("Generating synthetic expert panel ratings...")
    expert_ratings = validator.create_synthetic_expert_panel(all_defects, n_experts=3)
    
    # Perform validation
    print("Performing FIS validation...")
    validation_results = validator.validate_fis_performance(all_defects, expert_ratings)
    
    # Generate and display validation report
    print("\nValidation Report:")
    report = validator.generate_validation_report(validation_results, "fis_validation_report.txt")
    print(report)
    
    # Create validation plots
    print("Generating validation plots...")
    validator.plot_validation_results(validation_results, "fis_validation_plots.png")
    
    # Perform sensitivity analysis on first defect
    print("\nPerforming sensitivity analysis...")
    sensitivity_results = validator.sensitivity_analyzer.perform_sensitivity_analysis(all_defects[0])
    
    print(f"Base criticality score: {sensitivity_results['base_score']:.3f}")
    print(f"Most sensitive parameter: {sensitivity_results['most_sensitive_parameter']}")
    print("\nParameter sensitivities:")
    for param, sensitivity in sensitivity_results['average_sensitivity'].items():
        print(f"  {param}: {sensitivity:.3f}")
    
    print("\nValidation framework demonstration completed successfully!")