"""
Comprehensive Test Suite for Enhanced VISION_Fuzzy System
=========================================================

This module provides comprehensive testing for all components of the enhanced
fuzzy inference system, including validation of the research paper implementations.

Author: Based on research by Radiuk et al. (2025)
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import json
from typing import List, Dict
import matplotlib.pyplot as plt
import warnings

# Import the modules to test
from enhanced_fis_core import (
    EnhancedFuzzyInferenceSystem,
    ExpertCriticalityModels,
    DefectParameters,
    DefectType,
    ComponentType,
    CriticalityWeights,
    create_sample_defect_parameters
)

from validation_framework import (
    ValidationFramework,
    InterRaterReliabilityAnalyzer,
    SensitivityAnalyzer,
    ValidationResults
)

from integration_bridge import IntegrationBridge, IntegrationResult

# Suppress matplotlib warnings in tests
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

class TestExpertCriticalityModels(unittest.TestCase):
    """Test suite for expert-driven criticality models (Block 2)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.expert_models = ExpertCriticalityModels()
        
        # Create test defect parameters
        self.crack_params = DefectParameters(
            area_real_mm2=200.0,
            length_real_mm=15.0,
            width_avg_mm=2.0,
            perimeter_real_mm=35.0,
            curvature_avg=0.1,
            location_normalized=0.2,
            component_type=ComponentType.BLADE_ROOT,
            delta_t_max=10.0,
            thermal_gradient=0.5,
            thermal_contrast=0.6,
            confidence=0.9,
            defect_type=DefectType.CRACK
        )
        
        self.erosion_params = DefectParameters(
            area_real_mm2=150.0,
            length_real_mm=12.0,
            width_avg_mm=8.0,
            perimeter_real_mm=28.0,
            curvature_avg=0.05,
            location_normalized=0.7,
            component_type=ComponentType.BLADE_TIP,
            delta_t_max=5.0,
            thermal_gradient=0.3,
            thermal_contrast=0.4,
            confidence=0.85,
            defect_type=DefectType.EROSION
        )
        
        self.hotspot_params = DefectParameters(
            area_real_mm2=100.0,
            length_real_mm=10.0,
            width_avg_mm=10.0,
            perimeter_real_mm=30.0,
            curvature_avg=0.02,
            location_normalized=0.0,
            component_type=ComponentType.GENERATOR_HOUSING,
            delta_t_max=20.0,
            thermal_gradient=1.2,
            thermal_contrast=0.9,
            confidence=0.95,
            defect_type=DefectType.HOTSPOT
        )
    
    def test_crack_criticality_model(self):
        """Test crack criticality model implementation"""
        criticality = self.expert_models.crack_criticality_model(self.crack_params)
        
        # Verify output is in valid range
        self.assertGreaterEqual(criticality, 0.0)
        self.assertLessEqual(criticality, 5.0)
        
        # Test that blade root has higher criticality than tip for same defect
        tip_params = self.crack_params
        tip_params.component_type = ComponentType.BLADE_TIP
        tip_criticality = self.expert_models.crack_criticality_model(tip_params)
        
        self.assertGreater(criticality, tip_criticality)
    
    def test_erosion_criticality_model(self):
        """Test erosion criticality model implementation"""
        criticality = self.expert_models.erosion_criticality_model(self.erosion_params)
        
        # Verify output is in valid range
        self.assertGreaterEqual(criticality, 0.0)
        self.assertLessEqual(criticality, 5.0)
        
        # Test that larger area results in higher criticality
        large_area_params = self.erosion_params
        large_area_params.area_real_mm2 = 300.0
        large_criticality = self.expert_models.erosion_criticality_model(large_area_params)
        
        self.assertGreater(large_criticality, criticality)
    
    def test_hotspot_criticality_model(self):
        """Test hotspot criticality model implementation"""
        criticality = self.expert_models.hotspot_criticality_model(self.hotspot_params)
        
        # Verify output is in valid range
        self.assertGreaterEqual(criticality, 0.0)
        self.assertLessEqual(criticality, 5.0)
        
        # Test that higher temperature results in higher criticality
        high_temp_params = self.hotspot_params
        high_temp_params.delta_t_max = 30.0
        high_temp_criticality = self.expert_models.hotspot_criticality_model(high_temp_params)
        
        self.assertGreater(high_temp_criticality, criticality)
    
    def test_component_weighting_coefficients(self):
        """Test that component weighting coefficients are properly applied"""
        # Test crack weights
        self.assertEqual(CriticalityWeights.CRACK_WEIGHTS[ComponentType.BLADE_ROOT], 1.0)
        self.assertEqual(CriticalityWeights.CRACK_WEIGHTS[ComponentType.BLADE_TIP], 0.4)
        
        # Test erosion weights
        self.assertEqual(CriticalityWeights.EROSION_WEIGHTS[ComponentType.BLADE_ROOT], 0.9)
        self.assertEqual(CriticalityWeights.EROSION_WEIGHTS[ComponentType.BLADE_TIP], 0.3)
        
        # Test hotspot weights
        self.assertEqual(CriticalityWeights.HOTSPOT_WEIGHTS[ComponentType.GENERATOR_HOUSING], 1.0)
        self.assertEqual(CriticalityWeights.HOTSPOT_WEIGHTS[ComponentType.SPAR_CAP], 0.8)

class TestEnhancedFuzzyInferenceSystem(unittest.TestCase):
    """Test suite for enhanced fuzzy inference system (Block 3)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.fis = EnhancedFuzzyInferenceSystem()
        self.sample_defects = create_sample_defect_parameters()
    
    def test_fis_initialization(self):
        """Test that FIS initializes correctly"""
        self.assertIsNotNone(self.fis.fis_simulation)
        
        # Test that all required inputs and outputs are present
        input_labels = [ant.label for ant in self.fis.fis_simulation.ctrl.antecedents]
        output_labels = [cons.label for cons in self.fis.fis_simulation.ctrl.consequents]
        
        expected_inputs = ['DefectSize', 'Location', 'ThermalSignature', 'ExpertScore']
        expected_outputs = ['Criticality']
        
        for expected_input in expected_inputs:
            self.assertIn(expected_input, input_labels)
        
        for expected_output in expected_outputs:
            self.assertIn(expected_output, output_labels)
    
    def test_criticality_computation(self):
        """Test criticality computation for sample defects"""
        for i, defect in enumerate(self.sample_defects):
            with self.subTest(defect_id=i):
                criticality, results = self.fis.compute_final_criticality(defect)
                
                # Verify output is in valid range
                self.assertGreaterEqual(criticality, 0.0)
                self.assertLessEqual(criticality, 5.0)
                
                # Verify results structure
                self.assertIn('final_criticality', results)
                self.assertIn('expert_criticality', results)
                self.assertIn('epri_level', results)
                self.assertIn('defect_type', results)
                self.assertIn('component_type', results)
    
    def test_epri_level_mapping(self):
        """Test EPRI level mapping functionality"""
        test_scores = [0.5, 1.5, 2.5, 3.5, 4.5]
        expected_levels = [
            "Level 1: Monitor",
            "Level 2: Repair at next opportunity", 
            "Level 3: Repair soon",
            "Level 4: Urgent repair",
            "Level 5: Immediate action required"
        ]
        
        for score, expected_level in zip(test_scores, expected_levels):
            mapped_level = self.fis._map_to_epri_level(score)
            self.assertEqual(mapped_level, expected_level)
    
    def test_rule_consistency(self):
        """Test that fuzzy rules produce consistent results"""
        # Test that larger defects at critical locations produce higher scores
        small_defect = DefectParameters(
            area_real_mm2=50.0, length_real_mm=5.0, width_avg_mm=1.0,
            perimeter_real_mm=15.0, curvature_avg=0.05, location_normalized=0.8,
            component_type=ComponentType.BLADE_TIP, delta_t_max=2.0,
            thermal_gradient=0.1, thermal_contrast=0.2, confidence=0.8,
            defect_type=DefectType.CRACK
        )
        
        large_defect = DefectParameters(
            area_real_mm2=500.0, length_real_mm=25.0, width_avg_mm=3.0,
            perimeter_real_mm=60.0, curvature_avg=0.15, location_normalized=0.1,
            component_type=ComponentType.BLADE_ROOT, delta_t_max=18.0,
            thermal_gradient=0.8, thermal_contrast=0.9, confidence=0.95,
            defect_type=DefectType.CRACK
        )
        
        small_criticality, _ = self.fis.compute_final_criticality(small_defect)
        large_criticality, _ = self.fis.compute_final_criticality(large_defect)
        
        self.assertGreater(large_criticality, small_criticality)

class TestValidationFramework(unittest.TestCase):
    """Test suite for validation framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.fis = EnhancedFuzzyInferenceSystem()
        self.validator = ValidationFramework(self.fis)
        self.sample_defects = create_sample_defect_parameters()
    
    def test_fleiss_kappa_calculation(self):
        """Test Fleiss' Kappa calculation"""
        # Create test ratings matrix (3 raters, 5 items, ratings 0-4)
        test_ratings = np.array([
            [2, 2, 3],  # Item 1
            [1, 2, 1],  # Item 2
            [4, 4, 4],  # Item 3
            [0, 1, 0],  # Item 4
            [3, 3, 2]   # Item 5
        ])
        
        kappa, interpretation = self.validator.reliability_analyzer.compute_fleiss_kappa(test_ratings)
        
        # Verify kappa is in valid range
        self.assertGreaterEqual(kappa, -1.0)
        self.assertLessEqual(kappa, 1.0)
        
        # Verify interpretation is provided
        self.assertIsInstance(interpretation, str)
        self.assertGreater(len(interpretation), 0)
    
    def test_synthetic_expert_panel(self):
        """Test synthetic expert panel generation"""
        expert_ratings = self.validator.create_synthetic_expert_panel(self.sample_defects, n_experts=3)
        
        # Verify correct number of ratings
        expected_ratings = len(self.sample_defects) * 3
        self.assertEqual(len(expert_ratings), expected_ratings)
        
        # Verify rating structure
        for rating in expert_ratings:
            self.assertGreaterEqual(rating.criticality_score, 0.0)
            self.assertLessEqual(rating.criticality_score, 5.0)
            self.assertGreaterEqual(rating.confidence, 0.0)
            self.assertLessEqual(rating.confidence, 1.0)
    
    def test_fis_validation(self):
        """Test FIS validation against expert ratings"""
        expert_ratings = self.validator.create_synthetic_expert_panel(self.sample_defects, n_experts=3)
        validation_results = self.validator.validate_fis_performance(self.sample_defects, expert_ratings)
        
        # Verify validation results structure
        self.assertIsInstance(validation_results, ValidationResults)
        self.assertGreaterEqual(validation_results.fleiss_kappa, -1.0)
        self.assertLessEqual(validation_results.fleiss_kappa, 1.0)
        self.assertGreaterEqual(validation_results.mean_absolute_error, 0.0)
        self.assertGreaterEqual(validation_results.pearson_correlation, -1.0)
        self.assertLessEqual(validation_results.pearson_correlation, 1.0)
    
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis functionality"""
        sensitivity_results = self.validator.sensitivity_analyzer.perform_sensitivity_analysis(
            self.sample_defects[0]
        )
        
        # Verify results structure
        self.assertIn('base_score', sensitivity_results)
        self.assertIn('parameter_sensitivity', sensitivity_results)
        self.assertIn('average_sensitivity', sensitivity_results)
        self.assertIn('most_sensitive_parameter', sensitivity_results)
        
        # Verify base score is valid
        self.assertGreaterEqual(sensitivity_results['base_score'], 0.0)
        self.assertLessEqual(sensitivity_results['base_score'], 5.0)

class TestIntegrationBridge(unittest.TestCase):
    """Test suite for integration bridge"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bridge = IntegrationBridge()
        
        # Create test detection data
        self.test_detection_data = [
            {
                'defect_id': 0,
                'defect_type': 'crack',
                'component_type': 'blade_root',
                'area_mm2': 250.0,
                'length_mm': 18.0,
                'width_mm': 2.5,
                'location_normalized': 0.15,
                'thermal_delta_t': 15.0,
                'thermal_gradient': 0.7,
                'thermal_contrast': 0.8,
                'confidence': 0.92
            },
            {
                'defect_id': 1,
                'defect_type': 'erosion',
                'component_type': 'blade_tip',
                'area_mm2': 80.0,
                'length_mm': 8.0,
                'width_mm': 1.5,
                'location_normalized': 0.85,
                'thermal_delta_t': 4.0,
                'thermal_gradient': 0.2,
                'thermal_contrast': 0.3,
                'confidence': 0.87
            }
        ]
    
    def test_detection_processing(self):
        """Test processing of detection results"""
        results = self.bridge.process_detection_results(self.test_detection_data)
        
        # Verify correct number of results
        self.assertEqual(len(results), len(self.test_detection_data))
        
        # Verify result structure
        for result in results:
            self.assertIsInstance(result, IntegrationResult)
            self.assertGreaterEqual(result.final_criticality, 0.0)
            self.assertLessEqual(result.final_criticality, 5.0)
            self.assertGreaterEqual(result.expert_criticality, 0.0)
            self.assertLessEqual(result.expert_criticality, 5.0)
            self.assertIn('Level', result.epri_level)
    
    def test_defect_type_mapping(self):
        """Test defect type string to enum mapping"""
        test_cases = [
            ('crack', DefectType.CRACK),
            ('erosion', DefectType.EROSION),
            ('hotspot', DefectType.HOTSPOT),
            ('unknown', DefectType.CRACK)  # Default case
        ]
        
        for input_str, expected_enum in test_cases:
            result = self.bridge._map_defect_type(input_str)
            self.assertEqual(result, expected_enum)
    
    def test_component_type_mapping(self):
        """Test component type string to enum mapping"""
        test_cases = [
            ('blade_root', ComponentType.BLADE_ROOT),
            ('blade_mid_span', ComponentType.BLADE_MID_SPAN),
            ('blade_tip', ComponentType.BLADE_TIP),
            ('unknown', ComponentType.BLADE_MID_SPAN)  # Default case
        ]
        
        for input_str, expected_enum in test_cases:
            result = self.bridge._map_component_type(input_str)
            self.assertEqual(result, expected_enum)
    
    def test_results_saving(self):
        """Test saving results in different formats"""
        results = self.bridge.process_detection_results(self.test_detection_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test JSON saving
            json_path = os.path.join(temp_dir, 'results.json')
            self.bridge.config['output']['format'] = 'json'
            self.bridge.save_results(results, json_path)
            
            self.assertTrue(os.path.exists(json_path))
            
            # Verify JSON content
            with open(json_path, 'r') as f:
                saved_data = json.load(f)
            
            self.assertEqual(len(saved_data), len(results))
            
            # Test CSV saving
            csv_path = os.path.join(temp_dir, 'results.csv')
            self.bridge.config['output']['format'] = 'csv'
            self.bridge.save_results(results, csv_path)
            
            self.assertTrue(os.path.exists(csv_path))
            
            # Verify CSV content
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), len(results))
    
    def test_summary_report_generation(self):
        """Test summary report generation"""
        results = self.bridge.process_detection_results(self.test_detection_data)
        summary = self.bridge.generate_summary_report(results)
        
        # Verify summary structure
        expected_keys = [
            'total_defects', 'defect_type_distribution', 'component_type_distribution',
            'epri_level_distribution', 'criticality_statistics', 'most_critical_defect',
            'performance'
        ]
        
        for key in expected_keys:
            self.assertIn(key, summary)
        
        # Verify statistics
        self.assertEqual(summary['total_defects'], len(results))
        self.assertIn('mean', summary['criticality_statistics'])
        self.assertIn('defect_id', summary['most_critical_defect'])

class TestResearchPaperCompliance(unittest.TestCase):
    """Test suite to verify compliance with research paper specifications"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.fis = EnhancedFuzzyInferenceSystem()
        self.expert_models = ExpertCriticalityModels()
    
    def test_27_rule_knowledge_base(self):
        """Test that the FIS implements exactly 27 rules as specified in the paper"""
        # Count the number of rules in the FIS
        rule_count = len(self.fis.fis_simulation.ctrl.rules)
        self.assertEqual(rule_count, 27, "FIS should implement exactly 27 rules as per research paper")
    
    def test_membership_function_parameters(self):
        """Test that membership functions match the research paper specifications"""
        # Test defect size membership functions
        defect_size_antecedent = None
        for ant in self.fis.fis_simulation.ctrl.antecedents:
            if ant.label == 'DefectSize':
                defect_size_antecedent = ant
                break
        
        self.assertIsNotNone(defect_size_antecedent)
        
        # Verify the three linguistic terms exist
        expected_terms = ['Small', 'Medium', 'Large']
        actual_terms = list(defect_size_antecedent.terms.keys())
        
        for term in expected_terms:
            self.assertIn(term, actual_terms)
    
    def test_epri_alignment(self):
        """Test that output levels align with EPRI taxonomy"""
        criticality_consequent = None
        for cons in self.fis.fis_simulation.ctrl.consequents:
            if cons.label == 'Criticality':
                criticality_consequent = cons
                break
        
        self.assertIsNotNone(criticality_consequent)
        
        # Verify the five EPRI-aligned terms exist
        expected_terms = ['Negligible', 'Low', 'Medium', 'High', 'Severe']
        actual_terms = list(criticality_consequent.terms.keys())
        
        for term in expected_terms:
            self.assertIn(term, actual_terms)
    
    def test_component_weighting_compliance(self):
        """Test that component weighting coefficients match paper specifications"""
        # Test specific values mentioned in the paper
        self.assertEqual(
            CriticalityWeights.CRACK_WEIGHTS[ComponentType.BLADE_ROOT], 
            1.0, 
            "Blade root should have maximum crack weight of 1.0"
        )
        
        self.assertEqual(
            CriticalityWeights.HOTSPOT_WEIGHTS[ComponentType.GENERATOR_HOUSING], 
            1.0, 
            "Generator housing should have maximum hotspot weight of 1.0"
        )
    
    def test_three_block_architecture(self):
        """Test that the three-block architecture is properly implemented"""
        # Create test defect
        test_defect = DefectParameters(
            area_real_mm2=200.0, length_real_mm=15.0, width_avg_mm=2.0,
            perimeter_real_mm=35.0, curvature_avg=0.1, location_normalized=0.2,
            component_type=ComponentType.BLADE_ROOT, delta_t_max=10.0,
            thermal_gradient=0.5, thermal_contrast=0.6, confidence=0.9,
            defect_type=DefectType.CRACK
        )
        
        # Block 2: Expert model should produce a score
        expert_score = self.expert_models.compute_expert_criticality(test_defect)
        self.assertGreaterEqual(expert_score, 0.0)
        self.assertLessEqual(expert_score, 5.0)
        
        # Block 3: FIS should integrate expert score with other parameters
        final_score, results = self.fis.compute_final_criticality(test_defect)
        self.assertGreaterEqual(final_score, 0.0)
        self.assertLessEqual(final_score, 5.0)
        
        # Verify that expert score is included in the results
        self.assertIn('expert_criticality', results)
        self.assertEqual(results['expert_criticality'], expert_score)

def run_comprehensive_tests():
    """Run all test suites and generate a comprehensive report"""
    print("=" * 80)
    print("COMPREHENSIVE TEST SUITE FOR ENHANCED VISION_FUZZY SYSTEM")
    print("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestExpertCriticalityModels,
        TestEnhancedFuzzyInferenceSystem,
        TestValidationFramework,
        TestIntegrationBridge,
        TestResearchPaperCompliance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2]}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    import sys
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)