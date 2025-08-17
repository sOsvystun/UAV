"""
Integration Bridge between VISION_Recognition and VISION_Fuzzy
==============================================================

This module provides the integration layer between the C++ image processing
system and the Python fuzzy logic system, implementing the complete three-block
framework from the research paper.

Author: Based on research by Radiuk et al. (2025)
"""

import json
import numpy as np
import pandas as pd
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import subprocess

# Import the enhanced FIS components
from enhanced_fis_core import (
    EnhancedFuzzyInferenceSystem,
    DefectParameters,
    DefectType,
    ComponentType,
    ExpertCriticalityModels
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class IntegrationResult:
    """Results from the complete integration pipeline"""
    defect_id: int
    defect_type: str
    component_type: str
    
    # Block 1 results (from C++ image processing)
    geometric_features: Dict
    thermal_features: Dict
    
    # Block 2 results (expert models)
    expert_criticality: float
    
    # Block 3 results (fuzzy inference)
    final_criticality: float
    epri_level: str
    
    # Additional metadata
    confidence: float
    processing_time_ms: float

class IntegrationBridge:
    """
    Main integration bridge class that orchestrates the complete pipeline
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the integration bridge
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        self.fis = EnhancedFuzzyInferenceSystem()
        self.expert_models = ExpertCriticalityModels()
        
        logger.info("Integration bridge initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "processing": {
                "enable_validation": True,
                "enable_visualization": True,
                "save_intermediate_results": False
            },
            "fuzzy_system": {
                "enable_sensitivity_analysis": False,
                "confidence_threshold": 0.5
            },
            "output": {
                "format": "json",
                "include_metadata": True,
                "precision": 3
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    default_config.update(user_config)
                    logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Using default configuration")
        
        return default_config
    
    def process_detection_results(self, input_data: Dict) -> List[IntegrationResult]:
        """
        Process detection results from C++ system through the complete pipeline
        
        Args:
            input_data: Dictionary containing detection results from C++ system
            
        Returns:
            List of IntegrationResult objects
        """
        results = []
        
        if isinstance(input_data, list):
            detections = input_data
        else:
            detections = input_data.get('detections', [])
        
        logger.info(f"Processing {len(detections)} detections through integration pipeline")
        
        for i, detection in enumerate(detections):
            try:
                result = self._process_single_detection(detection, i)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing detection {i}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(results)} detections")
        return results
    
    def _process_single_detection(self, detection: Dict, detection_id: int) -> IntegrationResult:
        """
        Process a single detection through the complete three-block pipeline
        
        Args:
            detection: Detection data from C++ system
            detection_id: Unique identifier for this detection
            
        Returns:
            IntegrationResult object
        """
        import time
        start_time = time.time()
        
        # Extract data from C++ detection result
        defect_type_str = detection.get('defect_type', 'crack')
        component_type_str = detection.get('component_type', 'blade_mid_span')
        
        # Map string types to enums
        defect_type = self._map_defect_type(defect_type_str)
        component_type = self._map_component_type(component_type_str)
        
        # Create DefectParameters object for fuzzy system
        defect_params = DefectParameters(
            area_real_mm2=detection.get('area_mm2', 100.0),
            length_real_mm=detection.get('length_mm', 10.0),
            width_avg_mm=detection.get('width_mm', 2.0),
            perimeter_real_mm=detection.get('perimeter_mm', 25.0),
            curvature_avg=detection.get('curvature_avg', 0.1),
            location_normalized=detection.get('location_normalized', 0.5),
            component_type=component_type,
            delta_t_max=detection.get('thermal_delta_t', 5.0),
            thermal_gradient=detection.get('thermal_gradient', 0.3),
            thermal_contrast=detection.get('thermal_contrast', 0.5),
            confidence=detection.get('confidence', 0.8),
            defect_type=defect_type
        )
        
        # Block 2: Compute expert-driven criticality
        expert_criticality = self.expert_models.compute_expert_criticality(defect_params)
        
        # Block 3: Compute final criticality using fuzzy inference
        final_criticality, detailed_results = self.fis.compute_final_criticality(defect_params)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Create result object
        result = IntegrationResult(
            defect_id=detection_id,
            defect_type=defect_type_str,
            component_type=component_type_str,
            geometric_features={
                'area_mm2': defect_params.area_real_mm2,
                'length_mm': defect_params.length_real_mm,
                'width_mm': defect_params.width_avg_mm,
                'perimeter_mm': defect_params.perimeter_real_mm,
                'curvature': defect_params.curvature_avg,
                'location_normalized': defect_params.location_normalized
            },
            thermal_features={
                'delta_t_max': defect_params.delta_t_max,
                'thermal_gradient': defect_params.thermal_gradient,
                'thermal_contrast': defect_params.thermal_contrast
            },
            expert_criticality=expert_criticality,
            final_criticality=final_criticality,
            epri_level=detailed_results['epri_level'],
            confidence=defect_params.confidence,
            processing_time_ms=processing_time
        )
        
        logger.debug(f"Processed detection {detection_id}: "
                    f"Expert={expert_criticality:.3f}, "
                    f"Final={final_criticality:.3f}, "
                    f"EPRI={detailed_results['epri_level']}")
        
        return result
    
    def _map_defect_type(self, defect_type_str: str) -> DefectType:
        """Map string defect type to enum"""
        mapping = {
            'crack': DefectType.CRACK,
            'erosion': DefectType.EROSION,
            'hotspot': DefectType.HOTSPOT
        }
        return mapping.get(defect_type_str.lower(), DefectType.CRACK)
    
    def _map_component_type(self, component_type_str: str) -> ComponentType:
        """Map string component type to enum"""
        mapping = {
            'blade_root': ComponentType.BLADE_ROOT,
            'blade_mid_span': ComponentType.BLADE_MID_SPAN,
            'blade_tip': ComponentType.BLADE_TIP,
            'spar_cap': ComponentType.SPAR_CAP,
            'trailing_edge': ComponentType.TRAILING_EDGE,
            'nacelle_housing': ComponentType.NACELLE_HOUSING,
            'tower_section': ComponentType.TOWER_SECTION,
            'generator_housing': ComponentType.GENERATOR_HOUSING
        }
        return mapping.get(component_type_str.lower(), ComponentType.BLADE_MID_SPAN)
    
    def save_results(self, results: List[IntegrationResult], output_path: str):
        """
        Save integration results to file
        
        Args:
            results: List of IntegrationResult objects
            output_path: Path to save results
        """
        output_format = self.config['output']['format'].lower()
        precision = self.config['output']['precision']
        
        if output_format == 'json':
            self._save_json_results(results, output_path, precision)
        elif output_format == 'csv':
            self._save_csv_results(results, output_path, precision)
        else:
            logger.warning(f"Unknown output format: {output_format}, defaulting to JSON")
            self._save_json_results(results, output_path, precision)
    
    def _save_json_results(self, results: List[IntegrationResult], output_path: str, precision: int):
        """Save results in JSON format"""
        json_data = []
        
        for result in results:
            result_dict = asdict(result)
            
            # Round numerical values
            result_dict['expert_criticality'] = round(result_dict['expert_criticality'], precision)
            result_dict['final_criticality'] = round(result_dict['final_criticality'], precision)
            result_dict['confidence'] = round(result_dict['confidence'], precision)
            result_dict['processing_time_ms'] = round(result_dict['processing_time_ms'], 2)
            
            json_data.append(result_dict)
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path} (JSON format)")
    
    def _save_csv_results(self, results: List[IntegrationResult], output_path: str, precision: int):
        """Save results in CSV format"""
        # Flatten the results for CSV
        csv_data = []
        
        for result in results:
            row = {
                'defect_id': result.defect_id,
                'defect_type': result.defect_type,
                'component_type': result.component_type,
                'expert_criticality': round(result.expert_criticality, precision),
                'final_criticality': round(result.final_criticality, precision),
                'epri_level': result.epri_level,
                'confidence': round(result.confidence, precision),
                'processing_time_ms': round(result.processing_time_ms, 2),
                
                # Geometric features
                'area_mm2': round(result.geometric_features['area_mm2'], precision),
                'length_mm': round(result.geometric_features['length_mm'], precision),
                'width_mm': round(result.geometric_features['width_mm'], precision),
                'location_normalized': round(result.geometric_features['location_normalized'], precision),
                
                # Thermal features
                'thermal_delta_t': round(result.thermal_features['delta_t_max'], precision),
                'thermal_gradient': round(result.thermal_features['thermal_gradient'], precision),
                'thermal_contrast': round(result.thermal_features['thermal_contrast'], precision)
            }
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to {output_path} (CSV format)")
    
    def generate_summary_report(self, results: List[IntegrationResult]) -> Dict:
        """
        Generate summary statistics from integration results
        
        Args:
            results: List of IntegrationResult objects
            
        Returns:
            Dictionary containing summary statistics
        """
        if not results:
            return {"error": "No results to summarize"}
        
        # Basic statistics
        total_defects = len(results)
        defect_types = [r.defect_type for r in results]
        component_types = [r.component_type for r in results]
        criticality_scores = [r.final_criticality for r in results]
        epri_levels = [r.epri_level for r in results]
        
        # Count by type
        defect_type_counts = pd.Series(defect_types).value_counts().to_dict()
        component_type_counts = pd.Series(component_types).value_counts().to_dict()
        epri_level_counts = pd.Series(epri_levels).value_counts().to_dict()
        
        # Criticality statistics
        criticality_stats = {
            'mean': np.mean(criticality_scores),
            'median': np.median(criticality_scores),
            'std': np.std(criticality_scores),
            'min': np.min(criticality_scores),
            'max': np.max(criticality_scores)
        }
        
        # Find most critical defect
        max_criticality_idx = np.argmax(criticality_scores)
        most_critical = results[max_criticality_idx]
        
        # Processing performance
        processing_times = [r.processing_time_ms for r in results]
        avg_processing_time = np.mean(processing_times)
        
        summary = {
            'total_defects': total_defects,
            'defect_type_distribution': defect_type_counts,
            'component_type_distribution': component_type_counts,
            'epri_level_distribution': epri_level_counts,
            'criticality_statistics': criticality_stats,
            'most_critical_defect': {
                'defect_id': most_critical.defect_id,
                'defect_type': most_critical.defect_type,
                'component_type': most_critical.component_type,
                'criticality_score': most_critical.final_criticality,
                'epri_level': most_critical.epri_level
            },
            'performance': {
                'average_processing_time_ms': avg_processing_time,
                'total_processing_time_ms': sum(processing_times)
            }
        }
        
        return summary

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(
        description="Integration bridge between VISION_Recognition and VISION_Fuzzy systems"
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input JSON file with detection results from C++ system'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output file path for integration results'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Configuration file path (optional)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'csv'],
        default='json',
        help='Output format (default: json)'
    )
    
    parser.add_argument(
        '--summary', '-s',
        help='Path to save summary report (optional)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize integration bridge
        bridge = IntegrationBridge(args.config)
        
        # Override output format if specified
        if args.format:
            bridge.config['output']['format'] = args.format
        
        # Load input data
        logger.info(f"Loading detection results from {args.input}")
        with open(args.input, 'r') as f:
            input_data = json.load(f)
        
        # Process through integration pipeline
        logger.info("Processing detections through integration pipeline...")
        results = bridge.process_detection_results(input_data)
        
        if not results:
            logger.error("No results generated from input data")
            sys.exit(1)
        
        # Save results
        logger.info(f"Saving results to {args.output}")
        bridge.save_results(results, args.output)
        
        # Generate and save summary if requested
        if args.summary:
            logger.info(f"Generating summary report...")
            summary = bridge.generate_summary_report(results)
            
            with open(args.summary, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Summary report saved to {args.summary}")
            
            # Print key statistics
            print("\n=== INTEGRATION SUMMARY ===")
            print(f"Total defects processed: {summary['total_defects']}")
            print(f"Average criticality score: {summary['criticality_statistics']['mean']:.3f}")
            print(f"Most critical defect: ID {summary['most_critical_defect']['defect_id']} "
                  f"({summary['most_critical_defect']['defect_type']}) "
                  f"with score {summary['most_critical_defect']['criticality_score']:.3f}")
            print(f"Average processing time: {summary['performance']['average_processing_time_ms']:.2f} ms")
        
        logger.info("Integration processing completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()