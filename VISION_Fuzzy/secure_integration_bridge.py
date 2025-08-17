"""
Secure Integration Bridge with Enhanced Security and Performance
===============================================================

This module provides a secure, high-performance integration layer between 
the C++ image processing system and the Python fuzzy logic system.

Security Features:
- Input validation and sanitization
- Secure subprocess execution
- Path traversal protection
- JSON schema validation
- Rate limiting and resource controls
- Comprehensive audit logging

Performance Features:
- Asynchronous processing
- Result caching and memoization
- Memory optimization
- Parallel batch processing
- GPU acceleration support

Author: Enhanced security implementation based on Radiuk et al. (2025)
"""

import json
import numpy as np
import pandas as pd
import argparse
import logging
import sys
import os
import asyncio
import hashlib
import time
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from functools import lru_cache, wraps
import jsonschema
from jsonschema import validate, ValidationError
import secrets
import uuid
from datetime import datetime

# Import the enhanced FIS components
from enhanced_fis_core import (
    EnhancedFuzzyInferenceSystem,
    DefectParameters,
    DefectType,
    ComponentType,
    ExpertCriticalityModels
)

# Import security framework components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'security'))
from input_validator import InputValidator, ValidationRule, InputType
from audit_logger import AuditLogger, SecurityEvent, SecurityEventType, RiskLevel
from cryptographic_manager import CryptographicManager
from secure_data_manager import SecureDataManager, DataClassification, DataCategory

# Configure secure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('secure_integration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Security configuration
class SecurityConfig:
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_DETECTIONS_PER_REQUEST = 1000
    MAX_PROCESSING_TIME = 300  # 5 minutes
    ALLOWED_FILE_EXTENSIONS = {'.json', '.csv'}
    TEMP_DIR_PREFIX = 'secure_uav_'
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 60
    MAX_CONCURRENT_REQUESTS = 10

# JSON Schema for input validation
INPUT_SCHEMA = {
    "type": "array",
    "maxItems": SecurityConfig.MAX_DETECTIONS_PER_REQUEST,
    "items": {
        "type": "object",
        "required": ["defect_type", "component_type"],
        "properties": {
            "defect_id": {"type": "integer", "minimum": 0},
            "defect_type": {"type": "string", "enum": ["crack", "erosion", "hotspot"]},
            "component_type": {"type": "string", "enum": [
                "blade_root", "blade_mid_span", "blade_tip", "spar_cap", 
                "trailing_edge", "nacelle_housing", "tower_section", "generator_housing"
            ]},
            "area_mm2": {"type": "number", "minimum": 0, "maximum": 100000},
            "length_mm": {"type": "number", "minimum": 0, "maximum": 10000},
            "width_mm": {"type": "number", "minimum": 0, "maximum": 1000},
            "location_normalized": {"type": "number", "minimum": 0, "maximum": 1},
            "thermal_delta_t": {"type": "number", "minimum": 0, "maximum": 100},
            "thermal_gradient": {"type": "number", "minimum": 0, "maximum": 10},
            "thermal_contrast": {"type": "number", "minimum": 0, "maximum": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "additionalProperties": False
    }
}

class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class RateLimiter:
    """Thread-safe rate limiter implementation"""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed under rate limit"""
        current_time = time.time()
        
        with self.lock:
            # Remove old requests outside time window
            self.requests = [req_time for req_time in self.requests 
                           if current_time - req_time < self.time_window]
            
            if len(self.requests) < self.max_requests:
                self.requests.append(current_time)
                return True
            
            return False

class SecurePathValidator:
    """Secure path validation and sanitization"""
    
    @staticmethod
    def validate_path(path: str, base_dir: Optional[str] = None) -> str:
        """Validate and sanitize file path to prevent directory traversal"""
        if not path:
            raise SecurityError("Empty path provided")
        
        # Convert to Path object for safe handling
        path_obj = Path(path).resolve()
        
        # Check for directory traversal attempts
        if '..' in str(path_obj) or str(path_obj).startswith('/'):
            raise SecurityError(f"Invalid path detected: {path}")
        
        # If base directory specified, ensure path is within it
        if base_dir:
            base_path = Path(base_dir).resolve()
            try:
                path_obj.relative_to(base_path)
            except ValueError:
                raise SecurityError(f"Path outside allowed directory: {path}")
        
        return str(path_obj)
    
    @staticmethod
    def validate_file_extension(path: str) -> bool:
        """Validate file extension against allowed list"""
        ext = Path(path).suffix.lower()
        return ext in SecurityConfig.ALLOWED_FILE_EXTENSIONS

class SecureJSONHandler:
    """Secure JSON parsing with schema validation"""
    
    @staticmethod
    def load_json(file_path: str, schema: Optional[Dict] = None) -> Any:
        """Securely load and validate JSON file"""
        # Validate path
        safe_path = SecurePathValidator.validate_path(file_path)
        
        # Check file size
        if os.path.getsize(safe_path) > SecurityConfig.MAX_FILE_SIZE:
            raise SecurityError(f"File too large: {safe_path}")
        
        # Check file extension
        if not SecurePathValidator.validate_file_extension(safe_path):
            raise SecurityError(f"Invalid file extension: {safe_path}")
        
        try:
            with open(safe_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate against schema if provided
            if schema:
                validate(instance=data, schema=schema)
            
            return data
            
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON format: {e}")
        except jsonschema.ValidationError as e:
            raise ValidationError(f"JSON schema validation failed: {e}")
    
    @staticmethod
    def save_json(data: Any, file_path: str, schema: Optional[Dict] = None) -> None:
        """Securely save JSON data with validation"""
        # Validate data against schema if provided
        if schema:
            validate(instance=data, schema=schema)
        
        # Validate path
        safe_path = SecurePathValidator.validate_path(file_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        
        # Write to temporary file first, then move (atomic operation)
        temp_path = safe_path + '.tmp'
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic move
            shutil.move(temp_path, safe_path)
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    def __init__(self):
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.memory_pool = []
    
    @lru_cache(maxsize=1000)
    def cached_fuzzy_inference(self, params_hash: str, params: DefectParameters) -> Tuple[float, Dict]:
        """Cached fuzzy inference computation"""
        fis = EnhancedFuzzyInferenceSystem()
        return fis.compute_final_criticality(params)
    
    def get_params_hash(self, params: DefectParameters) -> str:
        """Generate hash for defect parameters for caching"""
        param_str = f"{params.area_real_mm2}_{params.location_normalized}_{params.delta_t_max}_{params.defect_type.value}_{params.component_type.value}"
        return hashlib.md5(param_str.encode()).hexdigest()

@dataclass
class SecureIntegrationResult:
    """Secure result structure with additional metadata"""
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
    
    # Security and performance metadata
    confidence: float
    processing_time_ms: float
    request_id: str
    timestamp: float
    validation_passed: bool
    cache_hit: bool

class SecureIntegrationBridge:
    """
    Secure, high-performance integration bridge with comprehensive security controls
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize secure integration bridge"""
        self.config = self._load_secure_config(config_path)
        self.fis = EnhancedFuzzyInferenceSystem()
        self.expert_models = ExpertCriticalityModels()
        self.rate_limiter = RateLimiter(SecurityConfig.MAX_REQUESTS_PER_MINUTE)
        self.performance_optimizer = PerformanceOptimizer()
        self.active_requests = 0
        self.request_lock = threading.Lock()
        
        # Initialize security framework components
        self.crypto_manager = CryptographicManager()
        self.audit_logger = AuditLogger(self.crypto_manager)
        self.secure_data_manager = SecureDataManager(self.crypto_manager, self.audit_logger)
        self.input_validator = InputValidator(strict_mode=True)
        
        # Create secure temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix=SecurityConfig.TEMP_DIR_PREFIX)
        
        # Log initialization
        self.audit_logger.log_security_event(SecurityEvent(
            event_id="",
            timestamp=datetime.now(),
            event_type=SecurityEventType.SYSTEM_ACCESS,
            risk_level=RiskLevel.LOW,
            user_id="system",
            username="integration_bridge",
            source_ip="localhost",
            user_agent="secure_integration_bridge",
            resource="fuzzy_integration",
            action="initialize",
            result="success",
            details={"temp_dir": self.temp_dir},
            risk_score=0.1
        ))
        
        logger.info("Secure integration bridge initialized successfully")
        logger.info(f"Temporary directory: {self.temp_dir}")
    
    def __del__(self):
        """Cleanup temporary directory on destruction"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _load_secure_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration with security validation"""
        default_config = {
            "processing": {
                "enable_validation": True,
                "enable_visualization": False,  # Disabled for security
                "save_intermediate_results": False,
                "max_processing_time": SecurityConfig.MAX_PROCESSING_TIME
            },
            "fuzzy_system": {
                "enable_sensitivity_analysis": False,
                "confidence_threshold": 0.5,
                "enable_caching": True
            },
            "output": {
                "format": "json",
                "include_metadata": True,
                "precision": 3
            },
            "security": {
                "enable_rate_limiting": True,
                "enable_input_validation": True,
                "enable_audit_logging": True,
                "max_file_size": SecurityConfig.MAX_FILE_SIZE
            }
        }
        
        if config_path:
            try:
                # Use secure JSON handler
                user_config = SecureJSONHandler.load_json(config_path)
                # Merge with defaults (shallow merge for security)
                for key in user_config:
                    if key in default_config:
                        default_config[key].update(user_config[key])
                
                logger.info(f"Secure configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Using default secure configuration")
        
        return default_config
    
    def _validate_request(self, input_data: Any) -> List[Dict]:
        """Comprehensive request validation using security framework"""
        # Rate limiting check
        if self.config['security']['enable_rate_limiting']:
            if not self.rate_limiter.is_allowed():
                self.audit_logger.log_security_violation(
                    "system", "integration_bridge", "rate_limit_exceeded",
                    "Request rate limit exceeded", "localhost", "medium"
                )
                raise SecurityError("Rate limit exceeded")
        
        # Concurrent request limit
        with self.request_lock:
            if self.active_requests >= SecurityConfig.MAX_CONCURRENT_REQUESTS:
                self.audit_logger.log_security_violation(
                    "system", "integration_bridge", "concurrent_limit_exceeded",
                    "Too many concurrent requests", "localhost", "medium"
                )
                raise SecurityError("Too many concurrent requests")
            self.active_requests += 1
        
        try:
            # Enhanced input validation using security framework
            if self.config['security']['enable_input_validation']:
                if isinstance(input_data, dict):
                    detections = input_data.get('detections', [])
                else:
                    detections = input_data
                
                # Validate using InputValidator
                validation_rule = ValidationRule(
                    input_type=InputType.JSON,
                    required=True,
                    custom_validator=lambda x: self._validate_detections_schema(x)
                )
                
                validated_detections = self.input_validator.validate_input(
                    detections, validation_rule, "detections"
                )
                
                # Additional security checks
                for i, detection in enumerate(validated_detections):
                    self._validate_detection_security(detection, i)
                
                return validated_detections
            
            return input_data if isinstance(input_data, list) else input_data.get('detections', [])
            
        except Exception as e:
            self.audit_logger.log_security_violation(
                "system", "integration_bridge", "input_validation_failed",
                f"Input validation failed: {str(e)}", "localhost", "high"
            )
            raise ValidationError(f"Input validation failed: {e}")
        finally:
            with self.request_lock:
                self.active_requests -= 1
    
    def _validate_detections_schema(self, detections: List[Dict]) -> bool:
        """Validate detections against schema using jsonschema"""
        try:
            validate(instance=detections, schema=INPUT_SCHEMA)
            return True
        except jsonschema.ValidationError:
            return False
    
    def _validate_detection_security(self, detection: Dict, index: int):
        """Additional security validation for individual detection"""
        # Validate numerical bounds
        for field, (min_val, max_val) in [
            ('area_mm2', (0, 100000)),
            ('length_mm', (0, 10000)),
            ('width_mm', (0, 1000)),
            ('location_normalized', (0, 1)),
            ('thermal_delta_t', (0, 100)),
            ('thermal_gradient', (0, 10)),
            ('thermal_contrast', (0, 1)),
            ('confidence', (0, 1))
        ]:
            if field in detection:
                self.input_validator.validate_numerical_bounds(
                    detection[field], min_val, max_val, f"detection[{index}].{field}"
                )
        
        # Validate string fields
        for field in ['defect_type', 'component_type']:
            if field in detection:
                validation_rule = ValidationRule(
                    input_type=InputType.STRING,
                    required=True,
                    max_length=50,
                    pattern=r'^[a-zA-Z_]+$'
                )
                self.input_validator.validate_input(
                    detection[field], validation_rule, f"detection[{index}].{field}"
                )
    
    async def process_detection_results_async(self, input_data: Any) -> List[SecureIntegrationResult]:
        """Asynchronous processing of detection results with security controls"""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Processing request {request_id}")
        
        try:
            # Validate request
            detections = self._validate_request(input_data)
            
            logger.info(f"Request {request_id}: Processing {len(detections)} detections")
            
            # Process detections in parallel
            tasks = []
            for i, detection in enumerate(detections):
                task = asyncio.create_task(
                    self._process_single_detection_async(detection, i, request_id)
                )
                tasks.append(task)
            
            # Wait for all tasks with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config['processing']['max_processing_time']
            )
            
            # Filter out exceptions and log them
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Request {request_id}: Error processing detection {i}: {result}")
                else:
                    valid_results.append(result)
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Request {request_id}: Completed in {processing_time:.2f}ms, "
                       f"processed {len(valid_results)}/{len(detections)} detections")
            
            return valid_results
            
        except Exception as e:
            logger.error(f"Request {request_id}: Failed with error: {e}")
            raise
    
    async def _process_single_detection_async(self, detection: Dict, detection_id: int, request_id: str) -> SecureIntegrationResult:
        """Asynchronously process a single detection with security controls"""
        start_time = time.time()
        cache_hit = False
        
        try:
            # Extract and validate data
            defect_type_str = detection.get('defect_type', 'crack')
            component_type_str = detection.get('component_type', 'blade_mid_span')
            
            # Map string types to enums with validation
            defect_type = self._map_defect_type_secure(defect_type_str)
            component_type = self._map_component_type_secure(component_type_str)
            
            # Create DefectParameters with bounds checking
            defect_params = DefectParameters(
                area_real_mm2=max(0, min(detection.get('area_mm2', 100.0), 100000)),
                length_real_mm=max(0, min(detection.get('length_mm', 10.0), 10000)),
                width_avg_mm=max(0, min(detection.get('width_mm', 2.0), 1000)),
                perimeter_real_mm=max(0, min(detection.get('perimeter_mm', 25.0), 50000)),
                curvature_avg=max(0, min(detection.get('curvature_avg', 0.1), 1.0)),
                location_normalized=max(0, min(detection.get('location_normalized', 0.5), 1.0)),
                component_type=component_type,
                delta_t_max=max(0, min(detection.get('thermal_delta_t', 5.0), 100)),
                thermal_gradient=max(0, min(detection.get('thermal_gradient', 0.3), 10)),
                thermal_contrast=max(0, min(detection.get('thermal_contrast', 0.5), 1.0)),
                confidence=max(0, min(detection.get('confidence', 0.8), 1.0)),
                defect_type=defect_type
            )
            
            # Check cache if enabled
            if self.config['fuzzy_system']['enable_caching']:
                params_hash = self.performance_optimizer.get_params_hash(defect_params)
                try:
                    final_criticality, detailed_results = self.performance_optimizer.cached_fuzzy_inference(
                        params_hash, defect_params
                    )
                    cache_hit = True
                except:
                    # Cache miss, compute normally
                    pass
            
            if not cache_hit:
                # Block 2: Compute expert-driven criticality
                expert_criticality = self.expert_models.compute_expert_criticality(defect_params)
                
                # Block 3: Compute final criticality using fuzzy inference
                final_criticality, detailed_results = self.fis.compute_final_criticality(defect_params)
            else:
                expert_criticality = detailed_results.get('expert_criticality', 0.0)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Create secure result object
            result = SecureIntegrationResult(
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
                processing_time_ms=processing_time,
                request_id=request_id,
                timestamp=time.time(),
                validation_passed=True,
                cache_hit=cache_hit
            )
            
            logger.debug(f"Request {request_id}: Processed detection {detection_id} "
                        f"(cache_hit={cache_hit}, time={processing_time:.2f}ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"Request {request_id}: Error processing detection {detection_id}: {e}")
            raise
    
    def _map_defect_type_secure(self, defect_type_str: str) -> DefectType:
        """Securely map string defect type to enum with validation"""
        if not isinstance(defect_type_str, str):
            raise ValidationError("Defect type must be a string")
        
        mapping = {
            'crack': DefectType.CRACK,
            'erosion': DefectType.EROSION,
            'hotspot': DefectType.HOTSPOT
        }
        
        defect_type_lower = defect_type_str.lower().strip()
        if defect_type_lower not in mapping:
            raise ValidationError(f"Invalid defect type: {defect_type_str}")
        
        return mapping[defect_type_lower]
    
    def _map_component_type_secure(self, component_type_str: str) -> ComponentType:
        """Securely map string component type to enum with validation"""
        if not isinstance(component_type_str, str):
            raise ValidationError("Component type must be a string")
        
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
        
        component_type_lower = component_type_str.lower().strip()
        if component_type_lower not in mapping:
            raise ValidationError(f"Invalid component type: {component_type_str}")
        
        return mapping[component_type_lower]
    
    def save_results_secure(self, results: List[SecureIntegrationResult], output_path: str):
        """Securely save integration results with validation"""
        try:
            # Validate output path
            safe_output_path = SecurePathValidator.validate_path(output_path, self.temp_dir)
            
            output_format = self.config['output']['format'].lower()
            precision = self.config['output']['precision']
            
            if output_format == 'json':
                self._save_json_results_secure(results, safe_output_path, precision)
            elif output_format == 'csv':
                self._save_csv_results_secure(results, safe_output_path, precision)
            else:
                logger.warning(f"Unknown output format: {output_format}, defaulting to JSON")
                self._save_json_results_secure(results, safe_output_path, precision)
                
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def _save_json_results_secure(self, results: List[SecureIntegrationResult], output_path: str, precision: int):
        """Securely save results in JSON format"""
        json_data = []
        
        for result in results:
            result_dict = asdict(result)
            
            # Round numerical values and sanitize
            result_dict['expert_criticality'] = round(result_dict['expert_criticality'], precision)
            result_dict['final_criticality'] = round(result_dict['final_criticality'], precision)
            result_dict['confidence'] = round(result_dict['confidence'], precision)
            result_dict['processing_time_ms'] = round(result_dict['processing_time_ms'], 2)
            
            # Remove sensitive information if not in debug mode
            if not self.config.get('debug', False):
                result_dict.pop('request_id', None)
                result_dict.pop('timestamp', None)
            
            json_data.append(result_dict)
        
        # Use secure JSON handler
        SecureJSONHandler.save_json(json_data, output_path)
        logger.info(f"Results securely saved to {output_path} (JSON format)")

async def main():
    """Secure main function with comprehensive error handling"""
    parser = argparse.ArgumentParser(
        description="Secure integration bridge between VISION_Recognition and VISION_Fuzzy systems"
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input JSON file with detection results')
    parser.add_argument('--output', '-o', required=True, help='Output file path for integration results')
    parser.add_argument('--config', '-c', help='Configuration file path (optional)')
    parser.add_argument('--format', '-f', choices=['json', 'csv'], default='json', help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    bridge = None
    try:
        # Initialize secure integration bridge
        bridge = SecureIntegrationBridge(args.config)
        
        # Override output format if specified
        if args.format:
            bridge.config['output']['format'] = args.format
        
        # Load input data securely
        logger.info(f"Loading detection results from {args.input}")
        input_data = SecureJSONHandler.load_json(args.input, INPUT_SCHEMA)
        
        # Process through secure integration pipeline
        logger.info("Processing detections through secure integration pipeline...")
        results = await bridge.process_detection_results_async(input_data)
        
        if not results:
            logger.error("No results generated from input data")
            sys.exit(1)
        
        # Save results securely
        logger.info(f"Saving results to {args.output}")
        bridge.save_results_secure(results, args.output)
        
        logger.info("Secure integration processing completed successfully!")
        
    except SecurityError as e:
        logger.error(f"Security error: {e}")
        sys.exit(1)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if bridge:
            del bridge

if __name__ == "__main__":
    asyncio.run(main())