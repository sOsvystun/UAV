"""
Input Validation and Sanitization Framework
==========================================

This module provides comprehensive input validation and sanitization to prevent
injection attacks, path traversal, and other security vulnerabilities.

Key Features:
- File path sanitization and validation
- JSON schema validation with strict type checking
- Numerical bounds checking and type validation
- Command argument sanitization
- SQL injection prevention
- XSS protection for web inputs

Author: Security Framework Team
"""

import os
import re
import json
import pathlib
import urllib.parse
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import jsonschema
from jsonschema import validate, ValidationError
import bleach
import html

# Configure logging
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom validation error with security context"""
    def __init__(self, message: str, field: str = None, value: Any = None, risk_level: str = "medium"):
        self.message = message
        self.field = field
        self.value = str(value)[:100] if value is not None else None  # Truncate for security
        self.risk_level = risk_level
        super().__init__(self.message)

class InputType(Enum):
    """Supported input types for validation"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    EMAIL = "email"
    URL = "url"
    FILE_PATH = "file_path"
    JSON = "json"
    COMMAND = "command"
    SQL_IDENTIFIER = "sql_identifier"

@dataclass
class ValidationRule:
    """Validation rule configuration"""
    input_type: InputType
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[callable] = None

class InputValidator:
    """
    Comprehensive input validation and sanitization system.
    
    This class provides methods to validate and sanitize various types of input
    to prevent security vulnerabilities including injection attacks, path traversal,
    and data corruption.
    """
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        
        # Compile regex patterns for performance
        self._compile_patterns()
        
        # Initialize bleach for HTML sanitization
        self.allowed_tags = ['b', 'i', 'u', 'em', 'strong', 'p', 'br']
        self.allowed_attributes = {}
        
        logger.info(f"InputValidator initialized (strict_mode={strict_mode})")
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient validation"""
        self.patterns = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'url': re.compile(r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'),
            'sql_identifier': re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$'),
            'safe_filename': re.compile(r'^[a-zA-Z0-9._-]+$'),
            'command_injection': re.compile(r'[;&|`$(){}[\]<>*?~!]'),
            'path_traversal': re.compile(r'\.\.[\\/]'),
            'script_tags': re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            'sql_injection': re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)", re.IGNORECASE)
        }
    
    def validate_input(self, value: Any, rule: ValidationRule, field_name: str = "unknown") -> Any:
        """
        Validate input against specified rule.
        
        Args:
            value: Input value to validate
            rule: ValidationRule specifying validation criteria
            field_name: Name of the field being validated
            
        Returns:
            Validated and sanitized value
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Check if required
            if rule.required and (value is None or value == ""):
                raise ValidationError(f"Field '{field_name}' is required", field_name, value, "high")
            
            # Skip validation for optional empty values
            if not rule.required and (value is None or value == ""):
                return None
            
            # Type-specific validation
            if rule.input_type == InputType.STRING:
                return self._validate_string(value, rule, field_name)
            elif rule.input_type == InputType.INTEGER:
                return self._validate_integer(value, rule, field_name)
            elif rule.input_type == InputType.FLOAT:
                return self._validate_float(value, rule, field_name)
            elif rule.input_type == InputType.BOOLEAN:
                return self._validate_boolean(value, rule, field_name)
            elif rule.input_type == InputType.EMAIL:
                return self._validate_email(value, rule, field_name)
            elif rule.input_type == InputType.URL:
                return self._validate_url(value, rule, field_name)
            elif rule.input_type == InputType.FILE_PATH:
                return self._validate_file_path(value, rule, field_name)
            elif rule.input_type == InputType.JSON:
                return self._validate_json(value, rule, field_name)
            elif rule.input_type == InputType.COMMAND:
                return self._validate_command(value, rule, field_name)
            elif rule.input_type == InputType.SQL_IDENTIFIER:
                return self._validate_sql_identifier(value, rule, field_name)
            else:
                raise ValidationError(f"Unsupported input type: {rule.input_type}", field_name, value)
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected validation error for field '{field_name}': {e}")
            raise ValidationError(f"Validation failed for field '{field_name}'", field_name, value, "high")
    
    def _validate_string(self, value: Any, rule: ValidationRule, field_name: str) -> str:
        """Validate string input"""
        if not isinstance(value, str):
            try:
                value = str(value)
            except:
                raise ValidationError(f"Cannot convert '{field_name}' to string", field_name, value)
        
        # Length validation
        if rule.min_length is not None and len(value) < rule.min_length:
            raise ValidationError(f"Field '{field_name}' must be at least {rule.min_length} characters", field_name, value)
        
        if rule.max_length is not None and len(value) > rule.max_length:
            raise ValidationError(f"Field '{field_name}' must be at most {rule.max_length} characters", field_name, value)
        
        # Pattern validation
        if rule.pattern and not re.match(rule.pattern, value):
            raise ValidationError(f"Field '{field_name}' does not match required pattern", field_name, value)
        
        # Allowed values validation
        if rule.allowed_values and value not in rule.allowed_values:
            raise ValidationError(f"Field '{field_name}' must be one of: {rule.allowed_values}", field_name, value)
        
        # Security checks
        if self.strict_mode:
            value = self._sanitize_string(value, field_name)
        
        # Custom validation
        if rule.custom_validator:
            if not rule.custom_validator(value):
                raise ValidationError(f"Custom validation failed for field '{field_name}'", field_name, value)
        
        return value
    
    def _validate_integer(self, value: Any, rule: ValidationRule, field_name: str) -> int:
        """Validate integer input"""
        try:
            if isinstance(value, str):
                value = int(value)
            elif not isinstance(value, int):
                value = int(value)
        except (ValueError, TypeError):
            raise ValidationError(f"Field '{field_name}' must be a valid integer", field_name, value)
        
        # Range validation
        if rule.min_value is not None and value < rule.min_value:
            raise ValidationError(f"Field '{field_name}' must be at least {rule.min_value}", field_name, value)
        
        if rule.max_value is not None and value > rule.max_value:
            raise ValidationError(f"Field '{field_name}' must be at most {rule.max_value}", field_name, value)
        
        # Allowed values validation
        if rule.allowed_values and value not in rule.allowed_values:
            raise ValidationError(f"Field '{field_name}' must be one of: {rule.allowed_values}", field_name, value)
        
        return value
    
    def _validate_float(self, value: Any, rule: ValidationRule, field_name: str) -> float:
        """Validate float input"""
        try:
            if isinstance(value, str):
                value = float(value)
            elif not isinstance(value, (int, float)):
                value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(f"Field '{field_name}' must be a valid number", field_name, value)
        
        # Range validation
        if rule.min_value is not None and value < rule.min_value:
            raise ValidationError(f"Field '{field_name}' must be at least {rule.min_value}", field_name, value)
        
        if rule.max_value is not None and value > rule.max_value:
            raise ValidationError(f"Field '{field_name}' must be at most {rule.max_value}", field_name, value)
        
        return float(value)
    
    def _validate_boolean(self, value: Any, rule: ValidationRule, field_name: str) -> bool:
        """Validate boolean input"""
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            lower_value = value.lower()
            if lower_value in ('true', '1', 'yes', 'on'):
                return True
            elif lower_value in ('false', '0', 'no', 'off'):
                return False
            else:
                raise ValidationError(f"Field '{field_name}' must be a valid boolean", field_name, value)
        elif isinstance(value, (int, float)):
            return bool(value)
        else:
            raise ValidationError(f"Field '{field_name}' must be a valid boolean", field_name, value)
    
    def _validate_email(self, value: Any, rule: ValidationRule, field_name: str) -> str:
        """Validate email input"""
        if not isinstance(value, str):
            raise ValidationError(f"Field '{field_name}' must be a string", field_name, value)
        
        # Basic format validation
        if not self.patterns['email'].match(value):
            raise ValidationError(f"Field '{field_name}' must be a valid email address", field_name, value)
        
        # Length validation
        if len(value) > 254:  # RFC 5321 limit
            raise ValidationError(f"Email address '{field_name}' is too long", field_name, value)
        
        return value.lower().strip()
    
    def _validate_url(self, value: Any, rule: ValidationRule, field_name: str) -> str:
        """Validate URL input"""
        if not isinstance(value, str):
            raise ValidationError(f"Field '{field_name}' must be a string", field_name, value)
        
        # Basic format validation
        if not self.patterns['url'].match(value):
            raise ValidationError(f"Field '{field_name}' must be a valid URL", field_name, value)
        
        # Parse and validate components
        try:
            parsed = urllib.parse.urlparse(value)
            if not parsed.scheme or not parsed.netloc:
                raise ValidationError(f"Field '{field_name}' must be a complete URL", field_name, value)
        except Exception:
            raise ValidationError(f"Field '{field_name}' must be a valid URL", field_name, value)
        
        return value.strip()
    
    def _validate_file_path(self, value: Any, rule: ValidationRule, field_name: str) -> str:
        """Validate and sanitize file path"""
        if not isinstance(value, str):
            raise ValidationError(f"Field '{field_name}' must be a string", field_name, value)
        
        # Check for path traversal attempts
        if self.patterns['path_traversal'].search(value):
            raise ValidationError(f"Path traversal detected in field '{field_name}'", field_name, value, "critical")
        
        # Normalize path
        try:
            normalized_path = os.path.normpath(value)
            
            # Ensure path doesn't escape intended directory
            if normalized_path.startswith('..') or '/..' in normalized_path or '\\\\..\\\\' in normalized_path:
                raise ValidationError(f"Invalid path in field '{field_name}'", field_name, value, "critical")
            
            # Check for null bytes
            if '\\x00' in value or '\\0' in value:
                raise ValidationError(f"Null byte detected in path '{field_name}'", field_name, value, "critical")
            
            return normalized_path
            
        except Exception as e:
            raise ValidationError(f"Invalid path format in field '{field_name}'", field_name, value)
    
    def _validate_json(self, value: Any, rule: ValidationRule, field_name: str) -> Dict:
        """Validate JSON input with schema validation"""
        if isinstance(value, dict):
            json_data = value
        elif isinstance(value, str):
            try:
                json_data = json.loads(value)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid JSON in field '{field_name}': {e}", field_name, value)
        else:
            raise ValidationError(f"Field '{field_name}' must be JSON string or dict", field_name, value)
        
        # Schema validation if provided in custom_validator
        if rule.custom_validator and callable(rule.custom_validator):
            try:
                rule.custom_validator(json_data)
            except Exception as e:
                raise ValidationError(f"JSON schema validation failed for '{field_name}': {e}", field_name, value)
        
        return json_data
    
    def _validate_command(self, value: Any, rule: ValidationRule, field_name: str) -> List[str]:
        """Validate and sanitize command arguments"""
        if isinstance(value, str):
            # Split command string into arguments
            import shlex
            try:
                args = shlex.split(value)
            except ValueError as e:
                raise ValidationError(f"Invalid command syntax in '{field_name}': {e}", field_name, value)
        elif isinstance(value, list):
            args = [str(arg) for arg in value]
        else:
            raise ValidationError(f"Field '{field_name}' must be command string or list", field_name, value)
        
        # Check for command injection patterns
        for arg in args:
            if self.patterns['command_injection'].search(arg):
                raise ValidationError(f"Command injection detected in '{field_name}'", field_name, value, "critical")
        
        return args
    
    def _validate_sql_identifier(self, value: Any, rule: ValidationRule, field_name: str) -> str:
        """Validate SQL identifier (table/column names)"""
        if not isinstance(value, str):
            raise ValidationError(f"Field '{field_name}' must be a string", field_name, value)
        
        # Check SQL identifier pattern
        if not self.patterns['sql_identifier'].match(value):
            raise ValidationError(f"Invalid SQL identifier in field '{field_name}'", field_name, value)
        
        # Check for SQL injection patterns
        if self.patterns['sql_injection'].search(value):
            raise ValidationError(f"SQL injection detected in field '{field_name}'", field_name, value, "critical")
        
        return value
    
    def _sanitize_string(self, value: str, field_name: str) -> str:
        """Sanitize string for security"""
        # Remove script tags
        value = self.patterns['script_tags'].sub('', value)
        
        # HTML escape
        value = html.escape(value)
        
        # Remove null bytes
        value = value.replace('\\x00', '').replace('\\0', '')
        
        return value
    
    def validate_file_path(self, path: str, allowed_extensions: List[str] = None, 
                          base_directory: str = None) -> str:
        """
        Comprehensive file path validation and sanitization.
        
        Args:
            path: File path to validate
            allowed_extensions: List of allowed file extensions
            base_directory: Base directory to restrict access to
            
        Returns:
            Sanitized and validated file path
            
        Raises:
            ValidationError: If path is invalid or unsafe
        """
        if not isinstance(path, str):
            raise ValidationError("File path must be a string", "file_path", path)
        
        # Basic sanitization
        path = path.strip()
        
        # Check for empty path
        if not path:
            raise ValidationError("File path cannot be empty", "file_path", path)
        
        # Check for path traversal
        if self.patterns['path_traversal'].search(path):
            raise ValidationError("Path traversal detected", "file_path", path, "critical")
        
        # Normalize path
        try:
            normalized_path = os.path.normpath(path)
        except Exception:
            raise ValidationError("Invalid path format", "file_path", path)
        
        # Check for attempts to escape base directory
        if base_directory:
            base_abs = os.path.abspath(base_directory)
            path_abs = os.path.abspath(os.path.join(base_directory, normalized_path))
            
            if not path_abs.startswith(base_abs):
                raise ValidationError("Path outside allowed directory", "file_path", path, "critical")
        
        # Validate file extension
        if allowed_extensions:
            file_ext = pathlib.Path(normalized_path).suffix.lower()
            if file_ext not in [ext.lower() for ext in allowed_extensions]:
                raise ValidationError(f"File extension not allowed. Allowed: {allowed_extensions}", 
                                    "file_path", path)
        
        # Check filename for dangerous characters
        filename = os.path.basename(normalized_path)
        if not self.patterns['safe_filename'].match(filename):
            raise ValidationError("Filename contains unsafe characters", "file_path", path)
        
        return normalized_path
    
    def validate_json_schema(self, data: Dict, schema: Dict, field_name: str = "json_data") -> Dict:
        """
        Validate JSON data against a schema.
        
        Args:
            data: JSON data to validate
            schema: JSON schema for validation
            field_name: Name of the field being validated
            
        Returns:
            Validated JSON data
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            validate(instance=data, schema=schema)
            return data
        except ValidationError as e:
            raise ValidationError(f"JSON schema validation failed for '{field_name}': {e.message}", 
                                field_name, data)
        except Exception as e:
            raise ValidationError(f"JSON validation error for '{field_name}': {e}", field_name, data)
    
    def sanitize_command_args(self, args: List[str]) -> List[str]:
        """
        Sanitize command arguments to prevent injection attacks.
        
        Args:
            args: List of command arguments
            
        Returns:
            Sanitized command arguments
            
        Raises:
            ValidationError: If dangerous patterns are detected
        """
        sanitized_args = []
        
        for i, arg in enumerate(args):
            if not isinstance(arg, str):
                arg = str(arg)
            
            # Check for injection patterns
            if self.patterns['command_injection'].search(arg):
                raise ValidationError(f"Command injection detected in argument {i}", f"arg_{i}", arg, "critical")
            
            # Basic sanitization
            arg = arg.strip()
            
            # Escape special characters if needed
            if self.strict_mode:
                # Quote arguments that contain spaces or special characters
                if ' ' in arg or any(c in arg for c in '&|<>'):
                    arg = f'"{arg}"'
            
            sanitized_args.append(arg)
        
        return sanitized_args
    
    def validate_numerical_bounds(self, value: Union[int, float], min_val: Union[int, float], 
                                max_val: Union[int, float], field_name: str = "numeric_value") -> Union[int, float]:
        """
        Validate numerical value within specified bounds.
        
        Args:
            value: Numerical value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            field_name: Name of the field being validated
            
        Returns:
            Validated numerical value
            
        Raises:
            ValidationError: If value is out of bounds
        """
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise ValidationError(f"Field '{field_name}' must be a number", field_name, value)
        
        if value < min_val:
            raise ValidationError(f"Field '{field_name}' must be at least {min_val}", field_name, value)
        
        if value > max_val:
            raise ValidationError(f"Field '{field_name}' must be at most {max_val}", field_name, value)
        
        return value
    
    def create_validation_schema(self, rules: Dict[str, ValidationRule]) -> Dict[str, Any]:
        """
        Create a validation schema from validation rules.
        
        Args:
            rules: Dictionary mapping field names to ValidationRule objects
            
        Returns:
            Validation schema dictionary
        """
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for field_name, rule in rules.items():
            if rule.required:
                schema["required"].append(field_name)
            
            field_schema = self._rule_to_schema(rule)
            schema["properties"][field_name] = field_schema
        
        return schema
    
    def _rule_to_schema(self, rule: ValidationRule) -> Dict[str, Any]:
        """Convert ValidationRule to JSON schema format"""
        schema = {}
        
        if rule.input_type == InputType.STRING:
            schema["type"] = "string"
            if rule.min_length is not None:
                schema["minLength"] = rule.min_length
            if rule.max_length is not None:
                schema["maxLength"] = rule.max_length
            if rule.pattern:
                schema["pattern"] = rule.pattern
            if rule.allowed_values:
                schema["enum"] = rule.allowed_values
                
        elif rule.input_type == InputType.INTEGER:
            schema["type"] = "integer"
            if rule.min_value is not None:
                schema["minimum"] = rule.min_value
            if rule.max_value is not None:
                schema["maximum"] = rule.max_value
            if rule.allowed_values:
                schema["enum"] = rule.allowed_values
                
        elif rule.input_type == InputType.FLOAT:
            schema["type"] = "number"
            if rule.min_value is not None:
                schema["minimum"] = rule.min_value
            if rule.max_value is not None:
                schema["maximum"] = rule.max_value
                
        elif rule.input_type == InputType.BOOLEAN:
            schema["type"] = "boolean"
            
        elif rule.input_type == InputType.EMAIL:
            schema["type"] = "string"
            schema["format"] = "email"
            
        elif rule.input_type == InputType.URL:
            schema["type"] = "string"
            schema["format"] = "uri"
            
        else:
            schema["type"] = "string"
        
        return schema


# Example usage and testing
if __name__ == "__main__":
    print("Input Validator Test Suite")
    print("=" * 50)
    
    validator = InputValidator(strict_mode=True)
    
    # Test file path validation
    print("\\n1. Testing File Path Validation...")
    try:
        safe_path = validator.validate_file_path("data/models/model.pkl", 
                                               allowed_extensions=['.pkl', '.json'],
                                               base_directory="data")
        print(f"✅ Safe path: {safe_path}")
    except ValidationError as e:
        print(f"❌ Path validation failed: {e}")
    
    try:
        validator.validate_file_path("../../../etc/passwd")
        print("❌ Path traversal not detected!")
    except ValidationError as e:
        print(f"✅ Path traversal detected: {e.message}")
    
    # Test command validation
    print("\\n2. Testing Command Validation...")
    try:
        safe_cmd = validator.sanitize_command_args(["python", "script.py", "--input", "data.json"])
        print(f"✅ Safe command: {safe_cmd}")
    except ValidationError as e:
        print(f"❌ Command validation failed: {e}")
    
    try:
        validator.sanitize_command_args(["python", "script.py; rm -rf /"])
        print("❌ Command injection not detected!")
    except ValidationError as e:
        print(f"✅ Command injection detected: {e.message}")
    
    # Test numerical validation
    print("\\n3. Testing Numerical Validation...")
    try:
        valid_num = validator.validate_numerical_bounds(2.5, 0.0, 10.0, "confidence_score")
        print(f"✅ Valid number: {valid_num}")
    except ValidationError as e:
        print(f"❌ Numerical validation failed: {e}")
    
    try:
        validator.validate_numerical_bounds(15.0, 0.0, 10.0, "confidence_score")
        print("❌ Bounds checking failed!")
    except ValidationError as e:
        print(f"✅ Bounds violation detected: {e.message}")
    
    # Test JSON schema validation
    print("\\n4. Testing JSON Schema Validation...")
    schema = {
        "type": "object",
        "properties": {
            "defect_type": {"type": "string", "enum": ["crack", "corrosion", "wear"]},
            "severity": {"type": "number", "minimum": 0, "maximum": 10}
        },
        "required": ["defect_type", "severity"]
    }
    
    try:
        valid_data = {"defect_type": "crack", "severity": 7.5}
        validated = validator.validate_json_schema(valid_data, schema)
        print(f"✅ Valid JSON: {validated}")
    except ValidationError as e:
        print(f"❌ JSON validation failed: {e}")
    
    try:
        invalid_data = {"defect_type": "unknown", "severity": 15}
        validator.validate_json_schema(invalid_data, schema)
        print("❌ JSON schema validation failed!")
    except ValidationError as e:
        print(f"✅ JSON schema violation detected: {e.message}")
    
    print("\\n" + "=" * 50)
    print("Input validation test suite completed")