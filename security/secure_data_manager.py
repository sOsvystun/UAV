"""
Secure Data Manager for Classification-Based Data Protection
===========================================================

This module provides enterprise-grade secure data management including:
- Classification-based data encryption and access control
- Data loss prevention (DLP) and compliance
- Secure data lifecycle management
- Data integrity verification and audit trails
- Backup and recovery with encryption
- GDPR and compliance support

Author: Security Framework Team
"""

import os
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sqlite3
from contextlib import contextmanager
import gzip
import base64

from .cryptographic_manager import CryptographicManager, KeyType, EncryptionAlgorithm, EncryptionResult
from .audit_logger import AuditLogger, SecurityEventType, RiskLevel
from .authentication_manager import AuthenticationCredentials, UserRole

# Configure logging
logger = logging.getLogger(__name__)

class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

class DataCategory(Enum):
    """Data category types"""
    INSPECTION_RESULTS = "inspection_results"
    USER_DATA = "user_data"
    SYSTEM_CONFIG = "system_config"
    AUDIT_LOGS = "audit_logs"
    MODEL_DATA = "model_data"
    SENSOR_DATA = "sensor_data"
    REPORTS = "reports"
    CREDENTIALS = "credentials"

class AccessOperation(Enum):
    """Data access operations"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXPORT = "export"
    SHARE = "share"
    BACKUP = "backup"

class DataStatus(Enum):
    """Data lifecycle status"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    QUARANTINED = "quarantined"

@dataclass
class DataRecord:
    """Secure data record metadata"""
    record_id: str
    classification: DataClassification
    category: DataCategory
    owner_id: str
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime]
    status: DataStatus
    encryption_key_id: str
    integrity_hash: str
    access_count: int
    last_accessed: Optional[datetime]
    tags: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'record_id': self.record_id,
            'classification': self.classification.value,
            'category': self.category.value,
            'owner_id': self.owner_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'status': self.status.value,
            'encryption_key_id': self.encryption_key_id,
            'integrity_hash': self.integrity_hash,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'tags': self.tags,
            'metadata': self.metadata
        }

@dataclass
class AccessPolicy:
    """Data access policy definition"""
    policy_id: str
    classification: DataClassification
    category: DataCategory
    allowed_roles: List[UserRole]
    allowed_operations: List[AccessOperation]
    conditions: Dict[str, Any]
    time_restrictions: Optional[Dict[str, Any]]
    ip_restrictions: Optional[List[str]]
    approval_required: bool
    audit_required: bool
    
@dataclass
class DataAccessRequest:
    """Data access request"""
    request_id: str
    user_credentials: AuthenticationCredentials
    record_id: str
    operation: AccessOperation
    justification: str
    requested_at: datetime
    approved: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

class SecureDataManager:
    """
    Comprehensive secure data management system with classification-based
    access control, encryption, and compliance features.
    """
    
    def __init__(self, crypto_manager: CryptographicManager, 
                 audit_logger: AuditLogger,
                 storage_path: str = "security/secure_data",
                 config_path: str = "security/data_config.json"):
        self.crypto = crypto_manager
        self.audit_logger = audit_logger
        self.storage_path = Path(storage_path)
        self.config_path = Path(config_path)
        
        # Create directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "data").mkdir(exist_ok=True)
        (self.storage_path / "backups").mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize database
        self.db_path = self.storage_path / "data_registry.db"
        self._initialize_database()
        
        # Access policies
        self.access_policies: Dict[str, AccessPolicy] = {}
        self._load_access_policies()
        
        # Pending access requests
        self.pending_requests: Dict[str, DataAccessRequest] = {}
        
        # Data encryption keys by classification
        self.classification_keys: Dict[DataClassification, str] = {}
        self._initialize_classification_keys()
        
        logger.info("SecureDataManager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load data management configuration"""
        default_config = {
            "encryption": {
                "algorithm": "aes_256_gcm",
                "key_rotation_days": 90,
                "compress_before_encrypt": True
            },
            "access_control": {
                "require_approval_for_restricted": True,
                "require_approval_for_export": True,
                "session_timeout_minutes": 30,
                "max_concurrent_access": 10
            },
            "data_lifecycle": {
                "auto_archive_days": 365,
                "auto_delete_days": 2555,  # 7 years
                "backup_retention_days": 90,
                "integrity_check_interval_hours": 24
            },
            "compliance": {
                "gdpr_enabled": True,
                "hipaa_enabled": False,
                "sox_enabled": False,
                "data_residency_required": False,
                "anonymization_required": False
            },
            "dlp": {
                "enabled": True,
                "scan_content": True,
                "block_sensitive_patterns": True,
                "quarantine_violations": True
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load data config: {e}")
        else:
            # Save default config
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            os.chmod(self.config_path, 0o600)
        
        return default_config
    
    def _initialize_database(self):
        """Initialize SQLite database for data registry"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT UNIQUE NOT NULL,
                    classification TEXT NOT NULL,
                    category TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    status TEXT NOT NULL,
                    encryption_key_id TEXT NOT NULL,
                    integrity_hash TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    tags TEXT,
                    metadata TEXT,
                    file_path TEXT NOT NULL,
                    file_size INTEGER NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    ip_address TEXT,
                    success BOOLEAN NOT NULL,
                    details TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS access_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    justification TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    approved BOOLEAN DEFAULT FALSE,
                    approved_by TEXT,
                    approved_at TEXT,
                    status TEXT DEFAULT 'pending'
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_records_classification ON data_records(classification)')\n            conn.execute('CREATE INDEX IF NOT EXISTS idx_records_owner ON data_records(owner_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_records_status ON data_records(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_access_log_record ON access_log(record_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_access_log_user ON access_log(user_id)')
            
            conn.commit()
    
    def _load_access_policies(self):
        """Load data access policies"""
        policies_file = self.storage_path / "access_policies.json"
        
        # Default access policies
        default_policies = {
            "public_read": AccessPolicy(
                policy_id="public_read",
                classification=DataClassification.PUBLIC,
                category=DataCategory.REPORTS,
                allowed_roles=[UserRole.VIEWER, UserRole.OPERATOR, UserRole.ENGINEER, UserRole.ADMIN],
                allowed_operations=[AccessOperation.READ],
                conditions={},
                time_restrictions=None,
                ip_restrictions=None,
                approval_required=False,
                audit_required=True
            ),
            "confidential_access": AccessPolicy(
                policy_id="confidential_access",
                classification=DataClassification.CONFIDENTIAL,
                category=DataCategory.INSPECTION_RESULTS,
                allowed_roles=[UserRole.ENGINEER, UserRole.ADMIN],
                allowed_operations=[AccessOperation.READ, AccessOperation.WRITE],
                conditions={"mfa_required": True},
                time_restrictions={"business_hours_only": True},
                ip_restrictions=None,
                approval_required=False,
                audit_required=True
            ),
            "restricted_access": AccessPolicy(
                policy_id="restricted_access",
                classification=DataClassification.RESTRICTED,
                category=DataCategory.CREDENTIALS,
                allowed_roles=[UserRole.ADMIN, UserRole.SECURITY_OFFICER],
                allowed_operations=[AccessOperation.READ],
                conditions={"mfa_required": True, "approval_required": True},
                time_restrictions={"business_hours_only": True},
                ip_restrictions=None,
                approval_required=True,
                audit_required=True
            )
        }
        
        if policies_file.exists():
            try:
                with open(policies_file, 'r') as f:
                    policies_data = json.load(f)
                
                for policy_id, policy_data in policies_data.items():
                    policy = AccessPolicy(
                        policy_id=policy_data['policy_id'],
                        classification=DataClassification(policy_data['classification']),
                        category=DataCategory(policy_data['category']),
                        allowed_roles=[UserRole(r) for r in policy_data['allowed_roles']],
                        allowed_operations=[AccessOperation(o) for o in policy_data['allowed_operations']],
                        conditions=policy_data.get('conditions', {}),
                        time_restrictions=policy_data.get('time_restrictions'),
                        ip_restrictions=policy_data.get('ip_restrictions'),
                        approval_required=policy_data.get('approval_required', False),
                        audit_required=policy_data.get('audit_required', True)
                    )
                    self.access_policies[policy_id] = policy
                
            except Exception as e:
                logger.error(f"Failed to load access policies: {e}")
                self.access_policies = default_policies
        else:
            self.access_policies = default_policies
            self._save_access_policies()
    
    def _save_access_policies(self):
        """Save access policies"""
        policies_file = self.storage_path / "access_policies.json"
        policies_data = {}
        
        for policy_id, policy in self.access_policies.items():
            policies_data[policy_id] = {
                'policy_id': policy.policy_id,
                'classification': policy.classification.value,
                'category': policy.category.value,
                'allowed_roles': [r.value for r in policy.allowed_roles],
                'allowed_operations': [o.value for o in policy.allowed_operations],
                'conditions': policy.conditions,
                'time_restrictions': policy.time_restrictions,
                'ip_restrictions': policy.ip_restrictions,
                'approval_required': policy.approval_required,
                'audit_required': policy.audit_required
            }
        
        with open(policies_file, 'w') as f:
            json.dump(policies_data, f, indent=2)
        os.chmod(policies_file, 0o600)
    
    def _initialize_classification_keys(self):
        """Initialize encryption keys for each classification level"""
        for classification in DataClassification:
            key_id = f"data_key_{classification.value}"
            
            # Check if key already exists
            existing_key = None
            for existing_key_id, key_info in self.crypto.keys.items():
                if (key_info.key_type == KeyType.DATA_ENCRYPTION_KEY and 
                    key_info.metadata and 
                    key_info.metadata.get('classification') == classification.value):
                    existing_key = existing_key_id
                    break
            
            if existing_key:
                self.classification_keys[classification] = existing_key
            else:
                # Create new key for this classification
                new_key_id = self.crypto.generate_key(
                    KeyType.DATA_ENCRYPTION_KEY,
                    EncryptionAlgorithm.AES_256_GCM,
                    expires_in_days=self.config['encryption']['key_rotation_days']
                )
                
                # Add classification metadata
                self.crypto.keys[new_key_id].metadata = {'classification': classification.value}
                self.crypto._save_keys()
                
                self.classification_keys[classification] = new_key_id
        
        logger.info(f"Initialized encryption keys for {len(self.classification_keys)} classification levels")
    
    def store_data(self, data: Union[str, bytes, Dict], 
                   classification: DataClassification,
                   category: DataCategory,
                   owner_credentials: AuthenticationCredentials,
                   tags: List[str] = None,
                   metadata: Dict[str, Any] = None,
                   expires_in_days: int = None) -> str:
        """
        Store data securely with classification-based encryption.
        
        Args:
            data: Data to store (string, bytes, or dict)
            classification: Data classification level
            category: Data category
            owner_credentials: Owner's credentials
            tags: Optional tags for categorization
            metadata: Optional metadata
            expires_in_days: Optional expiration in days
            
        Returns:
            Record ID of stored data
        """
        # Validate access permissions
        if not self._check_store_permission(classification, category, owner_credentials):
            raise PermissionError(f"Insufficient permissions to store {classification.value} data")
        
        # Generate record ID
        record_id = f"rec_{int(time.time())}_{self.crypto.generate_secure_token(16)}"
        
        # Prepare data for storage
        if isinstance(data, dict):
            data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        # Compress if enabled
        if self.config['encryption']['compress_before_encrypt']:
            data_bytes = gzip.compress(data_bytes)
        
        # DLP scanning
        if self.config['dlp']['enabled']:
            dlp_result = self._scan_for_sensitive_data(data_bytes, classification)
            if dlp_result['violations']:
                if self.config['dlp']['quarantine_violations']:
                    classification = DataClassification.RESTRICTED
                    logger.warning(f"DLP violations detected, upgrading classification to RESTRICTED")
        
        # Encrypt data
        encryption_key_id = self.classification_keys[classification]
        encryption_result = self.crypto.encrypt_data(data_bytes, encryption_key_id)
        
        # Calculate integrity hash
        integrity_hash = hashlib.sha256(data_bytes).hexdigest()
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        # Create data record
        record = DataRecord(
            record_id=record_id,
            classification=classification,
            category=category,
            owner_id=owner_credentials.user_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            expires_at=expires_at,
            status=DataStatus.ACTIVE,
            encryption_key_id=encryption_key_id,
            integrity_hash=integrity_hash,
            access_count=0,
            last_accessed=None,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Store encrypted data to file
        file_path = self._get_data_file_path(record_id)
        self._write_encrypted_file(file_path, encryption_result)
        
        # Store record metadata in database
        self._store_record_metadata(record, file_path, len(data_bytes))
        
        # Log data storage event
        self.audit_logger.log_data_access(
            owner_credentials.user_id,
            owner_credentials.username,
            record_id,
            "store",
            "system",
            True,
            classification.value,
            1
        )
        
        logger.info(f"Stored {classification.value} data record: {record_id}")
        return record_id
    
    def retrieve_data(self, record_id: str, 
                     user_credentials: AuthenticationCredentials,
                     justification: str = "") -> Any:
        """
        Retrieve and decrypt data with access control.
        
        Args:
            record_id: Record ID to retrieve
            user_credentials: User's credentials
            justification: Justification for access
            
        Returns:
            Decrypted data
        """
        # Get record metadata
        record = self._get_record_metadata(record_id)
        if not record:
            raise ValueError(f"Record not found: {record_id}")
        
        # Check if record is active
        if record.status != DataStatus.ACTIVE:
            raise ValueError(f"Record is not active: {record.status.value}")
        
        # Check expiration
        if record.expires_at and datetime.now() > record.expires_at:
            raise ValueError("Record has expired")
        
        # Check access permissions
        access_granted, requires_approval = self._check_access_permission(
            record, AccessOperation.READ, user_credentials
        )
        
        if requires_approval:
            # Create access request
            request_id = self._create_access_request(
                record_id, AccessOperation.READ, user_credentials, justification
            )
            raise PermissionError(f"Access requires approval. Request ID: {request_id}")
        
        if not access_granted:
            # Log failed access attempt
            self._log_access_attempt(record_id, user_credentials, AccessOperation.READ, False, "Access denied")
            raise PermissionError("Access denied")
        
        try:
            # Read encrypted file
            file_path = self._get_data_file_path(record_id)
            encryption_result = self._read_encrypted_file(file_path)
            
            # Decrypt data
            decrypted_data = self.crypto.decrypt_data(encryption_result)
            
            # Decompress if needed
            if self.config['encryption']['compress_before_encrypt']:
                try:
                    decrypted_data = gzip.decompress(decrypted_data)
                except:
                    pass  # Data might not be compressed
            
            # Verify integrity
            current_hash = hashlib.sha256(decrypted_data).hexdigest()
            if current_hash != record.integrity_hash:
                logger.error(f"Integrity check failed for record {record_id}")
                raise ValueError("Data integrity verification failed")
            
            # Update access statistics
            self._update_access_statistics(record_id)
            
            # Log successful access
            self._log_access_attempt(record_id, user_credentials, AccessOperation.READ, True, "")
            
            # Try to parse as JSON, fallback to string, then bytes
            try:
                return json.loads(decrypted_data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                try:
                    return decrypted_data.decode('utf-8')
                except UnicodeDecodeError:
                    return decrypted_data
        
        except Exception as e:
            # Log failed access attempt
            self._log_access_attempt(record_id, user_credentials, AccessOperation.READ, False, str(e))
            raise
    
    def update_data(self, record_id: str, new_data: Union[str, bytes, Dict],
                   user_credentials: AuthenticationCredentials,
                   justification: str = "") -> bool:
        """
        Update existing data record.
        
        Args:
            record_id: Record ID to update
            new_data: New data content
            user_credentials: User's credentials
            justification: Justification for update
            
        Returns:
            True if successful
        """
        # Get record metadata
        record = self._get_record_metadata(record_id)
        if not record:
            raise ValueError(f"Record not found: {record_id}")
        
        # Check access permissions
        access_granted, requires_approval = self._check_access_permission(
            record, AccessOperation.WRITE, user_credentials
        )
        
        if requires_approval:
            request_id = self._create_access_request(
                record_id, AccessOperation.WRITE, user_credentials, justification
            )
            raise PermissionError(f"Update requires approval. Request ID: {request_id}")
        
        if not access_granted:
            self._log_access_attempt(record_id, user_credentials, AccessOperation.WRITE, False, "Access denied")
            raise PermissionError("Access denied")
        
        try:
            # Prepare new data
            if isinstance(new_data, dict):
                data_bytes = json.dumps(new_data, sort_keys=True).encode('utf-8')
            elif isinstance(new_data, str):
                data_bytes = new_data.encode('utf-8')
            else:
                data_bytes = new_data
            
            # Compress if enabled
            if self.config['encryption']['compress_before_encrypt']:
                data_bytes = gzip.compress(data_bytes)
            
            # Encrypt new data
            encryption_result = self.crypto.encrypt_data(data_bytes, record.encryption_key_id)
            
            # Calculate new integrity hash
            new_integrity_hash = hashlib.sha256(data_bytes).hexdigest()
            
            # Update file
            file_path = self._get_data_file_path(record_id)
            self._write_encrypted_file(file_path, encryption_result)
            
            # Update record metadata
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE data_records 
                    SET updated_at = ?, integrity_hash = ?, file_size = ?
                    WHERE record_id = ?
                ''', (
                    datetime.now().isoformat(),
                    new_integrity_hash,
                    len(data_bytes),
                    record_id
                ))
                conn.commit()
            
            # Log successful update
            self._log_access_attempt(record_id, user_credentials, AccessOperation.WRITE, True, "")
            
            logger.info(f"Updated data record: {record_id}")
            return True
        
        except Exception as e:
            self._log_access_attempt(record_id, user_credentials, AccessOperation.WRITE, False, str(e))
            raise
    
    def delete_data(self, record_id: str, 
                   user_credentials: AuthenticationCredentials,
                   justification: str = "",
                   secure_delete: bool = True) -> bool:
        """
        Delete data record securely.
        
        Args:
            record_id: Record ID to delete
            user_credentials: User's credentials
            justification: Justification for deletion
            secure_delete: Whether to perform secure deletion
            
        Returns:
            True if successful
        """
        # Get record metadata
        record = self._get_record_metadata(record_id)
        if not record:
            raise ValueError(f"Record not found: {record_id}")
        
        # Check access permissions
        access_granted, requires_approval = self._check_access_permission(
            record, AccessOperation.DELETE, user_credentials
        )
        
        if requires_approval:
            request_id = self._create_access_request(
                record_id, AccessOperation.DELETE, user_credentials, justification
            )
            raise PermissionError(f"Deletion requires approval. Request ID: {request_id}")
        
        if not access_granted:
            self._log_access_attempt(record_id, user_credentials, AccessOperation.DELETE, False, "Access denied")
            raise PermissionError("Access denied")
        
        try:
            # Update record status to deleted
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE data_records 
                    SET status = ?, updated_at = ?
                    WHERE record_id = ?
                ''', (
                    DataStatus.DELETED.value,
                    datetime.now().isoformat(),
                    record_id
                ))
                conn.commit()
            
            # Secure file deletion if requested
            if secure_delete:
                file_path = self._get_data_file_path(record_id)
                self._secure_delete_file(file_path)
            
            # Log successful deletion
            self._log_access_attempt(record_id, user_credentials, AccessOperation.DELETE, True, "")
            
            logger.info(f"Deleted data record: {record_id}")
            return True
        
        except Exception as e:
            self._log_access_attempt(record_id, user_credentials, AccessOperation.DELETE, False, str(e))
            raise
    
    def _check_store_permission(self, classification: DataClassification, 
                               category: DataCategory,
                               credentials: AuthenticationCredentials) -> bool:
        """Check if user can store data with given classification"""
        # Find applicable policy
        for policy in self.access_policies.values():
            if (policy.classification == classification and 
                policy.category == category and
                credentials.role in policy.allowed_roles and
                AccessOperation.WRITE in policy.allowed_operations):
                return True
        
        # Default permissions based on role
        if classification == DataClassification.PUBLIC:
            return True
        elif classification == DataClassification.INTERNAL:
            return credentials.role in [UserRole.OPERATOR, UserRole.ENGINEER, UserRole.ADMIN]
        elif classification == DataClassification.CONFIDENTIAL:
            return credentials.role in [UserRole.ENGINEER, UserRole.ADMIN]
        elif classification in [DataClassification.RESTRICTED, DataClassification.TOP_SECRET]:
            return credentials.role in [UserRole.ADMIN, UserRole.SECURITY_OFFICER]
        
        return False
    
    def _check_access_permission(self, record: DataRecord, 
                               operation: AccessOperation,
                               credentials: AuthenticationCredentials) -> Tuple[bool, bool]:
        """
        Check access permission for data record.
        
        Returns:
            Tuple of (access_granted, requires_approval)
        """
        # Owner always has access (except for restricted data)
        if (record.owner_id == credentials.user_id and 
            record.classification != DataClassification.RESTRICTED):
            return True, False
        
        # Find applicable policy
        applicable_policy = None
        for policy in self.access_policies.values():
            if (policy.classification == record.classification and 
                policy.category == record.category):
                applicable_policy = policy
                break
        
        if not applicable_policy:
            # Default deny
            return False, False
        
        # Check role permission
        if credentials.role not in applicable_policy.allowed_roles:
            return False, False
        
        # Check operation permission
        if operation not in applicable_policy.allowed_operations:
            return False, False
        
        # Check conditions
        if applicable_policy.conditions:
            if applicable_policy.conditions.get('mfa_required', False) and not credentials.mfa_verified:
                return False, False
        
        # Check time restrictions
        if applicable_policy.time_restrictions:
            if applicable_policy.time_restrictions.get('business_hours_only', False):
                current_hour = datetime.now().hour
                if current_hour < 8 or current_hour > 18:  # Outside business hours
                    return False, True  # Requires approval
        
        # Check if approval is required
        requires_approval = applicable_policy.approval_required
        
        return True, requires_approval
    
    def _create_access_request(self, record_id: str, operation: AccessOperation,
                              credentials: AuthenticationCredentials, justification: str) -> str:
        """Create access request for approval"""
        request_id = f"req_{int(time.time())}_{self.crypto.generate_secure_token(8)}"
        
        request = DataAccessRequest(
            request_id=request_id,
            user_credentials=credentials,
            record_id=record_id,
            operation=operation,
            justification=justification,
            requested_at=datetime.now()
        )
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO access_requests (
                    request_id, user_id, record_id, operation, justification, requested_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                request_id,
                credentials.user_id,
                record_id,
                operation.value,
                justification,
                datetime.now().isoformat()
            ))
            conn.commit()
        
        self.pending_requests[request_id] = request
        
        logger.info(f"Created access request: {request_id} for record {record_id}")
        return request_id
    
    def _scan_for_sensitive_data(self, data: bytes, classification: DataClassification) -> Dict[str, Any]:
        """Scan data for sensitive patterns (DLP)"""
        violations = []
        
        try:
            # Convert to string for pattern matching
            text = data.decode('utf-8', errors='ignore')
            
            # Common sensitive patterns
            patterns = {
                'credit_card': r'\\b(?:\\d{4}[-\\s]?){3}\\d{4}\\b',
                'ssn': r'\\b\\d{3}-\\d{2}-\\d{4}\\b',
                'email': r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b',
                'phone': r'\\b\\d{3}-\\d{3}-\\d{4}\\b',
                'api_key': r'\\b[A-Za-z0-9]{32,}\\b'
            }
            
            import re
            for pattern_name, pattern in patterns.items():
                matches = re.findall(pattern, text)
                if matches:
                    violations.append({
                        'type': pattern_name,
                        'count': len(matches),
                        'severity': 'high' if pattern_name in ['credit_card', 'ssn'] else 'medium'
                    })
        
        except Exception as e:
            logger.warning(f"DLP scanning failed: {e}")
        
        return {
            'violations': violations,
            'scan_timestamp': datetime.now().isoformat()
        }
    
    def _get_data_file_path(self, record_id: str) -> Path:
        """Get file path for data record"""
        # Use first 2 characters of record ID for directory structure
        subdir = record_id[4:6] if len(record_id) > 6 else "00"
        dir_path = self.storage_path / "data" / subdir
        dir_path.mkdir(exist_ok=True)
        return dir_path / f"{record_id}.enc"
    
    def _write_encrypted_file(self, file_path: Path, encryption_result: EncryptionResult):
        """Write encrypted data to file"""
        # Create file metadata
        file_metadata = {
            'algorithm': encryption_result.algorithm.value,
            'key_id': encryption_result.key_id,
            'nonce': base64.b64encode(encryption_result.nonce).decode() if encryption_result.nonce else None,
            'tag': base64.b64encode(encryption_result.tag).decode() if encryption_result.tag else None,
            'metadata': encryption_result.metadata
        }
        
        # Write metadata and encrypted data
        with open(file_path, 'wb') as f:
            # Write metadata length (4 bytes)
            metadata_json = json.dumps(file_metadata).encode('utf-8')
            f.write(len(metadata_json).to_bytes(4, 'big'))
            
            # Write metadata
            f.write(metadata_json)
            
            # Write encrypted data
            f.write(encryption_result.ciphertext)
        
        # Set restrictive permissions
        os.chmod(file_path, 0o600)
    
    def _read_encrypted_file(self, file_path: Path) -> EncryptionResult:
        """Read encrypted data from file"""
        with open(file_path, 'rb') as f:
            # Read metadata length
            metadata_length = int.from_bytes(f.read(4), 'big')
            
            # Read metadata
            metadata_json = f.read(metadata_length)
            file_metadata = json.loads(metadata_json.decode('utf-8'))
            
            # Read encrypted data
            ciphertext = f.read()
        
        # Reconstruct EncryptionResult
        return EncryptionResult(
            ciphertext=ciphertext,
            nonce=base64.b64decode(file_metadata['nonce']) if file_metadata['nonce'] else None,
            tag=base64.b64decode(file_metadata['tag']) if file_metadata['tag'] else None,
            algorithm=EncryptionAlgorithm(file_metadata['algorithm']),
            key_id=file_metadata['key_id'],
            metadata=file_metadata['metadata']
        )
    
    def _store_record_metadata(self, record: DataRecord, file_path: Path, file_size: int):
        """Store record metadata in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO data_records (
                    record_id, classification, category, owner_id, created_at, updated_at,
                    expires_at, status, encryption_key_id, integrity_hash, access_count,
                    last_accessed, tags, metadata, file_path, file_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.record_id,
                record.classification.value,
                record.category.value,
                record.owner_id,
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
                record.expires_at.isoformat() if record.expires_at else None,
                record.status.value,
                record.encryption_key_id,
                record.integrity_hash,
                record.access_count,
                record.last_accessed.isoformat() if record.last_accessed else None,
                json.dumps(record.tags),
                json.dumps(record.metadata),
                str(file_path),
                file_size
            ))
            conn.commit()
    
    def _get_record_metadata(self, record_id: str) -> Optional[DataRecord]:
        """Get record metadata from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM data_records WHERE record_id = ?
            ''', (record_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return DataRecord(
                record_id=row[1],
                classification=DataClassification(row[2]),
                category=DataCategory(row[3]),
                owner_id=row[4],
                created_at=datetime.fromisoformat(row[5]),
                updated_at=datetime.fromisoformat(row[6]),
                expires_at=datetime.fromisoformat(row[7]) if row[7] else None,
                status=DataStatus(row[8]),
                encryption_key_id=row[9],
                integrity_hash=row[10],
                access_count=row[11],
                last_accessed=datetime.fromisoformat(row[12]) if row[12] else None,
                tags=json.loads(row[13]) if row[13] else [],
                metadata=json.loads(row[14]) if row[14] else {}
            )
    
    def _update_access_statistics(self, record_id: str):
        """Update access statistics for record"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE data_records 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE record_id = ?
            ''', (datetime.now().isoformat(), record_id))
            conn.commit()
    
    def _log_access_attempt(self, record_id: str, credentials: AuthenticationCredentials,
                           operation: AccessOperation, success: bool, details: str):
        """Log data access attempt"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO access_log (
                    record_id, user_id, username, operation, timestamp, success, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                record_id,
                credentials.user_id,
                credentials.username,
                operation.value,
                datetime.now().isoformat(),
                success,
                details
            ))
            conn.commit()
    
    def _secure_delete_file(self, file_path: Path):
        """Securely delete file by overwriting with random data"""
        if not file_path.exists():
            return
        
        try:
            # Get file size
            file_size = file_path.stat().st_size
            
            # Overwrite with random data multiple times
            with open(file_path, 'r+b') as f:
                for _ in range(3):  # 3 passes
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            # Delete the file
            file_path.unlink()
            
        except Exception as e:
            logger.error(f"Secure file deletion failed: {e}")
            # Fallback to regular deletion
            try:
                file_path.unlink()
            except:
                pass
    
    def search_records(self, filters: Dict[str, Any], 
                      user_credentials: AuthenticationCredentials,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search data records with filters.
        
        Args:
            filters: Search filters
            user_credentials: User's credentials
            limit: Maximum results
            
        Returns:
            List of record metadata (without sensitive data)
        """
        query = 'SELECT * FROM data_records WHERE status = ?'
        params = [DataStatus.ACTIVE.value]
        
        # Apply filters
        if 'classification' in filters:
            query += ' AND classification = ?'
            params.append(filters['classification'])
        
        if 'category' in filters:
            query += ' AND category = ?'
            params.append(filters['category'])
        
        if 'owner_id' in filters:
            query += ' AND owner_id = ?'
            params.append(filters['owner_id'])
        
        if 'tags' in filters:
            query += ' AND tags LIKE ?'
            params.append(f'%{filters["tags"]}%')
        
        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            
            for row in cursor.fetchall():
                record = DataRecord(
                    record_id=row[1],
                    classification=DataClassification(row[2]),
                    category=DataCategory(row[3]),
                    owner_id=row[4],
                    created_at=datetime.fromisoformat(row[5]),
                    updated_at=datetime.fromisoformat(row[6]),
                    expires_at=datetime.fromisoformat(row[7]) if row[7] else None,
                    status=DataStatus(row[8]),
                    encryption_key_id=row[9],
                    integrity_hash=row[10],
                    access_count=row[11],
                    last_accessed=datetime.fromisoformat(row[12]) if row[12] else None,
                    tags=json.loads(row[13]) if row[13] else [],
                    metadata=json.loads(row[14]) if row[14] else {}
                )
                
                # Check if user can see this record
                access_granted, _ = self._check_access_permission(
                    record, AccessOperation.READ, user_credentials
                )
                
                if access_granted:
                    # Return metadata without sensitive information
                    record_info = record.to_dict()
                    record_info.pop('encryption_key_id', None)
                    record_info.pop('integrity_hash', None)
                    results.append(record_info)
        
        return results
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get data management statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Total records by classification
            cursor = conn.execute('''
                SELECT classification, COUNT(*) FROM data_records 
                WHERE status = ? GROUP BY classification
            ''', (DataStatus.ACTIVE.value,))
            records_by_classification = dict(cursor.fetchall())
            
            # Total records by category
            cursor = conn.execute('''
                SELECT category, COUNT(*) FROM data_records 
                WHERE status = ? GROUP BY category
            ''', (DataStatus.ACTIVE.value,))
            records_by_category = dict(cursor.fetchall())
            
            # Storage usage
            cursor = conn.execute('''
                SELECT SUM(file_size) FROM data_records WHERE status = ?
            ''', (DataStatus.ACTIVE.value,))
            total_storage = cursor.fetchone()[0] or 0
            
            # Recent access activity
            yesterday = (datetime.now() - timedelta(hours=24)).isoformat()
            cursor = conn.execute('''
                SELECT COUNT(*) FROM access_log WHERE timestamp > ?
            ''', (yesterday,))
            recent_access_count = cursor.fetchone()[0]
            
            # Pending requests
            cursor = conn.execute('''
                SELECT COUNT(*) FROM access_requests WHERE approved = FALSE
            ''')
            pending_requests = cursor.fetchone()[0]
        
        return {
            'total_active_records': sum(records_by_classification.values()),
            'records_by_classification': records_by_classification,
            'records_by_category': records_by_category,
            'total_storage_bytes': total_storage,
            'recent_access_count_24h': recent_access_count,
            'pending_access_requests': pending_requests,
            'classification_keys_count': len(self.classification_keys)
        }


# Example usage and testing
if __name__ == "__main__":
    print("Secure Data Manager Test Suite")
    print("=" * 50)
    
    from cryptographic_manager import CryptographicManager
    from audit_logger import AuditLogger
    from authentication_manager import AuthenticationManager, AuthenticationCredentials, UserRole
    
    # Initialize components
    crypto = CryptographicManager()
    audit = AuditLogger(crypto)
    auth = AuthenticationManager(crypto)
    data_manager = SecureDataManager(crypto, audit)
    
    # Create test credentials
    test_credentials = AuthenticationCredentials(
        user_id="test_user_001",
        username="testuser",
        role=UserRole.ENGINEER,
        permissions=["read", "write", "configure"],
        session_token="test_session_token",
        expires_at=datetime.now() + timedelta(hours=1),
        mfa_verified=True
    )
    
    # Test data storage
    print("\\n1. Testing Data Storage...")
    test_data = {
        "inspection_id": "INS_001",
        "turbine_id": "TURB_001",
        "defects": [
            {"type": "crack", "severity": 7.5, "location": "blade_tip"},
            {"type": "corrosion", "severity": 4.2, "location": "hub"}
        ],
        "timestamp": datetime.now().isoformat(),
        "inspector": "John Doe"
    }
    
    try:
        record_id = data_manager.store_data(
            data=test_data,
            classification=DataClassification.CONFIDENTIAL,
            category=DataCategory.INSPECTION_RESULTS,
            owner_credentials=test_credentials,
            tags=["inspection", "turbine", "defects"],
            metadata={"source": "automated_inspection", "version": "1.0"}
        )
        print(f"✅ Data stored successfully: {record_id}")
    except Exception as e:
        print(f"❌ Data storage failed: {e}")
        record_id = None
    
    # Test data retrieval
    if record_id:
        print("\\n2. Testing Data Retrieval...")
        try:
            retrieved_data = data_manager.retrieve_data(
                record_id=record_id,
                user_credentials=test_credentials,
                justification="Testing data retrieval functionality"
            )
            print(f"✅ Data retrieved successfully")
            print(f"   Inspection ID: {retrieved_data.get('inspection_id')}")
            print(f"   Defects found: {len(retrieved_data.get('defects', []))}")
        except Exception as e:
            print(f"❌ Data retrieval failed: {e}")
    
    # Test search functionality
    print("\\n3. Testing Data Search...")
    try:
        search_results = data_manager.search_records(
            filters={"classification": "confidential", "category": "inspection_results"},
            user_credentials=test_credentials,
            limit=10
        )
        print(f"✅ Found {len(search_results)} matching records")
        for result in search_results[:3]:  # Show first 3
            print(f"   Record: {result['record_id'][:16]}... ({result['classification']})")
    except Exception as e:
        print(f"❌ Data search failed: {e}")
    
    # Test statistics
    print("\\n4. Data Management Statistics:")
    try:
        stats = data_manager.get_data_statistics()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"     {k}: {v}")
            else:
                print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ Statistics retrieval failed: {e}")
    
    print("\\n" + "=" * 50)
    print("Secure data manager test suite completed")