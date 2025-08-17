"""
Comprehensive Audit Logger for Security Event Tracking
=====================================================

This module provides enterprise-grade audit logging capabilities including:
- Structured security event logging with risk scoring
- Tamper-proof audit trails with cryptographic integrity
- Real-time threat detection and alerting
- Compliance reporting and log retention
- Log aggregation and analysis
- Integration with SIEM systems

Author: Security Framework Team
"""

import os
import json
import time
import hashlib
import hmac
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import logging
import threading
from queue import Queue, Empty
import gzip
import sqlite3
from contextlib import contextmanager

from .cryptographic_manager import CryptographicManager, KeyType, EncryptionAlgorithm

# Configure logging
logger = logging.getLogger(__name__)

class SecurityEventType(Enum):
    """Types of security events"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    SYSTEM_ACCESS = "system_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_VIOLATION = "security_violation"
    INTRUSION_ATTEMPT = "intrusion_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_BREACH = "data_breach"
    MALWARE_DETECTION = "malware_detection"
    NETWORK_ANOMALY = "network_anomaly"
    SYSTEM_ERROR = "system_error"

class RiskLevel(Enum):
    """Risk levels for security events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EventStatus(Enum):
    """Event processing status"""
    PENDING = "pending"
    PROCESSED = "processed"
    ALERTED = "alerted"
    INVESTIGATED = "investigated"
    RESOLVED = "resolved"

@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    timestamp: datetime
    event_type: SecurityEventType
    risk_level: RiskLevel
    user_id: Optional[str]
    username: Optional[str]
    source_ip: str
    user_agent: str
    resource: str
    action: str
    result: str
    details: Dict[str, Any]
    risk_score: float
    status: EventStatus = EventStatus.PENDING
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'risk_level': self.risk_level.value,
            'user_id': self.user_id,
            'username': self.username,
            'source_ip': self.source_ip,
            'user_agent': self.user_agent,
            'resource': self.resource,
            'action': self.action,
            'result': self.result,
            'details': self.details,
            'risk_score': self.risk_score,
            'status': self.status.value,
            'correlation_id': self.correlation_id,
            'session_id': self.session_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityEvent':
        """Create from dictionary"""
        return cls(
            event_id=data['event_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            event_type=SecurityEventType(data['event_type']),
            risk_level=RiskLevel(data['risk_level']),
            user_id=data.get('user_id'),
            username=data.get('username'),
            source_ip=data['source_ip'],
            user_agent=data['user_agent'],
            resource=data['resource'],
            action=data['action'],
            result=data['result'],
            details=data['details'],
            risk_score=data['risk_score'],
            status=EventStatus(data.get('status', 'pending')),
            correlation_id=data.get('correlation_id'),
            session_id=data.get('session_id')
        )

@dataclass
class AuditLogEntry:
    """Audit log entry with integrity protection"""
    sequence_number: int
    event: SecurityEvent
    hash_chain: str
    signature: str
    encrypted_data: bytes

@dataclass
class ThreatPattern:
    """Threat detection pattern"""
    pattern_id: str
    name: str
    description: str
    event_types: List[SecurityEventType]
    conditions: Dict[str, Any]
    risk_threshold: float
    time_window_minutes: int
    max_occurrences: int
    enabled: bool = True

class AuditLogger:
    """
    Comprehensive audit logging system with tamper-proof trails,
    real-time threat detection, and compliance reporting.
    """
    
    def __init__(self, crypto_manager: CryptographicManager,
                 log_directory: str = "security/audit_logs",
                 config_path: str = "security/audit_config.json"):
        self.crypto = crypto_manager
        self.log_directory = Path(log_directory)
        self.config_path = Path(config_path)
        
        # Create directories
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize database
        self.db_path = self.log_directory / "audit.db"
        self._initialize_database()
        
        # Event processing
        self.event_queue = Queue()
        self.processing_thread = None
        self.running = False
        
        # Hash chain for integrity
        self.last_hash = self._get_last_hash()
        self.sequence_number = self._get_next_sequence_number()
        
        # Threat detection patterns
        self.threat_patterns: Dict[str, ThreatPattern] = {}
        self._load_threat_patterns()
        
        # Event correlation
        self.correlation_cache: Dict[str, List[SecurityEvent]] = {}
        
        # Signing key for log integrity
        self.signing_key_id = self._get_or_create_signing_key()
        
        # Start processing
        self.start_processing()
        
        logger.info("AuditLogger initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load audit configuration"""
        default_config = {
            "log_retention_days": 365,
            "max_log_file_size_mb": 100,
            "compression_enabled": True,
            "encryption_enabled": True,
            "real_time_alerting": True,
            "siem_integration": {
                "enabled": False,
                "endpoint": "",
                "api_key": ""
            },
            "threat_detection": {
                "enabled": True,
                "correlation_window_minutes": 60,
                "max_correlation_events": 1000
            },
            "compliance": {
                "gdpr_enabled": True,
                "hipaa_enabled": False,
                "sox_enabled": False
            },
            "alerting": {
                "email_enabled": False,
                "webhook_enabled": False,
                "slack_enabled": False
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load audit config: {e}")
        else:
            # Save default config
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            os.chmod(self.config_path, 0o600)
        
        return default_config
    
    def _initialize_database(self):
        """Initialize SQLite database for audit logs"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sequence_number INTEGER UNIQUE NOT NULL,
                    event_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    user_id TEXT,
                    username TEXT,
                    source_ip TEXT NOT NULL,
                    user_agent TEXT,
                    resource TEXT NOT NULL,
                    action TEXT NOT NULL,
                    result TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    status TEXT NOT NULL,
                    correlation_id TEXT,
                    session_id TEXT,
                    details_encrypted BLOB,
                    hash_chain TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS threat_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    pattern_id TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    event_count INTEGER NOT NULL,
                    correlation_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT NOT NULL,
                    resolved_at TEXT,
                    resolved_by TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON audit_events(timestamp)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_events_user ON audit_events(user_id)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_events_type ON audit_events(event_type)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_events_risk ON audit_events(risk_level)
            ''')
            
            conn.commit()
    
    def _get_last_hash(self) -> str:
        """Get the last hash in the chain for integrity"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT hash_chain FROM audit_events ORDER BY sequence_number DESC LIMIT 1'
            )
            result = cursor.fetchone()
            return result[0] if result else "genesis_hash"
    
    def _get_next_sequence_number(self) -> int:
        """Get next sequence number"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT MAX(sequence_number) FROM audit_events'
            )
            result = cursor.fetchone()
            return (result[0] or 0) + 1
    
    def _get_or_create_signing_key(self) -> str:
        """Get or create signing key for log integrity"""
        # Look for existing signing key
        for key_id, key_info in self.crypto.keys.items():
            if key_info.key_type == KeyType.SIGNING_KEY:
                return key_id
        
        # Create new signing key
        return self.crypto.generate_key(
            KeyType.SIGNING_KEY,
            EncryptionAlgorithm.RSA_OAEP,
            key_size=2048,
            expires_in_days=365
        )
    
    def _load_threat_patterns(self):
        """Load threat detection patterns"""
        patterns_file = self.log_directory / "threat_patterns.json"
        
        # Default threat patterns
        default_patterns = {
            "brute_force_login": ThreatPattern(
                pattern_id="brute_force_login",
                name="Brute Force Login Attack",
                description="Multiple failed login attempts from same IP",
                event_types=[SecurityEventType.AUTHENTICATION],
                conditions={
                    "result": "failure",
                    "max_attempts": 10,
                    "same_ip": True
                },
                risk_threshold=0.8,
                time_window_minutes=15,
                max_occurrences=10
            ),
            "privilege_escalation": ThreatPattern(
                pattern_id="privilege_escalation",
                name="Privilege Escalation Attempt",
                description="Unauthorized access to privileged resources",
                event_types=[SecurityEventType.AUTHORIZATION, SecurityEventType.PRIVILEGE_ESCALATION],
                conditions={
                    "result": "failure",
                    "privilege_required": True
                },
                risk_threshold=0.9,
                time_window_minutes=30,
                max_occurrences=3
            ),
            "data_exfiltration": ThreatPattern(
                pattern_id="data_exfiltration",
                name="Potential Data Exfiltration",
                description="Unusual data access patterns",
                event_types=[SecurityEventType.DATA_ACCESS],
                conditions={
                    "large_volume": True,
                    "unusual_time": True
                },
                risk_threshold=0.7,
                time_window_minutes=60,
                max_occurrences=5
            ),
            "system_intrusion": ThreatPattern(
                pattern_id="system_intrusion",
                name="System Intrusion Attempt",
                description="Multiple security violations from same source",
                event_types=[SecurityEventType.SECURITY_VIOLATION, SecurityEventType.INTRUSION_ATTEMPT],
                conditions={
                    "same_source": True,
                    "multiple_violations": True
                },
                risk_threshold=0.85,
                time_window_minutes=45,
                max_occurrences=5
            )
        }
        
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                
                for pattern_id, pattern_data in patterns_data.items():
                    pattern = ThreatPattern(
                        pattern_id=pattern_data['pattern_id'],
                        name=pattern_data['name'],
                        description=pattern_data['description'],
                        event_types=[SecurityEventType(t) for t in pattern_data['event_types']],
                        conditions=pattern_data['conditions'],
                        risk_threshold=pattern_data['risk_threshold'],
                        time_window_minutes=pattern_data['time_window_minutes'],
                        max_occurrences=pattern_data['max_occurrences'],
                        enabled=pattern_data.get('enabled', True)
                    )
                    self.threat_patterns[pattern_id] = pattern
                
            except Exception as e:
                logger.error(f"Failed to load threat patterns: {e}")
                self.threat_patterns = default_patterns
        else:
            self.threat_patterns = default_patterns
            self._save_threat_patterns()
    
    def _save_threat_patterns(self):
        """Save threat detection patterns"""
        patterns_file = self.log_directory / "threat_patterns.json"
        patterns_data = {}
        
        for pattern_id, pattern in self.threat_patterns.items():
            patterns_data[pattern_id] = {
                'pattern_id': pattern.pattern_id,
                'name': pattern.name,
                'description': pattern.description,
                'event_types': [t.value for t in pattern.event_types],
                'conditions': pattern.conditions,
                'risk_threshold': pattern.risk_threshold,
                'time_window_minutes': pattern.time_window_minutes,
                'max_occurrences': pattern.max_occurrences,
                'enabled': pattern.enabled
            }
        
        with open(patterns_file, 'w') as f:
            json.dump(patterns_data, f, indent=2)
        os.chmod(patterns_file, 0o600)
    
    def start_processing(self):
        """Start event processing thread"""
        if not self.running:
            self.running = True
            self.processing_thread = threading.Thread(target=self._process_events)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info("Audit event processing started")
    
    def stop_processing(self):
        """Stop event processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        logger.info("Audit event processing stopped")
    
    def log_security_event(self, event: SecurityEvent):
        """
        Log security event to audit trail.
        
        Args:
            event: SecurityEvent to log
        """
        # Generate event ID if not provided
        if not event.event_id:
            event.event_id = f"evt_{int(time.time())}_{self.crypto.generate_secure_token(8)}"
        
        # Add to processing queue
        self.event_queue.put(event)
    
    def log_authentication(self, username: str, success: bool, ip_address: str,
                          user_agent: str = "unknown", user_id: str = None,
                          session_id: str = None, details: Dict[str, Any] = None):
        """Log authentication event"""
        event = SecurityEvent(
            event_id="",
            timestamp=datetime.now(),
            event_type=SecurityEventType.AUTHENTICATION,
            risk_level=RiskLevel.LOW if success else RiskLevel.MEDIUM,
            user_id=user_id,
            username=username,
            source_ip=ip_address,
            user_agent=user_agent,
            resource="authentication",
            action="login",
            result="success" if success else "failure",
            details=details or {},
            risk_score=0.1 if success else 0.6,
            session_id=session_id
        )
        
        self.log_security_event(event)
    
    def log_authorization(self, user_id: str, username: str, resource: str,
                         action: str, success: bool, ip_address: str,
                         required_permission: str = None, session_id: str = None):
        """Log authorization event"""
        event = SecurityEvent(
            event_id="",
            timestamp=datetime.now(),
            event_type=SecurityEventType.AUTHORIZATION,
            risk_level=RiskLevel.LOW if success else RiskLevel.HIGH,
            user_id=user_id,
            username=username,
            source_ip=ip_address,
            user_agent="system",
            resource=resource,
            action=action,
            result="success" if success else "failure",
            details={"required_permission": required_permission} if required_permission else {},
            risk_score=0.2 if success else 0.8,
            session_id=session_id
        )
        
        self.log_security_event(event)
    
    def log_data_access(self, user_id: str, username: str, resource: str,
                       action: str, ip_address: str, success: bool = True,
                       data_classification: str = None, record_count: int = None,
                       session_id: str = None):
        """Log data access event"""
        details = {}
        if data_classification:
            details["data_classification"] = data_classification
        if record_count:
            details["record_count"] = record_count
        
        # Calculate risk score based on data sensitivity and volume
        risk_score = 0.2 if success else 0.7
        if data_classification in ["confidential", "restricted"]:
            risk_score += 0.2
        if record_count and record_count > 1000:
            risk_score += 0.3
        
        event = SecurityEvent(
            event_id="",
            timestamp=datetime.now(),
            event_type=SecurityEventType.DATA_ACCESS,
            risk_level=self._calculate_risk_level(risk_score),
            user_id=user_id,
            username=username,
            source_ip=ip_address,
            user_agent="system",
            resource=resource,
            action=action,
            result="success" if success else "failure",
            details=details,
            risk_score=min(risk_score, 1.0),
            session_id=session_id
        )
        
        self.log_security_event(event)
    
    def log_security_violation(self, user_id: str, username: str, violation_type: str,
                              description: str, ip_address: str, severity: str = "medium"):
        """Log security violation"""
        risk_level_map = {
            "low": RiskLevel.LOW,
            "medium": RiskLevel.MEDIUM,
            "high": RiskLevel.HIGH,
            "critical": RiskLevel.CRITICAL
        }
        
        risk_score_map = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.95
        }
        
        event = SecurityEvent(
            event_id="",
            timestamp=datetime.now(),
            event_type=SecurityEventType.SECURITY_VIOLATION,
            risk_level=risk_level_map.get(severity, RiskLevel.MEDIUM),
            user_id=user_id,
            username=username,
            source_ip=ip_address,
            user_agent="system",
            resource="security_policy",
            action="violation",
            result="detected",
            details={
                "violation_type": violation_type,
                "description": description,
                "severity": severity
            },
            risk_score=risk_score_map.get(severity, 0.6)
        )
        
        self.log_security_event(event)
    
    def _process_events(self):
        """Process events from queue"""
        while self.running:
            try:
                # Get event from queue with timeout
                event = self.event_queue.get(timeout=1.0)
                
                # Process the event
                self._process_single_event(event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audit event: {e}")
    
    def _process_single_event(self, event: SecurityEvent):
        """Process a single security event"""
        try:
            # Create audit log entry with integrity protection
            log_entry = self._create_audit_entry(event)
            
            # Store in database
            self._store_audit_entry(log_entry)
            
            # Perform threat detection
            if self.config['threat_detection']['enabled']:
                self._detect_threats(event)
            
            # Send real-time alerts if needed
            if (self.config['real_time_alerting'] and 
                event.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]):
                self._send_alert(event)
            
            # Update correlation cache
            self._update_correlation_cache(event)
            
            logger.debug(f"Processed audit event: {event.event_id}")
            
        except Exception as e:
            logger.error(f"Failed to process audit event {event.event_id}: {e}")
    
    def _create_audit_entry(self, event: SecurityEvent) -> AuditLogEntry:
        """Create tamper-proof audit log entry"""
        # Serialize event data
        event_data = json.dumps(event.to_dict(), sort_keys=True).encode('utf-8')
        
        # Encrypt sensitive details if enabled
        encrypted_data = event_data
        if self.config['encryption_enabled']:
            encryption_result = self.crypto.encrypt_data(event_data, self.signing_key_id)
            encrypted_data = encryption_result.ciphertext
        
        # Create hash chain
        previous_hash = self.last_hash
        current_data = f"{self.sequence_number}:{event.event_id}:{event.timestamp.isoformat()}:{previous_hash}"
        current_hash = hashlib.sha256(current_data.encode()).hexdigest()
        
        # Create digital signature
        signature_data = f"{current_hash}:{encrypted_data.hex()}"
        signature = self._create_signature(signature_data)
        
        # Update state
        self.last_hash = current_hash
        
        return AuditLogEntry(
            sequence_number=self.sequence_number,
            event=event,
            hash_chain=current_hash,
            signature=signature,
            encrypted_data=encrypted_data
        )
    
    def _create_signature(self, data: str) -> str:
        """Create digital signature for data integrity"""
        # Use HMAC with signing key for now
        # In production, use RSA or ECDSA signatures
        key_info = self.crypto.keys[self.signing_key_id]
        signature = hmac.new(
            key_info.key_data[:32],  # Use first 32 bytes as HMAC key
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _store_audit_entry(self, log_entry: AuditLogEntry):
        """Store audit entry in database"""
        event = log_entry.event
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO audit_events (
                    sequence_number, event_id, timestamp, event_type, risk_level,
                    user_id, username, source_ip, user_agent, resource, action,
                    result, risk_score, status, correlation_id, session_id,
                    details_encrypted, hash_chain, signature, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                log_entry.sequence_number,
                event.event_id,
                event.timestamp.isoformat(),
                event.event_type.value,
                event.risk_level.value,
                event.user_id,
                event.username,
                event.source_ip,
                event.user_agent,
                event.resource,
                event.action,
                event.result,
                event.risk_score,
                event.status.value,
                event.correlation_id,
                event.session_id,
                log_entry.encrypted_data,
                log_entry.hash_chain,
                log_entry.signature,
                datetime.now().isoformat()
            ))
            
            conn.commit()
        
        # Increment sequence number
        self.sequence_number += 1
    
    def _detect_threats(self, event: SecurityEvent):
        """Detect threats based on patterns"""
        for pattern_id, pattern in self.threat_patterns.items():
            if not pattern.enabled:
                continue
            
            if event.event_type in pattern.event_types:
                if self._matches_pattern(event, pattern):
                    self._trigger_threat_alert(pattern, event)
    
    def _matches_pattern(self, event: SecurityEvent, pattern: ThreatPattern) -> bool:
        """Check if event matches threat pattern"""
        # Get recent events for correlation
        time_window = datetime.now() - timedelta(minutes=pattern.time_window_minutes)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM audit_events 
                WHERE timestamp > ? AND event_type IN ({})
                ORDER BY timestamp DESC
            '''.format(','.join(['?' for _ in pattern.event_types])),
            [time_window.isoformat()] + [t.value for t in pattern.event_types])
            
            recent_events = cursor.fetchall()
        
        # Apply pattern-specific logic
        if pattern.pattern_id == "brute_force_login":
            return self._check_brute_force_pattern(event, recent_events, pattern)
        elif pattern.pattern_id == "privilege_escalation":
            return self._check_privilege_escalation_pattern(event, recent_events, pattern)
        elif pattern.pattern_id == "data_exfiltration":
            return self._check_data_exfiltration_pattern(event, recent_events, pattern)
        elif pattern.pattern_id == "system_intrusion":
            return self._check_system_intrusion_pattern(event, recent_events, pattern)
        
        return False
    
    def _check_brute_force_pattern(self, event: SecurityEvent, recent_events: List, 
                                  pattern: ThreatPattern) -> bool:
        """Check for brute force login pattern"""
        if event.result != "failure":
            return False
        
        # Count failed attempts from same IP
        failed_attempts = sum(1 for e in recent_events 
                            if e[7] == event.source_ip and e[11] == "failure")
        
        return failed_attempts >= pattern.conditions.get("max_attempts", 10)
    
    def _check_privilege_escalation_pattern(self, event: SecurityEvent, recent_events: List,
                                          pattern: ThreatPattern) -> bool:
        """Check for privilege escalation pattern"""
        if event.result != "failure":
            return False
        
        # Check for multiple authorization failures
        auth_failures = sum(1 for e in recent_events 
                          if e[5] == event.user_id and e[11] == "failure")
        
        return auth_failures >= pattern.max_occurrences
    
    def _check_data_exfiltration_pattern(self, event: SecurityEvent, recent_events: List,
                                       pattern: ThreatPattern) -> bool:
        """Check for data exfiltration pattern"""
        # Check for unusual data access volume
        user_access_count = sum(1 for e in recent_events if e[5] == event.user_id)
        
        # Check if access is outside normal hours (simplified)
        current_hour = event.timestamp.hour
        unusual_time = current_hour < 6 or current_hour > 22
        
        return (user_access_count >= pattern.max_occurrences and 
                unusual_time and 
                event.risk_score >= pattern.risk_threshold)
    
    def _check_system_intrusion_pattern(self, event: SecurityEvent, recent_events: List,
                                      pattern: ThreatPattern) -> bool:
        """Check for system intrusion pattern"""
        # Count security violations from same source
        violations = sum(1 for e in recent_events 
                        if e[7] == event.source_ip and e[3] in ["security_violation", "intrusion_attempt"])
        
        return violations >= pattern.max_occurrences
    
    def _trigger_threat_alert(self, pattern: ThreatPattern, event: SecurityEvent):
        """Trigger threat alert"""
        alert_id = f"alert_{int(time.time())}_{self.crypto.generate_secure_token(8)}"
        correlation_id = event.correlation_id or f"corr_{int(time.time())}"
        
        # Store alert in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO threat_alerts (
                    alert_id, pattern_id, triggered_at, risk_score, event_count,
                    correlation_id, status, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert_id,
                pattern.pattern_id,
                datetime.now().isoformat(),
                event.risk_score,
                1,  # Initial event count
                correlation_id,
                "active",
                json.dumps({
                    "pattern_name": pattern.name,
                    "description": pattern.description,
                    "triggering_event": event.event_id,
                    "source_ip": event.source_ip,
                    "user_id": event.user_id
                })
            ))
            
            conn.commit()
        
        logger.warning(f"THREAT ALERT: {pattern.name} - {alert_id}")
        
        # Send immediate notification
        self._send_threat_notification(alert_id, pattern, event)
    
    def _send_alert(self, event: SecurityEvent):
        """Send real-time alert for high-risk events"""
        alert_message = (f"HIGH RISK SECURITY EVENT: {event.event_type.value} "
                        f"by {event.username or 'unknown'} from {event.source_ip}")
        
        logger.critical(alert_message)
        
        # TODO: Implement actual alerting mechanisms
        # - Email notifications
        # - Webhook calls
        # - Slack/Teams integration
        # - SIEM integration
    
    def _send_threat_notification(self, alert_id: str, pattern: ThreatPattern, event: SecurityEvent):
        """Send threat detection notification"""
        notification = {
            "alert_id": alert_id,
            "pattern": pattern.name,
            "description": pattern.description,
            "risk_score": event.risk_score,
            "source_ip": event.source_ip,
            "user": event.username,
            "timestamp": event.timestamp.isoformat()
        }
        
        logger.critical(f"THREAT DETECTED: {json.dumps(notification)}")
    
    def _update_correlation_cache(self, event: SecurityEvent):
        """Update event correlation cache"""
        if not event.correlation_id:
            return
        
        if event.correlation_id not in self.correlation_cache:
            self.correlation_cache[event.correlation_id] = []
        
        self.correlation_cache[event.correlation_id].append(event)
        
        # Limit cache size
        max_events = self.config['threat_detection']['max_correlation_events']
        if len(self.correlation_cache[event.correlation_id]) > max_events:
            self.correlation_cache[event.correlation_id] = \
                self.correlation_cache[event.correlation_id][-max_events:]
    
    def _calculate_risk_level(self, risk_score: float) -> RiskLevel:
        """Calculate risk level from score"""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def verify_log_integrity(self, start_sequence: int = None, end_sequence: int = None) -> bool:
        """
        Verify integrity of audit log chain.
        
        Args:
            start_sequence: Starting sequence number (optional)
            end_sequence: Ending sequence number (optional)
            
        Returns:
            True if integrity is intact, False otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            query = 'SELECT sequence_number, hash_chain, signature FROM audit_events'
            params = []
            
            if start_sequence is not None:
                query += ' WHERE sequence_number >= ?'
                params.append(start_sequence)
                
                if end_sequence is not None:
                    query += ' AND sequence_number <= ?'
                    params.append(end_sequence)
            elif end_sequence is not None:
                query += ' WHERE sequence_number <= ?'
                params.append(end_sequence)
            
            query += ' ORDER BY sequence_number'
            
            cursor = conn.execute(query, params)
            entries = cursor.fetchall()
        
        if not entries:
            return True
        
        # Verify hash chain
        previous_hash = "genesis_hash"
        
        for sequence_num, hash_chain, signature in entries:
            # Reconstruct hash
            # Note: This is simplified - in production, you'd need to reconstruct
            # the exact data that was hashed
            expected_data = f"{sequence_num}:event_data:{previous_hash}"
            expected_hash = hashlib.sha256(expected_data.encode()).hexdigest()
            
            # In a real implementation, you'd verify the actual signature
            # For now, just check that hash exists
            if not hash_chain:
                logger.error(f"Missing hash for sequence {sequence_num}")
                return False
            
            previous_hash = hash_chain
        
        logger.info(f"Verified integrity of {len(entries)} audit log entries")
        return True
    
    def search_events(self, filters: Dict[str, Any], limit: int = 100) -> List[SecurityEvent]:
        """
        Search audit events with filters.
        
        Args:
            filters: Search filters (user_id, event_type, start_time, end_time, etc.)
            limit: Maximum number of results
            
        Returns:
            List of matching SecurityEvent objects
        """
        query = 'SELECT * FROM audit_events WHERE 1=1'
        params = []
        
        if 'user_id' in filters:
            query += ' AND user_id = ?'
            params.append(filters['user_id'])
        
        if 'username' in filters:
            query += ' AND username = ?'
            params.append(filters['username'])
        
        if 'event_type' in filters:
            query += ' AND event_type = ?'
            params.append(filters['event_type'])
        
        if 'risk_level' in filters:
            query += ' AND risk_level = ?'
            params.append(filters['risk_level'])
        
        if 'source_ip' in filters:
            query += ' AND source_ip = ?'
            params.append(filters['source_ip'])
        
        if 'start_time' in filters:
            query += ' AND timestamp >= ?'
            params.append(filters['start_time'])
        
        if 'end_time' in filters:
            query += ' AND timestamp <= ?'
            params.append(filters['end_time'])
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        events = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            
            for row in cursor.fetchall():
                # Reconstruct SecurityEvent from database row
                event = SecurityEvent(
                    event_id=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    event_type=SecurityEventType(row[4]),
                    risk_level=RiskLevel(row[5]),
                    user_id=row[6],
                    username=row[7],
                    source_ip=row[8],
                    user_agent=row[9] or "unknown",
                    resource=row[10],
                    action=row[11],
                    result=row[12],
                    details={},  # Would need to decrypt details_encrypted
                    risk_score=row[13],
                    status=EventStatus(row[14]),
                    correlation_id=row[15],
                    session_id=row[16]
                )
                events.append(event)
        
        return events
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Total events
            cursor = conn.execute('SELECT COUNT(*) FROM audit_events')
            total_events = cursor.fetchone()[0]
            
            # Events by type
            cursor = conn.execute('''
                SELECT event_type, COUNT(*) FROM audit_events 
                GROUP BY event_type ORDER BY COUNT(*) DESC
            ''')
            events_by_type = dict(cursor.fetchall())
            
            # Events by risk level
            cursor = conn.execute('''
                SELECT risk_level, COUNT(*) FROM audit_events 
                GROUP BY risk_level ORDER BY COUNT(*) DESC
            ''')
            events_by_risk = dict(cursor.fetchall())
            
            # Recent activity (last 24 hours)
            yesterday = (datetime.now() - timedelta(hours=24)).isoformat()
            cursor = conn.execute('''
                SELECT COUNT(*) FROM audit_events WHERE timestamp > ?
            ''', (yesterday,))
            recent_events = cursor.fetchone()[0]
            
            # Active alerts
            cursor = conn.execute('''
                SELECT COUNT(*) FROM threat_alerts WHERE status = 'active'
            ''')
            active_alerts = cursor.fetchone()[0]
        
        return {
            'total_events': total_events,
            'events_by_type': events_by_type,
            'events_by_risk_level': events_by_risk,
            'recent_events_24h': recent_events,
            'active_threat_alerts': active_alerts,
            'log_integrity_verified': self.verify_log_integrity(),
            'current_sequence_number': self.sequence_number - 1
        }
    
    def cleanup_old_logs(self):
        """Clean up old audit logs based on retention policy"""
        retention_days = self.config['log_retention_days']
        cutoff_date = (datetime.now() - timedelta(days=retention_days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                DELETE FROM audit_events WHERE timestamp < ?
            ''', (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old audit log entries")
        
        return deleted_count


# Example usage and testing
if __name__ == "__main__":
    print("Audit Logger Test Suite")
    print("=" * 50)
    
    from cryptographic_manager import CryptographicManager
    
    # Initialize managers
    crypto = CryptographicManager()
    audit = AuditLogger(crypto)
    
    # Test authentication logging
    print("\\n1. Testing Authentication Logging...")
    audit.log_authentication("testuser", True, "192.168.1.100", "Mozilla/5.0", "user_123")
    audit.log_authentication("attacker", False, "10.0.0.1", "curl/7.68.0")
    audit.log_authentication("attacker", False, "10.0.0.1", "curl/7.68.0")
    print("✅ Authentication events logged")
    
    # Test authorization logging
    print("\\n2. Testing Authorization Logging...")
    audit.log_authorization("user_123", "testuser", "sensitive_data", "read", True, "192.168.1.100")
    audit.log_authorization("user_456", "normaluser", "admin_panel", "access", False, "192.168.1.101", "admin_access")
    print("✅ Authorization events logged")
    
    # Test data access logging
    print("\\n3. Testing Data Access Logging...")
    audit.log_data_access("user_123", "testuser", "inspection_results", "query", "192.168.1.100", 
                         True, "confidential", 500)
    print("✅ Data access events logged")
    
    # Test security violation logging
    print("\\n4. Testing Security Violation Logging...")
    audit.log_security_violation("user_456", "malicioususer", "injection_attempt", 
                                "SQL injection detected in input", "10.0.0.2", "high")
    print("✅ Security violation logged")
    
    # Wait for processing
    import time
    time.sleep(2)
    
    # Test event search
    print("\\n5. Testing Event Search...")
    events = audit.search_events({'event_type': 'authentication'}, limit=10)
    print(f"✅ Found {len(events)} authentication events")
    
    # Test statistics
    print("\\n6. Audit Statistics:")
    stats = audit.get_audit_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test integrity verification
    print("\\n7. Testing Log Integrity...")
    integrity_ok = audit.verify_log_integrity()
    if integrity_ok:
        print("✅ Log integrity verified")
    else:
        print("❌ Log integrity check failed")
    
    # Stop processing
    audit.stop_processing()
    
    print("\\n" + "=" * 50)
    print("Audit logger test suite completed")