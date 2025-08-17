"""
Real-Time Security Monitor and Threat Detection System
=====================================================

This module provides real-time security monitoring capabilities including:
- Continuous threat detection and analysis
- Behavioral anomaly detection
- Network traffic analysis
- System resource monitoring
- Automated incident response
- Security dashboard and alerting

Author: Security Framework Team
"""

import os
import json
import time
import threading
import psutil
import socket
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import logging
from queue import Queue, Empty
import statistics
from collections import defaultdict, deque
import ipaddress

from .audit_logger import AuditLogger, SecurityEvent, SecurityEventType, RiskLevel
from .cryptographic_manager import CryptographicManager

# Configure logging
logger = logging.getLogger(__name__)

class MonitoringMode(Enum):
    """Security monitoring modes"""
    PASSIVE = "passive"
    ACTIVE = "active"
    AGGRESSIVE = "aggressive"

class ThreatLevel(Enum):
    """Threat severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ResponseAction(Enum):
    """Automated response actions"""
    LOG_ONLY = "log_only"
    ALERT = "alert"
    BLOCK_IP = "block_ip"
    DISABLE_USER = "disable_user"
    QUARANTINE = "quarantine"
    SHUTDOWN = "shutdown"

@dataclass
class ThreatDetection:
    """Threat detection result"""
    detection_id: str
    timestamp: datetime
    threat_type: str
    threat_level: ThreatLevel
    source_ip: str
    target_resource: str
    description: str
    indicators: Dict[str, Any]
    confidence_score: float
    recommended_actions: List[ResponseAction]
    correlation_events: List[str]

@dataclass
class SystemMetrics:
    """System performance and security metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_connections: int
    active_processes: int
    failed_logins: int
    security_events: int
    threat_level: ThreatLevel

@dataclass
class NetworkConnection:
    """Network connection information"""
    local_address: str
    local_port: int
    remote_address: str
    remote_port: int
    status: str
    process_id: int
    process_name: str
    timestamp: datetime

@dataclass
class BehavioralProfile:
    """User behavioral profile for anomaly detection"""
    user_id: str
    username: str
    typical_login_hours: List[int]
    typical_ip_ranges: List[str]
    average_session_duration: float
    typical_resources: List[str]
    login_frequency: float
    risk_baseline: float
    last_updated: datetime

class SecurityMonitor:
    """
    Real-time security monitoring system with threat detection,
    behavioral analysis, and automated response capabilities.
    """
    
    def __init__(self, audit_logger: AuditLogger, crypto_manager: CryptographicManager,
                 config_path: str = "security/monitor_config.json"):
        self.audit_logger = audit_logger
        self.crypto = crypto_manager
        self.config_path = Path(config_path)
        
        # Load configuration
        self.config = self._load_config()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_threads = []
        
        # Data storage
        self.system_metrics_history = deque(maxlen=1000)
        self.network_connections = {}
        self.behavioral_profiles = {}
        self.active_threats = {}
        self.blocked_ips = set()
        
        # Event processing
        self.event_queue = Queue()
        self.threat_callbacks: List[Callable] = []
        
        # Anomaly detection
        self.baseline_metrics = {}
        self.anomaly_thresholds = {}
        
        # Load behavioral profiles
        self._load_behavioral_profiles()
        
        # Initialize baseline metrics
        self._initialize_baselines()
        
        logger.info("SecurityMonitor initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load security monitoring configuration"""
        default_config = {
            "monitoring_mode": "active",
            "scan_interval_seconds": 30,
            "threat_detection": {
                "enabled": True,
                "behavioral_analysis": True,
                "network_monitoring": True,
                "system_monitoring": True,
                "anomaly_detection": True
            },
            "thresholds": {
                "cpu_usage_critical": 90.0,
                "memory_usage_critical": 85.0,
                "disk_usage_critical": 95.0,
                "failed_login_threshold": 5,
                "connection_threshold": 1000,
                "anomaly_score_threshold": 0.7
            },
            "response_actions": {
                "auto_block_enabled": True,
                "auto_disable_users": False,
                "quarantine_enabled": False,
                "alert_threshold": "medium"
            },
            "behavioral_analysis": {
                "learning_period_days": 30,
                "min_samples": 10,
                "anomaly_sensitivity": 0.8,
                "profile_update_interval": 3600
            },
            "network_monitoring": {
                "monitor_external_connections": True,
                "suspicious_ports": [22, 23, 135, 139, 445, 1433, 3389],
                "allowed_ip_ranges": ["192.168.0.0/16", "10.0.0.0/8", "172.16.0.0/12"],
                "geo_blocking_enabled": False
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load monitor config: {e}")
        else:
            # Save default config
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            os.chmod(self.config_path, 0o600)
        
        return default_config
    
    def _load_behavioral_profiles(self):
        """Load user behavioral profiles"""
        profiles_file = Path("security/behavioral_profiles.json")
        
        if profiles_file.exists():
            try:
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)
                
                for user_id, profile_data in profiles_data.items():
                    profile = BehavioralProfile(
                        user_id=profile_data['user_id'],
                        username=profile_data['username'],
                        typical_login_hours=profile_data['typical_login_hours'],
                        typical_ip_ranges=profile_data['typical_ip_ranges'],
                        average_session_duration=profile_data['average_session_duration'],
                        typical_resources=profile_data['typical_resources'],
                        login_frequency=profile_data['login_frequency'],
                        risk_baseline=profile_data['risk_baseline'],
                        last_updated=datetime.fromisoformat(profile_data['last_updated'])
                    )
                    self.behavioral_profiles[user_id] = profile
                
                logger.info(f"Loaded {len(self.behavioral_profiles)} behavioral profiles")
                
            except Exception as e:
                logger.error(f"Failed to load behavioral profiles: {e}")
    
    def _save_behavioral_profiles(self):
        """Save user behavioral profiles"""
        profiles_file = Path("security/behavioral_profiles.json")
        profiles_data = {}
        
        for user_id, profile in self.behavioral_profiles.items():
            profiles_data[user_id] = {
                'user_id': profile.user_id,
                'username': profile.username,
                'typical_login_hours': profile.typical_login_hours,
                'typical_ip_ranges': profile.typical_ip_ranges,
                'average_session_duration': profile.average_session_duration,
                'typical_resources': profile.typical_resources,
                'login_frequency': profile.login_frequency,
                'risk_baseline': profile.risk_baseline,
                'last_updated': profile.last_updated.isoformat()
            }
        
        with open(profiles_file, 'w') as f:
            json.dump(profiles_data, f, indent=2)
        os.chmod(profiles_file, 0o600)
    
    def _initialize_baselines(self):
        """Initialize baseline metrics for anomaly detection"""
        # Collect initial system metrics
        for _ in range(10):
            metrics = self._collect_system_metrics()
            self.system_metrics_history.append(metrics)
            time.sleep(1)
        
        # Calculate baseline values
        if self.system_metrics_history:
            cpu_values = [m.cpu_usage for m in self.system_metrics_history]
            memory_values = [m.memory_usage for m in self.system_metrics_history]
            
            self.baseline_metrics = {
                'cpu_mean': statistics.mean(cpu_values),
                'cpu_stdev': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0,
                'memory_mean': statistics.mean(memory_values),
                'memory_stdev': statistics.stdev(memory_values) if len(memory_values) > 1 else 0
            }
            
            # Set anomaly thresholds (3 standard deviations)
            self.anomaly_thresholds = {
                'cpu_high': self.baseline_metrics['cpu_mean'] + 3 * self.baseline_metrics['cpu_stdev'],
                'memory_high': self.baseline_metrics['memory_mean'] + 3 * self.baseline_metrics['memory_stdev']
            }
        
        logger.info("Initialized baseline metrics for anomaly detection")
    
    def start_monitoring(self):
        """Start security monitoring"""
        if self.monitoring_active:
            logger.warning("Security monitoring already active")
            return
        
        self.monitoring_active = True
        
        # Start monitoring threads
        threads = [
            threading.Thread(target=self._system_monitoring_loop, name="SystemMonitor"),
            threading.Thread(target=self._network_monitoring_loop, name="NetworkMonitor"),
            threading.Thread(target=self._threat_detection_loop, name="ThreatDetector"),
            threading.Thread(target=self._behavioral_analysis_loop, name="BehavioralAnalyzer")
        ]
        
        for thread in threads:
            thread.daemon = True
            thread.start()
            self.monitoring_threads.append(thread)
        
        logger.info("Security monitoring started")
    
    def stop_monitoring(self):
        """Stop security monitoring"""
        self.monitoring_active = False
        
        # Wait for threads to finish
        for thread in self.monitoring_threads:
            thread.join(timeout=5)
        
        self.monitoring_threads.clear()
        logger.info("Security monitoring stopped")
    
    def _system_monitoring_loop(self):
        """System resource monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.system_metrics_history.append(metrics)
                
                # Check for anomalies
                self._detect_system_anomalies(metrics)
                
                # Sleep until next scan
                time.sleep(self.config['scan_interval_seconds'])
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(5)
    
    def _network_monitoring_loop(self):
        """Network connection monitoring loop"""
        while self.monitoring_active:
            try:
                # Monitor network connections
                connections = self._get_network_connections()
                self._analyze_network_connections(connections)
                
                # Sleep until next scan
                time.sleep(self.config['scan_interval_seconds'])
                
            except Exception as e:
                logger.error(f"Error in network monitoring: {e}")
                time.sleep(5)
    
    def _threat_detection_loop(self):
        """Threat detection processing loop"""
        while self.monitoring_active:
            try:
                # Process events from audit logger
                self._process_security_events()
                
                # Analyze active threats
                self._analyze_active_threats()
                
                # Sleep briefly
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in threat detection: {e}")
                time.sleep(5)
    
    def _behavioral_analysis_loop(self):
        """Behavioral analysis loop"""
        while self.monitoring_active:
            try:
                # Update behavioral profiles
                self._update_behavioral_profiles()
                
                # Detect behavioral anomalies
                self._detect_behavioral_anomalies()
                
                # Sleep until next analysis
                time.sleep(self.config['behavioral_analysis']['profile_update_interval'])
                
            except Exception as e:
                logger.error(f"Error in behavioral analysis: {e}")
                time.sleep(60)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        # Network connections
        connections = psutil.net_connections()
        network_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
        
        # Active processes
        active_processes = len(psutil.pids())
        
        # Get recent security events count
        recent_events = self._count_recent_security_events()
        failed_logins = self._count_recent_failed_logins()
        
        # Calculate threat level
        threat_level = self._calculate_system_threat_level(
            cpu_usage, memory_usage, disk_usage, failed_logins, recent_events
        )
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_connections=network_connections,
            active_processes=active_processes,
            failed_logins=failed_logins,
            security_events=recent_events,
            threat_level=threat_level
        )
    
    def _get_network_connections(self) -> List[NetworkConnection]:
        """Get current network connections"""
        connections = []
        
        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.laddr and conn.raddr:
                    # Get process information
                    process_name = "unknown"
                    if conn.pid:
                        try:
                            process = psutil.Process(conn.pid)
                            process_name = process.name()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    
                    connection = NetworkConnection(
                        local_address=conn.laddr.ip,
                        local_port=conn.laddr.port,
                        remote_address=conn.raddr.ip,
                        remote_port=conn.raddr.port,
                        status=conn.status,
                        process_id=conn.pid or 0,
                        process_name=process_name,
                        timestamp=datetime.now()
                    )
                    connections.append(connection)
        
        except Exception as e:
            logger.error(f"Error getting network connections: {e}")
        
        return connections
    
    def _analyze_network_connections(self, connections: List[NetworkConnection]):
        """Analyze network connections for threats"""
        suspicious_connections = []
        
        for conn in connections:
            risk_score = 0.0
            indicators = {}
            
            # Check for suspicious ports
            if conn.remote_port in self.config['network_monitoring']['suspicious_ports']:
                risk_score += 0.3
                indicators['suspicious_port'] = conn.remote_port
            
            # Check for external connections
            if self._is_external_ip(conn.remote_address):
                risk_score += 0.2
                indicators['external_connection'] = True
                
                # Check if IP is in allowed ranges
                if not self._is_ip_allowed(conn.remote_address):
                    risk_score += 0.4
                    indicators['unauthorized_external'] = True
            
            # Check for blocked IPs
            if conn.remote_address in self.blocked_ips:
                risk_score += 0.8
                indicators['blocked_ip_connection'] = True
            
            # Check for unusual process
            if conn.process_name in ['nc', 'netcat', 'telnet', 'nmap']:
                risk_score += 0.5
                indicators['suspicious_process'] = conn.process_name
            
            # If risk score is high, create threat detection
            if risk_score >= 0.6:
                threat = ThreatDetection(
                    detection_id=f"net_{int(time.time())}_{self.crypto.generate_secure_token(8)}",
                    timestamp=datetime.now(),
                    threat_type="suspicious_network_connection",
                    threat_level=ThreatLevel.HIGH if risk_score >= 0.8 else ThreatLevel.MEDIUM,
                    source_ip=conn.remote_address,
                    target_resource=f"{conn.local_address}:{conn.local_port}",
                    description=f"Suspicious network connection from {conn.remote_address}:{conn.remote_port}",
                    indicators=indicators,
                    confidence_score=risk_score,
                    recommended_actions=[ResponseAction.ALERT, ResponseAction.BLOCK_IP] if risk_score >= 0.8 else [ResponseAction.ALERT],
                    correlation_events=[]
                )
                
                self._handle_threat_detection(threat)
    
    def _detect_system_anomalies(self, metrics: SystemMetrics):
        """Detect system performance anomalies"""
        anomalies = []
        
        # Check CPU usage
        if metrics.cpu_usage > self.config['thresholds']['cpu_usage_critical']:
            anomalies.append({
                'type': 'high_cpu_usage',
                'value': metrics.cpu_usage,
                'threshold': self.config['thresholds']['cpu_usage_critical']
            })
        
        # Check memory usage
        if metrics.memory_usage > self.config['thresholds']['memory_usage_critical']:
            anomalies.append({
                'type': 'high_memory_usage',
                'value': metrics.memory_usage,
                'threshold': self.config['thresholds']['memory_usage_critical']
            })
        
        # Check disk usage
        if metrics.disk_usage > self.config['thresholds']['disk_usage_critical']:
            anomalies.append({
                'type': 'high_disk_usage',
                'value': metrics.disk_usage,
                'threshold': self.config['thresholds']['disk_usage_critical']
            })
        
        # Check failed logins
        if metrics.failed_logins > self.config['thresholds']['failed_login_threshold']:
            anomalies.append({
                'type': 'excessive_failed_logins',
                'value': metrics.failed_logins,
                'threshold': self.config['thresholds']['failed_login_threshold']
            })
        
        # Statistical anomaly detection
        if self.baseline_metrics:
            if (metrics.cpu_usage > self.anomaly_thresholds.get('cpu_high', 100) or
                metrics.memory_usage > self.anomaly_thresholds.get('memory_high', 100)):
                anomalies.append({
                    'type': 'statistical_anomaly',
                    'cpu_deviation': abs(metrics.cpu_usage - self.baseline_metrics['cpu_mean']),
                    'memory_deviation': abs(metrics.memory_usage - self.baseline_metrics['memory_mean'])
                })
        
        # Create threat detections for significant anomalies
        for anomaly in anomalies:
            if anomaly['type'] in ['high_cpu_usage', 'high_memory_usage', 'excessive_failed_logins']:
                threat_level = ThreatLevel.HIGH if anomaly['value'] > anomaly['threshold'] * 1.2 else ThreatLevel.MEDIUM
                
                threat = ThreatDetection(
                    detection_id=f"sys_{int(time.time())}_{self.crypto.generate_secure_token(8)}",
                    timestamp=datetime.now(),
                    threat_type="system_anomaly",
                    threat_level=threat_level,
                    source_ip="localhost",
                    target_resource="system",
                    description=f"System anomaly detected: {anomaly['type']}",
                    indicators=anomaly,
                    confidence_score=0.7,
                    recommended_actions=[ResponseAction.ALERT],
                    correlation_events=[]
                )
                
                self._handle_threat_detection(threat)
    
    def _process_security_events(self):
        """Process security events for threat detection"""
        # This would integrate with the audit logger to get recent events
        # For now, we'll simulate processing
        pass
    
    def _analyze_active_threats(self):
        """Analyze and correlate active threats"""
        current_time = datetime.now()
        
        # Remove old threats (older than 1 hour)
        expired_threats = []
        for threat_id, threat in self.active_threats.items():
            if current_time - threat.timestamp > timedelta(hours=1):
                expired_threats.append(threat_id)
        
        for threat_id in expired_threats:
            del self.active_threats[threat_id]
        
        # Correlate threats by source IP
        ip_threats = defaultdict(list)
        for threat in self.active_threats.values():
            ip_threats[threat.source_ip].append(threat)
        
        # Look for patterns
        for source_ip, threats in ip_threats.items():
            if len(threats) >= 3:  # Multiple threats from same IP
                self._create_correlated_threat(source_ip, threats)
    
    def _create_correlated_threat(self, source_ip: str, threats: List[ThreatDetection]):
        """Create correlated threat from multiple detections"""
        correlation_id = f"corr_{int(time.time())}_{self.crypto.generate_secure_token(8)}"
        
        # Calculate combined risk score
        combined_score = min(sum(t.confidence_score for t in threats) / len(threats) + 0.3, 1.0)
        
        correlated_threat = ThreatDetection(
            detection_id=correlation_id,
            timestamp=datetime.now(),
            threat_type="correlated_attack",
            threat_level=ThreatLevel.CRITICAL,
            source_ip=source_ip,
            target_resource="multiple",
            description=f"Correlated attack detected from {source_ip} ({len(threats)} incidents)",
            indicators={
                'threat_count': len(threats),
                'threat_types': [t.threat_type for t in threats],
                'time_span': (max(t.timestamp for t in threats) - min(t.timestamp for t in threats)).total_seconds()
            },
            confidence_score=combined_score,
            recommended_actions=[ResponseAction.BLOCK_IP, ResponseAction.ALERT],
            correlation_events=[t.detection_id for t in threats]
        )
        
        self._handle_threat_detection(correlated_threat)
    
    def _update_behavioral_profiles(self):
        """Update user behavioral profiles"""
        # This would analyze recent user activity to update profiles
        # Implementation would query audit logs for user behavior patterns
        pass
    
    def _detect_behavioral_anomalies(self):
        """Detect behavioral anomalies in user activity"""
        # This would compare current user behavior against established profiles
        # Implementation would analyze login patterns, resource access, etc.
        pass
    
    def _handle_threat_detection(self, threat: ThreatDetection):
        """Handle detected threat"""
        # Store threat
        self.active_threats[threat.detection_id] = threat
        
        # Log security event
        self.audit_logger.log_security_event(SecurityEvent(
            event_id=threat.detection_id,
            timestamp=threat.timestamp,
            event_type=SecurityEventType.INTRUSION_ATTEMPT,
            risk_level=RiskLevel.CRITICAL if threat.threat_level == ThreatLevel.CRITICAL else RiskLevel.HIGH,
            user_id=None,
            username=None,
            source_ip=threat.source_ip,
            user_agent="security_monitor",
            resource=threat.target_resource,
            action="threat_detection",
            result="detected",
            details={
                'threat_type': threat.threat_type,
                'indicators': threat.indicators,
                'confidence_score': threat.confidence_score,
                'recommended_actions': [a.value for a in threat.recommended_actions]
            },
            risk_score=threat.confidence_score
        ))
        
        # Execute automated responses
        self._execute_response_actions(threat)
        
        # Notify callbacks
        for callback in self.threat_callbacks:
            try:
                callback(threat)
            except Exception as e:
                logger.error(f"Error in threat callback: {e}")
        
        logger.warning(f"THREAT DETECTED: {threat.threat_type} from {threat.source_ip} (confidence: {threat.confidence_score:.2f})")
    
    def _execute_response_actions(self, threat: ThreatDetection):
        """Execute automated response actions"""
        if not self.config['response_actions']['auto_block_enabled']:
            return
        
        for action in threat.recommended_actions:
            try:
                if action == ResponseAction.BLOCK_IP:
                    self._block_ip_address(threat.source_ip)
                elif action == ResponseAction.ALERT:
                    self._send_threat_alert(threat)
                elif action == ResponseAction.DISABLE_USER and self.config['response_actions']['auto_disable_users']:
                    # Would disable user account
                    pass
                elif action == ResponseAction.QUARANTINE and self.config['response_actions']['quarantine_enabled']:
                    # Would quarantine affected resources
                    pass
            except Exception as e:
                logger.error(f"Error executing response action {action}: {e}")
    
    def _block_ip_address(self, ip_address: str):
        """Block IP address"""
        self.blocked_ips.add(ip_address)
        logger.warning(f"BLOCKED IP ADDRESS: {ip_address}")
        
        # In a real implementation, this would:
        # - Add firewall rules
        # - Update network ACLs
        # - Notify network security devices
    
    def _send_threat_alert(self, threat: ThreatDetection):
        """Send threat alert notification"""
        alert_data = {
            'detection_id': threat.detection_id,
            'threat_type': threat.threat_type,
            'threat_level': threat.threat_level.value,
            'source_ip': threat.source_ip,
            'confidence': threat.confidence_score,
            'description': threat.description,
            'timestamp': threat.timestamp.isoformat()
        }
        
        logger.critical(f"THREAT ALERT: {json.dumps(alert_data)}")
        
        # In a real implementation, this would:
        # - Send email notifications
        # - Post to Slack/Teams
        # - Call webhook endpoints
        # - Update SIEM systems
    
    def _count_recent_security_events(self) -> int:
        """Count recent security events"""
        # This would query the audit logger for recent events
        # For now, return a simulated count
        return len(self.active_threats)
    
    def _count_recent_failed_logins(self) -> int:
        """Count recent failed login attempts"""
        # This would query the audit logger for failed authentication events
        # For now, return a simulated count
        return 0
    
    def _calculate_system_threat_level(self, cpu: float, memory: float, 
                                     disk: float, failed_logins: int, events: int) -> ThreatLevel:
        """Calculate overall system threat level"""
        risk_score = 0.0
        
        # Resource usage contribution
        if cpu > 80:
            risk_score += 0.2
        if memory > 80:
            risk_score += 0.2
        if disk > 90:
            risk_score += 0.3
        
        # Security events contribution
        if failed_logins > 5:
            risk_score += 0.4
        if events > 10:
            risk_score += 0.3
        
        # Convert to threat level
        if risk_score >= 0.8:
            return ThreatLevel.CRITICAL
        elif risk_score >= 0.6:
            return ThreatLevel.HIGH
        elif risk_score >= 0.3:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _is_external_ip(self, ip_address: str) -> bool:
        """Check if IP address is external"""
        try:
            ip = ipaddress.ip_address(ip_address)
            return not ip.is_private
        except ValueError:
            return False
    
    def _is_ip_allowed(self, ip_address: str) -> bool:
        """Check if IP address is in allowed ranges"""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            for range_str in self.config['network_monitoring']['allowed_ip_ranges']:
                network = ipaddress.ip_network(range_str, strict=False)
                if ip in network:
                    return True
            
            return False
        except ValueError:
            return False
    
    def add_threat_callback(self, callback: Callable[[ThreatDetection], None]):
        """Add callback for threat notifications"""
        self.threat_callbacks.append(callback)
    
    def remove_threat_callback(self, callback: Callable[[ThreatDetection], None]):
        """Remove threat callback"""
        if callback in self.threat_callbacks:
            self.threat_callbacks.remove(callback)
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        return self._collect_system_metrics()
    
    def get_active_threats(self) -> List[ThreatDetection]:
        """Get list of active threats"""
        return list(self.active_threats.values())
    
    def get_blocked_ips(self) -> List[str]:
        """Get list of blocked IP addresses"""
        return list(self.blocked_ips)
    
    def unblock_ip(self, ip_address: str):
        """Unblock IP address"""
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            logger.info(f"UNBLOCKED IP ADDRESS: {ip_address}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring system status"""
        current_metrics = self.get_current_metrics()
        
        return {
            'monitoring_active': self.monitoring_active,
            'monitoring_mode': self.config['monitoring_mode'],
            'active_threads': len([t for t in self.monitoring_threads if t.is_alive()]),
            'active_threats': len(self.active_threats),
            'blocked_ips': len(self.blocked_ips),
            'current_threat_level': current_metrics.threat_level.value,
            'system_metrics': {
                'cpu_usage': current_metrics.cpu_usage,
                'memory_usage': current_metrics.memory_usage,
                'disk_usage': current_metrics.disk_usage,
                'network_connections': current_metrics.network_connections,
                'active_processes': current_metrics.active_processes
            },
            'behavioral_profiles': len(self.behavioral_profiles),
            'baseline_established': bool(self.baseline_metrics)
        }


# Example usage and testing
if __name__ == "__main__":
    print("Security Monitor Test Suite")
    print("=" * 50)
    
    from cryptographic_manager import CryptographicManager
    from audit_logger import AuditLogger
    
    # Initialize components
    crypto = CryptographicManager()
    audit = AuditLogger(crypto)
    monitor = SecurityMonitor(audit, crypto)
    
    # Add threat callback
    def threat_handler(threat: ThreatDetection):
        print(f"🚨 THREAT CALLBACK: {threat.threat_type} from {threat.source_ip}")
    
    monitor.add_threat_callback(threat_handler)
    
    # Test system metrics collection
    print("\\n1. Testing System Metrics Collection...")
    metrics = monitor.get_current_metrics()
    print(f"✅ CPU: {metrics.cpu_usage:.1f}%, Memory: {metrics.memory_usage:.1f}%, Threat Level: {metrics.threat_level.value}")
    
    # Start monitoring
    print("\\n2. Starting Security Monitoring...")
    monitor.start_monitoring()
    print("✅ Security monitoring started")
    
    # Wait for some monitoring activity
    print("\\n3. Monitoring for 10 seconds...")
    time.sleep(10)
    
    # Check monitoring status
    print("\\n4. Monitoring Status:")
    status = monitor.get_monitoring_status()
    for key, value in status.items():
        if key != 'system_metrics':
            print(f"   {key}: {value}")
        else:
            print(f"   system_metrics:")
            for metric, val in value.items():
                print(f"     {metric}: {val}")
    
    # Test threat detection (simulate)
    print("\\n5. Testing Threat Detection...")
    fake_threat = ThreatDetection(
        detection_id="test_threat_001",
        timestamp=datetime.now(),
        threat_type="test_threat",
        threat_level=ThreatLevel.HIGH,
        source_ip="192.168.1.100",
        target_resource="test_resource",
        description="Test threat for demonstration",
        indicators={'test': True},
        confidence_score=0.85,
        recommended_actions=[ResponseAction.ALERT],
        correlation_events=[]
    )
    
    monitor._handle_threat_detection(fake_threat)
    
    # Check active threats
    active_threats = monitor.get_active_threats()
    print(f"✅ Active threats: {len(active_threats)}")
    
    # Stop monitoring
    print("\\n6. Stopping Security Monitoring...")
    monitor.stop_monitoring()
    print("✅ Security monitoring stopped")
    
    print("\\n" + "=" * 50)
    print("Security monitor test suite completed")