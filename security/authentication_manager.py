"""
Authentication Manager for Secure User Authentication
===================================================

This module provides comprehensive user authentication services including:
- Multi-factor authentication (MFA)
- Session management with secure tokens
- Account lockout and rate limiting
- Password policy enforcement
- User lifecycle management
- Authentication audit logging

Author: Security Framework Team
"""

import os
import json
import time
import secrets
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import logging
import pyotp
import qrcode
from io import BytesIO
import base64

from .cryptographic_manager import CryptographicManager, KeyType, EncryptionAlgorithm

# Configure logging
logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles with different access levels"""
    VIEWER = "viewer"
    OPERATOR = "operator"
    ENGINEER = "engineer"
    ADMIN = "admin"
    SECURITY_OFFICER = "security_officer"

class AuthenticationMethod(Enum):
    """Supported authentication methods"""
    PASSWORD = "password"
    MFA_TOTP = "mfa_totp"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"

class SessionStatus(Enum):
    """Session status types"""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    LOCKED = "locked"

@dataclass
class UserAccount:
    """User account information"""
    user_id: str
    username: str
    email: str
    role: UserRole
    password_hash: str
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    api_keys: List[str] = None
    permissions: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.api_keys is None:
            self.api_keys = []
        if self.permissions is None:
            self.permissions = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AuthenticationCredentials:
    """Authentication credentials container"""
    user_id: str
    username: str
    role: UserRole
    permissions: List[str]
    session_token: str
    expires_at: datetime
    mfa_verified: bool = False
    last_activity: Optional[datetime] = None
    session_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.session_metadata is None:
            self.session_metadata = {}

@dataclass
class SessionInfo:
    """Active session information"""
    session_id: str
    user_id: str
    username: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    mfa_verified: bool
    status: SessionStatus
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AuthenticationAttempt:
    """Authentication attempt record"""
    timestamp: datetime
    username: str
    ip_address: str
    user_agent: str
    method: AuthenticationMethod
    success: bool
    failure_reason: Optional[str] = None
    risk_score: float = 0.0

class AuthenticationManager:
    """
    Comprehensive authentication manager providing secure user authentication,
    session management, and multi-factor authentication.
    """
    
    def __init__(self, crypto_manager: CryptographicManager, 
                 config_path: str = "security/auth_config.json"):
        self.crypto = crypto_manager
        self.config_path = Path(config_path)
        
        # Load configuration
        self.config = self._load_config()
        
        # Storage paths
        self.users_db_path = Path("security/users.json")
        self.sessions_db_path = Path("security/sessions.json")
        self.attempts_db_path = Path("security/auth_attempts.json")
        
        # In-memory storage
        self.users: Dict[str, UserAccount] = {}
        self.active_sessions: Dict[str, SessionInfo] = {}
        self.auth_attempts: List[AuthenticationAttempt] = []
        
        # Load data
        self._load_users_db()
        self._load_sessions_db()
        self._load_attempts_db()
        
        # Create default admin if no users exist
        if not self.users:
            self._create_default_admin()
        
        logger.info("AuthenticationManager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load authentication configuration"""
        default_config = {
            "session_timeout_hours": 8,
            "max_failed_attempts": 5,
            "lockout_duration_minutes": 30,
            "password_policy": {
                "min_length": 12,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special": True,
                "max_age_days": 90,
                "history_count": 5
            },
            "mfa_policy": {
                "required_for_admin": True,
                "required_for_api": False,
                "backup_codes_count": 10
            },
            "session_policy": {
                "max_concurrent_sessions": 3,
                "require_ip_validation": True,
                "idle_timeout_minutes": 30
            },
            "rate_limiting": {
                "max_attempts_per_minute": 5,
                "max_attempts_per_hour": 20,
                "ban_duration_minutes": 60
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load auth config: {e}")
        else:
            # Save default config
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            os.chmod(self.config_path, 0o600)
        
        return default_config
    
    def _load_users_db(self):
        """Load users database"""
        if self.users_db_path.exists():
            try:
                with open(self.users_db_path, 'r') as f:
                    users_data = json.load(f)
                
                for username, user_data in users_data.items():
                    user_account = UserAccount(
                        user_id=user_data['user_id'],
                        username=username,
                        email=user_data['email'],
                        role=UserRole(user_data['role']),
                        password_hash=user_data['password_hash'],
                        created_at=datetime.fromisoformat(user_data['created_at']),
                        last_login=datetime.fromisoformat(user_data['last_login']) if user_data.get('last_login') else None,
                        failed_attempts=user_data.get('failed_attempts', 0),
                        locked_until=datetime.fromisoformat(user_data['locked_until']) if user_data.get('locked_until') else None,
                        mfa_enabled=user_data.get('mfa_enabled', False),
                        mfa_secret=user_data.get('mfa_secret'),
                        api_keys=user_data.get('api_keys', []),
                        permissions=user_data.get('permissions', []),
                        metadata=user_data.get('metadata', {})
                    )
                    self.users[username] = user_account
                
                logger.info(f"Loaded {len(self.users)} user accounts")
                
            except Exception as e:
                logger.error(f"Failed to load users database: {e}")
    
    def _save_users_db(self):
        """Save users database"""
        users_data = {}
        
        for username, user_account in self.users.items():
            users_data[username] = {
                'user_id': user_account.user_id,
                'email': user_account.email,
                'role': user_account.role.value,
                'password_hash': user_account.password_hash,
                'created_at': user_account.created_at.isoformat(),
                'last_login': user_account.last_login.isoformat() if user_account.last_login else None,
                'failed_attempts': user_account.failed_attempts,
                'locked_until': user_account.locked_until.isoformat() if user_account.locked_until else None,
                'mfa_enabled': user_account.mfa_enabled,
                'mfa_secret': user_account.mfa_secret,
                'api_keys': user_account.api_keys,
                'permissions': user_account.permissions,
                'metadata': user_account.metadata
            }
        
        with open(self.users_db_path, 'w') as f:
            json.dump(users_data, f, indent=2)
        os.chmod(self.users_db_path, 0o600)
    
    def _load_sessions_db(self):
        """Load active sessions database"""
        if self.sessions_db_path.exists():
            try:
                with open(self.sessions_db_path, 'r') as f:
                    sessions_data = json.load(f)
                
                current_time = datetime.now()
                
                for session_id, session_data in sessions_data.items():
                    expires_at = datetime.fromisoformat(session_data['expires_at'])
                    
                    # Only load non-expired sessions
                    if expires_at > current_time:
                        session_info = SessionInfo(
                            session_id=session_id,
                            user_id=session_data['user_id'],
                            username=session_data['username'],
                            created_at=datetime.fromisoformat(session_data['created_at']),
                            expires_at=expires_at,
                            last_activity=datetime.fromisoformat(session_data['last_activity']),
                            ip_address=session_data['ip_address'],
                            user_agent=session_data.get('user_agent', 'unknown'),
                            mfa_verified=session_data.get('mfa_verified', False),
                            status=SessionStatus(session_data.get('status', 'active')),
                            metadata=session_data.get('metadata', {})
                        )
                        self.active_sessions[session_id] = session_info
                
                logger.info(f"Loaded {len(self.active_sessions)} active sessions")
                
            except Exception as e:
                logger.error(f"Failed to load sessions database: {e}")
    
    def _save_sessions_db(self):
        """Save active sessions database"""
        sessions_data = {}
        
        for session_id, session_info in self.active_sessions.items():
            sessions_data[session_id] = {
                'user_id': session_info.user_id,
                'username': session_info.username,
                'created_at': session_info.created_at.isoformat(),
                'expires_at': session_info.expires_at.isoformat(),
                'last_activity': session_info.last_activity.isoformat(),
                'ip_address': session_info.ip_address,
                'user_agent': session_info.user_agent,
                'mfa_verified': session_info.mfa_verified,
                'status': session_info.status.value,
                'metadata': session_info.metadata
            }
        
        with open(self.sessions_db_path, 'w') as f:
            json.dump(sessions_data, f, indent=2)
        os.chmod(self.sessions_db_path, 0o600)
    
    def _load_attempts_db(self):
        """Load authentication attempts database"""
        if self.attempts_db_path.exists():
            try:
                with open(self.attempts_db_path, 'r') as f:
                    attempts_data = json.load(f)
                
                # Only load recent attempts (last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                for attempt_data in attempts_data:
                    timestamp = datetime.fromisoformat(attempt_data['timestamp'])
                    if timestamp > cutoff_time:
                        attempt = AuthenticationAttempt(
                            timestamp=timestamp,
                            username=attempt_data['username'],
                            ip_address=attempt_data['ip_address'],
                            user_agent=attempt_data.get('user_agent', 'unknown'),
                            method=AuthenticationMethod(attempt_data['method']),
                            success=attempt_data['success'],
                            failure_reason=attempt_data.get('failure_reason'),
                            risk_score=attempt_data.get('risk_score', 0.0)
                        )
                        self.auth_attempts.append(attempt)
                
                logger.info(f"Loaded {len(self.auth_attempts)} recent authentication attempts")
                
            except Exception as e:
                logger.error(f"Failed to load attempts database: {e}")
    
    def _save_attempts_db(self):
        """Save authentication attempts database"""
        attempts_data = []
        
        for attempt in self.auth_attempts:
            attempts_data.append({
                'timestamp': attempt.timestamp.isoformat(),
                'username': attempt.username,
                'ip_address': attempt.ip_address,
                'user_agent': attempt.user_agent,
                'method': attempt.method.value,
                'success': attempt.success,
                'failure_reason': attempt.failure_reason,
                'risk_score': attempt.risk_score
            })
        
        with open(self.attempts_db_path, 'w') as f:
            json.dump(attempts_data, f, indent=2)
        os.chmod(self.attempts_db_path, 0o600)
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin_password = self.crypto.generate_secure_token(16)
        
        admin_user = UserAccount(
            user_id="admin_001",
            username="admin",
            email="admin@uav-inspection.local",
            role=UserRole.ADMIN,
            password_hash=self.crypto.hash_password(admin_password),
            created_at=datetime.now(),
            permissions=["*"],  # All permissions
            mfa_enabled=True
        )
        
        # Generate MFA secret
        admin_user.mfa_secret = pyotp.random_base32()
        
        self.users["admin"] = admin_user
        self._save_users_db()
        
        logger.warning(f"Created default admin user with password: {admin_password}")
        logger.warning("Please change the default password immediately!")
    
    def create_user(self, username: str, email: str, password: str, 
                   role: UserRole, permissions: List[str] = None) -> str:
        """
        Create new user account.
        
        Args:
            username: Unique username
            email: User email address
            password: User password
            role: User role
            permissions: List of permissions (optional)
            
        Returns:
            User ID of created user
            
        Raises:
            ValueError: If user already exists or validation fails
        """
        # Validate inputs
        if username in self.users:
            raise ValueError(f"User already exists: {username}")
        
        if not self._validate_password(password):
            raise ValueError("Password does not meet policy requirements")
        
        if not self._validate_email(email):
            raise ValueError("Invalid email address")
        
        # Generate user ID
        user_id = f"user_{int(time.time())}_{secrets.token_hex(4)}"
        
        # Create user account
        user_account = UserAccount(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            password_hash=self.crypto.hash_password(password),
            created_at=datetime.now(),
            permissions=permissions or self._get_default_permissions(role)
        )
        
        # Enable MFA for admin users by default
        if role == UserRole.ADMIN or self.config['mfa_policy']['required_for_admin']:
            user_account.mfa_enabled = True
            user_account.mfa_secret = pyotp.random_base32()
        
        self.users[username] = user_account
        self._save_users_db()
        
        logger.info(f"Created user account: {username} ({role.value})")
        return user_id
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: str = "unknown", user_agent: str = "unknown",
                         mfa_code: str = None) -> Optional[AuthenticationCredentials]:
        """
        Authenticate user with username and password.
        
        Args:
            username: Username
            password: Password
            ip_address: Client IP address
            user_agent: Client user agent
            mfa_code: MFA code (if MFA enabled)
            
        Returns:
            AuthenticationCredentials if successful, None otherwise
        """
        attempt = AuthenticationAttempt(
            timestamp=datetime.now(),
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            method=AuthenticationMethod.PASSWORD,
            success=False
        )
        
        try:
            # Check rate limiting
            if self._is_rate_limited(username, ip_address):
                attempt.failure_reason = "rate_limited"
                attempt.risk_score = 0.9
                self._record_attempt(attempt)
                return None
            
            # Check if user exists
            if username not in self.users:
                attempt.failure_reason = "user_not_found"
                attempt.risk_score = 0.6
                self._record_attempt(attempt)
                return None
            
            user_account = self.users[username]
            
            # Check if account is locked
            if self._is_account_locked(user_account):
                attempt.failure_reason = "account_locked"
                attempt.risk_score = 0.8
                self._record_attempt(attempt)
                return None
            
            # Verify password
            if not self.crypto.verify_password(password, user_account.password_hash):
                user_account.failed_attempts += 1
                
                # Lock account if too many failures
                if user_account.failed_attempts >= self.config['max_failed_attempts']:
                    lockout_duration = timedelta(minutes=self.config['lockout_duration_minutes'])
                    user_account.locked_until = datetime.now() + lockout_duration
                    logger.warning(f"Account locked due to failed attempts: {username}")
                
                self._save_users_db()
                
                attempt.failure_reason = "invalid_password"
                attempt.risk_score = 0.7
                self._record_attempt(attempt)
                return None
            
            # Verify MFA if enabled
            if user_account.mfa_enabled:
                if not mfa_code:
                    attempt.failure_reason = "mfa_required"
                    attempt.risk_score = 0.3
                    self._record_attempt(attempt)
                    return None
                
                if not self._verify_mfa_code(user_account, mfa_code):
                    attempt.failure_reason = "invalid_mfa"
                    attempt.risk_score = 0.8
                    self._record_attempt(attempt)
                    return None
            
            # Authentication successful
            user_account.failed_attempts = 0
            user_account.last_login = datetime.now()
            self._save_users_db()
            
            # Create session
            session_token = self._create_session(user_account, ip_address, user_agent)
            
            credentials = AuthenticationCredentials(
                user_id=user_account.user_id,
                username=username,
                role=user_account.role,
                permissions=user_account.permissions,
                session_token=session_token,
                expires_at=datetime.now() + timedelta(hours=self.config['session_timeout_hours']),
                mfa_verified=user_account.mfa_enabled,
                last_activity=datetime.now()
            )
            
            attempt.success = True
            attempt.risk_score = 0.1
            self._record_attempt(attempt)
            
            logger.info(f"User authenticated successfully: {username} from {ip_address}")
            return credentials
            
        except Exception as e:
            attempt.failure_reason = f"system_error: {str(e)}"
            attempt.risk_score = 0.5
            self._record_attempt(attempt)
            logger.error(f"Authentication error for {username}: {e}")
            return None
    
    def validate_session(self, session_token: str, 
                        ip_address: str = None) -> Optional[AuthenticationCredentials]:
        """
        Validate session token and return credentials.
        
        Args:
            session_token: Session token to validate
            ip_address: Client IP address for validation
            
        Returns:
            AuthenticationCredentials if valid, None otherwise
        """
        if session_token not in self.active_sessions:
            return None
        
        session_info = self.active_sessions[session_token]
        
        # Check if session expired
        if datetime.now() > session_info.expires_at:
            self._revoke_session(session_token)
            return None
        
        # Check if session is active
        if session_info.status != SessionStatus.ACTIVE:
            return None
        
        # Validate IP address if required
        if (self.config['session_policy']['require_ip_validation'] and 
            ip_address and session_info.ip_address != ip_address):
            logger.warning(f"IP address mismatch for session {session_token}: {session_info.ip_address} vs {ip_address}")
            self._revoke_session(session_token)
            return None
        
        # Check idle timeout
        idle_timeout = timedelta(minutes=self.config['session_policy']['idle_timeout_minutes'])
        if datetime.now() - session_info.last_activity > idle_timeout:
            self._revoke_session(session_token)
            return None
        
        # Update last activity
        session_info.last_activity = datetime.now()
        self._save_sessions_db()
        
        # Get user account
        username = session_info.username
        if username not in self.users:
            self._revoke_session(session_token)
            return None
        
        user_account = self.users[username]
        
        return AuthenticationCredentials(
            user_id=user_account.user_id,
            username=username,
            role=user_account.role,
            permissions=user_account.permissions,
            session_token=session_token,
            expires_at=session_info.expires_at,
            mfa_verified=session_info.mfa_verified,
            last_activity=session_info.last_activity
        )
    
    def _create_session(self, user_account: UserAccount, 
                       ip_address: str, user_agent: str) -> str:
        """Create new session for user"""
        # Check concurrent session limit
        user_sessions = [s for s in self.active_sessions.values() 
                        if s.user_id == user_account.user_id and s.status == SessionStatus.ACTIVE]
        
        max_sessions = self.config['session_policy']['max_concurrent_sessions']
        if len(user_sessions) >= max_sessions:
            # Revoke oldest session
            oldest_session = min(user_sessions, key=lambda s: s.last_activity)
            self._revoke_session(oldest_session.session_id)
        
        # Generate session token
        session_token = self.crypto.generate_secure_token(32)
        
        # Create session info
        session_info = SessionInfo(
            session_id=session_token,
            user_id=user_account.user_id,
            username=user_account.username,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=self.config['session_timeout_hours']),
            last_activity=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent,
            mfa_verified=user_account.mfa_enabled,
            status=SessionStatus.ACTIVE
        )
        
        self.active_sessions[session_token] = session_info
        self._save_sessions_db()
        
        return session_token
    
    def _revoke_session(self, session_token: str):
        """Revoke session"""
        if session_token in self.active_sessions:
            self.active_sessions[session_token].status = SessionStatus.REVOKED
            del self.active_sessions[session_token]
            self._save_sessions_db()
    
    def logout_user(self, session_token: str):
        """Logout user and revoke session"""
        if session_token in self.active_sessions:
            username = self.active_sessions[session_token].username
            self._revoke_session(session_token)
            logger.info(f"User logged out: {username}")
    
    def _is_account_locked(self, user_account: UserAccount) -> bool:
        """Check if account is locked"""
        if user_account.locked_until:
            if datetime.now() < user_account.locked_until:
                return True
            else:
                # Unlock account
                user_account.locked_until = None
                user_account.failed_attempts = 0
                self._save_users_db()
        
        return False
    
    def _is_rate_limited(self, username: str, ip_address: str) -> bool:
        """Check if user/IP is rate limited"""
        now = datetime.now()
        
        # Check attempts in last minute
        minute_ago = now - timedelta(minutes=1)
        recent_attempts = [a for a in self.auth_attempts 
                          if a.timestamp > minute_ago and 
                          (a.username == username or a.ip_address == ip_address)]
        
        if len(recent_attempts) >= self.config['rate_limiting']['max_attempts_per_minute']:
            return True
        
        # Check attempts in last hour
        hour_ago = now - timedelta(hours=1)
        hourly_attempts = [a for a in self.auth_attempts 
                          if a.timestamp > hour_ago and 
                          (a.username == username or a.ip_address == ip_address)]
        
        if len(hourly_attempts) >= self.config['rate_limiting']['max_attempts_per_hour']:
            return True
        
        return False
    
    def _verify_mfa_code(self, user_account: UserAccount, mfa_code: str) -> bool:
        """Verify MFA TOTP code"""
        if not user_account.mfa_secret:
            return False
        
        totp = pyotp.TOTP(user_account.mfa_secret)
        return totp.verify(mfa_code, valid_window=1)  # Allow 30-second window
    
    def _record_attempt(self, attempt: AuthenticationAttempt):
        """Record authentication attempt"""
        self.auth_attempts.append(attempt)
        
        # Keep only recent attempts
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.auth_attempts = [a for a in self.auth_attempts if a.timestamp > cutoff_time]
        
        self._save_attempts_db()
    
    def _validate_password(self, password: str) -> bool:
        """Validate password against policy"""
        policy = self.config['password_policy']
        
        if len(password) < policy['min_length']:
            return False
        
        if policy['require_uppercase'] and not any(c.isupper() for c in password):
            return False
        
        if policy['require_lowercase'] and not any(c.islower() for c in password):
            return False
        
        if policy['require_numbers'] and not any(c.isdigit() for c in password):
            return False
        
        if policy['require_special'] and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False
        
        return True
    
    def _validate_email(self, email: str) -> bool:
        """Basic email validation"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _get_default_permissions(self, role: UserRole) -> List[str]:
        """Get default permissions for role"""
        permission_map = {
            UserRole.VIEWER: ["read"],
            UserRole.OPERATOR: ["read", "operate"],
            UserRole.ENGINEER: ["read", "operate", "configure"],
            UserRole.ADMIN: ["*"],
            UserRole.SECURITY_OFFICER: ["read", "audit", "security"]
        }
        
        return permission_map.get(role, ["read"])
    
    def enable_mfa(self, username: str) -> Tuple[str, str]:
        """
        Enable MFA for user and return secret and QR code.
        
        Args:
            username: Username to enable MFA for
            
        Returns:
            Tuple of (secret, qr_code_data_url)
        """
        if username not in self.users:
            raise ValueError(f"User not found: {username}")
        
        user_account = self.users[username]
        
        # Generate MFA secret
        secret = pyotp.random_base32()
        user_account.mfa_secret = secret
        user_account.mfa_enabled = True
        
        # Generate QR code
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_account.email,
            issuer_name="UAV Inspection Suite"
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        qr_code_data = base64.b64encode(buffer.getvalue()).decode()
        qr_code_url = f"data:image/png;base64,{qr_code_data}"
        
        self._save_users_db()
        
        logger.info(f"MFA enabled for user: {username}")
        return secret, qr_code_url
    
    def disable_mfa(self, username: str):
        """Disable MFA for user"""
        if username not in self.users:
            raise ValueError(f"User not found: {username}")
        
        user_account = self.users[username]
        user_account.mfa_enabled = False
        user_account.mfa_secret = None
        
        self._save_users_db()
        
        logger.info(f"MFA disabled for user: {username}")
    
    def change_password(self, username: str, old_password: str, new_password: str):
        """Change user password"""
        if username not in self.users:
            raise ValueError(f"User not found: {username}")
        
        user_account = self.users[username]
        
        # Verify old password
        if not self.crypto.verify_password(old_password, user_account.password_hash):
            raise ValueError("Invalid current password")
        
        # Validate new password
        if not self._validate_password(new_password):
            raise ValueError("New password does not meet policy requirements")
        
        # Update password
        user_account.password_hash = self.crypto.hash_password(new_password)
        self._save_users_db()
        
        logger.info(f"Password changed for user: {username}")
    
    def get_user_sessions(self, username: str) -> List[Dict[str, Any]]:
        """Get active sessions for user"""
        if username not in self.users:
            raise ValueError(f"User not found: {username}")
        
        user_id = self.users[username].user_id
        user_sessions = []
        
        for session_token, session_info in self.active_sessions.items():
            if session_info.user_id == user_id:
                user_sessions.append({
                    'session_id': session_token[:16] + "...",  # Truncate for security
                    'created_at': session_info.created_at.isoformat(),
                    'last_activity': session_info.last_activity.isoformat(),
                    'ip_address': session_info.ip_address,
                    'user_agent': session_info.user_agent,
                    'status': session_info.status.value
                })
        
        return user_sessions
    
    def revoke_all_sessions(self, username: str):
        """Revoke all sessions for user"""
        if username not in self.users:
            raise ValueError(f"User not found: {username}")
        
        user_id = self.users[username].user_id
        sessions_to_revoke = []
        
        for session_token, session_info in self.active_sessions.items():
            if session_info.user_id == user_id:
                sessions_to_revoke.append(session_token)
        
        for session_token in sessions_to_revoke:
            self._revoke_session(session_token)
        
        logger.info(f"Revoked {len(sessions_to_revoke)} sessions for user: {username}")
    
    def get_authentication_stats(self) -> Dict[str, Any]:
        """Get authentication statistics"""
        now = datetime.now()
        
        # Recent attempts (last 24 hours)
        recent_attempts = [a for a in self.auth_attempts if a.timestamp > now - timedelta(hours=24)]
        successful_attempts = [a for a in recent_attempts if a.success]
        failed_attempts = [a for a in recent_attempts if not a.success]
        
        # Active sessions
        active_sessions = len(self.active_sessions)
        
        # Locked accounts
        locked_accounts = sum(1 for user in self.users.values() if self._is_account_locked(user))
        
        return {
            'total_users': len(self.users),
            'active_sessions': active_sessions,
            'locked_accounts': locked_accounts,
            'recent_attempts_24h': len(recent_attempts),
            'successful_attempts_24h': len(successful_attempts),
            'failed_attempts_24h': len(failed_attempts),
            'success_rate_24h': len(successful_attempts) / len(recent_attempts) if recent_attempts else 0,
            'mfa_enabled_users': sum(1 for user in self.users.values() if user.mfa_enabled)
        }


# Example usage and testing
if __name__ == "__main__":
    print("Authentication Manager Test Suite")
    print("=" * 50)
    
    from cryptographic_manager import CryptographicManager
    
    # Initialize managers
    crypto = CryptographicManager()
    auth = AuthenticationManager(crypto)
    
    # Test user creation
    print("\\n1. Testing User Creation...")
    try:
        user_id = auth.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePassword123!",
            role=UserRole.ENGINEER,
            permissions=["read", "operate", "configure"]
        )
        print(f"✅ Created user: {user_id}")
    except Exception as e:
        print(f"❌ User creation failed: {e}")
    
    # Test authentication
    print("\\n2. Testing Authentication...")
    credentials = auth.authenticate_user("testuser", "SecurePassword123!", "127.0.0.1", "test-client")
    
    if credentials:
        print(f"✅ Authentication successful for {credentials.username}")
        print(f"   Role: {credentials.role.value}")
        print(f"   Permissions: {credentials.permissions}")
        
        # Test session validation
        print("\\n3. Testing Session Validation...")
        validated_creds = auth.validate_session(credentials.session_token, "127.0.0.1")
        
        if validated_creds:
            print("✅ Session validation successful")
        else:
            print("❌ Session validation failed")
        
        # Test logout
        print("\\n4. Testing Logout...")
        auth.logout_user(credentials.session_token)
        
        # Verify session is invalid
        invalid_creds = auth.validate_session(credentials.session_token)
        if not invalid_creds:
            print("✅ Logout successful - session invalidated")
        else:
            print("❌ Logout failed - session still valid")
    else:
        print("❌ Authentication failed")
    
    # Test MFA
    print("\\n5. Testing MFA Setup...")
    try:
        secret, qr_code = auth.enable_mfa("testuser")
        print(f"✅ MFA enabled with secret: {secret[:8]}...")
        print(f"   QR code length: {len(qr_code)} characters")
    except Exception as e:
        print(f"❌ MFA setup failed: {e}")
    
    # Test statistics
    print("\\n6. Authentication Statistics:")
    stats = auth.get_authentication_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\\n" + "=" * 50)
    print("Authentication manager test suite completed")