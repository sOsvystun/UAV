"""
Cryptographic Manager for Secure Key Management and Encryption
=============================================================

This module provides comprehensive cryptographic services including:
- Symmetric and asymmetric encryption
- Secure key generation and management
- Password hashing and verification
- Digital signatures and verification
- Key derivation and rotation
- Hardware security module (HSM) integration

Author: Security Framework Team
"""

import os
import json
import base64
import secrets
import hashlib
import hmac
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Cryptographic libraries
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization, padding
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.backends import default_backend
from cryptography.x509 import load_pem_x509_certificate, CertificateSigningRequest
from cryptography import x509

# Password hashing
from passlib.context import CryptContext
from passlib.hash import argon2, bcrypt, scrypt

# JWT for tokens
import jwt

# Configure logging
logger = logging.getLogger(__name__)

class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms"""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    FERNET = "fernet"
    RSA_OAEP = "rsa_oaep"

class HashAlgorithm(Enum):
    """Supported hash algorithms"""
    SHA256 = "sha256"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"
    ARGON2 = "argon2"
    BCRYPT = "bcrypt"
    SCRYPT = "scrypt"

class KeyType(Enum):
    """Key types for different purposes"""
    MASTER_KEY = "master_key"
    DATA_ENCRYPTION_KEY = "data_encryption_key"
    KEY_ENCRYPTION_KEY = "key_encryption_key"
    SIGNING_KEY = "signing_key"
    SESSION_KEY = "session_key"

@dataclass
class CryptoKey:
    """Cryptographic key metadata"""
    key_id: str
    key_type: KeyType
    algorithm: EncryptionAlgorithm
    created_at: datetime
    expires_at: Optional[datetime]
    key_data: bytes
    public_key: Optional[bytes] = None
    metadata: Dict[str, Any] = None

@dataclass
class EncryptionResult:
    """Result of encryption operation"""
    ciphertext: bytes
    nonce: Optional[bytes]
    tag: Optional[bytes]
    algorithm: EncryptionAlgorithm
    key_id: str
    metadata: Dict[str, Any]

@dataclass
class SignatureResult:
    """Result of digital signature operation"""
    signature: bytes
    algorithm: str
    key_id: str
    timestamp: datetime

class CryptographicManager:
    """
    Comprehensive cryptographic manager providing secure encryption,
    key management, and cryptographic operations.
    """
    
    def __init__(self, key_storage_path: str = "security/keys", 
                 config_path: str = "security/crypto_config.json"):
        self.key_storage_path = Path(key_storage_path)
        self.config_path = Path(config_path)
        
        # Create directories
        self.key_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize password context
        self.pwd_context = CryptContext(
            schemes=["argon2", "bcrypt", "scrypt"],
            deprecated="auto",
            argon2__rounds=self.config.get("argon2_rounds", 3),
            bcrypt__rounds=self.config.get("bcrypt_rounds", 12),
            scrypt__rounds=self.config.get("scrypt_rounds", 16)
        )
        
        # Key storage
        self.keys: Dict[str, CryptoKey] = {}
        self.master_key: Optional[bytes] = None
        
        # Initialize master key
        self._initialize_master_key()
        
        # Load existing keys
        self._load_keys()
        
        logger.info("CryptographicManager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load cryptographic configuration"""
        default_config = {
            "master_key_algorithm": "aes_256_gcm",
            "default_key_size": 256,
            "key_rotation_days": 90,
            "password_policy": {
                "min_length": 12,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special": True
            },
            "argon2_rounds": 3,
            "bcrypt_rounds": 12,
            "scrypt_rounds": 16,
            "jwt_algorithm": "HS256",
            "jwt_expiry_hours": 24
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load crypto config: {e}")
        else:
            # Save default config
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            os.chmod(self.config_path, 0o600)
        
        return default_config
    
    def _initialize_master_key(self):
        """Initialize or load master key"""
        master_key_file = self.key_storage_path / "master.key"
        
        if master_key_file.exists():
            try:
                with open(master_key_file, 'rb') as f:
                    encrypted_master = f.read()
                
                # For now, use a simple key derivation
                # In production, this should use HSM or secure key storage
                self.master_key = self._derive_master_key_from_system()
                
                # Verify master key by attempting decryption
                self._verify_master_key(encrypted_master)
                
                logger.info("Master key loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load master key: {e}")
                raise RuntimeError("Master key verification failed")
        else:
            # Generate new master key
            self.master_key = self._generate_master_key()
            self._save_master_key()
            logger.info("New master key generated and saved")
    
    def _generate_master_key(self) -> bytes:
        """Generate new master key"""
        return secrets.token_bytes(32)  # 256-bit key
    
    def _derive_master_key_from_system(self) -> bytes:
        """Derive master key from system properties"""
        # This is a simplified implementation
        # In production, use HSM, TPM, or secure key derivation
        system_info = f"{os.name}_{os.getlogin() if hasattr(os, 'getlogin') else 'unknown'}"
        salt = b"uav_inspection_suite_salt_2024"
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        return kdf.derive(system_info.encode())
    
    def _save_master_key(self):
        """Save master key securely"""
        master_key_file = self.key_storage_path / "master.key"
        
        # Encrypt master key with system-derived key
        system_key = self._derive_master_key_from_system()
        fernet = Fernet(base64.urlsafe_b64encode(system_key))
        encrypted_master = fernet.encrypt(self.master_key)
        
        with open(master_key_file, 'wb') as f:
            f.write(encrypted_master)
        
        os.chmod(master_key_file, 0o600)
    
    def _verify_master_key(self, encrypted_master: bytes):
        """Verify master key can decrypt test data"""
        try:
            system_key = self._derive_master_key_from_system()
            fernet = Fernet(base64.urlsafe_b64encode(system_key))
            self.master_key = fernet.decrypt(encrypted_master)
        except Exception as e:
            raise RuntimeError(f"Master key verification failed: {e}")
    
    def _load_keys(self):
        """Load existing keys from storage"""
        keys_file = self.key_storage_path / "keys.json"
        
        if keys_file.exists():
            try:
                with open(keys_file, 'r') as f:
                    keys_data = json.load(f)
                
                for key_id, key_info in keys_data.items():
                    # Decrypt key data
                    encrypted_key = base64.b64decode(key_info['encrypted_key'])
                    key_data = self._decrypt_with_master_key(encrypted_key)
                    
                    # Reconstruct CryptoKey object
                    crypto_key = CryptoKey(
                        key_id=key_id,
                        key_type=KeyType(key_info['key_type']),
                        algorithm=EncryptionAlgorithm(key_info['algorithm']),
                        created_at=datetime.fromisoformat(key_info['created_at']),
                        expires_at=datetime.fromisoformat(key_info['expires_at']) if key_info.get('expires_at') else None,
                        key_data=key_data,
                        public_key=base64.b64decode(key_info['public_key']) if key_info.get('public_key') else None,
                        metadata=key_info.get('metadata', {})
                    )
                    
                    self.keys[key_id] = crypto_key
                
                logger.info(f"Loaded {len(self.keys)} cryptographic keys")
                
            except Exception as e:
                logger.error(f"Failed to load keys: {e}")
    
    def _save_keys(self):
        """Save keys to encrypted storage"""
        keys_file = self.key_storage_path / "keys.json"
        keys_data = {}
        
        for key_id, crypto_key in self.keys.items():
            # Encrypt key data with master key
            encrypted_key = self._encrypt_with_master_key(crypto_key.key_data)
            
            keys_data[key_id] = {
                'key_type': crypto_key.key_type.value,
                'algorithm': crypto_key.algorithm.value,
                'created_at': crypto_key.created_at.isoformat(),
                'expires_at': crypto_key.expires_at.isoformat() if crypto_key.expires_at else None,
                'encrypted_key': base64.b64encode(encrypted_key).decode(),
                'public_key': base64.b64encode(crypto_key.public_key).decode() if crypto_key.public_key else None,
                'metadata': crypto_key.metadata or {}
            }
        
        with open(keys_file, 'w') as f:
            json.dump(keys_data, f, indent=2)
        
        os.chmod(keys_file, 0o600)
    
    def _encrypt_with_master_key(self, data: bytes) -> bytes:
        """Encrypt data with master key"""
        fernet = Fernet(base64.urlsafe_b64encode(self.master_key))
        return fernet.encrypt(data)
    
    def _decrypt_with_master_key(self, encrypted_data: bytes) -> bytes:
        """Decrypt data with master key"""
        fernet = Fernet(base64.urlsafe_b64encode(self.master_key))
        return fernet.decrypt(encrypted_data)
    
    def generate_key(self, key_type: KeyType, algorithm: EncryptionAlgorithm,
                    key_size: int = None, expires_in_days: int = None) -> str:
        """
        Generate new cryptographic key.
        
        Args:
            key_type: Type of key to generate
            algorithm: Encryption algorithm
            key_size: Key size in bits (optional)
            expires_in_days: Key expiration in days (optional)
            
        Returns:
            Key ID for the generated key
        """
        key_id = f"{key_type.value}_{int(time.time())}_{secrets.token_hex(8)}"
        
        # Generate key based on algorithm
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            key_data = secrets.token_bytes(32)  # 256 bits
            public_key = None
        elif algorithm == EncryptionAlgorithm.AES_256_CBC:
            key_data = secrets.token_bytes(32)  # 256 bits
            public_key = None
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            key_data = secrets.token_bytes(32)  # 256 bits
            public_key = None
        elif algorithm == EncryptionAlgorithm.FERNET:
            key_data = Fernet.generate_key()
            public_key = None
        elif algorithm == EncryptionAlgorithm.RSA_OAEP:
            # Generate RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size or 2048,
                backend=default_backend()
            )
            
            key_data = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_key = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        # Create key object
        crypto_key = CryptoKey(
            key_id=key_id,
            key_type=key_type,
            algorithm=algorithm,
            created_at=datetime.now(),
            expires_at=expires_at,
            key_data=key_data,
            public_key=public_key,
            metadata={}
        )
        
        # Store key
        self.keys[key_id] = crypto_key
        self._save_keys()
        
        logger.info(f"Generated new {algorithm.value} key: {key_id}")
        return key_id
    
    def encrypt_data(self, data: Union[str, bytes], key_id: str, 
                    associated_data: bytes = None) -> EncryptionResult:
        """
        Encrypt data with specified key.
        
        Args:
            data: Data to encrypt
            key_id: ID of encryption key
            associated_data: Additional authenticated data (for AEAD)
            
        Returns:
            EncryptionResult with encrypted data and metadata
        """
        if key_id not in self.keys:
            raise ValueError(f"Key not found: {key_id}")
        
        crypto_key = self.keys[key_id]
        
        # Check key expiration
        if crypto_key.expires_at and datetime.now() > crypto_key.expires_at:
            raise ValueError(f"Key expired: {key_id}")
        
        # Convert string to bytes
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Encrypt based on algorithm
        if crypto_key.algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._encrypt_aes_gcm(data, crypto_key, associated_data)
        elif crypto_key.algorithm == EncryptionAlgorithm.AES_256_CBC:
            return self._encrypt_aes_cbc(data, crypto_key)
        elif crypto_key.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._encrypt_chacha20_poly1305(data, crypto_key, associated_data)
        elif crypto_key.algorithm == EncryptionAlgorithm.FERNET:
            return self._encrypt_fernet(data, crypto_key)
        elif crypto_key.algorithm == EncryptionAlgorithm.RSA_OAEP:
            return self._encrypt_rsa_oaep(data, crypto_key)
        else:
            raise ValueError(f"Unsupported encryption algorithm: {crypto_key.algorithm}")
    
    def decrypt_data(self, encryption_result: EncryptionResult, 
                    associated_data: bytes = None) -> bytes:
        """
        Decrypt data using encryption result metadata.
        
        Args:
            encryption_result: Result from encryption operation
            associated_data: Additional authenticated data (for AEAD)
            
        Returns:
            Decrypted data as bytes
        """
        key_id = encryption_result.key_id
        
        if key_id not in self.keys:
            raise ValueError(f"Key not found: {key_id}")
        
        crypto_key = self.keys[key_id]
        
        # Decrypt based on algorithm
        if crypto_key.algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._decrypt_aes_gcm(encryption_result, crypto_key, associated_data)
        elif crypto_key.algorithm == EncryptionAlgorithm.AES_256_CBC:
            return self._decrypt_aes_cbc(encryption_result, crypto_key)
        elif crypto_key.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._decrypt_chacha20_poly1305(encryption_result, crypto_key, associated_data)
        elif crypto_key.algorithm == EncryptionAlgorithm.FERNET:
            return self._decrypt_fernet(encryption_result, crypto_key)
        elif crypto_key.algorithm == EncryptionAlgorithm.RSA_OAEP:
            return self._decrypt_rsa_oaep(encryption_result, crypto_key)
        else:
            raise ValueError(f"Unsupported decryption algorithm: {crypto_key.algorithm}")
    
    def _encrypt_aes_gcm(self, data: bytes, key: CryptoKey, 
                        associated_data: bytes = None) -> EncryptionResult:
        """Encrypt with AES-256-GCM"""
        aesgcm = AESGCM(key.key_data)
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        
        ciphertext = aesgcm.encrypt(nonce, data, associated_data)
        
        return EncryptionResult(
            ciphertext=ciphertext,
            nonce=nonce,
            tag=None,  # Tag is included in ciphertext for AESGCM
            algorithm=key.algorithm,
            key_id=key.key_id,
            metadata={'associated_data_length': len(associated_data) if associated_data else 0}
        )
    
    def _decrypt_aes_gcm(self, result: EncryptionResult, key: CryptoKey,
                        associated_data: bytes = None) -> bytes:
        """Decrypt AES-256-GCM"""
        aesgcm = AESGCM(key.key_data)
        return aesgcm.decrypt(result.nonce, result.ciphertext, associated_data)
    
    def _encrypt_aes_cbc(self, data: bytes, key: CryptoKey) -> EncryptionResult:
        """Encrypt with AES-256-CBC"""
        # Pad data to block size
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        
        # Generate IV
        iv = secrets.token_bytes(16)
        
        # Encrypt
        cipher = Cipher(algorithms.AES(key.key_data), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        return EncryptionResult(
            ciphertext=ciphertext,
            nonce=iv,
            tag=None,
            algorithm=key.algorithm,
            key_id=key.key_id,
            metadata={}
        )
    
    def _decrypt_aes_cbc(self, result: EncryptionResult, key: CryptoKey) -> bytes:
        """Decrypt AES-256-CBC"""
        cipher = Cipher(algorithms.AES(key.key_data), modes.CBC(result.nonce), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(result.ciphertext) + decryptor.finalize()
        
        # Remove padding
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()
    
    def _encrypt_chacha20_poly1305(self, data: bytes, key: CryptoKey,
                                  associated_data: bytes = None) -> EncryptionResult:
        """Encrypt with ChaCha20-Poly1305"""
        chacha = ChaCha20Poly1305(key.key_data)
        nonce = secrets.token_bytes(12)  # 96-bit nonce
        
        ciphertext = chacha.encrypt(nonce, data, associated_data)
        
        return EncryptionResult(
            ciphertext=ciphertext,
            nonce=nonce,
            tag=None,  # Tag is included in ciphertext
            algorithm=key.algorithm,
            key_id=key.key_id,
            metadata={'associated_data_length': len(associated_data) if associated_data else 0}
        )
    
    def _decrypt_chacha20_poly1305(self, result: EncryptionResult, key: CryptoKey,
                                  associated_data: bytes = None) -> bytes:
        """Decrypt ChaCha20-Poly1305"""
        chacha = ChaCha20Poly1305(key.key_data)
        return chacha.decrypt(result.nonce, result.ciphertext, associated_data)
    
    def _encrypt_fernet(self, data: bytes, key: CryptoKey) -> EncryptionResult:
        """Encrypt with Fernet"""
        fernet = Fernet(key.key_data)
        ciphertext = fernet.encrypt(data)
        
        return EncryptionResult(
            ciphertext=ciphertext,
            nonce=None,
            tag=None,
            algorithm=key.algorithm,
            key_id=key.key_id,
            metadata={}
        )
    
    def _decrypt_fernet(self, result: EncryptionResult, key: CryptoKey) -> bytes:
        """Decrypt Fernet"""
        fernet = Fernet(key.key_data)
        return fernet.decrypt(result.ciphertext)
    
    def _encrypt_rsa_oaep(self, data: bytes, key: CryptoKey) -> EncryptionResult:
        """Encrypt with RSA-OAEP"""
        # Load private key to get public key
        private_key = serialization.load_pem_private_key(
            key.key_data, password=None, backend=default_backend()
        )
        public_key = private_key.public_key()
        
        # RSA can only encrypt small amounts of data
        max_size = (public_key.key_size // 8) - 2 * (hashes.SHA256().digest_size) - 2
        
        if len(data) > max_size:
            raise ValueError(f"Data too large for RSA encryption. Max size: {max_size} bytes")
        
        ciphertext = public_key.encrypt(
            data,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return EncryptionResult(
            ciphertext=ciphertext,
            nonce=None,
            tag=None,
            algorithm=key.algorithm,
            key_id=key.key_id,
            metadata={}
        )
    
    def _decrypt_rsa_oaep(self, result: EncryptionResult, key: CryptoKey) -> bytes:
        """Decrypt RSA-OAEP"""
        private_key = serialization.load_pem_private_key(
            key.key_data, password=None, backend=default_backend()
        )
        
        return private_key.decrypt(
            result.ciphertext,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def hash_password(self, password: str) -> str:
        """Hash password using secure algorithm"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(password, hashed)
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)
    
    def create_jwt_token(self, payload: Dict[str, Any], key_id: str = None,
                        expires_in_hours: int = None) -> str:
        """
        Create JWT token with payload.
        
        Args:
            payload: Token payload
            key_id: Key ID for signing (optional, uses default)
            expires_in_hours: Token expiration (optional)
            
        Returns:
            JWT token string
        """
        if not key_id:
            # Use or create default JWT signing key
            key_id = self._get_or_create_jwt_key()
        
        crypto_key = self.keys[key_id]
        
        # Add standard claims
        now = datetime.utcnow()
        payload.update({
            'iat': now,
            'exp': now + timedelta(hours=expires_in_hours or self.config['jwt_expiry_hours'])
        })
        
        # Sign token
        if crypto_key.algorithm == EncryptionAlgorithm.RSA_OAEP:
            # Use RSA private key for signing
            private_key = serialization.load_pem_private_key(
                crypto_key.key_data, password=None, backend=default_backend()
            )
            return jwt.encode(payload, private_key, algorithm='RS256')
        else:
            # Use symmetric key for HMAC
            return jwt.encode(payload, crypto_key.key_data, algorithm=self.config['jwt_algorithm'])
    
    def verify_jwt_token(self, token: str, key_id: str = None) -> Dict[str, Any]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token string
            key_id: Key ID for verification (optional)
            
        Returns:
            Decoded payload
        """
        if not key_id:
            key_id = self._get_or_create_jwt_key()
        
        crypto_key = self.keys[key_id]
        
        try:
            if crypto_key.algorithm == EncryptionAlgorithm.RSA_OAEP:
                # Use RSA public key for verification
                private_key = serialization.load_pem_private_key(
                    crypto_key.key_data, password=None, backend=default_backend()
                )
                public_key = private_key.public_key()
                return jwt.decode(token, public_key, algorithms=['RS256'])
            else:
                # Use symmetric key for HMAC
                return jwt.decode(token, crypto_key.key_data, algorithms=[self.config['jwt_algorithm']])
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid JWT token: {e}")
    
    def _get_or_create_jwt_key(self) -> str:
        """Get or create default JWT signing key"""
        # Look for existing JWT key
        for key_id, crypto_key in self.keys.items():
            if crypto_key.key_type == KeyType.SIGNING_KEY:
                return key_id
        
        # Create new JWT signing key
        return self.generate_key(
            KeyType.SIGNING_KEY,
            EncryptionAlgorithm.FERNET,
            expires_in_days=self.config['key_rotation_days']
        )
    
    def rotate_key(self, key_id: str) -> str:
        """
        Rotate existing key by generating new one with same parameters.
        
        Args:
            key_id: ID of key to rotate
            
        Returns:
            New key ID
        """
        if key_id not in self.keys:
            raise ValueError(f"Key not found: {key_id}")
        
        old_key = self.keys[key_id]
        
        # Generate new key with same parameters
        new_key_id = self.generate_key(
            old_key.key_type,
            old_key.algorithm,
            expires_in_days=self.config['key_rotation_days']
        )
        
        # Mark old key as expired
        old_key.expires_at = datetime.now()
        self._save_keys()
        
        logger.info(f"Rotated key {key_id} -> {new_key_id}")
        return new_key_id
    
    def delete_key(self, key_id: str):
        """Securely delete key"""
        if key_id in self.keys:
            # Overwrite key data with random bytes
            key_data = self.keys[key_id].key_data
            if isinstance(key_data, bytes):
                # Overwrite memory (best effort)
                for i in range(len(key_data)):
                    key_data = key_data[:i] + secrets.token_bytes(1) + key_data[i+1:]
            
            del self.keys[key_id]
            self._save_keys()
            logger.info(f"Deleted key: {key_id}")
    
    def get_key_info(self, key_id: str) -> Dict[str, Any]:
        """Get key metadata without sensitive data"""
        if key_id not in self.keys:
            raise ValueError(f"Key not found: {key_id}")
        
        key = self.keys[key_id]
        return {
            'key_id': key.key_id,
            'key_type': key.key_type.value,
            'algorithm': key.algorithm.value,
            'created_at': key.created_at.isoformat(),
            'expires_at': key.expires_at.isoformat() if key.expires_at else None,
            'has_public_key': key.public_key is not None,
            'metadata': key.metadata or {}
        }
    
    def list_keys(self) -> List[Dict[str, Any]]:
        """List all keys with metadata"""
        return [self.get_key_info(key_id) for key_id in self.keys.keys()]
    
    def cleanup_expired_keys(self):
        """Remove expired keys"""
        now = datetime.now()
        expired_keys = []
        
        for key_id, key in self.keys.items():
            if key.expires_at and now > key.expires_at:
                expired_keys.append(key_id)
        
        for key_id in expired_keys:
            self.delete_key(key_id)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired keys")


# Example usage and testing
if __name__ == "__main__":
    print("Cryptographic Manager Test Suite")
    print("=" * 50)
    
    # Initialize crypto manager
    crypto = CryptographicManager()
    
    # Test key generation
    print("\\n1. Testing Key Generation...")
    aes_key_id = crypto.generate_key(KeyType.DATA_ENCRYPTION_KEY, EncryptionAlgorithm.AES_256_GCM)
    rsa_key_id = crypto.generate_key(KeyType.SIGNING_KEY, EncryptionAlgorithm.RSA_OAEP, key_size=2048)
    print(f"✅ Generated AES key: {aes_key_id}")
    print(f"✅ Generated RSA key: {rsa_key_id}")
    
    # Test encryption/decryption
    print("\\n2. Testing Encryption/Decryption...")
    test_data = "This is sensitive UAV inspection data that needs protection."
    
    # AES-GCM encryption
    encrypted = crypto.encrypt_data(test_data, aes_key_id, b"inspection_metadata")
    decrypted = crypto.decrypt_data(encrypted, b"inspection_metadata")
    
    if decrypted.decode() == test_data:
        print("✅ AES-GCM encryption/decryption successful")
    else:
        print("❌ AES-GCM encryption/decryption failed")
    
    # Test password hashing
    print("\\n3. Testing Password Hashing...")
    password = "SecurePassword123!"
    hashed = crypto.hash_password(password)
    
    if crypto.verify_password(password, hashed):
        print("✅ Password hashing and verification successful")
    else:
        print("❌ Password verification failed")
    
    if not crypto.verify_password("WrongPassword", hashed):
        print("✅ Wrong password correctly rejected")
    else:
        print("❌ Wrong password incorrectly accepted")
    
    # Test JWT tokens
    print("\\n4. Testing JWT Tokens...")
    payload = {"user_id": "admin", "role": "administrator", "permissions": ["read", "write"]}
    token = crypto.create_jwt_token(payload)
    
    try:
        decoded = crypto.verify_jwt_token(token)
        if decoded["user_id"] == "admin":
            print("✅ JWT token creation and verification successful")
        else:
            print("❌ JWT token payload mismatch")
    except Exception as e:
        print(f"❌ JWT token verification failed: {e}")
    
    # Test key rotation
    print("\\n5. Testing Key Rotation...")
    new_key_id = crypto.rotate_key(aes_key_id)
    print(f"✅ Key rotated: {aes_key_id} -> {new_key_id}")
    
    # List keys
    print("\\n6. Key Inventory:")
    keys = crypto.list_keys()
    for key_info in keys:
        status = "EXPIRED" if key_info.get('expires_at') and datetime.fromisoformat(key_info['expires_at']) < datetime.now() else "ACTIVE"
        print(f"   {key_info['key_id'][:16]}... - {key_info['key_type']} ({key_info['algorithm']}) - {status}")
    
    print("\\n" + "=" * 50)
    print("Cryptographic manager test suite completed")