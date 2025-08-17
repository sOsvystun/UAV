use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    pub name: String,
    pub version: String,
    pub environment: Environment,
    pub server: ServerConfig,
    pub logging: LoggingConfig,
    pub metrics: MetricsConfig,
    pub tracing: TracingConfig,
    #[cfg(feature = "database")]
    pub database: Option<DatabaseConfig>,
    #[cfg(feature = "kubernetes")]
    pub kubernetes: Option<KubernetesConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: Option<u32>,
    pub timeout_seconds: Option<u64>,
    pub tls: Option<TlsConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    pub cert_file: String,
    pub key_file: String,
    pub ca_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: LogFormat,
    pub output: LogOutput,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub port: u16,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    pub enabled: bool,
    pub jaeger_endpoint: Option<String>,
    pub service_name: String,
    pub sample_rate: f64,
}

#[cfg(feature = "database")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,
    pub connection_timeout_seconds: u64,
    pub idle_timeout_seconds: u64,
}

#[cfg(feature = "kubernetes")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesConfig {
    pub namespace: String,
    pub in_cluster: bool,
    pub kubeconfig_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Environment {
    Development,
    Testing,
    Staging,
    Production,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    Json,
    Pretty,
    Compact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogOutput {
    Stdout,
    Stderr,
    File(String),
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            name: "uav-service".to_string(),
            version: "0.1.0".to_string(),
            environment: Environment::Development,
            server: ServerConfig::default(),
            logging: LoggingConfig::default(),
            metrics: MetricsConfig::default(),
            tracing: TracingConfig::default(),
            #[cfg(feature = "database")]
            database: None,
            #[cfg(feature = "kubernetes")]
            kubernetes: None,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 50051,
            max_connections: Some(1000),
            timeout_seconds: Some(30),
            tls: None,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: LogFormat::Pretty,
            output: LogOutput::Stdout,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 9090,
            path: "/metrics".to_string(),
        }
    }
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            jaeger_endpoint: None,
            service_name: "uav-service".to_string(),
            sample_rate: 0.1,
        }
    }
}

pub fn load_config() -> crate::Result<ServiceConfig> {
    let mut settings = config::Config::builder()
        .add_source(config::File::with_name("config/default").required(false))
        .add_source(config::File::with_name("config/local").required(false))
        .add_source(config::Environment::with_prefix("UAV").separator("_"));

    // Add environment-specific config
    if let Ok(env) = std::env::var("UAV_ENVIRONMENT") {
        settings = settings.add_source(
            config::File::with_name(&format!("config/{}", env)).required(false)
        );
    }

    let config = settings.build()?;
    Ok(config.try_deserialize()?)
}

// Environment variable helpers
pub fn get_env_or_default(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

pub fn get_env_or_error(key: &str) -> crate::Result<String> {
    std::env::var(key).map_err(|_| crate::Error::Config(
        config::ConfigError::NotFound(format!("Environment variable {} not found", key))
    ))
}