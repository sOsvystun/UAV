use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uav_common::{config::ServiceConfig, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    pub service: ServiceConfig,
    pub workflow: WorkflowConfig,
    pub services: ServiceEndpoints,
    pub quality_gates: QualityGatesConfig,
    pub notifications: NotificationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowConfig {
    pub max_concurrent_missions: usize,
    pub default_timeout_seconds: u64,
    pub retry_attempts: usize,
    pub retry_backoff_seconds: u64,
    pub enable_parallel_processing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoints {
    pub trajectory_service: ServiceEndpoint,
    pub detection_service: ServiceEndpoint,
    pub criticality_service: ServiceEndpoint,
    pub reporting_service: ServiceEndpoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    pub url: String,
    pub timeout_seconds: u64,
    pub max_retries: usize,
    pub health_check_interval_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGatesConfig {
    pub image_quality: ImageQualityConfig,
    pub detection_quality: DetectionQualityConfig,
    pub coverage_quality: CoverageQualityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageQualityConfig {
    pub min_sharpness_score: f64,
    pub max_blur_percentage: f64,
    pub min_brightness_score: f64,
    pub max_noise_level: f64,
    pub enable_auto_retry: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionQualityConfig {
    pub min_confidence_threshold: f64,
    pub min_detections_per_component: i32,
    pub require_consensus: bool,
    pub max_false_positive_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageQualityConfig {
    pub min_surface_coverage_percentage: f64,
    pub min_overlap_percentage: f64,
    pub require_complete_coverage: bool,
    pub max_gap_size_meters: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    pub enabled: bool,
    pub channels: Vec<NotificationChannel>,
    pub templates: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    pub name: String,
    pub channel_type: NotificationChannelType,
    pub config: HashMap<String, String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotificationChannelType {
    Email,
    Slack,
    Webhook,
    Sms,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            service: ServiceConfig::default(),
            workflow: WorkflowConfig::default(),
            services: ServiceEndpoints::default(),
            quality_gates: QualityGatesConfig::default(),
            notifications: NotificationConfig::default(),
        }
    }
}

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            max_concurrent_missions: 10,
            default_timeout_seconds: 300,
            retry_attempts: 3,
            retry_backoff_seconds: 5,
            enable_parallel_processing: true,
        }
    }
}

impl Default for ServiceEndpoints {
    fn default() -> Self {
        Self {
            trajectory_service: ServiceEndpoint {
                url: "http://trajectory-service:50051".to_string(),
                timeout_seconds: 30,
                max_retries: 3,
                health_check_interval_seconds: 30,
            },
            detection_service: ServiceEndpoint {
                url: "http://detection-service:50051".to_string(),
                timeout_seconds: 60,
                max_retries: 3,
                health_check_interval_seconds: 30,
            },
            criticality_service: ServiceEndpoint {
                url: "http://criticality-service:50051".to_string(),
                timeout_seconds: 30,
                max_retries: 3,
                health_check_interval_seconds: 30,
            },
            reporting_service: ServiceEndpoint {
                url: "http://reporting-service:50051".to_string(),
                timeout_seconds: 120,
                max_retries: 3,
                health_check_interval_seconds: 30,
            },
        }
    }
}

impl Default for QualityGatesConfig {
    fn default() -> Self {
        Self {
            image_quality: ImageQualityConfig {
                min_sharpness_score: 0.7,
                max_blur_percentage: 10.0,
                min_brightness_score: 0.3,
                max_noise_level: 0.2,
                enable_auto_retry: true,
            },
            detection_quality: DetectionQualityConfig {
                min_confidence_threshold: 0.8,
                min_detections_per_component: 1,
                require_consensus: false,
                max_false_positive_rate: 0.1,
            },
            coverage_quality: CoverageQualityConfig {
                min_surface_coverage_percentage: 85.0,
                min_overlap_percentage: 20.0,
                require_complete_coverage: false,
                max_gap_size_meters: 2.0,
            },
        }
    }
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            channels: vec![],
            templates: HashMap::new(),
        }
    }
}

impl GatewayConfig {
    pub fn load(config_path: &str) -> Result<Self> {
        let mut settings = config::Config::builder()
            .add_source(config::File::with_name(config_path).required(false))
            .add_source(config::File::with_name("config/gateway").required(false))
            .add_source(config::Environment::with_prefix("UAV_GATEWAY").separator("_"));

        // Add environment-specific config
        if let Ok(env) = std::env::var("UAV_ENVIRONMENT") {
            settings = settings.add_source(
                config::File::with_name(&format!("config/gateway-{}", env)).required(false)
            );
        }

        let config = settings.build()?;
        Ok(config.try_deserialize()?)
    }
    
    pub fn validate(&self) -> Result<()> {
        // Validate service endpoints
        for endpoint in [
            &self.services.trajectory_service,
            &self.services.detection_service,
            &self.services.criticality_service,
            &self.services.reporting_service,
        ] {
            if endpoint.url.is_empty() {
                return Err(uav_common::Error::Config(
                    config::ConfigError::Message("Service endpoint URL cannot be empty".to_string())
                ));
            }
            
            if endpoint.timeout_seconds == 0 {
                return Err(uav_common::Error::Config(
                    config::ConfigError::Message("Service timeout must be greater than 0".to_string())
                ));
            }
        }
        
        // Validate quality gates
        let img_quality = &self.quality_gates.image_quality;
        if img_quality.min_sharpness_score < 0.0 || img_quality.min_sharpness_score > 1.0 {
            return Err(uav_common::Error::Config(
                config::ConfigError::Message("Image sharpness score must be between 0 and 1".to_string())
            ));
        }
        
        let det_quality = &self.quality_gates.detection_quality;
        if det_quality.min_confidence_threshold < 0.0 || det_quality.min_confidence_threshold > 1.0 {
            return Err(uav_common::Error::Config(
                config::ConfigError::Message("Detection confidence threshold must be between 0 and 1".to_string())
            ));
        }
        
        let cov_quality = &self.quality_gates.coverage_quality;
        if cov_quality.min_surface_coverage_percentage < 0.0 || cov_quality.min_surface_coverage_percentage > 100.0 {
            return Err(uav_common::Error::Config(
                config::ConfigError::Message("Surface coverage percentage must be between 0 and 100".to_string())
            ));
        }
        
        Ok(())
    }
}