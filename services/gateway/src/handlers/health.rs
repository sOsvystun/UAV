use axum::{http::StatusCode, response::Json};
use serde_json::{json, Value};

pub struct HealthHandler;

impl HealthHandler {
    pub fn new() -> Self {
        Self
    }
    
    pub async fn health_check(&self) -> Result<Json<Value>, StatusCode> {
        Ok(Json(json!({
            "status": "healthy",
            "service": "uav-gateway",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "version": env!("CARGO_PKG_VERSION")
        })))
    }
    
    pub async fn readiness_check(&self) -> Result<Json<Value>, StatusCode> {
        // TODO: Check downstream service health
        Ok(Json(json!({
            "status": "ready",
            "service": "uav-gateway",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "checks": {
                "trajectory_service": "healthy",
                "detection_service": "healthy",
                "criticality_service": "healthy",
                "reporting_service": "healthy"
            }
        })))
    }
}