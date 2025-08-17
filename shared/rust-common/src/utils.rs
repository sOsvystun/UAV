use chrono::{DateTime, Utc};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

// Time utilities
pub fn current_timestamp_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

pub fn timestamp_to_datetime(timestamp_ms: i64) -> DateTime<Utc> {
    DateTime::from_timestamp_millis(timestamp_ms).unwrap_or_else(|| Utc::now())
}

pub fn datetime_to_timestamp(datetime: DateTime<Utc>) -> i64 {
    datetime.timestamp_millis()
}

// ID generation utilities
pub fn generate_correlation_id() -> String {
    Uuid::new_v4().to_string()
}

pub fn generate_request_id() -> String {
    format!("req_{}", Uuid::new_v4().simple())
}

pub fn generate_mission_id() -> String {
    format!("mission_{}", Uuid::new_v4().simple())
}

// String utilities
pub fn sanitize_filename(filename: &str) -> String {
    filename
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c => c,
        })
        .collect()
}

pub fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

// Math utilities
pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

pub fn normalize_angle_degrees(angle: f64) -> f64 {
    let mut normalized = angle % 360.0;
    if normalized < 0.0 {
        normalized += 360.0;
    }
    normalized
}

pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * std::f64::consts::PI / 180.0
}

pub fn radians_to_degrees(radians: f64) -> f64 {
    radians * 180.0 / std::f64::consts::PI
}

// Distance calculations
pub fn euclidean_distance_2d(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
}

pub fn euclidean_distance_3d(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) -> f64 {
    ((x2 - x1).powi(2) + (y2 - y1).powi(2) + (z2 - z1).powi(2)).sqrt()
}

// Coordinate system utilities
use crate::types::Point3D;

pub fn calculate_distance(p1: &Point3D, p2: &Point3D) -> f64 {
    euclidean_distance_3d(p1.x, p1.y, p1.z, p2.x, p2.y, p2.z)
}

pub fn calculate_midpoint(p1: &Point3D, p2: &Point3D) -> Point3D {
    Point3D {
        x: (p1.x + p2.x) / 2.0,
        y: (p1.y + p2.y) / 2.0,
        z: (p1.z + p2.z) / 2.0,
    }
}

pub fn interpolate_point(p1: &Point3D, p2: &Point3D, t: f64) -> Point3D {
    let t = clamp(t, 0.0, 1.0);
    Point3D {
        x: p1.x + t * (p2.x - p1.x),
        y: p1.y + t * (p2.y - p1.y),
        z: p1.z + t * (p2.z - p1.z),
    }
}

// File utilities
use std::path::Path;

pub fn ensure_directory_exists(path: &Path) -> std::io::Result<()> {
    if !path.exists() {
        std::fs::create_dir_all(path)?;
    }
    Ok(())
}

pub fn get_file_extension(filename: &str) -> Option<&str> {
    Path::new(filename).extension()?.to_str()
}

pub fn get_file_stem(filename: &str) -> Option<&str> {
    Path::new(filename).file_stem()?.to_str()
}

// Retry utilities
use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;

pub async fn retry_with_backoff<F, Fut, T, E>(
    mut operation: F,
    max_attempts: usize,
    initial_delay: Duration,
    backoff_factor: f64,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    let mut delay = initial_delay;
    
    for attempt in 1..=max_attempts {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(error) => {
                if attempt == max_attempts {
                    return Err(error);
                }
                
                tracing::warn!(
                    attempt = attempt,
                    max_attempts = max_attempts,
                    delay_ms = delay.as_millis(),
                    "Operation failed, retrying"
                );
                
                sleep(delay).await;
                delay = Duration::from_millis((delay.as_millis() as f64 * backoff_factor) as u64);
            }
        }
    }
    
    unreachable!()
}

// Configuration helpers
pub fn parse_duration_from_string(s: &str) -> Result<Duration, String> {
    let s = s.trim().to_lowercase();
    
    if let Some(num_str) = s.strip_suffix("ms") {
        let ms: u64 = num_str.parse().map_err(|_| "Invalid milliseconds")?;
        Ok(Duration::from_millis(ms))
    } else if let Some(num_str) = s.strip_suffix('s') {
        let secs: u64 = num_str.parse().map_err(|_| "Invalid seconds")?;
        Ok(Duration::from_secs(secs))
    } else if let Some(num_str) = s.strip_suffix('m') {
        let mins: u64 = num_str.parse().map_err(|_| "Invalid minutes")?;
        Ok(Duration::from_secs(mins * 60))
    } else if let Some(num_str) = s.strip_suffix('h') {
        let hours: u64 = num_str.parse().map_err(|_| "Invalid hours")?;
        Ok(Duration::from_secs(hours * 3600))
    } else {
        // Default to seconds if no unit specified
        let secs: u64 = s.parse().map_err(|_| "Invalid duration format")?;
        Ok(Duration::from_secs(secs))
    }
}

// Health check utilities
use crate::pb::common::{HealthCheckRequest, HealthCheckResponse};

pub fn create_health_check_response(
    service_name: &str,
    is_healthy: bool,
) -> HealthCheckResponse {
    use crate::pb::common::health_check_response::ServingStatus;
    
    HealthCheckResponse {
        status: if is_healthy {
            ServingStatus::Serving as i32
        } else {
            ServingStatus::NotServing as i32
        },
    }
}

// Async utilities
use futures::future::BoxFuture;

pub type AsyncResult<T> = BoxFuture<'static, crate::Result<T>>;

pub fn boxed_future<T: Send + 'static>(
    future: impl Future<Output = crate::Result<T>> + Send + 'static,
) -> AsyncResult<T> {
    Box::pin(future)
}

// Testing utilities
#[cfg(test)]
pub mod test_utils {
    use super::*;
    use crate::types::*;
    
    pub fn create_test_point3d(x: f64, y: f64, z: f64) -> Point3D {
        Point3D { x, y, z }
    }
    
    pub fn create_test_weather_conditions() -> WeatherConditions {
        WeatherConditions {
            wind_speed_ms: 5.0,
            wind_direction_degrees: 180.0,
            temperature_celsius: 20.0,
            humidity_percent: 60.0,
            visibility_km: 10.0,
            timestamp: Utc::now(),
        }
    }
    
    pub fn create_test_mission_id() -> MissionId {
        MissionId::new()
    }
    
    pub fn create_test_turbine_id() -> TurbineId {
        TurbineId::new("TEST_TURBINE_001".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_normalize_angle_degrees() {
        assert_eq!(normalize_angle_degrees(0.0), 0.0);
        assert_eq!(normalize_angle_degrees(360.0), 0.0);
        assert_eq!(normalize_angle_degrees(450.0), 90.0);
        assert_eq!(normalize_angle_degrees(-90.0), 270.0);
    }
    
    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5, 0, 10), 5);
        assert_eq!(clamp(-5, 0, 10), 0);
        assert_eq!(clamp(15, 0, 10), 10);
    }
    
    #[test]
    fn test_euclidean_distance_2d() {
        assert_eq!(euclidean_distance_2d(0.0, 0.0, 3.0, 4.0), 5.0);
    }
    
    #[test]
    fn test_parse_duration_from_string() {
        assert_eq!(parse_duration_from_string("100ms").unwrap(), Duration::from_millis(100));
        assert_eq!(parse_duration_from_string("5s").unwrap(), Duration::from_secs(5));
        assert_eq!(parse_duration_from_string("2m").unwrap(), Duration::from_secs(120));
        assert_eq!(parse_duration_from_string("1h").unwrap(), Duration::from_secs(3600));
        assert_eq!(parse_duration_from_string("30").unwrap(), Duration::from_secs(30));
    }
    
    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("test/file:name*.txt"), "test_file_name_.txt");
    }
}