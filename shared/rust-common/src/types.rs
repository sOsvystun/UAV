use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use validator::Validate;

// Common domain types

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MissionId(pub Uuid);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TurbineId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DefectId(pub Uuid);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ReportId(pub Uuid);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TrajectoryId(pub Uuid);

impl MissionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    pub fn from_string(s: &str) -> Result<Self, uuid::Error> {
        Ok(Self(Uuid::parse_str(s)?))
    }
    
    pub fn as_str(&self) -> String {
        self.0.to_string()
    }
}

impl DefectId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    pub fn from_string(s: &str) -> Result<Self, uuid::Error> {
        Ok(Self(Uuid::parse_str(s)?))
    }
    
    pub fn as_str(&self) -> String {
        self.0.to_string()
    }
}

impl ReportId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    pub fn from_string(s: &str) -> Result<Self, uuid::Error> {
        Ok(Self(Uuid::parse_str(s)?))
    }
    
    pub fn as_str(&self) -> String {
        self.0.to_string()
    }
}

impl TrajectoryId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    pub fn from_string(s: &str) -> Result<Self, uuid::Error> {
        Ok(Self(Uuid::parse_str(s)?))
    }
    
    pub fn as_str(&self) -> String {
        self.0.to_string()
    }
}

impl TurbineId {
    pub fn new(id: String) -> Self {
        Self(id)
    }
    
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// Coordinate system types
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct BoundingBox {
    #[validate(range(min = 0.0))]
    pub x: f64,
    #[validate(range(min = 0.0))]
    pub y: f64,
    #[validate(range(min = 0.0))]
    pub width: f64,
    #[validate(range(min = 0.0))]
    pub height: f64,
}

// Image types
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ImageMetadata {
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub timestamp: DateTime<Utc>,
    pub camera_id: Option<String>,
    pub exposure_settings: Option<ExposureSettings>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageFormat {
    Jpeg,
    Png,
    Tiff,
    Raw,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExposureSettings {
    pub iso: u32,
    pub shutter_speed: f64,
    pub aperture: f64,
    pub focal_length: f64,
}

// Weather and environmental types
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct WeatherConditions {
    #[validate(range(min = 0.0))]
    pub wind_speed_ms: f64,
    #[validate(range(min = 0.0, max = 360.0))]
    pub wind_direction_degrees: f64,
    pub temperature_celsius: f64,
    #[validate(range(min = 0.0, max = 100.0))]
    pub humidity_percent: f64,
    #[validate(range(min = 0.0))]
    pub visibility_km: f64,
    pub timestamp: DateTime<Utc>,
}

// Defect types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DefectType {
    Crack,
    Erosion,
    Rust,
    PaintLoss,
    Delamination,
    LightningDamage,
    IceDamage,
    ForeignObject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DefectSeverity {
    Negligible,
    Low,
    Medium,
    High,
    Severe,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct Defect {
    pub id: DefectId,
    pub defect_type: DefectType,
    #[validate(range(min = 0.0, max = 1.0))]
    pub confidence: f64,
    pub bounding_box: BoundingBox,
    pub world_coordinates: Option<Point3D>,
    pub severity: DefectSeverity,
    pub size_pixels: f64,
    pub size_meters: Option<f64>,
    pub thermal_signature: Option<ThermalSignature>,
    pub detected_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalSignature {
    pub temperature_celsius: f64,
    pub temperature_delta: f64,
    pub pattern: ThermalPattern,
    pub thermal_contrast: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermalPattern {
    Hotspot,
    ColdSpot,
    Gradient,
    Uniform,
}

// Mission and trajectory types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MissionStatus {
    Planned,
    Starting,
    InProgress,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrajectoryType {
    Spiral,
    Helical,
    OffsetLine,
    Grid,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct Waypoint {
    pub position: Point3D,
    #[validate(range(min = 0.0, max = 360.0))]
    pub heading_degrees: f64,
    #[validate(range(min = 0.0))]
    pub speed_ms: f64,
    pub action: WaypointAction,
    #[validate(range(min = 0.0))]
    pub dwell_time_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WaypointAction {
    FlyTo,
    Hover,
    CaptureImage,
    StartVideo,
    StopVideo,
    Land,
    Takeoff,
}

// Component types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComponentType {
    Blade,
    Tower,
    Nacelle,
    Hub,
    Spinner,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComponentSection {
    Root,
    MidSpan,
    Tip,
    LeadingEdge,
    TrailingEdge,
}

// Criticality assessment types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CriticalityLevel {
    Negligible,
    Low,
    Medium,
    High,
    Severe,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaintenanceAction {
    Monitor,
    ScheduleInspection,
    MinorRepair,
    MajorRepair,
    ComponentReplacement,
    ImmediateShutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UrgencyLevel {
    Routine,
    Scheduled,
    Priority,
    Urgent,
    Emergency,
}

// Conversion traits for protobuf interop
impl From<Point3D> for crate::pb::common::Point3D {
    fn from(point: Point3D) -> Self {
        Self {
            x: point.x,
            y: point.y,
            z: point.z,
        }
    }
}

impl From<crate::pb::common::Point3D> for Point3D {
    fn from(point: crate::pb::common::Point3D) -> Self {
        Self {
            x: point.x,
            y: point.y,
            z: point.z,
        }
    }
}

impl From<BoundingBox> for crate::pb::common::BoundingBox {
    fn from(bbox: BoundingBox) -> Self {
        Self {
            x: bbox.x,
            y: bbox.y,
            width: bbox.width,
            height: bbox.height,
        }
    }
}

impl From<crate::pb::common::BoundingBox> for BoundingBox {
    fn from(bbox: crate::pb::common::BoundingBox) -> Self {
        Self {
            x: bbox.x,
            y: bbox.y,
            width: bbox.width,
            height: bbox.height,
        }
    }
}