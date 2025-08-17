use validator::{Validate, ValidationError, ValidationErrors};
use crate::pb::common::*;
use crate::pb::detection::*;
use crate::pb::trajectory::*;

pub trait ValidateProto {
    fn validate_proto(&self) -> Result<(), ValidationErrors>;
}

// Image validation
impl ValidateProto for Image {
    fn validate_proto(&self) -> Result<(), ValidationErrors> {
        let mut errors = ValidationErrors::new();
        
        if self.data.is_empty() {
            errors.add("data", ValidationError::new("empty_image_data"));
        }
        
        if self.format.is_empty() {
            errors.add("format", ValidationError::new("empty_format"));
        }
        
        if !["jpeg", "jpg", "png", "tiff", "raw"].contains(&self.format.to_lowercase().as_str()) {
            errors.add("format", ValidationError::new("unsupported_format"));
        }
        
        if self.width == 0 || self.height == 0 {
            errors.add("dimensions", ValidationError::new("invalid_dimensions"));
        }
        
        // Check reasonable image size limits
        if self.data.len() > 50 * 1024 * 1024 { // 50MB limit
            errors.add("data", ValidationError::new("image_too_large"));
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// Point3D validation
impl ValidateProto for Point3D {
    fn validate_proto(&self) -> Result<(), ValidationErrors> {
        let mut errors = ValidationErrors::new();
        
        if !self.x.is_finite() {
            errors.add("x", ValidationError::new("invalid_coordinate"));
        }
        
        if !self.y.is_finite() {
            errors.add("y", ValidationError::new("invalid_coordinate"));
        }
        
        if !self.z.is_finite() {
            errors.add("z", ValidationError::new("invalid_coordinate"));
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// BoundingBox validation
impl ValidateProto for BoundingBox {
    fn validate_proto(&self) -> Result<(), ValidationErrors> {
        let mut errors = ValidationErrors::new();
        
        if self.x < 0.0 || !self.x.is_finite() {
            errors.add("x", ValidationError::new("invalid_x_coordinate"));
        }
        
        if self.y < 0.0 || !self.y.is_finite() {
            errors.add("y", ValidationError::new("invalid_y_coordinate"));
        }
        
        if self.width <= 0.0 || !self.width.is_finite() {
            errors.add("width", ValidationError::new("invalid_width"));
        }
        
        if self.height <= 0.0 || !self.height.is_finite() {
            errors.add("height", ValidationError::new("invalid_height"));
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// TurbineGeometry validation
impl ValidateProto for TurbineGeometry {
    fn validate_proto(&self) -> Result<(), ValidationErrors> {
        let mut errors = ValidationErrors::new();
        
        if self.turbine_id.is_empty() {
            errors.add("turbine_id", ValidationError::new("empty_turbine_id"));
        }
        
        if self.blade_points.is_empty() {
            errors.add("blade_points", ValidationError::new("no_blade_points"));
        }
        
        // Validate all points
        for (i, point) in self.blade_points.iter().enumerate() {
            if let Err(point_errors) = point.validate_proto() {
                for (field, field_errors) in point_errors.field_errors() {
                    for error in field_errors {
                        let mut new_error = error.clone();
                        new_error.code = format!("blade_points[{}].{}", i, error.code).into();
                        errors.add(&format!("blade_points[{}].{}", i, field), new_error);
                    }
                }
            }
        }
        
        if let Some(ref tower_base) = self.tower_base {
            if let Err(point_errors) = tower_base.validate_proto() {
                for (field, field_errors) in point_errors.field_errors() {
                    for error in field_errors {
                        let mut new_error = error.clone();
                        new_error.code = format!("tower_base.{}", error.code).into();
                        errors.add(&format!("tower_base.{}", field), new_error);
                    }
                }
            }
        }
        
        if let Some(ref nacelle_center) = self.nacelle_center {
            if let Err(point_errors) = nacelle_center.validate_proto() {
                for (field, field_errors) in point_errors.field_errors() {
                    for error in field_errors {
                        let mut new_error = error.clone();
                        new_error.code = format!("nacelle_center.{}", error.code).into();
                        errors.add(&format!("nacelle_center.{}", field), new_error);
                    }
                }
            }
        }
        
        if self.blade_length <= 0.0 || !self.blade_length.is_finite() {
            errors.add("blade_length", ValidationError::new("invalid_blade_length"));
        }
        
        if self.tower_height <= 0.0 || !self.tower_height.is_finite() {
            errors.add("tower_height", ValidationError::new("invalid_tower_height"));
        }
        
        if self.rotor_diameter <= 0.0 || !self.rotor_diameter.is_finite() {
            errors.add("rotor_diameter", ValidationError::new("invalid_rotor_diameter"));
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// WeatherConditions validation
impl ValidateProto for WeatherConditions {
    fn validate_proto(&self) -> Result<(), ValidationErrors> {
        let mut errors = ValidationErrors::new();
        
        if self.wind_speed_ms < 0.0 || !self.wind_speed_ms.is_finite() {
            errors.add("wind_speed_ms", ValidationError::new("invalid_wind_speed"));
        }
        
        if self.wind_direction_degrees < 0.0 || self.wind_direction_degrees >= 360.0 || !self.wind_direction_degrees.is_finite() {
            errors.add("wind_direction_degrees", ValidationError::new("invalid_wind_direction"));
        }
        
        if !self.temperature_celsius.is_finite() {
            errors.add("temperature_celsius", ValidationError::new("invalid_temperature"));
        }
        
        if self.humidity_percent < 0.0 || self.humidity_percent > 100.0 || !self.humidity_percent.is_finite() {
            errors.add("humidity_percent", ValidationError::new("invalid_humidity"));
        }
        
        if self.visibility_km < 0.0 || !self.visibility_km.is_finite() {
            errors.add("visibility_km", ValidationError::new("invalid_visibility"));
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// DetectDefectsRequest validation
impl ValidateProto for DetectDefectsRequest {
    fn validate_proto(&self) -> Result<(), ValidationErrors> {
        let mut errors = ValidationErrors::new();
        
        if self.request_id.is_empty() {
            errors.add("request_id", ValidationError::new("empty_request_id"));
        }
        
        if let Some(ref rgb_image) = self.rgb_image {
            if let Err(image_errors) = rgb_image.validate_proto() {
                for (field, field_errors) in image_errors.field_errors() {
                    for error in field_errors {
                        let mut new_error = error.clone();
                        new_error.code = format!("rgb_image.{}", error.code).into();
                        errors.add(&format!("rgb_image.{}", field), new_error);
                    }
                }
            }
        }
        
        if let Some(ref thermal_image) = self.thermal_image {
            if let Err(image_errors) = thermal_image.validate_proto() {
                for (field, field_errors) in image_errors.field_errors() {
                    for error in field_errors {
                        let mut new_error = error.clone();
                        new_error.code = format!("thermal_image.{}", error.code).into();
                        errors.add(&format!("thermal_image.{}", field), new_error);
                    }
                }
            }
        }
        
        if self.turbine_id.is_empty() {
            errors.add("turbine_id", ValidationError::new("empty_turbine_id"));
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// TrajectoryParameters validation
impl ValidateProto for TrajectoryParameters {
    fn validate_proto(&self) -> Result<(), ValidationErrors> {
        let mut errors = ValidationErrors::new();
        
        if self.standoff_distance_meters <= 0.0 || !self.standoff_distance_meters.is_finite() {
            errors.add("standoff_distance_meters", ValidationError::new("invalid_standoff_distance"));
        }
        
        if self.max_speed_ms <= 0.0 || !self.max_speed_ms.is_finite() {
            errors.add("max_speed_ms", ValidationError::new("invalid_max_speed"));
        }
        
        if self.min_speed_ms <= 0.0 || !self.min_speed_ms.is_finite() {
            errors.add("min_speed_ms", ValidationError::new("invalid_min_speed"));
        }
        
        if self.min_speed_ms >= self.max_speed_ms {
            errors.add("speed_range", ValidationError::new("invalid_speed_range"));
        }
        
        if self.max_acceleration_ms2 <= 0.0 || !self.max_acceleration_ms2.is_finite() {
            errors.add("max_acceleration_ms2", ValidationError::new("invalid_acceleration"));
        }
        
        if self.safety_margin_meters < 0.0 || !self.safety_margin_meters.is_finite() {
            errors.add("safety_margin_meters", ValidationError::new("invalid_safety_margin"));
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// Validation helper functions
pub fn validate_confidence(confidence: f64) -> Result<(), ValidationError> {
    if confidence < 0.0 || confidence > 1.0 || !confidence.is_finite() {
        Err(ValidationError::new("invalid_confidence"))
    } else {
        Ok(())
    }
}

pub fn validate_percentage(value: f64) -> Result<(), ValidationError> {
    if value < 0.0 || value > 100.0 || !value.is_finite() {
        Err(ValidationError::new("invalid_percentage"))
    } else {
        Ok(())
    }
}

pub fn validate_positive_number(value: f64) -> Result<(), ValidationError> {
    if value <= 0.0 || !value.is_finite() {
        Err(ValidationError::new("invalid_positive_number"))
    } else {
        Ok(())
    }
}

pub fn validate_non_negative_number(value: f64) -> Result<(), ValidationError> {
    if value < 0.0 || !value.is_finite() {
        Err(ValidationError::new("invalid_non_negative_number"))
    } else {
        Ok(())
    }
}

pub fn validate_string_not_empty(value: &str) -> Result<(), ValidationError> {
    if value.trim().is_empty() {
        Err(ValidationError::new("empty_string"))
    } else {
        Ok(())
    }
}

// Validation middleware for gRPC services
use tonic::{Request, Response, Status};

pub fn validate_request<T: ValidateProto>(request: &T) -> Result<(), Status> {
    request.validate_proto().map_err(|errors| {
        let error_messages: Vec<String> = errors
            .field_errors()
            .iter()
            .flat_map(|(field, field_errors)| {
                field_errors.iter().map(move |error| {
                    format!("{}: {}", field, error.code)
                })
            })
            .collect();
        
        Status::invalid_argument(format!("Validation failed: {}", error_messages.join(", ")))
    })
}