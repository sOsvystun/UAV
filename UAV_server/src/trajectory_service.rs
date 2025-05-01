use tonic::{Request, Response, Status};
use detection::defect_detector_server::DefectDetector;
use detection::{DetectionRequest, DetectionResponse, Defect};
use common::Point3D;
use rand::Rng;
use std::ffi::CString;
use std::os::raw::c_char;
use std::slice;

#[derive(Debug, Default)]
pub struct DetectionService {}

#[tonic::async_trait]
impl DefectDetector for DetectionService {
    async fn detect_defects(
        &self,
        request: Request<DetectionRequest>,
    ) -> Result<Response<DetectionResponse>, Status> {
        let image_data = request.into_inner().image;

        let defects = unsafe { detect_defects_external(&image_data.data) };

        Ok(Response::new(DetectionResponse { defects }))
    }
}

extern "C" {
    fn process_image_data(image_bytes: *const u8, image_length: usize) -> *const c_char;
}

unsafe fn detect_defects_external(image_bytes: &[u8]) -> Vec<Defect> {
    let result_ptr = process_image_data(image_bytes.as_ptr(), image_bytes.len());
    let c_str = CString::from_raw(result_ptr as *mut c_char);
    let defect_data = c_str.to_str().unwrap_or("");
    parse_defects(defect_data)
}

fn parse_defects(data: &str) -> Vec<Defect> {
    let mut defects = Vec::new();

    for line in data.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 6 {
            defects.push(Defect {
                id: parts[0].to_string(),
                type_: parts[1].to_string(),
                confidence: parts[2].parse().unwrap_or(0.0),
                location: Some(Point3D {
                    x: parts[3].parse().unwrap_or(0.0),
                    y: parts[4].parse().unwrap_or(0.0),
                    z: parts[5].parse().unwrap_or(0.0),
                }),
            });
        }
    }

    defects
}

fn classify_defect(confidence: f64) -> &'static str {
    match confidence {
        c if c >= 0.95 => "Critical",
        c if c >= 0.85 => "High",
        c if c >= 0.75 => "Moderate",
        _ => "Low",
    }
}

fn calculate_severity(defect_type: &str, confidence: f64) -> f64 {
    let base_severity = match defect_type {
        "crack" => 0.9,
        "erosion" => 0.7,
        "rust" => 0.6,
        "paint_loss" => 0.4,
        _ => 0.5,
    };
    base_severity * confidence
}

fn estimate_repair_cost(severity: f64) -> f64 {
    severity * 1000.0
}

fn aggregate_defect_information(defects: &[Defect]) -> Vec<(Defect, &'static str, f64, f64)> {
    defects.iter().map(|defect| {
        let classification = classify_defect(defect.confidence);
        let severity = calculate_severity(&defect.type_, defect.confidence);
        let repair_cost = estimate_repair_cost(severity);
        (defect.clone(), classification, severity, repair_cost)
    }).collect()
}

fn filter_critical_defects(defects: &[Defect]) -> Vec<Defect> {
    defects.iter().filter(|defect| defect.confidence >= 0.9).cloned().collect()
}

fn simulate_external_defect_analysis(image_bytes: &[u8]) -> String {
    let mut rng = rand::thread_rng();
    let types = vec!["crack", "erosion", "rust", "paint_loss"];

    (0..rng.gen_range(1..10)).map(|i| {
        let defect_type = types[rng.gen_range(0..types.len())];
        let confidence = rng.gen_range(0.70..0.99);
        let location_x = rng.gen_range(0.0..100.0);
        let location_y = rng.gen_range(0.0..100.0);
        let location_z = rng.gen_range(0.0..50.0);

        format!("D{:03},{},{},{},{},{}", i + 1, defect_type, confidence, location_x, location_y, location_z)
    }).collect::<Vec<String>>().join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_defect() {
        assert_eq!(classify_defect(0.96), "Critical");
        assert_eq!(classify_defect(0.86), "High");
        assert_eq!(classify_defect(0.76), "Moderate");
        assert_eq!(classify_defect(0.65), "Low");
    }

    #[test]
    fn test_calculate_severity() {
        assert_eq!(calculate_severity("crack", 0.9), 0.81);
        assert_eq!(calculate_severity("erosion", 0.8), 0.56);
    }

    #[test]
    fn test_estimate_repair_cost() {
        assert_eq!(estimate_repair_cost(0.81), 810.0);
    }
}
