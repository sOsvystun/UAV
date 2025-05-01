use tonic::{Request, Response, Status};
use detection::defect_detector_server::DefectDetector;
use detection::{DetectionRequest, DetectionResponse, Defect};
use common::Point3D;
use rand::Rng;

#[derive(Debug, Default)]
pub struct DetectionService {}

#[tonic::async_trait]
impl DefectDetector for DetectionService {
    async fn detect_defects(
        &self,
        request: Request<DetectionRequest>,
    ) -> Result<Response<DetectionResponse>, Status> {
        let image_data = request.into_inner().image;
        let defects = detect_defects_yolo(&image_data.data);

        Ok(Response::new(DetectionResponse { defects }))
    }
}

fn detect_defects_yolo(image_bytes: &[u8]) -> Vec<Defect> {
    let mut rng = rand::thread_rng();
    let types = vec!["crack", "erosion", "rust", "paint_loss"];

    (0..rng.gen_range(1..10)).map(|i| {
        Defect {
            id: format!("D{:03}", i + 1),
            type_: types[rng.gen_range(0..types.len())].into(),
            confidence: rng.gen_range(0.70..0.99),
            location: Some(Point3D {
                x: rng.gen_range(0.0..100.0),
                y: rng.gen_range(0.0..100.0),
                z: rng.gen_range(0.0..50.0),
            }),
        }
    }).collect()
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
