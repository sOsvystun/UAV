use serde::{Deserialize, Serialize};
use serde_json::Result as JsonResult;
use std::fs;
use tonic::{Request, Response, Status};
use criticality::criticality_assessor_server::CriticalityAssessor;
use criticality::{CriticalityRequest, CriticalityResponse, DefectCriticality};
use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct CriticalityAssessorService {
    expert_weights: HashMap<String, f64>,
    historical_data: HashMap<String, Vec<f64>>,
}

#[derive(Serialize, Deserialize, Debug)]
struct ExpertData {
    defect_type: String,
    weight: f64,
}

impl CriticalityAssessorService {
    pub fn new(expert_data_file: &str, historical_data_file: &str) -> Self {
        let data = fs::read_to_string(expert_data_file).expect("Unable to read expert data file");
        let expert_data: Vec<ExpertData> = serde_json::from_str(&data).expect("JSON was not well-formatted");
        let expert_weights = expert_data.into_iter().map(|d| (d.defect_type, d.weight)).collect();

        let historical_data = Self::load_historical_data(historical_data_file).unwrap_or_default();

        CriticalityAssessorService { expert_weights, historical_data }
    }

    fn load_historical_data(file_path: &str) -> JsonResult<HashMap<String, Vec<f64>>> {
        let data = fs::read_to_string(file_path)?;
        serde_json::from_str(&data)
    }

    fn save_historical_data(&self, file_path: &str) -> JsonResult<()> {
        let data = serde_json::to_string(&self.historical_data)?;
        fs::write(file_path, data).map_err(|e| serde_json::Error::custom(e.to_string()))
    }

    fn update_historical_data(&mut self, defect_type: &str, criticality: f64) {
        self.historical_data.entry(defect_type.to_string()).or_default().push(criticality);
    }
}

#[tonic::async_trait]
impl CriticalityAssessor for CriticalityAssessorService {
    async fn assess_criticality(
        &self,
        request: Request<CriticalityRequest>,
    ) -> Result<Response<CriticalityResponse>, Status> {
        let defects = request.into_inner().defects;

        let mut ratings = Vec::new();

        for defect in defects {
            let expert_weight = self.expert_weights.get(&defect.type_).unwrap_or(&1.0);
            let historical_factor = self.calculate_historical_factor(&defect.type_);
            let criticality_index = calculate_criticality(defect.confidence, *expert_weight, historical_factor);
            let recommendation = generate_recommendation(criticality_index);

            ratings.push(DefectCriticality {
                id: defect.id,
                criticality_index,
                recommendation,
            });
        }

        Ok(Response::new(CriticalityResponse { ratings }))
    }
}

fn calculate_criticality(confidence: f64, expert_weight: f64, historical_factor: f64) -> f64 {
    (confidence * expert_weight + historical_factor) / 2.0
}

fn generate_recommendation(criticality_index: f64) -> String {
    if criticality_index >= 0.9 {
        "Immediate action required".into()
    } else if criticality_index >= 0.75 {
        "High priority, schedule maintenance".into()
    } else if criticality_index >= 0.5 {
        "Moderate priority, monitor closely".into()
    } else {
        "Low priority, routine inspection sufficient".into()
    }
}

impl CriticalityAssessorService {
    fn calculate_historical_factor(&self, defect_type: &str) -> f64 {
        if let Some(history) = self.historical_data.get(defect_type) {
            history.iter().sum::<f64>() / history.len() as f64
        } else {
            0.5
        }
    }

    pub fn periodic_update_weights(&mut self) {
        for (defect_type, history) in &self.historical_data {
            let avg_criticality = history.iter().sum::<f64>() / history.len() as f64;
            let adjusted_weight = 1.0 + avg_criticality;
            self.expert_weights.insert(defect_type.clone(), adjusted_weight);
        }
    }
}

fn update_expert_weights_from_json(service: &mut CriticalityAssessorService, json_str: &str) -> JsonResult<()> {
    let expert_data: Vec<ExpertData> = serde_json::from_str(json_str)?;
    service.expert_weights.clear();
    service.expert_weights = expert_data.into_iter().map(|d| (d.defect_type, d.weight)).collect();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_criticality() {
        assert_eq!(calculate_criticality(0.8, 1.2, 0.6), 0.84);
        assert_eq!(calculate_criticality(0.5, 1.0, 0.5), 0.5);
    }

    #[test]
    fn test_generate_recommendation() {
        assert_eq!(generate_recommendation(0.95), "Immediate action required");
        assert_eq!(generate_recommendation(0.8), "High priority, schedule maintenance");
        assert_eq!(generate_recommendation(0.6), "Moderate priority, monitor closely");
        assert_eq!(generate_recommendation(0.4), "Low priority, routine inspection sufficient");
    }

    #[test]
    fn test_update_expert_weights_from_json() {
        let mut service = CriticalityAssessorService::default();
        let json_data = r#"[
            {"defect_type": "crack", "weight": 1.5},
            {"defect_type": "erosion", "weight": 1.2}
        ]"#;

        update_expert_weights_from_json(&mut service, json_data).unwrap();

        assert_eq!(service.expert_weights.get("crack"), Some(&1.5));
        assert_eq!(service.expert_weights.get("erosion"), Some(&1.2));
    }
}