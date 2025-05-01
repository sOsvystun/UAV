use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;

#[derive(Serialize, Deserialize)]
struct ExpertDefectParameters {
    defect_types: HashMap<String, DefectExpertParams>,
}

#[derive(Serialize, Deserialize)]
struct DefectExpertParams {
    membership_function: String,
    parameters: Vec<f64>,
    expert_weight: f64,
}

fn load_expert_data(file_path: &str) -> ExpertDefectParameters {
    let data = fs::read_to_string(file_path).expect("Unable to read expert data file");
    serde_json::from_str(&data).expect("JSON was not well-formatted")
}

fn membership_gaussian(x: f64, mean: f64, sigma: f64) -> f64 {
    (-((x - mean).powi(2)) / (2.0 * sigma.powi(2))).exp()
}

fn membership_trapezoidal(x: f64, a: f64, b: f64, c: f64, d: f64) -> f64 {
    if x < a || x > d {
        0.0
    } else if x < b {
        (x - a) / (b - a)
    } else if x <= c {
        1.0
    } else {
        (d - x) / (d - c)
    }
}

fn membership_triangular(x: f64, a: f64, b: f64, c: f64) -> f64 {
    if x < a || x > c {
        0.0
    } else if x < b {
        (x - a) / (b - a)
    } else {
        (c - x) / (c - b)
    }
}

fn calculate_expert_membership(defect_type: &str, value: f64, params: &ExpertDefectParameters) -> f64 {
    if let Some(defect_params) = params.defect_types.get(defect_type) {
        match defect_params.membership_function.as_str() {
            "gaussian" => membership_gaussian(value, defect_params.parameters[0], defect_params.parameters[1]),
            "trapezoidal" => membership_trapezoidal(value, defect_params.parameters[0], defect_params.parameters[1], defect_params.parameters[2], defect_params.parameters[3]),
            "triangular" => membership_triangular(value, defect_params.parameters[0], defect_params.parameters[1], defect_params.parameters[2]),
            _ => 0.0,
        }
    } else {
        0.0
    }
}

fn calculate_criticality(confidence: f64, expert_membership: f64, expert_weight: f64) -> f64 {
    (confidence * expert_weight + expert_membership) / (expert_weight + 1.0)
}

fn normalize_membership_values(memberships: &[f64]) -> Vec<f64> {
    let max = memberships.iter().cloned().fold(f64::NAN, f64::max);
    memberships.iter().map(|&val| val / max).collect()
}

fn aggregate_memberships(memberships: &[f64]) -> f64 {
    memberships.iter().sum::<f64>() / memberships.len() as f64
}

fn adjust_weights_based_on_feedback(weights: &mut HashMap<String, f64>, feedback: &HashMap<String, f64>) {
    for (defect_type, &feedback_value) in feedback.iter() {
        if let Some(weight) = weights.get_mut(defect_type) {
            *weight = (*weight + feedback_value) / 2.0;
        }
    }
}

fn fuzzy_logic_inference(defect_type: &str, value: f64, params: &ExpertDefectParameters) -> f64 {
    let expert_membership = calculate_expert_membership(defect_type, value, params);
    let confidence_values = vec![0.8, 0.85, 0.9];
    let memberships: Vec<f64> = confidence_values
        .iter()
        .map(|&conf| calculate_criticality(conf, expert_membership, params.defect_types.get(defect_type).unwrap().expert_weight))
        .collect();

    aggregate_memberships(&normalize_membership_values(&memberships))
}

fn save_updated_parameters(params: &ExpertDefectParameters, file_path: &str) {
    let data = serde_json::to_string(params).expect("Failed to serialize parameters");
    fs::write(file_path, data).expect("Unable to write file");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_membership_functions() {
        assert!(membership_gaussian(4.8, 4.8, 0.5) > 0.99);
        assert_eq!(membership_trapezoidal(4.6, 4.5, 4.7, 5.0, 5.0), 0.5);
        assert_eq!(membership_triangular(4.75, 4.6, 4.9, 5.0), 0.5);
    }

    #[test]
    fn test_calculate_criticality() {
        let criticality = calculate_criticality(0.9, 0.95, 1.5);
        assert!(criticality > 0.9);
    }

    #[test]
    fn test_fuzzy_logic_inference() {
        let mut defect_types = HashMap::new();
        defect_types.insert(
            "crack".to_string(),
            DefectExpertParams {
                membership_function: "gaussian".to_string(),
                parameters: vec![4.8, 0.5],
                expert_weight: 1.5,
            },
        );
        let params = ExpertDefectParameters { defect_types };
        let inference = fuzzy_logic_inference("crack", 4.8, &params);
        assert!(inference > 0.8);
    }
}
