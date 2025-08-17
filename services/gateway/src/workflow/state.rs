use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uav_common::{
    pb::gateway::*,
    pb::common::*,
    pb::trajectory::*,
    utils::current_timestamp_ms,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionContext {
    pub mission_id: String,
    pub turbine_id: String,
    pub operator_id: String,
    pub turbine_geometry: Option<TurbineGeometry>,
    pub weather_conditions: Option<WeatherConditions>,
    pub mission_parameters: Option<MissionParameters>,
    pub status: MissionStatus,
    pub current_stage: WorkflowStage,
    pub completion_percentage: f64,
    pub start_timestamp_ms: i64,
    pub estimated_completion_timestamp_ms: i64,
    pub stage_statuses: HashMap<WorkflowStage, StageStatus>,
    pub statistics: MissionStatistics,
    pub trajectory_plan: Option<TrajectoryPlan>,
    pub captured_images: Vec<String>, // Image file paths
    pub detected_defects: Vec<String>, // Defect IDs
    pub generated_reports: Vec<String>, // Report IDs
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageStatus {
    pub status: StageStatus,
    pub completion_percentage: f64,
    pub message: String,
    pub start_timestamp_ms: i64,
    pub end_timestamp_ms: i64,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionStatistics {
    pub images_captured: i32,
    pub defects_detected: i32,
    pub critical_defects: i32,
    pub flight_time_minutes: f64,
    pub data_processed_gb: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkflowState {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl MissionContext {
    pub fn new(
        mission_id: String,
        turbine_id: String,
        operator_id: String,
        turbine_geometry: Option<TurbineGeometry>,
        weather_conditions: Option<WeatherConditions>,
        mission_parameters: Option<MissionParameters>,
    ) -> Self {
        let now = current_timestamp_ms();
        
        Self {
            mission_id,
            turbine_id,
            operator_id,
            turbine_geometry,
            weather_conditions,
            mission_parameters,
            status: MissionStatus::Planned,
            current_stage: WorkflowStage::Initialization,
            completion_percentage: 0.0,
            start_timestamp_ms: now,
            estimated_completion_timestamp_ms: now + (25 * 60 * 1000), // 25 minutes default
            stage_statuses: HashMap::new(),
            statistics: MissionStatistics::default(),
            trajectory_plan: None,
            captured_images: Vec::new(),
            detected_defects: Vec::new(),
            generated_reports: Vec::new(),
        }
    }
    
    pub fn update_stage(&mut self, stage: WorkflowStage, status: StageStatus) {
        self.current_stage = stage;
        self.stage_statuses.insert(stage, status);
        
        // Update overall completion percentage based on stage progress
        self.completion_percentage = self.calculate_overall_progress();
        
        // Update mission status based on stage status
        match status.status {
            StageStatus::Failed => {
                self.status = MissionStatus::Failed;
            }
            StageStatus::Completed if stage == WorkflowStage::Completion => {
                self.status = MissionStatus::Completed;
                self.completion_percentage = 100.0;
            }
            _ => {
                if self.status == MissionStatus::Planned {
                    self.status = MissionStatus::InProgress;
                }
            }
        }
    }
    
    fn calculate_overall_progress(&self) -> f64 {
        let stage_weights = [
            (WorkflowStage::Initialization, 5.0),
            (WorkflowStage::TrajectoryPlanning, 10.0),
            (WorkflowStage::FlightExecution, 30.0),
            (WorkflowStage::ImageCapture, 25.0),
            (WorkflowStage::DefectDetection, 20.0),
            (WorkflowStage::CriticalityAssessment, 5.0),
            (WorkflowStage::ReportGeneration, 5.0),
        ];
        
        let mut total_progress = 0.0;
        let mut total_weight = 0.0;
        
        for (stage, weight) in stage_weights {
            total_weight += weight;
            
            if let Some(status) = self.stage_statuses.get(&stage) {
                let stage_progress = match status.status {
                    StageStatus::Completed => 100.0,
                    StageStatus::Running => status.completion_percentage,
                    StageStatus::Failed => 0.0,
                    _ => 0.0,
                };
                
                total_progress += (stage_progress / 100.0) * weight;
            }
        }
        
        if total_weight > 0.0 {
            (total_progress / total_weight) * 100.0
        } else {
            0.0
        }
    }
    
    pub fn add_captured_image(&mut self, image_path: String) {
        self.captured_images.push(image_path);
        self.statistics.images_captured = self.captured_images.len() as i32;
    }
    
    pub fn add_detected_defect(&mut self, defect_id: String, is_critical: bool) {
        self.detected_defects.push(defect_id);
        self.statistics.defects_detected = self.detected_defects.len() as i32;
        
        if is_critical {
            self.statistics.critical_defects += 1;
        }
    }
    
    pub fn add_generated_report(&mut self, report_id: String) {
        self.generated_reports.push(report_id);
    }
    
    pub fn update_flight_time(&mut self, additional_minutes: f64) {
        self.statistics.flight_time_minutes += additional_minutes;
    }
    
    pub fn update_data_processed(&mut self, additional_gb: f64) {
        self.statistics.data_processed_gb += additional_gb;
    }
}

impl Default for MissionStatistics {
    fn default() -> Self {
        Self {
            images_captured: 0,
            defects_detected: 0,
            critical_defects: 0,
            flight_time_minutes: 0.0,
            data_processed_gb: 0.0,
        }
    }
}

impl Default for StageStatus {
    fn default() -> Self {
        Self {
            status: StageStatus::Pending,
            completion_percentage: 0.0,
            message: String::new(),
            start_timestamp_ms: current_timestamp_ms(),
            end_timestamp_ms: 0,
            error_message: None,
        }
    }
}

impl StageStatus {
    pub fn new_pending(message: String) -> Self {
        Self {
            status: StageStatus::Pending,
            message,
            ..Default::default()
        }
    }
    
    pub fn new_running(message: String) -> Self {
        Self {
            status: StageStatus::Running,
            message,
            start_timestamp_ms: current_timestamp_ms(),
            ..Default::default()
        }
    }
    
    pub fn new_completed(message: String) -> Self {
        Self {
            status: StageStatus::Completed,
            completion_percentage: 100.0,
            message,
            end_timestamp_ms: current_timestamp_ms(),
            ..Default::default()
        }
    }
    
    pub fn new_failed(message: String, error: String) -> Self {
        Self {
            status: StageStatus::Failed,
            message,
            end_timestamp_ms: current_timestamp_ms(),
            error_message: Some(error),
            ..Default::default()
        }
    }
    
    pub fn update_progress(&mut self, percentage: f64, message: Option<String>) {
        self.completion_percentage = percentage.clamp(0.0, 100.0);
        
        if let Some(msg) = message {
            self.message = msg;
        }
        
        if percentage >= 100.0 {
            self.status = StageStatus::Completed;
            self.end_timestamp_ms = current_timestamp_ms();
        }
    }
}