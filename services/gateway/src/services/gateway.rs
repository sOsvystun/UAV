use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tonic::{Request, Response, Status};
use tracing::{error, info, instrument, warn};
use uuid::Uuid;

use uav_common::{
    metrics::Metrics,
    pb::gateway::{
        uav_gateway_service_server::UavGatewayService,
        *,
    },
    pb::common::*,
    types::*,
    utils::*,
    Result,
};

use crate::{
    config::GatewayConfig,
    workflow::{WorkflowEngine, WorkflowState, MissionContext},
};

pub struct GatewayService {
    config: GatewayConfig,
    metrics: Metrics,
    workflow_engine: Arc<WorkflowEngine>,
    active_missions: Arc<RwLock<HashMap<String, Arc<Mutex<MissionContext>>>>>,
}

impl GatewayService {
    pub async fn new(config: GatewayConfig, metrics: Metrics) -> Result<Self> {
        config.validate()?;
        
        let workflow_engine = Arc::new(WorkflowEngine::new(config.clone()).await?);
        let active_missions = Arc::new(RwLock::new(HashMap::new()));
        
        Ok(Self {
            config,
            metrics,
            workflow_engine,
            active_missions,
        })
    }
    
    async fn get_mission_context(&self, mission_id: &str) -> Option<Arc<Mutex<MissionContext>>> {
        self.active_missions.read().await.get(mission_id).cloned()
    }
    
    async fn add_mission_context(&self, mission_id: String, context: MissionContext) {
        self.active_missions.write().await.insert(
            mission_id,
            Arc::new(Mutex::new(context))
        );
    }
    
    async fn remove_mission_context(&self, mission_id: &str) {
        self.active_missions.write().await.remove(mission_id);
    }
}

#[tonic::async_trait]
impl UavGatewayService for GatewayService {
    #[instrument(skip(self))]
    async fn health_check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> std::result::Result<Response<HealthCheckResponse>, Status> {
        let req = request.into_inner();
        
        info!(service = %req.service, "Health check requested");
        
        // Check if all downstream services are healthy
        let is_healthy = self.workflow_engine.check_services_health().await;
        
        let response = HealthCheckResponse {
            status: if is_healthy {
                health_check_response::ServingStatus::Serving as i32
            } else {
                health_check_response::ServingStatus::NotServing as i32
            },
        };
        
        Ok(Response::new(response))
    }
    
    #[instrument(skip(self))]
    async fn start_inspection_mission(
        &self,
        request: Request<StartInspectionMissionRequest>,
    ) -> std::result::Result<Response<StartInspectionMissionResponse>, Status> {
        let req = request.into_inner();
        
        info!(
            mission_id = %req.mission_id,
            turbine_id = %req.turbine_id,
            operator_id = %req.operator_id,
            "Starting inspection mission"
        );
        
        // Validate request
        if req.mission_id.is_empty() {
            return Err(Status::invalid_argument("Mission ID cannot be empty"));
        }
        
        if req.turbine_id.is_empty() {
            return Err(Status::invalid_argument("Turbine ID cannot be empty"));
        }
        
        // Check if mission already exists
        if self.get_mission_context(&req.mission_id).await.is_some() {
            return Err(Status::already_exists("Mission already in progress"));
        }
        
        // Validate turbine geometry
        if let Some(ref geometry) = req.turbine_geometry {
            if geometry.blade_points.is_empty() {
                return Err(Status::invalid_argument("Turbine geometry must include blade points"));
            }
        } else {
            return Err(Status::invalid_argument("Turbine geometry is required"));
        }
        
        // Create mission context
        let mission_context = MissionContext::new(
            req.mission_id.clone(),
            req.turbine_id.clone(),
            req.operator_id.clone(),
            req.turbine_geometry.clone(),
            req.weather_conditions.clone(),
            req.mission_parameters.clone(),
        );
        
        // Add to active missions
        self.add_mission_context(req.mission_id.clone(), mission_context).await;
        
        // Start the workflow
        match self.workflow_engine.start_mission(&req.mission_id).await {
            Ok(trajectory_plan) => {
                self.metrics.record_request("start_mission", "gateway", "success", 0.0);
                
                let response = StartInspectionMissionResponse {
                    result: Some(start_inspection_mission_response::Result::MissionResult(
                        MissionStartResult {
                            mission_id: req.mission_id,
                            initial_status: MissionStatus::Starting as i32,
                            planned_trajectory: Some(trajectory_plan),
                            estimated_duration_minutes: 25.0, // TODO: Calculate from trajectory
                        }
                    )),
                };
                
                Ok(Response::new(response))
            }
            Err(e) => {
                error!(error = %e, mission_id = %req.mission_id, "Failed to start mission");
                self.remove_mission_context(&req.mission_id).await;
                self.metrics.record_error("workflow_start_failed", "gateway");
                
                let response = StartInspectionMissionResponse {
                    result: Some(start_inspection_mission_response::Result::Error(
                        ErrorResponse {
                            code: ErrorCode::InternalError as i32,
                            message: "Failed to start mission workflow".to_string(),
                            details: e.to_string(),
                            timestamp_ms: current_timestamp_ms(),
                        }
                    )),
                };
                
                Ok(Response::new(response))
            }
        }
    }
    
    #[instrument(skip(self))]
    async fn get_mission_status(
        &self,
        request: Request<GetMissionStatusRequest>,
    ) -> std::result::Result<Response<GetMissionStatusResponse>, Status> {
        let req = request.into_inner();
        
        info!(mission_id = %req.mission_id, "Getting mission status");
        
        if req.mission_id.is_empty() {
            return Err(Status::invalid_argument("Mission ID cannot be empty"));
        }
        
        match self.get_mission_context(&req.mission_id).await {
            Some(context_arc) => {
                let context = context_arc.lock().await;
                
                let response = GetMissionStatusResponse {
                    status: context.status as i32,
                    progress: Some(MissionProgress {
                        current_stage: context.current_stage as i32,
                        overall_completion_percentage: context.completion_percentage,
                        start_timestamp_ms: context.start_timestamp_ms,
                        estimated_completion_timestamp_ms: context.estimated_completion_timestamp_ms,
                        statistics: Some(MissionStatistics {
                            total_images_captured: context.statistics.images_captured,
                            total_defects_detected: context.statistics.defects_detected,
                            critical_defects_count: context.statistics.critical_defects,
                            flight_time_minutes: context.statistics.flight_time_minutes,
                            data_processed_gb: context.statistics.data_processed_gb,
                        }),
                    }),
                    stage_statuses: context.stage_statuses.iter().map(|(stage, status)| {
                        WorkflowStageStatus {
                            stage: *stage as i32,
                            status: status.status as i32,
                            completion_percentage: status.completion_percentage,
                            status_message: status.message.clone(),
                            start_timestamp_ms: status.start_timestamp_ms,
                            end_timestamp_ms: status.end_timestamp_ms,
                        }
                    }).collect(),
                };
                
                Ok(Response::new(response))
            }
            None => {
                warn!(mission_id = %req.mission_id, "Mission not found");
                Err(Status::not_found("Mission not found"))
            }
        }
    }
    
    #[instrument(skip(self))]
    async fn stop_mission(
        &self,
        request: Request<StopMissionRequest>,
    ) -> std::result::Result<Response<StopMissionResponse>, Status> {
        let req = request.into_inner();
        
        info!(
            mission_id = %req.mission_id,
            reason = %req.reason,
            emergency = req.emergency_stop,
            "Stopping mission"
        );
        
        if req.mission_id.is_empty() {
            return Err(Status::invalid_argument("Mission ID cannot be empty"));
        }
        
        match self.workflow_engine.stop_mission(&req.mission_id, req.emergency_stop).await {
            Ok(_) => {
                self.remove_mission_context(&req.mission_id).await;
                
                let response = StopMissionResponse {
                    success: true,
                    message: "Mission stopped successfully".to_string(),
                };
                
                Ok(Response::new(response))
            }
            Err(e) => {
                error!(error = %e, mission_id = %req.mission_id, "Failed to stop mission");
                
                let response = StopMissionResponse {
                    success: false,
                    message: format!("Failed to stop mission: {}", e),
                };
                
                Ok(Response::new(response))
            }
        }
    }
    
    type ExecuteInspectionWorkflowStream = tokio_stream::wrappers::ReceiverStream<
        std::result::Result<WorkflowStatusUpdate, Status>
    >;
    
    #[instrument(skip(self))]
    async fn execute_inspection_workflow(
        &self,
        request: Request<ExecuteInspectionWorkflowRequest>,
    ) -> std::result::Result<Response<Self::ExecuteInspectionWorkflowStream>, Status> {
        let req = request.into_inner();
        
        info!(mission_id = %req.mission_id, "Executing inspection workflow");
        
        if req.mission_id.is_empty() {
            return Err(Status::invalid_argument("Mission ID cannot be empty"));
        }
        
        let mission_context = self.get_mission_context(&req.mission_id).await
            .ok_or_else(|| Status::not_found("Mission not found"))?;
        
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let workflow_engine = self.workflow_engine.clone();
        let mission_id = req.mission_id.clone();
        
        // Spawn workflow execution task
        tokio::spawn(async move {
            if let Err(e) = workflow_engine.execute_workflow(&mission_id, tx.clone()).await {
                error!(error = %e, mission_id = %mission_id, "Workflow execution failed");
                
                let error_update = WorkflowStatusUpdate {
                    mission_id: mission_id.clone(),
                    current_stage: WorkflowStage::Completion as i32,
                    completion_percentage: 0.0,
                    status_message: format!("Workflow failed: {}", e),
                    stage_data: None,
                    logs: vec![format!("ERROR: {}", e)],
                };
                
                let _ = tx.send(Ok(error_update)).await;
            }
        });
        
        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }
}