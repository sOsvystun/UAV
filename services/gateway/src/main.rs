mod config;
mod handlers;
mod middleware;
mod services;
mod workflow;

use anyhow::Result;
use clap::Parser;
use std::net::SocketAddr;
use tonic::transport::Server;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tracing::{info, warn};

use uav_common::{
    config::ServiceConfig,
    logging::{init_logging, CorrelationIdLayer},
    metrics::{Metrics, MetricsLayer},
    pb::gateway::uav_gateway_service_server::UavGatewayServiceServer,
};

use crate::{
    config::GatewayConfig,
    handlers::health::HealthHandler,
    middleware::AuthLayer,
    services::gateway::GatewayService,
};

#[derive(Parser)]
#[command(name = "uav-gateway")]
#[command(about = "UAV Gateway Service - orchestrates inspection workflows")]
struct Args {
    #[arg(short, long, default_value = "config/gateway.toml")]
    config: String,
    
    #[arg(short, long)]
    port: Option<u16>,
    
    #[arg(long)]
    metrics_port: Option<u16>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Load configuration
    let mut config = GatewayConfig::load(&args.config)?;
    
    // Override with CLI args
    if let Some(port) = args.port {
        config.service.server.port = port;
    }
    if let Some(metrics_port) = args.metrics_port {
        config.service.metrics.port = metrics_port;
    }
    
    // Initialize logging and tracing
    init_logging(&config.service.logging, &config.service.tracing)?;
    
    info!(
        service = %config.service.name,
        version = %config.service.version,
        port = config.service.server.port,
        "Starting UAV Gateway Service"
    );
    
    // Initialize metrics
    let metrics = Metrics::new(&config.service.name)?;
    
    // Create service instances
    let gateway_service = GatewayService::new(config.clone(), metrics.clone()).await?;
    
    // Build gRPC server
    let grpc_addr: SocketAddr = format!("{}:{}", 
        config.service.server.host, 
        config.service.server.port
    ).parse()?;
    
    let grpc_server = Server::builder()
        .layer(
            ServiceBuilder::new()
                .layer(CorrelationIdLayer)
                .layer(MetricsLayer::new(metrics.clone()))
                .layer(AuthLayer::new())
                .layer(CorsLayer::permissive())
        )
        .add_service(UavGatewayServiceServer::new(gateway_service))
        .add_service(tonic_health::server::health_reporter())
        .add_service(tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(tonic_reflection::server::FILE_DESCRIPTOR_SET)
            .build()?)
        .serve(grpc_addr);
    
    // Build HTTP server for health checks and metrics
    let health_handler = HealthHandler::new();
    let http_app = axum::Router::new()
        .route("/health", axum::routing::get(health_handler.health_check))
        .route("/ready", axum::routing::get(health_handler.readiness_check))
        .route("/metrics", axum::routing::get(move || async move {
            use prometheus::Encoder;
            let encoder = prometheus::TextEncoder::new();
            let metric_families = metrics.registry().gather();
            match encoder.encode_to_string(&metric_families) {
                Ok(output) => axum::response::Response::builder()
                    .header("content-type", "text/plain; version=0.0.4")
                    .body(output)
                    .unwrap(),
                Err(e) => {
                    warn!(error = %e, "Failed to encode metrics");
                    axum::response::Response::builder()
                        .status(500)
                        .body("Failed to encode metrics".to_string())
                        .unwrap()
                }
            }
        }));
    
    let http_addr: SocketAddr = format!("{}:{}", 
        config.service.server.host, 
        config.service.metrics.port
    ).parse()?;
    
    let http_server = axum::serve(
        tokio::net::TcpListener::bind(http_addr).await?,
        http_app
    );
    
    info!(
        grpc_addr = %grpc_addr,
        http_addr = %http_addr,
        "Servers starting"
    );
    
    // Run both servers concurrently
    tokio::select! {
        result = grpc_server => {
            if let Err(e) = result {
                warn!(error = %e, "gRPC server failed");
            }
        }
        result = http_server => {
            if let Err(e) = result {
                warn!(error = %e, "HTTP server failed");
            }
        }
    }
    
    info!("Shutting down UAV Gateway Service");
    Ok(())
}