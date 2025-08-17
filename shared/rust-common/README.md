# UAV Common Rust Library

This library provides shared utilities, types, and gRPC definitions for the UAV Wind-Turbine Inspection Suite.

## Features

- **Configuration Management**: Environment-based configuration with validation
- **Error Handling**: Comprehensive error types with gRPC integration
- **Logging & Tracing**: Structured logging with OpenTelemetry support
- **Metrics**: Prometheus metrics collection
- **Validation**: Request validation for gRPC services
- **Types**: Common domain types with protobuf conversion
- **Utilities**: Helper functions for common operations

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
uav-common = { path = "../shared/rust-common" }

# Optional features
uav-common = { path = "../shared/rust-common", features = ["database", "kubernetes"] }
```

### Basic Setup

```rust
use uav_common::{config, logging, metrics, Result};

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration
    let config = config::load_config()?;
    
    // Initialize logging
    logging::init_logging(&config.logging, &config.tracing)?;
    
    // Initialize metrics
    let metrics = metrics::Metrics::new(&config.name)?;
    
    // Your service logic here
    
    Ok(())
}
```

### gRPC Service with Validation

```rust
use uav_common::{pb::detection::*, validation::ValidateProto, Result};
use tonic::{Request, Response, Status};

#[tonic::async_trait]
impl DefectDetectionService for MyService {
    async fn detect_defects(
        &self,
        request: Request<DetectDefectsRequest>,
    ) -> Result<Response<DetectDefectsResponse>, Status> {
        let req = request.into_inner();
        
        // Validate request
        req.validate_proto().map_err(|e| Status::invalid_argument(e.to_string()))?;
        
        // Process request
        // ...
        
        Ok(Response::new(response))
    }
}
```

## Configuration

The library supports configuration through:
- Configuration files (`config/default.toml`, `config/local.toml`, etc.)
- Environment variables (prefixed with `UAV_`)
- Command line arguments (when using clap integration)

Example configuration:

```toml
name = "trajectory-service"
version = "0.1.0"
environment = "development"

[server]
host = "0.0.0.0"
port = 50051
max_connections = 1000
timeout_seconds = 30

[logging]
level = "info"
format = "pretty"
output = "stdout"

[metrics]
enabled = true
port = 9090
path = "/metrics"

[tracing]
enabled = true
jaeger_endpoint = "http://localhost:14268/api/traces"
service_name = "trajectory-service"
sample_rate = 0.1
```

## Features

### `database`
Enables PostgreSQL database support with sqlx.

### `kubernetes`
Enables Kubernetes client support for service discovery and deployment.

## Development

To build the library:

```bash
cargo build
```

To run tests:

```bash
cargo test
```

To generate documentation:

```bash
cargo doc --open
```