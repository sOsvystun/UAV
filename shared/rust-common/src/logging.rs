use crate::config::{LogFormat, LogOutput, LoggingConfig, TracingConfig};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

pub fn init_logging(logging_config: &LoggingConfig, tracing_config: &TracingConfig) -> crate::Result<()> {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&logging_config.level));

    let mut layers = Vec::new();

    // Console layer
    let console_layer = match logging_config.format {
        LogFormat::Json => {
            tracing_subscriber::fmt::layer()
                .json()
                .with_target(true)
                .with_thread_ids(true)
                .with_thread_names(true)
                .boxed()
        }
        LogFormat::Pretty => {
            tracing_subscriber::fmt::layer()
                .pretty()
                .with_target(true)
                .with_thread_ids(true)
                .with_thread_names(true)
                .boxed()
        }
        LogFormat::Compact => {
            tracing_subscriber::fmt::layer()
                .compact()
                .with_target(true)
                .boxed()
        }
    };

    layers.push(console_layer.with_filter(env_filter.clone()));

    // File layer if specified
    if let LogOutput::File(path) = &logging_config.output {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        
        let file_layer = tracing_subscriber::fmt::layer()
            .json()
            .with_writer(file)
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .boxed();
        
        layers.push(file_layer.with_filter(env_filter.clone()));
    }

    // Jaeger tracing layer
    if tracing_config.enabled {
        if let Some(jaeger_endpoint) = &tracing_config.jaeger_endpoint {
            let tracer = opentelemetry_jaeger::new_agent_pipeline()
                .with_endpoint(jaeger_endpoint)
                .with_service_name(&tracing_config.service_name)
                .install_simple()?;

            let telemetry_layer = tracing_opentelemetry::layer()
                .with_tracer(tracer)
                .boxed();
            
            layers.push(telemetry_layer.with_filter(env_filter));
        }
    }

    tracing_subscriber::registry()
        .with(layers)
        .init();

    Ok(())
}

// Structured logging macros
#[macro_export]
macro_rules! log_request {
    ($method:expr, $path:expr) => {
        tracing::info!(
            method = $method,
            path = $path,
            "Processing request"
        );
    };
    ($method:expr, $path:expr, $($field:tt)*) => {
        tracing::info!(
            method = $method,
            path = $path,
            $($field)*,
            "Processing request"
        );
    };
}

#[macro_export]
macro_rules! log_response {
    ($method:expr, $path:expr, $status:expr, $duration_ms:expr) => {
        tracing::info!(
            method = $method,
            path = $path,
            status = $status,
            duration_ms = $duration_ms,
            "Request completed"
        );
    };
}

#[macro_export]
macro_rules! log_error {
    ($error:expr) => {
        tracing::error!(
            error = %$error,
            "Error occurred"
        );
    };
    ($error:expr, $($field:tt)*) => {
        tracing::error!(
            error = %$error,
            $($field)*,
            "Error occurred"
        );
    };
}

// Correlation ID utilities
use std::task::{Context, Poll};
use std::pin::Pin;
use tonic::{Request, Response, Status};
use tower::{Layer, Service};
use uuid::Uuid;

#[derive(Clone)]
pub struct CorrelationIdLayer;

impl<S> Layer<S> for CorrelationIdLayer {
    type Service = CorrelationIdService<S>;

    fn layer(&self, service: S) -> Self::Service {
        CorrelationIdService { inner: service }
    }
}

#[derive(Clone)]
pub struct CorrelationIdService<S> {
    inner: S,
}

impl<S, ReqBody> Service<Request<ReqBody>> for CorrelationIdService<S>
where
    S: Service<Request<ReqBody>, Response = Response<tonic::body::BoxBody>, Error = std::convert::Infallible>
        + Clone
        + Send
        + 'static,
    S::Future: Send + 'static,
    ReqBody: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: Request<ReqBody>) -> Self::Future {
        let correlation_id = req
            .metadata()
            .get("x-correlation-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_else(|| {
                let id = Uuid::new_v4().to_string();
                req.metadata_mut().insert(
                    "x-correlation-id",
                    id.parse().unwrap(),
                );
                Box::leak(id.into_boxed_str())
            });

        let span = tracing::info_span!("request", correlation_id = correlation_id);
        let _enter = span.enter();

        let mut service = self.inner.clone();
        Box::pin(async move {
            service.call(req).await
        })
    }
}