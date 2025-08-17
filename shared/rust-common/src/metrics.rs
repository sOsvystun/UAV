use prometheus::{
    Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramVec, IntCounter, IntCounterVec,
    IntGauge, IntGaugeVec, Opts, Registry,
};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct Metrics {
    registry: Arc<Registry>,
    
    // Request metrics
    pub requests_total: CounterVec,
    pub request_duration_seconds: HistogramVec,
    pub requests_in_flight: IntGaugeVec,
    
    // Service-specific metrics
    pub defects_detected_total: IntCounterVec,
    pub images_processed_total: IntCounterVec,
    pub trajectories_planned_total: IntCounter,
    pub reports_generated_total: IntCounter,
    
    // System metrics
    pub active_connections: IntGauge,
    pub memory_usage_bytes: Gauge,
    pub cpu_usage_percent: Gauge,
    
    // Error metrics
    pub errors_total: CounterVec,
    pub timeouts_total: CounterVec,
}

impl Metrics {
    pub fn new(service_name: &str) -> prometheus::Result<Self> {
        let registry = Arc::new(Registry::new());
        
        let requests_total = CounterVec::new(
            Opts::new("requests_total", "Total number of requests")
                .namespace(service_name),
            &["method", "endpoint", "status"],
        )?;
        
        let request_duration_seconds = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "request_duration_seconds",
                "Request duration in seconds",
            )
            .namespace(service_name)
            .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]),
            &["method", "endpoint"],
        )?;
        
        let requests_in_flight = IntGaugeVec::new(
            Opts::new("requests_in_flight", "Number of requests currently being processed")
                .namespace(service_name),
            &["method", "endpoint"],
        )?;
        
        let defects_detected_total = IntCounterVec::new(
            Opts::new("defects_detected_total", "Total number of defects detected")
                .namespace(service_name),
            &["defect_type", "severity"],
        )?;
        
        let images_processed_total = IntCounterVec::new(
            Opts::new("images_processed_total", "Total number of images processed")
                .namespace(service_name),
            &["image_type", "status"],
        )?;
        
        let trajectories_planned_total = IntCounter::new(
            "trajectories_planned_total",
            "Total number of trajectories planned",
        )?;
        
        let reports_generated_total = IntCounter::new(
            "reports_generated_total",
            "Total number of reports generated",
        )?;
        
        let active_connections = IntGauge::new(
            "active_connections",
            "Number of active connections",
        )?;
        
        let memory_usage_bytes = Gauge::new(
            "memory_usage_bytes",
            "Memory usage in bytes",
        )?;
        
        let cpu_usage_percent = Gauge::new(
            "cpu_usage_percent",
            "CPU usage percentage",
        )?;
        
        let errors_total = CounterVec::new(
            Opts::new("errors_total", "Total number of errors")
                .namespace(service_name),
            &["error_type", "endpoint"],
        )?;
        
        let timeouts_total = CounterVec::new(
            Opts::new("timeouts_total", "Total number of timeouts")
                .namespace(service_name),
            &["endpoint"],
        )?;
        
        // Register all metrics
        registry.register(Box::new(requests_total.clone()))?;
        registry.register(Box::new(request_duration_seconds.clone()))?;
        registry.register(Box::new(requests_in_flight.clone()))?;
        registry.register(Box::new(defects_detected_total.clone()))?;
        registry.register(Box::new(images_processed_total.clone()))?;
        registry.register(Box::new(trajectories_planned_total.clone()))?;
        registry.register(Box::new(reports_generated_total.clone()))?;
        registry.register(Box::new(active_connections.clone()))?;
        registry.register(Box::new(memory_usage_bytes.clone()))?;
        registry.register(Box::new(cpu_usage_percent.clone()))?;
        registry.register(Box::new(errors_total.clone()))?;
        registry.register(Box::new(timeouts_total.clone()))?;
        
        Ok(Self {
            registry,
            requests_total,
            request_duration_seconds,
            requests_in_flight,
            defects_detected_total,
            images_processed_total,
            trajectories_planned_total,
            reports_generated_total,
            active_connections,
            memory_usage_bytes,
            cpu_usage_percent,
            errors_total,
            timeouts_total,
        })
    }
    
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
    
    pub fn record_request(&self, method: &str, endpoint: &str, status: &str, duration: f64) {
        self.requests_total
            .with_label_values(&[method, endpoint, status])
            .inc();
        
        self.request_duration_seconds
            .with_label_values(&[method, endpoint])
            .observe(duration);
    }
    
    pub fn record_defect(&self, defect_type: &str, severity: &str) {
        self.defects_detected_total
            .with_label_values(&[defect_type, severity])
            .inc();
    }
    
    pub fn record_image_processed(&self, image_type: &str, status: &str) {
        self.images_processed_total
            .with_label_values(&[image_type, status])
            .inc();
    }
    
    pub fn record_error(&self, error_type: &str, endpoint: &str) {
        self.errors_total
            .with_label_values(&[error_type, endpoint])
            .inc();
    }
    
    pub fn record_timeout(&self, endpoint: &str) {
        self.timeouts_total
            .with_label_values(&[endpoint])
            .inc();
    }
    
    pub fn set_active_connections(&self, count: i64) {
        self.active_connections.set(count);
    }
    
    pub fn set_memory_usage(&self, bytes: f64) {
        self.memory_usage_bytes.set(bytes);
    }
    
    pub fn set_cpu_usage(&self, percent: f64) {
        self.cpu_usage_percent.set(percent);
    }
}

// Middleware for automatic request metrics
use std::task::{Context, Poll};
use std::pin::Pin;
use tonic::{Request, Response};
use tower::{Layer, Service};
use std::time::Instant;

#[derive(Clone)]
pub struct MetricsLayer {
    metrics: Metrics,
}

impl MetricsLayer {
    pub fn new(metrics: Metrics) -> Self {
        Self { metrics }
    }
}

impl<S> Layer<S> for MetricsLayer {
    type Service = MetricsService<S>;

    fn layer(&self, service: S) -> Self::Service {
        MetricsService {
            inner: service,
            metrics: self.metrics.clone(),
        }
    }
}

#[derive(Clone)]
pub struct MetricsService<S> {
    inner: S,
    metrics: Metrics,
}

impl<S, ReqBody> Service<Request<ReqBody>> for MetricsService<S>
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

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        let start = Instant::now();
        let method = "grpc"; // Could extract actual gRPC method
        let endpoint = req.uri().path();
        
        self.metrics.requests_in_flight
            .with_label_values(&[method, endpoint])
            .inc();
        
        let metrics = self.metrics.clone();
        let mut service = self.inner.clone();
        
        Box::pin(async move {
            let result = service.call(req).await;
            
            let duration = start.elapsed().as_secs_f64();
            let status = if result.is_ok() { "success" } else { "error" };
            
            metrics.record_request(method, endpoint, status, duration);
            metrics.requests_in_flight
                .with_label_values(&[method, endpoint])
                .dec();
            
            result
        })
    }
}