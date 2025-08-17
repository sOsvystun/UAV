use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),

    #[error("gRPC transport error: {0}")]
    GrpcTransport(#[from] tonic::transport::Error),

    #[error("Validation error: {0}")]
    Validation(#[from] validator::ValidationErrors),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Already exists: {0}")]
    AlreadyExists(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Service unavailable: {0}")]
    Unavailable(String),

    #[cfg(feature = "database")]
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[cfg(feature = "kubernetes")]
    #[error("Kubernetes error: {0}")]
    Kubernetes(#[from] kube::Error),
}

impl From<Error> for tonic::Status {
    fn from(err: Error) -> Self {
        match err {
            Error::NotFound(msg) => tonic::Status::not_found(msg),
            Error::AlreadyExists(msg) => tonic::Status::already_exists(msg),
            Error::PermissionDenied(msg) => tonic::Status::permission_denied(msg),
            Error::Validation(err) => tonic::Status::invalid_argument(format!("Validation failed: {}", err)),
            Error::Timeout(msg) => tonic::Status::deadline_exceeded(msg),
            Error::Unavailable(msg) => tonic::Status::unavailable(msg),
            _ => tonic::Status::internal(err.to_string()),
        }
    }
}

// Helper macros for common error patterns
#[macro_export]
macro_rules! not_found {
    ($msg:expr) => {
        $crate::Error::NotFound($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::Error::NotFound(format!($fmt, $($arg)*))
    };
}

#[macro_export]
macro_rules! internal_error {
    ($msg:expr) => {
        $crate::Error::Internal($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::Error::Internal(format!($fmt, $($arg)*))
    };
}

#[macro_export]
macro_rules! validation_error {
    ($msg:expr) => {
        $crate::Error::Parse($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::Error::Parse(format!($fmt, $($arg)*))
    };
}