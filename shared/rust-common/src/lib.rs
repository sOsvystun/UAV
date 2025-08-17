pub mod config;
pub mod error;
pub mod logging;
pub mod metrics;
pub mod proto;
pub mod types;
pub mod utils;
pub mod validation;

#[cfg(feature = "database")]
pub mod database;

#[cfg(feature = "kubernetes")]
pub mod kubernetes;

// Re-export commonly used types
pub use error::{Error, Result};
pub use types::*;

// Re-export proto modules for convenience
pub mod pb {
    pub mod common {
        pub use crate::proto::uav::common::v1::*;
    }
    pub mod trajectory {
        pub use crate::proto::uav::trajectory::v1::*;
    }
    pub mod detection {
        pub use crate::proto::uav::detection::v1::*;
    }
    pub mod criticality {
        pub use crate::proto::uav::criticality::v1::*;
    }
    pub mod reporting {
        pub use crate::proto::uav::reporting::v1::*;
    }
    pub mod gateway {
        pub use crate::proto::uav::gateway::v1::*;
    }
}