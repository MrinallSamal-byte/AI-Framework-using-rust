//! # NeuralFrame Core
//!
//! High-performance HTTP server, routing engine, middleware pipeline,
//! and typed extractors for AI-native web applications.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use neuralframe_core::prelude::*;
//!
//! #[tokio::main]
//! async fn main() {
//!     NeuralFrame::new()
//!         .get("/hello", |_req| async {
//!             Response::ok().json(serde_json::json!({"message": "Hello, NeuralFrame!"}))
//!         })
//!         .bind("0.0.0.0:8080")
//!         .run()
//!         .await
//!         .expect("server failed");
//! }
//! ```

pub mod error;
pub mod extractors;
pub mod middleware;
pub mod response;
pub mod router;
pub mod server;

/// Prelude module re-exporting commonly used types.
pub mod prelude {
    pub use crate::error::*;
    pub use crate::extractors::*;
    pub use crate::middleware::{
        BodyLimitMiddleware, CompressionMiddleware, CorsMiddleware, LoggingMiddleware, Middleware,
        MiddlewareChain, RateLimitMiddleware, RequestIdMiddleware, TimeoutMiddleware,
    };
    pub use crate::response::{IntoResponse, Response, StreamingResponse};
    pub use crate::router::{Route, Router};
    pub use crate::server::NeuralFrame;
    pub use serde::{Deserialize, Serialize};
    pub use serde_json;
}
