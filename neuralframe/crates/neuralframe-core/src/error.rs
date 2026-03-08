//! Error types for NeuralFrame Core.
//!
//! All errors implement `std::error::Error` and provide meaningful
//! context for debugging and logging.

use std::fmt;

/// The primary error type for NeuralFrame core operations.
#[derive(Debug)]
pub enum NeuralError {
    /// Route not found for the given path and method.
    RouteNotFound {
        /// The HTTP method that was requested.
        method: String,
        /// The path that was requested.
        path: String,
    },

    /// Failed to deserialize request body.
    DeserializationError {
        /// Description of what failed to deserialize.
        context: String,
        /// The underlying error message.
        source: String,
    },

    /// Failed to serialize response body.
    SerializationError {
        /// Description of what failed to serialize.
        context: String,
        /// The underlying error message.
        source: String,
    },

    /// Middleware rejected the request.
    MiddlewareRejection {
        /// Name of the middleware that rejected.
        middleware: String,
        /// Reason for rejection.
        reason: String,
        /// HTTP status code to return.
        status_code: u16,
    },

    /// Request rate limit exceeded.
    RateLimitExceeded {
        /// Maximum allowed requests per window.
        limit: u64,
        /// Window duration in seconds.
        window_secs: u64,
    },

    /// Server binding error.
    BindError {
        /// The address that failed to bind.
        address: String,
        /// The underlying error.
        source: String,
    },

    /// Hyper/HTTP error.
    HttpError(String),

    /// I/O error.
    IoError(std::io::Error),

    /// Generic internal error.
    Internal(String),
}

impl fmt::Display for NeuralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RouteNotFound { method, path } => {
                write!(f, "no route found for {} {}", method, path)
            }
            Self::DeserializationError { context, source } => {
                write!(f, "deserialization failed ({}): {}", context, source)
            }
            Self::SerializationError { context, source } => {
                write!(f, "serialization failed ({}): {}", context, source)
            }
            Self::MiddlewareRejection {
                middleware,
                reason,
                status_code,
            } => {
                write!(
                    f,
                    "middleware '{}' rejected request ({}): {}",
                    middleware, status_code, reason
                )
            }
            Self::RateLimitExceeded { limit, window_secs } => {
                write!(
                    f,
                    "rate limit exceeded: {} requests per {}s",
                    limit, window_secs
                )
            }
            Self::BindError { address, source } => {
                write!(f, "failed to bind to {}: {}", address, source)
            }
            Self::HttpError(msg) => write!(f, "HTTP error: {}", msg),
            Self::IoError(e) => write!(f, "I/O error: {}", e),
            Self::Internal(msg) => write!(f, "internal error: {}", msg),
        }
    }
}

impl std::error::Error for NeuralError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for NeuralError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

impl From<serde_json::Error> for NeuralError {
    fn from(e: serde_json::Error) -> Self {
        Self::DeserializationError {
            context: "JSON".to_string(),
            source: e.to_string(),
        }
    }
}

/// Result type alias using [`NeuralError`].
pub type NeuralResult<T> = Result<T, NeuralError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_error_display_route_not_found() {
        let err = NeuralError::RouteNotFound {
            method: "GET".into(),
            path: "/missing".into(),
        };
        assert_eq!(err.to_string(), "no route found for GET /missing");
    }

    #[test]
    fn test_error_display_rate_limit() {
        let err = NeuralError::RateLimitExceeded {
            limit: 100,
            window_secs: 60,
        };
        assert_eq!(
            err.to_string(),
            "rate limit exceeded: 100 requests per 60s"
        );
    }

    #[test]
    fn test_error_display_middleware_rejection() {
        let err = NeuralError::MiddlewareRejection {
            middleware: "auth".into(),
            reason: "invalid token".into(),
            status_code: 401,
        };
        assert!(err.to_string().contains("auth"));
        assert!(err.to_string().contains("401"));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err = NeuralError::from(io_err);
        assert!(matches!(err, NeuralError::IoError(_)));
        assert!(err.source().is_some());
    }

    #[test]
    fn test_error_from_serde_json() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid").unwrap_err();
        let err = NeuralError::from(json_err);
        assert!(matches!(err, NeuralError::DeserializationError { .. }));
    }
}
