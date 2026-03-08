//! Error types for the LLM client.

use std::fmt;

/// Errors that can occur during LLM operations.
#[derive(Debug)]
pub enum LLMError {
    /// HTTP request failed.
    RequestFailed {
        /// HTTP status code.
        status: u16,
        /// Error message from the provider.
        message: String,
    },

    /// Failed to parse the provider's response.
    ParseError {
        /// Description of what failed.
        context: String,
        /// Underlying error message.
        source: String,
    },

    /// Stream connection was lost.
    StreamError(String),

    /// Authentication error (invalid API key, etc.).
    AuthError(String),

    /// Rate limit exceeded.
    RateLimited {
        /// Seconds until the rate limit resets.
        retry_after: Option<u64>,
    },

    /// Request timed out.
    Timeout {
        /// Timeout duration in seconds.
        timeout_secs: u64,
    },

    /// Token count exceeds model's context window.
    TokenLimitExceeded {
        /// Tokens in the request.
        token_count: u32,
        /// Maximum allowed tokens.
        max_tokens: u32,
    },

    /// Provider is not available.
    ProviderUnavailable(String),

    /// All providers in the failover chain failed.
    AllProvidersFailed(Vec<LLMError>),

    /// Tool execution failed.
    ToolError {
        /// Tool name.
        tool: String,
        /// Error details.
        source: String,
    },

    /// Configuration error.
    ConfigError(String),
}

impl fmt::Display for LLMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequestFailed { status, message } => {
                write!(f, "LLM request failed (HTTP {}): {}", status, message)
            }
            Self::ParseError { context, source } => {
                write!(f, "failed to parse {}: {}", context, source)
            }
            Self::StreamError(msg) => write!(f, "stream error: {}", msg),
            Self::AuthError(msg) => write!(f, "authentication error: {}", msg),
            Self::RateLimited { retry_after } => {
                if let Some(secs) = retry_after {
                    write!(f, "rate limited, retry after {}s", secs)
                } else {
                    write!(f, "rate limited")
                }
            }
            Self::Timeout { timeout_secs } => {
                write!(f, "request timed out after {}s", timeout_secs)
            }
            Self::TokenLimitExceeded {
                token_count,
                max_tokens,
            } => {
                write!(
                    f,
                    "token count {} exceeds limit {}",
                    token_count, max_tokens
                )
            }
            Self::ProviderUnavailable(name) => {
                write!(f, "provider '{}' is unavailable", name)
            }
            Self::AllProvidersFailed(errors) => {
                write!(f, "all {} providers failed", errors.len())
            }
            Self::ToolError { tool, source } => {
                write!(f, "tool '{}' failed: {}", tool, source)
            }
            Self::ConfigError(msg) => write!(f, "configuration error: {}", msg),
        }
    }
}

impl std::error::Error for LLMError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = LLMError::RequestFailed {
            status: 429,
            message: "Too many requests".into(),
        };
        assert!(err.to_string().contains("429"));

        let err = LLMError::RateLimited {
            retry_after: Some(30),
        };
        assert!(err.to_string().contains("30s"));

        let err = LLMError::TokenLimitExceeded {
            token_count: 5000,
            max_tokens: 4096,
        };
        assert!(err.to_string().contains("5000"));
    }
}
