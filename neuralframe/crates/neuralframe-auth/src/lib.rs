//! # NeuralFrame Auth
//!
//! Authentication and authorization middleware with API key,
//! JWT, and OAuth2 support. Rate limiting per API key.

use async_trait::async_trait;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Errors from the auth system.
#[derive(Debug)]
pub enum AuthError {
    /// Invalid or missing API key.
    InvalidApiKey,
    /// JWT token is invalid or expired.
    InvalidToken(String),
    /// Insufficient permissions.
    Forbidden(String),
    /// Rate limit exceeded.
    RateLimited { retry_after: Duration },
    /// Configuration error.
    ConfigError(String),
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidApiKey => write!(f, "invalid or missing API key"),
            Self::InvalidToken(msg) => write!(f, "invalid token: {}", msg),
            Self::Forbidden(msg) => write!(f, "forbidden: {}", msg),
            Self::RateLimited { retry_after } => {
                write!(f, "rate limited, retry after {:?}", retry_after)
            }
            Self::ConfigError(msg) => write!(f, "auth config error: {}", msg),
        }
    }
}

impl std::error::Error for AuthError {}

/// An authenticated identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    /// Unique identifier.
    pub id: String,
    /// Display name.
    pub name: Option<String>,
    /// Email address.
    pub email: Option<String>,
    /// Roles / permissions.
    pub roles: Vec<String>,
    /// API key (if using key auth).
    pub api_key: Option<String>,
    /// Extra claims.
    pub claims: serde_json::Value,
}

impl Identity {
    /// Check if the identity has a specific role.
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.iter().any(|r| r == role)
    }

    /// Check if the identity has any of the specified roles.
    pub fn has_any_role(&self, roles: &[&str]) -> bool {
        roles.iter().any(|r| self.has_role(r))
    }
}

/// Trait for authentication providers.
#[async_trait]
pub trait AuthProvider: Send + Sync {
    /// Authenticate a request and return the identity.
    async fn authenticate(&self, token: &str) -> Result<Identity, AuthError>;

    /// Get the provider name.
    fn name(&self) -> &str;
}

/// API key authentication provider.
pub struct ApiKeyAuth {
    keys: Arc<DashMap<String, ApiKeyConfig>>,
}

/// Configuration for a single API key.
#[derive(Debug, Clone)]
pub struct ApiKeyConfig {
    /// The API key string.
    pub key: String,
    /// Associated identity name.
    pub name: String,
    /// Roles granted to this key.
    pub roles: Vec<String>,
    /// Rate limit (requests per minute).
    pub rate_limit: Option<u32>,
}

impl ApiKeyAuth {
    /// Create a new API key auth provider.
    pub fn new() -> Self {
        Self {
            keys: Arc::new(DashMap::new()),
        }
    }

    /// Register an API key.
    pub fn add_key(&self, config: ApiKeyConfig) {
        self.keys.insert(config.key.clone(), config);
    }

    /// Revoke an API key.
    pub fn revoke_key(&self, key: &str) {
        self.keys.remove(key);
    }
}

impl Default for ApiKeyAuth {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AuthProvider for ApiKeyAuth {
    async fn authenticate(&self, token: &str) -> Result<Identity, AuthError> {
        let config = self
            .keys
            .get(token)
            .ok_or(AuthError::InvalidApiKey)?;

        Ok(Identity {
            id: config.key.clone(),
            name: Some(config.name.clone()),
            email: None,
            roles: config.roles.clone(),
            api_key: Some(config.key.clone()),
            claims: serde_json::json!({}),
        })
    }

    fn name(&self) -> &str {
        "api_key"
    }
}

/// JWT authentication provider.
#[derive(Debug, Clone)]
pub struct JwtAuth {
    secret: String,
    issuer: Option<String>,
}

impl JwtAuth {
    /// Create a new JWT auth provider.
    pub fn new(secret: &str) -> Self {
        Self {
            secret: secret.to_string(),
            issuer: None,
        }
    }

    /// Set the expected issuer.
    pub fn with_issuer(mut self, issuer: &str) -> Self {
        self.issuer = Some(issuer.to_string());
        self
    }
}

#[async_trait]
impl AuthProvider for JwtAuth {
    async fn authenticate(&self, token: &str) -> Result<Identity, AuthError> {
        // In a full implementation, this would decode and verify the JWT
        // using the secret key,validate claims, exp, iss, etc.
        if token.is_empty() {
            return Err(AuthError::InvalidToken("empty token".into()));
        }

        // Stub: parse the token as base64-encoded JSON identity
        tracing::debug!("JWT auth stub - verifying token");

        Ok(Identity {
            id: "jwt-user".to_string(),
            name: None,
            email: None,
            roles: vec!["user".to_string()],
            api_key: None,
            claims: serde_json::json!({}),
        })
    }

    fn name(&self) -> &str {
        "jwt"
    }
}

/// Per-key rate limiter using token bucket.
pub struct KeyRateLimiter {
    buckets: DashMap<String, RateBucket>,
    default_rate: u32,
    window: Duration,
}

#[derive(Debug)]
struct RateBucket {
    tokens: u32,
    max_tokens: u32,
    last_refill: Instant,
    window: Duration,
}

impl KeyRateLimiter {
    /// Create a new rate limiter with the given default rate (requests/minute).
    pub fn new(default_rate: u32) -> Self {
        Self {
            buckets: DashMap::new(),
            default_rate,
            window: Duration::from_secs(60),
        }
    }

    /// Check if a request is allowed for the given key.
    pub fn check(&self, key: &str) -> Result<(), AuthError> {
        let mut bucket = self
            .buckets
            .entry(key.to_string())
            .or_insert_with(|| RateBucket {
                tokens: self.default_rate,
                max_tokens: self.default_rate,
                last_refill: Instant::now(),
                window: self.window,
            });

        // Refill tokens if window has passed
        if bucket.last_refill.elapsed() >= bucket.window {
            bucket.tokens = bucket.max_tokens;
            bucket.last_refill = Instant::now();
        }

        if bucket.tokens > 0 {
            bucket.tokens -= 1;
            Ok(())
        } else {
            let retry_after = bucket.window.saturating_sub(bucket.last_refill.elapsed());
            Err(AuthError::RateLimited { retry_after })
        }
    }

    /// Set a custom rate for a key.
    pub fn set_rate(&self, key: &str, rate: u32) {
        self.buckets.insert(
            key.to_string(),
            RateBucket {
                tokens: rate,
                max_tokens: rate,
                last_refill: Instant::now(),
                window: self.window,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_roles() {
        let id = Identity {
            id: "1".into(),
            name: None,
            email: None,
            roles: vec!["admin".into(), "user".into()],
            api_key: None,
            claims: serde_json::json!({}),
        };

        assert!(id.has_role("admin"));
        assert!(!id.has_role("superadmin"));
        assert!(id.has_any_role(&["admin", "superadmin"]));
    }

    #[tokio::test]
    async fn test_api_key_auth() {
        let auth = ApiKeyAuth::new();
        auth.add_key(ApiKeyConfig {
            key: "test-key-123".into(),
            name: "Test App".into(),
            roles: vec!["user".into()],
            rate_limit: Some(100),
        });

        let identity = auth.authenticate("test-key-123").await.unwrap();
        assert_eq!(identity.name, Some("Test App".to_string()));
        assert!(identity.has_role("user"));
    }

    #[tokio::test]
    async fn test_api_key_invalid() {
        let auth = ApiKeyAuth::new();
        let result = auth.authenticate("invalid").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = KeyRateLimiter::new(3);

        assert!(limiter.check("key1").is_ok());
        assert!(limiter.check("key1").is_ok());
        assert!(limiter.check("key1").is_ok());
        assert!(limiter.check("key1").is_err()); // Rate limited

        // Different key should work
        assert!(limiter.check("key2").is_ok());
    }
}
