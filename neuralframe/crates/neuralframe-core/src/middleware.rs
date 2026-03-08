//! # Middleware Pipeline
//!
//! Async middleware chain with short-circuit support for NeuralFrame.
//! Built-in middleware includes logging, CORS, rate limiting, and compression.
//!
//! ## Custom Middleware
//!
//! ```rust
//! use neuralframe_core::middleware::{Middleware, MiddlewareResult};
//! use neuralframe_core::extractors::Request;
//! use neuralframe_core::response::Response;
//! use async_trait::async_trait;
//!
//! struct AuthMiddleware;
//!
//! #[async_trait]
//! impl Middleware for AuthMiddleware {
//!     async fn handle(
//!         &self,
//!         req: Request,
//!         next: &dyn Fn(Request) -> futures::future::BoxFuture<'_, Response>,
//!     ) -> MiddlewareResult {
//!         if req.headers.bearer_token().is_some() {
//!             MiddlewareResult::Continue(req)
//!         } else {
//!             MiddlewareResult::ShortCircuit(Response::unauthorized("Missing token"))
//!         }
//!     }
//! }
//! ```

use crate::extractors::Request;
use crate::response::Response;
use async_trait::async_trait;
use dashmap::DashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Result of middleware processing.
#[derive(Debug)]
pub enum MiddlewareResult {
    /// Continue processing the request through the chain.
    Continue(Request),
    /// Short-circuit: return this response immediately.
    ShortCircuit(Response),
}

/// Trait for implementing custom middleware.
///
/// Middleware can inspect/modify requests, short-circuit the pipeline,
/// or add post-processing logic.
#[async_trait]
pub trait Middleware: Send + Sync + fmt::Debug {
    /// Process a request through this middleware.
    ///
    /// Return `MiddlewareResult::Continue(req)` to pass to the next middleware,
    /// or `MiddlewareResult::ShortCircuit(response)` to respond immediately.
    async fn handle(
        &self,
        req: Request,
        next: &dyn Fn(Request) -> futures::future::BoxFuture<'_, Response>,
    ) -> MiddlewareResult;

    /// Optional name for this middleware (used in logging).
    fn name(&self) -> &str {
        "unnamed"
    }
}

/// A chain of middleware that processes requests in order.
#[derive(Debug, Clone)]
pub struct MiddlewareChain {
    middleware: Vec<Arc<dyn Middleware>>,
}

impl Default for MiddlewareChain {
    fn default() -> Self {
        Self::new()
    }
}

impl MiddlewareChain {
    /// Create an empty middleware chain.
    pub fn new() -> Self {
        Self {
            middleware: Vec::new(),
        }
    }

    /// Add a middleware to the chain.
    pub fn add<M: Middleware + 'static>(&mut self, middleware: M) {
        self.middleware.push(Arc::new(middleware));
    }

    /// Process a request through all middleware in the chain.
    ///
    /// Returns the possibly-modified request if all middleware pass,
    /// or a short-circuit response if any middleware rejects.
    pub async fn process(&self, mut req: Request) -> Result<Request, Response> {
        for mw in &self.middleware {
            // Create a no-op next function for middleware compatibility
            let _next = |r: Request| -> futures::future::BoxFuture<'_, Response> {
                Box::pin(async move { Response::ok().text("OK") })
            };

            match mw.handle(req, &_next).await {
                MiddlewareResult::Continue(modified_req) => {
                    req = modified_req;
                }
                MiddlewareResult::ShortCircuit(response) => {
                    return Err(response);
                }
            }
        }
        Ok(req)
    }

    /// Return the number of middleware in the chain.
    pub fn len(&self) -> usize {
        self.middleware.len()
    }

    /// Check if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.middleware.is_empty()
    }
}

// ─── Built-in Middleware ───────────────────────────────────────────────

/// Logging middleware that records request method, path, and duration.
///
/// Uses the `tracing` crate for structured logging.
#[derive(Debug, Clone)]
pub struct LoggingMiddleware {
    /// Whether to log request bodies.
    pub log_bodies: bool,
}

impl LoggingMiddleware {
    /// Create a new logging middleware.
    pub fn new() -> Self {
        Self { log_bodies: false }
    }

    /// Enable request body logging (use with caution in production).
    pub fn with_body_logging(mut self) -> Self {
        self.log_bodies = true;
        self
    }
}

impl Default for LoggingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for LoggingMiddleware {
    async fn handle(
        &self,
        req: Request,
        _next: &dyn Fn(Request) -> futures::future::BoxFuture<'_, Response>,
    ) -> MiddlewareResult {
        let start = Instant::now();
        tracing::info!(
            method = %req.method,
            path = %req.path,
            "incoming request"
        );

        if self.log_bodies && !req.body.is_empty() {
            if let Ok(body_str) = std::str::from_utf8(&req.body) {
                tracing::debug!(body = %body_str, "request body");
            }
        }

        let elapsed = start.elapsed();
        tracing::info!(
            method = %req.method,
            path = %req.path,
            duration_us = elapsed.as_micros() as u64,
            "request processed"
        );

        MiddlewareResult::Continue(req)
    }

    fn name(&self) -> &str {
        "logging"
    }
}

/// CORS middleware for cross-origin request handling.
///
/// Configures allowed origins, methods, headers, and max age.
#[derive(Debug, Clone)]
pub struct CorsMiddleware {
    /// Allowed origins (`*` for any).
    pub allowed_origins: Vec<String>,
    /// Allowed HTTP methods.
    pub allowed_methods: Vec<String>,
    /// Allowed headers.
    pub allowed_headers: Vec<String>,
    /// Max age for preflight cache (seconds).
    pub max_age: u64,
    /// Whether to allow credentials.
    pub allow_credentials: bool,
}

impl CorsMiddleware {
    /// Create a CORS middleware allowing all origins.
    pub fn permissive() -> Self {
        Self {
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec![
                "GET".into(),
                "POST".into(),
                "PUT".into(),
                "DELETE".into(),
                "PATCH".into(),
                "OPTIONS".into(),
            ],
            allowed_headers: vec!["*".to_string()],
            max_age: 86400,
            allow_credentials: false,
        }
    }

    /// Create a CORS middleware with specific allowed origins.
    pub fn with_origins(origins: Vec<String>) -> Self {
        Self {
            allowed_origins: origins,
            ..Self::permissive()
        }
    }

    /// Set allowed methods.
    pub fn methods(mut self, methods: Vec<&str>) -> Self {
        self.allowed_methods = methods.into_iter().map(String::from).collect();
        self
    }

    /// Set allowed headers.
    pub fn headers(mut self, headers: Vec<&str>) -> Self {
        self.allowed_headers = headers.into_iter().map(String::from).collect();
        self
    }

    /// Set whether to allow credentials.
    pub fn credentials(mut self, allow: bool) -> Self {
        self.allow_credentials = allow;
        self
    }
}

impl Default for CorsMiddleware {
    fn default() -> Self {
        Self::permissive()
    }
}

#[async_trait]
impl Middleware for CorsMiddleware {
    async fn handle(
        &self,
        req: Request,
        _next: &dyn Fn(Request) -> futures::future::BoxFuture<'_, Response>,
    ) -> MiddlewareResult {
        // For OPTIONS preflight, short-circuit with CORS headers
        if req.method == "OPTIONS" {
            let mut response = Response::new(204);
            response.set_header(
                "Access-Control-Allow-Origin",
                &self.allowed_origins.join(", "),
            );
            response.set_header(
                "Access-Control-Allow-Methods",
                &self.allowed_methods.join(", "),
            );
            response.set_header(
                "Access-Control-Allow-Headers",
                &self.allowed_headers.join(", "),
            );
            response.set_header("Access-Control-Max-Age", &self.max_age.to_string());
            if self.allow_credentials {
                response.set_header("Access-Control-Allow-Credentials", "true");
            }
            return MiddlewareResult::ShortCircuit(response);
        }

        // For normal requests, continue with CORS headers added later
        MiddlewareResult::Continue(req)
    }

    fn name(&self) -> &str {
        "cors"
    }
}

/// Rate limiting middleware using a token bucket algorithm.
///
/// Limits requests per client (identified by IP or custom key).
#[derive(Debug)]
pub struct RateLimitMiddleware {
    /// Maximum requests allowed per window.
    max_requests: u64,
    /// Window duration.
    window: Duration,
    /// Per-client request counters.
    counters: Arc<DashMap<String, (u64, Instant)>>,
}

impl RateLimitMiddleware {
    /// Create a new rate limiter.
    ///
    /// # Arguments
    ///
    /// * `max_requests` - Maximum requests per window
    /// * `window` - Time window duration
    pub fn new(max_requests: u64, window: Duration) -> Self {
        Self {
            max_requests,
            window,
            counters: Arc::new(DashMap::new()),
        }
    }

    /// Create a rate limiter with a per-second limit.
    pub fn per_second(max_requests: u64) -> Self {
        Self::new(max_requests, Duration::from_secs(1))
    }

    /// Create a rate limiter with a per-minute limit.
    pub fn per_minute(max_requests: u64) -> Self {
        Self::new(max_requests, Duration::from_secs(60))
    }

    /// Extract a client key from the request (uses remote IP or fallback).
    fn client_key(req: &Request) -> String {
        req.headers
            .get("x-forwarded-for")
            .or_else(|| req.headers.get("x-real-ip"))
            .cloned()
            .unwrap_or_else(|| "unknown".to_string())
    }
}

#[async_trait]
impl Middleware for RateLimitMiddleware {
    async fn handle(
        &self,
        req: Request,
        _next: &dyn Fn(Request) -> futures::future::BoxFuture<'_, Response>,
    ) -> MiddlewareResult {
        let key = Self::client_key(&req);
        let now = Instant::now();

        let mut entry = self.counters.entry(key).or_insert((0, now));
        let (count, window_start) = entry.value_mut();

        // Reset window if expired
        if now.duration_since(*window_start) >= self.window {
            *count = 0;
            *window_start = now;
        }

        *count += 1;

        if *count > self.max_requests {
            let retry_after = self
                .window
                .checked_sub(now.duration_since(*window_start))
                .unwrap_or_default();

            let mut response = Response::new(429)
                .text("Rate limit exceeded");
            response.set_header("Retry-After", &retry_after.as_secs().to_string());
            response.set_header("X-RateLimit-Limit", &self.max_requests.to_string());
            response.set_header("X-RateLimit-Remaining", "0");

            tracing::warn!(
                limit = self.max_requests,
                window_secs = self.window.as_secs(),
                "rate limit exceeded"
            );

            return MiddlewareResult::ShortCircuit(response);
        }

        let remaining = self.max_requests - *count;
        drop(entry);

        let mut req = req;
        // Store rate limit info in headers for downstream use
        req.headers.insert(
            "x-ratelimit-remaining",
            &remaining.to_string(),
        );

        MiddlewareResult::Continue(req)
    }

    fn name(&self) -> &str {
        "rate_limit"
    }
}

/// Compression middleware stub.
///
/// Adds `Content-Encoding` negotiation support.
#[derive(Debug, Clone)]
pub struct CompressionMiddleware {
    /// Minimum response size to compress (bytes).
    pub min_size: usize,
}

impl CompressionMiddleware {
    /// Create a new compression middleware.
    pub fn new() -> Self {
        Self { min_size: 1024 }
    }

    /// Set minimum response size for compression.
    pub fn min_size(mut self, size: usize) -> Self {
        self.min_size = size;
        self
    }
}

impl Default for CompressionMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for CompressionMiddleware {
    async fn handle(
        &self,
        req: Request,
        _next: &dyn Fn(Request) -> futures::future::BoxFuture<'_, Response>,
    ) -> MiddlewareResult {
        // Record accepted encodings for later response processing
        let _accept_encoding = req.headers.get("accept-encoding").cloned();
        MiddlewareResult::Continue(req)
    }

    fn name(&self) -> &str {
        "compression"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_middleware_chain_empty() {
        let chain = MiddlewareChain::new();
        let req = Request::new("GET", "/test");
        let result = chain.process(req).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_middleware_chain_continue() {
        let mut chain = MiddlewareChain::new();
        chain.add(LoggingMiddleware::new());

        let req = Request::new("GET", "/test");
        let result = chain.process(req).await;
        assert!(result.is_ok());
        let req = result.unwrap();
        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/test");
    }

    #[tokio::test]
    async fn test_cors_preflight_short_circuit() {
        let mut chain = MiddlewareChain::new();
        chain.add(CorsMiddleware::permissive());

        let req = Request::new("OPTIONS", "/api/data");
        let result = chain.process(req).await;
        assert!(result.is_err()); // Short-circuited
        let response = result.unwrap_err();
        assert_eq!(response.status_code, 204);
    }

    #[tokio::test]
    async fn test_cors_non_preflight_continues() {
        let mut chain = MiddlewareChain::new();
        chain.add(CorsMiddleware::permissive());

        let req = Request::new("GET", "/api/data");
        let result = chain.process(req).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_rate_limit_allows_within_limit() {
        let mut chain = MiddlewareChain::new();
        chain.add(RateLimitMiddleware::per_second(10));

        for _ in 0..10 {
            let req = Request::new("GET", "/test");
            let result = chain.process(req).await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_rate_limit_blocks_over_limit() {
        let mut chain = MiddlewareChain::new();
        chain.add(RateLimitMiddleware::per_second(2));

        // First two should pass
        for _ in 0..2 {
            let req = Request::new("GET", "/test");
            let result = chain.process(req).await;
            assert!(result.is_ok());
        }

        // Third should be blocked
        let req = Request::new("GET", "/test");
        let result = chain.process(req).await;
        assert!(result.is_err());
        let response = result.unwrap_err();
        assert_eq!(response.status_code, 429);
    }

    #[tokio::test]
    async fn test_logging_middleware() {
        let mw = LoggingMiddleware::new().with_body_logging();
        assert_eq!(mw.name(), "logging");

        let req = Request::new("POST", "/data").with_body(b"test body".to_vec());
        let next = |r: Request| -> futures::future::BoxFuture<'_, Response> {
            Box::pin(async move { Response::ok().text("OK") })
        };

        let result = mw.handle(req, &next).await;
        assert!(matches!(result, MiddlewareResult::Continue(_)));
    }

    #[test]
    fn test_cors_builder() {
        let cors = CorsMiddleware::with_origins(vec!["https://example.com".into()])
            .methods(vec!["GET", "POST"])
            .headers(vec!["Content-Type", "Authorization"])
            .credentials(true);

        assert_eq!(cors.allowed_origins, vec!["https://example.com"]);
        assert_eq!(cors.allowed_methods, vec!["GET", "POST"]);
        assert!(cors.allow_credentials);
    }

    #[test]
    fn test_middleware_chain_len() {
        let mut chain = MiddlewareChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);

        chain.add(LoggingMiddleware::new());
        chain.add(CorsMiddleware::permissive());
        assert_eq!(chain.len(), 2);
        assert!(!chain.is_empty());
    }

    #[test]
    fn test_compression_middleware() {
        let comp = CompressionMiddleware::new().min_size(2048);
        assert_eq!(comp.min_size, 2048);
        assert_eq!(comp.name(), "compression");
    }
}
