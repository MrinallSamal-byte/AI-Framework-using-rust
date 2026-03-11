//! # Middleware Pipeline
//!
//! Async middleware chain with short-circuit support for NeuralFrame.
//! Built-in middleware includes logging, CORS, rate limiting, compression,
//! request IDs, body limits, and timeouts.
//!
//! ## Custom Middleware
//!
//! ```rust
//! use neuralframe_core::middleware::{Middleware, MiddlewareResult};
//! use neuralframe_core::extractors::Request;
//! use neuralframe_core::response::Response;
//! use async_trait::async_trait;
//!
//! #[derive(Debug)]
//! struct AuthMiddleware;
//!
//! #[async_trait]
//! impl Middleware for AuthMiddleware {
//!     async fn handle(&self, req: Request) -> MiddlewareResult {
//!         if req.headers.bearer_token().is_some() {
//!             MiddlewareResult::Continue(req)
//!         } else {
//!             MiddlewareResult::ShortCircuit(Response::unauthorized("Missing token"))
//!         }
//!     }
//! }
//! ```

use crate::extractors::Request;
use crate::response::{Response, ResponseBody};
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
#[async_trait]
pub trait Middleware: Send + Sync + fmt::Debug {
    /// Process a request through this middleware.
    async fn handle(&self, req: Request) -> MiddlewareResult;

    /// Optionally transform the response after the handler runs.
    fn post_process(&self, _req: &Request, response: Response) -> Response {
        response
    }

    /// Return the middleware name.
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
    pub async fn process(&self, mut req: Request) -> Result<Request, Response> {
        for mw in &self.middleware {
            match mw.handle(req).await {
                MiddlewareResult::Continue(modified_req) => {
                    req = modified_req;
                }
                MiddlewareResult::ShortCircuit(response) => return Err(response),
            }
        }
        Ok(req)
    }

    /// Post-process a response through middleware in reverse order.
    pub fn post_process_response(&self, req: &Request, mut response: Response) -> Response {
        for mw in self.middleware.iter().rev() {
            response = mw.post_process(req, response);
        }
        response
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

/// Logging middleware that records request method, path, and duration.
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

    /// Enable request body logging.
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
    async fn handle(&self, req: Request) -> MiddlewareResult {
        let start = Instant::now();
        tracing::info!(method = %req.method, path = %req.path, "incoming request");
        if self.log_bodies && !req.body.is_empty() {
            if let Ok(body_str) = std::str::from_utf8(&req.body) {
                tracing::debug!(body = %body_str, "request body");
            }
        }
        tracing::info!(
            method = %req.method,
            path = %req.path,
            duration_us = start.elapsed().as_micros() as u64,
            "request processed"
        );
        MiddlewareResult::Continue(req)
    }

    fn name(&self) -> &str {
        "logging"
    }
}

/// CORS middleware for cross-origin request handling.
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
    /// Create a permissive CORS middleware.
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
    async fn handle(&self, req: Request) -> MiddlewareResult {
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
        MiddlewareResult::Continue(req)
    }

    fn name(&self) -> &str {
        "cors"
    }
}

/// Rate limiting middleware using a fixed window algorithm.
#[derive(Debug)]
pub struct RateLimitMiddleware {
    max_requests: u64,
    window: Duration,
    counters: Arc<DashMap<String, (u64, Instant)>>,
}

impl RateLimitMiddleware {
    /// Create a new rate limiter.
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
    async fn handle(&self, req: Request) -> MiddlewareResult {
        let key = Self::client_key(&req);
        let now = Instant::now();
        let mut entry = self.counters.entry(key).or_insert((0, now));
        let (count, window_start) = entry.value_mut();
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
            let mut response = Response::new(429).text("Rate limit exceeded");
            response.set_header("Retry-After", &retry_after.as_secs().to_string());
            response.set_header("X-RateLimit-Limit", &self.max_requests.to_string());
            response.set_header("X-RateLimit-Remaining", "0");
            return MiddlewareResult::ShortCircuit(response);
        }
        let remaining = self.max_requests.saturating_sub(*count);
        drop(entry);
        let mut req = req;
        req.headers
            .insert("x-ratelimit-remaining", &remaining.to_string());
        MiddlewareResult::Continue(req)
    }

    fn name(&self) -> &str {
        "rate_limit"
    }
}

/// Compression middleware that gzips large responses for gzip-capable clients.
#[derive(Debug, Clone)]
pub struct CompressionMiddleware {
    /// Minimum response size to compress in bytes.
    pub min_size: usize,
}

impl CompressionMiddleware {
    /// Create a new compression middleware.
    pub fn new() -> Self {
        Self { min_size: 1024 }
    }

    /// Set the minimum response size to compress.
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
    async fn handle(&self, req: Request) -> MiddlewareResult {
        MiddlewareResult::Continue(req)
    }

    fn post_process(&self, req: &Request, mut response: Response) -> Response {
        let accepts_gzip = req
            .headers
            .get("accept-encoding")
            .map(|v| v.contains("gzip"))
            .unwrap_or(false);
        if !accepts_gzip {
            return response;
        }
        let body = response.body_bytes();
        if body.len() < self.min_size {
            return response;
        }
        use flate2::{write::GzEncoder, Compression};
        use std::io::Write;
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        if encoder.write_all(&body).is_err() {
            return response;
        }
        match encoder.finish() {
            Ok(compressed) => {
                response.body = ResponseBody::Bytes(compressed);
                response.set_header("Content-Encoding", "gzip");
                response.set_header("Vary", "Accept-Encoding");
                response
            }
            Err(_) => response,
        }
    }

    fn name(&self) -> &str {
        "compression"
    }
}

/// Middleware that ensures each request carries a request ID header.
#[derive(Debug, Clone)]
pub struct RequestIdMiddleware {
    header_name: String,
}

impl RequestIdMiddleware {
    /// Create a new request ID middleware.
    pub fn new() -> Self {
        Self {
            header_name: "x-request-id".to_string(),
        }
    }

    /// Use a custom request ID header name.
    pub fn with_header(mut self, name: &str) -> Self {
        self.header_name = name.to_lowercase();
        self
    }
}

impl Default for RequestIdMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for RequestIdMiddleware {
    async fn handle(&self, mut req: Request) -> MiddlewareResult {
        if req.headers.get(&self.header_name).is_none() {
            req.headers
                .insert(&self.header_name, &uuid::Uuid::new_v4().to_string());
        }
        MiddlewareResult::Continue(req)
    }

    fn name(&self) -> &str {
        "request_id"
    }
}

/// Middleware that rejects requests larger than a configured body size.
#[derive(Debug, Clone)]
pub struct BodyLimitMiddleware {
    max_bytes: usize,
}

impl BodyLimitMiddleware {
    /// Create a new body limit middleware.
    pub fn new(max_bytes: usize) -> Self {
        Self { max_bytes }
    }

    /// Create a new body limit middleware using megabytes.
    pub fn megabytes(mb: usize) -> Self {
        Self::new(mb.saturating_mul(1024 * 1024))
    }
}

#[async_trait]
impl Middleware for BodyLimitMiddleware {
    async fn handle(&self, req: Request) -> MiddlewareResult {
        if req.body.len() > self.max_bytes {
            return MiddlewareResult::ShortCircuit(Response::new(413).json(serde_json::json!({
                "error": "Payload Too Large",
                "message": format!("body exceeds {} bytes", self.max_bytes),
                "limit_bytes": self.max_bytes,
            })));
        }
        MiddlewareResult::Continue(req)
    }

    fn name(&self) -> &str {
        "body_limit"
    }
}

/// Middleware that annotates requests with a configured timeout budget.
#[derive(Debug, Clone)]
pub struct TimeoutMiddleware {
    timeout: Duration,
}

impl TimeoutMiddleware {
    /// Create a timeout middleware with the given duration.
    pub fn new(timeout: Duration) -> Self {
        Self { timeout }
    }

    /// Create a timeout middleware with whole seconds.
    pub fn seconds(secs: u64) -> Self {
        Self::new(Duration::from_secs(secs))
    }

    /// Create a timeout middleware with milliseconds.
    pub fn milliseconds(ms: u64) -> Self {
        Self::new(Duration::from_millis(ms))
    }
}

#[async_trait]
impl Middleware for TimeoutMiddleware {
    async fn handle(&self, mut req: Request) -> MiddlewareResult {
        req.headers
            .insert("x-timeout-ms", &self.timeout.as_millis().to_string());
        MiddlewareResult::Continue(req)
    }

    fn name(&self) -> &str {
        "timeout"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_middleware_chain_empty() {
        let chain = MiddlewareChain::new();
        let result = chain.process(Request::new("GET", "/test")).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_middleware_chain_continue() {
        let mut chain = MiddlewareChain::new();
        chain.add(LoggingMiddleware::new());
        let result = chain.process(Request::new("GET", "/test")).await;
        assert!(result.is_ok());
        let req = result.expect("request");
        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/test");
    }

    #[tokio::test]
    async fn test_cors_preflight_short_circuit() {
        let mut chain = MiddlewareChain::new();
        chain.add(CorsMiddleware::permissive());
        let result = chain.process(Request::new("OPTIONS", "/api/data")).await;
        assert!(result.is_err());
        assert_eq!(result.expect_err("response").status_code, 204);
    }

    #[tokio::test]
    async fn test_cors_non_preflight_continues() {
        let mut chain = MiddlewareChain::new();
        chain.add(CorsMiddleware::permissive());
        assert!(chain
            .process(Request::new("GET", "/api/data"))
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn test_rate_limit_allows_within_limit() {
        let mut chain = MiddlewareChain::new();
        chain.add(RateLimitMiddleware::per_second(10));
        for _ in 0..10 {
            assert!(chain.process(Request::new("GET", "/test")).await.is_ok());
        }
    }

    #[tokio::test]
    async fn test_rate_limit_blocks_over_limit() {
        let mut chain = MiddlewareChain::new();
        chain.add(RateLimitMiddleware::per_second(2));
        for _ in 0..2 {
            assert!(chain.process(Request::new("GET", "/test")).await.is_ok());
        }
        let result = chain.process(Request::new("GET", "/test")).await;
        assert!(result.is_err());
        assert_eq!(result.expect_err("response").status_code, 429);
    }

    #[tokio::test]
    async fn test_logging_middleware() {
        let mw = LoggingMiddleware::new().with_body_logging();
        let req = Request::new("POST", "/data").with_body(b"test body".to_vec());
        let result = mw.handle(req).await;
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

    #[tokio::test]
    async fn test_compression_compresses_large_body() {
        let mut chain = MiddlewareChain::new();
        chain.add(CompressionMiddleware::new().min_size(100));
        let mut req = Request::new("GET", "/data");
        req.headers.insert("accept-encoding", "gzip, deflate");
        let response = Response::ok().text(&"x".repeat(2048));
        let compressed = chain.post_process_response(&req, response);
        assert_eq!(
            compressed
                .headers
                .get("Content-Encoding")
                .map(String::as_str),
            Some("gzip")
        );
        assert!(compressed.body_bytes().len() < 2048);
    }

    #[tokio::test]
    async fn test_compression_skips_small_body() {
        let mut chain = MiddlewareChain::new();
        chain.add(CompressionMiddleware::new().min_size(1024));
        let mut req = Request::new("GET", "/data");
        req.headers.insert("accept-encoding", "gzip");
        let result = chain.post_process_response(&req, Response::ok().text("small"));
        assert!(result.headers.get("Content-Encoding").is_none());
    }

    #[tokio::test]
    async fn test_compression_skips_without_accept_header() {
        let mut chain = MiddlewareChain::new();
        chain.add(CompressionMiddleware::new().min_size(100));
        let req = Request::new("GET", "/data");
        let response = Response::ok().text(&"x".repeat(2048));
        let result = chain.post_process_response(&req, response);
        assert!(result.headers.get("Content-Encoding").is_none());
    }

    #[tokio::test]
    async fn test_request_id_generated_when_absent() {
        let mut chain = MiddlewareChain::new();
        chain.add(RequestIdMiddleware::new());
        let result = chain.process(Request::new("GET", "/test")).await;
        assert!(result
            .expect("request")
            .headers
            .get("x-request-id")
            .is_some());
    }

    #[tokio::test]
    async fn test_request_id_preserved_when_present() {
        let mut chain = MiddlewareChain::new();
        chain.add(RequestIdMiddleware::new());
        let mut req = Request::new("GET", "/test");
        req.headers.insert("x-request-id", "existing-id-123");
        let result = chain.process(req).await.expect("request");
        assert_eq!(
            result.headers.get("x-request-id").map(String::as_str),
            Some("existing-id-123")
        );
    }

    #[tokio::test]
    async fn test_body_limit_allows_small_body() {
        let mut chain = MiddlewareChain::new();
        chain.add(BodyLimitMiddleware::new(1024));
        let req = Request::new("POST", "/upload").with_body(vec![0u8; 100]);
        assert!(chain.process(req).await.is_ok());
    }

    #[tokio::test]
    async fn test_body_limit_rejects_large_body() {
        let mut chain = MiddlewareChain::new();
        chain.add(BodyLimitMiddleware::new(1024));
        let req = Request::new("POST", "/upload").with_body(vec![0u8; 1025]);
        let result = chain.process(req).await;
        assert!(result.is_err());
        assert_eq!(result.expect_err("response").status_code, 413);
    }

    #[tokio::test]
    async fn test_timeout_header_added() {
        let mut chain = MiddlewareChain::new();
        chain.add(TimeoutMiddleware::seconds(30));
        let result = chain.process(Request::new("GET", "/test")).await;
        assert!(result
            .expect("request")
            .headers
            .get("x-timeout-ms")
            .is_some());
    }
}
