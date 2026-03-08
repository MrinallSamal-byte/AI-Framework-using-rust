//! # NeuralFrame Server
//!
//! Application builder and HTTP server runner.
//!
//! ## Examples
//!
//! ```rust,no_run
//! use neuralframe_core::server::NeuralFrame;
//! use neuralframe_core::response::Response;
//!
//! #[tokio::main]
//! async fn main() {
//!     NeuralFrame::new()
//!         .get("/health", |_| async {
//!             Response::ok().json(serde_json::json!({"status": "healthy"}))
//!         })
//!         .bind("0.0.0.0:8080")
//!         .run()
//!         .await
//!         .expect("server failed to start");
//! }
//! ```

use crate::error::{NeuralError, NeuralResult};
use crate::extractors::Request;
use crate::middleware::{Middleware, MiddlewareChain};
use crate::response::Response;
use crate::router::Router;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Type alias for an async handler function.
pub type HandlerFn = Arc<
    dyn Fn(Request) -> Pin<Box<dyn Future<Output = Response> + Send>> + Send + Sync,
>;

/// The NeuralFrame application builder.
///
/// Provides a fluent API for configuring routes, middleware,
/// and server settings.
///
/// # Examples
///
/// ```rust,no_run
/// use neuralframe_core::server::NeuralFrame;
/// use neuralframe_core::response::Response;
///
/// #[tokio::main]
/// async fn main() {
///     NeuralFrame::new()
///         .get("/", |_| async { Response::ok().text("Hello!") })
///         .post("/echo", |req| async move {
///             let body = req.body_string().unwrap_or_default();
///             Response::ok().text(&body)
///         })
///         .bind("0.0.0.0:3000")
///         .run()
///         .await
///         .expect("failed to start");
/// }
/// ```
pub struct NeuralFrame {
    router: Router<HandlerFn>,
    middleware: MiddlewareChain,
    bind_address: String,
    worker_threads: Option<usize>,
    app_name: String,
}

impl NeuralFrame {
    /// Create a new NeuralFrame application.
    pub fn new() -> Self {
        Self {
            router: Router::new(),
            middleware: MiddlewareChain::new(),
            bind_address: "0.0.0.0:8080".to_string(),
            worker_threads: None,
            app_name: "NeuralFrame".to_string(),
        }
    }

    /// Set the application name (used in logging).
    pub fn name(mut self, name: &str) -> Self {
        self.app_name = name.to_string();
        self
    }

    /// Register a handler for any HTTP method and path.
    ///
    /// # Arguments
    ///
    /// * `method` - HTTP method
    /// * `path` - URL path pattern
    /// * `handler` - Async handler function
    pub fn route<F, Fut>(mut self, method: &str, path: &str, handler: F) -> Self
    where
        F: Fn(Request) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Response> + Send + 'static,
    {
        let handler: HandlerFn = Arc::new(move |req| Box::pin(handler(req)));
        self.router.add_route(method, path, handler);
        self
    }

    /// Register a GET handler.
    pub fn get<F, Fut>(self, path: &str, handler: F) -> Self
    where
        F: Fn(Request) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Response> + Send + 'static,
    {
        self.route("GET", path, handler)
    }

    /// Register a POST handler.
    pub fn post<F, Fut>(self, path: &str, handler: F) -> Self
    where
        F: Fn(Request) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Response> + Send + 'static,
    {
        self.route("POST", path, handler)
    }

    /// Register a PUT handler.
    pub fn put<F, Fut>(self, path: &str, handler: F) -> Self
    where
        F: Fn(Request) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Response> + Send + 'static,
    {
        self.route("PUT", path, handler)
    }

    /// Register a DELETE handler.
    pub fn delete<F, Fut>(self, path: &str, handler: F) -> Self
    where
        F: Fn(Request) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Response> + Send + 'static,
    {
        self.route("DELETE", path, handler)
    }

    /// Register a PATCH handler.
    pub fn patch<F, Fut>(self, path: &str, handler: F) -> Self
    where
        F: Fn(Request) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Response> + Send + 'static,
    {
        self.route("PATCH", path, handler)
    }

    /// Add middleware to the application.
    pub fn middleware<M: Middleware + 'static>(mut self, mw: M) -> Self {
        self.middleware.add(mw);
        self
    }

    /// Set the address to bind to.
    pub fn bind(mut self, address: &str) -> Self {
        self.bind_address = address.to_string();
        self
    }

    /// Set the number of Tokio worker threads.
    pub fn workers(mut self, count: usize) -> Self {
        self.worker_threads = Some(count);
        self
    }

    /// Get a reference to the router.
    pub fn router(&self) -> &Router<HandlerFn> {
        &self.router
    }

    /// Get the bind address.
    pub fn address(&self) -> &str {
        &self.bind_address
    }

    /// Process a single request through middleware and routing.
    ///
    /// This is the core request handling logic, separated from the
    /// HTTP transport layer for testability.
    pub async fn handle_request(&self, request: Request) -> Response {
        // Process through middleware chain
        let request = match self.middleware.process(request).await {
            Ok(req) => req,
            Err(response) => return response,
        };

        // Match route
        let method = &request.method;
        let path = &request.path;

        match self.router.match_route(method, path) {
            Some(route_match) => {
                let mut request = request;
                request.params = crate::extractors::PathParams::new(route_match.params);
                let handler = route_match.handler;
                handler(request).await
            }
            None => Response::not_found(&format!(
                "No route found for {} {}",
                method, path
            )),
        }
    }

    /// Start the HTTP server.
    ///
    /// This will bind to the configured address and start accepting
    /// connections. The server runs until the process is terminated.
    ///
    /// # Errors
    ///
    /// Returns an error if the server fails to bind to the address.
    pub async fn run(self) -> NeuralResult<()> {
        tracing::info!(
            app = %self.app_name,
            address = %self.bind_address,
            routes = self.router.route_count(),
            middleware = self.middleware.len(),
            "starting NeuralFrame server"
        );

        let addr: std::net::SocketAddr =
            self.bind_address.parse().map_err(|e: std::net::AddrParseError| {
                NeuralError::BindError {
                    address: self.bind_address.clone(),
                    source: e.to_string(),
                }
            })?;

        let app = Arc::new(self);

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| NeuralError::BindError {
                address: addr.to_string(),
                source: e.to_string(),
            })?;

        tracing::info!(address = %addr, "server listening");

        loop {
            let (stream, remote_addr) = listener.accept().await.map_err(|e| {
                NeuralError::IoError(e)
            })?;

            let app = Arc::clone(&app);

            tokio::spawn(async move {
                let io = hyper_util::rt::TokioIo::new(stream);
                let service = hyper::service::service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
                    let app = Arc::clone(&app);
                    async move {
                        let method = req.method().to_string();
                        let path = req.uri().path().to_string();
                        let query_string = req.uri().query().map(String::from);

                        // Extract headers
                        let mut headers = crate::extractors::Headers::new();
                        for (key, value) in req.headers() {
                            if let Ok(v) = value.to_str() {
                                headers.insert(key.as_str(), v);
                            }
                        }

                        // Collect body
                        use http_body_util::BodyExt;
                        let body_bytes = req
                            .into_body()
                            .collect()
                            .await
                            .map(|collected| collected.to_bytes().to_vec())
                            .unwrap_or_default();

                        let request = Request {
                            method,
                            path,
                            query_string,
                            headers,
                            params: crate::extractors::PathParams::default(),
                            body: body_bytes,
                        };

                        let response = app.handle_request(request).await;

                        // Convert to hyper response
                        let mut builder =
                            hyper::Response::builder().status(response.status_code);

                        for (key, value) in &response.headers {
                            builder = builder.header(key.as_str(), value.as_str());
                        }

                        let body = response.body_bytes();
                        let hyper_response = builder
                            .body(http_body_util::Full::new(bytes::Bytes::from(body)))
                            .unwrap_or_else(|_| {
                                hyper::Response::builder()
                                    .status(500)
                                    .body(http_body_util::Full::new(bytes::Bytes::from(
                                        "Internal Server Error",
                                    )))
                                    .expect("building error response should not fail")
                            });

                        Ok::<_, hyper::Error>(hyper_response)
                    }
                });

                if let Err(e) = hyper_util::server::conn::auto::Builder::new(
                    hyper_util::rt::TokioExecutor::new(),
                )
                .serve_connection(io, service)
                .await
                {
                    tracing::error!(
                        remote_addr = %remote_addr,
                        error = %e,
                        "connection error"
                    );
                }
            });
        }
    }
}

impl Default for NeuralFrame {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::middleware::LoggingMiddleware;

    #[tokio::test]
    async fn test_neuralframe_route_matching() {
        let app = NeuralFrame::new()
            .get("/hello", |_req| async {
                Response::ok().text("Hello!")
            });

        let req = Request::new("GET", "/hello");
        let resp = app.handle_request(req).await;
        assert_eq!(resp.status_code, 200);
        assert_eq!(resp.body_string().unwrap(), "Hello!");
    }

    #[tokio::test]
    async fn test_neuralframe_route_not_found() {
        let app = NeuralFrame::new()
            .get("/hello", |_req| async {
                Response::ok().text("Hello!")
            });

        let req = Request::new("GET", "/missing");
        let resp = app.handle_request(req).await;
        assert_eq!(resp.status_code, 404);
    }

    #[tokio::test]
    async fn test_neuralframe_post_handler() {
        let app = NeuralFrame::new()
            .post("/echo", |req| async move {
                let body = req.body_string().unwrap_or_default();
                Response::ok().text(&body)
            });

        let req = Request::new("POST", "/echo")
            .with_body(b"Hello, NeuralFrame!".to_vec());
        let resp = app.handle_request(req).await;
        assert_eq!(resp.status_code, 200);
        assert_eq!(resp.body_string().unwrap(), "Hello, NeuralFrame!");
    }

    #[tokio::test]
    async fn test_neuralframe_path_params() {
        let app = NeuralFrame::new()
            .get("/users/:id", |req| async move {
                let id = req.params.get("id").cloned().unwrap_or_default();
                Response::ok().json(serde_json::json!({"user_id": id}))
            });

        let req = Request::new("GET", "/users/42");
        let resp = app.handle_request(req).await;
        assert_eq!(resp.status_code, 200);
        let body = resp.body_string().unwrap();
        assert!(body.contains("42"));
    }

    #[tokio::test]
    async fn test_neuralframe_json_body() {
        use serde::Deserialize;

        #[derive(Deserialize)]
        struct Message {
            text: String,
        }

        let app = NeuralFrame::new()
            .post("/chat", |req| async move {
                match req.json::<Message>() {
                    Ok(json) => {
                        Response::ok().json(serde_json::json!({"echo": json.text}))
                    }
                    Err(_) => Response::bad_request("Invalid JSON"),
                }
            });

        let req = Request::new("POST", "/chat")
            .with_body(br#"{"text":"Hello!"}"#.to_vec());
        let resp = app.handle_request(req).await;
        assert_eq!(resp.status_code, 200);
        let body = resp.body_string().unwrap();
        assert!(body.contains("Hello!"));
    }

    #[tokio::test]
    async fn test_neuralframe_with_middleware() {
        let app = NeuralFrame::new()
            .middleware(LoggingMiddleware::new())
            .get("/test", |_req| async {
                Response::ok().text("OK")
            });

        let req = Request::new("GET", "/test");
        let resp = app.handle_request(req).await;
        assert_eq!(resp.status_code, 200);
    }

    #[tokio::test]
    async fn test_neuralframe_multiple_methods() {
        let app = NeuralFrame::new()
            .get("/resource", |_req| async {
                Response::ok().text("GET")
            })
            .post("/resource", |_req| async {
                Response::created().text("POST")
            })
            .put("/resource", |_req| async {
                Response::ok().text("PUT")
            })
            .delete("/resource", |_req| async {
                Response::no_content()
            })
            .patch("/resource", |_req| async {
                Response::ok().text("PATCH")
            });

        let resp = app.handle_request(Request::new("GET", "/resource")).await;
        assert_eq!(resp.body_string().unwrap(), "GET");

        let resp = app.handle_request(Request::new("POST", "/resource")).await;
        assert_eq!(resp.status_code, 201);

        let resp = app.handle_request(Request::new("DELETE", "/resource")).await;
        assert_eq!(resp.status_code, 204);
    }

    #[test]
    fn test_neuralframe_builder() {
        let app = NeuralFrame::new()
            .name("TestApp")
            .bind("127.0.0.1:3000")
            .workers(4);

        assert_eq!(app.address(), "127.0.0.1:3000");
        assert_eq!(app.app_name, "TestApp");
        assert_eq!(app.worker_threads, Some(4));
    }

    #[tokio::test]
    async fn test_middleware_short_circuit() {
        use crate::middleware::{CorsMiddleware};

        let app = NeuralFrame::new()
            .middleware(CorsMiddleware::permissive())
            .get("/api/data", |_req| async {
                Response::ok().text("data")
            });

        // OPTIONS preflight should be short-circuited by CORS
        let req = Request::new("OPTIONS", "/api/data");
        let resp = app.handle_request(req).await;
        assert_eq!(resp.status_code, 204);

        // Normal GET should reach the handler
        let req = Request::new("GET", "/api/data");
        let resp = app.handle_request(req).await;
        assert_eq!(resp.status_code, 200);
    }
}
