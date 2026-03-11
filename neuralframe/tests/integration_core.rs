use neuralframe_core::middleware::MiddlewareResult;
use neuralframe_core::prelude::*;

#[tokio::test]
async fn test_full_request_cycle_with_middleware() {
    let app = NeuralFrame::new()
        .middleware(LoggingMiddleware::new())
        .middleware(RequestIdMiddleware::new())
        .get("/hello", |req| async move {
            let has_id = req.headers.get("x-request-id").is_some();
            Response::ok().json(serde_json::json!({"has_request_id": has_id}))
        });
    let resp = app.handle_request(Request::new("GET", "/hello")).await;
    assert_eq!(resp.status_code, 200);
    assert!(resp.body_string().unwrap_or_default().contains("true"));
}

#[tokio::test]
async fn test_rate_limit_enforced() {
    let app = NeuralFrame::new()
        .middleware(RateLimitMiddleware::per_second(2))
        .get("/test", |_| async { Response::ok().text("ok") });
    for _ in 0..2 {
        let resp = app.handle_request(Request::new("GET", "/test")).await;
        assert_eq!(resp.status_code, 200);
    }
    let resp = app.handle_request(Request::new("GET", "/test")).await;
    assert_eq!(resp.status_code, 429);
}

#[tokio::test]
async fn test_cors_preflight_headers() {
    let app = NeuralFrame::new()
        .middleware(CorsMiddleware::permissive())
        .get("/api", |_| async { Response::ok().text("data") });
    let resp = app.handle_request(Request::new("OPTIONS", "/api")).await;
    assert_eq!(resp.status_code, 204);
    assert!(resp.headers.contains_key("Access-Control-Allow-Origin"));
    assert!(resp.headers.contains_key("Access-Control-Allow-Methods"));
}

#[tokio::test]
async fn test_body_limit_blocks_large_payload() {
    let app = NeuralFrame::new()
        .middleware(BodyLimitMiddleware::new(100))
        .post("/upload", |_| async { Response::ok().text("ok") });
    let req = Request::new("POST", "/upload").with_body(vec![0u8; 200]);
    let resp = app.handle_request(req).await;
    assert_eq!(resp.status_code, 413);
}

#[tokio::test]
async fn test_body_limit_allows_small_payload() {
    let app = NeuralFrame::new()
        .middleware(BodyLimitMiddleware::new(1024))
        .post("/upload", |_| async { Response::ok().text("ok") });
    let req = Request::new("POST", "/upload").with_body(vec![0u8; 50]);
    let resp = app.handle_request(req).await;
    assert_eq!(resp.status_code, 200);
}

#[tokio::test]
async fn test_request_id_propagated() {
    let app = NeuralFrame::new()
        .middleware(RequestIdMiddleware::new())
        .get("/test", |req| async move {
            let id = req.headers.get("x-request-id").cloned().unwrap_or_default();
            Response::ok().json(serde_json::json!({"id": id}))
        });
    let resp = app.handle_request(Request::new("GET", "/test")).await;
    let body = resp.body_string().unwrap_or_default();
    assert!(body.contains("\"id\":"));
    let val: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
    assert!(!val["id"].as_str().unwrap_or("").is_empty());
}

#[tokio::test]
async fn test_timeout_returns_503() {
    let app = NeuralFrame::new()
        .handler_timeout(std::time::Duration::from_millis(50))
        .get("/slow", |_| async {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            Response::ok().text("never")
        });
    let resp = app.handle_request(Request::new("GET", "/slow")).await;
    assert_eq!(resp.status_code, 503);
}

#[tokio::test]
async fn test_fast_handler_not_timed_out() {
    let app = NeuralFrame::new()
        .handler_timeout(std::time::Duration::from_secs(5))
        .get("/fast", |_| async { Response::ok().text("fast") });
    let resp = app.handle_request(Request::new("GET", "/fast")).await;
    assert_eq!(resp.status_code, 200);
}

#[tokio::test]
async fn test_path_params_extracted() {
    let app = NeuralFrame::new().get("/users/:id/posts/:post_id", |req| async move {
        let uid = req.params.get("id").cloned().unwrap_or_default();
        let pid = req.params.get("post_id").cloned().unwrap_or_default();
        Response::ok().json(serde_json::json!({"uid": uid, "pid": pid}))
    });
    let resp = app
        .handle_request(Request::new("GET", "/users/42/posts/7"))
        .await;
    let body = resp.body_string().unwrap_or_default();
    assert!(body.contains("\"42\""));
    assert!(body.contains("\"7\""));
}

#[tokio::test]
async fn test_wildcard_captures_full_path() {
    let app = NeuralFrame::new().get("/files/*path", |req| async move {
        let path = req.params.get("path").cloned().unwrap_or_default();
        Response::ok().text(&path)
    });
    let resp = app
        .handle_request(Request::new("GET", "/files/docs/api/index.html"))
        .await;
    assert_eq!(
        resp.body_string().unwrap_or_default(),
        "docs/api/index.html"
    );
}

#[tokio::test]
async fn test_404_on_unknown_route() {
    let app = NeuralFrame::new().get("/known", |_| async { Response::ok().text("ok") });
    let resp = app.handle_request(Request::new("GET", "/unknown")).await;
    assert_eq!(resp.status_code, 404);
}

#[tokio::test]
async fn test_compression_applied_to_large_response() {
    let app = NeuralFrame::new()
        .middleware(CompressionMiddleware::new().min_size(100))
        .get("/large", |_| async {
            Response::ok().text(&"x".repeat(2048))
        });
    let mut req = Request::new("GET", "/large");
    req.headers.insert("accept-encoding", "gzip");
    let resp = app.handle_request(req).await;
    assert_eq!(
        resp.headers.get("Content-Encoding").map(String::as_str),
        Some("gzip")
    );
    assert!(resp.body_bytes().len() < 2048);
}

#[tokio::test]
async fn test_multiple_middleware_run_in_order() {
    use std::sync::{Arc, Mutex};

    let order = Arc::new(Mutex::new(Vec::<String>::new()));

    #[derive(Debug, Clone)]
    struct OrderMiddleware {
        name: String,
        order: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl Middleware for OrderMiddleware {
        async fn handle(&self, req: Request) -> MiddlewareResult {
            if let Ok(mut guard) = self.order.lock() {
                guard.push(self.name.clone());
            }
            MiddlewareResult::Continue(req)
        }

        fn name(&self) -> &str {
            "order"
        }
    }

    let app = NeuralFrame::new()
        .middleware(OrderMiddleware {
            name: "first".into(),
            order: Arc::clone(&order),
        })
        .middleware(OrderMiddleware {
            name: "second".into(),
            order: Arc::clone(&order),
        })
        .get("/test", |_| async { Response::ok().text("ok") });

    let _ = app.handle_request(Request::new("GET", "/test")).await;
    let recorded = order.lock().map(|g| g.clone()).unwrap_or_default();
    assert_eq!(recorded, vec!["first", "second"]);
}
