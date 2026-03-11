//! # Response Types
//!
//! HTTP response types including streaming responses for LLM token output.
//!
//! ## Examples
//!
//! ```rust
//! use neuralframe_core::response::Response;
//!
//! // Simple JSON response
//! let resp = Response::ok().json(serde_json::json!({"status": "ok"}));
//!
//! // Error response
//! let resp = Response::not_found("Resource not found");
//!
//! // Streaming response for LLM tokens
//! let resp = Response::ok().sse_stream(vec!["token1", "token2"]);
//! ```

use serde::Serialize;
use std::collections::HashMap;
use std::fmt;

/// An HTTP response.
#[derive(Debug, Clone)]
pub struct Response {
    /// HTTP status code.
    pub status_code: u16,
    /// Response headers.
    pub headers: HashMap<String, String>,
    /// Response body.
    pub body: ResponseBody,
}

/// The body of an HTTP response.
#[derive(Debug, Clone)]
pub enum ResponseBody {
    /// Empty body.
    Empty,
    /// Text body.
    Text(String),
    /// Binary body.
    Bytes(Vec<u8>),
    /// Server-Sent Events stream with collected event data.
    SseEvents(Vec<String>),
}

impl Response {
    /// Create a new response with the given status code.
    pub fn new(status_code: u16) -> Self {
        Self {
            status_code,
            headers: HashMap::new(),
            body: ResponseBody::Empty,
        }
    }

    /// Create a 200 OK response.
    pub fn ok() -> Self {
        Self::new(200)
    }

    /// Create a 201 Created response.
    pub fn created() -> Self {
        Self::new(201)
    }

    /// Create a 204 No Content response.
    pub fn no_content() -> Self {
        Self::new(204)
    }

    /// Create a 400 Bad Request response with a message.
    pub fn bad_request(message: &str) -> Self {
        Self::new(400).json(serde_json::json!({
            "error": "Bad Request",
            "message": message
        }))
    }

    /// Create a 401 Unauthorized response with a message.
    pub fn unauthorized(message: &str) -> Self {
        Self::new(401).json(serde_json::json!({
            "error": "Unauthorized",
            "message": message
        }))
    }

    /// Create a 403 Forbidden response with a message.
    pub fn forbidden(message: &str) -> Self {
        Self::new(403).json(serde_json::json!({
            "error": "Forbidden",
            "message": message
        }))
    }

    /// Create a 404 Not Found response with a message.
    pub fn not_found(message: &str) -> Self {
        Self::new(404).json(serde_json::json!({
            "error": "Not Found",
            "message": message
        }))
    }

    /// Create a 500 Internal Server Error response.
    pub fn internal_error(message: &str) -> Self {
        Self::new(500).json(serde_json::json!({
            "error": "Internal Server Error",
            "message": message
        }))
    }

    /// Set the body as JSON.
    pub fn json<T: Serialize>(mut self, value: T) -> Self {
        match serde_json::to_vec(&value) {
            Ok(bytes) => {
                self.body = ResponseBody::Bytes(bytes);
                self.headers
                    .insert("Content-Type".to_string(), "application/json".to_string());
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to serialize JSON response");
                self.status_code = 500;
                self.body = ResponseBody::Text("Internal serialization error".to_string());
            }
        }
        self
    }

    /// Set the body as plain text.
    pub fn text(mut self, text: &str) -> Self {
        self.body = ResponseBody::Text(text.to_string());
        self.headers
            .insert("Content-Type".to_string(), "text/plain".to_string());
        self
    }

    /// Set the body as HTML.
    pub fn html(mut self, html: &str) -> Self {
        self.body = ResponseBody::Text(html.to_string());
        self.headers
            .insert("Content-Type".to_string(), "text/html".to_string());
        self
    }

    /// Set this response as an SSE stream with the given events.
    ///
    /// Each item is formatted as an SSE `data:` line. Accepts any iterator
    /// of string-like values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use neuralframe_core::response::Response;
    ///
    /// let events = vec!["hello", "world"];
    /// let resp = Response::ok().sse_stream(events);
    /// assert!(resp.is_streaming());
    /// ```
    pub fn sse_stream(mut self, events: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let collected: Vec<String> = events.into_iter().map(|e| e.into()).collect();
        self.body = ResponseBody::SseEvents(collected);
        self.headers
            .insert("Content-Type".to_string(), "text/event-stream".to_string());
        self.headers
            .insert("Cache-Control".to_string(), "no-cache".to_string());
        self.headers
            .insert("Connection".to_string(), "keep-alive".to_string());
        self
    }

    /// Set a response header.
    pub fn set_header(&mut self, key: &str, value: &str) {
        self.headers.insert(key.to_string(), value.to_string());
    }

    /// Set a response header (builder pattern).
    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.set_header(key, value);
        self
    }

    /// Get the body as bytes.
    pub fn body_bytes(&self) -> Vec<u8> {
        match &self.body {
            ResponseBody::Empty => Vec::new(),
            ResponseBody::Text(text) => text.as_bytes().to_vec(),
            ResponseBody::Bytes(bytes) => bytes.clone(),
            ResponseBody::SseEvents(events) => {
                let mut buf = String::new();
                for event in events {
                    buf.push_str(&format!("data: {}\n\n", event));
                }
                buf.into_bytes()
            }
        }
    }

    /// Get the body as a string (if possible).
    pub fn body_string(&self) -> Option<String> {
        match &self.body {
            ResponseBody::Empty => Some(String::new()),
            ResponseBody::Text(text) => Some(text.clone()),
            ResponseBody::Bytes(bytes) => String::from_utf8(bytes.clone()).ok(),
            ResponseBody::SseEvents(events) => {
                let mut buf = String::new();
                for event in events {
                    buf.push_str(&format!("data: {}\n\n", event));
                }
                Some(buf)
            }
        }
    }

    /// Check if this is a streaming response.
    pub fn is_streaming(&self) -> bool {
        matches!(self.body, ResponseBody::SseEvents(_))
    }

    /// Get the content length.
    pub fn content_length(&self) -> usize {
        match &self.body {
            ResponseBody::Empty => 0,
            ResponseBody::Text(text) => text.len(),
            ResponseBody::Bytes(bytes) => bytes.len(),
            ResponseBody::SseEvents(events) => events
                .iter()
                .map(|e| "data: ".len() + e.len() + "\n\n".len())
                .sum(),
        }
    }
}

impl fmt::Display for Response {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Response({}, {} bytes)",
            self.status_code,
            self.content_length()
        )
    }
}

/// Trait for types that can be converted into a `Response`.
pub trait IntoResponse {
    /// Convert this type into a `Response`.
    fn into_response(self) -> Response;
}

impl IntoResponse for Response {
    fn into_response(self) -> Response {
        self
    }
}

impl IntoResponse for String {
    fn into_response(self) -> Response {
        Response::ok().text(&self)
    }
}

impl IntoResponse for &str {
    fn into_response(self) -> Response {
        Response::ok().text(self)
    }
}

impl IntoResponse for serde_json::Value {
    fn into_response(self) -> Response {
        Response::ok().json(self)
    }
}

impl<T: Serialize> IntoResponse for (u16, T) {
    fn into_response(self) -> Response {
        Response::new(self.0).json(self.1)
    }
}

/// A streaming response wrapper that emits SSE events.
///
/// Used for streaming LLM tokens back to the client.
///
/// # Examples
///
/// ```rust
/// use neuralframe_core::response::StreamingResponse;
///
/// let stream = StreamingResponse::new();
/// let event = stream.format_event("token", r#"{"content":"Hello"}"#);
/// assert!(event.contains("event: token"));
/// ```
#[derive(Debug, Clone)]
pub struct StreamingResponse {
    /// SSE event counter.
    event_id: u64,
}

impl StreamingResponse {
    /// Create a new streaming response.
    pub fn new() -> Self {
        Self { event_id: 0 }
    }

    /// Format a single SSE event.
    ///
    /// # Arguments
    ///
    /// * `event_type` - The event type name
    /// * `data` - The event data payload
    pub fn format_event(&self, event_type: &str, data: &str) -> String {
        let mut event = String::new();
        event.push_str(&format!("id: {}\n", self.event_id));
        event.push_str(&format!("event: {}\n", event_type));
        for line in data.lines() {
            event.push_str(&format!("data: {}\n", line));
        }
        event.push('\n');
        event
    }

    /// Format the stream termination event.
    pub fn format_done(&self) -> String {
        "event: done\ndata: [DONE]\n\n".to_string()
    }

    /// Format a heartbeat/keep-alive comment.
    pub fn format_heartbeat(&self) -> String {
        ": heartbeat\n\n".to_string()
    }

    /// Increment the event counter and return the new ID.
    pub fn next_id(&mut self) -> u64 {
        self.event_id += 1;
        self.event_id
    }
}

impl Default for StreamingResponse {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_ok() {
        let resp = Response::ok();
        assert_eq!(resp.status_code, 200);
        assert_eq!(resp.content_length(), 0);
    }

    #[test]
    fn test_response_json() {
        let resp = Response::ok().json(serde_json::json!({"key": "value"}));
        assert_eq!(resp.status_code, 200);
        assert_eq!(
            resp.headers.get("Content-Type").unwrap(),
            "application/json"
        );
        let body = resp.body_string().unwrap();
        assert!(body.contains("key"));
        assert!(body.contains("value"));
    }

    #[test]
    fn test_response_text() {
        let resp = Response::ok().text("Hello, world!");
        assert_eq!(resp.body_string().unwrap(), "Hello, world!");
        assert_eq!(resp.headers.get("Content-Type").unwrap(), "text/plain");
    }

    #[test]
    fn test_response_html() {
        let resp = Response::ok().html("<h1>Hello</h1>");
        assert_eq!(resp.body_string().unwrap(), "<h1>Hello</h1>");
        assert_eq!(resp.headers.get("Content-Type").unwrap(), "text/html");
    }

    #[test]
    fn test_response_not_found() {
        let resp = Response::not_found("User not found");
        assert_eq!(resp.status_code, 404);
        let body = resp.body_string().unwrap();
        assert!(body.contains("Not Found"));
        assert!(body.contains("User not found"));
    }

    #[test]
    fn test_response_unauthorized() {
        let resp = Response::unauthorized("Invalid token");
        assert_eq!(resp.status_code, 401);
    }

    #[test]
    fn test_response_with_header() {
        let resp = Response::ok().with_header("X-Custom", "value").text("data");
        assert_eq!(resp.headers.get("X-Custom").unwrap(), "value");
    }

    #[test]
    fn test_response_sse_stream() {
        let events = vec!["hello", "world"];
        let resp = Response::ok().sse_stream(events);
        assert!(resp.is_streaming());
        assert_eq!(
            resp.headers.get("Content-Type").unwrap(),
            "text/event-stream"
        );
        assert_eq!(resp.headers.get("Cache-Control").unwrap(), "no-cache");
    }

    #[test]
    fn test_response_sse_stream_body() {
        let events = vec!["hello".to_string(), "world".to_string()];
        let resp = Response::ok().sse_stream(events);
        let body = resp.body_string().unwrap();
        assert!(body.contains("data: hello\n\n"));
        assert!(body.contains("data: world\n\n"));
        assert_eq!(resp.content_length(), body.len());
    }

    #[test]
    fn test_response_sse_stream_empty() {
        let resp = Response::ok().sse_stream(Vec::<String>::new());
        assert!(resp.is_streaming());
        assert_eq!(resp.body_bytes(), Vec::<u8>::new());
        assert_eq!(resp.content_length(), 0);
    }

    #[test]
    fn test_response_display() {
        let resp = Response::ok().text("Hello");
        let display = format!("{}", resp);
        assert!(display.contains("200"));
    }

    #[test]
    fn test_into_response_string() {
        let resp = "Hello".into_response();
        assert_eq!(resp.status_code, 200);
        assert_eq!(resp.body_string().unwrap(), "Hello");
    }

    #[test]
    fn test_into_response_json_value() {
        let resp = serde_json::json!({"ok": true}).into_response();
        assert_eq!(resp.status_code, 200);
        assert!(resp.body_string().unwrap().contains("true"));
    }

    #[test]
    fn test_into_response_tuple() {
        let resp = (201u16, serde_json::json!({"id": 1})).into_response();
        assert_eq!(resp.status_code, 201);
    }

    #[test]
    fn test_streaming_response_format() {
        let stream = StreamingResponse::new();
        let event = stream.format_event("token", r#"{"content":"Hi"}"#);
        assert!(event.contains("id: 0"));
        assert!(event.contains("event: token"));
        assert!(event.contains(r#"data: {"content":"Hi"}"#));
    }

    #[test]
    fn test_streaming_response_done() {
        let stream = StreamingResponse::new();
        let done = stream.format_done();
        assert!(done.contains("event: done"));
        assert!(done.contains("[DONE]"));
    }

    #[test]
    fn test_streaming_response_heartbeat() {
        let stream = StreamingResponse::new();
        let hb = stream.format_heartbeat();
        assert!(hb.contains("heartbeat"));
    }

    #[test]
    fn test_streaming_response_next_id() {
        let mut stream = StreamingResponse::new();
        assert_eq!(stream.next_id(), 1);
        assert_eq!(stream.next_id(), 2);
        assert_eq!(stream.next_id(), 3);
    }

    #[test]
    fn test_response_body_bytes() {
        let resp = Response::ok().text("Hello");
        assert_eq!(resp.body_bytes(), b"Hello");

        let resp = Response::ok();
        assert!(resp.body_bytes().is_empty());
    }
}
