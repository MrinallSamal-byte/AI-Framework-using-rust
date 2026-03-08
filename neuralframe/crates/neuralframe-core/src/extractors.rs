//! # Typed Extractors
//!
//! Type-safe request data extraction from HTTP requests.
//! Extractors automatically deserialize path parameters, query strings,
//! JSON bodies, and headers.
//!
//! ## Examples
//!
//! ```rust
//! use neuralframe_core::extractors::{Json, Query, PathParams, Headers};
//! use serde::Deserialize;
//!
//! #[derive(Deserialize)]
//! struct CreateUser {
//!     name: String,
//!     email: String,
//! }
//!
//! // The handler would receive `Json<CreateUser>` automatically
//! ```

use crate::error::{NeuralError, NeuralResult};
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::fmt;

/// Extract a JSON body from the request.
///
/// Automatically deserializes the request body into type `T`.
///
/// # Examples
///
/// ```rust
/// use neuralframe_core::extractors::Json;
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct ChatMessage {
///     content: String,
/// }
///
/// let json_body = Json(ChatMessage { content: "Hello".to_string() });
/// assert_eq!(json_body.content, "Hello");
/// ```
#[derive(Debug, Clone)]
pub struct Json<T>(pub T);

impl<T> Json<T> {
    /// Unwrap the inner value.
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> std::ops::Deref for Json<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Json<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: DeserializeOwned> Json<T> {
    /// Parse JSON from a byte slice.
    pub fn from_bytes(bytes: &[u8]) -> NeuralResult<Self> {
        let value: T = serde_json::from_slice(bytes).map_err(|e| {
            NeuralError::DeserializationError {
                context: "JSON body".to_string(),
                source: e.to_string(),
            }
        })?;
        Ok(Json(value))
    }

    /// Parse JSON from a string.
    pub fn from_str(s: &str) -> NeuralResult<Self> {
        let value: T = serde_json::from_str(s).map_err(|e| {
            NeuralError::DeserializationError {
                context: "JSON body".to_string(),
                source: e.to_string(),
            }
        })?;
        Ok(Json(value))
    }
}

/// Extract query parameters from the URL.
///
/// Parses query string `?key=value&other=123` into a typed struct.
///
/// # Examples
///
/// ```rust
/// use neuralframe_core::extractors::Query;
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct Pagination {
///     page: u32,
///     limit: u32,
/// }
/// ```
#[derive(Debug, Clone)]
pub struct Query<T>(pub T);

impl<T> Query<T> {
    /// Unwrap the inner value.
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> std::ops::Deref for Query<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: DeserializeOwned> Query<T> {
    /// Parse query parameters from a query string.
    ///
    /// # Arguments
    ///
    /// * `query_string` - The raw query string (without the leading `?`)
    pub fn from_query_string(query_string: &str) -> NeuralResult<Self> {
        let pairs: Vec<(String, String)> = query_string
            .split('&')
            .filter(|s| !s.is_empty())
            .filter_map(|pair| {
                let mut parts = pair.splitn(2, '=');
                let key = parts.next()?;
                let value = parts.next().unwrap_or("");
                let key = percent_encoding::percent_decode_str(key)
                    .decode_utf8()
                    .ok()?
                    .to_string();
                let value = percent_encoding::percent_decode_str(value)
                    .decode_utf8()
                    .ok()?
                    .to_string();
                Some((key, value))
            })
            .collect();

        // Convert to a JSON object for serde deserialization
        let map: serde_json::Map<String, serde_json::Value> = pairs
            .into_iter()
            .map(|(k, v)| (k, serde_json::Value::String(v)))
            .collect();

        let value = serde_json::Value::Object(map);
        let result: T = serde_json::from_value(value).map_err(|e| {
            NeuralError::DeserializationError {
                context: "query parameters".to_string(),
                source: e.to_string(),
            }
        })?;

        Ok(Query(result))
    }
}

/// Extracted path parameters from the URL.
///
/// Contains named parameters extracted by the router (e.g., `:id` → `"42"`).
///
/// # Examples
///
/// ```rust
/// use neuralframe_core::extractors::PathParams;
/// use std::collections::HashMap;
///
/// let mut params = HashMap::new();
/// params.insert("id".to_string(), "42".to_string());
/// let path = PathParams::new(params);
///
/// assert_eq!(path.get("id"), Some(&"42".to_string()));
/// assert_eq!(path.get_as::<u32>("id"), Some(42));
/// ```
#[derive(Debug, Clone, Default)]
pub struct PathParams {
    params: HashMap<String, String>,
}

impl PathParams {
    /// Create a new `PathParams` from a map.
    pub fn new(params: HashMap<String, String>) -> Self {
        Self { params }
    }

    /// Get a raw parameter value by name.
    pub fn get(&self, name: &str) -> Option<&String> {
        self.params.get(name)
    }

    /// Get a parameter value parsed as type `T`.
    ///
    /// Returns `None` if the parameter doesn't exist or can't be parsed.
    pub fn get_as<T: std::str::FromStr>(&self, name: &str) -> Option<T> {
        self.params.get(name).and_then(|v| v.parse().ok())
    }

    /// Get a parameter or return an error.
    pub fn require(&self, name: &str) -> NeuralResult<&str> {
        self.params.get(name).map(|s| s.as_str()).ok_or_else(|| {
            NeuralError::DeserializationError {
                context: format!("path parameter '{}'", name),
                source: "parameter not found".to_string(),
            }
        })
    }

    /// Return the number of extracted parameters.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Check if there are no parameters.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
}

/// HTTP headers extracted from the request.
///
/// Provides convenient access to common headers with type conversion.
///
/// # Examples
///
/// ```rust
/// use neuralframe_core::extractors::Headers;
///
/// let mut headers = Headers::new();
/// headers.insert("content-type", "application/json");
/// headers.insert("authorization", "Bearer token123");
///
/// assert_eq!(headers.get("content-type"), Some(&"application/json".to_string()));
/// assert_eq!(headers.content_type(), Some("application/json"));
/// ```
#[derive(Debug, Clone, Default)]
pub struct Headers {
    inner: HashMap<String, String>,
}

impl Headers {
    /// Create a new empty `Headers`.
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Create `Headers` from a map.
    pub fn from_map(map: HashMap<String, String>) -> Self {
        let normalized: HashMap<String, String> = map
            .into_iter()
            .map(|(k, v)| (k.to_lowercase(), v))
            .collect();
        Self { inner: normalized }
    }

    /// Insert a header (key is lowercased automatically).
    pub fn insert(&mut self, key: &str, value: &str) {
        self.inner
            .insert(key.to_lowercase(), value.to_string());
    }

    /// Get a header value by name (case-insensitive).
    pub fn get(&self, key: &str) -> Option<&String> {
        self.inner.get(&key.to_lowercase())
    }

    /// Get the `Content-Type` header.
    pub fn content_type(&self) -> Option<&str> {
        self.inner.get("content-type").map(|s| s.as_str())
    }

    /// Get the `Authorization` header.
    pub fn authorization(&self) -> Option<&str> {
        self.inner.get("authorization").map(|s| s.as_str())
    }

    /// Get the `User-Agent` header.
    pub fn user_agent(&self) -> Option<&str> {
        self.inner.get("user-agent").map(|s| s.as_str())
    }

    /// Extract a Bearer token from the Authorization header.
    pub fn bearer_token(&self) -> Option<&str> {
        self.authorization()
            .and_then(|auth| auth.strip_prefix("Bearer "))
    }

    /// Extract an API key from a custom header.
    pub fn api_key(&self, header_name: &str) -> Option<&str> {
        self.inner.get(&header_name.to_lowercase()).map(|s| s.as_str())
    }

    /// Return the number of headers.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if there are no headers.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterate over all headers.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &String)> {
        self.inner.iter()
    }
}

impl fmt::Display for Headers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Headers({} entries)", self.inner.len())
    }
}

/// An incoming HTTP request with all extracted data.
///
/// This is the primary request type passed to handlers, providing
/// access to the body, path parameters, query string, headers, and method.
#[derive(Debug, Clone)]
pub struct Request {
    /// HTTP method.
    pub method: String,
    /// Request path.
    pub path: String,
    /// Raw query string.
    pub query_string: Option<String>,
    /// Request headers.
    pub headers: Headers,
    /// Path parameters extracted by the router.
    pub params: PathParams,
    /// Raw request body bytes.
    pub body: Vec<u8>,
}

impl Request {
    /// Create a new request (primarily for testing).
    pub fn new(method: &str, path: &str) -> Self {
        Self {
            method: method.to_uppercase(),
            path: path.to_string(),
            query_string: None,
            headers: Headers::new(),
            params: PathParams::default(),
            body: Vec::new(),
        }
    }

    /// Set the request body as JSON.
    pub fn with_json_body<T: serde::Serialize>(mut self, body: &T) -> NeuralResult<Self> {
        self.body = serde_json::to_vec(body).map_err(|e| NeuralError::SerializationError {
            context: "request body".to_string(),
            source: e.to_string(),
        })?;
        self.headers.insert("content-type", "application/json");
        Ok(self)
    }

    /// Set the raw request body.
    pub fn with_body(mut self, body: Vec<u8>) -> Self {
        self.body = body;
        self
    }

    /// Set path parameters.
    pub fn with_params(mut self, params: HashMap<String, String>) -> Self {
        self.params = PathParams::new(params);
        self
    }

    /// Set a query string.
    pub fn with_query(mut self, query: &str) -> Self {
        self.query_string = Some(query.to_string());
        self
    }

    /// Parse the body as JSON into type `T`.
    pub fn json<T: DeserializeOwned>(&self) -> NeuralResult<Json<T>> {
        Json::from_bytes(&self.body)
    }

    /// Parse query parameters into type `T`.
    pub fn query<T: DeserializeOwned>(&self) -> NeuralResult<Query<T>> {
        let qs = self.query_string.as_deref().unwrap_or("");
        Query::from_query_string(qs)
    }

    /// Get the body as a UTF-8 string.
    pub fn body_string(&self) -> NeuralResult<String> {
        String::from_utf8(self.body.clone()).map_err(|e| NeuralError::DeserializationError {
            context: "body string".to_string(),
            source: e.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestBody {
        name: String,
        age: u32,
    }

    #[test]
    fn test_json_from_bytes() {
        let json = Json::<TestBody>::from_bytes(br#"{"name":"Alice","age":30}"#).unwrap();
        assert_eq!(json.name, "Alice");
        assert_eq!(json.age, 30);
    }

    #[test]
    fn test_json_from_str() {
        let json = Json::<TestBody>::from_str(r#"{"name":"Bob","age":25}"#).unwrap();
        assert_eq!(json.name, "Bob");
        assert_eq!(json.age, 25);
    }

    #[test]
    fn test_json_invalid() {
        let result = Json::<TestBody>::from_bytes(b"not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_json_into_inner() {
        let json = Json(TestBody {
            name: "Test".into(),
            age: 1,
        });
        let inner = json.into_inner();
        assert_eq!(inner.name, "Test");
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestQuery {
        page: String,
        limit: String,
    }

    #[test]
    fn test_query_from_string() {
        let query = Query::<TestQuery>::from_query_string("page=1&limit=10").unwrap();
        assert_eq!(query.page, "1");
        assert_eq!(query.limit, "10");
    }

    #[test]
    fn test_query_empty() {
        #[derive(Debug, Deserialize)]
        struct Empty {}
        let query = Query::<Empty>::from_query_string("");
        assert!(query.is_ok());
    }

    #[test]
    fn test_query_url_encoded() {
        #[derive(Debug, Deserialize)]
        struct Q {
            q: String,
        }
        let query = Query::<Q>::from_query_string("q=hello%20world").unwrap();
        assert_eq!(query.q, "hello world");
    }

    #[test]
    fn test_path_params() {
        let mut map = HashMap::new();
        map.insert("id".to_string(), "42".to_string());
        map.insert("name".to_string(), "alice".to_string());
        let params = PathParams::new(map);

        assert_eq!(params.get("id"), Some(&"42".to_string()));
        assert_eq!(params.get_as::<u32>("id"), Some(42));
        assert_eq!(params.get("name"), Some(&"alice".to_string()));
        assert!(params.get("missing").is_none());
        assert_eq!(params.len(), 2);
        assert!(!params.is_empty());
    }

    #[test]
    fn test_path_params_require() {
        let params = PathParams::default();
        assert!(params.require("id").is_err());

        let mut map = HashMap::new();
        map.insert("id".to_string(), "42".to_string());
        let params = PathParams::new(map);
        assert_eq!(params.require("id").unwrap(), "42");
    }

    #[test]
    fn test_headers() {
        let mut headers = Headers::new();
        headers.insert("Content-Type", "application/json");
        headers.insert("Authorization", "Bearer mytoken");
        headers.insert("X-API-Key", "key123");

        assert_eq!(headers.content_type(), Some("application/json"));
        assert_eq!(headers.authorization(), Some("Bearer mytoken"));
        assert_eq!(headers.bearer_token(), Some("mytoken"));
        assert_eq!(headers.api_key("x-api-key"), Some("key123"));
        assert_eq!(headers.len(), 3);
    }

    #[test]
    fn test_headers_case_insensitive() {
        let mut headers = Headers::new();
        headers.insert("CONTENT-TYPE", "text/html");

        assert_eq!(headers.get("content-type"), Some(&"text/html".to_string()));
        assert_eq!(headers.get("Content-Type"), Some(&"text/html".to_string()));
    }

    #[test]
    fn test_request_builder() {
        let req = Request::new("GET", "/users")
            .with_query("page=1&limit=10")
            .with_body(b"hello".to_vec());

        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/users");
        assert_eq!(req.query_string, Some("page=1&limit=10".to_string()));
        assert_eq!(req.body, b"hello");
    }

    #[test]
    fn test_request_json_body() {
        #[derive(serde::Serialize, Deserialize, Debug, PartialEq)]
        struct Msg {
            text: String,
        }
        let req = Request::new("POST", "/chat")
            .with_json_body(&Msg {
                text: "hello".into(),
            })
            .unwrap();

        let body: Json<Msg> = req.json().unwrap();
        assert_eq!(body.text, "hello");
    }

    #[test]
    fn test_request_body_string() {
        let req = Request::new("POST", "/echo").with_body(b"hello world".to_vec());
        assert_eq!(req.body_string().unwrap(), "hello world");
    }
}
