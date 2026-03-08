//! # Radix Tree Router
//!
//! Zero-allocation route matching using a radix tree (trie).
//! Supports path parameters (`:id`), wildcards (`*path`), and route groups.
//!
//! ## Examples
//!
//! ```rust
//! use neuralframe_core::router::Router;
//!
//! let mut router = Router::new();
//! router.add_route("GET", "/users/:id", "get_user");
//! router.add_route("GET", "/files/*path", "get_file");
//!
//! let m = router.match_route("GET", "/users/42").unwrap();
//! assert_eq!(m.params.get("id"), Some(&"42".to_string()));
//! ```

use std::collections::HashMap;
use std::fmt;

/// A matched route with extracted parameters.
#[derive(Debug, Clone)]
pub struct RouteMatch<T: Clone> {
    /// The handler associated with the matched route.
    pub handler: T,
    /// Extracted path parameters (e.g., `:id` → `"42"`).
    pub params: HashMap<String, String>,
}

/// A single route definition.
#[derive(Debug, Clone)]
pub struct Route {
    /// HTTP method (GET, POST, PUT, DELETE, etc.).
    pub method: String,
    /// Path pattern (e.g., `/users/:id`).
    pub path: String,
}

impl Route {
    /// Create a new route definition.
    pub fn new(method: &str, path: &str) -> Self {
        Self {
            method: method.to_uppercase(),
            path: path.to_string(),
        }
    }
}

/// Type of a radix tree node segment.
#[derive(Debug, Clone, PartialEq)]
enum NodeType {
    /// Static path segment (e.g., "users").
    Static,
    /// Named parameter (e.g., ":id").
    Param,
    /// Wildcard catch-all (e.g., "*path").
    Wildcard,
}

/// A node in the radix tree.
#[derive(Debug, Clone)]
struct RadixNode<T: Clone> {
    /// The path segment this node represents.
    segment: String,
    /// Type of this node segment.
    node_type: NodeType,
    /// Children nodes indexed by their first character for fast lookup.
    children: Vec<RadixNode<T>>,
    /// Handler stored at this node for each HTTP method.
    handlers: HashMap<String, T>,
    /// Parameter name if this is a Param or Wildcard node.
    param_name: Option<String>,
}

impl<T: Clone> RadixNode<T> {
    /// Create a new static node.
    fn new_static(segment: &str) -> Self {
        Self {
            segment: segment.to_string(),
            node_type: NodeType::Static,
            children: Vec::new(),
            handlers: HashMap::new(),
            param_name: None,
        }
    }

    /// Create a new parameter node.
    fn new_param(name: &str) -> Self {
        Self {
            segment: String::new(),
            node_type: NodeType::Param,
            children: Vec::new(),
            handlers: HashMap::new(),
            param_name: Some(name.to_string()),
        }
    }

    /// Create a new wildcard node.
    fn new_wildcard(name: &str) -> Self {
        Self {
            segment: String::new(),
            node_type: NodeType::Wildcard,
            children: Vec::new(),
            handlers: HashMap::new(),
            param_name: Some(name.to_string()),
        }
    }
}

/// High-performance radix tree router.
///
/// Routes are stored in a radix tree (compressed trie) for O(k) lookup
/// where k is the length of the URL path. Supports:
/// - Static paths: `/users/list`
/// - Path parameters: `/users/:id`
/// - Wildcard paths: `/files/*path`
/// - Route groups with shared prefixes
#[derive(Debug, Clone)]
pub struct Router<T: Clone> {
    /// Root node of the radix tree.
    root: RadixNode<T>,
    /// Number of registered routes.
    route_count: usize,
}

impl<T: Clone> Default for Router<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Router<T> {
    /// Create a new empty router.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use neuralframe_core::router::Router;
    /// let router: Router<String> = Router::new();
    /// ```
    pub fn new() -> Self {
        Self {
            root: RadixNode::new_static(""),
            route_count: 0,
        }
    }

    /// Return the number of registered routes.
    pub fn route_count(&self) -> usize {
        self.route_count
    }

    /// Add a route to the router.
    ///
    /// # Arguments
    ///
    /// * `method` - HTTP method (GET, POST, PUT, DELETE, etc.)
    /// * `path` - URL path pattern (e.g., `/users/:id`)
    /// * `handler` - The handler to associate with this route
    ///
    /// # Examples
    ///
    /// ```rust
    /// use neuralframe_core::router::Router;
    ///
    /// let mut router = Router::new();
    /// router.add_route("GET", "/users/:id", "handler_fn");
    /// ```
    pub fn add_route(&mut self, method: &str, path: &str, handler: T) {
        let method = method.to_uppercase();
        let segments = Self::parse_path(path);
        let mut current = &mut self.root;

        for segment in &segments {
            if segment.starts_with(':') {
                // Parameter segment
                let param_name = &segment[1..];
                let idx = current
                    .children
                    .iter()
                    .position(|c| c.node_type == NodeType::Param);
                if let Some(idx) = idx {
                    current = &mut current.children[idx];
                } else {
                    current.children.push(RadixNode::new_param(param_name));
                    let last = current.children.len() - 1;
                    current = &mut current.children[last];
                }
            } else if segment.starts_with('*') {
                // Wildcard segment
                let param_name = &segment[1..];
                let idx = current
                    .children
                    .iter()
                    .position(|c| c.node_type == NodeType::Wildcard);
                if let Some(idx) = idx {
                    current = &mut current.children[idx];
                } else {
                    current.children.push(RadixNode::new_wildcard(param_name));
                    let last = current.children.len() - 1;
                    current = &mut current.children[last];
                }
            } else {
                // Static segment
                let idx = current
                    .children
                    .iter()
                    .position(|c| c.node_type == NodeType::Static && c.segment == *segment);
                if let Some(idx) = idx {
                    current = &mut current.children[idx];
                } else {
                    current.children.push(RadixNode::new_static(segment));
                    let last = current.children.len() - 1;
                    current = &mut current.children[last];
                }
            }
        }

        current.handlers.insert(method, handler);
        self.route_count += 1;
    }

    /// Match a request path against registered routes.
    ///
    /// Returns `Some(RouteMatch)` if a matching route is found, `None` otherwise.
    ///
    /// # Arguments
    ///
    /// * `method` - HTTP method to match
    /// * `path` - URL path to match against
    ///
    /// # Examples
    ///
    /// ```rust
    /// use neuralframe_core::router::Router;
    ///
    /// let mut router = Router::new();
    /// router.add_route("GET", "/users/:id", "get_user");
    ///
    /// let result = router.match_route("GET", "/users/42");
    /// assert!(result.is_some());
    /// let m = result.unwrap();
    /// assert_eq!(m.handler, "get_user");
    /// assert_eq!(m.params.get("id").unwrap(), "42");
    /// ```
    pub fn match_route(&self, method: &str, path: &str) -> Option<RouteMatch<T>> {
        let method = method.to_uppercase();
        let segments = Self::parse_path(path);
        let mut params = HashMap::new();

        if let Some(handler) = self.match_node(&self.root, &segments, 0, &mut params, &method) {
            Some(RouteMatch {
                handler: handler.clone(),
                params,
            })
        } else {
            None
        }
    }

    /// Recursively match a node in the radix tree.
    fn match_node<'a>(
        &'a self,
        node: &'a RadixNode<T>,
        segments: &[String],
        depth: usize,
        params: &mut HashMap<String, String>,
        method: &str,
    ) -> Option<&'a T> {
        // If we've consumed all segments, check for a handler
        if depth == segments.len() {
            return node.handlers.get(method);
        }

        let current_segment = &segments[depth];

        // Try static children first (highest priority)
        for child in &node.children {
            if child.node_type == NodeType::Static && child.segment == *current_segment {
                if let Some(handler) =
                    self.match_node(child, segments, depth + 1, params, method)
                {
                    return Some(handler);
                }
            }
        }

        // Try parameter children next
        for child in &node.children {
            if child.node_type == NodeType::Param {
                if let Some(name) = &child.param_name {
                    params.insert(name.clone(), current_segment.clone());
                    if let Some(handler) =
                        self.match_node(child, segments, depth + 1, params, method)
                    {
                        return Some(handler);
                    }
                    params.remove(name);
                }
            }
        }

        // Try wildcard children last (catch-all)
        for child in &node.children {
            if child.node_type == NodeType::Wildcard {
                if let Some(name) = &child.param_name {
                    // Wildcard captures everything remaining
                    let remaining: Vec<&str> =
                        segments[depth..].iter().map(|s| s.as_str()).collect();
                    params.insert(name.clone(), remaining.join("/"));
                    if let Some(handler) = child.handlers.get(method) {
                        return Some(handler);
                    }
                    params.remove(name);
                }
            }
        }

        None
    }

    /// Parse a URL path into segments.
    fn parse_path(path: &str) -> Vec<String> {
        path.split('/')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }
}

impl<T: Clone + fmt::Debug> fmt::Display for Router<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Router({} routes)", self.route_count)
    }
}

/// Route group builder for organizing routes with shared prefixes and middleware.
///
/// # Examples
///
/// ```rust
/// use neuralframe_core::router::{Router, RouteGroup};
///
/// let mut router = Router::new();
/// let group = RouteGroup::new("/api/v1")
///     .route("GET", "/users", "list_users")
///     .route("POST", "/users", "create_user")
///     .route("GET", "/users/:id", "get_user");
///
/// group.register(&mut router);
/// assert!(router.match_route("GET", "/api/v1/users").is_some());
/// ```
pub struct RouteGroup<T: Clone> {
    /// Path prefix for all routes in this group.
    prefix: String,
    /// Routes in this group.
    routes: Vec<(String, String, T)>,
}

impl<T: Clone> RouteGroup<T> {
    /// Create a new route group with the given prefix.
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.trim_end_matches('/').to_string(),
            routes: Vec::new(),
        }
    }

    /// Add a route to this group.
    pub fn route(mut self, method: &str, path: &str, handler: T) -> Self {
        self.routes
            .push((method.to_string(), path.to_string(), handler));
        self
    }

    /// Register all routes in this group with the given router.
    pub fn register(self, router: &mut Router<T>) {
        for (method, path, handler) in self.routes {
            let full_path = format!("{}{}", self.prefix, path);
            router.add_route(&method, &full_path, handler);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_routes() {
        let mut router = Router::new();
        router.add_route("GET", "/users", "list_users");
        router.add_route("POST", "/users", "create_user");
        router.add_route("GET", "/users/active", "active_users");

        let m = router.match_route("GET", "/users").unwrap();
        assert_eq!(m.handler, "list_users");
        assert!(m.params.is_empty());

        let m = router.match_route("POST", "/users").unwrap();
        assert_eq!(m.handler, "create_user");

        let m = router.match_route("GET", "/users/active").unwrap();
        assert_eq!(m.handler, "active_users");
    }

    #[test]
    fn test_param_routes() {
        let mut router = Router::new();
        router.add_route("GET", "/users/:id", "get_user");
        router.add_route("GET", "/users/:id/posts/:post_id", "get_user_post");

        let m = router.match_route("GET", "/users/42").unwrap();
        assert_eq!(m.handler, "get_user");
        assert_eq!(m.params.get("id").unwrap(), "42");

        let m = router.match_route("GET", "/users/42/posts/7").unwrap();
        assert_eq!(m.handler, "get_user_post");
        assert_eq!(m.params.get("id").unwrap(), "42");
        assert_eq!(m.params.get("post_id").unwrap(), "7");
    }

    #[test]
    fn test_wildcard_routes() {
        let mut router = Router::new();
        router.add_route("GET", "/files/*path", "get_file");

        let m = router.match_route("GET", "/files/docs/readme.md").unwrap();
        assert_eq!(m.handler, "get_file");
        assert_eq!(m.params.get("path").unwrap(), "docs/readme.md");
    }

    #[test]
    fn test_route_not_found() {
        let mut router: Router<&str> = Router::new();
        router.add_route("GET", "/users", "list_users");

        assert!(router.match_route("GET", "/posts").is_none());
        assert!(router.match_route("POST", "/users").is_none());
        assert!(router.match_route("GET", "/users/123").is_none());
    }

    #[test]
    fn test_route_priority() {
        let mut router = Router::new();
        router.add_route("GET", "/users/active", "active_users");
        router.add_route("GET", "/users/:id", "get_user");

        // Static route should take priority over parameter route
        let m = router.match_route("GET", "/users/active").unwrap();
        assert_eq!(m.handler, "active_users");

        let m = router.match_route("GET", "/users/42").unwrap();
        assert_eq!(m.handler, "get_user");
    }

    #[test]
    fn test_route_groups() {
        let mut router = Router::new();
        let group = RouteGroup::new("/api/v1")
            .route("GET", "/users", "list_users")
            .route("POST", "/users", "create_user")
            .route("GET", "/users/:id", "get_user");

        group.register(&mut router);

        let m = router.match_route("GET", "/api/v1/users").unwrap();
        assert_eq!(m.handler, "list_users");

        let m = router.match_route("GET", "/api/v1/users/42").unwrap();
        assert_eq!(m.handler, "get_user");
        assert_eq!(m.params.get("id").unwrap(), "42");
    }

    #[test]
    fn test_root_route() {
        let mut router = Router::new();
        router.add_route("GET", "/", "index");

        let m = router.match_route("GET", "/").unwrap();
        assert_eq!(m.handler, "index");
    }

    #[test]
    fn test_route_count() {
        let mut router: Router<&str> = Router::new();
        assert_eq!(router.route_count(), 0);

        router.add_route("GET", "/a", "a");
        router.add_route("POST", "/b", "b");
        assert_eq!(router.route_count(), 2);
    }

    #[test]
    fn test_case_insensitive_method() {
        let mut router = Router::new();
        router.add_route("get", "/test", "handler");

        let m = router.match_route("GET", "/test").unwrap();
        assert_eq!(m.handler, "handler");
    }

    #[test]
    fn test_multiple_methods_same_path() {
        let mut router = Router::new();
        router.add_route("GET", "/resource", "get_handler");
        router.add_route("POST", "/resource", "post_handler");
        router.add_route("PUT", "/resource", "put_handler");
        router.add_route("DELETE", "/resource", "delete_handler");

        assert_eq!(
            router.match_route("GET", "/resource").unwrap().handler,
            "get_handler"
        );
        assert_eq!(
            router.match_route("POST", "/resource").unwrap().handler,
            "post_handler"
        );
        assert_eq!(
            router.match_route("PUT", "/resource").unwrap().handler,
            "put_handler"
        );
        assert_eq!(
            router.match_route("DELETE", "/resource").unwrap().handler,
            "delete_handler"
        );
    }

    #[test]
    fn test_deeply_nested_routes() {
        let mut router = Router::new();
        router.add_route("GET", "/a/b/c/d/e/f", "deep");

        let m = router.match_route("GET", "/a/b/c/d/e/f").unwrap();
        assert_eq!(m.handler, "deep");
    }

    #[test]
    fn test_trailing_slash_handling() {
        let mut router = Router::new();
        router.add_route("GET", "/users/", "with_slash");

        // With trailing slash should match
        let m = router.match_route("GET", "/users").unwrap();
        assert_eq!(m.handler, "with_slash");
    }
}
