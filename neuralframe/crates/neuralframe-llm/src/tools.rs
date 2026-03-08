//! # Tool / Function Calling System
//!
//! Define tools that LLMs can call, with automatic JSON schema generation
//! and parallel tool execution.

use crate::error::LLMError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Definition of a tool that an LLM can call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// The unique name of the tool.
    pub name: String,
    /// A description of what the tool does.
    pub description: String,
    /// JSON Schema for the tool's parameters.
    pub parameters: serde_json::Value,
}

impl ToolDefinition {
    /// Create a new tool definition.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use neuralframe_llm::tools::ToolDefinition;
    ///
    /// let tool = ToolDefinition::new(
    ///     "get_weather",
    ///     "Get the current weather for a location",
    ///     serde_json::json!({
    ///         "type": "object",
    ///         "properties": {
    ///             "location": {
    ///                 "type": "string",
    ///                 "description": "City name"
    ///             }
    ///         },
    ///         "required": ["location"]
    ///     }),
    /// );
    /// ```
    pub fn new(name: &str, description: &str, parameters: serde_json::Value) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            parameters,
        }
    }

    /// Create a tool definition with no parameters.
    pub fn no_params(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }
    }
}

/// Result of executing a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// The tool call ID (for correlating with the request).
    pub call_id: String,
    /// The tool name.
    pub name: String,
    /// The result content (usually JSON string).
    pub content: String,
    /// Whether the tool execution succeeded.
    pub success: bool,
}

impl ToolResult {
    /// Create a successful tool result.
    pub fn success(call_id: &str, name: &str, content: &str) -> Self {
        Self {
            call_id: call_id.to_string(),
            name: name.to_string(),
            content: content.to_string(),
            success: true,
        }
    }

    /// Create a failed tool result.
    pub fn failure(call_id: &str, name: &str, error: &str) -> Self {
        Self {
            call_id: call_id.to_string(),
            name: name.to_string(),
            content: format!("Error: {}", error),
            success: false,
        }
    }
}

/// Trait for executable tools.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Get the tool definition.
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with the given arguments.
    async fn execute(&self, arguments: serde_json::Value) -> Result<String, LLMError>;

    /// Get the tool name.
    fn name(&self) -> &str {
        &self.definition().name
    }
}

/// Registry for managing available tools.
///
/// Stores tool implementations and dispatches calls to them.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new empty tool registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool.
    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        let name = tool.definition().name.clone();
        self.tools.insert(name, Arc::new(tool));
    }

    /// Get all tool definitions (for sending to the LLM).
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    /// Execute a tool by name.
    pub async fn execute(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, LLMError> {
        let tool = self.tools.get(name).ok_or(LLMError::ToolError {
            tool: name.to_string(),
            source: "tool not found in registry".to_string(),
        })?;

        tool.execute(arguments).await
    }

    /// Execute multiple tool calls in parallel.
    pub async fn execute_parallel(
        &self,
        calls: Vec<crate::types::ToolCall>,
    ) -> Vec<ToolResult> {
        let mut handles = Vec::new();

        for call in calls {
            let registry = self.tools.clone();
            let handle = tokio::spawn(async move {
                let arguments: serde_json::Value =
                    serde_json::from_str(&call.arguments).unwrap_or(serde_json::Value::Null);

                match registry.get(&call.name) {
                    Some(tool) => match tool.execute(arguments).await {
                        Ok(content) => {
                            ToolResult::success(&call.id, &call.name, &content)
                        }
                        Err(e) => ToolResult::failure(
                            &call.id,
                            &call.name,
                            &e.to_string(),
                        ),
                    },
                    None => ToolResult::failure(
                        &call.id,
                        &call.name,
                        "tool not found",
                    ),
                }
            });
            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            if let Ok(result) = handle.await {
                results.push(result);
            }
        }
        results
    }

    /// Get the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to generate JSON Schema from a type description.
pub fn json_schema_object(properties: Vec<(&str, &str, &str, bool)>) -> serde_json::Value {
    let mut props = serde_json::Map::new();
    let mut required = Vec::new();

    for (name, type_str, description, is_required) in properties {
        props.insert(
            name.to_string(),
            serde_json::json!({
                "type": type_str,
                "description": description
            }),
        );
        if is_required {
            required.push(serde_json::Value::String(name.to_string()));
        }
    }

    serde_json::json!({
        "type": "object",
        "properties": props,
        "required": required
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_definition() {
        let tool = ToolDefinition::new(
            "search",
            "Search the web",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }),
        );
        assert_eq!(tool.name, "search");
        assert_eq!(tool.description, "Search the web");
    }

    #[test]
    fn test_tool_definition_no_params() {
        let tool = ToolDefinition::no_params("get_time", "Get the current time");
        assert_eq!(tool.name, "get_time");
        assert!(tool.parameters.get("properties").is_some());
    }

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success("call_1", "search", "found 10 results");
        assert!(result.success);
        assert_eq!(result.content, "found 10 results");
    }

    #[test]
    fn test_tool_result_failure() {
        let result = ToolResult::failure("call_1", "search", "timeout");
        assert!(!result.success);
        assert!(result.content.contains("Error"));
    }

    #[test]
    fn test_json_schema_helper() {
        let schema = json_schema_object(vec![
            ("location", "string", "City name", true),
            ("units", "string", "Temperature units", false),
        ]);

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["location"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "location");
    }

    #[test]
    fn test_tool_registry() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }
}
