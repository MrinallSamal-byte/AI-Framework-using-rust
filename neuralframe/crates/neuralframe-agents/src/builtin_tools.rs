use async_trait::async_trait;
use neuralframe_llm::{
    error::LLMError,
    tools::{Tool, ToolDefinition},
};

/// Echo tool that returns the provided input string unchanged.
pub struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "echo",
            "Returns the input string unchanged",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Text to echo"}
                },
                "required": ["input"]
            }),
        )
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<String, LLMError> {
        Ok(arguments["input"].as_str().unwrap_or("").to_string())
    }
}

/// Tool that returns the current UTC time in ISO-8601 format.
pub struct CurrentTimeTool;

#[async_trait]
impl Tool for CurrentTimeTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::no_params("current_time", "Returns the current UTC time as ISO-8601")
    }

    async fn execute(&self, _arguments: serde_json::Value) -> Result<String, LLMError> {
        Ok(chrono::Utc::now().to_rfc3339())
    }
}
