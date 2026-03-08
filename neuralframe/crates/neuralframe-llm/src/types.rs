//! Core types shared across LLM providers.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Role in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System instructions.
    System,
    /// User message.
    User,
    /// Assistant response.
    Assistant,
    /// Tool/function result.
    Tool,
}

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// The role of the message sender.
    pub role: Role,
    /// The text content of the message.
    pub content: String,
    /// Function/tool call name (if role is Tool).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool call ID (for correlating tool results).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Create a system message.
    pub fn system(content: &str) -> Self {
        Self {
            role: Role::System,
            content: content.to_string(),
            name: None,
            tool_call_id: None,
        }
    }

    /// Create a user message.
    pub fn user(content: &str) -> Self {
        Self {
            role: Role::User,
            content: content.to_string(),
            name: None,
            tool_call_id: None,
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: content.to_string(),
            name: None,
            tool_call_id: None,
        }
    }

    /// Create a tool result message.
    pub fn tool_result(call_id: &str, name: &str, content: &str) -> Self {
        Self {
            role: Role::Tool,
            content: content.to_string(),
            name: Some(name.to_string()),
            tool_call_id: Some(call_id.to_string()),
        }
    }
}

/// A request to complete a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// The model to use (e.g., "gpt-4o", "claude-3.5-sonnet").
    pub model: String,
    /// The conversation messages.
    pub messages: Vec<Message>,
    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Sampling temperature (0.0 - 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-p sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Whether to stream the response.
    #[serde(default)]
    pub stream: bool,
    /// Tool definitions for function calling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<crate::tools::ToolDefinition>>,
    /// Additional provider-specific parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<HashMap<String, serde_json::Value>>,
}

impl CompletionRequest {
    /// Create a new completion request with the given model.
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            messages: Vec::new(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            stream: false,
            tools: None,
            extra: None,
        }
    }

    /// Add a system message.
    pub fn system(mut self, content: &str) -> Self {
        self.messages.push(Message::system(content));
        self
    }

    /// Add a user message.
    pub fn user(mut self, content: &str) -> Self {
        self.messages.push(Message::user(content));
        self
    }

    /// Add an assistant message.
    pub fn assistant(mut self, content: &str) -> Self {
        self.messages.push(Message::assistant(content));
        self
    }

    /// Set max tokens.
    pub fn max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Set temperature.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set top-p.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set stop sequences.
    pub fn stop(mut self, stops: Vec<String>) -> Self {
        self.stop = Some(stops);
        self
    }

    /// Enable streaming.
    pub fn streaming(mut self) -> Self {
        self.stream = true;
        self
    }

    /// Add tool definitions.
    pub fn with_tools(mut self, tools: Vec<crate::tools::ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }
}

/// A completion response from an LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// The generated content.
    pub content: String,
    /// The model used.
    pub model: String,
    /// Token usage statistics.
    pub usage: Usage,
    /// The finish reason.
    pub finish_reason: Option<FinishReason>,
    /// Tool calls requested by the model.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
}

/// Token usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
    /// Total tokens used.
    pub total_tokens: u32,
}

/// Why the model stopped generating.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Normal completion.
    Stop,
    /// Reached max token limit.
    Length,
    /// Model wants to call tools.
    ToolCalls,
    /// Content was filtered.
    ContentFilter,
}

/// A tool call requested by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this call.
    pub id: String,
    /// Name of the tool to call.
    pub name: String,
    /// Arguments as a JSON string.
    pub arguments: String,
}

/// A single streaming token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    /// The text content of this token.
    pub content: String,
    /// Whether this is the final token.
    pub done: bool,
    /// Finish reason (only on final token).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

impl Token {
    /// Create a new token.
    pub fn text(content: &str) -> Self {
        Self {
            content: content.to_string(),
            done: false,
            finish_reason: None,
        }
    }

    /// Create a final/done token.
    pub fn done() -> Self {
        Self {
            content: String::new(),
            done: true,
            finish_reason: Some(FinishReason::Stop),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::system("Be helpful");
        assert_eq!(msg.role, Role::System);
        assert_eq!(msg.content, "Be helpful");

        let msg = Message::user("Hello");
        assert_eq!(msg.role, Role::User);

        let msg = Message::assistant("Hi there");
        assert_eq!(msg.role, Role::Assistant);

        let msg = Message::tool_result("call_1", "search", "results...");
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.tool_call_id.unwrap(), "call_1");
    }

    #[test]
    fn test_completion_request_builder() {
        let req = CompletionRequest::new("gpt-4o")
            .system("You are helpful")
            .user("Hello")
            .max_tokens(1000)
            .temperature(0.7)
            .streaming();

        assert_eq!(req.model, "gpt-4o");
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.max_tokens, Some(1000));
        assert_eq!(req.temperature, Some(0.7));
        assert!(req.stream);
    }

    #[test]
    fn test_token() {
        let t = Token::text("Hello");
        assert_eq!(t.content, "Hello");
        assert!(!t.done);

        let t = Token::done();
        assert!(t.done);
        assert!(t.content.is_empty());
    }

    #[test]
    fn test_serialization() {
        let msg = Message::user("test");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("user"));
        assert!(json.contains("test"));
    }
}
