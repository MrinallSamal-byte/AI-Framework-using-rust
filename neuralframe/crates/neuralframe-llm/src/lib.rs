//! # NeuralFrame LLM Client
//!
//! Universal LLM client abstraction with streaming, retry logic,
//! multi-provider support, and function/tool calling.
//!
//! ## Supported Providers
//!
//! - OpenAI (GPT-4o, GPT-4-turbo)
//! - Anthropic (Claude 3.5 Sonnet)
//! - Google (Gemini Pro)
//! - Ollama (local models)
//! - Groq (fast inference)
//!
//! ## Examples
//!
//! ```rust,no_run
//! use neuralframe_llm::prelude::*;
//!
//! #[tokio::main]
//! async fn main() {
//!     let client = OpenAIProvider::new("your-api-key");
//!     let req = CompletionRequest::new("gpt-4o")
//!         .system("You are a helpful assistant.")
//!         .user("Hello!");
//!     let response = client.complete(req).await.unwrap();
//!     println!("{}", response.content);
//! }
//! ```

pub mod error;
pub mod providers;
pub mod retry;
pub mod streaming;
pub mod tools;
pub mod types;

/// Prelude module re-exporting commonly used types.
pub mod prelude {
    pub use crate::error::*;
    pub use crate::providers::anthropic::AnthropicProvider;
    pub use crate::providers::google::GoogleProvider;
    pub use crate::providers::groq::GroqProvider;
    pub use crate::providers::ollama::OllamaProvider;
    pub use crate::providers::openai::OpenAIProvider;
    pub use crate::providers::LLMProvider;
    pub use crate::retry::RetryConfig;
    pub use crate::streaming::SseParser;
    pub use crate::tools::{Tool, ToolDefinition, ToolResult};
    pub use crate::types::*;
}
