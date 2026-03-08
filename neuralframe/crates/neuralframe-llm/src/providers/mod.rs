//! # LLM Providers
//!
//! Implementations of the `LLMProvider` trait for various LLM services.

pub mod anthropic;
pub mod google;
pub mod groq;
pub mod ollama;
pub mod openai;

use crate::error::LLMError;
use crate::types::{CompletionRequest, CompletionResponse, Token};
use async_trait::async_trait;
use std::pin::Pin;
use tokio_stream::Stream;

/// Core trait for LLM providers.
///
/// Implementors provide completions, streaming, and embeddings
/// through a unified interface.
///
/// # Examples
///
/// ```rust,no_run
/// use neuralframe_llm::providers::LLMProvider;
/// use neuralframe_llm::types::CompletionRequest;
///
/// async fn example(provider: &dyn LLMProvider) {
///     let req = CompletionRequest::new("gpt-4o")
///         .user("Hello!");
///     let response = provider.complete(req).await.unwrap();
///     println!("{}", response.content);
/// }
/// ```
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Generate a completion for the given request.
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LLMError>;

    /// Stream tokens for the given request.
    async fn stream(
        &self,
        req: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, LLMError>> + Send>>, LLMError>;

    /// Generate embeddings for the given text.
    async fn embed(&self, text: &str, model: &str) -> Result<Vec<f32>, LLMError>;

    /// Return the provider name.
    fn name(&self) -> &str;

    /// List available models.
    fn models(&self) -> Vec<String>;
}
