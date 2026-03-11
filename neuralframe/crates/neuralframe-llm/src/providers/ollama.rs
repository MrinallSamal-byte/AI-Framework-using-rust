//! Ollama provider (local models).

use crate::error::LLMError;
use crate::providers::LLMProvider;
use crate::types::{CompletionRequest, CompletionResponse, FinishReason, Token, Usage};
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use tokio_stream::Stream;

/// Ollama LLM provider for local model inference.
#[derive(Debug, Clone)]
pub struct OllamaProvider {
    base_url: String,
    client: reqwest::Client,
}

impl OllamaProvider {
    /// Create a new Ollama provider. Defaults to `http://localhost:11434`.
    pub fn new() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Set a custom base URL.
    pub fn with_url(mut self, url: &str) -> Self {
        self.base_url = url.trim_end_matches('/').to_string();
        self
    }
}

impl Default for OllamaProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
}

#[derive(Serialize, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
}

#[derive(Deserialize)]
struct OllamaResponse {
    message: Option<OllamaMessage>,
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

#[async_trait]
impl LLMProvider for OllamaProvider {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LLMError> {
        let messages: Vec<OllamaMessage> = req
            .messages
            .iter()
            .map(|m| OllamaMessage {
                role: match m.role {
                    crate::types::Role::System => "system".to_string(),
                    crate::types::Role::User => "user".to_string(),
                    crate::types::Role::Assistant => "assistant".to_string(),
                    crate::types::Role::Tool => "user".to_string(),
                },
                content: m.content.clone(),
            })
            .collect();

        let ollama_req = OllamaRequest {
            model: req.model.clone(),
            messages,
            stream: false,
            options: Some(OllamaOptions {
                temperature: req.temperature,
                top_p: req.top_p,
                num_predict: req.max_tokens,
            }),
        };

        let response = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&ollama_req)
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed {
                status: e.status().map(|s| s.as_u16()).unwrap_or(0),
                message: e.to_string(),
            })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(LLMError::RequestFailed {
                status,
                message: body,
            });
        }

        let resp: OllamaResponse = response.json().await.map_err(|e| LLMError::ParseError {
            context: "Ollama response".into(),
            source: e.to_string(),
        })?;

        let content = resp.message.map(|m| m.content).unwrap_or_default();

        let prompt_tokens = resp.prompt_eval_count.unwrap_or(0);
        let completion_tokens = resp.eval_count.unwrap_or(0);

        Ok(CompletionResponse {
            content,
            model: req.model,
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
            finish_reason: Some(FinishReason::Stop),
            tool_calls: Vec::new(),
        })
    }

    async fn stream(
        &self,
        req: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, LLMError>> + Send>>, LLMError> {
        let messages: Vec<OllamaMessage> = req
            .messages
            .iter()
            .map(|m| OllamaMessage {
                role: match m.role {
                    crate::types::Role::System => "system".to_string(),
                    crate::types::Role::User => "user".to_string(),
                    crate::types::Role::Assistant => "assistant".to_string(),
                    crate::types::Role::Tool => "user".to_string(),
                },
                content: m.content.clone(),
            })
            .collect();

        let ollama_req = OllamaRequest {
            model: req.model.clone(),
            messages,
            stream: true,
            options: Some(OllamaOptions {
                temperature: req.temperature,
                top_p: req.top_p,
                num_predict: req.max_tokens,
            }),
        };

        let response = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&ollama_req)
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed {
                status: e.status().map(|s| s.as_u16()).unwrap_or(0),
                message: e.to_string(),
            })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(LLMError::RequestFailed {
                status,
                message: body,
            });
        }

        let byte_stream = response.bytes_stream();
        let token_stream = byte_stream.filter_map(|result| async move {
            match result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    if let Ok(resp) = serde_json::from_str::<OllamaResponse>(&text) {
                        if resp.done {
                            return Some(Ok(Token::done()));
                        }
                        if let Some(msg) = resp.message {
                            if !msg.content.is_empty() {
                                return Some(Ok(Token::text(&msg.content)));
                            }
                        }
                    }
                    None
                }
                Err(e) => Some(Err(LLMError::StreamError(e.to_string()))),
            }
        });

        Ok(Box::pin(token_stream))
    }

    async fn embed(&self, text: &str, model: &str) -> Result<Vec<f32>, LLMError> {
        let body = serde_json::json!({
            "model": model,
            "prompt": text,
        });

        let response = self
            .client
            .post(format!("{}/api/embeddings", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed {
                status: e.status().map(|s| s.as_u16()).unwrap_or(0),
                message: e.to_string(),
            })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body_text = response.text().await.unwrap_or_default();
            return Err(LLMError::RequestFailed {
                status,
                message: body_text,
            });
        }

        #[derive(Deserialize)]
        struct EmbedResp {
            embedding: Vec<f32>,
        }

        let resp: EmbedResp = response.json().await.map_err(|e| LLMError::ParseError {
            context: "Ollama embedding".into(),
            source: e.to_string(),
        })?;

        Ok(resp.embedding)
    }

    fn name(&self) -> &str {
        "ollama"
    }

    fn models(&self) -> Vec<String> {
        vec![
            "llama3".into(),
            "llama3:70b".into(),
            "mistral".into(),
            "codellama".into(),
            "nomic-embed-text".into(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = OllamaProvider::new().with_url("http://localhost:11434");
        assert_eq!(provider.name(), "ollama");
    }
}
