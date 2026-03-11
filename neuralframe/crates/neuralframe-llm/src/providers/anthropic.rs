//! Anthropic provider (Claude 3.5 Sonnet, etc.).

use crate::error::LLMError;
use crate::providers::LLMProvider;
use crate::types::{CompletionRequest, CompletionResponse, FinishReason, Token, Usage};
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use tokio_stream::Stream;

/// Anthropic LLM provider for Claude models.
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
    api_version: String,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider.
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            client: reqwest::Client::new(),
            api_version: "2023-06-01".to_string(),
        }
    }

    /// Set a custom base URL.
    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = url.trim_end_matches('/').to_string();
        self
    }

    fn build_headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-api-key", self.api_key.parse().expect("valid header"));
        headers.insert(
            "anthropic-version",
            self.api_version.parse().expect("valid header"),
        );
        headers.insert(
            "Content-Type",
            "application/json".parse().expect("valid header"),
        );
        headers
    }
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(default)]
    stream: bool,
}

#[derive(Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    model: String,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    delta: Option<AnthropicDelta>,
}

#[derive(Deserialize)]
struct AnthropicDelta {
    text: Option<String>,
}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LLMError> {
        let system = req
            .messages
            .iter()
            .find(|m| m.role == crate::types::Role::System)
            .map(|m| m.content.clone());

        let messages: Vec<AnthropicMessage> = req
            .messages
            .iter()
            .filter(|m| m.role != crate::types::Role::System)
            .map(|m| AnthropicMessage {
                role: match m.role {
                    crate::types::Role::User => "user".to_string(),
                    crate::types::Role::Assistant => "assistant".to_string(),
                    _ => "user".to_string(),
                },
                content: m.content.clone(),
            })
            .collect();

        let anthropic_req = AnthropicRequest {
            model: req.model.clone(),
            messages,
            system,
            max_tokens: req.max_tokens.unwrap_or(4096),
            temperature: req.temperature,
            stream: false,
        };

        let response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .headers(self.build_headers())
            .json(&anthropic_req)
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed {
                status: e.status().map(|s| s.as_u16()).unwrap_or(0),
                message: e.to_string(),
            })?;

        let status = response.status().as_u16();
        if status == 429 {
            return Err(LLMError::RateLimited { retry_after: None });
        }
        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LLMError::RequestFailed {
                status,
                message: body,
            });
        }

        let resp: AnthropicResponse = response.json().await.map_err(|e| LLMError::ParseError {
            context: "Anthropic response".into(),
            source: e.to_string(),
        })?;

        let content = resp
            .content
            .first()
            .and_then(|c| c.text.clone())
            .unwrap_or_default();

        let finish_reason = resp.stop_reason.as_deref().map(|r| match r {
            "end_turn" => FinishReason::Stop,
            "max_tokens" => FinishReason::Length,
            "tool_use" => FinishReason::ToolCalls,
            _ => FinishReason::Stop,
        });

        Ok(CompletionResponse {
            content,
            model: resp.model,
            usage: Usage {
                prompt_tokens: resp.usage.input_tokens,
                completion_tokens: resp.usage.output_tokens,
                total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
            },
            finish_reason,
            tool_calls: Vec::new(),
        })
    }

    async fn stream(
        &self,
        req: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, LLMError>> + Send>>, LLMError> {
        let system = req
            .messages
            .iter()
            .find(|m| m.role == crate::types::Role::System)
            .map(|m| m.content.clone());

        let messages: Vec<AnthropicMessage> = req
            .messages
            .iter()
            .filter(|m| m.role != crate::types::Role::System)
            .map(|m| AnthropicMessage {
                role: match m.role {
                    crate::types::Role::User => "user".to_string(),
                    crate::types::Role::Assistant => "assistant".to_string(),
                    _ => "user".to_string(),
                },
                content: m.content.clone(),
            })
            .collect();

        let anthropic_req = AnthropicRequest {
            model: req.model.clone(),
            messages,
            system,
            max_tokens: req.max_tokens.unwrap_or(4096),
            temperature: req.temperature,
            stream: true,
        };

        let response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .headers(self.build_headers())
            .json(&anthropic_req)
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
                    for line in text.lines() {
                        let line = line.trim();
                        if let Some(data) = line.strip_prefix("data: ") {
                            if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data) {
                                match event.event_type.as_str() {
                                    "content_block_delta" => {
                                        if let Some(delta) = event.delta {
                                            if let Some(text) = delta.text {
                                                return Some(Ok(Token::text(&text)));
                                            }
                                        }
                                    }
                                    "message_stop" => {
                                        return Some(Ok(Token::done()));
                                    }
                                    _ => {}
                                }
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

    async fn embed(&self, _text: &str, _model: &str) -> Result<Vec<f32>, LLMError> {
        Err(LLMError::ProviderUnavailable(
            "Anthropic does not provide embeddings API".into(),
        ))
    }

    fn name(&self) -> &str {
        "anthropic"
    }

    fn models(&self) -> Vec<String> {
        vec![
            "claude-3-5-sonnet-20241022".into(),
            "claude-3-opus-20240229".into(),
            "claude-3-sonnet-20240229".into(),
            "claude-3-haiku-20240307".into(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = AnthropicProvider::new("test-key");
        assert_eq!(provider.name(), "anthropic");
        assert!(provider
            .models()
            .contains(&"claude-3-5-sonnet-20241022".to_string()));
    }
}
