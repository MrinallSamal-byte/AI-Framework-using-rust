//! Groq provider (fast inference).

use crate::error::LLMError;
use crate::providers::LLMProvider;
use crate::types::{CompletionRequest, CompletionResponse, FinishReason, Token, Usage};
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use tokio_stream::Stream;

/// Groq LLM provider for fast inference.
///
/// Uses the OpenAI-compatible chat completions API.
#[derive(Debug, Clone)]
pub struct GroqProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl GroqProvider {
    /// Create a new Groq provider.
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.groq.com/openai/v1".to_string(),
            client: reqwest::Client::new(),
        }
    }

    fn build_headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "Authorization",
            format!("Bearer {}", self.api_key)
                .parse()
                .expect("valid header"),
        );
        headers.insert(
            "Content-Type",
            "application/json".parse().expect("valid header"),
        );
        headers
    }
}

/// Groq uses OpenAI-compatible format
#[derive(Serialize)]
struct GroqRequest {
    model: String,
    messages: Vec<GroqMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
}

#[derive(Serialize, Deserialize)]
struct GroqMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct GroqResponse {
    choices: Vec<GroqChoice>,
    usage: Option<GroqUsage>,
    model: String,
}

#[derive(Deserialize)]
struct GroqChoice {
    message: Option<GroqMessage>,
    delta: Option<GroqDelta>,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct GroqDelta {
    content: Option<String>,
}

#[derive(Deserialize)]
struct GroqUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[async_trait]
impl LLMProvider for GroqProvider {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LLMError> {
        let messages: Vec<GroqMessage> = req
            .messages
            .iter()
            .map(|m| GroqMessage {
                role: match m.role {
                    crate::types::Role::System => "system".to_string(),
                    crate::types::Role::User => "user".to_string(),
                    crate::types::Role::Assistant => "assistant".to_string(),
                    crate::types::Role::Tool => "user".to_string(),
                },
                content: m.content.clone(),
            })
            .collect();

        let groq_req = GroqRequest {
            model: req.model.clone(),
            messages,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            stream: false,
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .headers(self.build_headers())
            .json(&groq_req)
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

        let resp: GroqResponse =
            response.json().await.map_err(|e| LLMError::ParseError {
                context: "Groq response".into(),
                source: e.to_string(),
            })?;

        let choice = resp.choices.first().ok_or(LLMError::ParseError {
            context: "Groq response".into(),
            source: "no choices".into(),
        })?;

        let content = choice
            .message
            .as_ref()
            .map(|m| m.content.clone())
            .unwrap_or_default();

        let finish_reason = choice.finish_reason.as_deref().map(|r| match r {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            _ => FinishReason::Stop,
        });

        let usage = resp
            .usage
            .map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            })
            .unwrap_or_default();

        Ok(CompletionResponse {
            content,
            model: resp.model,
            usage,
            finish_reason,
            tool_calls: Vec::new(),
        })
    }

    async fn stream(
        &self,
        req: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, LLMError>> + Send>>, LLMError> {
        let messages: Vec<GroqMessage> = req
            .messages
            .iter()
            .map(|m| GroqMessage {
                role: match m.role {
                    crate::types::Role::System => "system".to_string(),
                    crate::types::Role::User => "user".to_string(),
                    crate::types::Role::Assistant => "assistant".to_string(),
                    crate::types::Role::Tool => "user".to_string(),
                },
                content: m.content.clone(),
            })
            .collect();

        let groq_req = GroqRequest {
            model: req.model.clone(),
            messages,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            stream: true,
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .headers(self.build_headers())
            .json(&groq_req)
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
                            if data == "[DONE]" {
                                return Some(Ok(Token::done()));
                            }
                            if let Ok(chunk) = serde_json::from_str::<GroqResponse>(data) {
                                if let Some(choice) = chunk.choices.first() {
                                    if let Some(delta) = &choice.delta {
                                        if let Some(content) = &delta.content {
                                            return Some(Ok(Token::text(content)));
                                        }
                                    }
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
            "Groq does not provide embeddings API".into(),
        ))
    }

    fn name(&self) -> &str {
        "groq"
    }

    fn models(&self) -> Vec<String> {
        vec![
            "llama-3.1-70b-versatile".into(),
            "llama-3.1-8b-instant".into(),
            "mixtral-8x7b-32768".into(),
            "gemma2-9b-it".into(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = GroqProvider::new("test-key");
        assert_eq!(provider.name(), "groq");
        assert!(!provider.models().is_empty());
    }
}
