//! Google Gemini provider.

use crate::error::LLMError;
use crate::providers::LLMProvider;
use crate::types::{CompletionRequest, CompletionResponse, FinishReason, Token, Usage};
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use tokio_stream::Stream;

/// Google Gemini LLM provider.
#[derive(Debug, Clone)]
pub struct GoogleProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl GoogleProvider {
    /// Create a new Google Gemini provider.
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Set a custom base URL.
    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = url.trim_end_matches('/').to_string();
        self
    }
}

#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Serialize, Deserialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Deserialize)]
struct GeminiPart {
    text: String,
}

#[derive(Serialize)]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsage>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiContent>,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct GeminiUsage {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: Option<u32>,
    #[serde(rename = "totalTokenCount")]
    total_token_count: Option<u32>,
}

#[async_trait]
impl LLMProvider for GoogleProvider {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LLMError> {
        let system_instruction = req
            .messages
            .iter()
            .find(|m| m.role == crate::types::Role::System)
            .map(|m| GeminiContent {
                role: "user".to_string(),
                parts: vec![GeminiPart {
                    text: m.content.clone(),
                }],
            });

        let contents: Vec<GeminiContent> = req
            .messages
            .iter()
            .filter(|m| m.role != crate::types::Role::System)
            .map(|m| GeminiContent {
                role: match m.role {
                    crate::types::Role::User => "user".to_string(),
                    crate::types::Role::Assistant => "model".to_string(),
                    _ => "user".to_string(),
                },
                parts: vec![GeminiPart {
                    text: m.content.clone(),
                }],
            })
            .collect();

        let gemini_req = GeminiRequest {
            contents,
            system_instruction,
            generation_config: Some(GeminiGenerationConfig {
                max_output_tokens: req.max_tokens,
                temperature: req.temperature,
                top_p: req.top_p,
            }),
        };

        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url, req.model, self.api_key
        );

        let response = self
            .client
            .post(&url)
            .json(&gemini_req)
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

        let resp: GeminiResponse =
            response.json().await.map_err(|e| LLMError::ParseError {
                context: "Gemini response".into(),
                source: e.to_string(),
            })?;

        let candidate = resp
            .candidates
            .as_ref()
            .and_then(|c| c.first())
            .ok_or(LLMError::ParseError {
                context: "Gemini response".into(),
                source: "no candidates returned".into(),
            })?;

        let content = candidate
            .content
            .as_ref()
            .and_then(|c| c.parts.first())
            .map(|p| p.text.clone())
            .unwrap_or_default();

        let finish_reason = candidate.finish_reason.as_deref().map(|r| match r {
            "STOP" => FinishReason::Stop,
            "MAX_TOKENS" => FinishReason::Length,
            "SAFETY" => FinishReason::ContentFilter,
            _ => FinishReason::Stop,
        });

        let usage = resp
            .usage_metadata
            .map(|u| Usage {
                prompt_tokens: u.prompt_token_count.unwrap_or(0),
                completion_tokens: u.candidates_token_count.unwrap_or(0),
                total_tokens: u.total_token_count.unwrap_or(0),
            })
            .unwrap_or_default();

        Ok(CompletionResponse {
            content,
            model: req.model,
            usage,
            finish_reason,
            tool_calls: Vec::new(),
        })
    }

    async fn stream(
        &self,
        req: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, LLMError>> + Send>>, LLMError> {
        let system_instruction = req
            .messages
            .iter()
            .find(|m| m.role == crate::types::Role::System)
            .map(|m| GeminiContent {
                role: "user".to_string(),
                parts: vec![GeminiPart {
                    text: m.content.clone(),
                }],
            });

        let contents: Vec<GeminiContent> = req
            .messages
            .iter()
            .filter(|m| m.role != crate::types::Role::System)
            .map(|m| GeminiContent {
                role: match m.role {
                    crate::types::Role::User => "user".to_string(),
                    crate::types::Role::Assistant => "model".to_string(),
                    _ => "user".to_string(),
                },
                parts: vec![GeminiPart {
                    text: m.content.clone(),
                }],
            })
            .collect();

        let gemini_req = GeminiRequest {
            contents,
            system_instruction,
            generation_config: Some(GeminiGenerationConfig {
                max_output_tokens: req.max_tokens,
                temperature: req.temperature,
                top_p: req.top_p,
            }),
        };

        let url = format!(
            "{}/models/{}:streamGenerateContent?key={}&alt=sse",
            self.base_url, req.model, self.api_key
        );

        let response = self
            .client
            .post(&url)
            .json(&gemini_req)
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
                        if let Some(data) = line.strip_prefix("data: ") {
                            if let Ok(resp) = serde_json::from_str::<GeminiResponse>(data) {
                                if let Some(candidates) = &resp.candidates {
                                    if let Some(candidate) = candidates.first() {
                                        if let Some(content) = &candidate.content {
                                            if let Some(part) = content.parts.first() {
                                                return Some(Ok(Token::text(&part.text)));
                                            }
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

    async fn embed(&self, text: &str, model: &str) -> Result<Vec<f32>, LLMError> {
        let body = serde_json::json!({
            "model": format!("models/{}", model),
            "content": {
                "parts": [{"text": text}]
            }
        });

        let url = format!(
            "{}/models/{}:embedContent?key={}",
            self.base_url, model, self.api_key
        );

        let response = self
            .client
            .post(&url)
            .json(&body)
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

        #[derive(Deserialize)]
        struct EmbedResponse {
            embedding: EmbedValues,
        }
        #[derive(Deserialize)]
        struct EmbedValues {
            values: Vec<f32>,
        }

        let resp: EmbedResponse =
            response.json().await.map_err(|e| LLMError::ParseError {
                context: "Gemini embedding".into(),
                source: e.to_string(),
            })?;

        Ok(resp.embedding.values)
    }

    fn name(&self) -> &str {
        "google"
    }

    fn models(&self) -> Vec<String> {
        vec![
            "gemini-pro".into(),
            "gemini-1.5-pro".into(),
            "gemini-1.5-flash".into(),
            "embedding-001".into(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = GoogleProvider::new("test-key");
        assert_eq!(provider.name(), "google");
    }
}
