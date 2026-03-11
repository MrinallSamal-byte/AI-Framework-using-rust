//! OpenAI provider (GPT-4o, GPT-4-turbo, etc.).

use crate::error::LLMError;
use crate::providers::LLMProvider;
use crate::types::{CompletionRequest, CompletionResponse, FinishReason, Token, ToolCall, Usage};
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use tokio_stream::Stream;

/// OpenAI LLM provider.
///
/// # Examples
///
/// ```rust
/// use neuralframe_llm::providers::openai::OpenAIProvider;
///
/// let provider = OpenAIProvider::new("sk-your-api-key");
/// ```
#[derive(Debug, Clone)]
pub struct OpenAIProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
    organization: Option<String>,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider with the given API key.
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            client: reqwest::Client::new(),
            organization: None,
        }
    }

    /// Set a custom base URL (for proxies or Azure OpenAI).
    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = url.trim_end_matches('/').to_string();
        self
    }

    /// Set an organization ID.
    pub fn with_organization(mut self, org: &str) -> Self {
        self.organization = Some(org.to_string());
        self
    }

    /// Build the request headers.
    fn build_headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "Authorization",
            format!("Bearer {}", self.api_key)
                .parse()
                .expect("valid header value"),
        );
        headers.insert(
            "Content-Type",
            "application/json".parse().expect("valid header value"),
        );
        if let Some(ref org) = self.organization {
            headers.insert(
                "OpenAI-Organization",
                org.parse().expect("valid header value"),
            );
        }
        headers
    }
}

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(default)]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
}

#[derive(Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
    model: String,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: Option<OpenAIResponseMessage>,
    delta: Option<OpenAIDelta>,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

#[derive(Deserialize)]
struct OpenAIDelta {
    content: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIToolCall {
    id: String,
    function: OpenAIFunctionCall,
}

#[derive(Deserialize)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl From<&CompletionRequest> for OpenAIRequest {
    fn from(req: &CompletionRequest) -> Self {
        let messages = req
            .messages
            .iter()
            .map(|m| OpenAIMessage {
                role: match m.role {
                    crate::types::Role::System => "system".to_string(),
                    crate::types::Role::User => "user".to_string(),
                    crate::types::Role::Assistant => "assistant".to_string(),
                    crate::types::Role::Tool => "tool".to_string(),
                },
                content: m.content.clone(),
                name: m.name.clone(),
                tool_call_id: m.tool_call_id.clone(),
            })
            .collect();

        let tools = req.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    })
                })
                .collect()
        });

        OpenAIRequest {
            model: req.model.clone(),
            messages,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            stop: req.stop.clone(),
            stream: req.stream,
            tools,
        }
    }
}

#[async_trait]
impl LLMProvider for OpenAIProvider {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LLMError> {
        let openai_req = OpenAIRequest::from(&req);

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .headers(self.build_headers())
            .json(&openai_req)
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed {
                status: e.status().map(|s| s.as_u16()).unwrap_or(0),
                message: e.to_string(),
            })?;

        let status = response.status().as_u16();
        if status == 429 {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse().ok());
            return Err(LLMError::RateLimited { retry_after });
        }
        if status == 401 || status == 403 {
            let body = response.text().await.unwrap_or_default();
            return Err(LLMError::AuthError(body));
        }
        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LLMError::RequestFailed {
                status,
                message: body,
            });
        }

        let openai_resp: OpenAIResponse =
            response.json().await.map_err(|e| LLMError::ParseError {
                context: "OpenAI response".into(),
                source: e.to_string(),
            })?;

        let choice = openai_resp.choices.first().ok_or(LLMError::ParseError {
            context: "OpenAI response".into(),
            source: "no choices returned".into(),
        })?;

        let content = choice
            .message
            .as_ref()
            .and_then(|m| m.content.clone())
            .unwrap_or_default();

        let tool_calls = choice
            .message
            .as_ref()
            .and_then(|m| m.tool_calls.as_ref())
            .map(|calls| {
                calls
                    .iter()
                    .map(|c| ToolCall {
                        id: c.id.clone(),
                        name: c.function.name.clone(),
                        arguments: c.function.arguments.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        let finish_reason = choice.finish_reason.as_deref().map(|r| match r {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "tool_calls" => FinishReason::ToolCalls,
            "content_filter" => FinishReason::ContentFilter,
            _ => FinishReason::Stop,
        });

        let usage = openai_resp
            .usage
            .map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            })
            .unwrap_or_default();

        Ok(CompletionResponse {
            content,
            model: openai_resp.model,
            usage,
            finish_reason,
            tool_calls,
        })
    }

    async fn stream(
        &self,
        req: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, LLMError>> + Send>>, LLMError> {
        let mut openai_req = OpenAIRequest::from(&req);
        openai_req.stream = true;

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .headers(self.build_headers())
            .json(&openai_req)
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
                            if let Ok(chunk) = serde_json::from_str::<OpenAIResponse>(data) {
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

    async fn embed(&self, text: &str, model: &str) -> Result<Vec<f32>, LLMError> {
        let body = serde_json::json!({
            "model": model,
            "input": text
        });

        let response = self
            .client
            .post(format!("{}/embeddings", self.base_url))
            .headers(self.build_headers())
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
        struct EmbeddingResponse {
            data: Vec<EmbeddingData>,
        }

        #[derive(Deserialize)]
        struct EmbeddingData {
            embedding: Vec<f32>,
        }

        let resp: EmbeddingResponse = response.json().await.map_err(|e| LLMError::ParseError {
            context: "embedding response".into(),
            source: e.to_string(),
        })?;

        resp.data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or(LLMError::ParseError {
                context: "embedding response".into(),
                source: "no embeddings returned".into(),
            })
    }

    fn name(&self) -> &str {
        "openai"
    }

    fn models(&self) -> Vec<String> {
        vec![
            "gpt-4o".into(),
            "gpt-4o-mini".into(),
            "gpt-4-turbo".into(),
            "gpt-4".into(),
            "gpt-3.5-turbo".into(),
            "text-embedding-3-small".into(),
            "text-embedding-3-large".into(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = OpenAIProvider::new("test-key")
            .with_base_url("https://custom.api.com")
            .with_organization("org-123");

        assert_eq!(provider.name(), "openai");
        assert_eq!(provider.base_url, "https://custom.api.com");
        assert_eq!(provider.organization, Some("org-123".to_string()));
    }

    #[test]
    fn test_models() {
        let provider = OpenAIProvider::new("test");
        let models = provider.models();
        assert!(models.contains(&"gpt-4o".to_string()));
        assert!(models.contains(&"text-embedding-3-small".to_string()));
    }

    #[test]
    fn test_request_conversion() {
        let req = CompletionRequest::new("gpt-4o")
            .system("Be helpful")
            .user("Hello")
            .max_tokens(100)
            .temperature(0.7);

        let openai_req = OpenAIRequest::from(&req);
        assert_eq!(openai_req.model, "gpt-4o");
        assert_eq!(openai_req.messages.len(), 2);
        assert_eq!(openai_req.messages[0].role, "system");
        assert_eq!(openai_req.messages[1].role, "user");
        assert_eq!(openai_req.max_tokens, Some(100));
    }
}
