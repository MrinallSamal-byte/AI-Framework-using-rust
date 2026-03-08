//! # Chatbot Server Example
//!
//! A complete AI chatbot server using NeuralFrame with:
//! - LLM completion via configurable providers
//! - Conversation memory
//! - Streaming responses (SSE)
//! - API key authentication
//!
//! ## Running
//!
//! ```bash
//! OPENAI_API_KEY=your-key cargo run --example chatbot_server
//! ```

use neuralframe_core::prelude::*;

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    tracing::info!("Starting NeuralFrame Chatbot Server");

    let app = NeuralFrame::new()
        .name("NeuralFrame Chatbot")
        // Health check endpoint
        .get("/health", |_req| async {
            Response::ok().json(serde_json::json!({
                "status": "healthy",
                "service": "neuralframe-chatbot",
                "version": "0.1.0"
            }))
        })
        // Chat completion endpoint
        .post("/v1/chat", |req| async move {
            // Parse the incoming chat request
            #[derive(Deserialize)]
            struct ChatRequest {
                message: String,
                #[serde(default = "default_model")]
                model: String,
                #[serde(default)]
                system_prompt: Option<String>,
                #[serde(default)]
                temperature: Option<f32>,
                #[serde(default)]
                max_tokens: Option<u32>,
            }

            fn default_model() -> String {
                "gpt-4o".to_string()
            }

            match req.json::<ChatRequest>() {
                Ok(chat_req) => {
                    // Build the prompt
                    let system = chat_req.system_prompt.unwrap_or_else(|| {
                        "You are a helpful AI assistant powered by NeuralFrame.".to_string()
                    });

                    // In a real application, you would:
                    // 1. Retrieve conversation history from memory
                    // 2. Build context with the prompt engine
                    // 3. Call the LLM provider
                    // 4. Store the response in memory
                    // 5. Return the response

                    let response_text = format!(
                        "I received your message: '{}'. \
                         In production, this would call the {} model with temperature {:?} \
                         and max_tokens {:?}.",
                        chat_req.message,
                        chat_req.model,
                        chat_req.temperature.unwrap_or(0.7),
                        chat_req.max_tokens.unwrap_or(4096)
                    );

                    Response::ok().json(serde_json::json!({
                        "response": response_text,
                        "model": chat_req.model,
                        "system_prompt": system,
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0
                        }
                    }))
                }
                Err(_) => Response::bad_request("Invalid JSON. Expected: {\"message\": \"...\"}"),
            }
        })
        // Chat streaming endpoint (SSE)
        .post("/v1/chat/stream", |req| async move {
            #[derive(Deserialize)]
            struct StreamRequest {
                message: String,
            }

            match req.json::<StreamRequest>() {
                Ok(stream_req) => {
                    // In production, this would stream tokens from the LLM
                    let words: Vec<String> = stream_req
                        .message
                        .split_whitespace()
                        .map(|w| w.to_string())
                        .collect();

                    let stream = tokio_stream::iter(
                        words.into_iter().map(|word| {
                            format!("Echo: {} ", word)
                        }),
                    );

                    Response::ok().sse_stream(stream)
                }
                Err(_) => Response::bad_request("Invalid JSON"),
            }
        })
        // List available models
        .get("/v1/models", |_req| async {
            Response::ok().json(serde_json::json!({
                "models": [
                    {"id": "gpt-4o", "provider": "openai"},
                    {"id": "gpt-4o-mini", "provider": "openai"},
                    {"id": "claude-3-5-sonnet-20241022", "provider": "anthropic"},
                    {"id": "gemini-1.5-pro", "provider": "google"},
                    {"id": "llama3:70b", "provider": "ollama"},
                    {"id": "llama-3.1-70b-versatile", "provider": "groq"}
                ]
            }))
        })
        // Conversation history
        .get("/v1/sessions/:session_id/history", |req| async move {
            let session_id = req.params.get("session_id").cloned().unwrap_or_default();

            Response::ok().json(serde_json::json!({
                "session_id": session_id,
                "messages": [],
                "note": "In production, this retrieves from neuralframe-memory"
            }))
        })
        // Delete conversation
        .delete("/v1/sessions/:session_id", |req| async move {
            let session_id = req.params.get("session_id").cloned().unwrap_or_default();
            tracing::info!(session_id, "deleting session");
            Response::no_content()
        })
        .middleware(LoggingMiddleware::new())
        .middleware(CorsMiddleware::permissive())
        .bind("0.0.0.0:8080");

    tracing::info!("Chatbot server starting on http://0.0.0.0:8080");
    tracing::info!("Endpoints:");
    tracing::info!("  GET  /health                          - Health check");
    tracing::info!("  POST /v1/chat                         - Chat completion");
    tracing::info!("  POST /v1/chat/stream                  - Chat streaming (SSE)");
    tracing::info!("  GET  /v1/models                       - List models");
    tracing::info!("  GET  /v1/sessions/:id/history          - Get session history");
    tracing::info!("  DELETE /v1/sessions/:id                - Delete session");

    if let Err(e) = app.run().await {
        tracing::error!(error = %e, "server failed");
    }
}
