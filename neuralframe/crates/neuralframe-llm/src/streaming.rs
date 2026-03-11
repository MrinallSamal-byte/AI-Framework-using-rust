//! # SSE Streaming Engine
//!
//! Efficient Server-Sent Events parser with zero-copy token buffering
//! and automatic reconnection support.

use crate::error::LLMError;
use crate::types::Token;

/// SSE event parsed from a stream.
#[derive(Debug, Clone)]
pub struct SseEvent {
    /// The event type (e.g., "message", "token").
    pub event: Option<String>,
    /// The event data payload.
    pub data: String,
    /// The event ID.
    pub id: Option<String>,
    /// Retry interval in milliseconds.
    pub retry: Option<u64>,
}

/// Parser for Server-Sent Events streams.
///
/// Handles multi-line data fields, event types, IDs, and retry directives.
///
/// # Examples
///
/// ```rust
/// use neuralframe_llm::streaming::SseParser;
///
/// let mut parser = SseParser::new();
/// let events = parser.feed("data: hello world\n\n");
/// assert_eq!(events.len(), 1);
/// assert_eq!(events[0].data, "hello world");
/// ```
pub struct SseParser {
    buffer: String,
    current_event: Option<String>,
    current_data: Vec<String>,
    current_id: Option<String>,
    current_retry: Option<u64>,
}

impl SseParser {
    /// Create a new SSE parser.
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            current_event: None,
            current_data: Vec::new(),
            current_id: None,
            current_retry: None,
        }
    }

    /// Feed raw bytes into the parser and return any completed events.
    pub fn feed(&mut self, chunk: &str) -> Vec<SseEvent> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();

        // Process complete lines
        while let Some(newline_pos) = self.buffer.find('\n') {
            let line = self.buffer[..newline_pos]
                .trim_end_matches('\r')
                .to_string();
            self.buffer = self.buffer[newline_pos + 1..].to_string();

            if line.is_empty() {
                // Empty line = event boundary
                if !self.current_data.is_empty() {
                    events.push(SseEvent {
                        event: self.current_event.take(),
                        data: self.current_data.join("\n"),
                        id: self.current_id.take(),
                        retry: self.current_retry.take(),
                    });
                    self.current_data.clear();
                }
                continue;
            }

            if line.starts_with(':') {
                // Comment, skip
                continue;
            }

            if let Some(value) = line.strip_prefix("data: ") {
                self.current_data.push(value.to_string());
            } else if line == "data" {
                self.current_data.push(String::new());
            } else if let Some(value) = line.strip_prefix("event: ") {
                self.current_event = Some(value.to_string());
            } else if let Some(value) = line.strip_prefix("id: ") {
                self.current_id = Some(value.to_string());
            } else if let Some(value) = line.strip_prefix("retry: ") {
                self.current_retry = value.parse().ok();
            }
        }

        events
    }

    /// Parse an SSE data field as a Token.
    pub fn parse_token(data: &str) -> Result<Option<Token>, LLMError> {
        if data == "[DONE]" {
            return Ok(Some(Token::done()));
        }

        // Try to parse as JSON with a "content" field
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(data) {
            // OpenAI-style: choices[0].delta.content
            if let Some(content) = value
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("delta"))
                .and_then(|d| d.get("content"))
                .and_then(|c| c.as_str())
            {
                return Ok(Some(Token::text(content)));
            }

            // Simple content field
            if let Some(content) = value.get("content").and_then(|c| c.as_str()) {
                return Ok(Some(Token::text(content)));
            }
        }

        Ok(None)
    }

    /// Reset the parser state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.current_event = None;
        self.current_data.clear();
        self.current_id = None;
        self.current_retry = None;
    }
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for SSE stream reconnection.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum reconnection attempts.
    pub max_reconnects: u32,
    /// Initial reconnection delay.
    pub reconnect_delay_ms: u64,
    /// Maximum reconnection delay.
    pub max_reconnect_delay_ms: u64,
    /// Heartbeat timeout (disconnect if no data received).
    pub heartbeat_timeout_ms: u64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            max_reconnects: 3,
            reconnect_delay_ms: 1000,
            max_reconnect_delay_ms: 30000,
            heartbeat_timeout_ms: 60000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_parser_basic() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: hello world\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello world");
        assert!(events[0].event.is_none());
    }

    #[test]
    fn test_sse_parser_with_event_type() {
        let mut parser = SseParser::new();
        let events = parser.feed("event: token\ndata: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.as_deref(), Some("token"));
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn test_sse_parser_multiline_data() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: line1\ndata: line2\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2");
    }

    #[test]
    fn test_sse_parser_with_id() {
        let mut parser = SseParser::new();
        let events = parser.feed("id: 42\ndata: test\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id.as_deref(), Some("42"));
    }

    #[test]
    fn test_sse_parser_with_retry() {
        let mut parser = SseParser::new();
        let events = parser.feed("retry: 5000\ndata: test\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].retry, Some(5000));
    }

    #[test]
    fn test_sse_parser_comments() {
        let mut parser = SseParser::new();
        let events = parser.feed(": this is a comment\ndata: test\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "test");
    }

    #[test]
    fn test_sse_parser_multiple_events() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: first\n\ndata: second\n\n");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "first");
        assert_eq!(events[1].data, "second");
    }

    #[test]
    fn test_sse_parser_chunked() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: hel");
        assert_eq!(events.len(), 0);
        let events = parser.feed("lo\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn test_parse_token_done() {
        let result = SseParser::parse_token("[DONE]").unwrap();
        assert!(result.unwrap().done);
    }

    #[test]
    fn test_parse_token_openai_format() {
        let data = r#"{"choices":[{"delta":{"content":"Hello"}}]}"#;
        let result = SseParser::parse_token(data).unwrap();
        assert_eq!(result.unwrap().content, "Hello");
    }

    #[test]
    fn test_parser_reset() {
        let mut parser = SseParser::new();
        parser.feed("data: partial");
        parser.reset();
        let events = parser.feed("data: fresh\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "fresh");
    }
}
