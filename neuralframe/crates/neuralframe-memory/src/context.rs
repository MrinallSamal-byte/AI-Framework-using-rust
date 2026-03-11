//! Context assembly for fitting memory within token limits.

use crate::{MemoryEntry, MemoryError};

/// Context builder assembles relevant memory into a prompt context
/// that fits within token limits.
pub struct ContextBuilder {
    max_tokens: usize,
    entries: Vec<MemoryEntry>,
    system_context: Option<String>,
}

impl ContextBuilder {
    /// Create a new context builder with a token limit.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            entries: Vec::new(),
            system_context: None,
        }
    }

    /// Set system-level context.
    pub fn system(mut self, context: &str) -> Self {
        self.system_context = Some(context.to_string());
        self
    }

    /// Add memory entries (sorted by relevance).
    pub fn add_entries(mut self, entries: Vec<MemoryEntry>) -> Self {
        self.entries.extend(entries);
        self
    }

    /// Build the context string, fitting within token limits.
    ///
    /// Uses a greedy approach: adds entries by relevance until
    /// the token budget is exhausted.
    pub fn build(&self) -> Result<String, MemoryError> {
        let mut parts = Vec::new();
        let mut token_budget = self.max_tokens;

        // Add system context first (highest priority)
        if let Some(sys) = &self.system_context {
            let tokens = count_tokens(sys, "gpt-4");
            if tokens <= token_budget {
                parts.push(sys.clone());
                token_budget -= tokens;
            }
        }

        // Sort entries by relevance (highest first)
        let mut sorted = self.entries.clone();
        sorted.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Add entries until budget is exhausted
        for entry in &sorted {
            let formatted = format!("{}: {}", entry.role, entry.content);
            let tokens = count_tokens(&formatted, "gpt-4");
            if tokens <= token_budget {
                parts.push(formatted);
                token_budget -= tokens;
            }
        }

        Ok(parts.join("\n\n"))
    }

    /// Get the remaining token budget.
    pub fn remaining_tokens(&self) -> usize {
        let used: usize = self
            .entries
            .iter()
            .map(|e| count_tokens(&e.content, "gpt-4"))
            .sum::<usize>()
            + self
                .system_context
                .as_deref()
                .map_or(0, |text| count_tokens(text, "gpt-4"));
        self.max_tokens.saturating_sub(used)
    }
}

fn count_tokens(text: &str, model: &str) -> usize {
    tiktoken_rs::get_bpe_from_model(model)
        .or_else(|_| tiktoken_rs::get_bpe_from_model("gpt-4"))
        .map(|bpe| bpe.encode_with_special_tokens(text).len())
        .unwrap_or_else(|_| (text.len() as f64 / 4.0).ceil() as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_builder_basic() {
        let ctx = ContextBuilder::new(1000)
            .system("You are a helpful assistant.")
            .build()
            .unwrap();

        assert!(ctx.contains("You are a helpful assistant."));
    }

    #[test]
    fn test_context_builder_with_entries() {
        let entries = vec![
            MemoryEntry {
                id: "1".into(),
                session_id: "s1".into(),
                role: "user".into(),
                content: "Hello".into(),
                timestamp: chrono::Utc::now(),
                metadata: serde_json::json!({}),
                embedding: None,
                relevance: 1.0,
            },
            MemoryEntry {
                id: "2".into(),
                session_id: "s1".into(),
                role: "assistant".into(),
                content: "Hi there!".into(),
                timestamp: chrono::Utc::now(),
                metadata: serde_json::json!({}),
                embedding: None,
                relevance: 0.9,
            },
        ];

        let ctx = ContextBuilder::new(1000)
            .add_entries(entries)
            .build()
            .unwrap();

        assert!(ctx.contains("Hello"));
        assert!(ctx.contains("Hi there!"));
    }

    #[test]
    fn test_context_builder_token_limit() {
        let entries = vec![MemoryEntry {
            id: "1".into(),
            session_id: "s1".into(),
            role: "user".into(),
            content: "x".repeat(1000),
            timestamp: chrono::Utc::now(),
            metadata: serde_json::json!({}),
            embedding: None,
            relevance: 1.0,
        }];

        let ctx = ContextBuilder::new(10) // Very small budget
            .add_entries(entries)
            .build()
            .unwrap();

        // Entry is too large, should not be included
        assert!(ctx.is_empty() || count_tokens(&ctx, "gpt-4") <= 10);
    }

    #[test]
    fn test_count_tokens() {
        assert_eq!(count_tokens("", "gpt-4"), 0);
        assert!(count_tokens("test", "gpt-4") > 0);
        assert!(count_tokens("Hello, world!", "gpt-4") > 0);
    }
}
