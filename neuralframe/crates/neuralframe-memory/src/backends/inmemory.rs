//! In-memory memory store using DashMap for concurrent access.

use crate::{MemoryEntry, MemoryError, MemoryStore};
use async_trait::async_trait;
use dashmap::DashMap;
use std::sync::Arc;

/// In-memory store for development and testing.
///
/// Uses `DashMap` for lock-free concurrent access.
#[derive(Debug, Clone)]
pub struct InMemoryStore {
    sessions: Arc<DashMap<String, Vec<MemoryEntry>>>,
    max_entries_per_session: usize,
}

impl InMemoryStore {
    /// Create a new in-memory store.
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(DashMap::new()),
            max_entries_per_session: 1000,
        }
    }

    /// Set the maximum entries per session.
    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries_per_session = max;
        self
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryStore for InMemoryStore {
    async fn store(&self, session: &str, entry: MemoryEntry) -> Result<(), MemoryError> {
        let mut entries = self.sessions.entry(session.to_string()).or_default();
        entries.push(entry);

        // Trim to max entries (sliding window)
        if entries.len() > self.max_entries_per_session {
            let excess = entries.len() - self.max_entries_per_session;
            entries.drain(0..excess);
        }

        Ok(())
    }

    async fn retrieve(
        &self,
        session: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, MemoryError> {
        let entries = self
            .sessions
            .get(session)
            .map(|e| e.value().clone())
            .unwrap_or_default();

        // Simple text-based relevance scoring
        let query_lower = query.to_lowercase();
        let mut scored: Vec<(f32, MemoryEntry)> = entries
            .into_iter()
            .map(|entry| {
                let content_lower = entry.content.to_lowercase();
                let score = if content_lower.contains(&query_lower) {
                    1.0
                } else {
                    // Word overlap scoring
                    let query_words: std::collections::HashSet<&str> =
                        query_lower.split_whitespace().collect();
                    let content_words: std::collections::HashSet<&str> =
                        content_lower.split_whitespace().collect();
                    let intersection = query_words.intersection(&content_words).count();
                    if query_words.is_empty() {
                        0.0
                    } else {
                        intersection as f32 / query_words.len() as f32
                    }
                };
                (score, entry)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored
            .into_iter()
            .take(limit)
            .map(|(score, mut entry)| {
                entry.relevance = score;
                entry
            })
            .collect())
    }

    async fn get_all(&self, session: &str) -> Result<Vec<MemoryEntry>, MemoryError> {
        Ok(self
            .sessions
            .get(session)
            .map(|e| e.value().clone())
            .unwrap_or_default())
    }

    async fn clear(&self, session: &str) -> Result<(), MemoryError> {
        self.sessions.remove(session);
        Ok(())
    }

    async fn count(&self, session: &str) -> Result<usize, MemoryError> {
        Ok(self
            .sessions
            .get(session)
            .map(|e| e.len())
            .unwrap_or(0))
    }

    fn name(&self) -> &str {
        "inmemory"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let store = InMemoryStore::new();

        store
            .store("s1", MemoryEntry::new("s1", "user", "Hello!"))
            .await
            .unwrap();
        store
            .store(
                "s1",
                MemoryEntry::new("s1", "assistant", "Hi there!"),
            )
            .await
            .unwrap();

        let entries = store.get_all("s1").await.unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[tokio::test]
    async fn test_retrieve_with_query() {
        let store = InMemoryStore::new();

        store
            .store(
                "s1",
                MemoryEntry::new("s1", "user", "Tell me about Rust programming"),
            )
            .await
            .unwrap();
        store
            .store(
                "s1",
                MemoryEntry::new("s1", "user", "What's the weather?"),
            )
            .await
            .unwrap();

        let results = store.retrieve("s1", "Rust", 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Rust"));
    }

    #[tokio::test]
    async fn test_clear() {
        let store = InMemoryStore::new();

        store
            .store("s1", MemoryEntry::new("s1", "user", "Hello"))
            .await
            .unwrap();
        assert_eq!(store.count("s1").await.unwrap(), 1);

        store.clear("s1").await.unwrap();
        assert_eq!(store.count("s1").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_max_entries() {
        let store = InMemoryStore::new().with_max_entries(2);

        store
            .store("s1", MemoryEntry::new("s1", "user", "First"))
            .await
            .unwrap();
        store
            .store("s1", MemoryEntry::new("s1", "user", "Second"))
            .await
            .unwrap();
        store
            .store("s1", MemoryEntry::new("s1", "user", "Third"))
            .await
            .unwrap();

        let entries = store.get_all("s1").await.unwrap();
        assert_eq!(entries.len(), 2);
        // First entry should have been evicted
        assert_eq!(entries[0].content, "Second");
    }

    #[tokio::test]
    async fn test_separate_sessions() {
        let store = InMemoryStore::new();

        store
            .store("s1", MemoryEntry::new("s1", "user", "Hello s1"))
            .await
            .unwrap();
        store
            .store("s2", MemoryEntry::new("s2", "user", "Hello s2"))
            .await
            .unwrap();

        assert_eq!(store.count("s1").await.unwrap(), 1);
        assert_eq!(store.count("s2").await.unwrap(), 1);
    }
}
