//! Redis memory backend.

use crate::{MemoryEntry, MemoryError, MemoryStore};
use async_trait::async_trait;
use std::collections::HashSet;

/// Redis-backed memory store for distributed deployments.
///
/// Stores memory entries as JSON in Redis with session-based keys.
#[derive(Debug, Clone)]
pub struct RedisStore {
    client: redis::Client,
    prefix: String,
    max_entries: usize,
}

impl RedisStore {
    /// Create a new Redis store.
    pub fn new(connection_url: &str) -> Self {
        let client = redis::Client::open(connection_url)
            .expect("Invalid Redis connection URL");
        Self {
            client,
            prefix: "nf:memory:".to_string(),
            max_entries: 1000,
        }
    }

    /// Set a key prefix.
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefix = prefix.to_string();
        self
    }

    fn session_key(&self, session: &str) -> String {
        format!("{}{}", self.prefix, session)
    }

    async fn get_connection(&self) -> Result<redis::aio::MultiplexedConnection, MemoryError> {
        self.client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| MemoryError::ConnectionError(e.to_string()))
    }
}

#[async_trait]
impl MemoryStore for RedisStore {
    async fn store(&self, session: &str, entry: MemoryEntry) -> Result<(), MemoryError> {
        let key = self.session_key(session);
        let json = serde_json::to_string(&entry)
            .map_err(|e| MemoryError::SerializationError(e.to_string()))?;

        let mut conn = self.get_connection().await?;

        redis::pipe()
            .cmd("RPUSH")
            .arg(&key)
            .arg(&json)
            .cmd("LTRIM")
            .arg(&key)
            .arg(0i64)
            .arg((self.max_entries - 1) as i64)
            .query_async::<_, ()>(&mut conn)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        tracing::debug!(session, "storing memory entry to Redis");
        Ok(())
    }

    async fn retrieve(
        &self,
        session: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, MemoryError> {
        let all = self.get_all(session).await?;
        let query_lower = query.to_lowercase();

        let mut scored: Vec<(f32, MemoryEntry)> = all
            .into_iter()
            .map(|entry| {
                let content_lower = entry.content.to_lowercase();
                let score = if content_lower.contains(&query_lower) {
                    1.0
                } else {
                    let query_words: HashSet<&str> = query_lower.split_whitespace().collect();
                    let content_words: HashSet<&str> = content_lower.split_whitespace().collect();
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
        let key = self.session_key(session);
        let mut conn = self.get_connection().await?;

        let items: Vec<String> = redis::cmd("LRANGE")
            .arg(&key)
            .arg(0i64)
            .arg(-1i64)
            .query_async(&mut conn)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let mut entries = Vec::with_capacity(items.len());
        for item in items {
            let entry: MemoryEntry = serde_json::from_str(&item)
                .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
            entries.push(entry);
        }
        Ok(entries)
    }

    async fn clear(&self, session: &str) -> Result<(), MemoryError> {
        let key = self.session_key(session);
        let mut conn = self.get_connection().await?;

        redis::cmd("DEL")
            .arg(&key)
            .query_async::<_, ()>(&mut conn)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        tracing::debug!(session, "clearing memory in Redis");
        Ok(())
    }

    async fn count(&self, session: &str) -> Result<usize, MemoryError> {
        let key = self.session_key(session);
        let mut conn = self.get_connection().await?;

        let len: i64 = redis::cmd("LLEN")
            .arg(&key)
            .query_async(&mut conn)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        Ok(len as usize)
    }

    fn name(&self) -> &str {
        "redis"
    }
}

#[cfg(all(test, feature = "redis-tests"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_redis_store_and_retrieve() {
        let store = RedisStore::new("redis://127.0.0.1/");

        store.clear("test_s1").await.unwrap();

        store
            .store("test_s1", MemoryEntry::new("test_s1", "user", "Hello world"))
            .await
            .unwrap();
        store
            .store(
                "test_s1",
                MemoryEntry::new("test_s1", "assistant", "Hi there"),
            )
            .await
            .unwrap();
        store
            .store(
                "test_s1",
                MemoryEntry::new("test_s1", "user", "Tell me about Rust"),
            )
            .await
            .unwrap();

        assert_eq!(store.count("test_s1").await.unwrap(), 3);

        let results = store.retrieve("test_s1", "Rust", 2).await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].content.contains("Rust"));

        store.clear("test_s1").await.unwrap();
        assert_eq!(store.count("test_s1").await.unwrap(), 0);
    }
}
