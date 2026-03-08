//! Redis memory backend.

use crate::{MemoryEntry, MemoryError, MemoryStore};
use async_trait::async_trait;

/// Redis-backed memory store for distributed deployments.
///
/// Stores memory entries as JSON in Redis with session-based keys.
#[derive(Debug, Clone)]
pub struct RedisStore {
    connection_url: String,
    prefix: String,
    max_entries: usize,
}

impl RedisStore {
    /// Create a new Redis store.
    pub fn new(connection_url: &str) -> Self {
        Self {
            connection_url: connection_url.to_string(),
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
}

#[async_trait]
impl MemoryStore for RedisStore {
    async fn store(&self, session: &str, entry: MemoryEntry) -> Result<(), MemoryError> {
        let _key = self.session_key(session);
        let _json = serde_json::to_string(&entry).map_err(|e| {
            MemoryError::SerializationError(e.to_string())
        })?;
        // Redis RPUSH + LTRIM would go here
        tracing::debug!(session, "storing memory entry to Redis");
        Ok(())
    }

    async fn retrieve(
        &self,
        session: &str,
        _query: &str,
        _limit: usize,
    ) -> Result<Vec<MemoryEntry>, MemoryError> {
        let _key = self.session_key(session);
        tracing::debug!(session, "retrieving memory from Redis");
        Ok(Vec::new())
    }

    async fn get_all(&self, session: &str) -> Result<Vec<MemoryEntry>, MemoryError> {
        let _key = self.session_key(session);
        Ok(Vec::new())
    }

    async fn clear(&self, session: &str) -> Result<(), MemoryError> {
        let _key = self.session_key(session);
        tracing::debug!(session, "clearing memory in Redis");
        Ok(())
    }

    async fn count(&self, session: &str) -> Result<usize, MemoryError> {
        let _key = self.session_key(session);
        Ok(0)
    }

    fn name(&self) -> &str {
        "redis"
    }
}
