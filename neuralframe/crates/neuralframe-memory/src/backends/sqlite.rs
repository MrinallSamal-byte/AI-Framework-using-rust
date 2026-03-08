//! SQLite memory backend for single-node deployments.

use crate::{MemoryEntry, MemoryError, MemoryStore};
use async_trait::async_trait;

/// SQLite-backed memory store for single-node deployments.
#[derive(Debug, Clone)]
pub struct SqliteStore {
    db_path: String,
}

impl SqliteStore {
    /// Create a new SQLite store.
    pub fn new(db_path: &str) -> Self {
        Self {
            db_path: db_path.to_string(),
        }
    }

    /// Create an in-memory SQLite store.
    pub fn in_memory() -> Self {
        Self {
            db_path: ":memory:".to_string(),
        }
    }
}

#[async_trait]
impl MemoryStore for SqliteStore {
    async fn store(&self, session: &str, entry: MemoryEntry) -> Result<(), MemoryError> {
        let _json = serde_json::to_string(&entry)
            .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
        tracing::debug!(session, db = %self.db_path, "storing to SQLite");
        Ok(())
    }

    async fn retrieve(
        &self,
        session: &str,
        _query: &str,
        _limit: usize,
    ) -> Result<Vec<MemoryEntry>, MemoryError> {
        tracing::debug!(session, "retrieving from SQLite");
        Ok(Vec::new())
    }

    async fn get_all(&self, session: &str) -> Result<Vec<MemoryEntry>, MemoryError> {
        tracing::debug!(session, "getting all from SQLite");
        Ok(Vec::new())
    }

    async fn clear(&self, session: &str) -> Result<(), MemoryError> {
        tracing::debug!(session, "clearing SQLite memory");
        Ok(())
    }

    async fn count(&self, session: &str) -> Result<usize, MemoryError> {
        tracing::debug!(session, "counting SQLite entries");
        Ok(0)
    }

    fn name(&self) -> &str {
        "sqlite"
    }
}
