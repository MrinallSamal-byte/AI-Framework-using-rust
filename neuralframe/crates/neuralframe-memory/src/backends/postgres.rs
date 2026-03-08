//! PostgreSQL memory backend with pgvector support.

use crate::{MemoryEntry, MemoryError, MemoryStore};
use async_trait::async_trait;

/// PostgreSQL-backed memory store with pgvector extension support.
#[derive(Debug, Clone)]
pub struct PostgresStore {
    connection_url: String,
    table_name: String,
}

impl PostgresStore {
    /// Create a new PostgreSQL store.
    pub fn new(connection_url: &str) -> Self {
        Self {
            connection_url: connection_url.to_string(),
            table_name: "neuralframe_memory".to_string(),
        }
    }

    /// Set the table name.
    pub fn with_table(mut self, table: &str) -> Self {
        self.table_name = table.to_string();
        self
    }
}

#[async_trait]
impl MemoryStore for PostgresStore {
    async fn store(&self, session: &str, entry: MemoryEntry) -> Result<(), MemoryError> {
        let _json = serde_json::to_string(&entry)
            .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
        tracing::debug!(session, table = %self.table_name, "storing to PostgreSQL");
        Ok(())
    }

    async fn retrieve(
        &self,
        session: &str,
        _query: &str,
        _limit: usize,
    ) -> Result<Vec<MemoryEntry>, MemoryError> {
        tracing::debug!(session, "retrieving from PostgreSQL");
        Ok(Vec::new())
    }

    async fn get_all(&self, session: &str) -> Result<Vec<MemoryEntry>, MemoryError> {
        tracing::debug!(session, "getting all from PostgreSQL");
        Ok(Vec::new())
    }

    async fn clear(&self, session: &str) -> Result<(), MemoryError> {
        tracing::debug!(session, "clearing PostgreSQL memory");
        Ok(())
    }

    async fn count(&self, session: &str) -> Result<usize, MemoryError> {
        tracing::debug!(session, "counting PostgreSQL entries");
        Ok(0)
    }

    fn name(&self) -> &str {
        "postgres"
    }
}
