//! PostgreSQL memory backend with pgvector support.

use crate::{MemoryEntry, MemoryError, MemoryStore};
use async_trait::async_trait;
use sqlx::postgres::PgPoolOptions;
use sqlx::{Row, PgPool};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::OnceCell;

/// PostgreSQL-backed memory store with pgvector extension support.
#[derive(Debug, Clone)]
pub struct PostgresStore {
    connection_url: String,
    table_name: String,
    pool: Arc<OnceCell<PgPool>>,
}

impl PostgresStore {
    /// Create a new PostgreSQL store.
    pub fn new(connection_url: &str) -> Self {
        Self {
            connection_url: connection_url.to_string(),
            table_name: "neuralframe_memory".to_string(),
            pool: Arc::new(OnceCell::new()),
        }
    }

    /// Set the table name.
    pub fn with_table(mut self, table: &str) -> Self {
        self.table_name = table.to_string();
        self
    }

    /// Get or initialize the connection pool and create tables.
    async fn pool(&self) -> Result<&PgPool, MemoryError> {
        self.pool
            .get_or_try_init(|| async {
                let pool = PgPoolOptions::new()
                    .max_connections(5)
                    .connect(&self.connection_url)
                    .await
                    .map_err(|e| MemoryError::ConnectionError(e.to_string()))?;

                let create_table = format!(
                    "CREATE TABLE IF NOT EXISTS {} (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}',
                        embedding REAL[],
                        relevance REAL DEFAULT 0
                    )",
                    self.table_name
                );
                sqlx::query(&create_table)
                    .execute(&pool)
                    .await
                    .map_err(|e| MemoryError::StorageError(e.to_string()))?;

                let create_index = format!(
                    "CREATE INDEX IF NOT EXISTS idx_nf_session ON {}(session_id)",
                    self.table_name
                );
                sqlx::query(&create_index)
                    .execute(&pool)
                    .await
                    .map_err(|e| MemoryError::StorageError(e.to_string()))?;

                Ok(pool)
            })
            .await
    }
}

#[async_trait]
impl MemoryStore for PostgresStore {
    async fn store(&self, session: &str, entry: MemoryEntry) -> Result<(), MemoryError> {
        let pool = self.pool().await?;
        let metadata = entry.metadata.clone();
        let embedding: Option<Vec<f32>> = entry.embedding.clone();

        let query_str = format!(
            "INSERT INTO {} (id, session_id, role, content, timestamp, metadata, embedding, relevance)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
             ON CONFLICT (id) DO UPDATE SET
                 session_id = EXCLUDED.session_id,
                 role = EXCLUDED.role,
                 content = EXCLUDED.content,
                 timestamp = EXCLUDED.timestamp,
                 metadata = EXCLUDED.metadata,
                 embedding = EXCLUDED.embedding,
                 relevance = EXCLUDED.relevance",
            self.table_name
        );

        sqlx::query(&query_str)
            .bind(&entry.id)
            .bind(session)
            .bind(&entry.role)
            .bind(&entry.content)
            .bind(entry.timestamp)
            .bind(metadata)
            .bind(&embedding)
            .bind(entry.relevance)
            .execute(pool)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        tracing::debug!(session, table = %self.table_name, "storing to PostgreSQL");
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
        let pool = self.pool().await?;
        let query_str = format!(
            "SELECT id, session_id, role, content, timestamp, metadata, embedding, relevance
             FROM {} WHERE session_id = $1 ORDER BY timestamp ASC",
            self.table_name
        );

        let rows = sqlx::query(&query_str)
            .bind(session)
            .fetch_all(pool)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let mut entries = Vec::with_capacity(rows.len());
        for row in rows {
            let id: String = row.get("id");
            let session_id: String = row.get("session_id");
            let role: String = row.get("role");
            let content: String = row.get("content");
            let timestamp: chrono::DateTime<chrono::Utc> = row.get("timestamp");
            let metadata: serde_json::Value = row.get("metadata");
            let embedding: Option<Vec<f32>> = row.get("embedding");
            let relevance: f32 = row.get("relevance");

            entries.push(MemoryEntry {
                id,
                session_id,
                role,
                content,
                timestamp,
                metadata,
                embedding,
                relevance,
            });
        }

        Ok(entries)
    }

    async fn clear(&self, session: &str) -> Result<(), MemoryError> {
        let pool = self.pool().await?;
        let query_str = format!(
            "DELETE FROM {} WHERE session_id = $1",
            self.table_name
        );
        sqlx::query(&query_str)
            .bind(session)
            .execute(pool)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        tracing::debug!(session, "clearing PostgreSQL memory");
        Ok(())
    }

    async fn count(&self, session: &str) -> Result<usize, MemoryError> {
        let pool = self.pool().await?;
        let query_str = format!(
            "SELECT COUNT(*) as cnt FROM {} WHERE session_id = $1",
            self.table_name
        );
        let row = sqlx::query(&query_str)
            .bind(session)
            .fetch_one(pool)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        let count: i64 = row.get("cnt");
        Ok(count as usize)
    }

    fn name(&self) -> &str {
        "postgres"
    }
}

#[cfg(all(test, feature = "pg-tests"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_postgres_store_and_retrieve() {
        let url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://localhost/neuralframe_test".to_string());
        let store = PostgresStore::new(&url);

        store.clear("pg_test_s1").await.unwrap();

        store
            .store(
                "pg_test_s1",
                MemoryEntry::new("pg_test_s1", "user", "Hello world"),
            )
            .await
            .unwrap();
        store
            .store(
                "pg_test_s1",
                MemoryEntry::new("pg_test_s1", "assistant", "Hi there"),
            )
            .await
            .unwrap();
        store
            .store(
                "pg_test_s1",
                MemoryEntry::new("pg_test_s1", "user", "Tell me about Rust"),
            )
            .await
            .unwrap();

        assert_eq!(store.count("pg_test_s1").await.unwrap(), 3);

        let results = store.retrieve("pg_test_s1", "Rust", 2).await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].content.contains("Rust"));

        store.clear("pg_test_s1").await.unwrap();
        assert_eq!(store.count("pg_test_s1").await.unwrap(), 0);
    }
}
