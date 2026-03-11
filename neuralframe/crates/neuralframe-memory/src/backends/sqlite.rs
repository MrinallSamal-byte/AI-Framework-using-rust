//! SQLite memory backend for single-node deployments.

use crate::{MemoryEntry, MemoryError, MemoryStore};
use async_trait::async_trait;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};
use std::collections::HashSet;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::OnceCell;

/// SQLite-backed memory store for single-node deployments.
#[derive(Debug, Clone)]
pub struct SqliteStore {
    db_url: String,
    pool: Arc<OnceCell<SqlitePool>>,
}

impl SqliteStore {
    /// Create a new SQLite store.
    pub fn new(db_path: &str) -> Self {
        let db_url = if db_path == ":memory:" {
            "sqlite::memory:".to_string()
        } else {
            format!("sqlite://{}?mode=rwc", db_path)
        };
        Self {
            db_url,
            pool: Arc::new(OnceCell::new()),
        }
    }

    /// Create an in-memory SQLite store.
    pub fn in_memory() -> Self {
        Self::new(":memory:")
    }

    /// Get or initialize the connection pool and create tables.
    async fn pool(&self) -> Result<&SqlitePool, MemoryError> {
        self.pool
            .get_or_try_init(|| async {
                let opts = SqliteConnectOptions::from_str(&self.db_url)
                    .map_err(|e| MemoryError::ConnectionError(e.to_string()))?
                    .create_if_missing(true);

                let pool = SqlitePoolOptions::new()
                    .max_connections(5)
                    .connect_with(opts)
                    .await
                    .map_err(|e| MemoryError::ConnectionError(e.to_string()))?;

                sqlx::query(
                    "CREATE TABLE IF NOT EXISTS memory_entries (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        embedding BLOB,
                        relevance REAL DEFAULT 0
                    )",
                )
                .execute(&pool)
                .await
                .map_err(|e| MemoryError::StorageError(e.to_string()))?;

                sqlx::query("CREATE INDEX IF NOT EXISTS idx_session ON memory_entries(session_id)")
                    .execute(&pool)
                    .await
                    .map_err(|e| MemoryError::StorageError(e.to_string()))?;

                Ok(pool)
            })
            .await
    }
}

#[async_trait]
impl MemoryStore for SqliteStore {
    async fn store(&self, session: &str, entry: MemoryEntry) -> Result<(), MemoryError> {
        let pool = self.pool().await?;
        let metadata_str = serde_json::to_string(&entry.metadata)
            .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
        let timestamp_str = entry.timestamp.to_rfc3339();
        let embedding_blob: Option<Vec<u8>> = entry
            .embedding
            .as_ref()
            .map(|emb| emb.iter().flat_map(|f| f.to_le_bytes()).collect());

        sqlx::query(
            "INSERT OR REPLACE INTO memory_entries (id, session_id, role, content, timestamp, metadata, embedding, relevance)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&entry.id)
        .bind(session)
        .bind(&entry.role)
        .bind(&entry.content)
        .bind(&timestamp_str)
        .bind(&metadata_str)
        .bind(&embedding_blob)
        .bind(entry.relevance)
        .execute(pool)
        .await
        .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        tracing::debug!(session, "stored entry to SQLite");
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
        let rows = sqlx::query(
            "SELECT id, session_id, role, content, timestamp, metadata, embedding, relevance
             FROM memory_entries WHERE session_id = ? ORDER BY timestamp ASC",
        )
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
            let timestamp_str: String = row.get("timestamp");
            let metadata_str: String = row.get("metadata");
            let embedding_blob: Option<Vec<u8>> = row.get("embedding");
            let relevance: f64 = row.get("relevance");

            let timestamp = chrono::DateTime::parse_from_rfc3339(&timestamp_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
            let metadata: serde_json::Value = serde_json::from_str(&metadata_str)
                .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
            let embedding = embedding_blob.map(|blob| {
                blob.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            });

            entries.push(MemoryEntry {
                id,
                session_id,
                role,
                content,
                timestamp,
                metadata,
                embedding,
                relevance: relevance as f32,
            });
        }

        Ok(entries)
    }

    async fn clear(&self, session: &str) -> Result<(), MemoryError> {
        let pool = self.pool().await?;
        sqlx::query("DELETE FROM memory_entries WHERE session_id = ?")
            .bind(session)
            .execute(pool)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        tracing::debug!(session, "cleared SQLite memory");
        Ok(())
    }

    async fn count(&self, session: &str) -> Result<usize, MemoryError> {
        let pool = self.pool().await?;
        let row = sqlx::query("SELECT COUNT(*) as cnt FROM memory_entries WHERE session_id = ?")
            .bind(session)
            .fetch_one(pool)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        let count: i64 = row.get("cnt");
        Ok(count as usize)
    }

    fn name(&self) -> &str {
        "sqlite"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sqlite_store_and_retrieve() {
        let store = SqliteStore::in_memory();

        store
            .store("s1", MemoryEntry::new("s1", "user", "Hello world"))
            .await
            .unwrap();
        store
            .store("s1", MemoryEntry::new("s1", "assistant", "Hi there"))
            .await
            .unwrap();
        store
            .store("s1", MemoryEntry::new("s1", "user", "Tell me about Rust"))
            .await
            .unwrap();

        assert_eq!(store.count("s1").await.unwrap(), 3);

        let results = store.retrieve("s1", "Rust", 2).await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].content.contains("Rust"));
    }

    #[tokio::test]
    async fn test_sqlite_get_all() {
        let store = SqliteStore::in_memory();

        store
            .store("s1", MemoryEntry::new("s1", "user", "First"))
            .await
            .unwrap();
        store
            .store("s1", MemoryEntry::new("s1", "user", "Second"))
            .await
            .unwrap();

        let all = store.get_all("s1").await.unwrap();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn test_sqlite_clear_and_count() {
        let store = SqliteStore::in_memory();

        store
            .store("s1", MemoryEntry::new("s1", "user", "Hello"))
            .await
            .unwrap();
        store
            .store("s1", MemoryEntry::new("s1", "user", "World"))
            .await
            .unwrap();
        assert_eq!(store.count("s1").await.unwrap(), 2);

        store.clear("s1").await.unwrap();
        assert_eq!(store.count("s1").await.unwrap(), 0);
    }
}
