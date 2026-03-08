//! # NeuralFrame Memory Engine
//!
//! Multi-tier memory system for AI applications with conversation,
//! summary, vector, and entity memory types.

pub mod backends;
pub mod context;
pub mod types;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Errors from the memory engine.
#[derive(Debug)]
pub enum MemoryError {
    /// Storage backend error.
    StorageError(String),
    /// Session not found.
    SessionNotFound(String),
    /// Serialization error.
    SerializationError(String),
    /// Connection error.
    ConnectionError(String),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StorageError(msg) => write!(f, "storage error: {}", msg),
            Self::SessionNotFound(id) => write!(f, "session not found: {}", id),
            Self::SerializationError(msg) => write!(f, "serialization error: {}", msg),
            Self::ConnectionError(msg) => write!(f, "connection error: {}", msg),
        }
    }
}

impl std::error::Error for MemoryError {}

/// A single memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier.
    pub id: String,
    /// The session this entry belongs to.
    pub session_id: String,
    /// The role (user/assistant/system).
    pub role: String,
    /// The text content.
    pub content: String,
    /// Timestamp.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Optional metadata.
    pub metadata: serde_json::Value,
    /// Optional embedding vector.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// Relevance score (set during retrieval).
    #[serde(default)]
    pub relevance: f32,
}

impl MemoryEntry {
    /// Create a new memory entry.
    pub fn new(session_id: &str, role: &str, content: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            role: role.to_string(),
            content: content.to_string(),
            timestamp: chrono::Utc::now(),
            metadata: serde_json::json!({}),
            embedding: None,
            relevance: 0.0,
        }
    }

    /// Set metadata on this entry.
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set the embedding vector.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// Trait for memory storage backends.
///
/// Implementations provide store, retrieve, and clear operations
/// for session-based memory.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Store a memory entry.
    async fn store(&self, session: &str, entry: MemoryEntry) -> Result<(), MemoryError>;

    /// Retrieve memory entries matching a query.
    async fn retrieve(
        &self,
        session: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, MemoryError>;

    /// Get all entries for a session.
    async fn get_all(&self, session: &str) -> Result<Vec<MemoryEntry>, MemoryError>;

    /// Clear all memory for a session.
    async fn clear(&self, session: &str) -> Result<(), MemoryError>;

    /// Get the number of entries for a session.
    async fn count(&self, session: &str) -> Result<usize, MemoryError>;

    /// Get the backend name.
    fn name(&self) -> &str;
}

/// Prelude for convenience.
pub mod prelude {
    pub use crate::backends::inmemory::InMemoryStore;
    pub use crate::context::ContextBuilder;
    pub use crate::types::*;
    pub use crate::{MemoryEntry, MemoryError, MemoryStore};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_entry_creation() {
        let entry = MemoryEntry::new("session1", "user", "Hello!");
        assert_eq!(entry.session_id, "session1");
        assert_eq!(entry.role, "user");
        assert_eq!(entry.content, "Hello!");
        assert!(!entry.id.is_empty());
    }

    #[test]
    fn test_memory_entry_with_metadata() {
        let entry = MemoryEntry::new("s1", "user", "Hi")
            .with_metadata(serde_json::json!({"intent": "greeting"}));
        assert_eq!(entry.metadata["intent"], "greeting");
    }

    #[test]
    fn test_memory_entry_with_embedding() {
        let entry = MemoryEntry::new("s1", "user", "Hi")
            .with_embedding(vec![1.0, 0.0, 0.0]);
        assert_eq!(entry.embedding.unwrap().len(), 3);
    }
}
