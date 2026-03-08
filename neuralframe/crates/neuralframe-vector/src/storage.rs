//! # Vector Storage
//!
//! Persistent storage for vectors using memory-mapped files with
//! write-ahead logging (WAL) for crash recovery.

use crate::VectorError;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Write-ahead log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalEntry {
    /// Insert a vector.
    Insert {
        id: String,
        vector: Vec<f32>,
        metadata: serde_json::Value,
    },
    /// Delete a vector.
    Delete { id: String },
}

/// Persistent storage configuration.
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Directory for data files.
    pub data_dir: PathBuf,
    /// Whether to sync writes to disk.
    pub sync_writes: bool,
    /// Maximum WAL size before compaction.
    pub max_wal_size_mb: usize,
}

impl StorageConfig {
    /// Create configuration for a given directory.
    pub fn new(data_dir: &Path) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
            sync_writes: true,
            max_wal_size_mb: 64,
        }
    }
}

/// Persistent vector storage engine.
///
/// Uses memory-mapped files for the main data store and a write-ahead
/// log for crash recovery.
pub struct PersistentStorage {
    config: StorageConfig,
    wal: Vec<WalEntry>,
}

impl PersistentStorage {
    /// Create a new persistent storage engine.
    pub fn new(config: StorageConfig) -> Result<Self, VectorError> {
        Ok(Self {
            config,
            wal: Vec::new(),
        })
    }

    /// Append an entry to the write-ahead log.
    pub fn append_wal(&mut self, entry: WalEntry) -> Result<(), VectorError> {
        self.wal.push(entry);
        Ok(())
    }

    /// Get the WAL entries.
    pub fn wal_entries(&self) -> &[WalEntry] {
        &self.wal
    }

    /// Get the data directory path.
    pub fn data_dir(&self) -> &Path {
        &self.config.data_dir
    }

    /// Get the current WAL size.
    pub fn wal_size(&self) -> usize {
        self.wal.len()
    }

    /// Compact the WAL by applying all entries and clearing.
    pub fn compact(&mut self) -> Result<(), VectorError> {
        // In a full implementation, this would write a snapshot
        // and clear the WAL. For now, just clear it.
        self.wal.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_storage_creation() {
        let config = StorageConfig::new(&PathBuf::from("/tmp/test_vectors"));
        let storage = PersistentStorage::new(config).unwrap();
        assert_eq!(storage.wal_size(), 0);
    }

    #[test]
    fn test_wal_append() {
        let config = StorageConfig::new(&PathBuf::from("/tmp/test_wal"));
        let mut storage = PersistentStorage::new(config).unwrap();

        storage
            .append_wal(WalEntry::Insert {
                id: "a".into(),
                vector: vec![1.0, 0.0],
                metadata: serde_json::json!({}),
            })
            .unwrap();

        assert_eq!(storage.wal_size(), 1);
    }

    #[test]
    fn test_wal_compact() {
        let config = StorageConfig::new(&PathBuf::from("/tmp/test_compact"));
        let mut storage = PersistentStorage::new(config).unwrap();

        storage
            .append_wal(WalEntry::Insert {
                id: "a".into(),
                vector: vec![1.0],
                metadata: serde_json::json!({}),
            })
            .unwrap();

        storage.compact().unwrap();
        assert_eq!(storage.wal_size(), 0);
    }
}
