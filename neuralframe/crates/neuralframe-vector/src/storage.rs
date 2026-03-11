//! # Vector Storage
//!
//! Persistent storage for vectors using memory-mapped files with
//! write-ahead logging (WAL) for crash recovery.

use crate::{VectorEntry, VectorError};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
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
        std::fs::create_dir_all(&config.data_dir)
            .map_err(|e| VectorError::StorageError(e.to_string()))?;
        Ok(Self {
            config,
            wal: Vec::new(),
        })
    }

    /// Append an entry to the write-ahead log.
    pub fn append_wal(&mut self, entry: WalEntry) -> Result<(), VectorError> {
        let line =
            serde_json::to_string(&entry).map_err(|e| VectorError::StorageError(e.to_string()))?;
        let path = self.config.data_dir.join("wal.jsonl");
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| VectorError::StorageError(e.to_string()))?;
        writeln!(file, "{}", line).map_err(|e| VectorError::StorageError(e.to_string()))?;
        if self.config.sync_writes {
            file.sync_all()
                .map_err(|e| VectorError::StorageError(e.to_string()))?;
        }
        self.wal.push(entry);
        Ok(())
    }

    /// Load WAL entries from disk.
    pub fn load_wal(&self) -> Result<Vec<WalEntry>, VectorError> {
        let path = self.config.data_dir.join("wal.jsonl");
        if !path.exists() {
            return Ok(Vec::new());
        }
        let file =
            std::fs::File::open(&path).map_err(|e| VectorError::StorageError(e.to_string()))?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| VectorError::StorageError(e.to_string()))?;
            match serde_json::from_str::<WalEntry>(&line) {
                Ok(entry) => entries.push(entry),
                Err(e) => tracing::warn!(error = %e, "skipping malformed WAL line"),
            }
        }
        Ok(entries)
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

    /// Write a fresh snapshot and clear the WAL file.
    pub fn compact(&mut self, entries: &[VectorEntry]) -> Result<(), VectorError> {
        let snap_path = self.config.data_dir.join("snapshot.jsonl");
        let mut file = std::fs::File::create(&snap_path)
            .map_err(|e| VectorError::StorageError(e.to_string()))?;
        for entry in entries {
            let line = serde_json::to_string(entry)
                .map_err(|e| VectorError::StorageError(e.to_string()))?;
            writeln!(file, "{}", line).map_err(|e| VectorError::StorageError(e.to_string()))?;
        }
        let wal_path = self.config.data_dir.join("wal.jsonl");
        std::fs::File::create(&wal_path).map_err(|e| VectorError::StorageError(e.to_string()))?;
        self.wal.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DistanceMetric, VectorStore};

    #[test]
    fn test_storage_creation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let storage = PersistentStorage::new(config);
        assert!(storage.is_ok());
        assert_eq!(storage.expect("storage").wal_size(), 0);
    }

    #[test]
    fn test_wal_append_persists_to_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let mut storage = PersistentStorage::new(config.clone()).expect("storage");
        let result = storage.append_wal(WalEntry::Insert {
            id: "a".into(),
            vector: vec![1.0, 0.0],
            metadata: serde_json::json!({}),
        });
        assert!(result.is_ok());
        let path = config.data_dir.join("wal.jsonl");
        let content = std::fs::read_to_string(path).expect("wal");
        assert!(content.contains("\"id\":\"a\""));
    }

    #[test]
    fn test_compact_clears_wal_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let mut storage = PersistentStorage::new(config.clone()).expect("storage");
        let result = storage.append_wal(WalEntry::Insert {
            id: "a".into(),
            vector: vec![1.0],
            metadata: serde_json::json!({}),
        });
        assert!(result.is_ok());
        let compact = storage.compact(&[VectorEntry {
            id: "a".into(),
            vector: vec![1.0],
            metadata: serde_json::json!({}),
        }]);
        assert!(compact.is_ok());
        let content = std::fs::read_to_string(config.data_dir.join("wal.jsonl")).expect("wal");
        assert!(content.is_empty());
        assert_eq!(storage.wal_size(), 0);
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let store = VectorStore::new(4, DistanceMetric::Cosine);
        for i in 0..10usize {
            assert!(store
                .insert(
                    &format!("v{}", i),
                    vec![i as f32 / 10.0, 0.0, 0.0, 0.0],
                    serde_json::json!({"i": i}),
                )
                .is_ok());
        }
        assert!(store.save_to_disk(&config).is_ok());
        let loaded = VectorStore::load_from_disk(&config, 4, DistanceMetric::Cosine);
        assert!(loaded.is_ok());
        let loaded = loaded.expect("load");
        assert_eq!(loaded.len(), 10);
        let results = loaded.search(&[0.9, 0.0, 0.0, 0.0], 1, None);
        assert!(results.is_ok());
        assert!(!results.expect("search").is_empty());
    }

    #[test]
    fn test_wal_replay_after_snapshot() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let store = VectorStore::new(2, DistanceMetric::Cosine);
        assert!(store
            .insert("a", vec![1.0, 0.0], serde_json::json!({}))
            .is_ok());
        assert!(store.save_to_disk(&config).is_ok());

        let mut storage = PersistentStorage::new(config.clone()).expect("storage");
        assert!(storage
            .append_wal(WalEntry::Insert {
                id: "b".into(),
                vector: vec![0.0, 1.0],
                metadata: serde_json::json!({}),
            })
            .is_ok());

        let loaded = VectorStore::load_from_disk(&config, 2, DistanceMetric::Cosine).expect("load");
        assert_eq!(loaded.len(), 2);
        assert!(loaded.get("a").is_some());
        assert!(loaded.get("b").is_some());
    }
}
