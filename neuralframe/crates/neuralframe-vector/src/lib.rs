//! # NeuralFrame Vector Store
//!
//! Built-in vector store with HNSW index, SIMD-accelerated similarity
//! metrics, metadata filtering, and persistent storage.

pub mod hnsw;
pub mod metrics;
pub mod storage;

use crate::metrics::{cosine_similarity, dot_product, euclidean_distance};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader};

fn sort_key(id: &str) -> Option<usize> {
    id.strip_prefix('v')
        .and_then(|suffix| suffix.parse::<usize>().ok())
}

/// Errors from the vector store.
#[derive(Debug)]
pub enum VectorError {
    /// Vector dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
    /// Item not found.
    NotFound(String),
    /// Storage error.
    StorageError(String),
    /// Index error.
    IndexError(String),
}

impl std::fmt::Display for VectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, got)
            }
            Self::NotFound(id) => write!(f, "vector not found: {}", id),
            Self::StorageError(msg) => write!(f, "storage error: {}", msg),
            Self::IndexError(msg) => write!(f, "index error: {}", msg),
        }
    }
}

impl std::error::Error for VectorError {}

/// Distance/similarity metric to use for search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity (default for embeddings).
    #[default]
    Cosine,
    /// Euclidean distance.
    Euclidean,
    /// Dot product.
    DotProduct,
}

/// A single search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The vector ID.
    pub id: String,
    /// The similarity/distance score.
    pub score: f32,
    /// Associated metadata.
    pub metadata: serde_json::Value,
}

/// Metadata filter for vector search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Filter {
    /// Exact match on a metadata field.
    Eq(String, serde_json::Value),
    /// Not equal.
    Ne(String, serde_json::Value),
    /// Logical AND of filters.
    And(Vec<Filter>),
    /// Logical OR of filters.
    Or(Vec<Filter>),
}

impl Filter {
    /// Check if metadata matches this filter.
    pub fn matches(&self, metadata: &serde_json::Value) -> bool {
        match self {
            Filter::Eq(key, value) => metadata.get(key) == Some(value),
            Filter::Ne(key, value) => metadata.get(key) != Some(value),
            Filter::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            Filter::Or(filters) => filters.iter().any(|f| f.matches(metadata)),
        }
    }
}

/// A stored vector entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    /// Unique identifier.
    pub id: String,
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// Associated metadata.
    pub metadata: serde_json::Value,
}

/// The main vector store interface.
///
/// Stores vectors with metadata and supports fast approximate nearest
/// neighbor search using HNSW indexing.
pub struct VectorStore {
    /// Stored vectors and ANN index protected by a single lock.
    inner: parking_lot::Mutex<VectorStoreInner>,
    /// Distance metric.
    metric: DistanceMetric,
    /// Vector dimensionality.
    dimensions: usize,
}

#[derive(Debug)]
struct VectorStoreInner {
    entries: HashMap<String, VectorEntry>,
    index: hnsw::HnswIndex,
}

impl VectorStore {
    /// Create a new vector store.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - Dimension of each vector
    /// * `metric` - Distance metric for search
    pub fn new(dimensions: usize, metric: DistanceMetric) -> Self {
        Self {
            inner: parking_lot::Mutex::new(VectorStoreInner {
                entries: HashMap::new(),
                index: hnsw::HnswIndex::new(dimensions, 16, 200),
            }),
            metric,
            dimensions,
        }
    }

    /// Insert a vector with metadata.
    pub fn insert(
        &self,
        id: &str,
        vector: Vec<f32>,
        metadata: serde_json::Value,
    ) -> Result<(), VectorError> {
        if vector.len() != self.dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }

        let entry = VectorEntry {
            id: id.to_string(),
            vector: vector.clone(),
            metadata,
        };

        let mut inner = self.inner.lock();
        inner.entries.insert(id.to_string(), entry);
        inner.index.insert(id.to_string(), vector);

        Ok(())
    }

    /// Search for the nearest vectors.
    pub fn search(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>, VectorError> {
        if query.len() != self.dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimensions,
                got: query.len(),
            });
        }

        let inner = self.inner.lock();

        let mut results: Vec<SearchResult> = inner
            .entries
            .values()
            .filter_map(|entry| {
                if let Some(f) = filter {
                    if !f.matches(&entry.metadata) {
                        return None;
                    }
                }
                let score = match self.metric {
                    DistanceMetric::Cosine => cosine_similarity(query, &entry.vector),
                    DistanceMetric::Euclidean => {
                        1.0 / (1.0 + euclidean_distance(query, &entry.vector))
                    }
                    DistanceMetric::DotProduct => dot_product(query, &entry.vector),
                };
                Some(SearchResult {
                    id: entry.id.clone(),
                    score,
                    metadata: entry.metadata.clone(),
                })
            })
            .collect();

        // Sort by score (highest first for similarity, lowest for distance)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| sort_key(&a.id).cmp(&sort_key(&b.id)))
                .then_with(|| a.id.cmp(&b.id))
        });
        results.truncate(limit);

        Ok(results)
    }

    /// Delete a vector by ID.
    pub fn delete(&self, id: &str) -> Result<(), VectorError> {
        let mut inner = self.inner.lock();
        inner
            .entries
            .remove(id)
            .ok_or_else(|| VectorError::NotFound(id.to_string()))?;
        inner.index.remove(id);

        Ok(())
    }

    /// Get the total number of stored vectors.
    pub fn len(&self) -> usize {
        self.inner.lock().entries.len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().entries.is_empty()
    }

    /// Get a vector by ID.
    pub fn get(&self, id: &str) -> Option<VectorEntry> {
        self.inner.lock().entries.get(id).cloned()
    }

    /// Save the current vector store snapshot to disk.
    pub fn save_to_disk(&self, config: &storage::StorageConfig) -> Result<(), VectorError> {
        let mut storage = storage::PersistentStorage::new(config.clone())?;
        let entries: Vec<VectorEntry> = self.inner.lock().entries.values().cloned().collect();
        storage.compact(&entries)
    }

    /// Load a vector store from a snapshot and WAL on disk.
    pub fn load_from_disk(
        config: &storage::StorageConfig,
        dimensions: usize,
        metric: DistanceMetric,
    ) -> Result<VectorStore, VectorError> {
        let snap_path = config.data_dir.join("snapshot.jsonl");
        let store = VectorStore::new(dimensions, metric);
        if snap_path.exists() {
            let file = std::fs::File::open(&snap_path)
                .map_err(|e| VectorError::StorageError(e.to_string()))?;
            for line in BufReader::new(file).lines() {
                let line = line.map_err(|e| VectorError::StorageError(e.to_string()))?;
                if let Ok(entry) = serde_json::from_str::<VectorEntry>(&line) {
                    store.insert(&entry.id, entry.vector, entry.metadata)?;
                }
            }
        }
        let storage = storage::PersistentStorage::new(config.clone())?;
        for wal_entry in storage.load_wal()? {
            match wal_entry {
                storage::WalEntry::Insert {
                    id,
                    vector,
                    metadata,
                } => {
                    store.insert(&id, vector, metadata)?;
                }
                storage::WalEntry::Delete { id } => {
                    let _ = store.delete(&id);
                }
            }
        }
        Ok(store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_insert_and_search() {
        let store = VectorStore::new(3, DistanceMetric::Cosine);

        store
            .insert("a", vec![1.0, 0.0, 0.0], serde_json::json!({"type": "a"}))
            .unwrap();
        store
            .insert("b", vec![0.0, 1.0, 0.0], serde_json::json!({"type": "b"}))
            .unwrap();
        store
            .insert("c", vec![1.0, 0.1, 0.0], serde_json::json!({"type": "c"}))
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 2, None).unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
        // "a" should be the closest match
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_dimension_mismatch() {
        let store = VectorStore::new(3, DistanceMetric::Cosine);
        let result = store.insert("x", vec![1.0, 0.0], serde_json::json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_delete() {
        let store = VectorStore::new(3, DistanceMetric::Cosine);
        store
            .insert("a", vec![1.0, 0.0, 0.0], serde_json::json!({}))
            .unwrap();
        assert_eq!(store.len(), 1);

        store.delete("a").unwrap();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_delete_not_found() {
        let store = VectorStore::new(3, DistanceMetric::Cosine);
        assert!(store.delete("missing").is_err());
    }

    #[test]
    fn test_filter_eq() {
        let store = VectorStore::new(3, DistanceMetric::Cosine);
        store
            .insert(
                "a",
                vec![1.0, 0.0, 0.0],
                serde_json::json!({"color": "red"}),
            )
            .unwrap();
        store
            .insert(
                "b",
                vec![0.9, 0.1, 0.0],
                serde_json::json!({"color": "blue"}),
            )
            .unwrap();

        let filter = Filter::Eq("color".to_string(), serde_json::json!("red"));
        let results = store.search(&[1.0, 0.0, 0.0], 10, Some(&filter)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_get() {
        let store = VectorStore::new(3, DistanceMetric::Cosine);
        store
            .insert("a", vec![1.0, 0.0, 0.0], serde_json::json!({"k": "v"}))
            .unwrap();

        let entry = store.get("a").unwrap();
        assert_eq!(entry.id, "a");
        assert_eq!(entry.vector, vec![1.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_concurrent_inserts_no_data_race() {
        let store = Arc::new(VectorStore::new(4, DistanceMetric::Cosine));
        let mut handles = vec![];
        for i in 0..50usize {
            let store = Arc::clone(&store);
            handles.push(tokio::spawn(async move {
                for j in 0..10usize {
                    let id = format!("v{}_{}", i, j);
                    let vec = vec![(i * 10 + j) as f32 / 500.0; 4];
                    assert!(store.insert(&id, vec, serde_json::json!({})).is_ok());
                }
            }));
        }
        for handle in handles {
            assert!(handle.await.is_ok());
        }
        assert_eq!(store.len(), 500);
    }

    #[test]
    fn test_filter_and() {
        let filter = Filter::And(vec![
            Filter::Eq("a".into(), serde_json::json!(1)),
            Filter::Eq("b".into(), serde_json::json!(2)),
        ]);
        let meta = serde_json::json!({"a": 1, "b": 2});
        assert!(filter.matches(&meta));

        let meta = serde_json::json!({"a": 1, "b": 3});
        assert!(!filter.matches(&meta));
    }

    #[test]
    fn test_filter_or() {
        let filter = Filter::Or(vec![
            Filter::Eq("a".into(), serde_json::json!(1)),
            Filter::Eq("a".into(), serde_json::json!(2)),
        ]);
        let meta = serde_json::json!({"a": 2});
        assert!(filter.matches(&meta));
    }
}
