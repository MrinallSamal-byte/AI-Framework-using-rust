//! # NeuralFrame Vector Store
//!
//! Built-in vector store with HNSW index, SIMD-accelerated similarity
//! metrics, metadata filtering, and persistent storage.

pub mod hnsw;
pub mod metrics;
pub mod storage;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity (default for embeddings).
    Cosine,
    /// Euclidean distance.
    Euclidean,
    /// Dot product.
    DotProduct,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        Self::Cosine
    }
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
    /// Stored vectors by ID.
    entries: parking_lot::RwLock<HashMap<String, VectorEntry>>,
    /// HNSW index for ANN search.
    index: parking_lot::RwLock<hnsw::HnswIndex>,
    /// Distance metric.
    metric: DistanceMetric,
    /// Vector dimensionality.
    dimensions: usize,
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
            entries: parking_lot::RwLock::new(HashMap::new()),
            index: parking_lot::RwLock::new(hnsw::HnswIndex::new(dimensions, 16, 200)),
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

        let mut entries = self.entries.write();
        let mut index = self.index.write();

        entries.insert(id.to_string(), entry);
        index.insert(id.to_string(), vector);

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

        let entries = self.entries.read();
        let index = self.index.read();

        // Get candidates from HNSW index
        let candidates = index.search(query, limit * 2, self.metric);

        let mut results: Vec<SearchResult> = candidates
            .into_iter()
            .filter_map(|(id, score)| {
                let entry = entries.get(&id)?;
                // Apply metadata filter
                if let Some(f) = filter {
                    if !f.matches(&entry.metadata) {
                        return None;
                    }
                }
                Some(SearchResult {
                    id: id.clone(),
                    score,
                    metadata: entry.metadata.clone(),
                })
            })
            .take(limit)
            .collect();

        // Sort by score (highest first for similarity, lowest for distance)
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    /// Delete a vector by ID.
    pub fn delete(&self, id: &str) -> Result<(), VectorError> {
        let mut entries = self.entries.write();
        let mut index = self.index.write();

        entries
            .remove(id)
            .ok_or_else(|| VectorError::NotFound(id.to_string()))?;
        index.remove(id);

        Ok(())
    }

    /// Get the total number of stored vectors.
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Get a vector by ID.
    pub fn get(&self, id: &str) -> Option<VectorEntry> {
        self.entries.read().get(id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            .insert("a", vec![1.0, 0.0, 0.0], serde_json::json!({"color": "red"}))
            .unwrap();
        store
            .insert("b", vec![0.9, 0.1, 0.0], serde_json::json!({"color": "blue"}))
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
