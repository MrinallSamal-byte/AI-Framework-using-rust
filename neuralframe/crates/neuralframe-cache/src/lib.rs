//! # NeuralFrame Semantic Cache
//!
//! Intelligent caching for LLM responses using embedding similarity
//! to serve cached responses for semantically equivalent prompts.

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Errors from the cache.
#[derive(Debug)]
pub enum CacheError {
    /// Cache miss.
    Miss,
    /// Similarity below threshold.
    BelowThreshold { similarity: f32, threshold: f32 },
    /// Storage error.
    StorageError(String),
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Miss => write!(f, "cache miss"),
            Self::BelowThreshold { similarity, threshold } => {
                write!(f, "similarity {} below threshold {}", similarity, threshold)
            }
            Self::StorageError(msg) => write!(f, "cache storage error: {}", msg),
        }
    }
}

impl std::error::Error for CacheError {}

/// A cached response entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResponse {
    /// The original prompt text.
    pub prompt: String,
    /// The cached response.
    pub response: String,
    /// The prompt embedding.
    pub embedding: Vec<f32>,
    /// Model used.
    pub model: String,
    /// Number of cache hits.
    pub hits: u64,
    /// When the entry was created.
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Configuration for the semantic cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Similarity threshold (0.0-1.0) for cache hits.
    pub similarity_threshold: f32,
    /// Maximum number of cached entries.
    pub max_entries: usize,
    /// Time-to-live for cached entries.
    pub ttl: Duration,
    /// Whether to use exact match before semantic match.
    pub exact_match_first: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.95,
            max_entries: 10000,
            ttl: Duration::from_secs(3600),
            exact_match_first: true,
        }
    }
}

/// Semantic cache for LLM responses.
///
/// Uses embedding similarity to determine if a prompt is semantically
/// equivalent to a previously cached prompt. If so, returns the cached
/// response without calling the LLM.
pub struct SemanticCache {
    config: CacheConfig,
    entries: Arc<DashMap<String, CacheEntry>>,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    response: CachedResponse,
    created: Instant,
    last_accessed: Instant,
    hits: u64,
}

impl SemanticCache {
    /// Create a new semantic cache.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: Arc::new(DashMap::new()),
        }
    }

    /// Create a cache with default settings.
    pub fn default_cache() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Look up a response by exact prompt match.
    pub fn get_exact(&self, prompt: &str) -> Option<CachedResponse> {
        let key = Self::hash_prompt(prompt);
        self.entries.get_mut(&key).and_then(|mut entry| {
            if entry.created.elapsed() > self.config.ttl {
                None
            } else {
                entry.last_accessed = Instant::now();
                entry.hits += 1;
                Some(entry.response.clone())
            }
        })
    }

    /// Look up a response by semantic similarity.
    pub fn get_semantic(&self, embedding: &[f32]) -> Option<CachedResponse> {
        let mut best_match: Option<(f32, String, CachedResponse)> = None;

        for entry in self.entries.iter() {
            if entry.created.elapsed() > self.config.ttl {
                continue;
            }

            let similarity = cosine_similarity(embedding, &entry.response.embedding);
            if similarity >= self.config.similarity_threshold {
                if best_match
                    .as_ref()
                    .is_none_or(|(s, _, _)| similarity > *s)
                {
                    best_match = Some((
                        similarity,
                        entry.key().clone(),
                        entry.response.clone(),
                    ));
                }
            }
        }

        if let Some((_, key, response)) = best_match {
            if let Some(mut entry) = self.entries.get_mut(&key) {
                entry.last_accessed = Instant::now();
                entry.hits += 1;
            }
            Some(response)
        } else {
            None
        }
    }

    /// Store a prompt-response pair in the cache.
    pub fn store(
        &self,
        prompt: &str,
        response: &str,
        embedding: Vec<f32>,
        model: &str,
    ) {
        // Evict if at capacity
        if self.entries.len() >= self.config.max_entries {
            self.evict_lru();
        }

        let key = Self::hash_prompt(prompt);
        let cached = CachedResponse {
            prompt: prompt.to_string(),
            response: response.to_string(),
            embedding,
            model: model.to_string(),
            hits: 0,
            created_at: chrono::Utc::now(),
        };

        self.entries.insert(key, CacheEntry {
            response: cached,
            created: Instant::now(),
            last_accessed: Instant::now(),
            hits: 0,
        });
    }

    /// Record a cache hit.
    pub fn record_hit(&self, prompt: &str) {
        let key = Self::hash_prompt(prompt);
        if let Some(mut entry) = self.entries.get_mut(&key) {
            entry.hits += 1;
            entry.last_accessed = Instant::now();
        }
    }

    /// Clear expired entries.
    pub fn evict_expired(&self) {
        self.entries.retain(|_, entry| {
            entry.created.elapsed() <= self.config.ttl
        });
    }

    /// Evict least recently used entry.
    fn evict_lru(&self) {
        if let Some(oldest_key) = self
            .entries
            .iter()
            .min_by_key(|entry| entry.last_accessed)
            .map(|entry| entry.key().clone())
        {
            self.entries.remove(&oldest_key);
        }
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let total_entries = self.entries.len();
        let total_hits: u64 = self.entries.iter().map(|e| e.hits).sum();
        CacheStats {
            total_entries,
            total_hits,
            max_entries: self.config.max_entries,
        }
    }

    /// Clear the entire cache.
    pub fn clear(&self) {
        self.entries.clear();
    }

    /// Hash a prompt string for exact match lookups.
    fn hash_prompt(prompt: &str) -> String {
        blake3::hash(prompt.as_bytes()).to_hex().to_string()
    }
}

/// Cache statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total cached entries.
    pub total_entries: usize,
    /// Total cache hits.
    pub total_hits: u64,
    /// Maximum configured entries.
    pub max_entries: usize,
}

/// Simple cosine similarity implementation.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }
    dot / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let cache = SemanticCache::default_cache();
        cache.store(
            "Hello",
            "Hi there!",
            vec![1.0, 0.0, 0.0],
            "gpt-4o",
        );

        let result = cache.get_exact("Hello");
        assert!(result.is_some());
        assert_eq!(result.unwrap().response, "Hi there!");
    }

    #[test]
    fn test_cache_miss() {
        let cache = SemanticCache::default_cache();
        assert!(cache.get_exact("Missing").is_none());
    }

    #[test]
    fn test_semantic_match() {
        let cache = SemanticCache::new(CacheConfig {
            similarity_threshold: 0.99,
            ..Default::default()
        });

        cache.store(
            "Hello",
            "Hi!",
            vec![1.0, 0.0, 0.0],
            "gpt-4o",
        );

        // Exact same embedding should hit
        let result = cache.get_semantic(&[1.0, 0.0, 0.0]);
        assert!(result.is_some());

        // Orthogonal embedding should miss
        let result = cache.get_semantic(&[0.0, 1.0, 0.0]);
        assert!(result.is_none());
    }

    #[test]
    fn test_eviction() {
        let cache = SemanticCache::new(CacheConfig {
            max_entries: 2,
            ..Default::default()
        });

        cache.store("a", "1", vec![1.0], "m");
        cache.store("b", "2", vec![2.0], "m");
        cache.store("c", "3", vec![3.0], "m");

        assert!(cache.entries.len() <= 2);
    }

    #[test]
    fn test_lru_evicts_least_recently_accessed() {
        let cache = SemanticCache::new(CacheConfig {
            max_entries: 2,
            ..Default::default()
        });
        cache.store("A", "1", vec![1.0, 0.0, 0.0], "m");
        cache.store("B", "2", vec![0.0, 1.0, 0.0], "m");
        assert!(cache.get_exact("A").is_some());
        cache.store("C", "3", vec![0.0, 0.0, 1.0], "m");
        assert!(cache.get_exact("A").is_some());
        assert!(cache.get_exact("B").is_none());
    }

    #[test]
    fn test_statistics() {
        let cache = SemanticCache::default_cache();
        cache.store("a", "1", vec![1.0], "m");
        cache.record_hit("a");
        cache.record_hit("a");

        let stats = cache.stats();
        assert_eq!(stats.total_entries, 1);
        assert_eq!(stats.total_hits, 2);
    }

    #[test]
    fn test_clear() {
        let cache = SemanticCache::default_cache();
        cache.store("a", "1", vec![1.0], "m");
        cache.clear();
        assert_eq!(cache.stats().total_entries, 0);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }
}
