use neuralframe_cache::{CacheConfig, SemanticCache};
use std::time::Duration;

fn vec3(x: f32, y: f32, z: f32) -> Vec<f32> {
    vec![x, y, z]
}

#[test]
fn test_exact_match_hit() {
    let cache = SemanticCache::default_cache();
    cache.store(
        "hello world",
        "response text",
        vec3(1.0, 0.0, 0.0),
        "gpt-4o",
    );
    let hit = cache.get_exact("hello world");
    assert!(hit.is_some());
    assert_eq!(
        hit.unwrap_or_else(|| panic!("missing hit")).response,
        "response text"
    );
}

#[test]
fn test_exact_match_miss() {
    let cache = SemanticCache::default_cache();
    assert!(cache.get_exact("not stored").is_none());
}

#[test]
fn test_semantic_match_above_threshold() {
    let cache = SemanticCache::new(CacheConfig {
        similarity_threshold: 0.99,
        ..Default::default()
    });
    cache.store("prompt", "reply", vec3(1.0, 0.0, 0.0), "gpt-4o");
    let result = cache.get_semantic(&[1.0, 0.0, 0.0]);
    assert!(result.is_some());
}

#[test]
fn test_semantic_miss_below_threshold() {
    let cache = SemanticCache::new(CacheConfig {
        similarity_threshold: 0.99,
        ..Default::default()
    });
    cache.store("prompt", "reply", vec3(1.0, 0.0, 0.0), "gpt-4o");
    let result = cache.get_semantic(&[0.0, 1.0, 0.0]);
    assert!(result.is_none());
}

#[test]
fn test_ttl_expiry_returns_none() {
    let cache = SemanticCache::new(CacheConfig {
        ttl: Duration::from_millis(1),
        ..Default::default()
    });
    cache.store("prompt", "reply", vec3(1.0, 0.0, 0.0), "gpt-4o");
    std::thread::sleep(Duration::from_millis(10));
    assert!(cache.get_exact("prompt").is_none());
    assert!(cache.get_semantic(&[1.0, 0.0, 0.0]).is_none());
}

#[test]
fn test_lru_evicts_least_recently_accessed() {
    let cache = SemanticCache::new(CacheConfig {
        max_entries: 2,
        ..Default::default()
    });
    cache.store("a", "1", vec3(1.0, 0.0, 0.0), "gpt-4o");
    cache.store("b", "2", vec3(0.0, 1.0, 0.0), "gpt-4o");
    let _ = cache.get_exact("a");
    cache.store("c", "3", vec3(0.0, 0.0, 1.0), "gpt-4o");
    assert!(cache.get_exact("a").is_some());
    assert!(cache.get_exact("b").is_none());
}

#[test]
fn test_stats_track_hits() {
    let cache = SemanticCache::default_cache();
    cache.store("p", "r", vec3(1.0, 0.0, 0.0), "gpt-4o");
    cache.record_hit("p");
    cache.record_hit("p");
    let stats = cache.stats();
    assert_eq!(stats.total_entries, 1);
    assert_eq!(stats.total_hits, 2);
}

#[test]
fn test_evict_expired_removes_stale() {
    let cache = SemanticCache::new(CacheConfig {
        ttl: Duration::from_millis(1),
        max_entries: 100,
        ..Default::default()
    });
    cache.store("old", "response", vec3(1.0, 0.0, 0.0), "gpt-4o");
    std::thread::sleep(Duration::from_millis(10));
    cache.evict_expired();
    assert_eq!(cache.stats().total_entries, 0);
}

#[test]
fn test_clear_empties_cache() {
    let cache = SemanticCache::default_cache();
    cache.store("a", "1", vec3(1.0, 0.0, 0.0), "gpt-4o");
    cache.store("b", "2", vec3(0.0, 1.0, 0.0), "gpt-4o");
    cache.clear();
    assert_eq!(cache.stats().total_entries, 0);
}
