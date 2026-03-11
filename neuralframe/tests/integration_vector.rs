use neuralframe_vector::{DistanceMetric, Filter, VectorStore};
use std::sync::Arc;

#[test]
fn test_insert_search_delete_cycle() {
    let store = VectorStore::new(3, DistanceMetric::Cosine);
    assert!(store
        .insert("a", vec![1.0, 0.0, 0.0], serde_json::json!({}))
        .is_ok());
    assert!(store
        .insert("b", vec![0.0, 1.0, 0.0], serde_json::json!({}))
        .is_ok());
    assert_eq!(store.len(), 2);

    let results = store.search(&[1.0, 0.0, 0.0], 1, None).unwrap_or_default();
    assert_eq!(results.first().map(|r| r.id.as_str()), Some("a"));

    assert!(store.delete("a").is_ok());
    assert_eq!(store.len(), 1);
    assert!(store.get("a").is_none());
}

#[test]
fn test_filter_eq_narrows_results() {
    let store = VectorStore::new(3, DistanceMetric::Cosine);
    assert!(store
        .insert(
            "r",
            vec![1.0, 0.0, 0.0],
            serde_json::json!({"color": "red"})
        )
        .is_ok());
    assert!(store
        .insert(
            "b",
            vec![0.9, 0.1, 0.0],
            serde_json::json!({"color": "blue"})
        )
        .is_ok());
    let f = Filter::Eq("color".into(), serde_json::json!("red"));
    let results = store
        .search(&[1.0, 0.0, 0.0], 10, Some(&f))
        .unwrap_or_default();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "r");
}

#[test]
fn test_filter_and_compound() {
    let store = VectorStore::new(2, DistanceMetric::Cosine);
    assert!(store
        .insert("match", vec![1.0, 0.0], serde_json::json!({"a": 1, "b": 2}))
        .is_ok());
    assert!(store
        .insert(
            "nomatch",
            vec![0.9, 0.1],
            serde_json::json!({"a": 1, "b": 99})
        )
        .is_ok());
    let f = Filter::And(vec![
        Filter::Eq("a".into(), serde_json::json!(1)),
        Filter::Eq("b".into(), serde_json::json!(2)),
    ]);
    let results = store.search(&[1.0, 0.0], 10, Some(&f)).unwrap_or_default();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "match");
}

#[test]
fn test_persistence_roundtrip() {
    let dir = tempfile::tempdir().unwrap_or_else(|_| panic!("tempdir failed"));
    let config = neuralframe_vector::storage::StorageConfig::new(dir.path());

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

    let loaded = VectorStore::load_from_disk(&config, 4, DistanceMetric::Cosine)
        .unwrap_or_else(|_| panic!("load failed"));
    assert_eq!(loaded.len(), 10);
    let results = loaded
        .search(&[0.9, 0.0, 0.0, 0.0], 1, None)
        .unwrap_or_default();
    assert!(!results.is_empty());
}

#[test]
fn test_hnsw_recall_rate_above_80_percent() {
    use neuralframe_vector::metrics::cosine_similarity;
    let dim = 16usize;
    let n = 200usize;
    let store = VectorStore::new(dim, DistanceMetric::Cosine);
    let mut vecs: Vec<(String, Vec<f32>)> = Vec::new();

    for i in 0..n {
        let mut v = vec![0.0f32; dim];
        v[i % dim] = 1.0;
        v[(i + 1) % dim] = 0.5;
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let v: Vec<f32> = v.into_iter().map(|x| x / norm).collect();
        assert!(store
            .insert(&format!("v{}", i), v.clone(), serde_json::json!({}))
            .is_ok());
        vecs.push((format!("v{}", i), v));
    }

    let k = 5usize;
    let queries = &vecs[..20];
    let mut total_recall = 0.0f32;

    for (_, query) in queries {
        let hnsw_results = store.search(query, k, None).unwrap_or_default();
        let hnsw_ids: std::collections::HashSet<&str> =
            hnsw_results.iter().map(|r| r.id.as_str()).collect();

        let mut brute: Vec<(String, f32)> = vecs
            .iter()
            .map(|(id, v)| (id.clone(), cosine_similarity(query, v)))
            .collect();
        brute.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let brute_ids: std::collections::HashSet<&str> =
            brute[..k].iter().map(|(id, _)| id.as_str()).collect();

        let overlap = hnsw_ids.intersection(&brute_ids).count();
        total_recall += overlap as f32 / k as f32;
    }

    let avg_recall = total_recall / queries.len() as f32;
    assert!(avg_recall >= 0.8, "recall {} below 0.8", avg_recall);
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
                let val = (i * 10 + j) as f32 / 500.0;
                let _ = store.insert(&id, vec![val, val, val, val], serde_json::json!({}));
            }
        }));
    }
    for h in handles {
        let _ = h.await;
    }
    assert_eq!(store.len(), 500);
}

#[test]
fn test_hnsw_m1_no_panic() {
    let mut index = neuralframe_vector::hnsw::HnswIndex::new(3, 1, 50);
    for i in 0..10 {
        index.insert(format!("v{}", i), vec![i as f32, 0.0, 0.0]);
    }
    let results = index.search(&[1.0, 0.0, 0.0], 5, DistanceMetric::Cosine);
    assert!(!results.is_empty());
}
