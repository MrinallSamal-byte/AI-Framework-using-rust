//! Benchmarks for the vector store and similarity metrics.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    for dim in [128, 256, 384, 768, 1536].iter() {
        let a: Vec<f32> = (0..*dim).map(|i| (i as f32 / *dim as f32).sin()).collect();
        let b: Vec<f32> = (0..*dim).map(|i| (i as f32 / *dim as f32).cos()).collect();

        group.bench_with_input(BenchmarkId::new("dim", dim), dim, |bench, _| {
            bench.iter(|| {
                neuralframe_vector::metrics::cosine_similarity(black_box(&a), black_box(&b))
            });
        });
    }

    group.finish();
}

fn bench_euclidean_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_distance");

    for dim in [128, 384, 1536].iter() {
        let a: Vec<f32> = (0..*dim).map(|i| (i as f32 / *dim as f32).sin()).collect();
        let b: Vec<f32> = (0..*dim).map(|i| (i as f32 / *dim as f32).cos()).collect();

        group.bench_with_input(BenchmarkId::new("dim", dim), dim, |bench, _| {
            bench.iter(|| {
                neuralframe_vector::metrics::euclidean_distance(black_box(&a), black_box(&b))
            });
        });
    }

    group.finish();
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for dim in [128, 384, 1536].iter() {
        let a: Vec<f32> = (0..*dim).map(|i| (i as f32 / *dim as f32).sin()).collect();
        let b: Vec<f32> = (0..*dim).map(|i| (i as f32 / *dim as f32).cos()).collect();

        group.bench_with_input(BenchmarkId::new("dim", dim), dim, |bench, _| {
            bench.iter(|| neuralframe_vector::metrics::dot_product(black_box(&a), black_box(&b)));
        });
    }

    group.finish();
}

fn bench_vector_store_insert(c: &mut Criterion) {
    let dim = 384;
    c.bench_function("vector_store_insert", |bench| {
        bench.iter_batched(
            || {
                let store = neuralframe_vector::VectorStore::new(
                    dim,
                    neuralframe_vector::DistanceMetric::Cosine,
                );
                let vectors: Vec<Vec<f32>> = (0..100)
                    .map(|i| (0..dim).map(|j| ((i * dim + j) as f32).sin()).collect())
                    .collect();
                (store, vectors)
            },
            |(store, vectors)| {
                for (i, vec) in vectors.into_iter().enumerate() {
                    store
                        .insert(&format!("v{}", i), vec, serde_json::json!({}))
                        .ok();
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_vector_store_search(c: &mut Criterion) {
    let dim = 384;
    let store =
        neuralframe_vector::VectorStore::new(dim, neuralframe_vector::DistanceMetric::Cosine);

    // Pre-populate
    for i in 0..500 {
        let vec: Vec<f32> = (0..dim).map(|j| ((i * dim + j) as f32).sin()).collect();
        store
            .insert(&format!("v{}", i), vec, serde_json::json!({"idx": i}))
            .ok();
    }

    let query: Vec<f32> = (0..dim).map(|j| (j as f32 / dim as f32).sin()).collect();

    c.bench_function("vector_store_search_top10_from_500", |bench| {
        bench.iter(|| store.search(black_box(&query), black_box(10), None).ok());
    });
}

fn bench_hnsw_search(c: &mut Criterion) {
    let dim = 128;
    let mut index = neuralframe_vector::hnsw::HnswIndex::new(dim, 16, 200);

    for i in 0..1000 {
        let vec: Vec<f32> = (0..dim).map(|j| ((i * dim + j) as f32).sin()).collect();
        index.insert(format!("v{}", i), vec);
    }

    let query: Vec<f32> = (0..dim).map(|j| (j as f32).sin()).collect();

    c.bench_function("hnsw_search_top10_from_1000", |bench| {
        bench.iter(|| {
            index.search(
                black_box(&query),
                black_box(10),
                neuralframe_vector::DistanceMetric::Cosine,
            )
        });
    });
}

criterion_group!(
    benches,
    bench_cosine_similarity,
    bench_euclidean_distance,
    bench_dot_product,
    bench_vector_store_insert,
    bench_vector_store_search,
    bench_hnsw_search,
);
criterion_main!(benches);
