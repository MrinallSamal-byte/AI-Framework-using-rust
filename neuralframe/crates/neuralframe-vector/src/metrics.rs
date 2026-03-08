//! # Similarity Metrics
//!
//! Vector similarity and distance functions with SIMD acceleration potential.

/// Compute cosine similarity between two vectors.
///
/// Returns a value between -1.0 and 1.0, where 1.0 means identical direction.
///
/// # Examples
///
/// ```rust
/// use neuralframe_vector::metrics::cosine_similarity;
///
/// let a = vec![1.0, 0.0, 0.0];
/// let b = vec![1.0, 0.0, 0.0];
/// assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
///
/// let c = vec![0.0, 1.0, 0.0];
/// assert!(cosine_similarity(&a, &c).abs() < 1e-6);
/// ```
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    // Process in chunks of 4 for auto-vectorization
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        dot += a[base] * b[base]
            + a[base + 1] * b[base + 1]
            + a[base + 2] * b[base + 2]
            + a[base + 3] * b[base + 3];
        norm_a += a[base] * a[base]
            + a[base + 1] * a[base + 1]
            + a[base + 2] * a[base + 2]
            + a[base + 3] * a[base + 3];
        norm_b += b[base] * b[base]
            + b[base + 1] * b[base + 1]
            + b[base + 2] * b[base + 2]
            + b[base + 3] * b[base + 3];
    }

    let base = chunks * 4;
    for i in 0..remainder {
        dot += a[base + i] * b[base + i];
        norm_a += a[base + i] * a[base + i];
        norm_b += b[base + i] * b[base + i];
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }
    dot / denom
}

/// Compute Euclidean distance between two vectors.
///
/// Returns a non-negative value where 0.0 means identical vectors.
///
/// # Examples
///
/// ```rust
/// use neuralframe_vector::metrics::euclidean_distance;
///
/// let a = vec![0.0, 0.0, 0.0];
/// let b = vec![3.0, 4.0, 0.0];
/// assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
/// ```
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    let mut sum = 0.0f32;

    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        let d0 = a[base] - b[base];
        let d1 = a[base + 1] - b[base + 1];
        let d2 = a[base + 2] - b[base + 2];
        let d3 = a[base + 3] - b[base + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }

    let base = chunks * 4;
    for i in 0..remainder {
        let d = a[base + i] - b[base + i];
        sum += d * d;
    }

    sum.sqrt()
}

/// Compute dot product of two vectors.
///
/// # Examples
///
/// ```rust
/// use neuralframe_vector::metrics::dot_product;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
/// ```
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    let mut sum = 0.0f32;

    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        sum += a[base] * b[base]
            + a[base + 1] * b[base + 1]
            + a[base + 2] * b[base + 2]
            + a[base + 3] * b[base + 3];
    }

    let base = chunks * 4;
    for i in 0..remainder {
        sum += a[base + i] * b[base + i];
    }

    sum
}

/// Normalize a vector to unit length (L2 norm).
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-10 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_zero() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(euclidean_distance(&a, &a).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_345() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(dot_product(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        let norm = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let n = normalize(&v);
        assert_eq!(n, v);
    }

    #[test]
    fn test_high_dimensional() {
        let dim = 1536; // OpenAI embedding dimension
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) / dim as f32).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 + 0.5) / dim as f32).collect();

        let sim = cosine_similarity(&a, &b);
        assert!(sim > 0.9); // Similar vectors should have high similarity
        assert!(sim <= 1.0);
    }
}
