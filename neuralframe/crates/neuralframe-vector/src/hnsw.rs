//! # HNSW Index
//!
//! Hierarchical Navigable Small World graph for approximate nearest neighbor search.
//! Implements the core insertion and search algorithms.

use crate::metrics;
use crate::DistanceMetric;
use rand::Rng;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

/// A neighbor reference with distance.
#[derive(Debug, Clone)]
struct Neighbor {
    id: String,
    distance: f32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed for min-heap behavior
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}

/// A node in the HNSW graph.
#[derive(Debug, Clone)]
struct HnswNode {
    vector: Vec<f32>,
    /// Neighbors at each level.
    neighbors: Vec<Vec<String>>,
}

/// HNSW index for approximate nearest neighbor search.
///
/// Parameters:
/// - `m`: Maximum number of connections per node per level
/// - `ef_construction`: Size of dynamic candidate list during construction
pub struct HnswIndex {
    nodes: HashMap<String, HnswNode>,
    entry_point: Option<String>,
    max_level: usize,
    m: usize,
    ef_construction: usize,
    dimensions: usize,
    ml: f64,
}

impl HnswIndex {
    /// Create a new HNSW index.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - Vector dimensionality
    /// * `m` - Max connections per node per level (default: 16)
    /// * `ef_construction` - Construction quality factor (default: 200)
    pub fn new(dimensions: usize, m: usize, ef_construction: usize) -> Self {
        Self {
            nodes: HashMap::new(),
            entry_point: None,
            max_level: 0,
            m,
            ef_construction,
            dimensions,
            ml: 1.0 / (m as f64).ln(),
        }
    }

    /// Calculate a random level for a new node.
    fn random_level(&self) -> usize {
        if self.m <= 1 {
            return 0;
        }
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen::<f64>().max(f64::EPSILON);
        (-r.ln() * self.ml).floor() as usize
    }

    /// Insert a vector into the index.
    pub fn insert(&mut self, id: String, vector: Vec<f32>) {
        if vector.len() != self.dimensions {
            return;
        }
        let level = self.random_level();

        let node = HnswNode {
            vector,
            neighbors: vec![Vec::new(); level + 1],
        };

        if self.entry_point.is_none() {
            self.entry_point = Some(id.clone());
            self.max_level = level;
            self.nodes.insert(id, node);
            return;
        }

        let entry_id = self.entry_point.clone().unwrap();
        let mut current = entry_id;

        // Traverse from top level down to level + 1
        for lev in (level + 1..=self.max_level).rev() {
            current = self.find_closest_at_level(&node.vector, &current, lev);
        }

        // Insert at levels from min(level, max_level) down to 0
        let insert_level = level.min(self.max_level);
        for lev in (0..=insert_level).rev() {
            let neighbors = self.search_level(&node.vector, &current, self.ef_construction, lev);

            // Select M best neighbors
            let selected: Vec<String> = neighbors
                .into_iter()
                .take(self.m)
                .map(|(nid, _)| nid)
                .collect();

            // Add bidirectional connections
            for neighbor_id in &selected {
                if let Some(neighbor_node) = self.nodes.get_mut(neighbor_id) {
                    if lev < neighbor_node.neighbors.len() {
                        neighbor_node.neighbors[lev].push(id.clone());
                        // Prune if over limit
                        if neighbor_node.neighbors[lev].len() > self.m * 2 {
                            neighbor_node.neighbors[lev].truncate(self.m * 2);
                        }
                    }
                }
            }

            if let Some(node) = self.nodes.get_mut(&id) {
                if lev < node.neighbors.len() {
                    node.neighbors[lev] = selected;
                }
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(id.clone());
        }

        self.nodes.insert(id, node);
    }

    /// Remove a vector from the index.
    pub fn remove(&mut self, id: &str) {
        if let Some(removed) = self.nodes.remove(id) {
            // Remove all references to this node
            for level_neighbors in &removed.neighbors {
                for neighbor_id in level_neighbors {
                    if let Some(neighbor) = self.nodes.get_mut(neighbor_id) {
                        for level in &mut neighbor.neighbors {
                            level.retain(|nid| nid != id);
                        }
                    }
                }
            }

            // Update entry point if needed
            if self.entry_point.as_deref() == Some(id) {
                self.entry_point = self.nodes.keys().next().cloned();
            }
        }
    }

    /// Search for nearest neighbors.
    pub fn search(&self, query: &[f32], limit: usize, metric: DistanceMetric) -> Vec<(String, f32)> {
        if self.nodes.is_empty() || query.len() != self.dimensions {
            return Vec::new();
        }

        let entry_id = match &self.entry_point {
            Some(ep) => ep.clone(),
            None => return Vec::new(),
        };

        let mut current = entry_id;

        // Traverse from top to level 1
        for lev in (1..=self.max_level).rev() {
            current = self.find_closest_at_level(query, &current, lev);
        }

        // Search at level 0
        let candidates = self.search_level(query, &current, limit.max(self.ef_construction), 0);

        // Score and sort
        let mut results: Vec<(String, f32)> = candidates
            .into_iter()
            .map(|(id, _)| {
                let node = &self.nodes[&id];
                let score = match metric {
                    DistanceMetric::Cosine => metrics::cosine_similarity(query, &node.vector),
                    DistanceMetric::Euclidean => {
                        1.0 / (1.0 + metrics::euclidean_distance(query, &node.vector))
                    }
                    DistanceMetric::DotProduct => metrics::dot_product(query, &node.vector),
                };
                (id, score)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results.truncate(limit);
        results
    }

    /// Find the closest node at a specific level using greedy search.
    fn find_closest_at_level(&self, query: &[f32], start: &str, level: usize) -> String {
        let mut current = start.to_string();
        let mut current_dist = self.distance(query, start);

        loop {
            let mut changed = false;
            let neighbors = self
                .nodes
                .get(&current)
                .map(|n| {
                    if level < n.neighbors.len() {
                        n.neighbors[level].clone()
                    } else {
                        Vec::new()
                    }
                })
                .unwrap_or_default();

            for neighbor_id in neighbors {
                let dist = self.distance(query, &neighbor_id);
                if dist < current_dist {
                    current = neighbor_id;
                    current_dist = dist;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        current
    }

    /// Beam search at a specific level.
    fn search_level(
        &self,
        query: &[f32],
        start: &str,
        ef: usize,
        level: usize,
    ) -> Vec<(String, f32)> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = Vec::new();

        let start_dist = self.distance(query, start);
        visited.insert(start.to_string());
        candidates.push(Neighbor {
            id: start.to_string(),
            distance: start_dist,
        });
        results.push((start.to_string(), start_dist));

        while let Some(closest) = candidates.pop() {
            let neighbors = self
                .nodes
                .get(&closest.id)
                .map(|n| {
                    if level < n.neighbors.len() {
                        n.neighbors[level].clone()
                    } else {
                        Vec::new()
                    }
                })
                .unwrap_or_default();

            for neighbor_id in neighbors {
                if visited.contains(&neighbor_id) {
                    continue;
                }
                visited.insert(neighbor_id.clone());

                let dist = self.distance(query, &neighbor_id);
                candidates.push(Neighbor {
                    id: neighbor_id.clone(),
                    distance: dist,
                });
                results.push((neighbor_id, dist));

                if results.len() > ef {
                    results.sort_by(|a, b| {
                        a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
                    });
                    results.truncate(ef);
                }
            }
        }

        results
    }

    /// Calculate distance between a query and a stored node.
    fn distance(&self, query: &[f32], node_id: &str) -> f32 {
        match self.nodes.get(node_id) {
            Some(node) => {
                let sim = metrics::cosine_similarity(query, &node.vector);
                1.0 - sim // Convert similarity to distance
            }
            None => f32::MAX,
        }
    }

    /// Get the number of nodes in the index.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("nodes", &self.nodes.len())
            .field("max_level", &self.max_level)
            .field("m", &self.m)
            .field("ef_construction", &self.ef_construction)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_index() {
        let index = HnswIndex::new(3, 16, 200);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        let results = index.search(&[1.0, 0.0, 0.0], 5, DistanceMetric::Cosine);
        assert!(results.is_empty());
    }

    #[test]
    fn test_single_insert() {
        let mut index = HnswIndex::new(3, 16, 200);
        index.insert("a".into(), vec![1.0, 0.0, 0.0]);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_search_basic() {
        let mut index = HnswIndex::new(3, 16, 200);
        index.insert("a".into(), vec![1.0, 0.0, 0.0]);
        index.insert("b".into(), vec![0.0, 1.0, 0.0]);
        index.insert("c".into(), vec![0.0, 0.0, 1.0]);

        let results = index.search(&[1.0, 0.0, 0.0], 1, DistanceMetric::Cosine);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "a");
    }

    #[test]
    fn test_remove() {
        let mut index = HnswIndex::new(3, 16, 200);
        index.insert("a".into(), vec![1.0, 0.0, 0.0]);
        index.insert("b".into(), vec![0.0, 1.0, 0.0]);
        assert_eq!(index.len(), 2);

        index.remove("a");
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_many_vectors() {
        let mut index = HnswIndex::new(8, 16, 200);
        for i in 0..100 {
            let mut vec = vec![0.0f32; 8];
            vec[i % 8] = 1.0;
            vec[(i + 1) % 8] = 0.5;
            index.insert(format!("v{}", i), vec);
        }
        assert_eq!(index.len(), 100);

        let results = index.search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5, DistanceMetric::Cosine);
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_hnsw_m1_no_panic() {
        let mut index = HnswIndex::new(3, 1, 50);
        for i in 0..10 {
            index.insert(format!("v{}", i), vec![i as f32, 0.0, 0.0]);
        }
        let results = index.search(&[1.0, 0.0, 0.0], 5, DistanceMetric::Cosine);
        assert!(!results.is_empty());
    }
}
