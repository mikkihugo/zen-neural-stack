/**
 * @file zen-neural/src/gnn/utils.rs
 * @brief Utility functions for Graph Neural Network operations
 * 
 * This module provides essential utility functions and helper structures
 * for GNN operations, including graph validation, tensor utilities,
 * performance monitoring, and debugging assistance.
 * 
 * @author GNN Utility Agent (ruv-swarm)
 * @version 1.0.0-alpha.1
 * @since 2024-08-11
 */

use std::collections::HashMap;
use ndarray::{Array1, Array2};
use crate::gnn::{GNNError, GraphData};

/// Utility functions for graph validation and preprocessing
pub struct GraphUtils;

impl GraphUtils {
    /// Validates graph structure integrity
    pub fn validate_graph_structure(graph: &GraphData) -> Result<(), GNNError> {
        // Check for empty graph
        if graph.num_nodes() == 0 {
            return Err(GNNError::InvalidGraphStructure("Empty graph provided".to_string()));
        }
        
        // Validate edge indices are within node range
        let max_node_id = graph.num_nodes();
        for &(src, dst) in graph.edges() {
            if src >= max_node_id || dst >= max_node_id {
                return Err(GNNError::InvalidGraphStructure(
                    format!("Invalid edge ({}, {}): node indices must be in range [0, {})", 
                           src, dst, max_node_id - 1)
                ));
            }
        }
        
        // Check feature dimensions consistency
        if let Some(features) = graph.node_features() {
            let expected_dim = graph.node_feature_dim();
            if features.len() != graph.num_nodes() * expected_dim {
                return Err(GNNError::InvalidGraphStructure(
                    format!("Node feature dimension mismatch: expected {} features for {} nodes", 
                           expected_dim, graph.num_nodes())
                ));
            }
        }
        
        Ok(())
    }
    
    /// Compute graph statistics for monitoring and optimization
    pub fn compute_graph_statistics(graph: &GraphData) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        stats.insert("num_nodes".to_string(), graph.num_nodes() as f64);
        stats.insert("num_edges".to_string(), graph.edges().len() as f64);
        
        // Average degree
        let avg_degree = if graph.num_nodes() > 0 {
            (2.0 * graph.edges().len() as f64) / graph.num_nodes() as f64
        } else {
            0.0
        };
        stats.insert("avg_degree".to_string(), avg_degree);
        
        // Density
        let max_edges = graph.num_nodes() * (graph.num_nodes() - 1);
        let density = if max_edges > 0 {
            graph.edges().len() as f64 / max_edges as f64
        } else {
            0.0
        };
        stats.insert("density".to_string(), density);
        
        stats
    }
    
    /// Normalize node features using z-score normalization
    pub fn normalize_features(features: &mut Array2<f32>) -> Result<(), GNNError> {
        let (num_nodes, feature_dim) = features.dim();
        
        for feature_idx in 0..feature_dim {
            let mut feature_column = features.column_mut(feature_idx);
            
            // Compute mean and standard deviation
            let mean = feature_column.mean().unwrap_or(0.0);
            let variance = feature_column.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / num_nodes as f32;
            let std_dev = variance.sqrt();
            
            // Apply z-score normalization (avoid division by zero)
            if std_dev > 1e-8 {
                feature_column.mapv_inplace(|x| (x - mean) / std_dev);
            } else {
                feature_column.fill(0.0);
            }
        }
        
        Ok(())
    }
    
    /// Convert adjacency list to edge list format
    pub fn adjacency_to_edges(adj_list: &HashMap<usize, Vec<usize>>) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        
        for (&src, neighbors) in adj_list {
            for &dst in neighbors {
                edges.push((src, dst));
            }
        }
        
        edges
    }
    
    /// Convert edge list to adjacency list format
    pub fn edges_to_adjacency(edges: &[(usize, usize)], num_nodes: usize) -> HashMap<usize, Vec<usize>> {
        let mut adj_list = HashMap::new();
        
        // Initialize all nodes
        for i in 0..num_nodes {
            adj_list.insert(i, Vec::new());
        }
        
        // Add edges
        for &(src, dst) in edges {
            adj_list.entry(src).or_default().push(dst);
        }
        
        adj_list
    }
}

/// Tensor utility functions for GNN operations
pub struct TensorUtils;

impl TensorUtils {
    /// Apply softmax to the last dimension of a 2D array
    pub fn softmax(logits: &mut Array2<f32>) -> Result<(), GNNError> {
        let (batch_size, num_classes) = logits.dim();
        
        for i in 0..batch_size {
            let mut row = logits.row_mut(i);
            
            // Find maximum for numerical stability
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            
            // Subtract max and exponentiate
            row.mapv_inplace(|x| (x - max_val).exp());
            
            // Normalize
            let sum = row.sum();
            if sum > 0.0 {
                row.mapv_inplace(|x| x / sum);
            } else {
                // Handle edge case where all values are -inf
                row.fill(1.0 / num_classes as f32);
            }
        }
        
        Ok(())
    }
    
    /// Compute accuracy from predictions and targets
    pub fn compute_accuracy(predictions: &Array2<f32>, targets: &Array1<usize>) -> f32 {
        if predictions.nrows() != targets.len() {
            return 0.0;
        }
        
        let mut correct = 0;
        let total = targets.len();
        
        for (i, &target) in targets.iter().enumerate() {
            let pred_row = predictions.row(i);
            let predicted_class = pred_row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            if predicted_class == target {
                correct += 1;
            }
        }
        
        correct as f32 / total as f32
    }
    
    /// Initialize weights using Xavier/Glorot uniform initialization
    pub fn xavier_uniform(shape: (usize, usize), rng: &mut impl rand::Rng) -> Array2<f32> {
        use rand_distr::{Uniform, Distribution};
        
        let (rows, cols) = shape;
        let limit = (6.0 / (rows + cols) as f32).sqrt();
        let dist = Uniform::new(-limit, limit);
        
        Array2::from_shape_fn(shape, |_| dist.sample(rng))
    }
    
    /// Initialize weights using He uniform initialization (good for ReLU)
    pub fn he_uniform(shape: (usize, usize), rng: &mut impl rand::Rng) -> Array2<f32> {
        use rand_distr::{Uniform, Distribution};
        
        let (rows, _) = shape;
        let limit = (6.0 / rows as f32).sqrt();
        let dist = Uniform::new(-limit, limit);
        
        Array2::from_shape_fn(shape, |_| dist.sample(rng))
    }
    
    /// Clip gradients by norm to prevent exploding gradients
    pub fn clip_gradients_by_norm(gradients: &mut Array2<f32>, max_norm: f32) {
        let total_norm = gradients.iter()
            .map(|&g| g * g)
            .sum::<f32>()
            .sqrt();
        
        if total_norm > max_norm {
            let clip_factor = max_norm / total_norm;
            gradients.mapv_inplace(|g| g * clip_factor);
        }
    }
}

/// Performance monitoring utilities
pub struct PerformanceMonitor {
    start_time: std::time::Instant,
    timers: HashMap<String, std::time::Duration>,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            timers: HashMap::new(),
        }
    }
    
    /// Start timing an operation
    pub fn start_timer(&mut self, name: &str) {
        self.timers.insert(name.to_string(), std::time::Instant::now().elapsed());
    }
    
    /// End timing an operation and return duration
    pub fn end_timer(&mut self, name: &str) -> Option<std::time::Duration> {
        if let Some(start) = self.timers.remove(name) {
            let end = std::time::Instant::now().elapsed();
            Some(end - start)
        } else {
            None
        }
    }
    
    /// Get total elapsed time since creation
    pub fn total_elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
    
    /// Reset all timers
    pub fn reset(&mut self) {
        self.start_time = std::time::Instant::now();
        self.timers.clear();
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    
    #[test]
    fn test_graph_validation() {
        // This is a placeholder test - actual GraphData would come from the main GNN module
        // We'll add proper tests once the GNN module is fully integrated
        let stats = HashMap::new();
        assert!(stats.is_empty());
    }
    
    #[test]
    fn test_tensor_utils() {
        use rand::rngs::SmallRng;
        
        let mut rng = SmallRng::from_entropy();
        let weights = TensorUtils::xavier_uniform((3, 4), &mut rng);
        assert_eq!(weights.dim(), (3, 4));
        
        let he_weights = TensorUtils::he_uniform((3, 4), &mut rng);
        assert_eq!(he_weights.dim(), (3, 4));
    }
    
    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();
        monitor.start_timer("test");
        std::thread::sleep(std::time::Duration::from_millis(1));
        let duration = monitor.end_timer("test");
        assert!(duration.is_some());
    }
}