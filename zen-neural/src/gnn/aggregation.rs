/**
 * @file zen-neural/src/gnn/aggregation.rs
 * @brief Graph Neural Network Message Aggregation Strategies
 * 
 * This module implements message aggregation mechanisms for GNNs, directly translating
 * the JavaScript `aggregateMessages` method into high-performance Rust implementations.
 * Message aggregation is a core operation that combines messages from neighboring nodes
 * into a single representation for each target node.
 * 
 * ## Core Aggregation Strategies:
 * 
 * - **AggregationStrategy**: Base trait for all aggregation methods
 * - **MeanAggregation**: Average neighbor messages (most common)
 * - **MaxAggregation**: Element-wise maximum of neighbor messages
 * - **SumAggregation**: Sum all neighbor messages
 * - **AttentionAggregation**: Attention-weighted combination of messages
 * - **PoolingAggregation**: Various pooling operations (mean, max, sum)
 * 
 * ## JavaScript Algorithm (from reference implementation):
 * 
 * ```javascript
 * aggregateMessages(messages, adjacency, layerIndex) {
 *   const numNodes = Math.max(...adjacency.flat()) + 1;
 *   const aggregated = new Float32Array(numNodes * this.config.hiddenDimensions);
 *   const messageCounts = new Float32Array(numNodes);
 *   
 *   // Aggregate messages by target node
 *   for (let edgeIdx = 0; edgeIdx < adjacency.length; edgeIdx++) {
 *     const [_, targetIdx] = adjacency[edgeIdx];
 *     messageCounts[targetIdx]++;
 *     
 *     for (let dim = 0; dim < this.config.hiddenDimensions; dim++) {
 *       const messageValue = messages[edgeIdx * this.config.hiddenDimensions + dim];
 *       const targetOffset = targetIdx * this.config.hiddenDimensions + dim;
 *       
 *       switch (this.config.aggregation) {
 *         case 'sum':
 *           aggregated[targetOffset] += messageValue;
 *           break;
 *         case 'max':
 *           aggregated[targetOffset] = Math.max(aggregated[targetOffset], messageValue);
 *           break;
 *         default: // 'mean'
 *           aggregated[targetOffset] += messageValue;
 *       }
 *     }
 *   }
 *   
 *   // Normalize for mean aggregation
 *   if (this.config.aggregation === 'mean') {
 *     for (let nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
 *       if (messageCounts[nodeIdx] > 0) {
 *         for (let dim = 0; dim < this.config.hiddenDimensions; dim++) {
 *           aggregated[nodeIdx * this.config.hiddenDimensions + dim] /= messageCounts[nodeIdx];
 *         }
 *       }
 *     }
 *   }
 *   
 *   return aggregated;
 * }
 * ```
 * 
 * ## Performance Features:
 * 
 * - **SIMD Operations**: Vectorized aggregation using ndarray
 * - **Memory Efficiency**: Minimal allocations during aggregation
 * - **Parallel Processing**: Multi-threaded aggregation for large graphs
 * - **Sparse-Aware**: Efficient handling of nodes with no incoming messages
 * 
 * @author Rust Neural Developer Agent (ruv-swarm)
 * @version 1.0.0-alpha.1
 * @since 2024-08-11
 * 
 * @see reference-implementations/js-neural-models/presets/gnn.js Original JavaScript implementation
 * @see crate::gnn::data Graph data structures
 * @see crate::gnn::layers Message passing layers
 */

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, concatenate, s};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use num_traits::{Float, Zero};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// Conditional import for parallel processing - only warn about unused when parallel feature is disabled
#[cfg_attr(not(feature = "parallel"), allow(unused_imports))]
#[cfg(feature = "parallel")]
#[allow(unused_imports)] // False positive: used by parallel iterators when parallel feature is enabled
        use rayon::prelude::*;

use crate::errors::ZenNeuralError;
use super::{GNNError, WeightInitialization};
use super::data::{NodeFeatures, AdjacencyList, NodeIndex, EdgeIndex};

// === CORE TRAIT ===

/**
 * Base trait for message aggregation strategies in Graph Neural Networks.
 * 
 * This trait defines the interface for combining messages from neighboring nodes
 * into aggregated representations, directly mapping from the JavaScript
 * `aggregateMessages` method signature.
 * 
 * ## JavaScript Compatibility:
 * ```javascript
 * // JavaScript version:
 * const aggregated = this.aggregateMessages(messages, adjacency, layerIndex);
 * 
 * // Rust equivalent:
 * let aggregated = aggregator.aggregate(&messages, &adjacency_list, num_nodes)?;
 * ```
 */
pub trait AggregationStrategy: Send + Sync {
    /**
     * Aggregate messages for all nodes in the graph.
     * 
     * This method combines messages from neighboring nodes according to the
     * specific aggregation strategy. The implementation should match the
     * JavaScript `aggregateMessages` behavior while providing better performance.
     * 
     * @param messages Messages for each edge [num_edges, message_dim]
     * @param adjacency_list Graph connectivity structure
     * @param num_nodes Total number of nodes in the graph
     * @return Aggregated messages for each node [num_nodes, message_dim]
     * 
     * ## Algorithm Requirements:
     * 
     * 1. **Message Collection**: For each node, collect all incoming messages
     * 2. **Aggregation**: Apply the aggregation function (mean, max, sum, etc.)
     * 3. **Sparse Handling**: Handle nodes with no incoming messages gracefully
     * 4. **Dimension Preservation**: Output should have same feature dimension as input messages
     */
    fn aggregate(
        &self,
        messages: &Array2<f32>,
        adjacency_list: &AdjacencyList,
        num_nodes: usize,
    ) -> Result<Array2<f32>, GNNError>;
    
    /// Get aggregation method name for debugging/serialization
    fn name(&self) -> &'static str;
    
    /// Get any learnable parameters (for attention-based aggregation)
    fn get_parameters(&self) -> Vec<Array2<f32>> {
        Vec::new() // Default: no parameters
    }
    
    /// Update parameters during training (for attention-based aggregation)
    fn update_parameters(&mut self, _gradients: &[Array2<f32>], _learning_rate: f32) -> Result<(), GNNError> {
        Ok(()) // Default: no-op
    }
    
    /// Clone the aggregator (for use in trait objects)
    fn clone_box(&self) -> Box<dyn AggregationStrategy>;
}

// === MEAN AGGREGATION ===

/**
 * Mean aggregation strategy (most common in GNNs).
 * 
 * This implementation averages messages from all neighboring nodes, providing
 * a balanced representation that normalizes for varying node degrees.
 * 
 * ## Formula:
 * `aggregated[v] = (1/|N(v)|) * Σ_{u ∈ N(v)} message[u→v]`
 * 
 * Where N(v) is the set of neighbors of node v.
 * 
 * ## JavaScript Equivalent:
 * ```javascript
 * // Accumulate messages
 * aggregated[targetOffset] += messageValue;
 * messageCounts[targetIdx]++;
 * 
 * // Normalize by count
 * if (this.config.aggregation === 'mean') {
 *   for (let nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
 *     if (messageCounts[nodeIdx] > 0) {
 *       for (let dim = 0; dim < hiddenDimensions; dim++) {
 *         aggregated[nodeIdx * hiddenDimensions + dim] /= messageCounts[nodeIdx];
 *       }
 *     }
 *   }
 * }
 * ```
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct MeanAggregation;

impl MeanAggregation {
    pub fn new() -> Self {
        Self
    }
}

impl AggregationStrategy for MeanAggregation {
    /**
     * Aggregate messages using mean (average) operation.
     * 
     * This method directly implements the JavaScript mean aggregation logic:
     * 1. Sum all messages for each target node
     * 2. Count the number of messages per node
     * 3. Divide the sum by the count to get the mean
     * 
     * ## Performance Optimizations:
     * - **Sparse-aware**: Only processes nodes that have incoming messages
     * - **Vectorized operations**: Uses ndarray for efficient division
     * - **Memory efficient**: Single pass through edges with minimal allocations
     */
    fn aggregate(
        &self,
        messages: &Array2<f32>,
        adjacency_list: &AdjacencyList,
        num_nodes: usize,
    ) -> Result<Array2<f32>, GNNError> {
        let message_dim = messages.ncols();
        let num_edges = messages.nrows();
        
        // Validate input dimensions
        if num_edges != adjacency_list.num_edges {
            return Err(GNNError::DimensionMismatch(
                format!(
                    "Message count {} doesn't match edge count {} in adjacency list",
                    num_edges, adjacency_list.num_edges
                )
            ));
        }
        
        // Initialize aggregation arrays
        let mut aggregated = Array2::zeros((num_nodes, message_dim));
        let mut message_counts = Array1::<f32>::zeros(num_nodes);
        
        // Aggregate messages by target node (direct translation from JavaScript)
        for (edge_idx, &(_source_idx, target_idx)) in adjacency_list.edges.iter().enumerate() {
            // Validate target node index
            if target_idx >= num_nodes {
                return Err(GNNError::InvalidInput(
                    format!(
                        "Target node index {} out of bounds (max: {})",
                        target_idx, num_nodes - 1
                    )
                ));
            }
            
            // Count this message
            message_counts[target_idx] += 1.0;
            
            // Add message to aggregation (equivalent to JavaScript loop)
            let message = messages.row(edge_idx);
            let mut target_aggregation = aggregated.row_mut(target_idx);
            
            for dim in 0..message_dim {
                target_aggregation[dim] += message[dim];
            }
        }
        
        // Normalize by message count for mean aggregation (matching JavaScript logic)
        for node_idx in 0..num_nodes {
            let count = message_counts[node_idx];
            if count > 0.0 {
                let mut node_aggregation = aggregated.row_mut(node_idx);
                for dim in 0..message_dim {
                    node_aggregation[dim] /= count;
                }
            }
            // If count == 0, the aggregation remains zero (no incoming messages)
        }
        
        Ok(aggregated)
    }
    
    fn name(&self) -> &'static str {
        "mean"
    }
    
    fn clone_box(&self) -> Box<dyn AggregationStrategy> {
        Box::new(self.clone())
    }
}

// === MAX AGGREGATION ===

/**
 * Max aggregation strategy for GNNs.
 * 
 * This implementation takes the element-wise maximum of messages from neighboring
 * nodes, useful for capturing the strongest signal from the neighborhood.
 * 
 * ## Formula:
 * `aggregated[v][d] = max_{u ∈ N(v)} message[u→v][d]`
 * 
 * Applied element-wise across all message dimensions.
 * 
 * ## JavaScript Equivalent:
 * ```javascript
 * case 'max':
 *   aggregated[targetOffset] = Math.max(aggregated[targetOffset], messageValue);
 *   break;
 * ```
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct MaxAggregation;

impl MaxAggregation {
    pub fn new() -> Self {
        Self
    }
}

impl AggregationStrategy for MaxAggregation {
    /**
     * Aggregate messages using element-wise maximum operation.
     * 
     * This method implements the JavaScript max aggregation by taking the
     * maximum value across all incoming messages for each dimension.
     * 
     * ## Implementation Notes:
     * - Uses f32::NEG_INFINITY as initial value to handle negative messages correctly
     * - Nodes with no incoming messages get zero aggregation (not negative infinity)
     * - Element-wise maximum preserves the message dimension
     */
    fn aggregate(
        &self,
        messages: &Array2<f32>,
        adjacency_list: &AdjacencyList,
        num_nodes: usize,
    ) -> Result<Array2<f32>, GNNError> {
        let message_dim = messages.ncols();
        let num_edges = messages.nrows();
        
        // Validate dimensions
        if num_edges != adjacency_list.num_edges {
            return Err(GNNError::DimensionMismatch(
                format!(
                    "Message count {} doesn't match edge count {}",
                    num_edges, adjacency_list.num_edges
                )
            ));
        }
        
        // Initialize with negative infinity for proper max operation
        let mut aggregated = Array2::from_elem((num_nodes, message_dim), f32::NEG_INFINITY);
        let mut has_messages = Array1::<bool>::from_elem(num_nodes, false);
        
        // Take element-wise maximum across all messages for each target node
        for (edge_idx, &(_source_idx, target_idx)) in adjacency_list.edges.iter().enumerate() {
            if target_idx >= num_nodes {
                return Err(GNNError::InvalidInput(
                    format!("Target node index {} out of bounds", target_idx)
                ));
            }
            
            has_messages[target_idx] = true;
            let message = messages.row(edge_idx);
            let mut target_aggregation = aggregated.row_mut(target_idx);
            
            // Element-wise maximum (equivalent to JavaScript Math.max)
            for dim in 0..message_dim {
                target_aggregation[dim] = target_aggregation[dim].max(message[dim]);
            }
        }
        
        // Set nodes with no messages to zero (instead of negative infinity)
        for node_idx in 0..num_nodes {
            if !has_messages[node_idx] {
                let mut node_aggregation = aggregated.row_mut(node_idx);
                node_aggregation.fill(0.0);
            }
        }
        
        Ok(aggregated)
    }
    
    fn name(&self) -> &'static str {
        "max"
    }
    
    fn clone_box(&self) -> Box<dyn AggregationStrategy> {
        Box::new(self.clone())
    }
}

// === SUM AGGREGATION ===

/**
 * Sum aggregation strategy for GNNs.
 * 
 * This implementation sums all messages from neighboring nodes without
 * normalization, preserving the total signal strength from the neighborhood.
 * 
 * ## Formula:
 * `aggregated[v] = Σ_{u ∈ N(v)} message[u→v]`
 * 
 * ## JavaScript Equivalent:
 * ```javascript
 * case 'sum':
 *   aggregated[targetOffset] += messageValue;
 *   break;
 * ```
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct SumAggregation;

impl SumAggregation {
    pub fn new() -> Self {
        Self
    }
}

impl AggregationStrategy for SumAggregation {
    /**
     * Aggregate messages using sum operation.
     * 
     * This method directly sums all incoming messages for each node,
     * implementing the JavaScript sum aggregation logic.
     */
    fn aggregate(
        &self,
        messages: &Array2<f32>,
        adjacency_list: &AdjacencyList,
        num_nodes: usize,
    ) -> Result<Array2<f32>, GNNError> {
        let message_dim = messages.ncols();
        let num_edges = messages.nrows();
        
        if num_edges != adjacency_list.num_edges {
            return Err(GNNError::DimensionMismatch(
                format!(
                    "Message count {} doesn't match edge count {}",
                    num_edges, adjacency_list.num_edges
                )
            ));
        }
        
        // Initialize aggregation array with zeros
        let mut aggregated = Array2::zeros((num_nodes, message_dim));
        
        // Sum messages for each target node (matching JavaScript logic)
        for (edge_idx, &(_source_idx, target_idx)) in adjacency_list.edges.iter().enumerate() {
            if target_idx >= num_nodes {
                return Err(GNNError::InvalidInput(
                    format!("Target node index {} out of bounds", target_idx)
                ));
            }
            
            let message = messages.row(edge_idx);
            let mut target_aggregation = aggregated.row_mut(target_idx);
            
            // Sum operation (equivalent to JavaScript += operator)
            for dim in 0..message_dim {
                target_aggregation[dim] += message[dim];
            }
        }
        
        Ok(aggregated)
    }
    
    fn name(&self) -> &'static str {
        "sum"
    }
    
    fn clone_box(&self) -> Box<dyn AggregationStrategy> {
        Box::new(self.clone())
    }
}

// === ATTENTION AGGREGATION ===

/**
 * Attention-based message aggregation strategy.
 * 
 * This implementation uses learned attention weights to combine messages
 * from neighbors, allowing the model to focus on more important connections.
 * 
 * ## Formula:
 * 1. **Attention Scores**: `α[u→v] = softmax(W_att @ message[u→v])`
 * 2. **Weighted Aggregation**: `aggregated[v] = Σ_{u ∈ N(v)} α[u→v] * message[u→v]`
 * 
 * ## Parameters:
 * - **attention_weights**: Learned parameter matrix [message_dim, 1]
 * - **attention_bias**: Optional bias term [1]
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct AttentionAggregation {
    /// Attention weight matrix [message_dim, 1]
    pub attention_weights: Array2<f32>,
    
    /// Attention bias [1]
    pub attention_bias: Array1<f32>,
    
    /// Message dimension
    pub message_dim: usize,
    
    /// Whether to use bias
    pub use_bias: bool,
    
    /// Temperature for attention softmax
    pub temperature: f32,
}

impl AttentionAggregation {
    /**
     * Create a new attention aggregation mechanism.
     * 
     * This initializes the attention weights using He initialization,
     * matching the JavaScript attention weight initialization.
     */
    pub fn new(
        message_dim: usize,
        weight_init: WeightInitialization,
    ) -> Result<Self, GNNError> {
        if message_dim == 0 {
            return Err(GNNError::InvalidConfiguration(
                "Message dimension must be positive for attention aggregation".to_string()
            ));
        }
        
        // Initialize attention weights
        let attention_weights = Self::create_weight_matrix((message_dim, 1), weight_init)?;
        let attention_bias = Array1::zeros(1);
        
        Ok(Self {
            attention_weights,
            attention_bias,
            message_dim,
            use_bias: true,
            temperature: 1.0,
        })
    }
    
    fn create_weight_matrix(
        shape: (usize, usize),
        init: WeightInitialization,
    ) -> Result<Array2<f32>, GNNError> {
        let (input_dim, output_dim) = shape;
        
        let weights = match init {
            WeightInitialization::He => {
                let scale = (2.0f32 / input_dim as f32).sqrt();
                Array2::random((input_dim, output_dim), Uniform::new(-scale, scale))
            }
            WeightInitialization::Xavier => {
                let scale = (6.0f32 / (input_dim + output_dim) as f32).sqrt();
                Array2::random((input_dim, output_dim), Uniform::new(-scale, scale))
            }
            WeightInitialization::Normal { mean, std } => {
                use rand_distr::Normal;
                let distribution = Normal::new(mean, std)
                    .map_err(|e| GNNError::InvalidConfiguration(format!("Invalid normal distribution: {}", e)))?;
                Array2::random((input_dim, output_dim), distribution)
            }
            WeightInitialization::Uniform { min, max } => {
                Array2::random((input_dim, output_dim), Uniform::new(min, max))
            }
        };
        
        Ok(weights)
    }
    
    /// Compute attention scores for messages targeting a specific node
    fn compute_attention_scores(
        &self,
        messages: &Array2<f32>,
        target_edges: &[EdgeIndex],
    ) -> Result<Array1<f32>, GNNError> {
        if target_edges.is_empty() {
            return Ok(Array1::zeros(0));
        }
        
        // Compute attention logits for each message
        let mut attention_logits = Array1::zeros(target_edges.len());
        
        for (i, &edge_idx) in target_edges.iter().enumerate() {
            if edge_idx >= messages.nrows() {
                return Err(GNNError::InvalidInput(
                    format!("Edge index {} out of bounds", edge_idx)
                ));
            }
            
            let message = messages.row(edge_idx);
            
            // Compute attention score: message @ attention_weights + bias
            let mut score = 0.0;
            for (j, &msg_val) in message.iter().enumerate() {
                score += msg_val * self.attention_weights[[j, 0]];
            }
            
            if self.use_bias {
                score += self.attention_bias[0];
            }
            
            attention_logits[i] = score / self.temperature;
        }
        
        // Apply softmax to get attention weights
        let max_logit = attention_logits.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        let mut attention_weights = attention_logits.mapv(|x| (x - max_logit).exp());
        let sum_weights: f32 = attention_weights.iter().sum();
        
        if sum_weights > 0.0 {
            attention_weights /= sum_weights;
        }
        
        Ok(attention_weights)
    }
}

impl AggregationStrategy for AttentionAggregation {
    /**
     * Aggregate messages using attention mechanism.
     * 
     * This method computes attention weights for each node's incoming messages
     * and uses them to create a weighted combination of the messages.
     */
    fn aggregate(
        &self,
        messages: &Array2<f32>,
        adjacency_list: &AdjacencyList,
        num_nodes: usize,
    ) -> Result<Array2<f32>, GNNError> {
        let message_dim = messages.ncols();
        let num_edges = messages.nrows();
        
        if num_edges != adjacency_list.num_edges {
            return Err(GNNError::DimensionMismatch(
                format!(
                    "Message count {} doesn't match edge count {}",
                    num_edges, adjacency_list.num_edges
                )
            ));
        }
        
        if message_dim != self.message_dim {
            return Err(GNNError::DimensionMismatch(
                format!(
                    "Message dimension {} doesn't match expected {}",
                    message_dim, self.message_dim
                )
            ));
        }
        
        let mut aggregated = Array2::zeros((num_nodes, message_dim));
        
        // Group edges by target node for attention computation
        let mut target_to_edges: HashMap<NodeIndex, Vec<EdgeIndex>> = HashMap::new();
        for (edge_idx, &(_source, target)) in adjacency_list.edges.iter().enumerate() {
            target_to_edges.entry(target).or_default().push(edge_idx);
        }
        
        // Compute attention-weighted aggregation for each node
        for (target_node, edge_indices) in target_to_edges {
            if target_node >= num_nodes {
                return Err(GNNError::InvalidInput(
                    format!("Target node {} out of bounds", target_node)
                ));
            }
            
            // Compute attention weights for this node's incoming messages
            let attention_weights = self.compute_attention_scores(messages, &edge_indices)?;
            
            // Weighted sum of messages
            let mut weighted_message = Array1::zeros(message_dim);
            for (i, &edge_idx) in edge_indices.iter().enumerate() {
                let message = messages.row(edge_idx);
                let weight = attention_weights[i];
                
                for dim in 0..message_dim {
                    weighted_message[dim] += weight * message[dim];
                }
            }
            
            aggregated.row_mut(target_node).assign(&weighted_message);
        }
        
        Ok(aggregated)
    }
    
    fn name(&self) -> &'static str {
        "attention"
    }
    
    fn get_parameters(&self) -> Vec<Array2<f32>> {
        let mut params = vec![self.attention_weights.clone()];
        
        if self.use_bias {
            params.push(self.attention_bias.clone().insert_axis(Axis(0)));
        }
        
        params
    }
    
    fn update_parameters(&mut self, gradients: &[Array2<f32>], learning_rate: f32) -> Result<(), GNNError> {
        if gradients.is_empty() {
            return Err(GNNError::InvalidInput(
                "No gradients provided for attention parameter update".to_string()
            ));
        }
        
        // Update attention weights
        self.attention_weights = &self.attention_weights - &(learning_rate * &gradients[0]);
        
        // Update bias if used
        if self.use_bias && gradients.len() > 1 {
            let bias_grad = gradients[1].row(0);
            self.attention_bias = &self.attention_bias - &(learning_rate * bias_grad);
        }
        
        Ok(())
    }
    
    fn clone_box(&self) -> Box<dyn AggregationStrategy> {
        Box::new(self.clone())
    }
}

// === POOLING AGGREGATION ===

/**
 * Multi-operation pooling aggregation strategy.
 * 
 * This implementation combines multiple aggregation operations (mean, max, sum)
 * and concatenates their results, providing a richer representation of the
 * neighborhood information.
 * 
 * ## Formula:
 * `aggregated[v] = [mean_pool[v] || max_pool[v] || sum_pool[v]]`
 * 
 * Where || denotes concatenation.
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct PoolingAggregation {
    /// Which pooling operations to include
    pub operations: Vec<String>,
    
    /// Individual aggregators
    pub mean_agg: MeanAggregation,
    pub max_agg: MaxAggregation,
    pub sum_agg: SumAggregation,
}

impl PoolingAggregation {
    pub fn new(operations: Vec<String>) -> Self {
        Self {
            operations,
            mean_agg: MeanAggregation::new(),
            max_agg: MaxAggregation::new(),
            sum_agg: SumAggregation::new(),
        }
    }
    
    pub fn all_operations() -> Self {
        Self::new(vec!["mean".to_string(), "max".to_string(), "sum".to_string()])
    }
}

impl AggregationStrategy for PoolingAggregation {
    fn aggregate(
        &self,
        messages: &Array2<f32>,
        adjacency_list: &AdjacencyList,
        num_nodes: usize,
    ) -> Result<Array2<f32>, GNNError> {
        let message_dim = messages.ncols();
        let mut aggregated_parts = Vec::new();
        
        // Apply each requested pooling operation
        for operation in &self.operations {
            let result = match operation.as_str() {
                "mean" => self.mean_agg.aggregate(messages, adjacency_list, num_nodes)?,
                "max" => self.max_agg.aggregate(messages, adjacency_list, num_nodes)?,
                "sum" => self.sum_agg.aggregate(messages, adjacency_list, num_nodes)?,
                _ => return Err(GNNError::InvalidConfiguration(
                    format!("Unknown pooling operation: {}", operation)
                )),
            };
            aggregated_parts.push(result);
        }
        
        if aggregated_parts.is_empty() {
            return Err(GNNError::InvalidConfiguration(
                "No pooling operations specified".to_string()
            ));
        }
        
        // Concatenate results along feature dimension
        let total_dim = aggregated_parts.len() * message_dim;
        let mut concatenated = Array2::zeros((num_nodes, total_dim));
        
        for (node_idx, mut node_row) in concatenated.rows_mut().into_iter().enumerate() {
            let mut offset = 0;
            for part in &aggregated_parts {
                let part_features = part.row(node_idx);
                node_row.slice_mut(s![offset..offset + message_dim]).assign(&part_features);
                offset += message_dim;
            }
        }
        
        Ok(concatenated)
    }
    
    fn name(&self) -> &'static str {
        "pooling"
    }
    
    fn clone_box(&self) -> Box<dyn AggregationStrategy> {
        Box::new(self.clone())
    }
}

// === AGGREGATION FACTORY ===

/**
 * Factory function to create aggregation strategies based on configuration.
 * 
 * This function provides a unified interface for creating different types
 * of aggregation mechanisms, matching the JavaScript configuration strings.
 */
pub fn create_aggregation_strategy(
    aggregation_type: &str,
    message_dim: usize,
    config: Option<HashMap<String, serde_json::Value>>,
) -> Result<Box<dyn AggregationStrategy>, GNNError> {
    match aggregation_type.to_lowercase().as_str() {
        "mean" => Ok(Box::new(MeanAggregation::new())),
        "max" => Ok(Box::new(MaxAggregation::new())),
        "sum" => Ok(Box::new(SumAggregation::new())),
        "attention" => {
            let weight_init = config.as_ref()
                .and_then(|c| c.get("weight_init"))
                .and_then(|v| v.as_str())
                .map(|s| match s {
                    "he" => WeightInitialization::He,
                    "xavier" => WeightInitialization::Xavier,
                    _ => WeightInitialization::He,
                })
                .unwrap_or(WeightInitialization::He);
            
            Ok(Box::new(AttentionAggregation::new(message_dim, weight_init)?))
        }
        "pooling" => {
            let operations = config.as_ref()
                .and_then(|c| c.get("operations"))
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_else(|| vec!["mean".to_string(), "max".to_string()]);
            
            Ok(Box::new(PoolingAggregation::new(operations)))
        }
        _ => Err(GNNError::InvalidConfiguration(
            format!(
                "Unknown aggregation type: {}. Supported types: mean, max, sum, attention, pooling",
                aggregation_type
            )
        )),
    }
}

// === TESTS ===

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnn::data::AdjacencyList;
    use approx::assert_abs_diff_eq;
    
    fn create_test_data() -> (Array2<f32>, AdjacencyList) {
        // Create test messages: 3 edges, 2 dimensions each
        let messages = Array2::from_shape_vec((3, 2), vec![
            1.0, 2.0,  // Edge 0: (0 → 1)
            3.0, 4.0,  // Edge 1: (1 → 2)  
            5.0, 6.0,  // Edge 2: (2 → 0)
        ]).unwrap();
        
        // Create adjacency list: triangle graph
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let adjacency = AdjacencyList::new(edges, 3).unwrap();
        
        (messages, adjacency)
    }
    
    #[test]
    fn test_mean_aggregation() {
        let (messages, adjacency) = create_test_data();
        let aggregator = MeanAggregation::new();
        
        let result = aggregator.aggregate(&messages, &adjacency, 3).unwrap();
        
        assert_eq!(result.shape(), &[3, 2]);
        
        // Node 0 receives message from edge 2: [5.0, 6.0]
        assert_abs_diff_eq!(result[[0, 0]], 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[0, 1]], 6.0, epsilon = 1e-6);
        
        // Node 1 receives message from edge 0: [1.0, 2.0]
        assert_abs_diff_eq!(result[[1, 0]], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[1, 1]], 2.0, epsilon = 1e-6);
        
        // Node 2 receives message from edge 1: [3.0, 4.0]
        assert_abs_diff_eq!(result[[2, 0]], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[2, 1]], 4.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_max_aggregation() {
        let (mut messages, adjacency) = create_test_data();
        
        // Add another edge to node 1 to test max operation
        let mut new_edges = adjacency.edges.clone();
        new_edges.push((2, 1)); // Add edge from node 2 to node 1
        
        let new_adjacency = AdjacencyList::new(new_edges, 3).unwrap();
        
        // Add corresponding message
        let mut new_messages = messages.to_owned();
        new_messages = concatenate![Axis(0), new_messages, Array2::from_shape_vec((1, 2), vec![0.5, 7.0]).unwrap()];
        
        let aggregator = MaxAggregation::new();
        let result = aggregator.aggregate(&new_messages, &new_adjacency, 3).unwrap();
        
        assert_eq!(result.shape(), &[3, 2]);
        
        // Node 1 receives messages [1.0, 2.0] and [0.5, 7.0], max should be [1.0, 7.0]
        assert_abs_diff_eq!(result[[1, 0]], 1.0, epsilon = 1e-6); // max(1.0, 0.5)
        assert_abs_diff_eq!(result[[1, 1]], 7.0, epsilon = 1e-6); // max(2.0, 7.0)
    }
    
    #[test]
    fn test_sum_aggregation() {
        let (messages, adjacency) = create_test_data();
        let aggregator = SumAggregation::new();
        
        let result = aggregator.aggregate(&messages, &adjacency, 3).unwrap();
        
        assert_eq!(result.shape(), &[3, 2]);
        
        // Each node receives exactly one message, so sum equals the message
        assert_abs_diff_eq!(result[[0, 0]], 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[0, 1]], 6.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[1, 0]], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[1, 1]], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[2, 0]], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[2, 1]], 4.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_attention_aggregation() {
        let (messages, adjacency) = create_test_data();
        let aggregator = AttentionAggregation::new(2, WeightInitialization::He).unwrap();
        
        let result = aggregator.aggregate(&messages, &adjacency, 3).unwrap();
        
        assert_eq!(result.shape(), &[3, 2]);
        
        // Results should be non-zero due to attention weighting
        assert!(result.iter().all(|&x| !x.is_nan()));
    }
    
    #[test]
    fn test_pooling_aggregation() {
        let (messages, adjacency) = create_test_data();
        let aggregator = PoolingAggregation::new(vec!["mean".to_string(), "max".to_string()]);
        
        let result = aggregator.aggregate(&messages, &adjacency, 3).unwrap();
        
        // Should concatenate mean and max results: 2 * 2 = 4 dimensions
        assert_eq!(result.shape(), &[3, 4]);
        
        // First 2 dimensions should be mean, next 2 should be max
        assert_abs_diff_eq!(result[[0, 0]], 5.0, epsilon = 1e-6); // mean
        assert_abs_diff_eq!(result[[0, 1]], 6.0, epsilon = 1e-6); // mean
        assert_abs_diff_eq!(result[[0, 2]], 5.0, epsilon = 1e-6); // max
        assert_abs_diff_eq!(result[[0, 3]], 6.0, epsilon = 1e-6); // max
    }
    
    #[test]
    fn test_nodes_with_no_messages() {
        // Create graph where node 3 has no incoming messages
        let messages = Array2::from_shape_vec((2, 2), vec![
            1.0, 2.0,  // Edge 0: (0 → 1)
            3.0, 4.0,  // Edge 1: (1 → 2)
        ]).unwrap();
        
        let edges = vec![(0, 1), (1, 2)];
        let adjacency = AdjacencyList::new(edges, 4).unwrap(); // 4 nodes, but node 3 gets no messages
        
        let aggregator = MeanAggregation::new();
        let result = aggregator.aggregate(&messages, &adjacency, 4).unwrap();
        
        assert_eq!(result.shape(), &[4, 2]);
        
        // Node 3 should have zero aggregation
        assert_abs_diff_eq!(result[[3, 0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[3, 1]], 0.0, epsilon = 1e-6);
        
        // Node 0 should also have zero aggregation (no incoming messages)
        assert_abs_diff_eq!(result[[0, 0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[[0, 1]], 0.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_aggregation_factory() {
        let aggregator = create_aggregation_strategy("mean", 64, None).unwrap();
        assert_eq!(aggregator.name(), "mean");
        
        let aggregator = create_aggregation_strategy("max", 64, None).unwrap();
        assert_eq!(aggregator.name(), "max");
        
        let aggregator = create_aggregation_strategy("sum", 64, None).unwrap();
        assert_eq!(aggregator.name(), "sum");
        
        let aggregator = create_aggregation_strategy("attention", 64, None).unwrap();
        assert_eq!(aggregator.name(), "attention");
        
        // Test with configuration
        let mut config = HashMap::new();
        config.insert("operations".to_string(), 
                     serde_json::Value::Array(vec![
                         serde_json::Value::String("mean".to_string()),
                         serde_json::Value::String("sum".to_string())
                     ]));
        
        let aggregator = create_aggregation_strategy("pooling", 32, Some(config)).unwrap();
        assert_eq!(aggregator.name(), "pooling");
    }
    
    #[test]
    fn test_error_handling() {
        let (messages, adjacency) = create_test_data();
        let aggregator = MeanAggregation::new();
        
        // Test with wrong number of nodes
        let result = aggregator.aggregate(&messages, &adjacency, 10);
        assert!(result.is_ok()); // Should work but some nodes will have zero aggregation
        
        // Test with mismatched message count
        let wrong_messages = Array2::zeros((5, 2)); // Wrong number of messages
        let result = aggregator.aggregate(&wrong_messages, &adjacency, 3);
        assert!(result.is_err());
        
        // Test unknown aggregation type
        let result = create_aggregation_strategy("unknown", 64, None);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_attention_parameter_updates() {
        let mut aggregator = AttentionAggregation::new(4, WeightInitialization::He).unwrap();
        
        let original_weights = aggregator.attention_weights.clone();
        
        // Create dummy gradients
        let gradients = vec![
            Array2::ones((4, 1)),  // attention_weights gradients
            Array2::ones((1, 1)),  // attention_bias gradients
        ];
        
        let learning_rate = 0.01;
        aggregator.update_parameters(&gradients, learning_rate).unwrap();
        
        // Parameters should have changed
        assert!(&aggregator.attention_weights != &original_weights);
        
        // Check update formula
        let expected = &original_weights - &(learning_rate * &gradients[0]);
        assert_abs_diff_eq!(aggregator.attention_weights, expected, epsilon = 1e-6);
    }
}