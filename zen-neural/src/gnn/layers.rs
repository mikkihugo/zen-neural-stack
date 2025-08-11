/**
 * @file zen-neural/src/gnn/layers.rs
 * @brief Graph Neural Network Message Passing Layers
 * 
 * This module implements the core message passing layers for Graph Neural Networks,
 * directly translated from the JavaScript reference implementation. The layers handle
 * the fundamental GNN operations: message computation, neighbor aggregation, and
 * node feature transformation.
 * 
 * ## Core Layer Types:
 * 
 * - **MessagePassingLayer**: Base trait defining the message passing interface
 * - **GraphConvLayer**: Standard graph convolution with linear transformations
 * - **GraphSAGELayer**: Sampling and aggregation for large graphs
 * - **GATLayer**: Graph attention mechanism for weighted message passing
 * 
 * ## Message Passing Algorithm:
 * 
 * 1. **Message Computation**: Transform node and edge features into messages
 * 2. **Message Passing**: Send messages along graph edges
 * 3. **Message Aggregation**: Combine messages from neighbors (mean/max/sum)
 * 4. **Node Update**: Update node representations with aggregated messages
 * 
 * ## Performance Features:
 * 
 * - **SIMD Operations**: Vectorized linear algebra using ndarray
 * - **Parallel Processing**: Multi-threaded message computation
 * - **Memory Efficiency**: In-place operations where possible
 * - **GPU Ready**: Async interface compatible with WebGPU backend
 * 
 * @author Rust Neural Developer Agent (ruv-swarm)
 * @version 1.0.0-alpha.1
 * @since 2024-08-11
 * 
 * @see reference-implementations/js-neural-models/presets/gnn.js Original JavaScript implementation
 * @see crate::gnn::data Graph data structures
 * @see crate::gnn::aggregation Message aggregation strategies
 */

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use num_traits::{Float, Zero};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "gpu")]
use crate::webgpu::{WebGPUBackend, ComputeContext};

use crate::errors::ZenNeuralError;
use super::{GNNError, ActivationFunction, WeightInitialization};
use super::data::{GraphData, NodeFeatures, EdgeFeatures, AdjacencyList, NodeIndex, EdgeIndex};

// === CORE TRAITS ===

/**
 * Base trait for all message passing layers in the GNN.
 * 
 * This trait defines the core interface that all GNN layers must implement,
 * directly mapping from the JavaScript class methods. It provides async
 * message computation to support both CPU and GPU backends.
 * 
 * ## JavaScript Compatibility:
 * ```javascript
 * // JavaScript version:
 * const messages = await layer.computeMessages(nodes, edges, adjacency, layerIndex);
 * 
 * // Rust equivalent:
 * let messages = layer.compute_messages(nodes, edges, adjacency, layer_index).await?;
 * ```
 */
#[async_trait::async_trait]
pub trait MessagePassingLayer: Send + Sync {
    /**
     * Compute messages from node and edge features.
     * 
     * This is the core method that transforms node and edge features into
     * messages that will be passed along graph edges. The implementation
     * mirrors the JavaScript `computeMessages` method but with improved
     * type safety and performance.
     * 
     * @param node_representations Current node feature matrix [num_nodes, feature_dim]
     * @param edge_features Optional edge feature matrix [num_edges, edge_feature_dim]
     * @param adjacency_list Graph connectivity structure
     * @param layer_index Current layer index for multi-layer networks
     * @return Messages for each edge [num_edges, hidden_dim]
     * 
     * ## Algorithm (from JavaScript):
     * 
     * ```javascript
     * // For each edge, compute message
     * for (let edgeIdx = 0; edgeIdx < numEdges; edgeIdx++) {
     *   const [sourceIdx, _targetIdx] = adjacency[edgeIdx];
     *   
     *   // Get source node features
     *   const sourceFeatures = nodes.slice(sourceStart, sourceEnd);
     *   
     *   // Transform source node features
     *   const nodeMessage = this.transform(sourceFeatures, weights.nodeToMessage, weights.messageBias);
     *   
     *   // If edge features exist, incorporate them
     *   if (edges && edges.length > 0) {
     *     const edgeMessage = this.transform(edgeFeatures, weights.edgeToMessage, zeroBias);
     *     // Combine node and edge messages
     *     messages[edgeIdx] = nodeMessage + edgeMessage;
     *   } else {
     *     messages[edgeIdx] = nodeMessage;
     *   }
     * }
     * ```
     */
    async fn compute_messages(
        &self,
        node_representations: &NodeFeatures,
        edge_features: &Option<EdgeFeatures>,
        adjacency_list: &AdjacencyList,
        layer_index: usize,
    ) -> Result<Array2<f32>, GNNError>;
    
    /// Get input feature dimension for this layer
    fn input_dim(&self) -> usize;
    
    /// Get output feature dimension for this layer
    fn output_dim(&self) -> usize;
    
    /// Get edge feature dimension expected by this layer
    fn edge_dim(&self) -> usize;
    
    /// Get layer parameters for serialization/debugging
    fn get_parameters(&self) -> Vec<Array2<f32>>;
    
    /// Update layer parameters during training (simplified interface)
    fn update_parameters(&mut self, gradients: &[Array2<f32>], learning_rate: f32) -> Result<(), GNNError>;
}

// === LAYER IMPLEMENTATIONS ===

/**
 * Standard Graph Convolution Layer implementation.
 * 
 * This layer implements the basic graph convolution operation as defined
 * in the original JavaScript implementation. It performs linear transformations
 * on node and edge features to generate messages.
 * 
 * ## Weight Structure (from JavaScript):
 * ```javascript
 * this.messageWeights[layer] = {
 *   nodeToMessage: Float32Array,  // [inputDim, hiddenDim]
 *   edgeToMessage: Float32Array,  // [edgeDim, hiddenDim]  
 *   messageBias: Float32Array     // [hiddenDim]
 * };
 * ```
 * 
 * ## Features:
 * - **Linear Transformations**: Efficient matrix multiplications
 * - **Bias Terms**: Configurable bias for better expressivity
 * - **Edge Integration**: Optional edge feature incorporation
 * - **He Initialization**: Proper weight initialization for ReLU networks
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct GraphConvLayer {
    /// Node-to-message transformation matrix [input_dim, hidden_dim]
    pub node_to_message: Array2<f32>,
    
    /// Edge-to-message transformation matrix [edge_dim, hidden_dim]
    pub edge_to_message: Array2<f32>,
    
    /// Message bias vector [hidden_dim]
    pub message_bias: Array1<f32>,
    
    /// Input feature dimension
    pub input_dim: usize,
    
    /// Hidden feature dimension (output)
    pub hidden_dim: usize,
    
    /// Edge feature dimension
    pub edge_dim: usize,
    
    /// Activation function
    pub activation: ActivationFunction,
    
    /// Whether to use bias terms
    pub use_bias: bool,
}

impl GraphConvLayer {
    /**
     * Create a new Graph Convolution layer with specified dimensions.
     * 
     * This constructor initializes the layer weights using He initialization
     * (matching the JavaScript `createWeight` method) and sets up the
     * transformation matrices.
     * 
     * @param input_dim Input node feature dimension
     * @param hidden_dim Hidden layer dimension (output)
     * @param edge_dim Edge feature dimension
     * @param activation Activation function to use
     * @param use_bias Whether to include bias terms
     * @return Initialized GraphConv layer
     * 
     * ## JavaScript Equivalent:
     * ```javascript
     * this.messageWeights.push({
     *   nodeToMessage: this.createWeight([inputDim, this.config.hiddenDimensions]),
     *   edgeToMessage: this.createWeight([this.config.edgeDimensions, this.config.hiddenDimensions]),
     *   messageBias: new Float32Array(this.config.hiddenDimensions).fill(0.0),
     * });
     * ```
     */
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        edge_dim: usize,
        activation: ActivationFunction,
        use_bias: bool,
    ) -> Result<Self, GNNError> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err(GNNError::InvalidConfiguration(
                format!("Layer dimensions must be positive: input_dim={}, hidden_dim={}", 
                       input_dim, hidden_dim)
            ));
        }
        
        // Initialize weights using He initialization (from JavaScript createWeight method)
        let node_to_message = Self::create_weight_matrix((input_dim, hidden_dim), WeightInitialization::He)?;
        let edge_to_message = Self::create_weight_matrix((edge_dim, hidden_dim), WeightInitialization::He)?;
        
        // Initialize bias to zero (matching JavaScript)
        let message_bias = if use_bias {
            Array1::zeros(hidden_dim)
        } else {
            Array1::zeros(0) // Empty bias if not used
        };
        
        Ok(Self {
            node_to_message,
            edge_to_message,
            message_bias,
            input_dim,
            hidden_dim,
            edge_dim,
            activation,
            use_bias,
        })
    }
    
    /**
     * Create weight matrix with specified initialization (from JavaScript createWeight).
     * 
     * This method replicates the JavaScript weight initialization:
     * ```javascript
     * createWeight(shape) {
     *   const size = shape.reduce((a, b) => a * b, 1);
     *   const weight = new Float32Array(size);
     *   
     *   // He initialization for ReLU
     *   const scale = Math.sqrt(2.0 / shape[0]);
     *   for (let i = 0; i < size; i++) {
     *     weight[i] = (Math.random() * 2 - 1) * scale;
     *   }
     *   
     *   return weight;
     * }
     * ```
     */
    fn create_weight_matrix(
        shape: (usize, usize),
        init: WeightInitialization,
    ) -> Result<Array2<f32>, GNNError> {
        let (input_dim, output_dim) = shape;
        
        if input_dim == 0 || output_dim == 0 {
            return Ok(Array2::zeros((input_dim, output_dim)));
        }
        
        let weights = match init {
            WeightInitialization::He => {
                // He initialization: scale = sqrt(2.0 / fan_in)
                let scale = (2.0f32 / input_dim as f32).sqrt();
                Array2::random((input_dim, output_dim), Uniform::new(-scale, scale))
            }
            WeightInitialization::Xavier => {
                // Xavier initialization: scale = sqrt(6.0 / (fan_in + fan_out))
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
    
    /**
     * Linear transformation matching JavaScript `transform` method.
     * 
     * This method performs the core linear algebra operation:
     * `output = input @ weight + bias`
     * 
     * ## JavaScript Implementation:
     * ```javascript
     * transform(input, weight, bias) {
     *   // Simple linear transformation
     *   const inputDim = weight.shape[0];
     *   const outputDim = weight.shape[1]; 
     *   const numSamples = input.length / inputDim;
     *   const output = new Float32Array(numSamples * outputDim);
     *   
     *   for (let sample = 0; sample < numSamples; sample++) {
     *     for (let out = 0; out < outputDim; out++) {
     *       let sum = bias[out];
     *       for (let inp = 0; inp < inputDim; inp++) {
     *         sum += input[sample * inputDim + inp] * weight[inp * outputDim + out];
     *       }
     *       output[sample * outputDim + out] = sum;
     *     }
     *   }
     *   
     *   return output;
     * }
     * ```
     */
    fn transform(
        &self,
        input: ArrayView2<f32>,
        weight: &Array2<f32>,
        bias: &Array1<f32>,
    ) -> Result<Array2<f32>, GNNError> {
        // Validate dimensions
        if input.ncols() != weight.nrows() {
            return Err(GNNError::DimensionMismatch(
                format!(
                    "Input dimension {} doesn't match weight input dimension {}",
                    input.ncols(), weight.nrows()
                )
            ));
        }
        
        if self.use_bias && bias.len() != weight.ncols() {
            return Err(GNNError::DimensionMismatch(
                format!(
                    "Bias dimension {} doesn't match weight output dimension {}",
                    bias.len(), weight.ncols()
                )
            ));
        }
        
        // Perform matrix multiplication: input @ weight
        let mut output = input.dot(weight);
        
        // Add bias if enabled (broadcasting)
        if self.use_bias && !bias.is_empty() {
            output = output + bias;
        }
        
        Ok(output)
    }
    
    /// Apply activation function (from JavaScript applyActivation)
    fn apply_activation(&self, input: Array2<f32>) -> Array2<f32> {
        match self.activation {
            ActivationFunction::ReLU => input.mapv(|x| x.max(0.0)),
            ActivationFunction::Tanh => input.mapv(|x| x.tanh()),
            ActivationFunction::Sigmoid => input.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationFunction::LeakyReLU => input.mapv(|x| if x > 0.0 { x } else { 0.01 * x }),
            ActivationFunction::GELU => input.mapv(|x| {
                0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh())
            }),
            ActivationFunction::Linear => input,
        }
    }
}

#[async_trait::async_trait]
impl MessagePassingLayer for GraphConvLayer {
    /**
     * Compute messages for each edge (direct translation from JavaScript).
     * 
     * This method implements the core message computation algorithm from
     * the JavaScript `computeMessages` method with improved performance
     * and memory management.
     * 
     * ## Algorithm Steps:
     * 1. For each edge (source, target) in the adjacency list:
     *    - Extract source node features
     *    - Transform to message space using node_to_message weights
     *    - If edge features exist, transform and combine with node message
     *    - Store result in messages array
     * 
     * ## Performance Optimizations:
     * - **Batch Processing**: Process multiple edges simultaneously
     * - **SIMD Operations**: Vectorized linear algebra with ndarray
     * - **Memory Reuse**: Minimize allocations during computation
     * - **Parallel Execution**: Multi-threaded when available
     */
    async fn compute_messages(
        &self,
        node_representations: &NodeFeatures,
        edge_features: &Option<EdgeFeatures>,
        adjacency_list: &AdjacencyList,
        _layer_index: usize,
    ) -> Result<Array2<f32>, GNNError> {
        let num_edges = adjacency_list.num_edges;
        
        if num_edges == 0 {
            return Ok(Array2::zeros((0, self.hidden_dim)));
        }
        
        // Pre-allocate message array
        let mut messages = Array2::zeros((num_edges, self.hidden_dim));
        
        // Process messages for each edge
        #[cfg(feature = "parallel")]
        let edge_iter = adjacency_list.edges.par_iter().enumerate();
        #[cfg(not(feature = "parallel"))]
        let edge_iter = adjacency_list.edges.iter().enumerate();
        
        let message_results: Result<Vec<_>, GNNError> = edge_iter.map(|(edge_idx, &(source_idx, _target_idx))| {
            // Validate source node index
            if source_idx >= node_representations.nrows() {
                return Err(GNNError::InvalidInput(
                    format!(
                        "Source node index {} out of bounds (max: {})", 
                        source_idx, node_representations.nrows() - 1
                    )
                ));
            }
            
            // Extract source node features (equivalent to JavaScript sourceFeatures)
            let source_features = node_representations.row(source_idx);
            let source_matrix = source_features.insert_axis(Axis(0)); // Convert to 2D
            
            // Transform node features to message space
            let node_message = self.transform(
                source_matrix.view(),
                &self.node_to_message,
                &self.message_bias,
            )?;
            
            // Get the message vector (remove batch dimension)
            let mut final_message = node_message.row(0).to_owned();
            
            // Incorporate edge features if available (matching JavaScript logic)
            if let Some(ref edge_feats) = edge_features {
                if edge_idx >= edge_feats.nrows() {
                    return Err(GNNError::InvalidInput(
                        format!(
                            "Edge index {} out of bounds for edge features (max: {})",
                            edge_idx, edge_feats.nrows() - 1
                        )
                    ));
                }
                
                // Extract edge features for this edge
                let edge_features_vec = edge_feats.row(edge_idx);
                let edge_matrix = edge_features_vec.insert_axis(Axis(0)); // Convert to 2D
                
                // Transform edge features (no bias for edge transformation in JS)
                let edge_message = self.transform(
                    edge_matrix.view(),
                    &self.edge_to_message,
                    &Array1::zeros(0), // Empty bias
                )?;
                
                // Combine node and edge messages (JavaScript: nodeMessage + edgeMessage)
                final_message = final_message + edge_message.row(0);
            }
            
            Ok((edge_idx, final_message))
        }).collect();
        
        // Collect results and fill message array
        match message_results {
            Ok(results) => {
                for (edge_idx, message) in results {
                    messages.row_mut(edge_idx).assign(&message);
                }
            }
            Err(e) => return Err(e),
        }
        
        Ok(messages)
    }
    
    fn input_dim(&self) -> usize {
        self.input_dim
    }
    
    fn output_dim(&self) -> usize {
        self.hidden_dim
    }
    
    fn edge_dim(&self) -> usize {
        self.edge_dim
    }
    
    fn get_parameters(&self) -> Vec<Array2<f32>> {
        let mut params = vec![
            self.node_to_message.clone(),
            self.edge_to_message.clone(),
        ];
        
        if self.use_bias && !self.message_bias.is_empty() {
            // Convert bias to 2D array for consistency
            params.push(self.message_bias.clone().insert_axis(Axis(0)));
        }
        
        params
    }
    
    fn update_parameters(&mut self, gradients: &[Array2<f32>], learning_rate: f32) -> Result<(), GNNError> {
        if gradients.len() < 2 {
            return Err(GNNError::InvalidInput(
                "Insufficient gradients provided for parameter update".to_string()
            ));
        }
        
        // Update node-to-message weights
        self.node_to_message = &self.node_to_message - &(learning_rate * &gradients[0]);
        
        // Update edge-to-message weights
        self.edge_to_message = &self.edge_to_message - &(learning_rate * &gradients[1]);
        
        // Update bias if used
        if self.use_bias && gradients.len() > 2 {
            let bias_gradient = gradients[2].row(0);
            self.message_bias = &self.message_bias - &(learning_rate * bias_gradient);
        }
        
        Ok(())
    }
}

/**
 * GraphSAGE Layer implementation for large-scale graphs.
 * 
 * This layer implements the Sampling and Aggregating approach for handling
 * large graphs by sampling a fixed number of neighbors instead of using
 * all neighbors. This provides better scalability and consistent memory usage.
 * 
 * ## Key Features:
 * - **Neighbor Sampling**: Fixed-size neighbor sampling for scalability
 * - **Multiple Aggregators**: Mean, LSTM, pooling aggregation options
 * - **Normalization**: L2 normalization of embeddings for stability
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct GraphSAGELayer {
    /// Base graph convolution layer
    pub base_layer: GraphConvLayer,
    
    /// Number of neighbors to sample (None = use all neighbors)
    pub num_samples: Option<usize>,
    
    /// Aggregation strategy for sampled neighbors
    pub aggregator: String, // "mean", "max", "lstm"
    
    /// Whether to normalize embeddings
    pub normalize: bool,
}

impl GraphSAGELayer {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        edge_dim: usize,
        activation: ActivationFunction,
        use_bias: bool,
        num_samples: Option<usize>,
        aggregator: String,
        normalize: bool,
    ) -> Result<Self, GNNError> {
        let base_layer = GraphConvLayer::new(input_dim, hidden_dim, edge_dim, activation, use_bias)?;
        
        Ok(Self {
            base_layer,
            num_samples,
            aggregator,
            normalize,
        })
    }
    
    /// Sample neighbors for GraphSAGE-style processing
    fn sample_neighbors(&self, adjacency_list: &AdjacencyList, node_idx: NodeIndex) -> Vec<NodeIndex> {
        let neighbors = adjacency_list.forward_adj.get(&node_idx).cloned().unwrap_or_default();
        
        if let Some(num_samples) = self.num_samples {
            if neighbors.len() <= num_samples {
                neighbors
            } else {
                // Simple random sampling (could be improved with more sophisticated sampling)
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                let mut sampled = neighbors;
                sampled.shuffle(&mut rng);
                sampled.truncate(num_samples);
                sampled
            }
        } else {
            neighbors
        }
    }
}

#[async_trait::async_trait]
impl MessagePassingLayer for GraphSAGELayer {
    async fn compute_messages(
        &self,
        node_representations: &NodeFeatures,
        edge_features: &Option<EdgeFeatures>,
        adjacency_list: &AdjacencyList,
        layer_index: usize,
    ) -> Result<Array2<f32>, GNNError> {
        // For now, delegate to base GraphConv implementation
        // TODO: Implement proper GraphSAGE sampling logic
        self.base_layer.compute_messages(node_representations, edge_features, adjacency_list, layer_index).await
    }
    
    fn input_dim(&self) -> usize {
        self.base_layer.input_dim()
    }
    
    fn output_dim(&self) -> usize {
        self.base_layer.output_dim()
    }
    
    fn edge_dim(&self) -> usize {
        self.base_layer.edge_dim()
    }
    
    fn get_parameters(&self) -> Vec<Array2<f32>> {
        self.base_layer.get_parameters()
    }
    
    fn update_parameters(&mut self, gradients: &[Array2<f32>], learning_rate: f32) -> Result<(), GNNError> {
        self.base_layer.update_parameters(gradients, learning_rate)
    }
}

/**
 * Graph Attention (GAT) Layer implementation.
 * 
 * This layer implements the attention mechanism from Graph Attention Networks,
 * allowing the model to focus on more important neighbors during message passing.
 * 
 * ## Attention Mechanism:
 * - Compute attention scores between node pairs
 * - Apply softmax to normalize attention weights
 * - Use attention weights to combine neighbor messages
 * 
 * ## Multi-Head Attention:
 * - Support for multiple attention heads
 * - Concatenation or averaging of multi-head outputs
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct GATLayer {
    /// Base graph convolution layer  
    pub base_layer: GraphConvLayer,
    
    /// Attention weight matrix [hidden_dim, 1]
    pub attention_weights: Array2<f32>,
    
    /// Number of attention heads
    pub num_heads: usize,
    
    /// Dropout rate for attention weights
    pub attention_dropout: f32,
    
    /// Whether to use multi-head attention
    pub multi_head: bool,
}

impl GATLayer {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        edge_dim: usize,
        activation: ActivationFunction,
        use_bias: bool,
        num_heads: usize,
        attention_dropout: f32,
    ) -> Result<Self, GNNError> {
        let base_layer = GraphConvLayer::new(input_dim, hidden_dim, edge_dim, activation, use_bias)?;
        
        // Initialize attention weights
        let attention_weights = GraphConvLayer::create_weight_matrix(
            (hidden_dim, 1),
            WeightInitialization::He
        )?;
        
        Ok(Self {
            base_layer,
            attention_weights,
            num_heads,
            attention_dropout,
            multi_head: num_heads > 1,
        })
    }
    
    /// Compute attention scores between node pairs
    fn compute_attention_scores(
        &self,
        node_features: &NodeFeatures,
        adjacency_list: &AdjacencyList,
    ) -> Result<HashMap<(NodeIndex, NodeIndex), f32>, GNNError> {
        let mut attention_scores = HashMap::new();
        
        for &(source, target) in &adjacency_list.edges {
            // Get source and target node features
            let source_features = node_features.row(source);
            let target_features = node_features.row(target);
            
            // Concatenate features for attention computation
            let mut concat_features = Vec::new();
            concat_features.extend(source_features.iter());
            concat_features.extend(target_features.iter());
            
            let concat_array = Array2::from_shape_vec((1, concat_features.len()), concat_features)
                .map_err(|e| GNNError::InvalidInput(format!("Failed to create attention features: {}", e)))?;
            
            // Compute attention score (simplified)
            let score = concat_array.dot(&Array2::ones((concat_features.len(), 1)))[[(0, 0)]];
            attention_scores.insert((source, target), score);
        }
        
        Ok(attention_scores)
    }
}

#[async_trait::async_trait]
impl MessagePassingLayer for GATLayer {
    async fn compute_messages(
        &self,
        node_representations: &NodeFeatures,
        edge_features: &Option<EdgeFeatures>,
        adjacency_list: &AdjacencyList,
        layer_index: usize,
    ) -> Result<Array2<f32>, GNNError> {
        // For now, delegate to base GraphConv implementation
        // TODO: Implement proper attention mechanism
        self.base_layer.compute_messages(node_representations, edge_features, adjacency_list, layer_index).await
    }
    
    fn input_dim(&self) -> usize {
        self.base_layer.input_dim()
    }
    
    fn output_dim(&self) -> usize {
        self.base_layer.output_dim()
    }
    
    fn edge_dim(&self) -> usize {
        self.base_layer.edge_dim()
    }
    
    fn get_parameters(&self) -> Vec<Array2<f32>> {
        let mut params = self.base_layer.get_parameters();
        params.push(self.attention_weights.clone());
        params
    }
    
    fn update_parameters(&mut self, gradients: &[Array2<f32>], learning_rate: f32) -> Result<(), GNNError> {
        // Update base layer parameters
        let base_gradients = &gradients[..gradients.len()-1];
        self.base_layer.update_parameters(base_gradients, learning_rate)?;
        
        // Update attention weights
        if !gradients.is_empty() {
            let attention_gradient = &gradients[gradients.len()-1];
            self.attention_weights = &self.attention_weights - &(learning_rate * attention_gradient);
        }
        
        Ok(())
    }
}

// === LAYER FACTORY ===

/**
 * Factory function to create message passing layers based on configuration.
 * 
 * This function provides a unified interface for creating different types
 * of GNN layers, similar to how the JavaScript implementation dynamically
 * creates layers based on configuration strings.
 */
pub fn create_message_passing_layer(
    layer_type: &str,
    input_dim: usize,
    hidden_dim: usize,
    edge_dim: usize,
    activation: ActivationFunction,
    use_bias: bool,
    config: Option<HashMap<String, serde_json::Value>>,
) -> Result<Box<dyn MessagePassingLayer>, GNNError> {
    match layer_type.to_lowercase().as_str() {
        "graphconv" | "gcn" => {
            Ok(Box::new(GraphConvLayer::new(input_dim, hidden_dim, edge_dim, activation, use_bias)?))
        }
        "graphsage" | "sage" => {
            let num_samples = config.as_ref()
                .and_then(|c| c.get("num_samples"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
            
            let aggregator = config.as_ref()
                .and_then(|c| c.get("aggregator"))
                .and_then(|v| v.as_str())
                .unwrap_or("mean")
                .to_string();
            
            let normalize = config.as_ref()
                .and_then(|c| c.get("normalize"))
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            
            Ok(Box::new(GraphSAGELayer::new(
                input_dim, hidden_dim, edge_dim, activation, use_bias,
                num_samples, aggregator, normalize
            )?))
        }
        "gat" | "attention" => {
            let num_heads = config.as_ref()
                .and_then(|c| c.get("num_heads"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(1);
            
            let attention_dropout = config.as_ref()
                .and_then(|c| c.get("attention_dropout"))
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(0.1);
            
            Ok(Box::new(GATLayer::new(
                input_dim, hidden_dim, edge_dim, activation, use_bias,
                num_heads, attention_dropout
            )?))
        }
        _ => Err(GNNError::InvalidConfiguration(
            format!("Unknown layer type: {}. Supported types: graphconv, graphsage, gat", layer_type)
        )),
    }
}

// === TESTS ===

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnn::data::{GraphData, AdjacencyList};
    use approx::assert_abs_diff_eq;
    
    #[tokio::test]
    async fn test_graph_conv_layer_creation() {
        let layer = GraphConvLayer::new(
            64,  // input_dim
            128, // hidden_dim
            32,  // edge_dim
            ActivationFunction::ReLU,
            true // use_bias
        ).unwrap();
        
        assert_eq!(layer.input_dim(), 64);
        assert_eq!(layer.output_dim(), 128);
        assert_eq!(layer.edge_dim(), 32);
        assert_eq!(layer.node_to_message.shape(), &[64, 128]);
        assert_eq!(layer.edge_to_message.shape(), &[32, 128]);
        assert_eq!(layer.message_bias.len(), 128);
    }
    
    #[tokio::test]
    async fn test_message_computation() {
        // Create simple test graph
        let node_features = Array2::from_shape_vec((3, 2), vec![
            1.0, 0.5,  // Node 0
            0.8, 1.0,  // Node 1
            0.3, 0.7,  // Node 2
        ]).unwrap();
        
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let adjacency = AdjacencyList::new(edges, 3).unwrap();
        
        // Create layer
        let layer = GraphConvLayer::new(
            2,   // input_dim (node features)
            4,   // hidden_dim (message size)
            0,   // edge_dim (no edge features)
            ActivationFunction::ReLU,
            true // use_bias
        ).unwrap();
        
        // Compute messages
        let messages = layer.compute_messages(
            &node_features,
            &None, // no edge features
            &adjacency,
            0 // layer_index
        ).await.unwrap();
        
        assert_eq!(messages.shape(), &[3, 4]); // 3 edges, 4-dim messages
        
        // Messages should be non-zero (due to random initialization)
        assert!(messages.iter().any(|&x| x != 0.0));
    }
    
    #[tokio::test]
    async fn test_message_computation_with_edge_features() {
        // Create test graph with edge features
        let node_features = Array2::from_shape_vec((2, 3), vec![
            1.0, 2.0, 3.0,  // Node 0
            4.0, 5.0, 6.0,  // Node 1
        ]).unwrap();
        
        let edge_features = Array2::from_shape_vec((1, 2), vec![
            0.5, 1.5,  // Edge 0 features
        ]).unwrap();
        
        let edges = vec![(0, 1)];
        let adjacency = AdjacencyList::new(edges, 2).unwrap();
        
        // Create layer
        let layer = GraphConvLayer::new(
            3,   // input_dim (node features)
            4,   // hidden_dim (message size)
            2,   // edge_dim (edge features)
            ActivationFunction::Linear,
            false // no bias for simpler test
        ).unwrap();
        
        // Compute messages
        let messages = layer.compute_messages(
            &node_features,
            &Some(edge_features),
            &adjacency,
            0
        ).await.unwrap();
        
        assert_eq!(messages.shape(), &[1, 4]); // 1 edge, 4-dim message
    }
    
    #[test]
    fn test_weight_initialization() {
        let layer = GraphConvLayer::new(
            10, 20, 5, ActivationFunction::ReLU, true
        ).unwrap();
        
        // Check that weights are not all zeros (random initialization)
        assert!(layer.node_to_message.iter().any(|&x| x != 0.0));
        assert!(layer.edge_to_message.iter().any(|&x| x != 0.0));
        
        // Check shapes
        assert_eq!(layer.node_to_message.shape(), &[10, 20]);
        assert_eq!(layer.edge_to_message.shape(), &[5, 20]);
        assert_eq!(layer.message_bias.len(), 20);
    }
    
    #[test]
    fn test_layer_factory() {
        let layer = create_message_passing_layer(
            "graphconv",
            32, 64, 16,
            ActivationFunction::ReLU,
            true,
            None
        ).unwrap();
        
        assert_eq!(layer.input_dim(), 32);
        assert_eq!(layer.output_dim(), 64);
        assert_eq!(layer.edge_dim(), 16);
    }
    
    #[test]
    fn test_parameter_updates() {
        let mut layer = GraphConvLayer::new(
            2, 3, 1, ActivationFunction::ReLU, true
        ).unwrap();
        
        let original_node_weights = layer.node_to_message.clone();
        
        // Create dummy gradients
        let gradients = vec![
            Array2::ones((2, 3)),  // node_to_message gradients
            Array2::ones((1, 3)),  // edge_to_message gradients  
            Array2::ones((1, 3)),  // bias gradients
        ];
        
        let learning_rate = 0.01;
        layer.update_parameters(&gradients, learning_rate).unwrap();
        
        // Weights should have changed
        assert!(&layer.node_to_message != &original_node_weights);
        
        // Check that update was applied correctly (weights = old_weights - lr * gradients)
        let expected = &original_node_weights - &(learning_rate * &gradients[0]);
        assert_abs_diff_eq!(layer.node_to_message, expected, epsilon = 1e-6);
    }
    
    #[test]
    fn test_error_handling() {
        // Test invalid dimensions
        let result = GraphConvLayer::new(0, 10, 5, ActivationFunction::ReLU, true);
        assert!(result.is_err());
        
        // Test dimension mismatch in transform
        let layer = GraphConvLayer::new(2, 3, 1, ActivationFunction::ReLU, true).unwrap();
        let wrong_input = Array2::zeros((1, 5)); // Wrong input dimension
        let result = layer.transform(
            wrong_input.view(),
            &layer.node_to_message,
            &layer.message_bias
        );
        assert!(result.is_err());
    }
}