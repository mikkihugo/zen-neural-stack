/**
 * @file zen-neural/src/gnn/data.rs
 * @brief Graph Neural Network Data Structures
 * 
 * This module implements the core data structures for Graph Neural Networks,
 * providing efficient representations for graphs, nodes, edges, and adjacency
 * relationships. All structures are designed for high-performance computation
 * with SIMD vectorization support and zero-copy operations where possible.
 * 
 * ## Key Data Structures:
 * 
 * - **GraphData**: Complete graph representation with nodes, edges, and connectivity
 * - **NodeFeatures**: Node feature matrices with efficient tensor operations
 * - **EdgeFeatures**: Edge feature matrices for heterogeneous graphs
 * - **AdjacencyList**: Memory-efficient graph connectivity representation
 * - **GraphBatch**: Batched graph processing for training efficiency
 * 
 * ## Performance Features:
 * 
 * - **SIMD Operations**: Vectorized tensor computations using ndarray
 * - **Memory Layout**: Cache-friendly data layout for graph traversal
 * - **Zero-Copy Views**: Efficient sub-graph and batch operations
 * - **Parallel Processing**: Multi-threaded graph operations with rayon
 * 
 * @author Rust Neural Developer Agent (ruv-swarm)
 * @version 1.0.0-alpha.1
 * @since 2024-08-11
 * 
 * @see reference-implementations/js-neural-models/presets/gnn.js Original JavaScript implementation
 * @see crate::gnn::layers Message passing layer implementations
 */

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis, Ix2, s};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use num_traits::{Float, Zero};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::errors::ZenNeuralError;
use super::GNNError;

// === TYPE ALIASES FOR CLARITY ===

/// Node feature matrix type: [num_nodes, node_feature_dim]
pub type NodeFeatures = Array2<f32>;

/// Edge feature matrix type: [num_edges, edge_feature_dim]  
pub type EdgeFeatures = Array2<f32>;

/// Node indices type for adjacency lists
pub type NodeIndex = usize;

/// Edge indices type
pub type EdgeIndex = usize;

// === CORE DATA STRUCTURES ===

/**
 * Complete graph data structure containing all information needed for GNN processing.
 * 
 * This structure mirrors the JavaScript `graphData` object but with stronger typing
 * and memory-efficient representations. It supports both homogeneous and heterogeneous
 * graphs with optional edge features.
 * 
 * ## Memory Layout:
 * - Node features stored as contiguous f32 arrays for SIMD operations
 * - Adjacency list optimized for cache-friendly graph traversal
 * - Optional edge features aligned with adjacency structure
 * 
 * ## JavaScript Compatibility:
 * ```javascript
 * // JavaScript version:
 * const graphData = {
 *   nodes: new Float32Array([...]), // [numNodes * nodeFeatureDim]
 *   edges: new Float32Array([...]), // [numEdges * edgeFeatureDim] 
 *   adjacency: [[0,1], [1,2], [2,0]] // Array of [source, target] pairs
 * };
 * 
 * // Rust equivalent:
 * let graph_data = GraphData::new(node_features, edge_features, adjacency_list)?;
 * ```
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct GraphData {
    /// Node feature matrix: [num_nodes, node_feature_dim]
    /// Equivalent to JavaScript `nodes` Float32Array reshaped
    pub node_features: NodeFeatures,
    
    /// Edge feature matrix: [num_edges, edge_feature_dim]  
    /// Equivalent to JavaScript `edges` Float32Array reshaped
    pub edge_features: Option<EdgeFeatures>,
    
    /// Graph connectivity structure optimized for message passing
    pub adjacency_list: AdjacencyList,
    
    /// Optional graph-level features for graph classification tasks
    pub graph_features: Option<Array1<f32>>,
    
    /// Metadata for the graph (labels, weights, etc.)
    pub metadata: GraphMetadata,
}

/**
 * Efficient adjacency list representation for graph connectivity.
 * 
 * This structure provides multiple views of graph connectivity optimized
 * for different GNN operations:
 * - Forward neighbor lookup for message passing
 * - Backward neighbor lookup for gradient computation  
 * - Edge indexing for edge feature integration
 * 
 * ## Memory Efficiency:
 * The adjacency list uses compact representations and avoids redundant storage.
 * For undirected graphs, edges are stored once with bidirectional lookup tables.
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct AdjacencyList {
    /// List of edges as (source, target) pairs
    /// Equivalent to JavaScript `adjacency` array
    pub edges: Vec<(NodeIndex, NodeIndex)>,
    
    /// Forward adjacency: node_id -> list of outgoing neighbors
    /// Optimized for message passing from source to target
    pub forward_adj: HashMap<NodeIndex, Vec<NodeIndex>>,
    
    /// Backward adjacency: node_id -> list of incoming neighbors  
    /// Optimized for message aggregation at target nodes
    pub backward_adj: HashMap<NodeIndex, Vec<NodeIndex>>,
    
    /// Edge index mapping: (source, target) -> edge_index
    /// For efficient edge feature lookup during message computation
    pub edge_index: HashMap<(NodeIndex, NodeIndex), EdgeIndex>,
    
    /// Total number of nodes in the graph
    pub num_nodes: usize,
    
    /// Total number of edges in the graph
    pub num_edges: usize,
}

/**
 * Graph metadata containing auxiliary information for training and evaluation.
 * 
 * This structure stores labels, weights, and other metadata needed for different
 * graph learning tasks without cluttering the main graph representation.
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct GraphMetadata {
    /// Node-level labels for node classification tasks
    pub node_labels: Option<Vec<usize>>,
    
    /// Graph-level labels for graph classification tasks
    pub graph_labels: Option<Vec<usize>>,
    
    /// Edge-level labels for link prediction tasks
    pub edge_labels: Option<Vec<usize>>,
    
    /// Node weights for weighted loss computation
    pub node_weights: Option<Array1<f32>>,
    
    /// Edge weights for weighted message passing
    pub edge_weights: Option<Array1<f32>>,
    
    /// Additional string metadata
    pub properties: HashMap<String, String>,
}

/**
 * Batched graph data structure for efficient mini-batch training.
 * 
 * This structure allows processing multiple graphs simultaneously, which is
 * essential for efficient training. It handles graphs of different sizes by
 * using padding and masking strategies.
 * 
 * ## Batching Strategy:
 * - **Padding**: Smaller graphs padded to batch maximum size
 * - **Masking**: Invalid nodes/edges masked during computation
 * - **Memory Pool**: Reusable memory allocation for training efficiency
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct GraphBatch {
    /// Batched node features: [batch_size, max_nodes, node_feature_dim]
    pub node_features: Array3<f32>,
    
    /// Batched edge features: [batch_size, max_edges, edge_feature_dim]
    pub edge_features: Option<Array3<f32>>,
    
    /// Adjacency lists for each graph in the batch
    pub adjacency_lists: Vec<AdjacencyList>,
    
    /// Node masks indicating valid nodes: [batch_size, max_nodes]
    pub node_masks: Array2<bool>,
    
    /// Edge masks indicating valid edges: [batch_size, max_edges]
    pub edge_masks: Option<Array2<bool>>,
    
    /// Batch metadata
    pub metadata: Vec<GraphMetadata>,
    
    /// Batch size
    pub batch_size: usize,
    
    /// Maximum number of nodes across all graphs
    pub max_nodes: usize,
    
    /// Maximum number of edges across all graphs
    pub max_edges: usize,
}

// === IMPLEMENTATION ===

impl GraphData {
    /**
     * Create a new graph data structure with validation.
     * 
     * This constructor validates input dimensions and builds efficient
     * lookup structures for graph operations.
     * 
     * @param node_features Node feature matrix [num_nodes, node_feature_dim]
     * @param edge_features Optional edge features [num_edges, edge_feature_dim]
     * @param adjacency_edges List of (source, target) pairs defining connectivity
     * @return GraphData structure ready for GNN processing
     * 
     * # Examples
     * 
     * ```rust
     * use ndarray::Array2;
     * use zen_neural::gnn::data::GraphData;
     * 
     * // Create simple triangle graph
     * let node_features = Array2::from_shape_vec((3, 4), vec![
     *     1.0, 0.5, 0.2, 0.1,  // Node 0
     *     0.8, 1.0, 0.1, 0.3,  // Node 1  
     *     0.3, 0.7, 0.9, 0.5   // Node 2
     * ])?;
     * 
     * let adjacency = vec![(0, 1), (1, 2), (2, 0)];
     * let graph = GraphData::new(node_features, None, adjacency)?;
     * 
     * assert_eq!(graph.num_nodes(), 3);
     * assert_eq!(graph.num_edges(), 3);
     * ```
     */
    pub fn new(
        node_features: NodeFeatures,
        edge_features: Option<EdgeFeatures>,
        adjacency_edges: Vec<(NodeIndex, NodeIndex)>,
    ) -> Result<Self, GNNError> {
        let num_nodes = node_features.nrows();
        let num_edges = adjacency_edges.len();
        
        // Validate edge features dimensions if provided
        if let Some(ref edges) = edge_features {
            if edges.nrows() != num_edges {
                return Err(GNNError::DimensionMismatch(
                    format!(
                        "Edge feature count mismatch: expected {}, got {}. Each edge in the adjacency list must have corresponding features.",
                        num_edges, edges.nrows()
                    )
                ));
            }
        }
        
        // Validate node indices in adjacency list
        for (source, target) in &adjacency_edges {
            if *source >= num_nodes || *target >= num_nodes {
                return Err(GNNError::InvalidInput(
                    format!(
                        "Invalid edge ({}, {}): node indices must be in range [0, {}]. Check your adjacency list for out-of-bounds node references.",
                        source, target, num_nodes - 1
                    )
                ));
            }
        }
        
        // Build adjacency list structure
        let adjacency_list = AdjacencyList::new(adjacency_edges, num_nodes)?;
        
        Ok(Self {
            node_features,
            edge_features,
            adjacency_list,
            graph_features: None,
            metadata: GraphMetadata::default(),
        })
    }
    
    /**
     * Create a graph from JavaScript-style flat arrays.
     * 
     * This method provides compatibility with the original JavaScript implementation
     * by accepting flat Float32Array-equivalent vectors and reshaping them appropriately.
     * 
     * @param node_data Flat array of node features [num_nodes * node_feature_dim]
     * @param node_feature_dim Feature dimension for each node
     * @param edge_data Optional flat array of edge features
     * @param edge_feature_dim Feature dimension for each edge
     * @param adjacency_edges List of (source, target) connectivity pairs
     * @return GraphData structure equivalent to JavaScript version
     */
    pub fn from_flat_arrays(
        node_data: Vec<f32>,
        node_feature_dim: usize,
        edge_data: Option<Vec<f32>>,
        edge_feature_dim: usize,
        adjacency_edges: Vec<(NodeIndex, NodeIndex)>,
    ) -> Result<Self, GNNError> {
        // Validate and reshape node features (matching JS logic)
        if node_data.len() % node_feature_dim != 0 {
            return Err(GNNError::DimensionMismatch(
                format!(
                    "Node data length {} is not divisible by feature dimension {}. This suggests a mismatch between your data shape and the specified dimensions.",
                    node_data.len(), node_feature_dim
                )
            ));
        }
        
        let num_nodes = node_data.len() / node_feature_dim;
        let node_features = Array2::from_shape_vec((num_nodes, node_feature_dim), node_data)
            .map_err(|e| GNNError::InvalidInput(
                format!("Failed to reshape node features: {}. Check your input data format.", e)
            ))?;
        
        // Reshape edge features if provided
        let edge_features = if let Some(edge_data) = edge_data {
            if edge_data.len() % edge_feature_dim != 0 {
                return Err(GNNError::DimensionMismatch(
                    format!(
                        "Edge data length {} is not divisible by feature dimension {}",
                        edge_data.len(), edge_feature_dim
                    )
                ));
            }
            
            let num_edges_from_data = edge_data.len() / edge_feature_dim;
            if num_edges_from_data != adjacency_edges.len() {
                return Err(GNNError::DimensionMismatch(
                    format!(
                        "Edge data suggests {} edges but adjacency list has {}",
                        num_edges_from_data, adjacency_edges.len()
                    )
                ));
            }
            
            Some(Array2::from_shape_vec((num_edges_from_data, edge_feature_dim), edge_data)
                .map_err(|e| GNNError::InvalidInput(format!("Failed to reshape edge features: {}", e)))?)
        } else {
            None
        };
        
        Self::new(node_features, edge_features, adjacency_edges)
    }
    
    /// Get number of nodes in the graph
    pub fn num_nodes(&self) -> usize {
        self.node_features.nrows()
    }
    
    /// Get number of edges in the graph
    pub fn num_edges(&self) -> usize {
        self.adjacency_list.num_edges
    }
    
    /// Get node feature dimension
    pub fn node_feature_dim(&self) -> usize {
        self.node_features.ncols()
    }
    
    /// Get edge feature dimension (0 if no edge features)
    pub fn edge_feature_dim(&self) -> usize {
        self.edge_features.as_ref().map_or(0, |edges| edges.ncols())
    }
    
    /// Get node features for a specific node
    pub fn node_features_for(&self, node_idx: NodeIndex) -> Result<ArrayView1<f32>, GNNError> {
        if node_idx >= self.num_nodes() {
            return Err(GNNError::InvalidInput(
                format!("Node index {} out of bounds (max: {})", node_idx, self.num_nodes() - 1)
            ));
        }
        
        Ok(self.node_features.row(node_idx))
    }
    
    /// Get edge features for a specific edge
    pub fn edge_features_for(&self, edge_idx: EdgeIndex) -> Result<ArrayView1<f32>, GNNError> {
        match &self.edge_features {
            Some(features) => {
                if edge_idx >= features.nrows() {
                    return Err(GNNError::InvalidInput(
                        format!("Edge index {} out of bounds (max: {})", edge_idx, features.nrows() - 1)
                    ));
                }
                Ok(features.row(edge_idx))
            }
            None => Err(GNNError::InvalidInput("No edge features available".to_string())),
        }
    }
    
    /// Get neighbors of a node (outgoing edges)
    pub fn neighbors(&self, node_idx: NodeIndex) -> Vec<NodeIndex> {
        self.adjacency_list.forward_adj.get(&node_idx).cloned().unwrap_or_default()
    }
    
    /// Get incoming neighbors of a node (for undirected graphs, same as neighbors)
    pub fn incoming_neighbors(&self, node_idx: NodeIndex) -> Vec<NodeIndex> {
        self.adjacency_list.backward_adj.get(&node_idx).cloned().unwrap_or_default()
    }
    
    /// Get degree (number of outgoing edges) for a node
    pub fn degree(&self, node_idx: NodeIndex) -> usize {
        self.neighbors(node_idx).len()
    }
    
    /// Get edge index for a given (source, target) pair
    pub fn edge_index(&self, source: NodeIndex, target: NodeIndex) -> Option<EdgeIndex> {
        self.adjacency_list.edge_index.get(&(source, target)).copied()
    }
    
    /// Add graph-level features for graph classification tasks
    pub fn with_graph_features(mut self, graph_features: Array1<f32>) -> Self {
        self.graph_features = Some(graph_features);
        self
    }
    
    /// Add metadata to the graph
    pub fn with_metadata(mut self, metadata: GraphMetadata) -> Self {
        self.metadata = metadata;
        self
    }
    
    /// Clone the graph with a subset of nodes (subgraph extraction)
    pub fn subgraph(&self, node_indices: &[NodeIndex]) -> Result<GraphData, GNNError> {
        let node_set: HashSet<_> = node_indices.iter().collect();
        
        // Extract node features for the subgraph
        let mut subgraph_node_features = Vec::new();
        let mut node_mapping = HashMap::new();
        
        for (new_idx, &old_idx) in node_indices.iter().enumerate() {
            if old_idx >= self.num_nodes() {
                return Err(GNNError::InvalidInput(
                    format!("Node index {} out of bounds for subgraph extraction", old_idx)
                ));
            }
            
            node_mapping.insert(old_idx, new_idx);
            
            // Copy node features
            let node_features = self.node_features.row(old_idx);
            subgraph_node_features.extend(node_features.iter());
        }
        
        let node_feature_dim = self.node_feature_dim();
        let subgraph_nodes = Array2::from_shape_vec(
            (node_indices.len(), node_feature_dim),
            subgraph_node_features
        ).map_err(|e| GNNError::InvalidInput(format!("Failed to create subgraph node features: {}", e)))?;
        
        // Extract edges that exist within the subgraph
        let mut subgraph_edges = Vec::new();
        let mut subgraph_edge_features = Vec::new();
        
        for (edge_idx, &(source, target)) in self.adjacency_list.edges.iter().enumerate() {
            if node_set.contains(&source) && node_set.contains(&target) {
                let new_source = node_mapping[&source];
                let new_target = node_mapping[&target];
                subgraph_edges.push((new_source, new_target));
                
                // Copy edge features if they exist
                if let Some(ref edge_features) = self.edge_features {
                    let edge_row = edge_features.row(edge_idx);
                    subgraph_edge_features.extend(edge_row.iter());
                }
            }
        }
        
        let subgraph_edge_features = if !subgraph_edge_features.is_empty() {
            let edge_feature_dim = self.edge_feature_dim();
            Some(Array2::from_shape_vec(
                (subgraph_edges.len(), edge_feature_dim),
                subgraph_edge_features
            ).map_err(|e| GNNError::InvalidInput(format!("Failed to create subgraph edge features: {}", e)))?)
        } else {
            None
        };
        
        Self::new(subgraph_nodes, subgraph_edge_features, subgraph_edges)
    }
}

impl AdjacencyList {
    /**
     * Create a new adjacency list from edge list.
     * 
     * This method builds efficient forward and backward lookup tables
     * for fast neighbor queries during message passing.
     * 
     * @param edges List of (source, target) pairs
     * @param num_nodes Total number of nodes in the graph
     * @return AdjacencyList with optimized lookup structures
     */
    pub fn new(edges: Vec<(NodeIndex, NodeIndex)>, num_nodes: usize) -> Result<Self, GNNError> {
        let num_edges = edges.len();
        
        // Build forward adjacency (outgoing edges)
        let mut forward_adj: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
        
        // Build backward adjacency (incoming edges)
        let mut backward_adj: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
        
        // Build edge index mapping
        let mut edge_index: HashMap<(NodeIndex, NodeIndex), EdgeIndex> = HashMap::new();
        
        for (edge_idx, &(source, target)) in edges.iter().enumerate() {
            // Forward adjacency
            forward_adj.entry(source).or_default().push(target);
            
            // Backward adjacency  
            backward_adj.entry(target).or_default().push(source);
            
            // Edge index mapping
            edge_index.insert((source, target), edge_idx);
        }
        
        Ok(Self {
            edges,
            forward_adj,
            backward_adj,
            edge_index,
            num_nodes,
            num_edges,
        })
    }
    
    /// Get maximum node ID referenced in the adjacency list
    pub fn max_node_id(&self) -> Option<NodeIndex> {
        self.edges.iter()
            .flat_map(|(source, target)| [*source, *target])
            .max()
    }
    
    /// Check if the graph is undirected (every edge has a reverse edge)
    pub fn is_undirected(&self) -> bool {
        self.edges.iter().all(|&(source, target)| {
            self.edge_index.contains_key(&(target, source))
        })
    }
    
    /// Get all unique node indices in the graph
    pub fn node_indices(&self) -> Vec<NodeIndex> {
        let mut nodes: Vec<_> = self.edges.iter()
            .flat_map(|(source, target)| [*source, *target])
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        
        nodes.sort();
        nodes
    }
    
    /// Convert to edge list format (compatible with JavaScript)
    pub fn to_edge_list(&self) -> Vec<(NodeIndex, NodeIndex)> {
        self.edges.clone()
    }
}

impl GraphBatch {
    /**
     * Create a batched graph structure from individual graphs.
     * 
     * This method handles graphs of different sizes by padding smaller graphs
     * and creating appropriate masks for valid computation.
     * 
     * @param graphs Vector of individual graph data structures
     * @return Batched graph ready for efficient mini-batch processing
     */
    pub fn from_graphs(graphs: Vec<GraphData>) -> Result<Self, GNNError> {
        if graphs.is_empty() {
            return Err(GNNError::InvalidInput("Cannot create batch from empty graph list".to_string()));
        }
        
        let batch_size = graphs.len();
        
        // Find maximum dimensions across all graphs
        let max_nodes = graphs.iter().map(|g| g.num_nodes()).max().unwrap();
        let max_edges = graphs.iter().map(|g| g.num_edges()).max().unwrap();
        
        let node_feature_dim = graphs[0].node_feature_dim();
        let edge_feature_dim = graphs[0].edge_feature_dim();
        
        // Validate consistent feature dimensions
        for graph in &graphs {
            if graph.node_feature_dim() != node_feature_dim {
                return Err(GNNError::DimensionMismatch(
                    format!(
                        "Inconsistent node feature dimensions in batch: expected {}, got {}",
                        node_feature_dim, graph.node_feature_dim()
                    )
                ));
            }
            
            if graph.edge_feature_dim() != edge_feature_dim {
                return Err(GNNError::DimensionMismatch(
                    format!(
                        "Inconsistent edge feature dimensions in batch: expected {}, got {}",
                        edge_feature_dim, graph.edge_feature_dim()
                    )
                ));
            }
        }
        
        // Create batched node features with padding
        let mut batched_node_features = Array3::zeros((batch_size, max_nodes, node_feature_dim));
        let mut node_masks = Array2::from_elem((batch_size, max_nodes), false);
        
        // Create batched edge features with padding if needed
        let mut batched_edge_features = if edge_feature_dim > 0 {
            Some(Array3::zeros((batch_size, max_edges, edge_feature_dim)))
        } else {
            None
        };
        let mut edge_masks = if edge_feature_dim > 0 {
            Some(Array2::from_elem((batch_size, max_edges), false))
        } else {
            None
        };
        
        let mut adjacency_lists = Vec::with_capacity(batch_size);
        let mut metadata = Vec::with_capacity(batch_size);
        
        // Fill batched arrays
        for (batch_idx, graph) in graphs.iter().enumerate() {
            let num_nodes = graph.num_nodes();
            let num_edges = graph.num_edges();
            
            // Copy node features and set mask
            batched_node_features.slice_mut(s![batch_idx, 0..num_nodes, ..])
                .assign(&graph.node_features);
            node_masks.slice_mut(s![batch_idx, 0..num_nodes]).fill(true);
            
            // Copy edge features and set mask if present
            if let (Some(ref mut batched_edges), Some(ref graph_edges)) = 
                (&mut batched_edge_features, &graph.edge_features) {
                batched_edges.slice_mut(s![batch_idx, 0..num_edges, ..])
                    .assign(graph_edges);
                
                if let Some(ref mut edge_mask) = edge_masks {
                    edge_mask.slice_mut(s![batch_idx, 0..num_edges]).fill(true);
                }
            }
            
            // Store adjacency list and metadata
            adjacency_lists.push(graph.adjacency_list.clone());
            metadata.push(graph.metadata.clone());
        }
        
        Ok(Self {
            node_features: batched_node_features,
            edge_features: batched_edge_features,
            adjacency_lists,
            node_masks,
            edge_masks,
            metadata,
            batch_size,
            max_nodes,
            max_edges,
        })
    }
    
    /// Extract individual graph from batch
    pub fn get_graph(&self, batch_idx: usize) -> Result<GraphData, GNNError> {
        if batch_idx >= self.batch_size {
            return Err(GNNError::InvalidInput(
                format!("Batch index {} out of bounds (batch size: {})", batch_idx, self.batch_size)
            ));
        }
        
        // Extract valid nodes based on mask
        let node_mask = self.node_masks.row(batch_idx);
        let num_valid_nodes = node_mask.iter().filter(|&&x| x).count();
        
        // Copy valid node features
        let mut node_features_vec = Vec::new();
        let node_feature_dim = self.node_features.shape()[2];
        
        for node_idx in 0..num_valid_nodes {
            let node_features = self.node_features.slice(s![batch_idx, node_idx, ..]);
            node_features_vec.extend(node_features.iter());
        }
        
        let node_features = Array2::from_shape_vec(
            (num_valid_nodes, node_feature_dim),
            node_features_vec
        ).map_err(|e| GNNError::InvalidInput(format!("Failed to extract node features: {}", e)))?;
        
        // Extract edge features if present
        let edge_features = if let Some(ref batched_edges) = self.edge_features {
            let edge_mask = self.edge_masks.as_ref().unwrap().row(batch_idx);
            let num_valid_edges = edge_mask.iter().filter(|&&x| x).count();
            
            let mut edge_features_vec = Vec::new();
            let edge_feature_dim = batched_edges.shape()[2];
            
            for edge_idx in 0..num_valid_edges {
                let edge_features = batched_edges.slice(s![batch_idx, edge_idx, ..]);
                edge_features_vec.extend(edge_features.iter());
            }
            
            Some(Array2::from_shape_vec(
                (num_valid_edges, edge_feature_dim),
                edge_features_vec
            ).map_err(|e| GNNError::InvalidInput(format!("Failed to extract edge features: {}", e)))?)
        } else {
            None
        };
        
        Ok(GraphData {
            node_features,
            edge_features,
            adjacency_list: self.adjacency_lists[batch_idx].clone(),
            graph_features: None,
            metadata: self.metadata[batch_idx].clone(),
        })
    }
}

// === UTILITY FUNCTIONS ===

/**
 * Generate a random graph for testing and benchmarking.
 * 
 * This function creates synthetic graphs with configurable parameters,
 * useful for testing GNN implementations and performance benchmarking.
 */
pub fn generate_random_graph(
    num_nodes: usize,
    num_edges: usize,
    node_feature_dim: usize,
    edge_feature_dim: usize,
    seed: Option<u64>,
) -> Result<GraphData, GNNError> {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    
    let mut rng = if let Some(seed) = seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_entropy()
    };
    
    // Generate random node features
    let node_features = Array2::random_using(
        (num_nodes, node_feature_dim),
        Uniform::new(-1.0f32, 1.0f32),
        &mut rng
    );
    
    // Generate random edges (avoiding self-loops and duplicates)
    let mut edges = Vec::new();
    let mut edge_set = HashSet::new();
    
    while edges.len() < num_edges && edge_set.len() < num_nodes * (num_nodes - 1) {
        let source = rng.gen_range(0..num_nodes);
        let target = rng.gen_range(0..num_nodes);
        
        if source != target && !edge_set.contains(&(source, target)) {
            edges.push((source, target));
            edge_set.insert((source, target));
        }
    }
    
    // Generate random edge features if requested
    let edge_features = if edge_feature_dim > 0 {
        Some(Array2::random_using(
            (edges.len(), edge_feature_dim),
            Uniform::new(-1.0f32, 1.0f32),
            &mut rng
        ))
    } else {
        None
    };
    
    GraphData::new(node_features, edge_features, edges)
}

// === TESTS ===

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_graph_data_creation() {
        let node_features = Array2::from_shape_vec((3, 2), vec![
            1.0, 2.0,  // Node 0
            3.0, 4.0,  // Node 1
            5.0, 6.0,  // Node 2
        ]).unwrap();
        
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let graph = GraphData::new(node_features, None, edges).unwrap();
        
        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(graph.num_edges(), 3);
        assert_eq!(graph.node_feature_dim(), 2);
        assert_eq!(graph.edge_feature_dim(), 0);
    }
    
    #[test]
    fn test_adjacency_list_functionality() {
        let edges = vec![(0, 1), (1, 2), (2, 0), (1, 0)]; // Mixed directed edges
        let adj = AdjacencyList::new(edges, 3).unwrap();
        
        assert_eq!(adj.forward_adj[&0], vec![1]);
        assert_eq!(adj.forward_adj[&1], vec![2, 0]);
        assert_eq!(adj.backward_adj[&1], vec![0, 1]);
        
        assert_eq!(adj.edge_index[&(0, 1)], 0);
        assert_eq!(adj.edge_index[&(1, 2)], 1);
        assert_eq!(adj.max_node_id(), Some(2));
    }
    
    #[test]
    fn test_from_flat_arrays() {
        let node_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 nodes, 2 features each
        let edges = vec![(0, 1), (1, 2)];
        
        let graph = GraphData::from_flat_arrays(
            node_data, 2, None, 0, edges
        ).unwrap();
        
        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(graph.node_feature_dim(), 2);
        
        let node_0 = graph.node_features_for(0).unwrap();
        assert_abs_diff_eq!(node_0[0], 1.0);
        assert_abs_diff_eq!(node_0[1], 2.0);
    }
    
    #[test]
    fn test_graph_batch_creation() {
        // Create two small graphs
        let graph1 = GraphData::from_flat_arrays(
            vec![1.0, 2.0, 3.0, 4.0], // 2 nodes, 2 features
            2, None, 0,
            vec![(0, 1)]
        ).unwrap();
        
        let graph2 = GraphData::from_flat_arrays(
            vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], // 3 nodes, 2 features  
            2, None, 0,
            vec![(0, 1), (1, 2)]
        ).unwrap();
        
        let batch = GraphBatch::from_graphs(vec![graph1, graph2]).unwrap();
        
        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.max_nodes, 3);
        assert_eq!(batch.max_edges, 2);
        
        // Test node masks
        assert!(batch.node_masks[[0, 0]]); // Graph 0, Node 0: valid
        assert!(batch.node_masks[[0, 1]]); // Graph 0, Node 1: valid
        assert!(!batch.node_masks[[0, 2]]); // Graph 0, Node 2: padded
        
        assert!(batch.node_masks[[1, 2]]); // Graph 1, Node 2: valid
    }
    
    #[test]
    fn test_subgraph_extraction() {
        let node_features = Array2::from_shape_vec((4, 2), vec![
            1.0, 2.0,  // Node 0
            3.0, 4.0,  // Node 1  
            5.0, 6.0,  // Node 2
            7.0, 8.0,  // Node 3
        ]).unwrap();
        
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        let graph = GraphData::new(node_features, None, edges).unwrap();
        
        // Extract subgraph with nodes 1 and 2
        let subgraph = graph.subgraph(&[1, 2]).unwrap();
        
        assert_eq!(subgraph.num_nodes(), 2);
        assert_eq!(subgraph.num_edges(), 1); // Only edge (1,2) maps to (0,1)
        
        let node_0 = subgraph.node_features_for(0).unwrap();
        assert_abs_diff_eq!(node_0[0], 3.0); // Original node 1 features
        assert_abs_diff_eq!(node_0[1], 4.0);
    }
    
    #[test]
    fn test_random_graph_generation() {
        let graph = generate_random_graph(10, 15, 32, 16, Some(42)).unwrap();
        
        assert_eq!(graph.num_nodes(), 10);
        assert_eq!(graph.num_edges(), 15);
        assert_eq!(graph.node_feature_dim(), 32);
        assert_eq!(graph.edge_feature_dim(), 16);
        
        // Test reproducibility with same seed
        let graph2 = generate_random_graph(10, 15, 32, 16, Some(42)).unwrap();
        assert_eq!(graph.node_features, graph2.node_features);
    }
    
    #[test]
    fn test_error_handling() {
        // Test dimension mismatch
        let node_features = Array2::zeros((3, 2));
        let edge_features = Array2::zeros((5, 3)); // Wrong number of edges
        let edges = vec![(0, 1), (1, 2)]; // Only 2 edges
        
        let result = GraphData::new(node_features, Some(edge_features), edges);
        assert!(result.is_err());
        
        // Test invalid node indices
        let node_features = Array2::zeros((2, 2));
        let edges = vec![(0, 3)]; // Node 3 doesn't exist
        
        let result = GraphData::new(node_features, None, edges);
        assert!(result.is_err());
    }
}