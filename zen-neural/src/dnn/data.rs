/**
 * @file zen-neural/src/dnn/data.rs
 * @brief Core data structures and tensor operations for DNN
 * 
 * This module implements the foundational data types for Deep Neural Networks,
 * including tensor representations, batch handling, and memory-efficient operations.
 * Built on ndarray for SIMD acceleration while maintaining compatibility with
 * JavaScript tensor patterns.
 * 
 * ## Core Components:
 * - **DNNTensor**: High-performance tensor wrapper around ndarray
 * - **TensorShape**: Shape validation and manipulation utilities
 * - **BatchData**: Efficient batch processing for training
 * - **TensorOps**: Common tensor operations optimized for neural networks
 * 
 * ## JavaScript Compatibility:
 * Maintains familiar tensor operations while adding type safety and performance.
 * 
 * @author DNN Core Developer Agent (ruv-swarm Phase 1)
 * @version 1.0.0-alpha.1
 * @since 2025-01-14
 */

use ndarray::{Array1, Array2, Array3, ArrayD, Axis, Dimension, IxDyn};
use num_traits::{Float, Zero, One};
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::DNNError;

// === CORE TENSOR TYPES ===

/**
 * High-performance tensor implementation for DNN operations.
 * 
 * This wraps ndarray::Array2<f32> with additional functionality specific
 * to neural network operations, including shape validation, batch handling,
 * and memory optimization.
 */
#[derive(Debug, Clone)]
pub struct DNNTensor {
    /// Underlying ndarray data (always 2D: [batch_size, features])
    pub data: Array2<f32>,
    /// Cached shape information for performance
    shape_cache: TensorShape,
    /// Whether tensor requires gradient computation
    requires_grad: bool,
}

impl DNNTensor {
    /// Create a new tensor from 2D array data
    pub fn new(data: Array2<f32>) -> Result<Self, DNNError> {
        let shape = TensorShape::new(data.dim().into_pattern().into());
        Ok(Self {
            shape_cache: shape,
            data,
            requires_grad: false,
        })
    }
    
    /// Create a tensor filled with zeros
    pub fn zeros(shape: &TensorShape) -> Result<Self, DNNError> {
        if shape.dims.len() != 2 {
            return Err(DNNError::InvalidInput(
                format!("DNNTensor only supports 2D tensors, got shape: {:?}", shape.dims)
            ));
        }
        
        let data = Array2::zeros((shape.dims[0], shape.dims[1]));
        Ok(Self {
            data,
            shape_cache: shape.clone(),
            requires_grad: false,
        })
    }
    
    /// Create a tensor filled with ones
    pub fn ones(shape: &TensorShape) -> Result<Self, DNNError> {
        if shape.dims.len() != 2 {
            return Err(DNNError::InvalidInput(
                format!("DNNTensor only supports 2D tensors, got shape: {:?}", shape.dims)
            ));
        }
        
        let data = Array2::ones((shape.dims[0], shape.dims[1]));
        Ok(Self {
            data,
            shape_cache: shape.clone(),
            requires_grad: false,
        })
    }
    
    /// Create a tensor from raw vector data with specified shape
    pub fn from_vec(vec_data: Vec<f32>, shape: &TensorShape) -> Result<Self, DNNError> {
        if shape.dims.len() != 2 {
            return Err(DNNError::InvalidInput(
                format!("DNNTensor only supports 2D tensors, got shape: {:?}", shape.dims)
            ));
        }
        
        let expected_len = shape.total_elements();
        if vec_data.len() != expected_len {
            return Err(DNNError::DimensionMismatch(
                format!("Vector length {} doesn't match shape {:?} (expected {})", 
                       vec_data.len(), shape.dims, expected_len)
            ));
        }
        
        let data = Array2::from_shape_vec((shape.dims[0], shape.dims[1]), vec_data)
            .map_err(|e| DNNError::InvalidInput(format!("Failed to create tensor: {}", e)))?;
        
        Ok(Self {
            data,
            shape_cache: shape.clone(),
            requires_grad: false,
        })
    }
    
    /// Get tensor shape
    pub fn shape(&self) -> &TensorShape {
        &self.shape_cache
    }
    
    /// Get batch size (first dimension)
    pub fn batch_size(&self) -> usize {
        self.shape_cache.dims[0]
    }
    
    /// Get feature dimension (second dimension)
    pub fn feature_dim(&self) -> usize {
        self.shape_cache.dims[1]
    }
    
    /// Check if tensor contains NaN or infinite values
    pub fn has_invalid_values(&self) -> bool {
        self.data.iter().any(|&x| !x.is_finite())
    }
    
    /// Set gradient requirement
    pub fn requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }
    
    /// Check if gradients are required
    pub fn requires_gradient(&self) -> bool {
        self.requires_grad
    }
    
    /// Get a view of the underlying data
    pub fn view(&self) -> ndarray::ArrayView2<f32> {
        self.data.view()
    }
    
    /// Get a mutable view of the underlying data
    pub fn view_mut(&mut self) -> ndarray::ArrayViewMut2<f32> {
        self.data.view_mut()
    }
    
    /// Convert to owned vector (flattened)
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.iter().cloned().collect()
    }
    
    /// Reshape tensor (must preserve total elements)
    pub fn reshape(&self, new_shape: &TensorShape) -> Result<Self, DNNError> {
        if new_shape.total_elements() != self.shape_cache.total_elements() {
            return Err(DNNError::DimensionMismatch(
                format!("Cannot reshape tensor with {} elements to shape with {} elements",
                       self.shape_cache.total_elements(), new_shape.total_elements())
            ));
        }
        
        let reshaped_data = self.data.clone().into_shape((new_shape.dims[0], new_shape.dims[1]))
            .map_err(|e| DNNError::InvalidInput(format!("Reshape failed: {}", e)))?;
        
        Ok(Self {
            data: reshaped_data,
            shape_cache: new_shape.clone(),
            requires_grad: self.requires_grad,
        })
    }
}

// === TENSOR SHAPE UTILITIES ===

/**
 * Tensor shape representation with validation and manipulation utilities.
 * 
 * For DNNs, we primarily work with 2D tensors [batch_size, features].
 * This provides shape validation and common operations.
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    /// Shape dimensions (always length 2 for DNN tensors)
    pub dims: Vec<usize>,
}

impl TensorShape {
    /// Create a new tensor shape
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }
    
    /// Create a 2D tensor shape
    pub fn new_2d(batch_size: usize, features: usize) -> Self {
        Self {
            dims: vec![batch_size, features],
        }
    }
    
    /// Get total number of elements
    pub fn total_elements(&self) -> usize {
        self.dims.iter().product()
    }
    
    /// Check if this shape is compatible with another for matrix operations
    pub fn is_compatible_for_matmul(&self, other: &TensorShape) -> bool {
        self.dims.len() == 2 && other.dims.len() == 2 && self.dims[1] == other.dims[0]
    }
    
    /// Get output shape for matrix multiplication
    pub fn matmul_output_shape(&self, other: &TensorShape) -> Result<TensorShape, DNNError> {
        if !self.is_compatible_for_matmul(other) {
            return Err(DNNError::DimensionMismatch(
                format!("Incompatible shapes for matrix multiplication: {:?} x {:?}", 
                       self.dims, other.dims)
            ));
        }
        
        Ok(TensorShape::new_2d(self.dims[0], other.dims[1]))
    }
    
    /// Check if shapes are broadcastable
    pub fn is_broadcastable_with(&self, other: &TensorShape) -> bool {
        if self.dims.len() != 2 || other.dims.len() != 2 {
            return false;
        }
        
        // For bias addition: [batch, features] + [features] or [1, features]
        other.dims[0] == 1 || other.dims[0] == self.dims[0]
    }
}

impl fmt::Display for TensorShape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.dims)
    }
}

// === BATCH DATA HANDLING ===

/**
 * Efficient batch data representation for training.
 * 
 * Handles batching of training samples with memory optimization
 * and parallel processing capabilities.
 */
#[derive(Debug, Clone)]
pub struct BatchData {
    /// Input tensor batch [batch_size, input_features]
    pub inputs: DNNTensor,
    /// Target tensor batch [batch_size, output_features]  
    pub targets: DNNTensor,
    /// Batch metadata
    pub metadata: BatchMetadata,
}

#[derive(Debug, Clone)]
pub struct BatchMetadata {
    /// Batch index in epoch
    pub batch_idx: usize,
    /// Total number of batches in epoch
    pub total_batches: usize,
    /// Epoch number
    pub epoch: u32,
    /// Sample indices in this batch
    pub sample_indices: Vec<usize>,
}

impl BatchData {
    /// Create a new batch
    pub fn new(inputs: DNNTensor, targets: DNNTensor, metadata: BatchMetadata) -> Result<Self, DNNError> {
        // Validate batch dimensions match
        if inputs.batch_size() != targets.batch_size() {
            return Err(DNNError::DimensionMismatch(
                format!("Input batch size {} doesn't match target batch size {}", 
                       inputs.batch_size(), targets.batch_size())
            ));
        }
        
        Ok(Self {
            inputs,
            targets,
            metadata,
        })
    }
    
    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.inputs.batch_size()
    }
    
    /// Shuffle the batch data
    pub fn shuffle(&mut self) {
        // For now, just shuffle indices - actual data shuffling would be more complex
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        self.metadata.sample_indices.shuffle(&mut rng);
    }
}

// === TENSOR OPERATIONS ===

/**
 * Common tensor operations optimized for neural networks.
 * 
 * These operations are frequently used in DNN forward/backward passes
 * and are optimized for performance with SIMD and parallel processing.
 */
pub struct TensorOps;

impl TensorOps {
    /// Matrix multiplication with bias addition (core dense layer operation)
    /// 
    /// Equivalent to JavaScript: output = input @ weight + bias
    /// Optimized using ndarray's BLAS backend for SIMD acceleration
    pub fn dense_forward(
        input: &DNNTensor,
        weight: &Array2<f32>,
        bias: Option<&Array1<f32>>
    ) -> Result<DNNTensor, DNNError> {
        // Validate dimensions
        if input.feature_dim() != weight.nrows() {
            return Err(DNNError::DimensionMismatch(
                format!("Input features {} don't match weight input dimension {}",
                       input.feature_dim(), weight.nrows())
            ));
        }
        
        // Matrix multiplication: [batch_size, input_features] @ [input_features, output_features]
        let mut output = input.data.dot(weight);
        
        // Add bias if provided: broadcasting [batch_size, output_features] + [output_features]
        if let Some(bias) = bias {
            if bias.len() != weight.ncols() {
                return Err(DNNError::DimensionMismatch(
                    format!("Bias dimension {} doesn't match weight output dimension {}",
                           bias.len(), weight.ncols())
                ));
            }
            
            // ndarray automatically broadcasts bias across batch dimension
            output = output + bias;
        }
        
        DNNTensor::new(output)
    }
    
    /// Element-wise addition with broadcasting
    pub fn add_broadcast(a: &DNNTensor, b: &DNNTensor) -> Result<DNNTensor, DNNError> {
        if !a.shape().is_broadcastable_with(b.shape()) {
            return Err(DNNError::DimensionMismatch(
                format!("Shapes {:?} and {:?} are not broadcastable", 
                       a.shape().dims, b.shape().dims)
            ));
        }
        
        let result = &a.data + &b.data;
        DNNTensor::new(result)
    }
    
    /// Element-wise multiplication
    pub fn multiply(a: &DNNTensor, b: &DNNTensor) -> Result<DNNTensor, DNNError> {
        if a.shape() != b.shape() {
            return Err(DNNError::DimensionMismatch(
                format!("Shapes {:?} and {:?} don't match for element-wise multiplication",
                       a.shape().dims, b.shape().dims)
            ));
        }
        
        let result = &a.data * &b.data;
        DNNTensor::new(result)
    }
    
    /// Sum along axis (for loss computation and gradients)
    pub fn sum_axis(tensor: &DNNTensor, axis: usize) -> Result<Array1<f32>, DNNError> {
        if axis >= tensor.shape().dims.len() {
            return Err(DNNError::InvalidInput(
                format!("Axis {} out of bounds for tensor with {} dimensions", 
                       axis, tensor.shape().dims.len())
            ));
        }
        
        let axis_ndarray = Axis(axis);
        Ok(tensor.data.sum_axis(axis_ndarray))
    }
    
    /// Mean along axis
    pub fn mean_axis(tensor: &DNNTensor, axis: usize) -> Result<Array1<f32>, DNNError> {
        if axis >= tensor.shape().dims.len() {
            return Err(DNNError::InvalidInput(
                format!("Axis {} out of bounds for tensor with {} dimensions", 
                       axis, tensor.shape().dims.len())
            ));
        }
        
        let axis_ndarray = Axis(axis);
        let mean = tensor.data.mean_axis(axis_ndarray).ok_or_else(|| {
            DNNError::InvalidInput("Failed to compute mean - empty tensor?".to_string())
        })?;
        Ok(mean)
    }
    
    /// Apply function element-wise (for activations)
    pub fn apply_elementwise<F>(tensor: &DNNTensor, func: F) -> Result<DNNTensor, DNNError>
    where
        F: Fn(f32) -> f32,
    {
        let result = tensor.data.mapv(func);
        DNNTensor::new(result)
    }
    
    /// Transpose tensor (swap batch and feature dimensions)
    pub fn transpose(tensor: &DNNTensor) -> Result<DNNTensor, DNNError> {
        let transposed = tensor.data.t().to_owned();
        let new_shape = TensorShape::new(vec![tensor.shape().dims[1], tensor.shape().dims[0]]);
        
        Ok(DNNTensor {
            data: transposed,
            shape_cache: new_shape,
            requires_grad: tensor.requires_grad,
        })
    }
    
    /// Concatenate tensors along batch dimension
    pub fn concat_batch(tensors: &[DNNTensor]) -> Result<DNNTensor, DNNError> {
        if tensors.is_empty() {
            return Err(DNNError::InvalidInput("Cannot concatenate empty tensor list".to_string()));
        }
        
        // Verify all tensors have the same feature dimension
        let feature_dim = tensors[0].feature_dim();
        for tensor in tensors.iter() {
            if tensor.feature_dim() != feature_dim {
                return Err(DNNError::DimensionMismatch(
                    format!("All tensors must have same feature dimension for batch concatenation")
                ));
            }
        }
        
        // Collect views and concatenate
        let views: Vec<_> = tensors.iter().map(|t| t.data.view()).collect();
        let concatenated = ndarray::concatenate(Axis(0), &views)
            .map_err(|e| DNNError::InvalidInput(format!("Concatenation failed: {}", e)))?;
        
        DNNTensor::new(concatenated)
    }
    
    /// Create batches from a dataset
    pub fn create_batches(
        inputs: &DNNTensor,
        targets: &DNNTensor,
        batch_size: usize,
        shuffle: bool,
    ) -> Result<Vec<BatchData>, DNNError> {
        let total_samples = inputs.batch_size();
        if total_samples != targets.batch_size() {
            return Err(DNNError::DimensionMismatch(
                format!("Input samples {} don't match target samples {}", 
                       total_samples, targets.batch_size())
            ));
        }
        
        if batch_size == 0 {
            return Err(DNNError::InvalidInput("Batch size must be greater than 0".to_string()));
        }
        
        // Create sample indices
        let mut indices: Vec<usize> = (0..total_samples).collect();
        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }
        
        // Create batches
        let mut batches = Vec::new();
        let total_batches = (total_samples + batch_size - 1) / batch_size; // Ceiling division
        
        for (batch_idx, chunk) in indices.chunks(batch_size).enumerate() {
            let batch_inputs = Self::select_rows(inputs, chunk)?;
            let batch_targets = Self::select_rows(targets, chunk)?;
            
            let metadata = BatchMetadata {
                batch_idx,
                total_batches,
                epoch: 0, // Will be set by trainer
                sample_indices: chunk.to_vec(),
            };
            
            batches.push(BatchData::new(batch_inputs, batch_targets, metadata)?);
        }
        
        Ok(batches)
    }
    
    /// Select specific rows from tensor (for batching)
    fn select_rows(tensor: &DNNTensor, indices: &[usize]) -> Result<DNNTensor, DNNError> {
        let mut selected_data = Array2::zeros((indices.len(), tensor.feature_dim()));
        
        for (new_row, &original_row) in indices.iter().enumerate() {
            if original_row >= tensor.batch_size() {
                return Err(DNNError::InvalidInput(
                    format!("Row index {} out of bounds for tensor with {} rows", 
                           original_row, tensor.batch_size())
                ));
            }
            
            selected_data.row_mut(new_row).assign(&tensor.data.row(original_row));
        }
        
        DNNTensor::new(selected_data)
    }
}

// === MEMORY OPTIMIZATION UTILITIES ===

/**
 * Memory pool for efficient tensor allocation and reuse.
 * 
 * Reduces allocation overhead during training by reusing tensor buffers.
 */
pub struct TensorPool {
    /// Pool of available tensors by shape
    available: std::collections::HashMap<TensorShape, Vec<DNNTensor>>,
    /// Maximum pool size per shape
    max_pool_size: usize,
}

impl TensorPool {
    /// Create a new tensor pool
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            available: std::collections::HashMap::new(),
            max_pool_size,
        }
    }
    
    /// Get a tensor from the pool or allocate new one
    pub fn get_tensor(&mut self, shape: &TensorShape) -> Result<DNNTensor, DNNError> {
        if let Some(pool) = self.available.get_mut(shape) {
            if let Some(tensor) = pool.pop() {
                return Ok(tensor);
            }
        }
        
        // Allocate new tensor if pool is empty
        DNNTensor::zeros(shape)
    }
    
    /// Return a tensor to the pool
    pub fn return_tensor(&mut self, mut tensor: DNNTensor) {
        let shape = tensor.shape().clone();
        
        // Clear tensor data (zero fill)
        tensor.data.fill(0.0);
        tensor.requires_grad = false;
        
        // Add to pool if under limit
        let pool = self.available.entry(shape).or_insert_with(Vec::new);
        if pool.len() < self.max_pool_size {
            pool.push(tensor);
        }
    }
    
    /// Clear all pools
    pub fn clear(&mut self) {
        self.available.clear();
    }
}

// === TESTS ===

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_tensor_creation() {
        let shape = TensorShape::new_2d(3, 4);
        let tensor = DNNTensor::zeros(&shape).unwrap();
        
        assert_eq!(tensor.batch_size(), 3);
        assert_eq!(tensor.feature_dim(), 4);
        assert_eq!(tensor.shape().total_elements(), 12);
    }
    
    #[test]
    fn test_tensor_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = TensorShape::new_2d(2, 3);
        let tensor = DNNTensor::from_vec(data, &shape).unwrap();
        
        assert_eq!(tensor.batch_size(), 2);
        assert_eq!(tensor.feature_dim(), 3);
        assert_eq!(tensor.data[[0, 0]], 1.0);
        assert_eq!(tensor.data[[1, 2]], 6.0);
    }
    
    #[test]
    fn test_dense_forward_operation() {
        // Create input tensor [2, 3] - 2 samples, 3 features each
        let input_data = Array2::from_shape_vec((2, 3), vec![
            1.0, 2.0, 3.0,  // Sample 1
            4.0, 5.0, 6.0   // Sample 2
        ]).unwrap();
        let input = DNNTensor::new(input_data).unwrap();
        
        // Create weight matrix [3, 2] - 3 input features to 2 output features
        let weight = Array2::from_shape_vec((3, 2), vec![
            0.1, 0.2,
            0.3, 0.4,
            0.5, 0.6
        ]).unwrap();
        
        // Create bias vector [2]
        let bias = Array1::from_vec(vec![0.1, 0.2]);
        
        // Forward pass
        let output = TensorOps::dense_forward(&input, &weight, Some(&bias)).unwrap();
        
        assert_eq!(output.batch_size(), 2);
        assert_eq!(output.feature_dim(), 2);
        
        // Manual calculation for verification:
        // Sample 1: [1,2,3] @ [[0.1,0.2],[0.3,0.4],[0.5,0.6]] + [0.1,0.2]
        //         = [1*0.1+2*0.3+3*0.5, 1*0.2+2*0.4+3*0.6] + [0.1,0.2]
        //         = [2.2, 2.8] + [0.1,0.2] = [2.3, 3.0]
        assert_relative_eq!(output.data[[0, 0]], 2.3, epsilon = 1e-6);
        assert_relative_eq!(output.data[[0, 1]], 3.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_batch_creation() {
        let inputs = DNNTensor::ones(&TensorShape::new_2d(10, 5)).unwrap();
        let targets = DNNTensor::zeros(&TensorShape::new_2d(10, 2)).unwrap();
        
        let batches = TensorOps::create_batches(&inputs, &targets, 3, false).unwrap();
        
        // Should create 4 batches: 3+3+3+1 = 10 samples
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0].batch_size(), 3);
        assert_eq!(batches[1].batch_size(), 3);
        assert_eq!(batches[2].batch_size(), 3);
        assert_eq!(batches[3].batch_size(), 1); // Last batch smaller
    }
    
    #[test]
    fn test_tensor_pool() {
        let mut pool = TensorPool::new(2);
        let shape = TensorShape::new_2d(5, 10);
        
        // Get tensor from empty pool (should allocate)
        let tensor1 = pool.get_tensor(&shape).unwrap();
        assert_eq!(tensor1.shape(), &shape);
        
        // Return tensor to pool
        pool.return_tensor(tensor1);
        
        // Get tensor again (should reuse from pool)
        let tensor2 = pool.get_tensor(&shape).unwrap();
        assert_eq!(tensor2.shape(), &shape);
    }
    
    #[test]
    fn test_shape_compatibility() {
        let shape1 = TensorShape::new_2d(5, 10);
        let shape2 = TensorShape::new_2d(10, 20);
        
        assert!(shape1.is_compatible_for_matmul(&shape2));
        assert!(!shape2.is_compatible_for_matmul(&shape1));
        
        let output_shape = shape1.matmul_output_shape(&shape2).unwrap();
        assert_eq!(output_shape.dims, vec![5, 20]);
    }
    
    #[test]
    fn test_invalid_values_detection() {
        let mut data = Array2::ones((2, 2));
        data[[0, 0]] = f32::NAN;
        data[[1, 1]] = f32::INFINITY;
        
        let tensor = DNNTensor::new(data).unwrap();
        assert!(tensor.has_invalid_values());
        
        let clean_tensor = DNNTensor::ones(&TensorShape::new_2d(2, 2)).unwrap();
        assert!(!clean_tensor.has_invalid_values());
    }
}