/**
 * @file zen-neural/src/dnn/utils.rs
 * @brief Utility Functions for Deep Neural Network Operations
 * 
 * This module provides essential utility functions for DNN operations including
 * tensor manipulations, numerical stability helpers, debug utilities, and
 * performance analysis tools. These utilities support the main DNN functionality
 * with type-safe and efficient implementations.
 * 
 * ## Core Utilities:
 * - **Tensor Operations**: Shape validation, tensor conversion, batch handling
 * - **Numerical Stability**: Overflow/underflow protection, NaN detection
 * - **Debug Helpers**: Model inspection, layer analysis, gradient checking
 * - **Performance Analysis**: Timing, memory usage, FLOP counting
 * - **Data Processing**: Normalization, scaling, preprocessing utilities
 * 
 * @author Module Resolver Agent (ruv-swarm)
 * @version 1.0.0-alpha.1
 * @since 2025-01-14
 */

use std::collections::HashMap;
use ndarray::{Array2, Axis, s};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::DNNError;
use super::data::{DNNTensor, TensorShape};
use super::{DNNConfig, ActivationType, WeightInitialization};

// === TENSOR UTILITIES ===

/// Tensor shape validation and manipulation utilities
pub struct TensorUtils;

impl TensorUtils {
    /// Validate tensor shape compatibility for matrix operations
    pub fn validate_shapes_for_matmul(
        a_shape: &[usize], 
        b_shape: &[usize]
    ) -> Result<(usize, usize, usize), DNNError> {
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(DNNError::InvalidInput(
                "Matrix multiplication requires 2D tensors".to_string()
            ));
        }
        
        let (m, k) = (a_shape[0], a_shape[1]);
        let (k2, n) = (b_shape[0], b_shape[1]);
        
        if k != k2 {
            return Err(DNNError::DimensionMismatch(
                format!("Inner dimensions must match: {} vs {}", k, k2)
            ));
        }
        
        Ok((m, k, n))
    }
    
    /// Calculate output shape for a layer given input shape and layer config
    pub fn calculate_output_shape(
        input_shape: &[usize],
        layer_type: &str,
        layer_config: &HashMap<String, usize>
    ) -> Result<Vec<usize>, DNNError> {
        match layer_type {
            "dense" | "linear" => {
                if input_shape.len() != 2 {
                    return Err(DNNError::InvalidInput(
                        "Dense layers require 2D input [batch, features]".to_string()
                    ));
                }
                
                let output_units = layer_config.get("units")
                    .ok_or_else(|| DNNError::InvalidConfiguration(
                        "Dense layer missing 'units' parameter".to_string()
                    ))?;
                    
                Ok(vec![input_shape[0], *output_units])
            },
            "dropout" => {
                // Dropout doesn't change shape
                Ok(input_shape.to_vec())
            },
            "batch_norm" => {
                // Batch normalization doesn't change shape
                Ok(input_shape.to_vec())
            },
            _ => Err(DNNError::InvalidConfiguration(
                format!("Unknown layer type: {}", layer_type)
            ))
        }
    }
    
    /// Convert flat vector to 2D tensor with specified batch size
    pub fn reshape_to_batch(data: Vec<f32>, batch_size: usize, feature_size: usize) -> Result<DNNTensor, DNNError> {
        if data.len() != batch_size * feature_size {
            return Err(DNNError::DimensionMismatch(
                format!("Data length {} doesn't match batch_size {} * feature_size {}", 
                    data.len(), batch_size, feature_size)
            ));
        }
        
        let array = Array2::from_shape_vec((batch_size, feature_size), data)
            .map_err(|e| DNNError::TensorError(e.to_string()))?;
            
        Ok(DNNTensor { data: array })
    }
    
    /// Extract batch from tensor
    pub fn extract_batch(tensor: &DNNTensor, batch_idx: usize) -> Result<DNNTensor, DNNError> {
        let shape = tensor.data.shape();
        if batch_idx >= shape[0] {
            return Err(DNNError::InvalidInput(
                format!("Batch index {} out of bounds for batch size {}", batch_idx, shape[0])
            ));
        }
        
        let batch_data = tensor.data.slice(s![batch_idx..batch_idx+1, ..]).to_owned();
        Ok(DNNTensor { data: batch_data })
    }
    
    /// Concatenate tensors along batch dimension
    pub fn concatenate_batches(tensors: &[DNNTensor]) -> Result<DNNTensor, DNNError> {
        if tensors.is_empty() {
            return Err(DNNError::InvalidInput("Cannot concatenate empty tensor list".to_string()));
        }
        
        // Validate all tensors have same feature dimension
        let feature_dim = tensors[0].data.shape()[1];
        for (i, tensor) in tensors.iter().enumerate() {
            if tensor.data.shape().len() != 2 {
                return Err(DNNError::InvalidInput(
                    format!("Tensor {} is not 2D", i)
                ));
            }
            if tensor.data.shape()[1] != feature_dim {
                return Err(DNNError::DimensionMismatch(
                    format!("Feature dimension mismatch: expected {}, got {} at tensor {}", 
                        feature_dim, tensor.data.shape()[1], i)
                ));
            }
        }
        
        let arrays: Vec<_> = tensors.iter().map(|t| t.data.view()).collect();
        let concatenated = ndarray::concatenate(Axis(0), &arrays)
            .map_err(|e| DNNError::TensorError(e.to_string()))?;
            
        Ok(DNNTensor { data: concatenated })
    }
}

// === NUMERICAL STABILITY UTILITIES ===

/// Numerical stability helpers for safe neural network operations
pub struct NumericalUtils;

impl NumericalUtils {
    /// Check for NaN or infinite values in tensor
    pub fn has_invalid_values(tensor: &DNNTensor) -> bool {
        tensor.data.iter().any(|&x| x.is_nan() || x.is_infinite())
    }
    
    /// Clip values to prevent overflow/underflow
    pub fn clip_values(tensor: &mut DNNTensor, min_val: f32, max_val: f32) {
        tensor.data.mapv_inplace(|x| x.clamp(min_val, max_val));
    }
    
    /// Apply numerical gradient clipping
    pub fn clip_gradients_by_norm(gradients: &mut [DNNTensor], max_norm: f32) -> f32 {
        // Calculate total gradient norm
        let total_norm: f32 = gradients.iter()
            .map(|grad| grad.data.iter().map(|&x| x * x).sum::<f32>())
            .sum::<f32>()
            .sqrt();
            
        if total_norm > max_norm {
            let clip_factor = max_norm / total_norm;
            for gradient in gradients.iter_mut() {
                gradient.data.mapv_inplace(|x| x * clip_factor);
            }
        }
        
        total_norm
    }
    
    /// Stable softmax computation to prevent overflow
    pub fn stable_softmax(logits: &mut DNNTensor) {
        let batch_size = logits.data.shape()[0];
        
        for i in 0..batch_size {
            let mut row = logits.data.slice_mut(s![i, ..]);
            
            // Subtract max for numerical stability
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            row.mapv_inplace(|x| x - max_val);
            
            // Compute exponentials
            row.mapv_inplace(|x| x.exp());
            
            // Normalize
            let sum: f32 = row.sum();
            if sum > 0.0 {
                row.mapv_inplace(|x| x / sum);
            }
        }
    }
    
    /// Safe log computation that handles zeros
    pub fn safe_log(tensor: &DNNTensor, epsilon: f32) -> DNNTensor {
        let data = tensor.data.mapv(|x| (x + epsilon).ln());
        DNNTensor { data }
    }
    
    /// Check for exploding gradients
    pub fn detect_exploding_gradients(gradients: &[DNNTensor], threshold: f32) -> bool {
        let max_gradient = gradients.iter()
            .map(|grad| grad.data.iter().fold(0.0f32, |a, &b| a.max(b.abs())))
            .fold(0.0f32, |a, b| a.max(b));
            
        max_gradient > threshold
    }
}

// === DEBUG AND ANALYSIS UTILITIES ===

/// Debug and inspection utilities for DNN models
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct DebugInfo {
    pub tensor_stats: TensorStats,
    pub layer_info: Vec<LayerDebugInfo>,
    pub memory_usage: usize,
    pub computation_graph: Vec<String>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct TensorStats {
    pub shape: Vec<usize>,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub has_nan: bool,
    pub has_inf: bool,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct LayerDebugInfo {
    pub layer_type: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub parameter_count: usize,
    pub activation_stats: Option<TensorStats>,
}

pub struct DebugUtils;

impl DebugUtils {
    /// Compute comprehensive tensor statistics
    pub fn compute_tensor_stats(tensor: &DNNTensor) -> TensorStats {
        let data = &tensor.data;
        let flat_data: Vec<f32> = data.iter().copied().collect();
        
        if flat_data.is_empty() {
            return TensorStats {
                shape: data.shape().to_vec(),
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                has_nan: false,
                has_inf: false,
            };
        }
        
        let mean = flat_data.iter().sum::<f32>() / flat_data.len() as f32;
        let variance = flat_data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / flat_data.len() as f32;
        let std = variance.sqrt();
        
        let min = flat_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = flat_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let has_nan = flat_data.iter().any(|&x| x.is_nan());
        let has_inf = flat_data.iter().any(|&x| x.is_infinite());
        
        TensorStats {
            shape: data.shape().to_vec(),
            mean,
            std,
            min,
            max,
            has_nan,
            has_inf,
        }
    }
    
    /// Print tensor summary for debugging
    pub fn print_tensor_summary(tensor: &DNNTensor, name: &str) {
        let stats = Self::compute_tensor_stats(tensor);
        println!("=== Tensor Summary: {} ===", name);
        println!("Shape: {:?}", stats.shape);
        println!("Mean: {:.6}, Std: {:.6}", stats.mean, stats.std);
        println!("Min: {:.6}, Max: {:.6}", stats.min, stats.max);
        if stats.has_nan {
            println!("⚠️ WARNING: Contains NaN values!");
        }
        if stats.has_inf {
            println!("⚠️ WARNING: Contains infinite values!");
        }
        println!();
    }
    
    /// Validate tensor for common issues
    pub fn validate_tensor_health(tensor: &DNNTensor) -> Vec<String> {
        let mut issues = Vec::new();
        let stats = Self::compute_tensor_stats(tensor);
        
        if stats.has_nan {
            issues.push("Contains NaN values".to_string());
        }
        
        if stats.has_inf {
            issues.push("Contains infinite values".to_string());
        }
        
        if stats.std == 0.0 {
            issues.push("All values are identical (zero variance)".to_string());
        }
        
        if stats.max - stats.min > 1000.0 {
            issues.push("Very large value range - may cause numerical instability".to_string());
        }
        
        if stats.mean.abs() > 100.0 {
            issues.push("Mean is very large - consider normalization".to_string());
        }
        
        issues
    }
}

// === PERFORMANCE UTILITIES ===

/// Performance analysis and profiling utilities
pub struct PerformanceUtils;

impl PerformanceUtils {
    /// Estimate FLOPS for a dense layer operation
    pub fn estimate_dense_layer_flops(
        batch_size: usize,
        input_dim: usize,
        output_dim: usize,
        has_bias: bool
    ) -> u64 {
        // Matrix multiplication: batch_size * input_dim * output_dim
        let matmul_flops = batch_size * input_dim * output_dim;
        
        // Bias addition: batch_size * output_dim (if has bias)
        let bias_flops = if has_bias { batch_size * output_dim } else { 0 };
        
        (matmul_flops + bias_flops) as u64
    }
    
    /// Estimate memory usage for a tensor
    pub fn estimate_tensor_memory(shape: &[usize], dtype_size: usize) -> usize {
        shape.iter().product::<usize>() * dtype_size
    }
    
    /// Calculate theoretical peak memory for a model
    pub fn estimate_peak_memory(
        config: &DNNConfig,
        batch_size: usize,
        include_gradients: bool
    ) -> usize {
        let mut total_memory = 0;
        
        // Parameters memory
        let mut current_dim = config.input_dim;
        for &hidden_dim in &config.hidden_layers {
            // Weights: current_dim * hidden_dim
            total_memory += current_dim * hidden_dim * 4; // 4 bytes per f32
            
            // Biases: hidden_dim
            if config.use_bias {
                total_memory += hidden_dim * 4;
            }
            
            current_dim = hidden_dim;
        }
        
        // Output layer
        total_memory += current_dim * config.output_dim * 4;
        if config.use_bias {
            total_memory += config.output_dim * 4;
        }
        
        // Activations memory (forward pass)
        let max_hidden_dim = config.hidden_layers.iter().max().unwrap_or(&config.input_dim);
        let activation_memory = batch_size * max_hidden_dim * 4;
        total_memory += activation_memory;
        
        // Gradients memory (if training)
        if include_gradients {
            total_memory *= 2; // Roughly double for gradients
        }
        
        total_memory
    }
}

// === DATA PREPROCESSING UTILITIES ===

/// Data preprocessing and normalization utilities
pub struct PreprocessingUtils;

impl PreprocessingUtils {
    /// Normalize tensor to zero mean and unit variance
    pub fn standardize(tensor: &mut DNNTensor) -> (f32, f32) {
        let data = &mut tensor.data;
        let flat_data: Vec<f32> = data.iter().copied().collect();
        
        if flat_data.is_empty() {
            return (0.0, 1.0);
        }
        
        let mean = flat_data.iter().sum::<f32>() / flat_data.len() as f32;
        let variance = flat_data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / flat_data.len() as f32;
        let std = variance.sqrt().max(1e-8); // Prevent division by zero
        
        data.mapv_inplace(|x| (x - mean) / std);
        
        (mean, std)
    }
    
    /// Normalize tensor to [0, 1] range
    pub fn min_max_normalize(tensor: &mut DNNTensor) -> (f32, f32) {
        let data = &mut tensor.data;
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let range = (max_val - min_val).max(1e-8); // Prevent division by zero
        
        data.mapv_inplace(|x| (x - min_val) / range);
        
        (min_val, max_val)
    }
    
    /// Add gaussian noise for regularization
    pub fn add_gaussian_noise(tensor: &mut DNNTensor, noise_std: f32, rng: &mut impl rand::Rng) {
        use rand::distributions::Distribution;
        use rand_distr::Normal;
        let normal = Normal::new(0.0, noise_std as f64).unwrap();
        
        tensor.data.mapv_inplace(|x| x + normal.sample(rng) as f32);
    }
}

// === WEIGHT INITIALIZATION UTILITIES ===

/// Weight initialization utilities for neural networks
pub struct InitializationUtils;

impl InitializationUtils {
    /// Initialize weights according to specified strategy
    pub fn initialize_weights(
        shape: (usize, usize),
        init_type: WeightInitialization,
        rng: &mut impl rand::Rng
    ) -> Array2<f32> {
        use rand::distributions::{Distribution, Uniform};
        use rand_distr::Normal as RandDistrNormal;
        
        match init_type {
            WeightInitialization::He => {
                let fan_in = shape.0 as f32;
                let std = (2.0 / fan_in).sqrt();
                let normal = RandDistrNormal::new(0.0, std as f64).unwrap();
                Array2::from_shape_fn(shape, |_| normal.sample(rng) as f32)
            },
            WeightInitialization::Xavier => {
                let fan_in = shape.0 as f32;
                let fan_out = shape.1 as f32;
                let std = (2.0 / (fan_in + fan_out)).sqrt();
                let normal = RandDistrNormal::new(0.0, std as f64).unwrap();
                Array2::from_shape_fn(shape, |_| normal.sample(rng) as f32)
            },
            WeightInitialization::LeCun => {
                let fan_in = shape.0 as f32;
                let std = (1.0 / fan_in).sqrt();
                let normal = RandDistrNormal::new(0.0, std as f64).unwrap();
                Array2::from_shape_fn(shape, |_| normal.sample(rng) as f32)
            },
            WeightInitialization::Normal { mean, std } => {
                let normal = RandDistrNormal::new(mean as f64, std as f64).unwrap();
                Array2::from_shape_fn(shape, |_| normal.sample(rng) as f32)
            },
            WeightInitialization::Uniform { min, max } => {
                let uniform = Uniform::new(min, max);
                Array2::from_shape_fn(shape, |_| uniform.sample(rng))
            },
        }
    }
    
    /// Initialize bias vector (usually zeros)
    pub fn initialize_bias(size: usize) -> Array2<f32> {
        Array2::zeros((1, size))
    }
    
    /// Calculate appropriate initialization standard deviation for activation type
    pub fn recommended_std_for_activation(activation: ActivationType, fan_in: usize) -> f32 {
        match activation {
            ActivationType::ReLU | ActivationType::LeakyReLU => {
                // He initialization
                (2.0 / fan_in as f32).sqrt()
            },
            ActivationType::Tanh | ActivationType::Sigmoid => {
                // Xavier initialization
                (1.0 / fan_in as f32).sqrt()
            },
            ActivationType::GELU | ActivationType::Swish => {
                // Similar to ReLU
                (2.0 / fan_in as f32).sqrt()
            },
            _ => {
                // Default Xavier
                (1.0 / fan_in as f32).sqrt()
            }
        }
    }
}

// === TESTS ===

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_tensor_shape_validation() {
        let result = TensorUtils::validate_shapes_for_matmul(&[2, 3], &[3, 4]);
        assert!(result.is_ok());
        let (m, k, n) = result.unwrap();
        assert_eq!((m, k, n), (2, 3, 4));
        
        // Test mismatch
        let result = TensorUtils::validate_shapes_for_matmul(&[2, 3], &[4, 5]);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_tensor_stats() {
        let data = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tensor = DNNTensor { data };
        
        let stats = DebugUtils::compute_tensor_stats(&tensor);
        assert_eq!(stats.shape, vec![2, 3]);
        assert!((stats.mean - 3.5).abs() < 1e-5);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 6.0);
        assert!(!stats.has_nan);
        assert!(!stats.has_inf);
    }
    
    #[test]
    fn test_numerical_stability() {
        let data = Array2::from_shape_vec((1, 3), vec![f32::NAN, 1.0, f32::INFINITY]).unwrap();
        let tensor = DNNTensor { data };
        
        assert!(NumericalUtils::has_invalid_values(&tensor));
    }
    
    #[test]
    fn test_flops_estimation() {
        let flops = PerformanceUtils::estimate_dense_layer_flops(32, 784, 128, true);
        // 32 * 784 * 128 (matmul) + 32 * 128 (bias) = 3,211,264 + 4,096 = 3,215,360
        assert_eq!(flops, 3_215_360);
    }
}