/**
 * @file zen-neural/src/dnn/activations.rs
 * @brief Activation Function Implementations for DNN
 * 
 * This module implements optimized activation functions for Deep Neural Networks,
 * replacing JavaScript element-wise operations with SIMD-accelerated implementations.
 * Provides both individual activation functions and activation layer wrappers.
 * 
 * ## Core Components:
 * - **ActivationLayer**: Layer wrapper for activation functions  
 * - **ActivationFunction**: Individual activation implementations
 * - **SIMD Optimizations**: Vectorized operations for performance
 * - **Gradient Computation**: Derivative calculations for backpropagation
 * 
 * ## JavaScript to Rust Translation:
 * 
 * ### JavaScript Original (CNN.js activation methods):
 * ```javascript
 * relu(input) {
 *   const output = new Float32Array(input.length);
 *   for (let i = 0; i < input.length; i++) {
 *     output[i] = Math.max(0, input[i]);
 *   }
 *   return output;
 * }
 * 
 * sigmoid(input) {
 *   const output = new Float32Array(input.length);
 *   for (let i = 0; i < input.length; i++) {
 *     output[i] = 1.0 / (1.0 + Math.exp(-input[i]));
 *   }
 *   return output;
 * }
 * 
 * softmax(input) {
 *   const [batchSize, size] = input.shape;
 *   const output = new Float32Array(input.length);
 *   
 *   for (let b = 0; b < batchSize; b++) {
 *     let maxVal = -Infinity;
 *     // Find max for numerical stability
 *     for (let i = 0; i < size; i++) {
 *       maxVal = Math.max(maxVal, input[offset + i]);
 *     }
 *     // Compute exp and sum...
 *   }
 * }
 * ```
 * 
 * ### Rust Optimized Version:
 * - **SIMD Element-wise**: `tensor.mapv(|x| activation_fn(x))` uses vectorization
 * - **Batch Processing**: Proper handling of tensor batches
 * - **Numerical Stability**: LogSumExp trick for softmax
 * - **Memory Efficiency**: In-place operations where safe
 * 
 * @author DNN Core Developer Agent (ruv-swarm Phase 1)
 * @version 1.0.0-alpha.1
 * @since 2025-01-14
 */

use ndarray::{Array1, Array2, Axis};
use num_traits::{Float, Zero};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::data::{DNNTensor, TensorOps};
use super::layers::DNNLayer;
use super::{DNNError, DNNTrainingMode, ActivationType};

// === ACTIVATION FUNCTION IMPLEMENTATIONS ===

/**
 * Collection of activation function implementations.
 * 
 * Each function is optimized for SIMD execution through ndarray's mapv operations,
 * providing significant performance improvements over JavaScript manual loops.
 */
pub struct ActivationFunctions;

impl ActivationFunctions {
    /**
     * Rectified Linear Unit (ReLU): f(x) = max(0, x)
     * 
     * Most common activation function in deep learning, providing:
     * - Simple computation: max(0, x)
     * - Non-saturating for positive values
     * - Sparse activation (zeros out negative values)
     * - Good gradient flow for positive inputs
     * 
     * Optimized from JavaScript loop to SIMD operation.
     */
    pub fn relu(input: &DNNTensor) -> Result<DNNTensor, DNNError> {
        TensorOps::apply_elementwise(input, |x| x.max(0.0))
    }
    
    /// ReLU derivative: f'(x) = 1 if x > 0, else 0
    pub fn relu_derivative(input: &DNNTensor) -> Result<DNNTensor, DNNError> {
        TensorOps::apply_elementwise(input, |x| if x > 0.0 { 1.0 } else { 0.0 })
    }
    
    /**
     * Leaky ReLU: f(x) = x if x > 0, else α*x (α = 0.01)
     * 
     * Addresses the "dying ReLU" problem by allowing small negative gradients:
     * - Non-zero gradient for negative inputs
     * - Helps with gradient flow in deep networks
     * - Small slope (0.01) for negative values
     */
    pub fn leaky_relu(input: &DNNTensor, alpha: f32) -> Result<DNNTensor, DNNError> {
        TensorOps::apply_elementwise(input, |x| if x > 0.0 { x } else { alpha * x })
    }
    
    /// Leaky ReLU derivative: f'(x) = 1 if x > 0, else α
    pub fn leaky_relu_derivative(input: &DNNTensor, alpha: f32) -> Result<DNNTensor, DNNError> {
        TensorOps::apply_elementwise(input, |x| if x > 0.0 { 1.0 } else { alpha })
    }
    
    /**
     * Hyperbolic Tangent: f(x) = tanh(x) = (e^x - e^-x) / (e^x + e^-x)
     * 
     * Classic activation function with properties:
     * - Output range: (-1, 1)
     * - Zero-centered (unlike sigmoid)
     * - Stronger gradients than sigmoid
     * - Good for RNNs and centered data
     */
    pub fn tanh(input: &DNNTensor) -> Result<DNNTensor, DNNError> {
        TensorOps::apply_elementwise(input, |x| x.tanh())
    }
    
    /// Tanh derivative: f'(x) = 1 - tanh²(x)
    pub fn tanh_derivative(input: &DNNTensor) -> Result<DNNTensor, DNNError> {
        TensorOps::apply_elementwise(input, |x| {
            let tanh_x = x.tanh();
            1.0 - tanh_x * tanh_x
        })
    }
    
    /**
     * Sigmoid: f(x) = 1 / (1 + e^-x)
     * 
     * Classic activation function:
     * - Output range: (0, 1)
     * - Smooth S-curve shape
     * - Good for binary classification
     * - Can suffer from vanishing gradients
     * 
     * Optimized from JavaScript manual exp calculation to built-in operations.
     */
    pub fn sigmoid(input: &DNNTensor) -> Result<DNNTensor, DNNError> {
        TensorOps::apply_elementwise(input, |x| {
            // Numerical stability: avoid overflow for large negative x
            if x < -50.0 {
                0.0
            } else if x > 50.0 {
                1.0
            } else {
                1.0 / (1.0 + (-x).exp())
            }
        })
    }
    
    /// Sigmoid derivative: f'(x) = sigmoid(x) * (1 - sigmoid(x))
    pub fn sigmoid_derivative(input: &DNNTensor) -> Result<DNNTensor, DNNError> {
        let sigmoid_out = Self::sigmoid(input)?;
        TensorOps::apply_elementwise(&sigmoid_out, |s| s * (1.0 - s))
    }
    
    /**
     * Gaussian Error Linear Unit (GELU): f(x) = x * Φ(x)
     * 
     * Modern activation function used in Transformers:
     * - Smooth approximation to ReLU
     * - Better gradient properties
     * - Non-monotonic around zero
     * - Used in BERT, GPT, and other transformer models
     * 
     * Approximation: f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
     */
    pub fn gelu(input: &DNNTensor) -> Result<DNNTensor, DNNError> {
        TensorOps::apply_elementwise(input, |x| {
            // GELU approximation used in many implementations
            0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
        })
    }
    
    /// GELU derivative (approximation)
    pub fn gelu_derivative(input: &DNNTensor) -> Result<DNNTensor, DNNError> {
        TensorOps::apply_elementwise(input, |x| {
            let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
            let tanh_inner = inner.tanh();
            let sech_squared = 1.0 - tanh_inner * tanh_inner;
            
            0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_squared * 0.7978845608 * (1.0 + 3.0 * 0.044715 * x * x)
        })
    }
    
    /**
     * Swish: f(x) = x * sigmoid(x)
     * 
     * Self-gated activation function:
     * - Smooth, non-monotonic
     * - Can go negative (unlike ReLU)
     * - Good empirical performance
     * - Used in EfficientNet and mobile architectures
     */
    pub fn swish(input: &DNNTensor) -> Result<DNNTensor, DNNError> {
        let sigmoid_out = Self::sigmoid(input)?;
        TensorOps::multiply(input, &sigmoid_out)
    }
    
    /// Swish derivative: f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    pub fn swish_derivative(input: &DNNTensor) -> Result<DNNTensor, DNNError> {
        let _sigmoid_out = Self::sigmoid(input)?;
        TensorOps::apply_elementwise(input, |x| {
            let sig_x = 1.0 / (1.0 + (-x).exp());
            sig_x + x * sig_x * (1.0 - sig_x)
        })
    }
    
    /**
     * Softmax: f(x_i) = exp(x_i) / Σ_j exp(x_j)
     * 
     * Converts logits to probabilities:
     * - Output sums to 1.0 across features
     * - Used for multi-class classification
     * - Applied along feature dimension
     * - Requires numerical stability (LogSumExp trick)
     * 
     * Major optimization from JavaScript nested loops to vectorized operations.
     */
    pub fn softmax(input: &DNNTensor) -> Result<DNNTensor, DNNError> {
        let batch_size = input.batch_size();
        let feature_dim = input.feature_dim();
        let mut output_data = input.data.clone();
        
        // Process each sample in the batch
        for batch_idx in 0..batch_size {
            let mut row = output_data.row_mut(batch_idx);
            
            // Find maximum for numerical stability (LogSumExp trick)
            let max_val = row.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
            
            // Subtract max and exponentiate
            row.mapv_inplace(|x| (x - max_val).exp());
            
            // Compute sum of exponentials
            let sum_exp: f32 = row.sum();
            
            // Normalize to get probabilities
            if sum_exp > 0.0 {
                row.mapv_inplace(|x| x / sum_exp);
            } else {
                // Handle edge case: all values were very negative
                row.fill(1.0 / feature_dim as f32);
            }
        }
        
        DNNTensor::new(output_data)
    }
    
    /// Softmax derivative (for cross-entropy loss, often combined)
    pub fn softmax_derivative(input: &DNNTensor) -> Result<DNNTensor, DNNError> {
        let softmax_out = Self::softmax(input)?;
        
        // For softmax, the Jacobian is more complex than element-wise derivatives
        // This is a simplified version - often softmax + cross-entropy are computed together
        TensorOps::apply_elementwise(&softmax_out, |s| s * (1.0 - s))
    }
    
    /**
     * Linear activation: f(x) = x (identity function)
     * 
     * No transformation applied:
     * - Used for regression outputs
     * - Pass-through layer
     * - Final layer for continuous outputs
     */
    pub fn linear(input: &DNNTensor) -> Result<DNNTensor, DNNError> {
        Ok(input.clone())
    }
    
    /// Linear derivative: f'(x) = 1
    pub fn linear_derivative(input: &DNNTensor) -> Result<DNNTensor, DNNError> {
        TensorOps::apply_elementwise(input, |_| 1.0)
    }
}

// === ACTIVATION LAYER WRAPPER ===

/**
 * Activation layer that can be inserted into neural network architectures.
 * 
 * Provides a layer interface for activation functions, allowing them to be
 * used as separate layers in the network architecture. This is the preferred
 * approach as it separates linear transformations from non-linear activations.
 */
#[derive(Debug, Clone)]
pub struct ActivationLayer {
    /// Type of activation function
    activation_type: ActivationType,
    
    /// Input/output dimension (same for activation layers)
    dimension: Option<usize>,
    
    /// Whether layer is compiled
    compiled: bool,
    
    /// Configuration parameters (e.g., alpha for LeakyReLU)
    config: ActivationConfig,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct ActivationConfig {
    /// Alpha parameter for LeakyReLU
    pub leaky_relu_alpha: f32,
    
    /// Temperature parameter for softmax (optional)
    pub softmax_temperature: f32,
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            leaky_relu_alpha: 0.01,
            softmax_temperature: 1.0,
        }
    }
}

impl ActivationLayer {
    /// Create a new activation layer
    pub fn new(activation_type: ActivationType) -> Self {
        Self {
            activation_type,
            dimension: None,
            compiled: false,
            config: ActivationConfig::default(),
        }
    }
    
    /// Create activation layer with custom configuration
    pub fn with_config(activation_type: ActivationType, config: ActivationConfig) -> Self {
        Self {
            activation_type,
            dimension: None,
            compiled: false,
            config,
        }
    }
    
    /// Apply the activation function
    fn apply_activation(&self, input: &DNNTensor) -> Result<DNNTensor, DNNError> {
        match self.activation_type {
            ActivationType::ReLU => ActivationFunctions::relu(input),
            ActivationType::LeakyReLU => ActivationFunctions::leaky_relu(input, self.config.leaky_relu_alpha),
            ActivationType::Tanh => ActivationFunctions::tanh(input),
            ActivationType::Sigmoid => ActivationFunctions::sigmoid(input),
            ActivationType::GELU => ActivationFunctions::gelu(input),
            ActivationType::Swish => ActivationFunctions::swish(input),
            ActivationType::Softmax => ActivationFunctions::softmax(input),
            ActivationType::Linear => ActivationFunctions::linear(input),
        }
    }
    
    /// Apply activation derivative for backpropagation
    fn apply_activation_derivative(&self, input: &DNNTensor) -> Result<DNNTensor, DNNError> {
        match self.activation_type {
            ActivationType::ReLU => ActivationFunctions::relu_derivative(input),
            ActivationType::LeakyReLU => ActivationFunctions::leaky_relu_derivative(input, self.config.leaky_relu_alpha),
            ActivationType::Tanh => ActivationFunctions::tanh_derivative(input),
            ActivationType::Sigmoid => ActivationFunctions::sigmoid_derivative(input),
            ActivationType::GELU => ActivationFunctions::gelu_derivative(input),
            ActivationType::Swish => ActivationFunctions::swish_derivative(input),
            ActivationType::Softmax => ActivationFunctions::softmax_derivative(input),
            ActivationType::Linear => ActivationFunctions::linear_derivative(input),
        }
    }
}

#[async_trait::async_trait]
impl DNNLayer for ActivationLayer {
    /**
     * Forward pass through activation layer.
     * 
     * Applies the activation function element-wise to the input tensor.
     * Performance is significantly improved over JavaScript through SIMD operations.
     */
    async fn forward(
        &self,
        input: &DNNTensor,
        _mode: DNNTrainingMode,
    ) -> Result<DNNTensor, DNNError> {
        if !self.compiled {
            return Err(DNNError::InvalidConfiguration(
                "Activation layer not compiled".to_string()
            ));
        }
        
        self.apply_activation(input)
    }
    
    /**
     * Backward pass through activation layer.
     * 
     * Computes gradients using the chain rule:
     * grad_input = grad_output * activation'(input)
     */
    async fn backward(
        &mut self,
        input: &DNNTensor,
        grad_output: &DNNTensor,
    ) -> Result<DNNTensor, DNNError> {
        if !self.compiled {
            return Err(DNNError::InvalidConfiguration(
                "Activation layer not compiled".to_string()
            ));
        }
        
        // Compute activation derivative
        let activation_grad = self.apply_activation_derivative(input)?;
        
        // Chain rule: grad_input = grad_output * activation'(input)
        TensorOps::multiply(grad_output, &activation_grad)
    }
    
    /// Compile the activation layer (dimension passes through unchanged)
    fn compile(&mut self, input_dim: usize) -> Result<usize, DNNError> {
        self.dimension = Some(input_dim);
        self.compiled = true;
        Ok(input_dim) // Output dimension same as input for activation layers
    }
    
    fn layer_type(&self) -> &'static str {
        match self.activation_type {
            ActivationType::ReLU => "ReLU",
            ActivationType::LeakyReLU => "LeakyReLU",
            ActivationType::Tanh => "Tanh",
            ActivationType::Sigmoid => "Sigmoid",
            ActivationType::GELU => "GELU",
            ActivationType::Swish => "Swish",
            ActivationType::Softmax => "Softmax",
            ActivationType::Linear => "Linear",
        }
    }
    
    /// Activation layers have no trainable parameters
    fn parameter_count(&self) -> usize {
        0
    }
    
    /// No parameters to return
    fn get_parameters(&self) -> Vec<Array2<f32>> {
        Vec::new()
    }
    
    /// No parameters to set
    fn set_parameters(&mut self, _params: &[Array2<f32>]) -> Result<(), DNNError> {
        Ok(()) // No-op for activation layers
    }
    
    /// No parameters to update
    fn update_parameters(&mut self, _gradients: &[Array2<f32>], _learning_rate: f32) -> Result<(), DNNError> {
        Ok(()) // No-op for activation layers
    }
    
    /// No state to reset
    fn reset(&mut self) {
        // No-op for activation layers
    }
    
    fn is_compiled(&self) -> bool {
        self.compiled
    }
}

// === ACTIVATION FACTORY ===

/**
 * Factory for creating activation layers.
 */
pub struct ActivationFactory;

impl ActivationFactory {
    /// Create ReLU activation layer
    pub fn relu() -> Box<dyn DNNLayer> {
        Box::new(ActivationLayer::new(ActivationType::ReLU))
    }
    
    /// Create LeakyReLU activation layer with custom alpha
    pub fn leaky_relu(alpha: f32) -> Box<dyn DNNLayer> {
        let config = ActivationConfig {
            leaky_relu_alpha: alpha,
            ..Default::default()
        };
        Box::new(ActivationLayer::with_config(ActivationType::LeakyReLU, config))
    }
    
    /// Create Tanh activation layer
    pub fn tanh() -> Box<dyn DNNLayer> {
        Box::new(ActivationLayer::new(ActivationType::Tanh))
    }
    
    /// Create Sigmoid activation layer
    pub fn sigmoid() -> Box<dyn DNNLayer> {
        Box::new(ActivationLayer::new(ActivationType::Sigmoid))
    }
    
    /// Create GELU activation layer
    pub fn gelu() -> Box<dyn DNNLayer> {
        Box::new(ActivationLayer::new(ActivationType::GELU))
    }
    
    /// Create Swish activation layer
    pub fn swish() -> Box<dyn DNNLayer> {
        Box::new(ActivationLayer::new(ActivationType::Swish))
    }
    
    /// Create Softmax activation layer
    pub fn softmax() -> Box<dyn DNNLayer> {
        Box::new(ActivationLayer::new(ActivationType::Softmax))
    }
    
    /// Create Linear activation layer (identity)
    pub fn linear() -> Box<dyn DNNLayer> {
        Box::new(ActivationLayer::new(ActivationType::Linear))
    }
}

// === UTILITY FUNCTIONS ===

/**
 * Activation utilities for common operations.
 */
pub struct ActivationUtils;

impl ActivationUtils {
    /// Get activation function by name (for configuration parsing)
    pub fn from_string(name: &str) -> Result<ActivationType, DNNError> {
        match name.to_lowercase().as_str() {
            "relu" => Ok(ActivationType::ReLU),
            "leaky_relu" | "leakyrelu" => Ok(ActivationType::LeakyReLU),
            "tanh" => Ok(ActivationType::Tanh),
            "sigmoid" => Ok(ActivationType::Sigmoid),
            "gelu" => Ok(ActivationType::GELU),
            "swish" => Ok(ActivationType::Swish),
            "softmax" => Ok(ActivationType::Softmax),
            "linear" | "identity" => Ok(ActivationType::Linear),
            _ => Err(DNNError::InvalidConfiguration(
                format!("Unknown activation function: {}", name)
            )),
        }
    }
    
    /// Check if activation is suitable for output layers
    pub fn is_output_activation(activation: ActivationType) -> bool {
        matches!(activation, 
            ActivationType::Softmax | 
            ActivationType::Sigmoid | 
            ActivationType::Linear
        )
    }
    
    /// Get recommended weight initialization for activation
    pub fn recommended_weight_init(activation: ActivationType) -> super::WeightInitialization {
        match activation {
            ActivationType::ReLU | ActivationType::LeakyReLU | ActivationType::GELU => {
                super::WeightInitialization::He
            }
            ActivationType::Tanh | ActivationType::Sigmoid => {
                super::WeightInitialization::Xavier
            }
            ActivationType::Swish | ActivationType::Linear | ActivationType::Softmax => {
                super::WeightInitialization::He // Default to He
            }
        }
    }
}

// === TESTS ===

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dnn::data::TensorShape;
    use approx::assert_relative_eq;
    
    fn create_test_tensor() -> DNNTensor {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let shape = TensorShape::new_2d(2, 3);
        DNNTensor::from_vec(data, &shape).unwrap()
    }
    
    #[test]
    fn test_relu_activation() {
        let input = create_test_tensor();
        let output = ActivationFunctions::relu(&input).unwrap();
        
        // ReLU should zero out negative values
        assert_eq!(output.data[[0, 0]], 0.0); // -2.0 -> 0.0
        assert_eq!(output.data[[0, 1]], 0.0); // -1.0 -> 0.0
        assert_eq!(output.data[[0, 2]], 0.0); // 0.0 -> 0.0
        assert_eq!(output.data[[1, 0]], 1.0); // 1.0 -> 1.0
        assert_eq!(output.data[[1, 1]], 2.0); // 2.0 -> 2.0
        assert_eq!(output.data[[1, 2]], 3.0); // 3.0 -> 3.0
    }
    
    #[test]
    fn test_leaky_relu_activation() {
        let input = create_test_tensor();
        let alpha = 0.1;
        let output = ActivationFunctions::leaky_relu(&input, alpha).unwrap();
        
        // Leaky ReLU should have small negative values
        assert_relative_eq!(output.data[[0, 0]], -0.2, epsilon = 1e-6); // -2.0 * 0.1
        assert_relative_eq!(output.data[[0, 1]], -0.1, epsilon = 1e-6); // -1.0 * 0.1
        assert_eq!(output.data[[0, 2]], 0.0); // 0.0 -> 0.0
        assert_eq!(output.data[[1, 0]], 1.0); // Positive values unchanged
    }
    
    #[test]
    fn test_sigmoid_activation() {
        let input = create_test_tensor();
        let output = ActivationFunctions::sigmoid(&input).unwrap();
        
        // All sigmoid outputs should be between 0 and 1
        for val in output.data.iter() {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
        
        // Sigmoid(0) should be 0.5
        assert_relative_eq!(output.data[[0, 2]], 0.5, epsilon = 1e-6);
    }
    
    #[test]
    fn test_tanh_activation() {
        let input = create_test_tensor();
        let output = ActivationFunctions::tanh(&input).unwrap();
        
        // All tanh outputs should be between -1 and 1
        for val in output.data.iter() {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
        
        // Tanh(0) should be 0
        assert_relative_eq!(output.data[[0, 2]], 0.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_gelu_activation() {
        let input = create_test_tensor();
        let output = ActivationFunctions::gelu(&input).unwrap();
        
        // GELU(0) should be 0
        assert_relative_eq!(output.data[[0, 2]], 0.0, epsilon = 1e-3);
        
        // GELU should be positive for positive inputs
        assert!(output.data[[1, 0]] > 0.0);
        assert!(output.data[[1, 1]] > 0.0);
        assert!(output.data[[1, 2]] > 0.0);
    }
    
    #[test]
    fn test_swish_activation() {
        let input = create_test_tensor();
        let output = ActivationFunctions::swish(&input).unwrap();
        
        // Swish(0) should be 0
        assert_relative_eq!(output.data[[0, 2]], 0.0, epsilon = 1e-6);
        
        // Swish can have negative values for negative inputs
        assert!(output.data[[0, 0]] < 0.0);
        assert!(output.data[[0, 1]] < 0.0);
    }
    
    #[test]
    fn test_softmax_activation() {
        let input = create_test_tensor();
        let output = ActivationFunctions::softmax(&input).unwrap();
        
        // Each row should sum to 1.0
        for batch_idx in 0..output.batch_size() {
            let row_sum: f32 = output.data.row(batch_idx).sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-6);
        }
        
        // All values should be positive
        for val in output.data.iter() {
            assert!(*val > 0.0);
        }
    }
    
    #[test]
    fn test_linear_activation() {
        let input = create_test_tensor();
        let output = ActivationFunctions::linear(&input).unwrap();
        
        // Linear should return identical tensor
        for i in 0..input.data.len() {
            assert_eq!(input.data.as_slice().unwrap()[i], output.data.as_slice().unwrap()[i]);
        }
    }
    
    #[tokio::test]
    async fn test_activation_layer() {
        let mut layer = ActivationLayer::new(ActivationType::ReLU);
        assert!(!layer.is_compiled());
        
        // Compile layer
        let output_dim = layer.compile(3).unwrap();
        assert_eq!(output_dim, 3); // Same as input for activation layers
        assert!(layer.is_compiled());
        
        // Test forward pass
        let input = create_test_tensor();
        let output = layer.forward(&input, DNNTrainingMode::Inference).await.unwrap();
        
        // Should be same as direct ReLU application
        let expected = ActivationFunctions::relu(&input).unwrap();
        assert_eq!(output.data, expected.data);
    }
    
    #[test]
    fn test_activation_factory() {
        let relu = ActivationFactory::relu();
        assert_eq!(relu.layer_type(), "ReLU");
        
        let sigmoid = ActivationFactory::sigmoid();
        assert_eq!(sigmoid.layer_type(), "Sigmoid");
        
        let softmax = ActivationFactory::softmax();
        assert_eq!(softmax.layer_type(), "Softmax");
    }
    
    #[test]
    fn test_activation_utils() {
        assert_eq!(ActivationUtils::from_string("relu").unwrap(), ActivationType::ReLU);
        assert_eq!(ActivationUtils::from_string("sigmoid").unwrap(), ActivationType::Sigmoid);
        assert_eq!(ActivationUtils::from_string("gelu").unwrap(), ActivationType::GELU);
        
        assert!(ActivationUtils::from_string("unknown").is_err());
        
        assert!(ActivationUtils::is_output_activation(ActivationType::Softmax));
        assert!(ActivationUtils::is_output_activation(ActivationType::Linear));
        assert!(!ActivationUtils::is_output_activation(ActivationType::ReLU));
    }
    
    #[tokio::test]
    async fn test_activation_derivatives() {
        let input = create_test_tensor();
        
        // Test ReLU derivative
        let relu_grad = ActivationFunctions::relu_derivative(&input).unwrap();
        assert_eq!(relu_grad.data[[0, 0]], 0.0); // Negative input
        assert_eq!(relu_grad.data[[1, 0]], 1.0); // Positive input
        
        // Test sigmoid derivative
        let sigmoid_grad = ActivationFunctions::sigmoid_derivative(&input).unwrap();
        
        // Sigmoid derivative should be positive and <= 0.25
        for val in sigmoid_grad.data.iter() {
            assert!(*val >= 0.0);
            assert!(*val <= 0.25);
        }
    }
    
    #[test]
    fn test_numerical_stability() {
        // Test extreme values
        let extreme_data = vec![-100.0, -50.0, 0.0, 50.0, 100.0, 1000.0];
        let shape = TensorShape::new_2d(2, 3);
        let extreme_input = DNNTensor::from_vec(extreme_data, &shape).unwrap();
        
        // Sigmoid should handle extreme values gracefully
        let sigmoid_out = ActivationFunctions::sigmoid(&extreme_input).unwrap();
        assert!(!sigmoid_out.has_invalid_values()); // No NaN or Inf
        
        // Values should be properly bounded
        for val in sigmoid_out.data.iter() {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
        
        // Softmax should handle extreme values
        let softmax_out = ActivationFunctions::softmax(&extreme_input).unwrap();
        assert!(!softmax_out.has_invalid_values());
        
        // Each row should still sum to 1.0
        for batch_idx in 0..softmax_out.batch_size() {
            let row_sum: f32 = softmax_out.data.row(batch_idx).sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-5);
        }
    }
}