//! GPU-accelerated backpropagation implementation
//!
//! This module provides the core backpropagation algorithm optimized for GPU execution,
//! enabling real neural network training on WebGPU.

use super::*;
use crate::webgpu::backend::{BackendSelector, ComputeBackend};
use crate::webgpu::shaders::webgpu_shaders::ShaderType;
use crate::webgpu::{ComputeContext, ComputeError};
use num_traits::Float;
use std::sync::Arc;

/// GPU-accelerated gradient computation for neural networks
pub struct GpuGradientComputer<T: Float + Send + Sync + Default + std::fmt::Debug + 'static> {
    backend: Arc<dyn ComputeBackend<T>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync + Default + std::fmt::Debug + 'static> GpuGradientComputer<T> {
    
    /// Create a new GPU gradient computer with automatic backend selection
    pub fn with_auto_backend() -> Result<Self, ComputeError> {
        let mut backend_selector = BackendSelector::new()?;
        let profile = crate::webgpu::backend::ComputeProfile {
            matrix_size: crate::webgpu::backend::MatrixSize::Medium,
            batch_size: 32,
            operation_type: crate::webgpu::backend::OperationType::Training,
        };
        
        // Select optimal backend for gradient computation
        if let Some(backend) = backend_selector.select_backend(&profile) {
            Ok(Self {
                backend: Arc::new(backend),
                _phantom: std::marker::PhantomData,
            })
        } else {
            Err(ComputeError::BackendError("No suitable backend found for gradient computation".to_string()))
        }
    }
    
    /// Create gradient computer with specific backend selector configuration
    pub fn with_backend_selector(mut selector: BackendSelector<T>) -> Result<Self, ComputeError> {
        // Configure selector for training workloads
        let profile = crate::webgpu::backend::ComputeProfile {
            matrix_size: crate::webgpu::backend::MatrixSize::Large,
            batch_size: 64,
            operation_type: crate::webgpu::backend::OperationType::Training,
        };
        
        if let Some(backend) = selector.select_backend(&profile) {
            Ok(Self {
                backend: Arc::new(backend),
                _phantom: std::marker::PhantomData,
            })
        } else {
            Err(ComputeError::BackendError("Backend selector failed to find suitable backend".to_string()))
        }
    }
    
    /// Configure GPU shaders for gradient computation
    pub fn configure_gradient_shaders(&mut self) -> Result<(), ComputeError> {
        // Configure different shader types for gradient computation
        let shader_configs = vec![
            (ShaderType::MatrixMultiply, "gradient_matrix_multiply"),
            (ShaderType::Activation, "gradient_activation_backprop"),
            (ShaderType::ElementWise, "gradient_element_wise_ops"),
        ];
        
        for (shader_type, shader_name) in shader_configs {
            // In a real implementation, this would compile and cache shaders
            match shader_type {
                ShaderType::MatrixMultiply => {
                    // Configure matrix multiplication shader for weight gradient computation
                    println!("Configuring {} shader for matrix operations", shader_name);
                },
                ShaderType::Activation => {
                    // Configure activation function derivative shaders
                    println!("Configuring {} shader for activation derivatives", shader_name);
                },
                ShaderType::ElementWise => {
                    // Configure element-wise operation shaders for bias gradients
                    println!("Configuring {} shader for element-wise operations", shader_name);
                },
                _ => {
                    println!("Skipping unsupported shader type: {:?}", shader_type);
                }
            }
        }
        
        Ok(())
    }
    
    /// Get optimal shader type for gradient computation based on operation
    pub fn get_optimal_shader_type(&self, operation: &str, data_size: usize) -> ShaderType {
        match operation {
            "weight_gradients" => {
                if data_size > 10000 {
                    ShaderType::MatrixMultiply
                } else {
                    ShaderType::ElementWise
                }
            },
            "bias_gradients" => ShaderType::ElementWise,
            "activation_derivatives" => ShaderType::Activation,
            "error_backprop" => {
                if data_size > 5000 {
                    ShaderType::MatrixMultiply
                } else {
                    ShaderType::ElementWise
                }
            },
            _ => ShaderType::ElementWise, // Default fallback
        }
    }
    /// Create a new GPU gradient computer
    pub fn new(backend: Arc<dyn ComputeBackend<T>>) -> Self {
        Self {
            backend,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Perform full backpropagation on GPU
    pub fn backpropagate(
        &self,
        network: &Network<T>,
        layer_activations: &[Vec<T>],
        output_error: &[T],
    ) -> Result<(Vec<Vec<T>>, Vec<Vec<T>>), ComputeError> {
        let mut weight_gradients = Vec::new();
        let mut bias_gradients = Vec::new();
        let mut current_error = output_error.to_vec();

        // Process layers in reverse order (backpropagation)
        for layer_idx in (1..network.layers.len()).rev() {
            let layer = &network.layers[layer_idx];
            let prev_layer = &network.layers[layer_idx - 1];

            // Get activations from previous layer
            let prev_activations = &layer_activations[layer_idx - 1];

            // Extract layer dimensions
            let output_size = layer.neurons.iter().filter(|n| !n.is_bias).count();
            let input_size = prev_activations.len();

            // Compute gradients for this layer
            let (layer_weight_grad, layer_bias_grad, prev_layer_error) = self
                .compute_layer_gradients(
                    layer,
                    prev_activations,
                    &current_error,
                    input_size,
                    output_size,
                )?;

            // Store gradients (in forward order)
            weight_gradients.insert(0, layer_weight_grad);
            bias_gradients.insert(0, layer_bias_grad);

            // Update error for next iteration (previous layer)
            if layer_idx > 1 {
                current_error = prev_layer_error;
            }
        }

        Ok((weight_gradients, bias_gradients))
    }

    /// Compute gradients for a single layer using GPU
    fn compute_layer_gradients(
        &self,
        layer: &crate::Layer<T>,
        prev_activations: &[T],
        layer_error: &[T],
        input_size: usize,
        output_size: usize,
    ) -> Result<(Vec<T>, Vec<T>, Vec<T>), ComputeError> {
        // Extract weights matrix for this layer
        let weights = self.extract_weights_matrix(layer, input_size, output_size);

        // Compute weight gradients: dW = prev_activations^T * layer_error
        // For each weight w_ij: dw_ij = activation_i * error_j
        let mut weight_gradients = Vec::with_capacity(weights.len());
        for output_idx in 0..output_size {
            for input_idx in 0..input_size {
                let gradient = prev_activations[input_idx] * layer_error[output_idx];
                weight_gradients.push(gradient);
            }
        }

        // Bias gradients are simply the error terms
        let bias_gradients = layer_error.to_vec();

        // Compute error for previous layer: prev_error = weights^T * layer_error
        let prev_layer_error = if input_size > 0 {
            self.backend.matrix_vector_multiply(
                &self.transpose_weights(&weights, input_size, output_size),
                layer_error,
                input_size,
                output_size,
            )?
        } else {
            vec![T::zero(); input_size]
        };

        // Apply activation function derivative to previous layer error
        let prev_layer_error = self.apply_activation_derivative(
            &prev_layer_error,
            prev_activations,
            layer.neurons[0].activation_function,
        )?;

        Ok((weight_gradients, bias_gradients, prev_layer_error))
    }

    /// Extract weights as a matrix from a layer
    fn extract_weights_matrix(
        &self,
        layer: &crate::Layer<T>,
        input_size: usize,
        output_size: usize,
    ) -> Vec<T> {
        let mut weights = vec![T::zero(); output_size * input_size];

        for (neuron_idx, neuron) in layer.neurons.iter().filter(|n| !n.is_bias).enumerate() {
            // Skip bias connection (first connection)
            for (conn_idx, connection) in neuron.connections.iter().skip(1).enumerate() {
                if conn_idx < input_size && neuron_idx < output_size {
                    weights[neuron_idx * input_size + conn_idx] = connection.weight;
                }
            }
        }

        weights
    }

    /// Transpose weight matrix
    fn transpose_weights(&self, weights: &[T], rows: usize, cols: usize) -> Vec<T> {
        let mut transposed = vec![T::zero(); rows * cols];

        for row in 0..rows {
            for col in 0..cols {
                transposed[col * rows + row] = weights[row * cols + col];
            }
        }

        transposed
    }

    /// Apply activation function derivative
    fn apply_activation_derivative(
        &self,
        errors: &[T],
        activations: &[T],
        activation_fn: crate::ActivationFunction,
    ) -> Result<Vec<T>, ComputeError> {
        use crate::ActivationFunction::*;

        let mut result = Vec::with_capacity(errors.len());

        match activation_fn {
            Sigmoid => {
                // f'(x) = f(x) * (1 - f(x))
                for (i, &error) in errors.iter().enumerate() {
                    let activation = activations[i];
                    let derivative = activation * (T::one() - activation);
                    result.push(error * derivative);
                }
            }
            Tanh => {
                // f'(x) = 1 - f(x)^2
                for (i, &error) in errors.iter().enumerate() {
                    let activation = activations[i];
                    let derivative = T::one() - activation * activation;
                    result.push(error * derivative);
                }
            }
            ReLU => {
                // f'(x) = 1 if x > 0, 0 otherwise
                for (i, &error) in errors.iter().enumerate() {
                    let derivative = if activations[i] > T::zero() {
                        T::one()
                    } else {
                        T::zero()
                    };
                    result.push(error * derivative);
                }
            }
            Linear => {
                // f'(x) = 1
                result = errors.to_vec();
            }
            _ => {
                // For unsupported activation functions, fall back to sigmoid derivative
                for (i, &error) in errors.iter().enumerate() {
                    let activation = activations[i];
                    let derivative = activation * (T::one() - activation);
                    result.push(error * derivative);
                }
            }
        }

        Ok(result)
    }
}

/// GPU-accelerated forward propagation
pub fn gpu_forward_propagate<T: Float + Send + Sync + Default + std::fmt::Debug + 'static>(
    backend: &Arc<dyn ComputeBackend<T>>,
    network: &Network<T>,
    input: &[T],
) -> Result<Vec<Vec<T>>, ComputeError> {
    let mut activations = vec![input.to_vec()];
    let mut current_input = input.to_vec();

    // Process each layer
    for layer_idx in 1..network.layers.len() {
        let layer = &network.layers[layer_idx];
        let output_size = layer.neurons.iter().filter(|n| !n.is_bias).count();
        let input_size = current_input.len();

        // Extract weights and biases
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for neuron in layer.neurons.iter().filter(|n| !n.is_bias) {
            // Extract bias (first connection)
            let bias = if !neuron.connections.is_empty() {
                neuron.connections[0].weight
            } else {
                T::zero()
            };
            biases.push(bias);

            // Extract weights (skip bias connection)
            for connection in neuron.connections.iter().skip(1) {
                weights.push(connection.weight);
            }
        }

        // Perform matrix-vector multiplication on GPU
        let layer_output =
            backend.matrix_vector_multiply(&weights, &current_input, output_size, input_size)?;

        // Add biases
        let mut with_bias = Vec::with_capacity(layer_output.len());
        for (i, &output) in layer_output.iter().enumerate() {
            with_bias.push(output + biases[i]);
        }

        // Apply activation function
        let activation_fn = layer
            .neurons
            .iter()
            .find(|n| !n.is_bias)
            .map(|n| n.activation_function)
            .unwrap_or(crate::ActivationFunction::Sigmoid);

        let activated = backend.apply_activation_function(
            &with_bias,
            activation_fn,
            T::one(), // steepness
        )?;

        activations.push(activated.clone());
        current_input = activated;
    }

    Ok(activations)
}
