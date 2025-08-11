//! High-performance optimizer implementations for zen-neural training
//!
//! This module provides zero-allocation, memory-efficient implementations of popular
//! optimization algorithms with SIMD acceleration where possible. All optimizers
//! implement the ZenOptimizer trait for unified usage in training pipelines.

use std::collections::HashMap;
use num_traits::Float;
use ndarray::{Array1, Array2};

#[cfg(feature = "simd")]
use crate::simd::SimdOps;

use crate::TrainingError;
use super::{ZenOptimizer, OptimizerType, NumericalUtils};

/// SGD with momentum optimizer
pub struct ZenSGD<T: Float> {
    learning_rate: T,
    momentum: T,
    weight_decay: T,
    dampening: T,
    nesterov: bool,
    
    // State buffers
    momentum_buffers: Vec<T>,
    initialized: bool,
}

impl<T: Float + Clone + Default> ZenSGD<T> {
    pub fn new(learning_rate: T, momentum: T) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay: T::zero(),
            dampening: T::zero(),
            nesterov: false,
            momentum_buffers: Vec::new(),
            initialized: false,
        }
    }
    
    pub fn with_weight_decay(mut self, weight_decay: T) -> Self {
        self.weight_decay = weight_decay;
        self
    }
    
    pub fn with_dampening(mut self, dampening: T) -> Self {
        self.dampening = dampening;
        self
    }
    
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl<T: Float + Clone + Default> ZenOptimizer<T> for ZenSGD<T> {
    fn initialize(&mut self, parameter_count: usize) -> Result<(), TrainingError> {
        self.momentum_buffers = vec![T::default(); parameter_count];
        self.initialized = true;
        Ok(())
    }
    
    fn update_parameters(&mut self, parameters: Vec<&mut T>, gradients: &[T]) -> Result<(), TrainingError> {
        if !self.initialized {
            return Err(TrainingError::TrainingFailed("Optimizer not initialized".to_string()));
        }
        
        if parameters.len() != gradients.len() || parameters.len() != self.momentum_buffers.len() {
            return Err(TrainingError::TrainingFailed("Parameter/gradient size mismatch".to_string()));
        }
        
        for (i, (param, &grad)) in parameters.into_iter().zip(gradients.iter()).enumerate() {
            // Apply weight decay to gradient
            let d_p = if self.weight_decay != T::zero() {
                grad + self.weight_decay * *param
            } else {
                grad
            };
            
            // Update momentum buffer
            if self.momentum != T::zero() {
                self.momentum_buffers[i] = self.momentum * self.momentum_buffers[i] 
                    + (T::one() - self.dampening) * d_p;
                
                // Apply update
                let update = if self.nesterov {
                    d_p + self.momentum * self.momentum_buffers[i]
                } else {
                    self.momentum_buffers[i]
                };
                
                *param = *param - self.learning_rate * update;
            } else {
                // Simple SGD without momentum
                *param = *param - self.learning_rate * d_p;
            }
        }
        
        Ok(())
    }
    
    fn set_learning_rate(&mut self, lr: T) {
        self.learning_rate = lr;
    }
    
    fn get_learning_rate(&self) -> T {
        self.learning_rate
    }
    
    fn reset(&mut self) {
        for buffer in &mut self.momentum_buffers {
            *buffer = T::default();
        }
    }
}

/// Adam optimizer with bias correction
pub struct ZenAdam<T: Float> {
    learning_rate: T,
    beta1: T,
    beta2: T,
    epsilon: T,
    weight_decay: T,
    amsgrad: bool,
    
    // State buffers
    m_buffers: Vec<T>,  // First moment
    v_buffers: Vec<T>,  // Second moment
    v_max_buffers: Vec<T>,  // Max second moment (AMSGrad)
    step_count: u64,
    initialized: bool,
}

impl<T: Float + Clone + Default> ZenAdam<T> {
    pub fn new(learning_rate: T, beta1: T, beta2: T, epsilon: T) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay: T::zero(),
            amsgrad: false,
            m_buffers: Vec::new(),
            v_buffers: Vec::new(),
            v_max_buffers: Vec::new(),
            step_count: 0,
            initialized: false,
        }
    }
    
    pub fn with_weight_decay(mut self, weight_decay: T) -> Self {
        self.weight_decay = weight_decay;
        self
    }
    
    pub fn with_amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }
}

impl<T: Float + Clone + Default> ZenOptimizer<T> for ZenAdam<T> {
    fn initialize(&mut self, parameter_count: usize) -> Result<(), TrainingError> {
        self.m_buffers = vec![T::default(); parameter_count];
        self.v_buffers = vec![T::default(); parameter_count];
        if self.amsgrad {
            self.v_max_buffers = vec![T::default(); parameter_count];
        }
        self.initialized = true;
        self.step_count = 0;
        Ok(())
    }
    
    fn update_parameters(&mut self, parameters: Vec<&mut T>, gradients: &[T]) -> Result<(), TrainingError> {
        if !self.initialized {
            return Err(TrainingError::TrainingFailed("Optimizer not initialized".to_string()));
        }
        
        if parameters.len() != gradients.len() || parameters.len() != self.m_buffers.len() {
            return Err(TrainingError::TrainingFailed("Parameter/gradient size mismatch".to_string()));
        }
        
        self.step_count += 1;
        
        // Bias correction terms
        let bias_correction1 = T::one() - self.beta1.powi(self.step_count as i32);
        let bias_correction2 = T::one() - self.beta2.powi(self.step_count as i32);
        
        // Corrected learning rate
        let corrected_lr = self.learning_rate * bias_correction2.sqrt() / bias_correction1;
        
        for (i, (param, &grad)) in parameters.into_iter().zip(gradients.iter()).enumerate() {
            // Apply weight decay to gradient (L2 regularization)
            let d_p = if self.weight_decay != T::zero() {
                grad + self.weight_decay * *param
            } else {
                grad
            };
            
            // Update biased first moment estimate
            self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (T::one() - self.beta1) * d_p;
            
            // Update biased second raw moment estimate
            self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (T::one() - self.beta2) * d_p * d_p;
            
            let denominator = if self.amsgrad {
                // AMSGrad: use max of current and previous v
                self.v_max_buffers[i] = if self.v_buffers[i] > self.v_max_buffers[i] {
                    self.v_buffers[i]
                } else {
                    self.v_max_buffers[i]
                };
                self.v_max_buffers[i].sqrt() + self.epsilon
            } else {
                self.v_buffers[i].sqrt() + self.epsilon
            };
            
            // Update parameter
            let update = corrected_lr * self.m_buffers[i] / denominator;
            *param = *param - update;
            
            // Check for numerical issues
            if !NumericalUtils::is_finite(*param) {
                return Err(TrainingError::TrainingFailed(
                    "Parameter became NaN or infinite during Adam update".to_string()
                ));
            }
        }
        
        Ok(())
    }
    
    fn set_learning_rate(&mut self, lr: T) {
        self.learning_rate = lr;
    }
    
    fn get_learning_rate(&self) -> T {
        self.learning_rate
    }
    
    fn reset(&mut self) {
        for buffer in &mut self.m_buffers {
            *buffer = T::default();
        }
        for buffer in &mut self.v_buffers {
            *buffer = T::default();
        }
        if self.amsgrad {
            for buffer in &mut self.v_max_buffers {
                *buffer = T::default();
            }
        }
        self.step_count = 0;
    }
}

/// AdamW optimizer with decoupled weight decay
pub struct ZenAdamW<T: Float> {
    learning_rate: T,
    beta1: T,
    beta2: T,
    epsilon: T,
    weight_decay: T,
    amsgrad: bool,
    
    // State buffers
    m_buffers: Vec<T>,
    v_buffers: Vec<T>,
    v_max_buffers: Vec<T>,
    step_count: u64,
    initialized: bool,
}

impl<T: Float + Clone + Default> ZenAdamW<T> {
    pub fn new(learning_rate: T, beta1: T, beta2: T, epsilon: T, weight_decay: T) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            amsgrad: false,
            m_buffers: Vec::new(),
            v_buffers: Vec::new(),
            v_max_buffers: Vec::new(),
            step_count: 0,
            initialized: false,
        }
    }
    
    pub fn with_amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }
}

impl<T: Float + Clone + Default> ZenOptimizer<T> for ZenAdamW<T> {
    fn initialize(&mut self, parameter_count: usize) -> Result<(), TrainingError> {
        self.m_buffers = vec![T::default(); parameter_count];
        self.v_buffers = vec![T::default(); parameter_count];
        if self.amsgrad {
            self.v_max_buffers = vec![T::default(); parameter_count];
        }
        self.initialized = true;
        self.step_count = 0;
        Ok(())
    }
    
    fn update_parameters(&mut self, parameters: Vec<&mut T>, gradients: &[T]) -> Result<(), TrainingError> {
        if !self.initialized {
            return Err(TrainingError::TrainingFailed("Optimizer not initialized".to_string()));
        }
        
        if parameters.len() != gradients.len() || parameters.len() != self.m_buffers.len() {
            return Err(TrainingError::TrainingFailed("Parameter/gradient size mismatch".to_string()));
        }
        
        self.step_count += 1;
        
        // Bias correction terms
        let bias_correction1 = T::one() - self.beta1.powi(self.step_count as i32);
        let bias_correction2 = T::one() - self.beta2.powi(self.step_count as i32);
        
        // Corrected learning rate
        let corrected_lr = self.learning_rate * bias_correction2.sqrt() / bias_correction1;
        
        for (i, (param, &grad)) in parameters.into_iter().zip(gradients.iter()).enumerate() {
            // Update biased first moment estimate (without weight decay in gradient)
            self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (T::one() - self.beta1) * grad;
            
            // Update biased second raw moment estimate
            self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (T::one() - self.beta2) * grad * grad;
            
            let denominator = if self.amsgrad {
                self.v_max_buffers[i] = if self.v_buffers[i] > self.v_max_buffers[i] {
                    self.v_buffers[i]
                } else {
                    self.v_max_buffers[i]
                };
                self.v_max_buffers[i].sqrt() + self.epsilon
            } else {
                self.v_buffers[i].sqrt() + self.epsilon
            };
            
            // Adam update
            let adam_update = corrected_lr * self.m_buffers[i] / denominator;
            
            // Decoupled weight decay (applied directly to parameters)
            let weight_decay_update = self.learning_rate * self.weight_decay * *param;
            
            // Combined update
            *param = *param - adam_update - weight_decay_update;
            
            if !NumericalUtils::is_finite(*param) {
                return Err(TrainingError::TrainingFailed(
                    "Parameter became NaN or infinite during AdamW update".to_string()
                ));
            }
        }
        
        Ok(())
    }
    
    fn set_learning_rate(&mut self, lr: T) {
        self.learning_rate = lr;
    }
    
    fn get_learning_rate(&self) -> T {
        self.learning_rate
    }
    
    fn reset(&mut self) {
        for buffer in &mut self.m_buffers {
            *buffer = T::default();
        }
        for buffer in &mut self.v_buffers {
            *buffer = T::default();
        }
        if self.amsgrad {
            for buffer in &mut self.v_max_buffers {
                *buffer = T::default();
            }
        }
        self.step_count = 0;
    }
}

/// RMSprop optimizer
pub struct ZenRMSprop<T: Float> {
    learning_rate: T,
    alpha: T,
    epsilon: T,
    weight_decay: T,
    momentum: T,
    centered: bool,
    
    // State buffers
    square_avg: Vec<T>,
    momentum_buffer: Vec<T>,
    grad_avg: Vec<T>, // For centered RMSprop
    initialized: bool,
}

impl<T: Float + Clone + Default> ZenRMSprop<T> {
    pub fn new(learning_rate: T, alpha: T, epsilon: T) -> Self {
        Self {
            learning_rate,
            alpha,
            epsilon,
            weight_decay: T::zero(),
            momentum: T::zero(),
            centered: false,
            square_avg: Vec::new(),
            momentum_buffer: Vec::new(),
            grad_avg: Vec::new(),
            initialized: false,
        }
    }
    
    pub fn with_weight_decay(mut self, weight_decay: T) -> Self {
        self.weight_decay = weight_decay;
        self
    }
    
    pub fn with_momentum(mut self, momentum: T) -> Self {
        self.momentum = momentum;
        self
    }
    
    pub fn with_centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }
}

impl<T: Float + Clone + Default> ZenOptimizer<T> for ZenRMSprop<T> {
    fn initialize(&mut self, parameter_count: usize) -> Result<(), TrainingError> {
        self.square_avg = vec![T::default(); parameter_count];
        if self.momentum != T::zero() {
            self.momentum_buffer = vec![T::default(); parameter_count];
        }
        if self.centered {
            self.grad_avg = vec![T::default(); parameter_count];
        }
        self.initialized = true;
        Ok(())
    }
    
    fn update_parameters(&mut self, parameters: Vec<&mut T>, gradients: &[T]) -> Result<(), TrainingError> {
        if !self.initialized {
            return Err(TrainingError::TrainingFailed("Optimizer not initialized".to_string()));
        }
        
        if parameters.len() != gradients.len() || parameters.len() != self.square_avg.len() {
            return Err(TrainingError::TrainingFailed("Parameter/gradient size mismatch".to_string()));
        }
        
        for (i, (param, &grad)) in parameters.into_iter().zip(gradients.iter()).enumerate() {
            // Apply weight decay
            let d_p = if self.weight_decay != T::zero() {
                grad + self.weight_decay * *param
            } else {
                grad
            };
            
            // Update moving average of squared gradients
            self.square_avg[i] = self.alpha * self.square_avg[i] + (T::one() - self.alpha) * d_p * d_p;
            
            let avg = if self.centered {
                // Update moving average of gradients
                self.grad_avg[i] = self.alpha * self.grad_avg[i] + (T::one() - self.alpha) * d_p;
                
                // Centered RMSprop
                let variance = self.square_avg[i] - self.grad_avg[i] * self.grad_avg[i];
                variance.sqrt() + self.epsilon
            } else {
                // Standard RMSprop
                self.square_avg[i].sqrt() + self.epsilon
            };
            
            let update = self.learning_rate * d_p / avg;
            
            if self.momentum != T::zero() {
                // Apply momentum
                self.momentum_buffer[i] = self.momentum * self.momentum_buffer[i] + update;
                *param = *param - self.momentum_buffer[i];
            } else {
                *param = *param - update;
            }
            
            if !NumericalUtils::is_finite(*param) {
                return Err(TrainingError::TrainingFailed(
                    "Parameter became NaN or infinite during RMSprop update".to_string()
                ));
            }
        }
        
        Ok(())
    }
    
    fn set_learning_rate(&mut self, lr: T) {
        self.learning_rate = lr;
    }
    
    fn get_learning_rate(&self) -> T {
        self.learning_rate
    }
    
    fn reset(&mut self) {
        for buffer in &mut self.square_avg {
            *buffer = T::default();
        }
        for buffer in &mut self.momentum_buffer {
            *buffer = T::default();
        }
        for buffer in &mut self.grad_avg {
            *buffer = T::default();
        }
    }
}

/// Adagrad optimizer
pub struct ZenAdagrad<T: Float> {
    learning_rate: T,
    epsilon: T,
    weight_decay: T,
    lr_decay: T,
    
    // State buffers
    sum_squares: Vec<T>,
    step_count: u64,
    initialized: bool,
}

impl<T: Float + Clone + Default> ZenAdagrad<T> {
    pub fn new(learning_rate: T, epsilon: T) -> Self {
        Self {
            learning_rate,
            epsilon,
            weight_decay: T::zero(),
            lr_decay: T::zero(),
            sum_squares: Vec::new(),
            step_count: 0,
            initialized: false,
        }
    }
    
    pub fn with_weight_decay(mut self, weight_decay: T) -> Self {
        self.weight_decay = weight_decay;
        self
    }
    
    pub fn with_lr_decay(mut self, lr_decay: T) -> Self {
        self.lr_decay = lr_decay;
        self
    }
}

impl<T: Float + Clone + Default> ZenOptimizer<T> for ZenAdagrad<T> {
    fn initialize(&mut self, parameter_count: usize) -> Result<(), TrainingError> {
        self.sum_squares = vec![T::default(); parameter_count];
        self.initialized = true;
        self.step_count = 0;
        Ok(())
    }
    
    fn update_parameters(&mut self, parameters: Vec<&mut T>, gradients: &[T]) -> Result<(), TrainingError> {
        if !self.initialized {
            return Err(TrainingError::TrainingFailed("Optimizer not initialized".to_string()));
        }
        
        if parameters.len() != gradients.len() || parameters.len() != self.sum_squares.len() {
            return Err(TrainingError::TrainingFailed("Parameter/gradient size mismatch".to_string()));
        }
        
        self.step_count += 1;
        
        // Effective learning rate with decay
        let effective_lr = self.learning_rate / 
            (T::one() + T::from(self.step_count - 1).unwrap() * self.lr_decay);
        
        for (i, (param, &grad)) in parameters.into_iter().zip(gradients.iter()).enumerate() {
            // Apply weight decay
            let d_p = if self.weight_decay != T::zero() {
                grad + self.weight_decay * *param
            } else {
                grad
            };
            
            // Update sum of squared gradients
            self.sum_squares[i] = self.sum_squares[i] + d_p * d_p;
            
            // Compute adaptive learning rate
            let std = self.sum_squares[i].sqrt() + self.epsilon;
            let update = effective_lr * d_p / std;
            
            *param = *param - update;
            
            if !NumericalUtils::is_finite(*param) {
                return Err(TrainingError::TrainingFailed(
                    "Parameter became NaN or infinite during Adagrad update".to_string()
                ));
            }
        }
        
        Ok(())
    }
    
    fn set_learning_rate(&mut self, lr: T) {
        self.learning_rate = lr;
    }
    
    fn get_learning_rate(&self) -> T {
        self.learning_rate
    }
    
    fn reset(&mut self) {
        for buffer in &mut self.sum_squares {
            *buffer = T::default();
        }
        self.step_count = 0;
    }
}

/// Factory function to create optimizers
pub fn create_optimizer<T: Float + Clone + Default + Send + Sync>(
    optimizer_type: &OptimizerType<T>
) -> Result<Box<dyn ZenOptimizer<T>>, TrainingError> {
    match optimizer_type {
        OptimizerType::SGD { momentum } => {
            Ok(Box::new(ZenSGD::new(T::from(0.01).unwrap(), *momentum)))
        }
        OptimizerType::Adam { beta1, beta2, epsilon } => {
            Ok(Box::new(ZenAdam::new(T::from(0.001).unwrap(), *beta1, *beta2, *epsilon)))
        }
        OptimizerType::AdamW { beta1, beta2, epsilon } => {
            Ok(Box::new(ZenAdamW::new(
                T::from(0.001).unwrap(), 
                *beta1, 
                *beta2, 
                *epsilon, 
                T::from(0.01).unwrap()
            )))
        }
        OptimizerType::RMSprop { alpha, epsilon } => {
            Ok(Box::new(ZenRMSprop::new(T::from(0.01).unwrap(), *alpha, *epsilon)))
        }
        OptimizerType::Adagrad { epsilon } => {
            Ok(Box::new(ZenAdagrad::new(T::from(0.01).unwrap(), *epsilon)))
        }
        _ => Err(TrainingError::TrainingFailed(
            "Optimizer type not yet implemented".to_string()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sgd_optimizer() {
        let mut sgd = ZenSGD::new(0.1f32, 0.9);
        sgd.initialize(3).unwrap();
        
        let mut params = vec![1.0f32, 2.0f32, 3.0f32];
        let param_refs: Vec<&mut f32> = params.iter_mut().collect();
        let gradients = vec![0.1f32, 0.2f32, 0.3f32];
        
        sgd.update_parameters(param_refs, &gradients).unwrap();
        
        // First update should be: param = param - lr * grad
        assert!((params[0] - 0.99).abs() < 1e-6);
        assert!((params[1] - 1.98).abs() < 1e-6);
        assert!((params[2] - 2.97).abs() < 1e-6);
    }
    
    #[test]
    fn test_adam_optimizer() {
        let mut adam = ZenAdam::new(0.001f32, 0.9, 0.999, 1e-8);
        adam.initialize(2).unwrap();
        
        let mut params = vec![1.0f32, 2.0f32];
        let param_refs: Vec<&mut f32> = params.iter_mut().collect();
        let gradients = vec![0.1f32, 0.2f32];
        
        adam.update_parameters(param_refs, &gradients).unwrap();
        
        // Parameters should be updated (exact values depend on Adam's complex math)
        assert!(params[0] < 1.0); // Should decrease
        assert!(params[1] < 2.0); // Should decrease
    }
    
    #[test]
    fn test_adamw_optimizer() {
        let mut adamw = ZenAdamW::new(0.001f32, 0.9, 0.999, 1e-8, 0.01);
        adamw.initialize(2).unwrap();
        
        let mut params = vec![1.0f32, 2.0f32];
        let param_refs: Vec<&mut f32> = params.iter_mut().collect();
        let gradients = vec![0.1f32, 0.2f32];
        
        adamw.update_parameters(param_refs, &gradients).unwrap();
        
        // Parameters should be updated with both Adam and weight decay
        assert!(params[0] < 1.0);
        assert!(params[1] < 2.0);
    }
    
    #[test]
    fn test_rmsprop_optimizer() {
        let mut rmsprop = ZenRMSprop::new(0.01f32, 0.99, 1e-8);
        rmsprop.initialize(2).unwrap();
        
        let mut params = vec![1.0f32, 2.0f32];
        let param_refs: Vec<&mut f32> = params.iter_mut().collect();
        let gradients = vec![0.1f32, 0.2f32];
        
        rmsprop.update_parameters(param_refs, &gradients).unwrap();
        
        assert!(params[0] < 1.0);
        assert!(params[1] < 2.0);
    }
    
    #[test]
    fn test_adagrad_optimizer() {
        let mut adagrad = ZenAdagrad::new(0.01f32, 1e-8);
        adagrad.initialize(2).unwrap();
        
        let mut params = vec![1.0f32, 2.0f32];
        let param_refs: Vec<&mut f32> = params.iter_mut().collect();
        let gradients = vec![0.1f32, 0.2f32];
        
        adagrad.update_parameters(param_refs, &gradients).unwrap();
        
        assert!(params[0] < 1.0);
        assert!(params[1] < 2.0);
    }
    
    #[test]
    fn test_optimizer_learning_rate() {
        let mut adam = ZenAdam::new(0.001f32, 0.9, 0.999, 1e-8);
        assert_eq!(adam.get_learning_rate(), 0.001);
        
        adam.set_learning_rate(0.01);
        assert_eq!(adam.get_learning_rate(), 0.01);
    }
    
    #[test]
    fn test_optimizer_reset() {
        let mut sgd = ZenSGD::new(0.1f32, 0.9);
        sgd.initialize(2).unwrap();
        
        // Update once to populate momentum buffers
        let mut params = vec![1.0f32, 2.0f32];
        let param_refs: Vec<&mut f32> = params.iter_mut().collect();
        let gradients = vec![0.1f32, 0.2f32];
        sgd.update_parameters(param_refs, &gradients).unwrap();
        
        // Reset should clear momentum buffers
        sgd.reset();
        
        // After reset, momentum buffers should be zero
        assert_eq!(sgd.momentum_buffers[0], 0.0);
        assert_eq!(sgd.momentum_buffers[1], 0.0);
    }
}