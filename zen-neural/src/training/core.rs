//! Core training infrastructure for zen-neural
//!
//! This module provides the fundamental building blocks for neural network training,
//! including gradient computation, automatic differentiation, and core training algorithms.

use std::collections::HashMap;
use num_traits::Float;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::TrainingError;
use super::{ZenLRScheduler, LRScheduleType, LRScheduleConfig};

/// Gradient computation utilities
pub struct GradientComputation;

impl GradientComputation {
    /// Compute gradients using automatic differentiation (simplified version)
    /// In a full implementation, this would use a proper AD system
    pub fn compute_gradients<T: Float + Clone>(
        parameters: &[T],
        loss_fn: impl Fn(&[T]) -> T,
        epsilon: T,
    ) -> Vec<T> {
        let mut gradients = Vec::with_capacity(parameters.len());
        
        for i in 0..parameters.len() {
            // Forward difference approximation
            let mut params_plus = parameters.to_vec();
            let mut params_minus = parameters.to_vec();
            
            params_plus[i] = params_plus[i] + epsilon;
            params_minus[i] = params_minus[i] - epsilon;
            
            let loss_plus = loss_fn(&params_plus);
            let loss_minus = loss_fn(&params_minus);
            
            let gradient = (loss_plus - loss_minus) / (T::from(2.0).unwrap() * epsilon);
            gradients.push(gradient);
        }
        
        gradients
    }
    
    /// Compute gradients using backpropagation for neural networks
    /// This is a simplified implementation - a full version would handle all activation functions
    pub fn backpropagate<T: Float + Clone>(
        activations: &[Vec<T>],
        weights: &[Array2<T>],
        biases: &[Array1<T>],
        target: &[T],
        loss_gradient: impl Fn(T, T) -> T,
    ) -> (Vec<Array2<T>>, Vec<Array1<T>>) {
        let num_layers = weights.len();
        let mut weight_gradients = Vec::with_capacity(num_layers);
        let mut bias_gradients = Vec::with_capacity(num_layers);
        
        // Initialize gradients
        for i in 0..num_layers {
            weight_gradients.push(Array2::zeros(weights[i].dim()));
            bias_gradients.push(Array1::zeros(biases[i].len()));
        }
        
        // Compute output layer error
        let output_layer = &activations[activations.len() - 1];
        let mut errors = Vec::with_capacity(output_layer.len());
        
        for (i, (&actual, &target_val)) in output_layer.iter().zip(target.iter()).enumerate() {
            let error = loss_gradient(actual, target_val) * Self::sigmoid_derivative(actual);
            errors.push(error);
        }
        
        // Backpropagate errors
        for layer_idx in (0..num_layers).rev() {
            let current_activations = &activations[layer_idx];
            let next_activations = &activations[layer_idx + 1];
            
            // Update gradients for this layer
            for (i, &error) in errors.iter().enumerate() {
                bias_gradients[layer_idx][i] = error;
                
                for (j, &activation) in current_activations.iter().enumerate() {
                    weight_gradients[layer_idx][[i, j]] = error * activation;
                }
            }
            
            // Compute errors for previous layer (if not input layer)
            if layer_idx > 0 {
                let mut prev_errors = vec![T::zero(); current_activations.len()];
                
                for (j, &current_activation) in current_activations.iter().enumerate() {
                    let mut error_sum = T::zero();
                    
                    for (i, &next_error) in errors.iter().enumerate() {
                        error_sum = error_sum + next_error * weights[layer_idx][[i, j]];
                    }
                    
                    prev_errors[j] = error_sum * Self::sigmoid_derivative(current_activation);
                }
                
                errors = prev_errors;
            }
        }
        
        (weight_gradients, bias_gradients)
    }
    
    /// Sigmoid activation function
    fn sigmoid<T: Float>(x: T) -> T {
        T::one() / (T::one() + (-x).exp())
    }
    
    /// Sigmoid derivative
    fn sigmoid_derivative<T: Float>(output: T) -> T {
        output * (T::one() - output)
    }
    
    /// ReLU activation function
    fn relu<T: Float>(x: T) -> T {
        if x > T::zero() { x } else { T::zero() }
    }
    
    /// ReLU derivative
    fn relu_derivative<T: Float>(x: T) -> T {
        if x > T::zero() { T::one() } else { T::zero() }
    }
    
    /// Tanh activation function
    fn tanh<T: Float>(x: T) -> T {
        x.tanh()
    }
    
    /// Tanh derivative
    fn tanh_derivative<T: Float>(output: T) -> T {
        T::one() - output * output
    }
}

/// Learning rate schedulers
pub struct ExponentialLRScheduler<T: Float> {
    initial_lr: T,
    current_lr: T,
    decay_rate: T,
    step_count: u32,
}

impl<T: Float> ExponentialLRScheduler<T> {
    pub fn new(initial_lr: T, decay_rate: T) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            decay_rate,
            step_count: 0,
        }
    }
}

impl<T: Float> ZenLRScheduler<T> for ExponentialLRScheduler<T> {
    fn step(&mut self, _metric: T) {
        self.step_count += 1;
        self.current_lr = self.initial_lr * self.decay_rate.powi(self.step_count as i32);
    }
    
    fn get_learning_rate(&self) -> T {
        self.current_lr
    }
    
    fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.step_count = 0;
    }
}

pub struct StepLRScheduler<T: Float> {
    initial_lr: T,
    current_lr: T,
    step_size: u32,
    gamma: T,
    step_count: u32,
}

impl<T: Float> StepLRScheduler<T> {
    pub fn new(initial_lr: T, step_size: u32, gamma: T) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            step_size,
            gamma,
            step_count: 0,
        }
    }
}

impl<T: Float> ZenLRScheduler<T> for StepLRScheduler<T> {
    fn step(&mut self, _metric: T) {
        self.step_count += 1;
        if self.step_count % self.step_size == 0 {
            self.current_lr = self.current_lr * self.gamma;
        }
    }
    
    fn get_learning_rate(&self) -> T {
        self.current_lr
    }
    
    fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.step_count = 0;
    }
}

pub struct CosineAnnealingLRScheduler<T: Float> {
    initial_lr: T,
    current_lr: T,
    t_max: u32,
    eta_min: T,
    step_count: u32,
}

impl<T: Float> CosineAnnealingLRScheduler<T> {
    pub fn new(initial_lr: T, t_max: u32, eta_min: T) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            t_max,
            eta_min,
            step_count: 0,
        }
    }
}

impl<T: Float> ZenLRScheduler<T> for CosineAnnealingLRScheduler<T> {
    fn step(&mut self, _metric: T) {
        self.step_count += 1;
        let t = T::from(self.step_count).unwrap();
        let t_max = T::from(self.t_max).unwrap();
        let pi = T::from(std::f64::consts::PI).unwrap();
        
        let cos_term = (T::one() + (pi * t / t_max).cos()) / T::from(2.0).unwrap();
        self.current_lr = self.eta_min + (self.initial_lr - self.eta_min) * cos_term;
    }
    
    fn get_learning_rate(&self) -> T {
        self.current_lr
    }
    
    fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.step_count = 0;
    }
}

pub struct ReduceOnPlateauLRScheduler<T: Float> {
    initial_lr: T,
    current_lr: T,
    factor: T,
    patience: u32,
    threshold: T,
    best_metric: T,
    epochs_without_improvement: u32,
    mode_min: bool,
}

impl<T: Float> ReduceOnPlateauLRScheduler<T> {
    pub fn new(initial_lr: T, factor: T, patience: u32, threshold: T, mode_min: bool) -> Self {
        let best_metric = if mode_min {
            T::from(f64::INFINITY).unwrap()
        } else {
            T::from(f64::NEG_INFINITY).unwrap()
        };
        
        Self {
            initial_lr,
            current_lr: initial_lr,
            factor,
            patience,
            threshold,
            best_metric,
            epochs_without_improvement: 0,
            mode_min,
        }
    }
}

impl<T: Float> ZenLRScheduler<T> for ReduceOnPlateauLRScheduler<T> {
    fn step(&mut self, metric: T) {
        let improved = if self.mode_min {
            metric < self.best_metric - self.threshold
        } else {
            metric > self.best_metric + self.threshold
        };
        
        if improved {
            self.best_metric = metric;
            self.epochs_without_improvement = 0;
        } else {
            self.epochs_without_improvement += 1;
            
            if self.epochs_without_improvement >= self.patience {
                self.current_lr = self.current_lr * self.factor;
                self.epochs_without_improvement = 0;
            }
        }
    }
    
    fn get_learning_rate(&self) -> T {
        self.current_lr
    }
    
    fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.epochs_without_improvement = 0;
        self.best_metric = if self.mode_min {
            T::from(f64::INFINITY).unwrap()
        } else {
            T::from(f64::NEG_INFINITY).unwrap()
        };
    }
}

/// Factory function to create learning rate schedulers
pub fn create_lr_scheduler<T: Float>(
    config: &LRScheduleConfig<T>, 
    initial_lr: T
) -> Result<Box<dyn ZenLRScheduler<T>>, TrainingError> {
    match &config.schedule_type {
        LRScheduleType::Exponential { decay_rate } => {
            Ok(Box::new(ExponentialLRScheduler::new(initial_lr, *decay_rate)))
        }
        LRScheduleType::StepLR { step_size, gamma } => {
            Ok(Box::new(StepLRScheduler::new(initial_lr, *step_size, *gamma)))
        }
        LRScheduleType::CosineAnnealing { t_max, eta_min } => {
            Ok(Box::new(CosineAnnealingLRScheduler::new(initial_lr, *t_max, *eta_min)))
        }
        LRScheduleType::ReduceOnPlateau { factor, patience, threshold } => {
            Ok(Box::new(ReduceOnPlateauLRScheduler::new(
                initial_lr, *factor, *patience, *threshold, true
            )))
        }
    }
}

/// Numerical utilities for training
pub struct NumericalUtils;

impl NumericalUtils {
    /// Check if a value is finite and not NaN
    pub fn is_finite<T: Float>(value: T) -> bool {
        value.is_finite() && !value.is_nan()
    }
    
    /// Clamp a value between min and max
    pub fn clamp<T: Float>(value: T, min_val: T, max_val: T) -> T {
        if value < min_val {
            min_val
        } else if value > max_val {
            max_val
        } else {
            value
        }
    }
    
    /// Compute the L2 norm of a vector
    pub fn l2_norm<T: Float>(vector: &[T]) -> T {
        vector.iter()
            .map(|x| *x * *x)
            .fold(T::zero(), |acc, x| acc + x)
            .sqrt()
    }
    
    /// Compute the L1 norm of a vector
    pub fn l1_norm<T: Float>(vector: &[T]) -> T {
        vector.iter()
            .map(|x| x.abs())
            .fold(T::zero(), |acc, x| acc + x)
    }
    
    /// Safe division with epsilon
    pub fn safe_divide<T: Float>(numerator: T, denominator: T, epsilon: T) -> T {
        numerator / (denominator + epsilon)
    }
    
    /// Exponential moving average
    pub fn exponential_moving_average<T: Float>(
        current: T,
        new_value: T,
        momentum: T,
    ) -> T {
        momentum * current + (T::one() - momentum) * new_value
    }
}

/// Training statistics collector
pub struct TrainingStats<T: Float> {
    pub gradient_norms: Vec<T>,
    pub parameter_norms: Vec<T>,
    pub loss_history: Vec<T>,
    pub learning_rates: Vec<T>,
    pub update_magnitudes: Vec<T>,
}

impl<T: Float + Clone + Default> TrainingStats<T> {
    pub fn new() -> Self {
        Self {
            gradient_norms: Vec::new(),
            parameter_norms: Vec::new(),
            loss_history: Vec::new(),
            learning_rates: Vec::new(),
            update_magnitudes: Vec::new(),
        }
    }
    
    pub fn record_gradient_norm(&mut self, norm: T) {
        self.gradient_norms.push(norm);
    }
    
    pub fn record_parameter_norm(&mut self, norm: T) {
        self.parameter_norms.push(norm);
    }
    
    pub fn record_loss(&mut self, loss: T) {
        self.loss_history.push(loss);
    }
    
    pub fn record_learning_rate(&mut self, lr: T) {
        self.learning_rates.push(lr);
    }
    
    pub fn record_update_magnitude(&mut self, magnitude: T) {
        self.update_magnitudes.push(magnitude);
    }
    
    pub fn get_average_gradient_norm(&self) -> Option<T> {
        if self.gradient_norms.is_empty() {
            None
        } else {
            let sum = self.gradient_norms.iter()
                .fold(T::zero(), |acc, x| acc + *x);
            Some(sum / T::from(self.gradient_norms.len()).unwrap())
        }
    }
    
    pub fn get_loss_trend(&self) -> Option<T> {
        if self.loss_history.len() < 2 {
            None
        } else {
            let n = self.loss_history.len();
            let recent_loss = self.loss_history[n - 1];
            let older_loss = self.loss_history[n - 2];
            Some(recent_loss - older_loss)
        }
    }
    
    pub fn clear(&mut self) {
        self.gradient_norms.clear();
        self.parameter_norms.clear();
        self.loss_history.clear();
        self.learning_rates.clear();
        self.update_magnitudes.clear();
    }
}

/// Activation functions and their derivatives
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    Tanh,
    LeakyReLU(f32),
    ELU(f32),
    Swish,
}

impl ActivationFunction {
    pub fn apply<T: Float>(&self, x: T) -> T {
        match self {
            ActivationFunction::Sigmoid => {
                T::one() / (T::one() + (-x).exp())
            }
            ActivationFunction::ReLU => {
                if x > T::zero() { x } else { T::zero() }
            }
            ActivationFunction::Tanh => {
                x.tanh()
            }
            ActivationFunction::LeakyReLU(alpha) => {
                let alpha_t = T::from(*alpha).unwrap();
                if x > T::zero() { x } else { alpha_t * x }
            }
            ActivationFunction::ELU(alpha) => {
                let alpha_t = T::from(*alpha).unwrap();
                if x > T::zero() { x } else { alpha_t * (x.exp() - T::one()) }
            }
            ActivationFunction::Swish => {
                let sigmoid = T::one() / (T::one() + (-x).exp());
                x * sigmoid
            }
        }
    }
    
    pub fn derivative<T: Float>(&self, output: T) -> T {
        match self {
            ActivationFunction::Sigmoid => {
                output * (T::one() - output)
            }
            ActivationFunction::ReLU => {
                if output > T::zero() { T::one() } else { T::zero() }
            }
            ActivationFunction::Tanh => {
                T::one() - output * output
            }
            ActivationFunction::LeakyReLU(alpha) => {
                let alpha_t = T::from(*alpha).unwrap();
                if output > T::zero() { T::one() } else { alpha_t }
            }
            ActivationFunction::ELU(alpha) => {
                let alpha_t = T::from(*alpha).unwrap();
                if output > T::zero() { T::one() } else { output + alpha_t }
            }
            ActivationFunction::Swish => {
                let sigmoid = T::one() / (T::one() + (-output).exp());
                sigmoid + output * sigmoid * (T::one() - sigmoid)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gradient_computation() {
        // Test simple quadratic function: f(x) = x^2
        let params = vec![2.0f32];
        let loss_fn = |p: &[f32]| p[0] * p[0];
        let epsilon = 1e-5;
        
        let gradients = GradientComputation::compute_gradients(&params, loss_fn, epsilon);
        
        // Expected gradient: 2 * x = 2 * 2 = 4
        assert!((gradients[0] - 4.0).abs() < 1e-3);
    }
    
    #[test]
    fn test_exponential_lr_scheduler() {
        let mut scheduler = ExponentialLRScheduler::new(0.1f32, 0.9);
        assert_eq!(scheduler.get_learning_rate(), 0.1);
        
        scheduler.step(0.0);
        assert!((scheduler.get_learning_rate() - 0.09).abs() < 1e-6);
        
        scheduler.step(0.0);
        assert!((scheduler.get_learning_rate() - 0.081).abs() < 1e-6);
    }
    
    #[test]
    fn test_step_lr_scheduler() {
        let mut scheduler = StepLRScheduler::new(0.1f32, 2, 0.5);
        assert_eq!(scheduler.get_learning_rate(), 0.1);
        
        scheduler.step(0.0);
        assert_eq!(scheduler.get_learning_rate(), 0.1);
        
        scheduler.step(0.0);
        assert_eq!(scheduler.get_learning_rate(), 0.05);
    }
    
    #[test]
    fn test_numerical_utils() {
        assert!(NumericalUtils::is_finite(1.0f32));
        assert!(!NumericalUtils::is_finite(f32::NAN));
        assert!(!NumericalUtils::is_finite(f32::INFINITY));
        
        assert_eq!(NumericalUtils::clamp(1.5f32, 0.0, 1.0), 1.0);
        assert_eq!(NumericalUtils::clamp(-0.5f32, 0.0, 1.0), 0.0);
        assert_eq!(NumericalUtils::clamp(0.5f32, 0.0, 1.0), 0.5);
        
        let vector = vec![3.0f32, 4.0f32];
        assert_eq!(NumericalUtils::l2_norm(&vector), 5.0);
        assert_eq!(NumericalUtils::l1_norm(&vector), 7.0);
    }
    
    #[test]
    fn test_activation_functions() {
        let sigmoid = ActivationFunction::Sigmoid;
        let relu = ActivationFunction::ReLU;
        let tanh = ActivationFunction::Tanh;
        
        // Test sigmoid
        let sig_output = sigmoid.apply(0.0f32);
        assert!((sig_output - 0.5).abs() < 1e-6);
        
        let sig_deriv = sigmoid.derivative(0.5f32);
        assert!((sig_deriv - 0.25).abs() < 1e-6);
        
        // Test ReLU
        assert_eq!(relu.apply(-1.0f32), 0.0);
        assert_eq!(relu.apply(1.0f32), 1.0);
        assert_eq!(relu.derivative(1.0f32), 1.0);
        assert_eq!(relu.derivative(-1.0f32), 0.0);
        
        // Test tanh
        let tanh_output = tanh.apply(0.0f32);
        assert!((tanh_output - 0.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_training_stats() {
        let mut stats = TrainingStats::<f32>::new();
        
        stats.record_gradient_norm(0.1);
        stats.record_gradient_norm(0.2);
        stats.record_gradient_norm(0.3);
        
        let avg_norm = stats.get_average_gradient_norm().unwrap();
        assert!((avg_norm - 0.2).abs() < 1e-6);
        
        stats.record_loss(1.0);
        stats.record_loss(0.8);
        
        let trend = stats.get_loss_trend().unwrap();
        assert!((trend - (-0.2)).abs() < 1e-6);
    }
}