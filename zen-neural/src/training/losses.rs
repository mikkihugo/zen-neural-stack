//! Comprehensive loss function library for zen-neural training
//!
//! This module provides high-performance implementations of common loss functions
//! used in neural network training, with support for various tasks including
//! regression, classification, and specialized applications.

use std::fmt::Debug;
use num_traits::Float;

use crate::TrainingError;
use super::{ZenLossFunction, LossType, NumericalUtils};

/// Mean Squared Error (MSE) loss for regression tasks
pub struct MSELoss<T: Float> {
    reduction: ReductionType,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Clone + Default> MSELoss<T> {
    pub fn new() -> Self {
        Self {
            reduction: ReductionType::Mean,
            _phantom: std::marker::PhantomData,
        }
    }
    
    pub fn with_reduction(mut self, reduction: ReductionType) -> Self {
        self.reduction = reduction;
        self
    }
}

impl<T: Float + Clone + Default> ZenLossFunction<T> for MSELoss<T> {
    fn compute_loss(&self, predictions: &[T], targets: &[T]) -> Result<T, TrainingError> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::InvalidData(
                "Predictions and targets must have the same length".to_string()
            ));
        }
        
        let squared_diffs: Vec<T> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let diff = pred - target;
                diff * diff
            })
            .collect();
        
        let loss = match self.reduction {
            ReductionType::Mean => {
                let sum: T = squared_diffs.iter().fold(T::zero(), |acc, &x| acc + x);
                sum / T::from(squared_diffs.len()).unwrap()
            }
            ReductionType::Sum => {
                squared_diffs.iter().fold(T::zero(), |acc, &x| acc + x)
            }
            ReductionType::None => {
                return Err(TrainingError::TrainingFailed(
                    "MSE loss with no reduction not supported for scalar output".to_string()
                ));
            }
        };
        
        Ok(loss)
    }
    
    fn compute_gradient(&self, predictions: &[T], targets: &[T]) -> Result<Vec<T>, TrainingError> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::InvalidData(
                "Predictions and targets must have the same length".to_string()
            ));
        }
        
        let n = T::from(predictions.len()).unwrap();
        let scale = match self.reduction {
            ReductionType::Mean => T::from(2.0).unwrap() / n,
            ReductionType::Sum => T::from(2.0).unwrap(),
            ReductionType::None => T::from(2.0).unwrap(),
        };
        
        let gradients: Vec<T> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| scale * (pred - target))
            .collect();
        
        Ok(gradients)
    }
}

/// Mean Absolute Error (MAE) loss for robust regression
pub struct MAELoss<T: Float> {
    reduction: ReductionType,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Clone + Default> MAELoss<T> {
    pub fn new() -> Self {
        Self {
            reduction: ReductionType::Mean,
            _phantom: std::marker::PhantomData,
        }
    }
    
    pub fn with_reduction(mut self, reduction: ReductionType) -> Self {
        self.reduction = reduction;
        self
    }
}

impl<T: Float + Clone + Default> ZenLossFunction<T> for MAELoss<T> {
    fn compute_loss(&self, predictions: &[T], targets: &[T]) -> Result<T, TrainingError> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::InvalidData(
                "Predictions and targets must have the same length".to_string()
            ));
        }
        
        let absolute_diffs: Vec<T> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - target).abs())
            .collect();
        
        let loss = match self.reduction {
            ReductionType::Mean => {
                let sum: T = absolute_diffs.iter().fold(T::zero(), |acc, &x| acc + x);
                sum / T::from(absolute_diffs.len()).unwrap()
            }
            ReductionType::Sum => {
                absolute_diffs.iter().fold(T::zero(), |acc, &x| acc + x)
            }
            ReductionType::None => {
                return Err(TrainingError::TrainingFailed(
                    "MAE loss with no reduction not supported for scalar output".to_string()
                ));
            }
        };
        
        Ok(loss)
    }
    
    fn compute_gradient(&self, predictions: &[T], targets: &[T]) -> Result<Vec<T>, TrainingError> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::InvalidData(
                "Predictions and targets must have the same length".to_string()
            ));
        }
        
        let n = T::from(predictions.len()).unwrap();
        let scale = match self.reduction {
            ReductionType::Mean => T::one() / n,
            ReductionType::Sum => T::one(),
            ReductionType::None => T::one(),
        };
        
        let gradients: Vec<T> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let diff = pred - target;
                if diff > T::zero() {
                    scale
                } else if diff < T::zero() {
                    -scale
                } else {
                    T::zero() // Subgradient at zero
                }
            })
            .collect();
        
        Ok(gradients)
    }
}

/// Cross Entropy loss for multi-class classification
pub struct CrossEntropyLoss<T: Float> {
    reduction: ReductionType,
    class_weights: Option<Vec<T>>,
    label_smoothing: T,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Clone + Default> CrossEntropyLoss<T> {
    pub fn new() -> Self {
        Self {
            reduction: ReductionType::Mean,
            class_weights: None,
            label_smoothing: T::zero(),
            _phantom: std::marker::PhantomData,
        }
    }
    
    pub fn with_reduction(mut self, reduction: ReductionType) -> Self {
        self.reduction = reduction;
        self
    }
    
    pub fn with_class_weights(mut self, weights: Vec<T>) -> Self {
        self.class_weights = Some(weights);
        self
    }
    
    pub fn with_label_smoothing(mut self, smoothing: T) -> Self {
        self.label_smoothing = smoothing;
        self
    }
    
    fn softmax(&self, logits: &[T]) -> Vec<T> {
        // Numerically stable softmax
        let max_logit = logits.iter().fold(T::neg_infinity(), |max, &x| {
            if x > max { x } else { max }
        });
        
        let exp_logits: Vec<T> = logits.iter()
            .map(|&x| (x - max_logit).exp())
            .collect();
        
        let sum_exp: T = exp_logits.iter().fold(T::zero(), |acc, &x| acc + x);
        
        exp_logits.into_iter().map(|x| x / sum_exp).collect()
    }
    
    fn apply_label_smoothing(&self, targets: &[T], num_classes: usize) -> Vec<T> {
        if self.label_smoothing == T::zero() {
            return targets.to_vec();
        }
        
        let smoothing = self.label_smoothing;
        let uniform_prob = smoothing / T::from(num_classes).unwrap();
        
        targets.iter().map(|&target| {
            (T::one() - smoothing) * target + uniform_prob
        }).collect()
    }
}

impl<T: Float + Clone + Default> ZenLossFunction<T> for CrossEntropyLoss<T> {
    fn compute_loss(&self, predictions: &[T], targets: &[T]) -> Result<T, TrainingError> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::InvalidData(
                "Predictions and targets must have the same length".to_string()
            ));
        }
        
        if predictions.is_empty() {
            return Err(TrainingError::InvalidData("Empty predictions".to_string()));
        }
        
        // Apply softmax to get probabilities
        let probabilities = self.softmax(predictions);
        
        // Apply label smoothing if specified
        let smooth_targets = self.apply_label_smoothing(targets, targets.len());
        
        // Compute cross-entropy loss
        let epsilon = T::from(1e-15).unwrap(); // For numerical stability
        let losses: Vec<T> = probabilities.iter()
            .zip(smooth_targets.iter())
            .enumerate()
            .map(|(i, (&prob, &target))| {
                let clipped_prob = NumericalUtils::clamp(prob, epsilon, T::one() - epsilon);
                let loss = -target * clipped_prob.ln();
                
                // Apply class weight if specified
                if let Some(ref weights) = self.class_weights {
                    if i < weights.len() {
                        loss * weights[i]
                    } else {
                        loss
                    }
                } else {
                    loss
                }
            })
            .collect();
        
        let total_loss = match self.reduction {
            ReductionType::Mean => {
                let sum: T = losses.iter().fold(T::zero(), |acc, &x| acc + x);
                sum / T::from(losses.len()).unwrap()
            }
            ReductionType::Sum => {
                losses.iter().fold(T::zero(), |acc, &x| acc + x)
            }
            ReductionType::None => {
                return Err(TrainingError::TrainingFailed(
                    "Cross-entropy loss with no reduction not supported for scalar output".to_string()
                ));
            }
        };
        
        Ok(total_loss)
    }
    
    fn compute_gradient(&self, predictions: &[T], targets: &[T]) -> Result<Vec<T>, TrainingError> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::InvalidData(
                "Predictions and targets must have the same length".to_string()
            ));
        }
        
        let probabilities = self.softmax(predictions);
        let smooth_targets = self.apply_label_smoothing(targets, targets.len());
        
        let n = T::from(predictions.len()).unwrap();
        let scale = match self.reduction {
            ReductionType::Mean => T::one() / n,
            ReductionType::Sum => T::one(),
            ReductionType::None => T::one(),
        };
        
        let gradients: Vec<T> = probabilities.iter()
            .zip(smooth_targets.iter())
            .enumerate()
            .map(|(i, (&prob, &target))| {
                let grad = scale * (prob - target);
                
                // Apply class weight if specified
                if let Some(ref weights) = self.class_weights {
                    if i < weights.len() {
                        grad * weights[i]
                    } else {
                        grad
                    }
                } else {
                    grad
                }
            })
            .collect();
        
        Ok(gradients)
    }
}

/// Binary Cross Entropy loss for binary classification
pub struct BinaryCrossEntropyLoss<T: Float> {
    reduction: ReductionType,
    pos_weight: Option<T>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Clone + Default> BinaryCrossEntropyLoss<T> {
    pub fn new() -> Self {
        Self {
            reduction: ReductionType::Mean,
            pos_weight: None,
            _phantom: std::marker::PhantomData,
        }
    }
    
    pub fn with_reduction(mut self, reduction: ReductionType) -> Self {
        self.reduction = reduction;
        self
    }
    
    pub fn with_pos_weight(mut self, weight: T) -> Self {
        self.pos_weight = Some(weight);
        self
    }
    
    fn sigmoid(&self, x: T) -> T {
        T::one() / (T::one() + (-x).exp())
    }
}

impl<T: Float + Clone + Default> ZenLossFunction<T> for BinaryCrossEntropyLoss<T> {
    fn compute_loss(&self, predictions: &[T], targets: &[T]) -> Result<T, TrainingError> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::InvalidData(
                "Predictions and targets must have the same length".to_string()
            ));
        }
        
        let epsilon = T::from(1e-15).unwrap();
        
        let losses: Vec<T> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let prob = self.sigmoid(pred);
                let clipped_prob = NumericalUtils::clamp(prob, epsilon, T::one() - epsilon);
                
                let loss = -target * clipped_prob.ln() - (T::one() - target) * (T::one() - clipped_prob).ln();
                
                // Apply positive class weight if specified
                if let Some(pos_weight) = self.pos_weight {
                    target * pos_weight * loss + (T::one() - target) * loss
                } else {
                    loss
                }
            })
            .collect();
        
        let total_loss = match self.reduction {
            ReductionType::Mean => {
                let sum: T = losses.iter().fold(T::zero(), |acc, &x| acc + x);
                sum / T::from(losses.len()).unwrap()
            }
            ReductionType::Sum => {
                losses.iter().fold(T::zero(), |acc, &x| acc + x)
            }
            ReductionType::None => {
                return Err(TrainingError::TrainingFailed(
                    "BCE loss with no reduction not supported for scalar output".to_string()
                ));
            }
        };
        
        Ok(total_loss)
    }
    
    fn compute_gradient(&self, predictions: &[T], targets: &[T]) -> Result<Vec<T>, TrainingError> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::InvalidData(
                "Predictions and targets must have the same length".to_string()
            ));
        }
        
        let n = T::from(predictions.len()).unwrap();
        let scale = match self.reduction {
            ReductionType::Mean => T::one() / n,
            ReductionType::Sum => T::one(),
            ReductionType::None => T::one(),
        };
        
        let gradients: Vec<T> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let prob = self.sigmoid(pred);
                let grad = scale * (prob - target);
                
                if let Some(pos_weight) = self.pos_weight {
                    target * pos_weight * grad + (T::one() - target) * grad
                } else {
                    grad
                }
            })
            .collect();
        
        Ok(gradients)
    }
}

/// Huber loss for robust regression (less sensitive to outliers than MSE)
pub struct HuberLoss<T: Float> {
    delta: T,
    reduction: ReductionType,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Clone + Default> HuberLoss<T> {
    pub fn new(delta: T) -> Self {
        Self {
            delta,
            reduction: ReductionType::Mean,
            _phantom: std::marker::PhantomData,
        }
    }
    
    pub fn with_reduction(mut self, reduction: ReductionType) -> Self {
        self.reduction = reduction;
        self
    }
}

impl<T: Float + Clone + Default> ZenLossFunction<T> for HuberLoss<T> {
    fn compute_loss(&self, predictions: &[T], targets: &[T]) -> Result<T, TrainingError> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::InvalidData(
                "Predictions and targets must have the same length".to_string()
            ));
        }
        
        let losses: Vec<T> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let diff = (pred - target).abs();
                if diff <= self.delta {
                    T::from(0.5).unwrap() * diff * diff
                } else {
                    self.delta * (diff - T::from(0.5).unwrap() * self.delta)
                }
            })
            .collect();
        
        let total_loss = match self.reduction {
            ReductionType::Mean => {
                let sum: T = losses.iter().fold(T::zero(), |acc, &x| acc + x);
                sum / T::from(losses.len()).unwrap()
            }
            ReductionType::Sum => {
                losses.iter().fold(T::zero(), |acc, &x| acc + x)
            }
            ReductionType::None => {
                return Err(TrainingError::TrainingFailed(
                    "Huber loss with no reduction not supported for scalar output".to_string()
                ));
            }
        };
        
        Ok(total_loss)
    }
    
    fn compute_gradient(&self, predictions: &[T], targets: &[T]) -> Result<Vec<T>, TrainingError> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::InvalidData(
                "Predictions and targets must have the same length".to_string()
            ));
        }
        
        let n = T::from(predictions.len()).unwrap();
        let scale = match self.reduction {
            ReductionType::Mean => T::one() / n,
            ReductionType::Sum => T::one(),
            ReductionType::None => T::one(),
        };
        
        let gradients: Vec<T> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let diff = pred - target;
                let abs_diff = diff.abs();
                
                if abs_diff <= self.delta {
                    scale * diff
                } else if diff > T::zero() {
                    scale * self.delta
                } else {
                    -scale * self.delta
                }
            })
            .collect();
        
        Ok(gradients)
    }
}

/// Focal loss for addressing class imbalance in classification
pub struct FocalLoss<T: Float> {
    alpha: T,
    gamma: T,
    reduction: ReductionType,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Clone + Default> FocalLoss<T> {
    pub fn new(alpha: T, gamma: T) -> Self {
        Self {
            alpha,
            gamma,
            reduction: ReductionType::Mean,
            _phantom: std::marker::PhantomData,
        }
    }
    
    pub fn with_reduction(mut self, reduction: ReductionType) -> Self {
        self.reduction = reduction;
        self
    }
    
    fn sigmoid(&self, x: T) -> T {
        T::one() / (T::one() + (-x).exp())
    }
}

impl<T: Float + Clone + Default> ZenLossFunction<T> for FocalLoss<T> {
    fn compute_loss(&self, predictions: &[T], targets: &[T]) -> Result<T, TrainingError> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::InvalidData(
                "Predictions and targets must have the same length".to_string()
            ));
        }
        
        let epsilon = T::from(1e-15).unwrap();
        
        let losses: Vec<T> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let prob = self.sigmoid(pred);
                let clipped_prob = NumericalUtils::clamp(prob, epsilon, T::one() - epsilon);
                
                // Focal loss computation
                let alpha_t = if target > T::from(0.5).unwrap() { self.alpha } else { T::one() - self.alpha };
                let p_t = if target > T::from(0.5).unwrap() { clipped_prob } else { T::one() - clipped_prob };
                
                let focal_weight = alpha_t * (T::one() - p_t).powf(self.gamma);
                let ce_loss = -p_t.ln();
                
                focal_weight * ce_loss
            })
            .collect();
        
        let total_loss = match self.reduction {
            ReductionType::Mean => {
                let sum: T = losses.iter().fold(T::zero(), |acc, &x| acc + x);
                sum / T::from(losses.len()).unwrap()
            }
            ReductionType::Sum => {
                losses.iter().fold(T::zero(), |acc, &x| acc + x)
            }
            ReductionType::None => {
                return Err(TrainingError::TrainingFailed(
                    "Focal loss with no reduction not supported for scalar output".to_string()
                ));
            }
        };
        
        Ok(total_loss)
    }
    
    fn compute_gradient(&self, predictions: &[T], targets: &[T]) -> Result<Vec<T>, TrainingError> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::InvalidData(
                "Predictions and targets must have the same length".to_string()
            ));
        }
        
        let n = T::from(predictions.len()).unwrap();
        let scale = match self.reduction {
            ReductionType::Mean => T::one() / n,
            ReductionType::Sum => T::one(),
            ReductionType::None => T::one(),
        };
        
        let gradients: Vec<T> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let prob = self.sigmoid(pred);
                let epsilon = T::from(1e-15).unwrap();
                let clipped_prob = NumericalUtils::clamp(prob, epsilon, T::one() - epsilon);
                
                // Focal loss gradient (simplified approximation)
                let alpha_t = if target > T::from(0.5).unwrap() { self.alpha } else { T::one() - self.alpha };
                let p_t = if target > T::from(0.5).unwrap() { clipped_prob } else { T::one() - clipped_prob };
                
                let focal_weight = alpha_t * (T::one() - p_t).powf(self.gamma);
                let grad = scale * focal_weight * (prob - target);
                
                grad
            })
            .collect();
        
        Ok(gradients)
    }
}

/// Reduction types for loss functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReductionType {
    Mean,  // Average loss over batch
    Sum,   // Sum loss over batch
    None,  // No reduction (return per-sample losses)
}

/// Factory function to create loss functions
pub fn create_loss_function<T: Float + Clone + Default + Send + Sync>(
    loss_type: LossType
) -> Result<Box<dyn ZenLossFunction<T>>, TrainingError> {
    match loss_type {
        LossType::MSE => Ok(Box::new(MSELoss::new())),
        LossType::MAE => Ok(Box::new(MAELoss::new())),
        LossType::CrossEntropy => Ok(Box::new(CrossEntropyLoss::new())),
        LossType::BinaryCrossEntropy => Ok(Box::new(BinaryCrossEntropyLoss::new())),
        LossType::Huber { delta } => Ok(Box::new(HuberLoss::new(T::from(delta).unwrap()))),
        LossType::Hinge => {
            // Hinge loss would be implemented similarly
            Err(TrainingError::TrainingFailed("Hinge loss not yet implemented".to_string()))
        }
        LossType::Custom => {
            Err(TrainingError::TrainingFailed("Custom loss functions must be provided directly".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mse_loss() {
        let mse = MSELoss::<f32>::new();
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.1, 1.9, 3.2];
        
        let loss = mse.compute_loss(&predictions, &targets).unwrap();
        let expected_loss = ((0.1 * 0.1) + (0.1 * 0.1) + (0.2 * 0.2)) / 3.0;
        
        assert!((loss - expected_loss).abs() < 1e-6);
        
        let gradients = mse.compute_gradient(&predictions, &targets).unwrap();
        assert_eq!(gradients.len(), 3);
        assert!((gradients[0] - (-2.0 * 0.1 / 3.0)).abs() < 1e-6);
    }
    
    #[test]
    fn test_mae_loss() {
        let mae = MAELoss::<f32>::new();
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.1, 1.8, 3.3];
        
        let loss = mae.compute_loss(&predictions, &targets).unwrap();
        let expected_loss = (0.1 + 0.2 + 0.3) / 3.0;
        
        assert!((loss - expected_loss).abs() < 1e-6);
        
        let gradients = mae.compute_gradient(&predictions, &targets).unwrap();
        assert_eq!(gradients.len(), 3);
        assert!((gradients[0] - (-1.0 / 3.0)).abs() < 1e-6);
        assert!((gradients[1] - (1.0 / 3.0)).abs() < 1e-6);
        assert!((gradients[2] - (-1.0 / 3.0)).abs() < 1e-6);
    }
    
    #[test]
    fn test_cross_entropy_loss() {
        let ce = CrossEntropyLoss::<f32>::new();
        let predictions = vec![2.0, 1.0, 0.1]; // logits
        let targets = vec![1.0, 0.0, 0.0]; // one-hot encoding
        
        let loss = ce.compute_loss(&predictions, &targets).unwrap();
        assert!(loss > 0.0); // Cross-entropy should be positive
        
        let gradients = ce.compute_gradient(&predictions, &targets).unwrap();
        assert_eq!(gradients.len(), 3);
    }
    
    #[test]
    fn test_binary_cross_entropy_loss() {
        let bce = BinaryCrossEntropyLoss::<f32>::new();
        let predictions = vec![0.5, -0.5]; // logits
        let targets = vec![1.0, 0.0]; // binary targets
        
        let loss = bce.compute_loss(&predictions, &targets).unwrap();
        assert!(loss > 0.0);
        
        let gradients = bce.compute_gradient(&predictions, &targets).unwrap();
        assert_eq!(gradients.len(), 2);
    }
    
    #[test]
    fn test_huber_loss() {
        let huber = HuberLoss::<f32>::new(1.0);
        let predictions = vec![0.0, 2.0]; // One within delta, one beyond
        let targets = vec![0.5, 0.0];
        
        let loss = huber.compute_loss(&predictions, &targets).unwrap();
        assert!(loss > 0.0);
        
        let gradients = huber.compute_gradient(&predictions, &targets).unwrap();
        assert_eq!(gradients.len(), 2);
    }
    
    #[test]
    fn test_focal_loss() {
        let focal = FocalLoss::<f32>::new(0.25, 2.0);
        let predictions = vec![0.1, 0.9]; // logits
        let targets = vec![1.0, 1.0]; // binary targets
        
        let loss = focal.compute_loss(&predictions, &targets).unwrap();
        assert!(loss > 0.0);
        
        let gradients = focal.compute_gradient(&predictions, &targets).unwrap();
        assert_eq!(gradients.len(), 2);
    }
    
    #[test]
    fn test_loss_with_different_reductions() {
        let mse_mean = MSELoss::<f32>::new().with_reduction(ReductionType::Mean);
        let mse_sum = MSELoss::<f32>::new().with_reduction(ReductionType::Sum);
        
        let predictions = vec![1.0, 2.0];
        let targets = vec![1.1, 1.9];
        
        let loss_mean = mse_mean.compute_loss(&predictions, &targets).unwrap();
        let loss_sum = mse_sum.compute_loss(&predictions, &targets).unwrap();
        
        assert!((loss_mean * 2.0 - loss_sum).abs() < 1e-6);
    }
    
    #[test]
    fn test_loss_creation_factory() {
        let mse_loss = create_loss_function::<f32>(LossType::MSE);
        assert!(mse_loss.is_ok());
        
        let mae_loss = create_loss_function::<f32>(LossType::MAE);
        assert!(mae_loss.is_ok());
        
        let ce_loss = create_loss_function::<f32>(LossType::CrossEntropy);
        assert!(ce_loss.is_ok());
        
        let bce_loss = create_loss_function::<f32>(LossType::BinaryCrossEntropy);
        assert!(bce_loss.is_ok());
        
        let huber_loss = create_loss_function::<f32>(LossType::Huber { delta: 1.0 });
        assert!(huber_loss.is_ok());
    }
}