//! Training configuration utilities and presets
//!
//! This module provides configuration builders, validation, and preset configurations
//! for common training scenarios. Includes templates for different types of neural
//! network training tasks and hyperparameter optimization utilities.

use std::collections::HashMap;
use num_traits::Float;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::TrainingError;
use super::{
    TrainingConfig, OptimizerType, LossType, LRScheduleConfig, LRScheduleType,
    EarlyStoppingConfig, EarlyStoppingMode, MetricType, CheckpointConfig,
    MemoryOptimizationConfig,
};

/// Configuration builder for training setup
pub struct TrainingConfigBuilder<T: Float> {
    config: TrainingConfig<T>,
}

impl<T: Float + Default + Clone> TrainingConfigBuilder<T> {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: TrainingConfig::default(),
        }
    }
    
    /// Set number of training epochs
    pub fn epochs(mut self, epochs: u32) -> Self {
        self.config.epochs = epochs;
        self
    }
    
    /// Set learning rate
    pub fn learning_rate(mut self, lr: T) -> Self {
        self.config.learning_rate = lr;
        self
    }
    
    /// Set batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }
    
    /// Configure Adam optimizer
    pub fn adam_optimizer(mut self, beta1: T, beta2: T, epsilon: T) -> Self {
        self.config.optimizer_type = OptimizerType::Adam { beta1, beta2, epsilon };
        self
    }
    
    /// Configure SGD optimizer
    pub fn sgd_optimizer(mut self, momentum: T) -> Self {
        self.config.optimizer_type = OptimizerType::SGD { momentum };
        self
    }
    
    /// Configure AdamW optimizer
    pub fn adamw_optimizer(mut self, beta1: T, beta2: T, epsilon: T) -> Self {
        self.config.optimizer_type = OptimizerType::AdamW { beta1, beta2, epsilon };
        self
    }
    
    /// Set loss function type
    pub fn loss_function(mut self, loss_type: LossType) -> Self {
        self.config.loss_type = loss_type;
        self
    }
    
    /// Configure exponential learning rate decay
    pub fn exponential_lr_decay(mut self, decay_rate: T, step_interval: u32) -> Self {
        self.config.lr_schedule = Some(LRScheduleConfig {
            schedule_type: LRScheduleType::Exponential { decay_rate },
            step_interval,
        });
        self
    }
    
    /// Configure step learning rate decay
    pub fn step_lr_decay(mut self, step_size: u32, gamma: T, step_interval: u32) -> Self {
        self.config.lr_schedule = Some(LRScheduleConfig {
            schedule_type: LRScheduleType::StepLR { step_size, gamma },
            step_interval,
        });
        self
    }
    
    /// Configure cosine annealing learning rate schedule
    pub fn cosine_annealing_lr(mut self, t_max: u32, eta_min: T, step_interval: u32) -> Self {
        self.config.lr_schedule = Some(LRScheduleConfig {
            schedule_type: LRScheduleType::CosineAnnealing { t_max, eta_min },
            step_interval,
        });
        self
    }
    
    /// Configure early stopping
    pub fn early_stopping(
        mut self,
        patience: u32,
        min_delta: T,
        monitor: MetricType,
        mode: EarlyStoppingMode,
    ) -> Self {
        self.config.early_stopping = Some(EarlyStoppingConfig {
            patience,
            min_delta,
            monitor,
            mode,
        });
        self
    }
    
    /// Set gradient clipping threshold
    pub fn gradient_clipping(mut self, clip_norm: T) -> Self {
        self.config.gradient_clip = Some(clip_norm);
        self
    }
    
    /// Set weight decay (L2 regularization)
    pub fn weight_decay(mut self, decay: T) -> Self {
        self.config.weight_decay = decay;
        self
    }
    
    /// Set dropout rate
    pub fn dropout(mut self, rate: T) -> Self {
        self.config.dropout = Some(rate);
        self
    }
    
    /// Set validation split ratio
    pub fn validation_split(mut self, ratio: T) -> Self {
        self.config.validation_split = ratio;
        self
    }
    
    /// Set random seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }
    
    /// Enable/disable parallel processing
    pub fn parallel(mut self, enable: bool) -> Self {
        self.config.parallel = enable;
        self
    }
    
    /// Configure memory optimization
    pub fn memory_optimization(mut self, config: MemoryOptimizationConfig) -> Self {
        self.config.memory_opts = config;
        self
    }
    
    /// Set metrics to collect
    pub fn metrics(mut self, metrics: Vec<MetricType>) -> Self {
        self.config.metrics = metrics;
        self
    }
    
    /// Configure checkpointing
    pub fn checkpointing(mut self, config: CheckpointConfig<T>) -> Self {
        self.config.checkpointing = config;
        self
    }
    
    /// Enable/disable verbose logging
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }
    
    /// Build the final configuration
    pub fn build(self) -> Result<TrainingConfig<T>, TrainingError> {
        self.validate_config(&self.config)?;
        Ok(self.config)
    }
    
    /// Validate the training configuration
    fn validate_config(&self, config: &TrainingConfig<T>) -> Result<(), TrainingError> {
        // Validate epochs
        if config.epochs == 0 {
            return Err(TrainingError::InvalidData("Epochs must be greater than 0".to_string()));
        }
        
        // Validate learning rate
        if config.learning_rate <= T::zero() {
            return Err(TrainingError::InvalidData("Learning rate must be positive".to_string()));
        }
        
        // Validate batch size
        if config.batch_size == 0 {
            return Err(TrainingError::InvalidData("Batch size must be greater than 0".to_string()));
        }
        
        // Validate validation split
        if config.validation_split < T::zero() || config.validation_split >= T::one() {
            return Err(TrainingError::InvalidData(
                "Validation split must be between 0 and 1".to_string()
            ));
        }
        
        // Validate weight decay
        if config.weight_decay < T::zero() {
            return Err(TrainingError::InvalidData("Weight decay must be non-negative".to_string()));
        }
        
        // Validate dropout
        if let Some(dropout) = config.dropout {
            if dropout < T::zero() || dropout >= T::one() {
                return Err(TrainingError::InvalidData(
                    "Dropout rate must be between 0 and 1".to_string()
                ));
            }
        }
        
        // Validate gradient clipping
        if let Some(clip_norm) = config.gradient_clip {
            if clip_norm <= T::zero() {
                return Err(TrainingError::InvalidData(
                    "Gradient clipping threshold must be positive".to_string()
                ));
            }
        }
        
        // Validate early stopping
        if let Some(ref early_stopping) = config.early_stopping {
            if early_stopping.patience == 0 {
                return Err(TrainingError::InvalidData(
                    "Early stopping patience must be greater than 0".to_string()
                ));
            }
            
            if early_stopping.min_delta < T::zero() {
                return Err(TrainingError::InvalidData(
                    "Early stopping min_delta must be non-negative".to_string()
                ));
            }
        }
        
        Ok(())
    }
}

/// Preset training configurations for common scenarios
pub struct TrainingPresets;

impl TrainingPresets {
    /// Fast training preset for quick experimentation
    pub fn fast_training<T: Float + Default>() -> TrainingConfigBuilder<T> {
        TrainingConfigBuilder::new()
            .epochs(50)
            .learning_rate(T::from(0.01).unwrap())
            .batch_size(64)
            .adam_optimizer(
                T::from(0.9).unwrap(),
                T::from(0.999).unwrap(),
                T::from(1e-8).unwrap(),
            )
            .gradient_clipping(T::from(1.0).unwrap())
            .verbose(true)
    }
    
    /// Production training preset for high-quality models
    pub fn production_training<T: Float + Default>() -> TrainingConfigBuilder<T> {
        TrainingConfigBuilder::new()
            .epochs(200)
            .learning_rate(T::from(0.001).unwrap())
            .batch_size(32)
            .adamw_optimizer(
                T::from(0.9).unwrap(),
                T::from(0.999).unwrap(),
                T::from(1e-8).unwrap(),
            )
            .weight_decay(T::from(0.01).unwrap())
            .gradient_clipping(T::from(0.5).unwrap())
            .exponential_lr_decay(T::from(0.95).unwrap(), 10)
            .early_stopping(15, T::from(1e-4).unwrap(), MetricType::Loss, EarlyStoppingMode::Min)
            .validation_split(T::from(0.15).unwrap())
            .memory_optimization(MemoryOptimizationConfig {
                use_memory_pools: true,
                gradient_accumulation: true,
                zero_allocation_loops: true,
                preallocate_buffers: true,
            })
            .metrics(vec![
                MetricType::Loss,
                MetricType::Accuracy,
                MetricType::F1Score,
                MetricType::MeanAbsoluteError,
            ])
            .verbose(true)
    }
    
    /// Research training preset for academic/research use
    pub fn research_training<T: Float + Default>() -> TrainingConfigBuilder<T> {
        TrainingConfigBuilder::new()
            .epochs(500)
            .learning_rate(T::from(0.0005).unwrap())
            .batch_size(16)
            .adam_optimizer(
                T::from(0.95).unwrap(),
                T::from(0.9999).unwrap(),
                T::from(1e-9).unwrap(),
            )
            .weight_decay(T::from(0.005).unwrap())
            .gradient_clipping(T::from(0.25).unwrap())
            .cosine_annealing_lr(500, T::from(1e-6).unwrap(), 1)
            .early_stopping(30, T::from(1e-5).unwrap(), MetricType::Loss, EarlyStoppingMode::Min)
            .validation_split(T::from(0.2).unwrap())
            .checkpointing(CheckpointConfig {
                enabled: true,
                save_interval: 10,
                save_best_only: true,
                monitor_metric: MetricType::Loss,
                min_improvement: T::from(1e-5).unwrap(),
            })
            .memory_optimization(MemoryOptimizationConfig::default())
            .verbose(true)
    }
    
    /// Binary classification preset
    pub fn binary_classification<T: Float + Default>() -> TrainingConfigBuilder<T> {
        TrainingConfigBuilder::new()
            .epochs(100)
            .learning_rate(T::from(0.01).unwrap())
            .batch_size(32)
            .adam_optimizer(
                T::from(0.9).unwrap(),
                T::from(0.999).unwrap(),
                T::from(1e-8).unwrap(),
            )
            .loss_function(LossType::BinaryCrossEntropy)
            .metrics(vec![
                MetricType::Loss,
                MetricType::Accuracy,
                MetricType::Precision,
                MetricType::Recall,
                MetricType::F1Score,
            ])
            .early_stopping(10, T::from(1e-4).unwrap(), MetricType::F1Score, EarlyStoppingMode::Max)
            .gradient_clipping(T::from(1.0).unwrap())
    }
    
    /// Multi-class classification preset
    pub fn multiclass_classification<T: Float + Default>() -> TrainingConfigBuilder<T> {
        TrainingConfigBuilder::new()
            .epochs(150)
            .learning_rate(T::from(0.005).unwrap())
            .batch_size(64)
            .adam_optimizer(
                T::from(0.9).unwrap(),
                T::from(0.999).unwrap(),
                T::from(1e-8).unwrap(),
            )
            .loss_function(LossType::CrossEntropy)
            .metrics(vec![
                MetricType::Loss,
                MetricType::Accuracy,
                MetricType::Precision,
                MetricType::Recall,
                MetricType::F1Score,
            ])
            .early_stopping(15, T::from(1e-4).unwrap(), MetricType::Accuracy, EarlyStoppingMode::Max)
            .step_lr_decay(50, T::from(0.5).unwrap(), 1)
            .gradient_clipping(T::from(2.0).unwrap())
    }
    
    /// Regression preset
    pub fn regression<T: Float + Default>() -> TrainingConfigBuilder<T> {
        TrainingConfigBuilder::new()
            .epochs(200)
            .learning_rate(T::from(0.001).unwrap())
            .batch_size(64)
            .adamw_optimizer(
                T::from(0.9).unwrap(),
                T::from(0.999).unwrap(),
                T::from(1e-8).unwrap(),
            )
            .loss_function(LossType::MSE)
            .weight_decay(T::from(0.01).unwrap())
            .metrics(vec![
                MetricType::Loss,
                MetricType::MeanAbsoluteError,
                MetricType::RootMeanSquaredError,
            ])
            .early_stopping(20, T::from(1e-5).unwrap(), MetricType::Loss, EarlyStoppingMode::Min)
            .gradient_clipping(T::from(1.0).unwrap())
    }
    
    /// Fine-tuning preset for transfer learning
    pub fn fine_tuning<T: Float + Default>() -> TrainingConfigBuilder<T> {
        TrainingConfigBuilder::new()
            .epochs(50)
            .learning_rate(T::from(0.0001).unwrap()) // Lower learning rate for fine-tuning
            .batch_size(16) // Smaller batch size
            .adam_optimizer(
                T::from(0.9).unwrap(),
                T::from(0.999).unwrap(),
                T::from(1e-8).unwrap(),
            )
            .weight_decay(T::from(0.001).unwrap()) // Light regularization
            .gradient_clipping(T::from(0.1).unwrap()) // Conservative clipping
            .exponential_lr_decay(T::from(0.9).unwrap(), 10)
            .early_stopping(10, T::from(1e-5).unwrap(), MetricType::Loss, EarlyStoppingMode::Min)
            .validation_split(T::from(0.1).unwrap())
    }
}

/// Hyperparameter search utilities
pub struct HyperparameterSearch<T: Float> {
    search_space: HashMap<String, ParameterRange<T>>,
}

impl<T: Float + Clone + Default> HyperparameterSearch<T> {
    pub fn new() -> Self {
        Self {
            search_space: HashMap::new(),
        }
    }
    
    /// Add learning rate search range
    pub fn add_learning_rate_range(mut self, min: T, max: T) -> Self {
        self.search_space.insert(
            "learning_rate".to_string(),
            ParameterRange::Continuous { min, max },
        );
        self
    }
    
    /// Add batch size search range
    pub fn add_batch_size_range(mut self, min: usize, max: usize) -> Self {
        self.search_space.insert(
            "batch_size".to_string(),
            ParameterRange::Discrete { 
                values: (min..=max).map(|x| T::from(x).unwrap()).collect()
            },
        );
        self
    }
    
    /// Add weight decay search range
    pub fn add_weight_decay_range(mut self, min: T, max: T) -> Self {
        self.search_space.insert(
            "weight_decay".to_string(),
            ParameterRange::Continuous { min, max },
        );
        self
    }
    
    /// Generate random configuration from search space
    pub fn sample_config(&self) -> Result<TrainingConfig<T>, TrainingError> {
        let mut config = TrainingConfig::default();
        
        for (param_name, range) in &self.search_space {
            let value = match range {
                ParameterRange::Continuous { min, max } => {
                    // Simple uniform sampling (would use proper random in real implementation)
                    *min + (*max - *min) * T::from(0.5).unwrap()
                }
                ParameterRange::Discrete { values } => {
                    if values.is_empty() {
                        return Err(TrainingError::InvalidData("Empty discrete range".to_string()));
                    }
                    values[0] // Would use random selection in real implementation
                }
            };
            
            // Apply sampled value to config
            match param_name.as_str() {
                "learning_rate" => config.learning_rate = value,
                "batch_size" => config.batch_size = value.to_usize().unwrap_or(32),
                "weight_decay" => config.weight_decay = value,
                _ => {}
            }
        }
        
        Ok(config)
    }
}

/// Parameter range for hyperparameter search
#[derive(Debug, Clone)]
pub enum ParameterRange<T: Float> {
    Continuous { min: T, max: T },
    Discrete { values: Vec<T> },
}

/// Configuration validation utilities
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate configuration for specific task type
    pub fn validate_for_task<T: Float + Default>(
        config: &TrainingConfig<T>,
        task_type: TaskType,
    ) -> Result<(), TrainingError> {
        match task_type {
            TaskType::BinaryClassification => {
                if config.loss_type != LossType::BinaryCrossEntropy 
                    && config.loss_type != LossType::Hinge {
                    return Err(TrainingError::InvalidData(
                        "Binary classification should use BCE or Hinge loss".to_string()
                    ));
                }
            }
            TaskType::MultiClassification => {
                if config.loss_type != LossType::CrossEntropy {
                    return Err(TrainingError::InvalidData(
                        "Multi-class classification should use CrossEntropy loss".to_string()
                    ));
                }
            }
            TaskType::Regression => {
                if config.loss_type != LossType::MSE 
                    && config.loss_type != LossType::MAE 
                    && !matches!(config.loss_type, LossType::Huber { .. }) {
                    return Err(TrainingError::InvalidData(
                        "Regression should use MSE, MAE, or Huber loss".to_string()
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    /// Suggest configuration improvements
    pub fn suggest_improvements<T: Float + Default>(
        config: &TrainingConfig<T>,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        // Check learning rate
        let lr_val = config.learning_rate.to_f64().unwrap_or(0.0);
        if lr_val > 0.1 {
            suggestions.push("Consider lowering learning rate (< 0.1) for more stable training".to_string());
        } else if lr_val < 1e-5 {
            suggestions.push("Consider increasing learning rate (> 1e-5) for faster convergence".to_string());
        }
        
        // Check batch size
        if config.batch_size > 512 {
            suggestions.push("Large batch sizes may hurt generalization, consider < 512".to_string());
        } else if config.batch_size < 8 {
            suggestions.push("Very small batch sizes may be unstable, consider >= 8".to_string());
        }
        
        // Check early stopping
        if config.early_stopping.is_none() {
            suggestions.push("Consider adding early stopping to prevent overfitting".to_string());
        }
        
        // Check gradient clipping
        if config.gradient_clip.is_none() {
            suggestions.push("Consider adding gradient clipping for training stability".to_string());
        }
        
        // Check regularization
        if config.weight_decay == T::zero() && config.dropout.is_none() {
            suggestions.push("Consider adding regularization (weight decay or dropout)".to_string());
        }
        
        suggestions
    }
}

/// Task types for configuration validation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskType {
    BinaryClassification,
    MultiClassification,
    Regression,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_builder() {
        let config = TrainingConfigBuilder::<f32>::new()
            .epochs(100)
            .learning_rate(0.01)
            .batch_size(32)
            .adam_optimizer(0.9, 0.999, 1e-8)
            .build()
            .unwrap();
        
        assert_eq!(config.epochs, 100);
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.batch_size, 32);
        
        match config.optimizer_type {
            OptimizerType::Adam { beta1, beta2, epsilon } => {
                assert_eq!(beta1, 0.9);
                assert_eq!(beta2, 0.999);
                assert_eq!(epsilon, 1e-8);
            }
            _ => panic!("Expected Adam optimizer"),
        }
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = TrainingConfig::<f32>::default();
        config.epochs = 0; // Invalid
        
        let builder = TrainingConfigBuilder { config };
        let result = builder.build();
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_presets() {
        let fast_config = TrainingPresets::fast_training::<f32>().build().unwrap();
        assert_eq!(fast_config.epochs, 50);
        assert!(fast_config.verbose);
        
        let prod_config = TrainingPresets::production_training::<f32>().build().unwrap();
        assert_eq!(prod_config.epochs, 200);
        assert!(prod_config.early_stopping.is_some());
        
        let research_config = TrainingPresets::research_training::<f32>().build().unwrap();
        assert_eq!(research_config.epochs, 500);
        assert!(research_config.checkpointing.enabled);
    }
    
    #[test]
    fn test_task_specific_presets() {
        let binary_config = TrainingPresets::binary_classification::<f32>().build().unwrap();
        assert_eq!(binary_config.loss_type, LossType::BinaryCrossEntropy);
        assert!(binary_config.metrics.contains(&MetricType::Precision));
        
        let multiclass_config = TrainingPresets::multiclass_classification::<f32>().build().unwrap();
        assert_eq!(multiclass_config.loss_type, LossType::CrossEntropy);
        
        let regression_config = TrainingPresets::regression::<f32>().build().unwrap();
        assert_eq!(regression_config.loss_type, LossType::MSE);
        assert!(regression_config.metrics.contains(&MetricType::MeanAbsoluteError));
    }
    
    #[test]
    fn test_config_validator() {
        let mut config = TrainingConfig::<f32>::default();
        config.loss_type = LossType::MSE;
        
        // Should validate successfully for regression
        assert!(ConfigValidator::validate_for_task(&config, TaskType::Regression).is_ok());
        
        // Should fail for binary classification
        assert!(ConfigValidator::validate_for_task(&config, TaskType::BinaryClassification).is_err());
    }
    
    #[test]
    fn test_hyperparameter_search() {
        let search = HyperparameterSearch::<f32>::new()
            .add_learning_rate_range(0.001, 0.1)
            .add_batch_size_range(16, 128)
            .add_weight_decay_range(0.0, 0.01);
        
        let config = search.sample_config().unwrap();
        
        // Values should be within expected ranges (simplified test)
        assert!(config.learning_rate >= 0.001);
        assert!(config.batch_size >= 16);
        assert!(config.weight_decay >= 0.0);
    }
    
    #[test]
    fn test_config_suggestions() {
        let mut config = TrainingConfig::<f32>::default();
        config.learning_rate = 1.0; // Very high
        config.batch_size = 4; // Very small
        
        let suggestions = ConfigValidator::suggest_improvements(&config);
        
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("learning rate")));
        assert!(suggestions.iter().any(|s| s.contains("batch")));
    }
}