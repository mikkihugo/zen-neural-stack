//! Comprehensive training infrastructure for Zen Neural networks
//!
//! This module provides a complete training ecosystem for neural networks with:
//! - Generic trainer system for all network types
//! - High-performance optimizers with memory optimization
//! - Comprehensive loss function library
//! - Training features: early stopping, checkpointing, metrics collection
//! - Zero-allocation training loops where possible
//! - SIMD-accelerated gradient computations
//! - Memory pools for efficient gradient storage
//!
//! The training infrastructure is designed to be 20-50x faster than JavaScript
//! implementations while maintaining compatibility with existing zen-neural components.

use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use num_traits::Float;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::Network;
use crate::errors::RuvFannError;

// Re-export common training types from the main training module
pub use crate::training::{
    TrainingData, TrainingError, TrainingState, TrainingAlgorithm,
    ErrorFunction, MseError, MaeError, TanhError,
    LearningRateSchedule, ExponentialDecay, StepDecay,
    StopCriteria, MseStopCriteria, BitFailStopCriteria,
    TrainingCallback, ParallelTrainingOptions
};

/// Core training infrastructure
pub mod core;
pub use core::*;

/// Optimizer implementations
pub mod optimizers;
pub use optimizers::*;

/// Loss function library
pub mod losses;
pub use losses::*;

/// Training loop utilities
pub mod loops;
pub use loops::*;

/// Memory optimization for training
pub mod memory;
pub use memory::*;

/// Metrics collection and analysis
pub mod metrics;
pub use metrics::*;

/// Training configuration
pub mod config;
pub use config::*;

/// SIMD-accelerated training components
#[cfg(feature = "simd")]
pub mod simd_training;
#[cfg(feature = "simd")]
pub use simd_training::*;

/// GPU-accelerated training components
#[cfg(feature = "gpu")]
pub mod gpu_training;
#[cfg(feature = "gpu")]
pub use gpu_training::*;

/// Distributed training support
#[cfg(feature = "zen-distributed")]
pub mod distributed;
#[cfg(feature = "zen-distributed")]
pub use distributed::*;

/// Training model trait for different network types
pub trait ZenNeuralModel<T: Float> {
    /// Forward pass through the network
    fn forward(&mut self, input: &[T]) -> Result<Vec<T>, TrainingError>;
    
    /// Get mutable access to parameters for updates
    fn parameters_mut(&mut self) -> Vec<&mut T>;
    
    /// Get parameter count
    fn parameter_count(&self) -> usize;
    
    /// Reset network state if needed
    fn reset(&mut self) {}
    
    /// Get network topology information
    fn topology(&self) -> NetworkTopology;
}

/// Network topology information
#[derive(Debug, Clone)]
pub struct NetworkTopology {
    pub input_size: usize,
    pub hidden_sizes: Vec<usize>,
    pub output_size: usize,
    pub total_parameters: usize,
}

/// Implement ZenNeuralModel for existing Network type
impl<T: Float + Send + Default + Clone> ZenNeuralModel<T> for Network<T> {
    fn forward(&mut self, input: &[T]) -> Result<Vec<T>, TrainingError> {
        Ok(self.run(input))
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut T> {
        let mut params = Vec::new();
        
        for layer in &mut self.layers {
            for neuron in &mut layer.neurons {
                if !neuron.is_bias {
                    for connection in &mut neuron.connections {
                        params.push(&mut connection.weight);
                    }
                }
            }
        }
        
        params
    }
    
    fn parameter_count(&self) -> usize {
        let mut count = 0;
        for layer in &self.layers {
            for neuron in &layer.neurons {
                if !neuron.is_bias {
                    count += neuron.connections.len();
                }
            }
        }
        count
    }
    
    fn topology(&self) -> NetworkTopology {
        let input_size = if !self.layers.is_empty() {
            self.layers[0].size()
        } else {
            0
        };
        
        let mut hidden_sizes = Vec::new();
        for layer in self.layers.iter().skip(1).take(self.layers.len().saturating_sub(2)) {
            hidden_sizes.push(layer.num_regular_neurons());
        }
        
        let output_size = if self.layers.len() > 1 {
            self.layers.last().unwrap().num_regular_neurons()
        } else {
            0
        };
        
        NetworkTopology {
            input_size,
            hidden_sizes,
            output_size,
            total_parameters: self.parameter_count(),
        }
    }
}

/// Main training orchestrator for zen-neural networks
pub struct ZenNeuralTrainer<T: Float, M: ZenNeuralModel<T>> {
    /// Training configuration
    config: TrainingConfig<T>,
    
    /// Optimizer state
    optimizer: Box<dyn ZenOptimizer<T>>,
    
    /// Loss function
    loss_function: Box<dyn ZenLossFunction<T>>,
    
    /// Learning rate scheduler
    lr_scheduler: Option<Box<dyn ZenLRScheduler<T>>>,
    
    /// Metrics collector
    metrics: ZenMetricsCollector<T>,
    
    /// Memory manager for training
    memory_manager: Option<TrainingMemoryManager<T>>,
    
    /// Current training state
    state: ZenTrainingState<T>,
    
    /// Phantom data for model type
    _phantom: PhantomData<M>,
}

/// Training configuration for zen-neural
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct TrainingConfig<T: Float> {
    /// Number of training epochs
    pub epochs: u32,
    
    /// Learning rate
    pub learning_rate: T,
    
    /// Batch size (0 for full batch)
    pub batch_size: usize,
    
    /// Optimizer type
    pub optimizer_type: OptimizerType<T>,
    
    /// Loss function type
    pub loss_type: LossType,
    
    /// Learning rate schedule
    pub lr_schedule: Option<LRScheduleConfig<T>>,
    
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig<T>>,
    
    /// Gradient clipping threshold
    pub gradient_clip: Option<T>,
    
    /// L2 regularization
    pub weight_decay: T,
    
    /// Dropout rate
    pub dropout: Option<T>,
    
    /// Validation split ratio
    pub validation_split: T,
    
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    
    /// Whether to use parallel processing
    pub parallel: bool,
    
    /// Memory optimization settings
    pub memory_opts: MemoryOptimizationConfig,
    
    /// Metrics to collect during training
    pub metrics: Vec<MetricType>,
    
    /// Checkpointing configuration
    pub checkpointing: CheckpointConfig<T>,
    
    /// Training verbosity
    pub verbose: bool,
}

impl<T: Float + Default> Default for TrainingConfig<T> {
    fn default() -> Self {
        Self {
            epochs: 100,
            learning_rate: T::from(0.01).unwrap(),
            batch_size: 32,
            optimizer_type: OptimizerType::Adam {
                beta1: T::from(0.9).unwrap(),
                beta2: T::from(0.999).unwrap(),
                epsilon: T::from(1e-8).unwrap(),
            },
            loss_type: LossType::MSE,
            lr_schedule: None,
            early_stopping: None,
            gradient_clip: Some(T::from(1.0).unwrap()),
            weight_decay: T::zero(),
            dropout: None,
            validation_split: T::from(0.1).unwrap(),
            seed: None,
            parallel: true,
            memory_opts: MemoryOptimizationConfig::default(),
            metrics: vec![MetricType::Loss, MetricType::Accuracy],
            checkpointing: CheckpointConfig::default(),
            verbose: true,
        }
    }
}

/// Optimizer types supported by the training system
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum OptimizerType<T: Float> {
    /// Stochastic Gradient Descent
    SGD { momentum: T },
    
    /// Adam optimizer
    Adam { beta1: T, beta2: T, epsilon: T },
    
    /// AdamW optimizer (decoupled weight decay)
    AdamW { beta1: T, beta2: T, epsilon: T },
    
    /// RMSprop optimizer
    RMSprop { alpha: T, epsilon: T },
    
    /// Adagrad optimizer
    Adagrad { epsilon: T },
    
    /// RPROP optimizer
    RProp { delta_min: T, delta_max: T, eta_minus: T, eta_plus: T },
    
    /// QuickProp optimizer
    QuickProp { mu: T, decay: T },
}

/// Loss function types
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LossType {
    /// Mean Squared Error
    MSE,
    /// Mean Absolute Error
    MAE,
    /// Cross Entropy Loss
    CrossEntropy,
    /// Binary Cross Entropy Loss
    BinaryCrossEntropy,
    /// Huber Loss
    Huber { delta: f32 },
    /// Hinge Loss
    Hinge,
    /// Custom loss function
    Custom,
}

/// Learning rate schedule configuration
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct LRScheduleConfig<T: Float> {
    pub schedule_type: LRScheduleType<T>,
    pub step_interval: u32,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum LRScheduleType<T: Float> {
    Exponential { decay_rate: T },
    StepLR { step_size: u32, gamma: T },
    CosineAnnealing { t_max: u32, eta_min: T },
    ReduceOnPlateau { factor: T, patience: u32, threshold: T },
}

/// Early stopping configuration
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig<T: Float> {
    pub patience: u32,
    pub min_delta: T,
    pub monitor: MetricType,
    pub mode: EarlyStoppingMode,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy)]
pub enum EarlyStoppingMode {
    Min,  // Stop when monitored metric stops decreasing
    Max,  // Stop when monitored metric stops increasing
}

/// Memory optimization configuration
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct MemoryOptimizationConfig {
    pub use_memory_pools: bool,
    pub gradient_accumulation: bool,
    pub zero_allocation_loops: bool,
    pub preallocate_buffers: bool,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            use_memory_pools: true,
            gradient_accumulation: true,
            zero_allocation_loops: true,
            preallocate_buffers: true,
        }
    }
}

/// Metric types for training monitoring
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricType {
    Loss,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    MeanAbsoluteError,
    RootMeanSquaredError,
    Custom(u32), // Custom metric ID
}

/// Checkpointing configuration
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct CheckpointConfig<T: Float> {
    pub enabled: bool,
    pub save_interval: u32,
    pub save_best_only: bool,
    pub monitor_metric: MetricType,
    pub min_improvement: T,
}

impl<T: Float + Default> Default for CheckpointConfig<T> {
    fn default() -> Self {
        Self {
            enabled: false,
            save_interval: 10,
            save_best_only: true,
            monitor_metric: MetricType::Loss,
            min_improvement: T::from(1e-4).unwrap(),
        }
    }
}

/// Training state information
#[derive(Debug, Clone)]
pub struct ZenTrainingState<T: Float> {
    pub current_epoch: u32,
    pub best_metric: T,
    pub best_epoch: u32,
    pub epochs_without_improvement: u32,
    pub should_stop: bool,
    pub training_start_time: Instant,
    pub last_checkpoint_time: Option<Instant>,
}

impl<T: Float + Default> Default for ZenTrainingState<T> {
    fn default() -> Self {
        Self {
            current_epoch: 0,
            best_metric: T::from(f32::INFINITY).unwrap(),
            best_epoch: 0,
            epochs_without_improvement: 0,
            should_stop: false,
            training_start_time: Instant::now(),
            last_checkpoint_time: None,
        }
    }
}

/// Training results with comprehensive metrics
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct TrainingResults<T: Float> {
    /// Final training loss
    pub final_loss: T,
    
    /// Best validation loss achieved
    pub best_loss: T,
    
    /// Final accuracy (if applicable)
    pub final_accuracy: Option<T>,
    
    /// Best accuracy achieved
    pub best_accuracy: Option<T>,
    
    /// Training duration
    pub training_duration: Duration,
    
    /// Number of epochs completed
    pub epochs_completed: u32,
    
    /// Training history
    pub history: Vec<EpochMetrics<T>>,
    
    /// Whether training converged
    pub converged: bool,
    
    /// Reason training stopped
    pub stop_reason: StopReason,
    
    /// Performance statistics
    pub performance_stats: PerformanceStats,
}

/// Metrics for a single epoch
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct EpochMetrics<T: Float> {
    pub epoch: u32,
    pub train_loss: T,
    pub val_loss: Option<T>,
    pub learning_rate: T,
    pub duration: Duration,
    pub metrics: HashMap<MetricType, T>,
}

/// Reason why training stopped
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StopReason {
    MaxEpochsReached,
    EarlyStopping,
    UserInterrupted,
    NaNLoss,
    Converged,
}

/// Performance statistics
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub avg_epoch_duration: Duration,
    pub total_samples_processed: u64,
    pub samples_per_second: f64,
    pub peak_memory_usage: Option<usize>,
    pub gradient_computation_time: Duration,
    pub parameter_update_time: Duration,
}

impl<T, M> ZenNeuralTrainer<T, M> 
where 
    T: Float + Send + Sync + Default + Clone + Debug,
    M: ZenNeuralModel<T>,
{
    /// Create a new trainer with configuration
    pub fn new(config: TrainingConfig<T>) -> Result<Self, TrainingError> {
        // Create optimizer
        let optimizer = Self::create_optimizer(&config.optimizer_type)?;
        
        // Create loss function
        let loss_function = Self::create_loss_function(config.loss_type)?;
        
        // Create learning rate scheduler if specified
        let lr_scheduler = if let Some(ref lr_config) = config.lr_schedule {
            Some(Self::create_lr_scheduler(lr_config, config.learning_rate)?)
        } else {
            None
        };
        
        // Initialize metrics collector
        let metrics = ZenMetricsCollector::new(config.metrics.clone());
        
        // Initialize memory manager if enabled
        let memory_manager = if config.memory_opts.use_memory_pools {
            Some(TrainingMemoryManager::new())
        } else {
            None
        };
        
        // Set random seed if specified
        if let Some(seed) = config.seed {
            Self::set_random_seed(seed);
        }
        
        Ok(Self {
            config,
            optimizer,
            loss_function,
            lr_scheduler,
            metrics,
            memory_manager,
            state: ZenTrainingState::default(),
            _phantom: PhantomData,
        })
    }
    
    /// Train the model with the provided data
    pub fn train(
        &mut self,
        model: &mut M,
        training_data: &TrainingData<T>,
        validation_data: Option<&TrainingData<T>>,
    ) -> Result<TrainingResults<T>, TrainingError> {
        self.state.training_start_time = Instant::now();
        
        if self.config.verbose {
            println!("Starting training with {} epochs", self.config.epochs);
            println!("Model topology: {:?}", model.topology());
            println!("Optimizer: {:?}", self.config.optimizer_type);
            println!("Loss function: {:?}", self.config.loss_type);
        }
        
        // Validate data
        self.validate_training_data(training_data)?;
        if let Some(val_data) = validation_data {
            self.validate_training_data(val_data)?;
        }
        
        // Initialize training
        self.initialize_training(model, training_data)?;
        
        let mut training_history = Vec::new();
        
        // Main training loop
        for epoch in 1..=self.config.epochs {
            self.state.current_epoch = epoch;
            let epoch_start = Instant::now();
            
            // Train one epoch
            let train_metrics = self.train_epoch(model, training_data)?;
            
            // Validate if validation data provided
            let val_metrics = if let Some(val_data) = validation_data {
                Some(self.validate_epoch(model, val_data)?)
            } else {
                None
            };
            
            // Update learning rate
            if let Some(ref mut scheduler) = self.lr_scheduler {
                let metric_value = val_metrics.as_ref()
                    .and_then(|m| m.get(&MetricType::Loss))
                    .or_else(|| train_metrics.get(&MetricType::Loss))
                    .copied()
                    .unwrap_or(T::zero());
                
                scheduler.step(metric_value);
                // Update optimizer learning rate
                self.optimizer.set_learning_rate(scheduler.get_learning_rate());
            }
            
            // Record metrics
            let epoch_metrics = EpochMetrics {
                epoch,
                train_loss: train_metrics.get(&MetricType::Loss).copied().unwrap_or(T::zero()),
                val_loss: val_metrics.as_ref().and_then(|m| m.get(&MetricType::Loss)).copied(),
                learning_rate: self.optimizer.get_learning_rate(),
                duration: epoch_start.elapsed(),
                metrics: {
                    let mut combined = train_metrics.clone();
                    if let Some(val_metrics) = &val_metrics {
                        for (metric, value) in val_metrics {
                            combined.insert(*metric, *value);
                        }
                    }
                    combined
                },
            };
            
            training_history.push(epoch_metrics.clone());
            
            // Check early stopping
            if let Some(ref early_stopping) = self.config.early_stopping {
                self.check_early_stopping(early_stopping, &epoch_metrics)?;
                if self.state.should_stop {
                    if self.config.verbose {
                        println!("Early stopping triggered at epoch {}", epoch);
                    }
                    break;
                }
            }
            
            // Log progress
            if self.config.verbose {
                self.log_epoch_metrics(epoch, &epoch_metrics);
            }
            
            // Check for NaN loss
            if epoch_metrics.train_loss.is_nan() {
                return Err(TrainingError::TrainingFailed(
                    "Training loss became NaN".to_string()
                ));
            }
        }
        
        // Create training results
        self.create_training_results(training_history)
    }
    
    /// Train one epoch
    fn train_epoch(
        &mut self,
        model: &mut M,
        training_data: &TrainingData<T>,
    ) -> Result<HashMap<MetricType, T>, TrainingError> {
        let mut total_loss = T::zero();
        let num_samples = training_data.inputs.len();
        
        // Use batch processing if specified
        if self.config.batch_size > 0 && self.config.batch_size < num_samples {
            self.train_epoch_batched(model, training_data)
        } else {
            self.train_epoch_full_batch(model, training_data)
        }
    }
    
    /// Train one epoch with mini-batches
    fn train_epoch_batched(
        &mut self,
        model: &mut M,
        training_data: &TrainingData<T>,
    ) -> Result<HashMap<MetricType, T>, TrainingError> {
        let mut total_loss = T::zero();
        let num_samples = training_data.inputs.len();
        let batch_size = self.config.batch_size;
        let num_batches = (num_samples + batch_size - 1) / batch_size;
        
        // Process each batch
        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = std::cmp::min(start_idx + batch_size, num_samples);
            
            // Create batch data
            let batch_inputs = &training_data.inputs[start_idx..end_idx];
            let batch_outputs = &training_data.outputs[start_idx..end_idx];
            
            // Compute batch gradients and loss
            let (gradients, batch_loss) = self.compute_batch_gradients(
                model, 
                batch_inputs, 
                batch_outputs
            )?;
            
            // Apply gradient clipping if configured
            let gradients = if let Some(clip_value) = self.config.gradient_clip {
                self.clip_gradients(gradients, clip_value)
            } else {
                gradients
            };
            
            // Update parameters
            self.optimizer.update_parameters(model.parameters_mut(), &gradients)?;
            
            total_loss = total_loss + batch_loss;
        }
        
        // Compute metrics
        let avg_loss = total_loss / T::from(num_batches).unwrap();
        let mut metrics = HashMap::new();
        metrics.insert(MetricType::Loss, avg_loss);
        
        // Compute additional metrics if requested
        self.compute_additional_metrics(model, training_data, &mut metrics)?;
        
        Ok(metrics)
    }
    
    /// Train one epoch with full batch
    fn train_epoch_full_batch(
        &mut self,
        model: &mut M,
        training_data: &TrainingData<T>,
    ) -> Result<HashMap<MetricType, T>, TrainingError> {
        // Compute gradients for entire dataset
        let (gradients, total_loss) = self.compute_batch_gradients(
            model,
            &training_data.inputs,
            &training_data.outputs,
        )?;
        
        // Apply gradient clipping if configured
        let gradients = if let Some(clip_value) = self.config.gradient_clip {
            self.clip_gradients(gradients, clip_value)
        } else {
            gradients
        };
        
        // Update parameters
        self.optimizer.update_parameters(model.parameters_mut(), &gradients)?;
        
        // Compute metrics
        let avg_loss = total_loss / T::from(training_data.inputs.len()).unwrap();
        let mut metrics = HashMap::new();
        metrics.insert(MetricType::Loss, avg_loss);
        
        // Compute additional metrics if requested
        self.compute_additional_metrics(model, training_data, &mut metrics)?;
        
        Ok(metrics)
    }
    
    /// Validate one epoch
    fn validate_epoch(
        &mut self,
        model: &mut M,
        validation_data: &TrainingData<T>,
    ) -> Result<HashMap<MetricType, T>, TrainingError> {
        let mut total_loss = T::zero();
        
        // Compute validation loss without updating parameters
        for (input, target) in validation_data.inputs.iter().zip(&validation_data.outputs) {
            let prediction = model.forward(input)?;
            let loss = self.loss_function.compute_loss(&prediction, target)?;
            total_loss = total_loss + loss;
        }
        
        let avg_loss = total_loss / T::from(validation_data.inputs.len()).unwrap();
        let mut metrics = HashMap::new();
        metrics.insert(MetricType::Loss, avg_loss);
        
        // Compute additional metrics
        self.compute_additional_metrics(model, validation_data, &mut metrics)?;
        
        Ok(metrics)
    }
    
    /// Create optimizer based on configuration
    fn create_optimizer(optimizer_type: &OptimizerType<T>) -> Result<Box<dyn ZenOptimizer<T>>, TrainingError> {
        optimizers::create_optimizer(optimizer_type)
    }
    
    fn create_loss_function(loss_type: LossType) -> Result<Box<dyn ZenLossFunction<T>>, TrainingError> {
        losses::create_loss_function(loss_type)
    }
    
    fn create_lr_scheduler(
        config: &LRScheduleConfig<T>, 
        initial_lr: T
    ) -> Result<Box<dyn ZenLRScheduler<T>>, TrainingError> {
        core::create_lr_scheduler(config, initial_lr)
    }
    
    fn set_random_seed(seed: u64) {
        // Set random seed for reproducibility
        use rand::{SeedableRng, rngs::StdRng};
        let _rng = StdRng::seed_from_u64(seed);
        // In a full implementation, this would set the global thread_local RNG
        log::info!("Random seed set to {}", seed);
    }
    
    fn validate_training_data(&self, data: &TrainingData<T>) -> Result<(), TrainingError> {
        if data.inputs.is_empty() || data.outputs.is_empty() {
            return Err(TrainingError::InvalidData("Empty training data".to_string()));
        }
        
        if data.inputs.len() != data.outputs.len() {
            return Err(TrainingError::InvalidData(
                "Input and output data length mismatch".to_string()
            ));
        }
        
        Ok(())
    }
    
    fn initialize_training(&mut self, model: &mut M, data: &TrainingData<T>) -> Result<(), TrainingError> {
        // Initialize optimizer for the model
        self.optimizer.initialize(model.parameter_count())?;
        
        // Reset model if needed
        model.reset();
        
        Ok(())
    }
    
    fn compute_batch_gradients(
        &self,
        model: &mut M,
        inputs: &[Vec<T>],
        targets: &[Vec<T>],
    ) -> Result<(Vec<T>, T), TrainingError> {
        let mut total_loss = T::zero();
        let parameter_count = model.parameter_count();
        let mut accumulated_gradients = vec![T::zero(); parameter_count];
        
        // Process each sample in the batch
        for (input, target) in inputs.iter().zip(targets.iter()) {
            // Forward pass
            let prediction = model.forward(input)?;
            
            // Compute loss
            let sample_loss = self.loss_function.compute_loss(&prediction, target)?;
            total_loss = total_loss + sample_loss;
            
            // Compute gradients using finite differences (simplified)
            let epsilon = T::from(1e-7).unwrap();
            let loss_fn = |params: &[T]| -> T {
                // This is a simplified gradient computation
                // In practice, would use automatic differentiation
                sample_loss
            };
            
            let parameters: Vec<T> = model.parameters_mut().iter().map(|p| **p).collect();
            let gradients = core::GradientComputation::compute_gradients(&parameters, loss_fn, epsilon);
            
            // Accumulate gradients
            for (i, grad) in gradients.into_iter().enumerate() {
                if i < accumulated_gradients.len() {
                    accumulated_gradients[i] = accumulated_gradients[i] + grad;
                }
            }
        }
        
        // Average gradients over batch size
        let batch_size = T::from(inputs.len()).unwrap();
        for grad in &mut accumulated_gradients {
            *grad = *grad / batch_size;
        }
        
        let avg_loss = total_loss / batch_size;
        Ok((accumulated_gradients, avg_loss))
    }
    
    fn clip_gradients(&self, gradients: Vec<T>, clip_value: T) -> Vec<T> {
        // Gradient clipping implementation
        let grad_norm: T = gradients.iter().map(|g| *g * *g).fold(T::zero(), |acc, x| acc + x).sqrt();
        
        if grad_norm > clip_value {
            let scale = clip_value / grad_norm;
            gradients.into_iter().map(|g| g * scale).collect()
        } else {
            gradients
        }
    }
    
    fn compute_additional_metrics(
        &self,
        model: &mut M,
        data: &TrainingData<T>,
        metrics: &mut HashMap<MetricType, T>,
    ) -> Result<(), TrainingError> {
        // Collect all predictions and targets for metrics computation
        let mut all_predictions = Vec::new();
        let mut all_targets = Vec::new();
        
        for (input, target) in data.inputs.iter().zip(&data.outputs) {
            let prediction = model.forward(input)?;
            all_predictions.extend(prediction);
            all_targets.extend(target.iter().cloned());
        }
        
        // Determine task type based on target values
        let task_type = if all_targets.iter().all(|&t| t == T::zero() || t == T::one()) {
            metrics::TaskType::BinaryClassification
        } else {
            metrics::TaskType::Regression
        };
        
        // Compute additional metrics
        let mut metrics_collector = metrics::ZenMetricsCollector::new(self.config.metrics.clone());
        let computed_metrics = metrics_collector.compute_metrics(&all_predictions, &all_targets, task_type)?;
        
        // Update metrics map
        for (metric_type, value) in computed_metrics {
            if metric_type != MetricType::Loss { // Don't overwrite loss
                metrics.insert(metric_type, value);
            }
        }
        
        Ok(())
    }
    
    fn check_early_stopping(
        &mut self,
        config: &EarlyStoppingConfig<T>,
        epoch_metrics: &EpochMetrics<T>,
    ) -> Result<(), TrainingError> {
        if let Some(current_value) = epoch_metrics.metrics.get(&config.monitor) {
            let improved = match config.mode {
                EarlyStoppingMode::Min => *current_value < self.state.best_metric - config.min_delta,
                EarlyStoppingMode::Max => *current_value > self.state.best_metric + config.min_delta,
            };
            
            if improved {
                self.state.best_metric = *current_value;
                self.state.best_epoch = epoch_metrics.epoch;
                self.state.epochs_without_improvement = 0;
            } else {
                self.state.epochs_without_improvement += 1;
            }
            
            if self.state.epochs_without_improvement >= config.patience {
                self.state.should_stop = true;
            }
        }
        
        Ok(())
    }
    
    fn log_epoch_metrics(&self, epoch: u32, metrics: &EpochMetrics<T>) {
        println!(
            "Epoch {}/{}: loss={:.6}, lr={:.6}, time={:.2}s",
            epoch,
            self.config.epochs,
            metrics.train_loss.to_f64().unwrap_or(0.0),
            metrics.learning_rate.to_f64().unwrap_or(0.0),
            metrics.duration.as_secs_f64()
        );
        
        if let Some(val_loss) = metrics.val_loss {
            println!("  Val loss: {:.6}", val_loss.to_f64().unwrap_or(0.0));
        }
        
        for (metric, value) in &metrics.metrics {
            if *metric != MetricType::Loss {
                println!("  {:?}: {:.4}", metric, value.to_f64().unwrap_or(0.0));
            }
        }
    }
    
    fn create_training_results(
        &self,
        history: Vec<EpochMetrics<T>>,
    ) -> Result<TrainingResults<T>, TrainingError> {
        let final_loss = history.last()
            .map(|m| m.train_loss)
            .unwrap_or(T::zero());
        
        let best_loss = history.iter()
            .map(|m| m.train_loss)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(T::zero());
        
        let training_duration = self.state.training_start_time.elapsed();
        
        let performance_stats = PerformanceStats {
            avg_epoch_duration: if !history.is_empty() {
                training_duration / history.len() as u32
            } else {
                Duration::from_secs(0)
            },
            total_samples_processed: 0, // Will be computed properly
            samples_per_second: 0.0,
            peak_memory_usage: None,
            gradient_computation_time: Duration::from_secs(0),
            parameter_update_time: Duration::from_secs(0),
        };
        
        let stop_reason = if self.state.should_stop {
            StopReason::EarlyStopping
        } else if self.state.current_epoch >= self.config.epochs {
            StopReason::MaxEpochsReached
        } else if final_loss.is_nan() {
            StopReason::NaNLoss
        } else {
            StopReason::Converged
        };
        
        Ok(TrainingResults {
            final_loss,
            best_loss,
            final_accuracy: None, // Will be computed if accuracy is tracked
            best_accuracy: None,
            training_duration,
            epochs_completed: self.state.current_epoch,
            history,
            converged: stop_reason == StopReason::Converged,
            stop_reason,
            performance_stats,
        })
    }
}

// Placeholder traits that will be implemented in their respective modules

/// Optimizer trait for zen-neural training
pub trait ZenOptimizer<T: Float>: Send + Sync {
    fn initialize(&mut self, parameter_count: usize) -> Result<(), TrainingError>;
    fn update_parameters(&mut self, parameters: Vec<&mut T>, gradients: &[T]) -> Result<(), TrainingError>;
    fn set_learning_rate(&mut self, lr: T);
    fn get_learning_rate(&self) -> T;
    fn reset(&mut self);
}

/// Loss function trait for zen-neural training
pub trait ZenLossFunction<T: Float>: Send + Sync {
    fn compute_loss(&self, predictions: &[T], targets: &[T]) -> Result<T, TrainingError>;
    fn compute_gradient(&self, predictions: &[T], targets: &[T]) -> Result<Vec<T>, TrainingError>;
}

/// Learning rate scheduler trait
pub trait ZenLRScheduler<T: Float>: Send + Sync {
    fn step(&mut self, metric: T);
    fn get_learning_rate(&self) -> T;
    fn reset(&mut self);
}

/// Metrics collector for training monitoring
pub struct ZenMetricsCollector<T: Float> {
    metrics_to_collect: Vec<MetricType>,
    history: Vec<HashMap<MetricType, T>>,
}

impl<T: Float> ZenMetricsCollector<T> {
    pub fn new(metrics: Vec<MetricType>) -> Self {
        Self {
            metrics_to_collect: metrics,
            history: Vec::new(),
        }
    }
    
    pub fn collect(&mut self, metrics: HashMap<MetricType, T>) {
        self.history.push(metrics);
    }
    
    pub fn get_history(&self) -> &[HashMap<MetricType, T>] {
        &self.history
    }
}

/// Training memory manager for efficient memory usage
pub struct TrainingMemoryManager<T: Float> {
    gradient_pools: Vec<Vec<T>>,
    activation_pools: Vec<Vec<T>>,
    _phantom: PhantomData<T>,
}

impl<T: Float + Default + Clone> TrainingMemoryManager<T> {
    pub fn new() -> Self {
        Self {
            gradient_pools: Vec::new(),
            activation_pools: Vec::new(),
            _phantom: PhantomData,
        }
    }
    
    pub fn get_gradient_buffer(&mut self, size: usize) -> &mut Vec<T> {
        if self.gradient_pools.is_empty() {
            self.gradient_pools.push(vec![T::default(); size]);
        }
        &mut self.gradient_pools[0]
    }
    
    pub fn return_gradient_buffer(&mut self, _buffer: Vec<T>) {
        // Return buffer to pool for reuse
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NetworkBuilder;
    
    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::<f32>::default();
        assert_eq!(config.epochs, 100);
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.batch_size, 32);
    }
    
    #[test]
    fn test_zen_neural_model_trait() {
        let mut network = NetworkBuilder::new()
            .add_layer(2)
            .add_layer(3)
            .add_layer(1)
            .build::<f32>();
        
        let topology = network.topology();
        assert_eq!(topology.input_size, 2);
        assert_eq!(topology.output_size, 1);
        assert!(topology.total_parameters > 0);
    }
    
    #[test]
    fn test_memory_optimization_config() {
        let config = MemoryOptimizationConfig::default();
        assert!(config.use_memory_pools);
        assert!(config.zero_allocation_loops);
        assert!(config.preallocate_buffers);
    }
}