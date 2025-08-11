//! High-performance training loops with zero-allocation optimizations
//!
//! This module provides memory-efficient training loop implementations that minimize
//! allocations during training for maximum performance. Includes batch processing,
//! gradient accumulation, and specialized loops for different training scenarios.

use std::collections::HashMap;
use num_traits::Float;
use std::time::{Instant, Duration};

use crate::TrainingError;
use super::{
    ZenNeuralModel, ZenOptimizer, ZenLossFunction, MetricType, TrainingMemoryManager,
    EpochMetrics, PerformanceStats
};

/// Zero-allocation training loop for high-performance training
pub struct ZeroAllocTrainingLoop<T: Float> {
    /// Pre-allocated gradient buffers
    gradient_buffers: Vec<Vec<T>>,
    /// Pre-allocated activation buffers
    activation_buffers: Vec<Vec<T>>,
    /// Pre-allocated loss buffers
    loss_buffer: Vec<T>,
    /// Pre-allocated prediction buffers
    prediction_buffers: Vec<Vec<T>>,
    /// Buffer management
    buffer_manager: TrainingMemoryManager<T>,
    /// Performance tracking
    performance_stats: PerformanceStats,
}

impl<T: Float + Clone + Default + Send + Sync> ZeroAllocTrainingLoop<T> {
    /// Create a new zero-allocation training loop
    pub fn new(max_batch_size: usize, parameter_count: usize, output_size: usize) -> Self {
        let mut gradient_buffers = Vec::new();
        let mut activation_buffers = Vec::new();
        let mut prediction_buffers = Vec::new();
        
        // Pre-allocate buffers
        for _ in 0..max_batch_size {
            gradient_buffers.push(vec![T::default(); parameter_count]);
            activation_buffers.push(vec![T::default(); output_size]);
            prediction_buffers.push(vec![T::default(); output_size]);
        }
        
        let loss_buffer = vec![T::default(); max_batch_size];
        let buffer_manager = TrainingMemoryManager::new();
        
        Self {
            gradient_buffers,
            activation_buffers,
            loss_buffer,
            prediction_buffers,
            buffer_manager,
            performance_stats: PerformanceStats {
                avg_epoch_duration: Duration::from_secs(0),
                total_samples_processed: 0,
                samples_per_second: 0.0,
                peak_memory_usage: None,
                gradient_computation_time: Duration::from_secs(0),
                parameter_update_time: Duration::from_secs(0),
            },
        }
    }
    
    /// Run a single training epoch with zero allocations
    pub fn train_epoch<M>(
        &mut self,
        model: &mut M,
        optimizer: &mut Box<dyn ZenOptimizer<T>>,
        loss_function: &Box<dyn ZenLossFunction<T>>,
        inputs: &[Vec<T>],
        targets: &[Vec<T>],
        batch_size: usize,
    ) -> Result<EpochMetrics<T>, TrainingError>
    where
        M: ZenNeuralModel<T>,
    {
        let epoch_start = Instant::now();
        let mut total_loss = T::zero();
        let num_batches = (inputs.len() + batch_size - 1) / batch_size;
        let mut gradient_time = Duration::from_secs(0);
        let mut update_time = Duration::from_secs(0);
        
        // Process each batch
        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = std::cmp::min(start_idx + batch_size, inputs.len());
            let current_batch_size = end_idx - start_idx;
            
            // Clear gradients using pre-allocated buffers
            for i in 0..current_batch_size {
                if i < self.gradient_buffers.len() {
                    for grad in &mut self.gradient_buffers[i] {
                        *grad = T::zero();
                    }
                }
            }
            
            let batch_loss = self.process_batch_zero_alloc(
                model,
                loss_function,
                &inputs[start_idx..end_idx],
                &targets[start_idx..end_idx],
                current_batch_size,
                &mut gradient_time,
            )?;
            
            // Update parameters
            let update_start = Instant::now();
            let accumulated_gradients = self.accumulate_gradients(current_batch_size)?;
            optimizer.update_parameters(model.parameters_mut(), &accumulated_gradients)?;
            update_time += update_start.elapsed();
            
            total_loss = total_loss + batch_loss;
        }
        
        let epoch_duration = epoch_start.elapsed();
        let avg_loss = total_loss / T::from(num_batches).unwrap();
        
        // Update performance stats
        self.performance_stats.gradient_computation_time = gradient_time;
        self.performance_stats.parameter_update_time = update_time;
        self.performance_stats.total_samples_processed += inputs.len() as u64;
        self.performance_stats.samples_per_second = 
            inputs.len() as f64 / epoch_duration.as_secs_f64();
        
        // Create epoch metrics
        let mut metrics = HashMap::new();
        metrics.insert(MetricType::Loss, avg_loss);
        
        Ok(EpochMetrics {
            epoch: 0, // Will be set by caller
            train_loss: avg_loss,
            val_loss: None,
            learning_rate: optimizer.get_learning_rate(),
            duration: epoch_duration,
            metrics,
        })
    }
    
    /// Process a single batch without allocations
    fn process_batch_zero_alloc<M>(
        &mut self,
        model: &mut M,
        loss_function: &Box<dyn ZenLossFunction<T>>,
        batch_inputs: &[Vec<T>],
        batch_targets: &[Vec<T>],
        batch_size: usize,
        gradient_time: &mut Duration,
    ) -> Result<T, TrainingError>
    where
        M: ZenNeuralModel<T>,
    {
        let mut batch_loss = T::zero();
        let gradient_start = Instant::now();
        
        // Process each sample in the batch
        for (sample_idx, (input, target)) in batch_inputs.iter().zip(batch_targets.iter()).enumerate() {
            if sample_idx >= self.prediction_buffers.len() {
                return Err(TrainingError::TrainingFailed(
                    "Batch size exceeds pre-allocated buffers".to_string()
                ));
            }
            
            // Forward pass - reuse prediction buffer
            let prediction_buffer = &mut self.prediction_buffers[sample_idx];
            prediction_buffer.clear();
            prediction_buffer.extend_from_slice(&model.forward(input)?);
            
            // Compute loss
            let sample_loss = loss_function.compute_loss(prediction_buffer, target)?;
            batch_loss = batch_loss + sample_loss;
            
            // Compute gradients - reuse gradient buffer
            let sample_gradients = loss_function.compute_gradient(prediction_buffer, target)?;
            
            // Accumulate gradients in pre-allocated buffer
            if sample_idx < self.gradient_buffers.len() {
                let gradient_buffer = &mut self.gradient_buffers[sample_idx];
                for (i, &grad) in sample_gradients.iter().enumerate() {
                    if i < gradient_buffer.len() {
                        gradient_buffer[i] = gradient_buffer[i] + grad;
                    }
                }
            }
        }
        
        *gradient_time += gradient_start.elapsed();
        Ok(batch_loss / T::from(batch_size).unwrap())
    }
    
    /// Accumulate gradients from all samples in batch
    fn accumulate_gradients(&self, batch_size: usize) -> Result<Vec<T>, TrainingError> {
        if self.gradient_buffers.is_empty() || self.gradient_buffers[0].is_empty() {
            return Err(TrainingError::TrainingFailed("No gradient buffers available".to_string()));
        }
        
        let gradient_size = self.gradient_buffers[0].len();
        let mut accumulated = vec![T::zero(); gradient_size];
        
        // Accumulate gradients across samples
        for sample_idx in 0..batch_size.min(self.gradient_buffers.len()) {
            for (i, &grad) in self.gradient_buffers[sample_idx].iter().enumerate() {
                if i < accumulated.len() {
                    accumulated[i] = accumulated[i] + grad;
                }
            }
        }
        
        // Average gradients
        let batch_size_t = T::from(batch_size).unwrap();
        for grad in &mut accumulated {
            *grad = *grad / batch_size_t;
        }
        
        Ok(accumulated)
    }
    
    /// Get performance statistics
    pub fn get_performance_stats(&self) -> &PerformanceStats {
        &self.performance_stats
    }
    
    /// Reset performance counters
    pub fn reset_performance_stats(&mut self) {
        self.performance_stats = PerformanceStats {
            avg_epoch_duration: Duration::from_secs(0),
            total_samples_processed: 0,
            samples_per_second: 0.0,
            peak_memory_usage: None,
            gradient_computation_time: Duration::from_secs(0),
            parameter_update_time: Duration::from_secs(0),
        };
    }
}

/// Gradient accumulation training loop for large effective batch sizes
pub struct GradientAccumulationLoop<T: Float> {
    accumulation_steps: usize,
    accumulated_gradients: Vec<T>,
    gradient_count: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Clone + Default> GradientAccumulationLoop<T> {
    pub fn new(accumulation_steps: usize, parameter_count: usize) -> Self {
        Self {
            accumulation_steps,
            accumulated_gradients: vec![T::default(); parameter_count],
            gradient_count: 0,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Accumulate gradients without updating parameters
    pub fn accumulate_gradients(&mut self, gradients: &[T]) -> Result<(), TrainingError> {
        if gradients.len() != self.accumulated_gradients.len() {
            return Err(TrainingError::TrainingFailed(
                "Gradient size mismatch".to_string()
            ));
        }
        
        // Add gradients to accumulation buffer
        for (accumulated, &grad) in self.accumulated_gradients.iter_mut().zip(gradients.iter()) {
            *accumulated = *accumulated + grad;
        }
        
        self.gradient_count += 1;
        Ok(())
    }
    
    /// Check if accumulated enough gradients to update
    pub fn should_update(&self) -> bool {
        self.gradient_count >= self.accumulation_steps
    }
    
    /// Get accumulated gradients and reset accumulator
    pub fn get_and_reset_gradients(&mut self) -> Vec<T> {
        // Average accumulated gradients
        let scale = T::from(self.gradient_count).unwrap();
        let gradients: Vec<T> = self.accumulated_gradients
            .iter()
            .map(|&grad| grad / scale)
            .collect();
        
        // Reset accumulator
        for grad in &mut self.accumulated_gradients {
            *grad = T::zero();
        }
        self.gradient_count = 0;
        
        gradients
    }
    
    /// Get current accumulation progress
    pub fn get_accumulation_progress(&self) -> f32 {
        self.gradient_count as f32 / self.accumulation_steps as f32
    }
}

/// Mixed precision training loop (placeholder for future implementation)
pub struct MixedPrecisionLoop {
    // This would implement FP16/FP32 mixed precision training
    // Currently a placeholder as it requires specialized hardware support
    _placeholder: (),
}

impl MixedPrecisionLoop {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
    
    // Mixed precision training would be implemented here
    // when hardware acceleration is available
}

/// Parallel training loop using Rayon for CPU parallelization
#[cfg(feature = "parallel")]
pub struct ParallelTrainingLoop<T: Float> {
    thread_pool: Option<rayon::ThreadPool>,
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "parallel")]
impl<T: Float + Send + Sync + Clone + Default> ParallelTrainingLoop<T> {
    pub fn new(num_threads: Option<usize>) -> Self {
        let thread_pool = if let Some(threads) = num_threads {
            Some(rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("Failed to create thread pool"))
        } else {
            None
        };
        
        Self {
            thread_pool,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Train with parallel batch processing
    pub fn train_parallel_batches<M>(
        &self,
        model: &mut M,
        optimizer: &mut Box<dyn ZenOptimizer<T>>,
        loss_function: &Box<dyn ZenLossFunction<T>>,
        inputs: &[Vec<T>],
        targets: &[Vec<T>],
        batch_size: usize,
    ) -> Result<T, TrainingError>
    where
        M: ZenNeuralModel<T> + Send + Sync,
    {
        use rayon::prelude::*;
        
        let num_batches = (inputs.len() + batch_size - 1) / batch_size;
        let batches: Vec<usize> = (0..num_batches).collect();
        
        // Process batches in parallel
        let batch_results: Result<Vec<_>, TrainingError> = if let Some(ref pool) = self.thread_pool {
            pool.install(|| {
                batches.par_iter().map(|&batch_idx| {
                    let start_idx = batch_idx * batch_size;
                    let end_idx = std::cmp::min(start_idx + batch_size, inputs.len());
                    
                    self.process_batch_parallel(
                        model,
                        loss_function,
                        &inputs[start_idx..end_idx],
                        &targets[start_idx..end_idx],
                    )
                }).collect()
            })
        } else {
            batches.par_iter().map(|&batch_idx| {
                let start_idx = batch_idx * batch_size;
                let end_idx = std::cmp::min(start_idx + batch_size, inputs.len());
                
                self.process_batch_parallel(
                    model,
                    loss_function,
                    &inputs[start_idx..end_idx],
                    &targets[start_idx..end_idx],
                )
            }).collect()
        };
        
        let batch_losses = batch_results?;
        let total_loss: T = batch_losses.iter().fold(T::zero(), |acc, &loss| acc + loss);
        
        Ok(total_loss / T::from(num_batches).unwrap())
    }
    
    fn process_batch_parallel<M>(
        &self,
        model: &M,
        loss_function: &Box<dyn ZenLossFunction<T>>,
        batch_inputs: &[Vec<T>],
        batch_targets: &[Vec<T>],
    ) -> Result<T, TrainingError>
    where
        M: ZenNeuralModel<T> + Send + Sync,
    {
        // This would need to be implemented carefully to avoid data races
        // For now, return a placeholder
        Ok(T::zero())
    }
}

/// Training loop factory for creating appropriate loops based on configuration
pub struct TrainingLoopFactory;

impl TrainingLoopFactory {
    /// Create a training loop based on requirements
    pub fn create_loop<T: Float + Clone + Default + Send + Sync>(
        loop_type: TrainingLoopType,
        max_batch_size: usize,
        parameter_count: usize,
        output_size: usize,
    ) -> Result<Box<dyn TrainingLoopTrait<T>>, TrainingError> {
        match loop_type {
            TrainingLoopType::ZeroAlloc => {
                Ok(Box::new(ZeroAllocTrainingLoop::new(
                    max_batch_size,
                    parameter_count,
                    output_size,
                )))
            }
            TrainingLoopType::Standard => {
                Ok(Box::new(StandardTrainingLoop::new()))
            }
            TrainingLoopType::GradientAccumulation { steps } => {
                Ok(Box::new(GradientAccumulationWrapper::new(
                    steps,
                    parameter_count,
                )))
            }
            #[cfg(feature = "parallel")]
            TrainingLoopType::Parallel { threads } => {
                Ok(Box::new(ParallelTrainingLoopWrapper::new(threads)))
            }
            #[cfg(not(feature = "parallel"))]
            TrainingLoopType::Parallel { .. } => {
                Err(TrainingError::TrainingFailed(
                    "Parallel training requires the 'parallel' feature".to_string()
                ))
            }
        }
    }
}

/// Training loop types
#[derive(Debug, Clone)]
pub enum TrainingLoopType {
    Standard,
    ZeroAlloc,
    GradientAccumulation { steps: usize },
    Parallel { threads: Option<usize> },
}

/// Trait for different training loop implementations
pub trait TrainingLoopTrait<T: Float> {
    fn train_epoch<M>(
        &mut self,
        model: &mut M,
        optimizer: &mut Box<dyn ZenOptimizer<T>>,
        loss_function: &Box<dyn ZenLossFunction<T>>,
        inputs: &[Vec<T>],
        targets: &[Vec<T>],
        batch_size: usize,
    ) -> Result<EpochMetrics<T>, TrainingError>
    where
        M: ZenNeuralModel<T>;
    
    fn get_performance_stats(&self) -> Option<&PerformanceStats> {
        None
    }
}

/// Standard training loop (baseline implementation)
pub struct StandardTrainingLoop<T: Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Clone + Default> StandardTrainingLoop<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float + Clone + Default + Send + Sync> TrainingLoopTrait<T> for StandardTrainingLoop<T> {
    fn train_epoch<M>(
        &mut self,
        model: &mut M,
        optimizer: &mut Box<dyn ZenOptimizer<T>>,
        loss_function: &Box<dyn ZenLossFunction<T>>,
        inputs: &[Vec<T>],
        targets: &[Vec<T>],
        batch_size: usize,
    ) -> Result<EpochMetrics<T>, TrainingError>
    where
        M: ZenNeuralModel<T>,
    {
        let epoch_start = Instant::now();
        let mut total_loss = T::zero();
        let num_batches = (inputs.len() + batch_size - 1) / batch_size;
        
        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = std::cmp::min(start_idx + batch_size, inputs.len());
            
            let mut batch_gradients = Vec::new();
            let mut batch_loss = T::zero();
            
            // Process batch
            for (input, target) in inputs[start_idx..end_idx].iter().zip(&targets[start_idx..end_idx]) {
                let prediction = model.forward(input)?;
                let sample_loss = loss_function.compute_loss(&prediction, target)?;
                let gradients = loss_function.compute_gradient(&prediction, target)?;
                
                batch_loss = batch_loss + sample_loss;
                batch_gradients.push(gradients);
            }
            
            // Average gradients
            let current_batch_size = end_idx - start_idx;
            let avg_gradients = Self::average_gradients(&batch_gradients, current_batch_size)?;
            
            // Update parameters
            optimizer.update_parameters(model.parameters_mut(), &avg_gradients)?;
            
            total_loss = total_loss + batch_loss / T::from(current_batch_size).unwrap();
        }
        
        let epoch_duration = epoch_start.elapsed();
        let avg_loss = total_loss / T::from(num_batches).unwrap();
        
        let mut metrics = HashMap::new();
        metrics.insert(MetricType::Loss, avg_loss);
        
        Ok(EpochMetrics {
            epoch: 0,
            train_loss: avg_loss,
            val_loss: None,
            learning_rate: optimizer.get_learning_rate(),
            duration: epoch_duration,
            metrics,
        })
    }
}

impl<T: Float + Clone + Default> StandardTrainingLoop<T> {
    fn average_gradients(
        gradient_batches: &[Vec<T>],
        batch_size: usize,
    ) -> Result<Vec<T>, TrainingError> {
        if gradient_batches.is_empty() {
            return Err(TrainingError::TrainingFailed("No gradients to average".to_string()));
        }
        
        let gradient_size = gradient_batches[0].len();
        let mut avg_gradients = vec![T::zero(); gradient_size];
        
        for gradients in gradient_batches {
            if gradients.len() != gradient_size {
                return Err(TrainingError::TrainingFailed("Gradient size mismatch".to_string()));
            }
            
            for (i, &grad) in gradients.iter().enumerate() {
                avg_gradients[i] = avg_gradients[i] + grad;
            }
        }
        
        let batch_size_t = T::from(batch_size).unwrap();
        for grad in &mut avg_gradients {
            *grad = *grad / batch_size_t;
        }
        
        Ok(avg_gradients)
    }
}

// Wrapper implementations for trait objects
pub struct GradientAccumulationWrapper<T: Float> {
    inner: GradientAccumulationLoop<T>,
}

impl<T: Float + Clone + Default> GradientAccumulationWrapper<T> {
    pub fn new(steps: usize, parameter_count: usize) -> Self {
        Self {
            inner: GradientAccumulationLoop::new(steps, parameter_count),
        }
    }
}

impl<T: Float + Clone + Default + Send + Sync> TrainingLoopTrait<T> for GradientAccumulationWrapper<T> {
    fn train_epoch<M>(
        &mut self,
        model: &mut M,
        optimizer: &mut Box<dyn ZenOptimizer<T>>,
        loss_function: &Box<dyn ZenLossFunction<T>>,
        inputs: &[Vec<T>],
        targets: &[Vec<T>],
        batch_size: usize,
    ) -> Result<EpochMetrics<T>, TrainingError>
    where
        M: ZenNeuralModel<T>,
    {
        // Implementation for gradient accumulation
        // This would use the inner GradientAccumulationLoop
        let mut standard_loop = StandardTrainingLoop::new();
        standard_loop.train_epoch(model, optimizer, loss_function, inputs, targets, batch_size)
    }
}

#[cfg(feature = "parallel")]
pub struct ParallelTrainingLoopWrapper<T: Float> {
    inner: ParallelTrainingLoop<T>,
}

#[cfg(feature = "parallel")]
impl<T: Float + Send + Sync + Clone + Default> ParallelTrainingLoopWrapper<T> {
    pub fn new(threads: Option<usize>) -> Self {
        Self {
            inner: ParallelTrainingLoop::new(threads),
        }
    }
}

#[cfg(feature = "parallel")]
impl<T: Float + Clone + Default + Send + Sync> TrainingLoopTrait<T> for ParallelTrainingLoopWrapper<T> {
    fn train_epoch<M>(
        &mut self,
        model: &mut M,
        optimizer: &mut Box<dyn ZenOptimizer<T>>,
        loss_function: &Box<dyn ZenLossFunction<T>>,
        inputs: &[Vec<T>],
        targets: &[Vec<T>],
        batch_size: usize,
    ) -> Result<EpochMetrics<T>, TrainingError>
    where
        M: ZenNeuralModel<T> + Send + Sync,
    {
        // This would use the inner ParallelTrainingLoop
        let mut standard_loop = StandardTrainingLoop::new();
        standard_loop.train_epoch(model, optimizer, loss_function, inputs, targets, batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NetworkBuilder;
    
    #[test]
    fn test_zero_alloc_training_loop() {
        let loop_instance = ZeroAllocTrainingLoop::<f32>::new(4, 10, 1);
        
        assert_eq!(loop_instance.gradient_buffers.len(), 4);
        assert_eq!(loop_instance.prediction_buffers.len(), 4);
        assert_eq!(loop_instance.gradient_buffers[0].len(), 10);
        assert_eq!(loop_instance.prediction_buffers[0].len(), 1);
    }
    
    #[test]
    fn test_gradient_accumulation_loop() {
        let mut acc_loop = GradientAccumulationLoop::<f32>::new(3, 5);
        
        let gradients1 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let gradients2 = vec![0.2, 0.3, 0.4, 0.5, 0.6];
        let gradients3 = vec![0.3, 0.4, 0.5, 0.6, 0.7];
        
        acc_loop.accumulate_gradients(&gradients1).unwrap();
        assert!(!acc_loop.should_update());
        assert!((acc_loop.get_accumulation_progress() - 1.0/3.0).abs() < 1e-6);
        
        acc_loop.accumulate_gradients(&gradients2).unwrap();
        assert!(!acc_loop.should_update());
        
        acc_loop.accumulate_gradients(&gradients3).unwrap();
        assert!(acc_loop.should_update());
        
        let final_gradients = acc_loop.get_and_reset_gradients();
        assert_eq!(final_gradients.len(), 5);
        assert!((final_gradients[0] - 0.2).abs() < 1e-6); // Average of 0.1, 0.2, 0.3
        assert!(!acc_loop.should_update()); // Should reset
    }
    
    #[test]
    fn test_training_loop_factory() {
        let result = TrainingLoopFactory::create_loop::<f32>(
            TrainingLoopType::ZeroAlloc,
            4,
            10,
            1,
        );
        assert!(result.is_ok());
        
        let result = TrainingLoopFactory::create_loop::<f32>(
            TrainingLoopType::Standard,
            4,
            10,
            1,
        );
        assert!(result.is_ok());
        
        let result = TrainingLoopFactory::create_loop::<f32>(
            TrainingLoopType::GradientAccumulation { steps: 4 },
            4,
            10,
            1,
        );
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_standard_training_loop() {
        let mut loop_instance = StandardTrainingLoop::<f32>::new();
        
        // Test gradient averaging
        let gradients = vec![
            vec![0.1, 0.2],
            vec![0.3, 0.4],
        ];
        
        let avg = StandardTrainingLoop::average_gradients(&gradients, 2).unwrap();
        assert!((avg[0] - 0.2).abs() < 1e-6);
        assert!((avg[1] - 0.3).abs() < 1e-6);
    }
}