//! Comprehensive training metrics collection and analysis
//!
//! This module provides extensive metrics collection capabilities for monitoring
//! neural network training progress, performance, and convergence. Includes
//! standard metrics like accuracy, precision, recall, as well as specialized
//! metrics for different types of learning tasks.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use num_traits::Float;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::TrainingError;
use super::{MetricType, EpochMetrics};

/// High-performance atomic counters for training metrics
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Default)]
pub struct AtomicTrainingCounters {
    /// Total training samples processed
    pub samples_processed: AtomicU64,
    /// Total training epochs completed
    pub epochs_completed: AtomicU64,
    /// Total forward passes executed
    pub forward_passes: AtomicU64,
    /// Total backward passes executed
    pub backward_passes: AtomicU64,
    /// Total gradient updates applied
    pub gradient_updates: AtomicU64,
    /// Total training time in microseconds
    pub training_time_us: AtomicU64,
    /// Total inference time in microseconds
    pub inference_time_us: AtomicU64,
    /// Total parameter updates count
    pub parameter_updates: AtomicU64,
    /// Total loss computations
    pub loss_computations: AtomicU64,
    /// Total metric computations
    pub metric_computations: AtomicU64,
    /// Training convergence checks
    pub convergence_checks: AtomicU64,
    /// Early stopping triggers
    pub early_stopping_triggers: AtomicU64,
    /// Learning rate adjustments
    pub lr_adjustments: AtomicU64,
    /// Batch processing operations
    pub batch_operations: AtomicU64,
    /// Memory allocation operations
    pub memory_allocations: AtomicU64,
    /// GPU operations (if available)
    pub gpu_operations: AtomicU64,
    /// Cache hits for optimization
    pub cache_hits: AtomicU64,
    /// Cache misses
    pub cache_misses: AtomicU64,
    /// Thread contention events
    pub thread_contentions: AtomicU64,
    /// Optimization recommendation triggers
    pub optimization_recommendations: AtomicU64,
}

/// Comprehensive metrics collector for training monitoring
pub struct ZenMetricsCollector<T: Float> {
    /// Metrics to collect
    enabled_metrics: Vec<MetricType>,
    /// Historical metrics data
    history: Vec<HashMap<MetricType, T>>,
    /// Running averages for smoothed metrics
    running_averages: HashMap<MetricType, T>,
    /// Best values achieved for each metric
    best_values: HashMap<MetricType, T>,
    /// Metric computation cache
    computation_cache: HashMap<MetricType, T>,
    /// Performance tracking
    computation_times: HashMap<MetricType, Duration>,
    /// Metric configurations
    config: MetricsConfig<T>,
    /// High-performance atomic counters
    pub atomic_counters: Arc<AtomicTrainingCounters>,
    /// Performance health scorer
    health_scorer: TrainingHealthScorer<T>,
    /// Optimization recommender
    optimizer_recommender: OptimizationRecommender<T>,
    /// Trend analyzer for convergence detection
    trend_analyzer: TrendAnalyzer<T>,
}

impl AtomicTrainingCounters {
    /// Create new atomic training counters
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Reset all counters to zero
    pub fn reset(&self) {
        self.samples_processed.store(0, Ordering::SeqCst);
        self.epochs_completed.store(0, Ordering::SeqCst);
        self.forward_passes.store(0, Ordering::SeqCst);
        self.backward_passes.store(0, Ordering::SeqCst);
        self.gradient_updates.store(0, Ordering::SeqCst);
        self.training_time_us.store(0, Ordering::SeqCst);
        self.inference_time_us.store(0, Ordering::SeqCst);
        self.parameter_updates.store(0, Ordering::SeqCst);
        self.loss_computations.store(0, Ordering::SeqCst);
        self.metric_computations.store(0, Ordering::SeqCst);
        self.convergence_checks.store(0, Ordering::SeqCst);
        self.early_stopping_triggers.store(0, Ordering::SeqCst);
        self.lr_adjustments.store(0, Ordering::SeqCst);
        self.batch_operations.store(0, Ordering::SeqCst);
        self.memory_allocations.store(0, Ordering::SeqCst);
        self.gpu_operations.store(0, Ordering::SeqCst);
        self.cache_hits.store(0, Ordering::SeqCst);
        self.cache_misses.store(0, Ordering::SeqCst);
        self.thread_contentions.store(0, Ordering::SeqCst);
        self.optimization_recommendations.store(0, Ordering::SeqCst);
    }
    
    /// Get comprehensive performance statistics
    pub fn get_performance_stats(&self) -> TrainingPerformanceStats {
        let samples = self.samples_processed.load(Ordering::SeqCst);
        let training_time = self.training_time_us.load(Ordering::SeqCst);
        let inference_time = self.inference_time_us.load(Ordering::SeqCst);
        let total_time = training_time + inference_time;
        
        let samples_per_second = if total_time > 0 {
            (samples as f64 * 1_000_000.0) / total_time as f64
        } else {
            0.0
        };
        
        let cache_total = self.cache_hits.load(Ordering::SeqCst) + self.cache_misses.load(Ordering::SeqCst);
        let cache_hit_rate = if cache_total > 0 {
            self.cache_hits.load(Ordering::SeqCst) as f64 / cache_total as f64
        } else {
            0.0
        };
        
        TrainingPerformanceStats {
            samples_processed: samples,
            epochs_completed: self.epochs_completed.load(Ordering::SeqCst),
            total_time_us: total_time,
            training_time_us: training_time,
            inference_time_us: inference_time,
            samples_per_second,
            forward_passes: self.forward_passes.load(Ordering::SeqCst),
            backward_passes: self.backward_passes.load(Ordering::SeqCst),
            gradient_updates: self.gradient_updates.load(Ordering::SeqCst),
            parameter_updates: self.parameter_updates.load(Ordering::SeqCst),
            cache_hit_rate,
            gpu_operations: self.gpu_operations.load(Ordering::SeqCst),
            thread_contentions: self.thread_contentions.load(Ordering::SeqCst),
            optimization_recommendations: self.optimization_recommendations.load(Ordering::SeqCst),
        }
    }
    
    /// Calculate training efficiency score (0.0-100.0)
    pub fn calculate_efficiency_score(&self) -> f64 {
        let stats = self.get_performance_stats();
        let mut score = 50.0; // Base score
        
        // Factor in cache hit rate (20% of score)
        score += stats.cache_hit_rate * 20.0;
        
        // Factor in samples per second (30% of score, normalized)
        let normalized_sps = (stats.samples_per_second / 1000.0).min(30.0);
        score += normalized_sps;
        
        // Penalize thread contentions (up to -20% of score)
        let contention_penalty = (stats.thread_contentions as f64 / stats.samples_processed.max(1) as f64) * 20.0;
        score -= contention_penalty.min(20.0);
        
        score.max(0.0).min(100.0)
    }
    
    /// Record a training sample being processed
    pub fn record_sample_processed(&self, count: u64) {
        self.samples_processed.fetch_add(count, Ordering::SeqCst);
    }
    
    /// Record an epoch completion
    pub fn record_epoch_completed(&self) {
        self.epochs_completed.fetch_add(1, Ordering::SeqCst);
    }
    
    /// Record forward pass execution
    pub fn record_forward_pass(&self, count: u64) {
        self.forward_passes.fetch_add(count, Ordering::SeqCst);
    }
    
    /// Record backward pass execution
    pub fn record_backward_pass(&self, count: u64) {
        self.backward_passes.fetch_add(count, Ordering::SeqCst);
    }
    
    /// Record training time
    pub fn record_training_time_us(&self, duration_us: u64) {
        self.training_time_us.fetch_add(duration_us, Ordering::SeqCst);
    }
    
    /// Record inference time
    pub fn record_inference_time_us(&self, duration_us: u64) {
        self.inference_time_us.fetch_add(duration_us, Ordering::SeqCst);
    }
    
    /// Record cache hit
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::SeqCst);
    }
    
    /// Record cache miss
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::SeqCst);
    }
    
    /// Record thread contention event
    pub fn record_thread_contention(&self) {
        self.thread_contentions.fetch_add(1, Ordering::SeqCst);
    }
    
    /// Record GPU operation
    pub fn record_gpu_operation(&self, count: u64) {
        self.gpu_operations.fetch_add(count, Ordering::SeqCst);
    }
}

impl<T: Float + Clone + Default + Send + Sync> ZenMetricsCollector<T> {
    /// Create a new metrics collector
    pub fn new(enabled_metrics: Vec<MetricType>) -> Self {
        Self {
            enabled_metrics,
            history: Vec::new(),
            running_averages: HashMap::new(),
            best_values: HashMap::new(),
            computation_cache: HashMap::new(),
            computation_times: HashMap::new(),
            config: MetricsConfig::default(),
            atomic_counters: Arc::new(AtomicTrainingCounters::new()),
            health_scorer: TrainingHealthScorer::new(),
            optimizer_recommender: OptimizationRecommender::new(),
            trend_analyzer: TrendAnalyzer::new(),
        }
    }
    
    /// Create a metrics collector with custom configuration
    pub fn with_config(enabled_metrics: Vec<MetricType>, config: MetricsConfig<T>) -> Self {
        Self {
            enabled_metrics,
            history: Vec::new(),
            running_averages: HashMap::new(),
            best_values: HashMap::new(),
            computation_cache: HashMap::new(),
            computation_times: HashMap::new(),
            config,
            atomic_counters: Arc::new(AtomicTrainingCounters::new()),
            health_scorer: TrainingHealthScorer::new(),
            optimizer_recommender: OptimizationRecommender::new(),
            trend_analyzer: TrendAnalyzer::new(),
        }
    }
    
    /// Get shared reference to atomic counters for thread-safe access
    pub fn get_atomic_counters(&self) -> Arc<AtomicTrainingCounters> {
        self.atomic_counters.clone()
    }
    
    /// Calculate comprehensive training health score
    pub fn calculate_training_health(&self) -> TrainingHealthScore<T> {
        let performance_stats = self.atomic_counters.get_performance_stats();
        self.health_scorer.calculate_health(&performance_stats, &self.history, &self.config)
    }
    
    /// Get optimization recommendations based on current metrics
    pub fn get_optimization_recommendations(&self) -> Vec<OptimizationRecommendation<T>> {
        let performance_stats = self.atomic_counters.get_performance_stats();
        self.optimizer_recommender.analyze_performance(&performance_stats, &self.history, &self.config)
    }
    
    /// Analyze training convergence trends
    pub fn analyze_convergence_trends(&self, metric_type: MetricType) -> ConvergenceTrend<T> {
        self.trend_analyzer.analyze_metric_trend(&self.history, metric_type, &self.config)
    }
    
    /// Record performance metrics automatically during training
    pub fn record_training_step(&self, duration: Duration, batch_size: usize, forward_time: Duration, backward_time: Duration) {
        let duration_us = duration.as_micros() as u64;
        let forward_us = forward_time.as_micros() as u64;
        let backward_us = backward_time.as_micros() as u64;
        
        self.atomic_counters.record_sample_processed(batch_size as u64);
        self.atomic_counters.record_training_time_us(duration_us);
        self.atomic_counters.record_forward_pass(1);
        self.atomic_counters.record_backward_pass(1);
        self.atomic_counters.training_time_us.fetch_add(forward_us + backward_us, Ordering::SeqCst);
        self.atomic_counters.gradient_updates.fetch_add(1, Ordering::SeqCst);
        self.atomic_counters.parameter_updates.fetch_add(1, Ordering::SeqCst);
        self.atomic_counters.metric_computations.fetch_add(1, Ordering::SeqCst);
    }
    
    /// Compute metrics for predictions and targets
    pub fn compute_metrics(
        &mut self,
        predictions: &[T],
        targets: &[T],
        task_type: TaskType,
    ) -> Result<HashMap<MetricType, T>, TrainingError> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::InvalidData(
                "Predictions and targets must have the same length".to_string()
            ));
        }
        
        let mut metrics = HashMap::new();
        
        for &metric_type in &self.enabled_metrics {
            let start_time = Instant::now();
            
            let value = match metric_type {
                MetricType::Loss => {
                    // Loss should be computed by loss function, use cached value
                    self.computation_cache.get(&MetricType::Loss)
                        .copied()
                        .unwrap_or(T::zero())
                }
                MetricType::Accuracy => {
                    self.compute_accuracy(predictions, targets, task_type)?
                }
                MetricType::Precision => {
                    self.compute_precision(predictions, targets, task_type)?
                }
                MetricType::Recall => {
                    self.compute_recall(predictions, targets, task_type)?
                }
                MetricType::F1Score => {
                    self.compute_f1_score(predictions, targets, task_type)?
                }
                MetricType::MeanAbsoluteError => {
                    self.compute_mae(predictions, targets)?
                }
                MetricType::RootMeanSquaredError => {
                    self.compute_rmse(predictions, targets)?
                }
                MetricType::Custom(id) => {
                    self.compute_custom_metric(id, predictions, targets)?
                }
            };
            
            let computation_time = start_time.elapsed();
            self.computation_times.insert(metric_type, computation_time);
            
            metrics.insert(metric_type, value);
        }
        
        // Update running averages
        self.update_running_averages(&metrics);
        
        // Update best values
        self.update_best_values(&metrics);
        
        Ok(metrics)
    }
    
    /// Set cached loss value (computed by loss function)
    pub fn set_loss_value(&mut self, loss: T) {
        self.computation_cache.insert(MetricType::Loss, loss);
    }
    
    /// Record metrics for an epoch
    pub fn record_epoch_metrics(&mut self, metrics: HashMap<MetricType, T>) {
        self.history.push(metrics);
        
        // Keep only recent history if configured
        if let Some(max_history) = self.config.max_history_size {
            if self.history.len() > max_history {
                self.history.drain(0..self.history.len() - max_history);
            }
        }
    }
    
    /// Get historical metrics
    pub fn get_history(&self) -> &[HashMap<MetricType, T>] {
        &self.history
    }
    
    /// Get running average for a metric
    pub fn get_running_average(&self, metric: MetricType) -> Option<T> {
        self.running_averages.get(&metric).copied()
    }
    
    /// Get best value achieved for a metric
    pub fn get_best_value(&self, metric: MetricType) -> Option<T> {
        self.best_values.get(&metric).copied()
    }
    
    /// Get computation time for a metric
    pub fn get_computation_time(&self, metric: MetricType) -> Option<Duration> {
        self.computation_times.get(&metric).copied()
    }
    
    /// Get recent trend for a metric (positive = improving, negative = deteriorating)
    pub fn get_metric_trend(&self, metric: MetricType, window_size: usize) -> Option<T> {
        if self.history.len() < 2 {
            return None;
        }
        
        let start_idx = if self.history.len() > window_size {
            self.history.len() - window_size
        } else {
            0
        };
        
        let recent_values: Vec<T> = self.history[start_idx..]
            .iter()
            .filter_map(|epoch_metrics| epoch_metrics.get(&metric).copied())
            .collect();
        
        if recent_values.len() < 2 {
            return None;
        }
        
        // Simple linear trend: (last - first) / window_size
        let first = recent_values[0];
        let last = recent_values[recent_values.len() - 1];
        let trend = (last - first) / T::from(recent_values.len()).unwrap();
        
        Some(trend)
    }
    
    /// Check if metric has improved recently
    pub fn has_improved_recently(
        &self,
        metric: MetricType,
        window_size: usize,
        min_improvement: T,
    ) -> bool {
        if let Some(trend) = self.get_metric_trend(metric, window_size) {
            // For loss metrics, improvement is negative trend
            // For accuracy metrics, improvement is positive trend
            match metric {
                MetricType::Loss | MetricType::MeanAbsoluteError | MetricType::RootMeanSquaredError => {
                    trend < -min_improvement
                }
                _ => trend > min_improvement,
            }
        } else {
            false
        }
    }
    
    /// Reset all metrics and history
    pub fn reset(&mut self) {
        self.history.clear();
        self.running_averages.clear();
        self.best_values.clear();
        self.computation_cache.clear();
        self.computation_times.clear();
        self.atomic_counters.reset();
        self.health_scorer.reset();
        self.optimizer_recommender.reset();
        self.trend_analyzer.reset();
    }
    
    // Metric computation methods
    
    fn compute_accuracy(
        &self,
        predictions: &[T],
        targets: &[T],
        task_type: TaskType,
    ) -> Result<T, TrainingError> {
        match task_type {
            TaskType::BinaryClassification => {
                let correct = predictions.iter()
                    .zip(targets.iter())
                    .filter(|(&pred, &target)| {
                        let pred_class = if pred > T::from(0.5).unwrap() { T::one() } else { T::zero() };
                        pred_class == target
                    })
                    .count();
                
                Ok(T::from(correct).unwrap() / T::from(predictions.len()).unwrap())
            }
            TaskType::MultiClassification => {
                // Assume predictions are class probabilities and targets are one-hot
                let correct = predictions.chunks(targets.len() / predictions.len())
                    .zip(targets.chunks(targets.len() / predictions.len()))
                    .filter(|(pred_chunk, target_chunk)| {
                        let pred_class = self.argmax(pred_chunk);
                        let target_class = self.argmax(target_chunk);
                        pred_class == target_class
                    })
                    .count();
                
                let num_samples = predictions.len() / (targets.len() / predictions.len());
                Ok(T::from(correct).unwrap() / T::from(num_samples).unwrap())
            }
            TaskType::Regression => {
                // For regression, use coefficient of determination (R²)
                self.compute_r_squared(predictions, targets)
            }
        }
    }
    
    fn compute_precision(
        &self,
        predictions: &[T],
        targets: &[T],
        task_type: TaskType,
    ) -> Result<T, TrainingError> {
        match task_type {
            TaskType::BinaryClassification => {
                let (tp, fp, _tn, _fn) = self.compute_confusion_matrix(predictions, targets)?;
                
                if tp + fp == T::zero() {
                    Ok(T::zero()) // No positive predictions
                } else {
                    Ok(tp / (tp + fp))
                }
            }
            TaskType::MultiClassification => {
                // Macro-averaged precision
                // This would require more complex implementation for multi-class
                Ok(T::zero()) // Placeholder
            }
            TaskType::Regression => {
                Err(TrainingError::InvalidData(
                    "Precision not applicable to regression tasks".to_string()
                ))
            }
        }
    }
    
    fn compute_recall(
        &self,
        predictions: &[T],
        targets: &[T],
        task_type: TaskType,
    ) -> Result<T, TrainingError> {
        match task_type {
            TaskType::BinaryClassification => {
                let (tp, _fp, _tn, fn_val) = self.compute_confusion_matrix(predictions, targets)?;
                
                if tp + fn_val == T::zero() {
                    Ok(T::zero()) // No positive targets
                } else {
                    Ok(tp / (tp + fn_val))
                }
            }
            TaskType::MultiClassification => {
                // Macro-averaged recall
                Ok(T::zero()) // Placeholder
            }
            TaskType::Regression => {
                Err(TrainingError::InvalidData(
                    "Recall not applicable to regression tasks".to_string()
                ))
            }
        }
    }
    
    fn compute_f1_score(
        &self,
        predictions: &[T],
        targets: &[T],
        task_type: TaskType,
    ) -> Result<T, TrainingError> {
        let precision = self.compute_precision(predictions, targets, task_type)?;
        let recall = self.compute_recall(predictions, targets, task_type)?;
        
        if precision + recall == T::zero() {
            Ok(T::zero())
        } else {
            Ok(T::from(2.0).unwrap() * precision * recall / (precision + recall))
        }
    }
    
    fn compute_mae(&self, predictions: &[T], targets: &[T]) -> Result<T, TrainingError> {
        let sum: T = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - target).abs())
            .fold(T::zero(), |acc, x| acc + x);
        
        Ok(sum / T::from(predictions.len()).unwrap())
    }
    
    fn compute_rmse(&self, predictions: &[T], targets: &[T]) -> Result<T, TrainingError> {
        let sum_squared: T = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let diff = pred - target;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x);
        
        let mse = sum_squared / T::from(predictions.len()).unwrap();
        Ok(mse.sqrt())
    }
    
    fn compute_r_squared(&self, predictions: &[T], targets: &[T]) -> Result<T, TrainingError> {
        // Coefficient of determination
        let target_mean: T = targets.iter().fold(T::zero(), |acc, &x| acc + x) 
            / T::from(targets.len()).unwrap();
        
        let ss_res: T = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let diff = target - pred;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x);
        
        let ss_tot: T = targets.iter()
            .map(|&target| {
                let diff = target - target_mean;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x);
        
        if ss_tot == T::zero() {
            Ok(T::zero())
        } else {
            Ok(T::one() - ss_res / ss_tot)
        }
    }
    
    fn compute_custom_metric(
        &self,
        _metric_id: u32,
        _predictions: &[T],
        _targets: &[T],
    ) -> Result<T, TrainingError> {
        // Custom metrics would be implemented based on metric_id
        Ok(T::zero()) // Placeholder
    }
    
    fn compute_confusion_matrix(
        &self,
        predictions: &[T],
        targets: &[T],
    ) -> Result<(T, T, T, T), TrainingError> {
        // Returns (true_positive, false_positive, true_negative, false_negative)
        let threshold = T::from(0.5).unwrap();
        
        let mut tp = T::zero();
        let mut fp = T::zero();
        let mut tn = T::zero();
        let mut fn_val = T::zero();
        
        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let pred_positive = pred > threshold;
            let target_positive = target > threshold;
            
            match (pred_positive, target_positive) {
                (true, true) => tp = tp + T::one(),
                (true, false) => fp = fp + T::one(),
                (false, false) => tn = tn + T::one(),
                (false, true) => fn_val = fn_val + T::one(),
            }
        }
        
        Ok((tp, fp, tn, fn_val))
    }
    
    fn argmax(&self, values: &[T]) -> usize {
        values.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap_or(0)
    }
    
    fn update_running_averages(&mut self, new_metrics: &HashMap<MetricType, T>) {
        let alpha = self.config.running_average_alpha;
        
        for (&metric, &new_value) in new_metrics {
            let avg = self.running_averages.entry(metric).or_insert(new_value);
            *avg = alpha * new_value + (T::one() - alpha) * *avg;
        }
    }
    
    fn update_best_values(&mut self, new_metrics: &HashMap<MetricType, T>) {
        for (&metric, &new_value) in new_metrics {
            let is_better = match metric {
                MetricType::Loss | MetricType::MeanAbsoluteError | MetricType::RootMeanSquaredError => {
                    // Lower is better
                    self.best_values.get(&metric)
                        .map(|&best| new_value < best)
                        .unwrap_or(true)
                }
                _ => {
                    // Higher is better
                    self.best_values.get(&metric)
                        .map(|&best| new_value > best)
                        .unwrap_or(true)
                }
            };
            
            if is_better {
                self.best_values.insert(metric, new_value);
            }
        }
    }
}

/// Task types for different metric computations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskType {
    BinaryClassification,
    MultiClassification,
    Regression,
}

/// Training performance statistics from atomic counters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct TrainingPerformanceStats {
    pub samples_processed: u64,
    pub epochs_completed: u64,
    pub total_time_us: u64,
    pub training_time_us: u64,
    pub inference_time_us: u64,
    pub samples_per_second: f64,
    pub forward_passes: u64,
    pub backward_passes: u64,
    pub gradient_updates: u64,
    pub parameter_updates: u64,
    pub cache_hit_rate: f64,
    pub gpu_operations: u64,
    pub thread_contentions: u64,
    pub optimization_recommendations: u64,
}

/// Comprehensive training health assessment
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct TrainingHealthScore<T: Float> {
    pub overall_score: T,
    pub performance_score: T,
    pub convergence_score: T,
    pub efficiency_score: T,
    pub stability_score: T,
    pub recommendations: Vec<String>,
    pub status: HealthStatus,
}

/// Health status classification
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

/// Optimization recommendation
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation<T: Float> {
    pub category: OptimizationCategory,
    pub priority: RecommendationPriority,
    pub description: String,
    pub estimated_improvement: T,
    pub implementation_difficulty: DifficultyLevel,
}

/// Optimization categories
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationCategory {
    LearningRate,
    BatchSize,
    Architecture,
    Regularization,
    DataProcessing,
    ComputeOptimization,
    MemoryOptimization,
}

/// Recommendation priority levels
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Implementation difficulty levels
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifficultyLevel {
    Easy,
    Medium,
    Hard,
    VeryHard,
}

/// Convergence trend analysis
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct ConvergenceTrend<T: Float> {
    pub trend_direction: TrendDirection,
    pub convergence_rate: T,
    pub stability: T,
    pub predicted_epochs_to_convergence: Option<u64>,
    pub confidence: T,
}

/// Trend direction classification
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Improving,
    Stable,
    Deteriorating,
    Oscillating,
    Converged,
}

/// Configuration for metrics collection
#[derive(Debug, Clone)]
pub struct MetricsConfig<T: Float> {
    /// Alpha parameter for exponential moving averages
    pub running_average_alpha: T,
    /// Maximum number of historical epochs to keep
    pub max_history_size: Option<usize>,
    /// Whether to cache intermediate computations
    pub enable_caching: bool,
    /// Precision threshold for binary classification
    pub binary_threshold: T,
    /// Convergence tolerance for trend analysis
    pub convergence_tolerance: T,
    /// Window size for trend analysis
    pub trend_window_size: usize,
    /// Minimum epochs before providing recommendations
    pub min_epochs_for_recommendations: usize,
}

impl<T: Float + Default> Default for MetricsConfig<T> {
    fn default() -> Self {
        Self {
            running_average_alpha: T::from(0.1).unwrap(),
            max_history_size: Some(1000),
            enable_caching: true,
            binary_threshold: T::from(0.5).unwrap(),
            convergence_tolerance: T::from(1e-6).unwrap(),
            trend_window_size: 20,
            min_epochs_for_recommendations: 5,
        }
    }
}

/// Metrics dashboard for real-time monitoring
pub struct MetricsDashboard<T: Float> {
    collectors: HashMap<String, ZenMetricsCollector<T>>,
    update_interval: Duration,
    last_update: Instant,
}

impl<T: Float + Clone + Default + Send + Sync> MetricsDashboard<T> {
    pub fn new(update_interval: Duration) -> Self {
        Self {
            collectors: HashMap::new(),
            update_interval,
            last_update: Instant::now(),
        }
    }
    
    /// Add a metrics collector
    pub fn add_collector(&mut self, name: String, collector: ZenMetricsCollector<T>) {
        self.collectors.insert(name, collector);
    }
    
    /// Update all collectors with new metrics
    pub fn update_metrics(
        &mut self,
        predictions: &[T],
        targets: &[T],
        task_type: TaskType,
    ) -> Result<(), TrainingError> {
        if self.last_update.elapsed() < self.update_interval {
            return Ok(()); // Skip update if too soon
        }
        
        for collector in self.collectors.values_mut() {
            collector.compute_metrics(predictions, targets, task_type)?;
        }
        
        self.last_update = Instant::now();
        Ok(())
    }
    
    /// Get metrics from a specific collector
    pub fn get_collector_metrics(&self, name: &str) -> Option<&ZenMetricsCollector<T>> {
        self.collectors.get(name)
    }
    
    /// Get summary of all metrics
    pub fn get_metrics_summary(&self) -> HashMap<String, HashMap<MetricType, T>> {
        let mut summary = HashMap::new();
        
        for (name, collector) in &self.collectors {
            if let Some(latest_metrics) = collector.history.last() {
                summary.insert(name.clone(), latest_metrics.clone());
            }
        }
        
        summary
    }
    
    /// Get comprehensive performance statistics from all collectors
    pub fn get_comprehensive_stats(&self) -> HashMap<String, TrainingPerformanceStats> {
        self.collectors.iter()
            .map(|(name, collector)| {
                (name.clone(), collector.atomic_counters.get_performance_stats())
            })
            .collect()
    }
    
    /// Get health assessment from all collectors
    pub fn get_health_assessment(&self) -> HashMap<String, TrainingHealthScore<T>> {
        self.collectors.iter()
            .map(|(name, collector)| {
                (name.clone(), collector.calculate_training_health())
            })
            .collect()
    }
    
    /// Get optimization recommendations from all collectors
    pub fn get_all_optimization_recommendations(&self) -> HashMap<String, Vec<OptimizationRecommendation<T>>> {
        self.collectors.iter()
            .map(|(name, collector)| {
                (name.clone(), collector.get_optimization_recommendations())
            })
            .collect()
    }
}

/// Training health scorer for comprehensive assessment
#[derive(Debug, Default)]
pub struct TrainingHealthScorer<T: Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Clone + Default + Send + Sync> TrainingHealthScorer<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
    
    pub fn calculate_health(
        &self,
        performance_stats: &TrainingPerformanceStats,
        history: &[HashMap<MetricType, T>],
        config: &MetricsConfig<T>,
    ) -> TrainingHealthScore<T> {
        let performance_score = self.calculate_performance_score(performance_stats);
        let convergence_score = self.calculate_convergence_score(history, config);
        let efficiency_score = T::from(performance_stats.cache_hit_rate * 100.0).unwrap();
        let stability_score = self.calculate_stability_score(history);
        
        let overall_score = (performance_score + convergence_score + efficiency_score + stability_score) / T::from(4.0).unwrap();
        
        let status = match overall_score {
            score if score >= T::from(90.0).unwrap() => HealthStatus::Excellent,
            score if score >= T::from(75.0).unwrap() => HealthStatus::Good,
            score if score >= T::from(50.0).unwrap() => HealthStatus::Fair,
            score if score >= T::from(25.0).unwrap() => HealthStatus::Poor,
            _ => HealthStatus::Critical,
        };
        
        let recommendations = self.generate_health_recommendations(&status, performance_stats);
        
        TrainingHealthScore {
            overall_score,
            performance_score,
            convergence_score,
            efficiency_score,
            stability_score,
            recommendations,
            status,
        }
    }
    
    fn calculate_performance_score(&self, stats: &TrainingPerformanceStats) -> T {
        let mut score = T::from(50.0).unwrap(); // Base score
        
        // Factor in samples per second (normalized to 0-50 range)
        let sps_score = T::from((stats.samples_per_second / 100.0).min(50.0)).unwrap();
        score = score + sps_score;
        
        // Penalize thread contentions
        let contention_penalty = if stats.samples_processed > 0 {
            T::from((stats.thread_contentions as f64 / stats.samples_processed as f64) * 25.0).unwrap()
        } else {
            T::zero()
        };
        score = score - contention_penalty;
        
        score.max(T::zero()).min(T::from(100.0).unwrap())
    }
    
    fn calculate_convergence_score(&self, history: &[HashMap<MetricType, T>], _config: &MetricsConfig<T>) -> T {
        if history.len() < 3 {
            return T::from(50.0).unwrap(); // Not enough data
        }
        
        // Check if loss is generally decreasing
        let recent_losses: Vec<T> = history.iter()
            .rev()
            .take(10)
            .filter_map(|metrics| metrics.get(&MetricType::Loss).copied())
            .collect();
        
        if recent_losses.len() < 2 {
            return T::from(50.0).unwrap();
        }
        
        let trend = recent_losses.first().unwrap() - recent_losses.last().unwrap();
        if trend > T::zero() {
            T::from(80.0).unwrap() // Loss is decreasing
        } else {
            T::from(30.0).unwrap() // Loss is not decreasing
        }
    }
    
    fn calculate_stability_score(&self, history: &[HashMap<MetricType, T>]) -> T {
        if history.len() < 5 {
            return T::from(50.0).unwrap();
        }
        
        // Calculate variance in recent loss values
        let recent_losses: Vec<T> = history.iter()
            .rev()
            .take(10)
            .filter_map(|metrics| metrics.get(&MetricType::Loss).copied())
            .collect();
        
        if recent_losses.len() < 3 {
            return T::from(50.0).unwrap();
        }
        
        let mean: T = recent_losses.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(recent_losses.len()).unwrap();
        let variance: T = recent_losses.iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x) / T::from(recent_losses.len()).unwrap();
        
        // Lower variance = higher stability score
        let stability_score = T::from(100.0).unwrap() - (variance * T::from(1000.0).unwrap()).min(T::from(100.0).unwrap());
        stability_score.max(T::zero())
    }
    
    fn generate_health_recommendations(&self, status: &HealthStatus, stats: &TrainingPerformanceStats) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        match status {
            HealthStatus::Critical | HealthStatus::Poor => {
                recommendations.push("Consider reducing learning rate".to_string());
                recommendations.push("Check data quality and preprocessing".to_string());
                
                if stats.thread_contentions > stats.samples_processed / 10 {
                    recommendations.push("High thread contention detected - consider reducing parallelism".to_string());
                }
            }
            HealthStatus::Fair => {
                recommendations.push("Monitor convergence closely".to_string());
                
                if stats.cache_hit_rate < 0.7 {
                    recommendations.push("Low cache hit rate - consider optimizing data access patterns".to_string());
                }
            }
            HealthStatus::Good | HealthStatus::Excellent => {
                if stats.samples_per_second < 50.0 {
                    recommendations.push("Consider increasing batch size for better throughput".to_string());
                }
            }
        }
        
        recommendations
    }
    
    pub fn reset(&mut self) {
        // Nothing to reset for now
    }
}

/// Optimization recommender for performance improvements
#[derive(Debug, Default)]
pub struct OptimizationRecommender<T: Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Clone + Default + Send + Sync> OptimizationRecommender<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
    
    pub fn analyze_performance(
        &self,
        performance_stats: &TrainingPerformanceStats,
        history: &[HashMap<MetricType, T>],
        config: &MetricsConfig<T>,
    ) -> Vec<OptimizationRecommendation<T>> {
        let mut recommendations = Vec::new();
        
        // Learning rate recommendations
        if let Some(lr_rec) = self.analyze_learning_rate(history, config) {
            recommendations.push(lr_rec);
        }
        
        // Batch size recommendations
        if let Some(batch_rec) = self.analyze_batch_size(performance_stats) {
            recommendations.push(batch_rec);
        }
        
        // Compute optimization recommendations
        if let Some(compute_rec) = self.analyze_compute_optimization(performance_stats) {
            recommendations.push(compute_rec);
        }
        
        // Memory optimization recommendations
        if let Some(memory_rec) = self.analyze_memory_optimization(performance_stats) {
            recommendations.push(memory_rec);
        }
        
        recommendations
    }
    
    fn analyze_learning_rate(&self, history: &[HashMap<MetricType, T>], _config: &MetricsConfig<T>) -> Option<OptimizationRecommendation<T>> {
        if history.len() < 10 {
            return None;
        }
        
        // Analyze loss progression
        let recent_losses: Vec<T> = history.iter()
            .rev()
            .take(10)
            .filter_map(|metrics| metrics.get(&MetricType::Loss).copied())
            .collect();
        
        if recent_losses.len() < 5 {
            return None;
        }
        
        let initial_loss = recent_losses.last().unwrap();
        let current_loss = recent_losses.first().unwrap();
        
        if *current_loss > *initial_loss {
            // Loss is increasing - reduce learning rate
            Some(OptimizationRecommendation {
                category: OptimizationCategory::LearningRate,
                priority: RecommendationPriority::High,
                description: "Loss is increasing - consider reducing learning rate by 50%".to_string(),
                estimated_improvement: T::from(15.0).unwrap(),
                implementation_difficulty: DifficultyLevel::Easy,
            })
        } else if (*initial_loss - *current_loss) < T::from(0.001).unwrap() {
            // Loss plateaued - might need adjustment
            Some(OptimizationRecommendation {
                category: OptimizationCategory::LearningRate,
                priority: RecommendationPriority::Medium,
                description: "Loss appears to have plateaued - consider learning rate scheduling".to_string(),
                estimated_improvement: T::from(10.0).unwrap(),
                implementation_difficulty: DifficultyLevel::Medium,
            })
        } else {
            None
        }
    }
    
    fn analyze_batch_size(&self, stats: &TrainingPerformanceStats) -> Option<OptimizationRecommendation<T>> {
        if stats.samples_per_second < 20.0 {
            Some(OptimizationRecommendation {
                category: OptimizationCategory::DataProcessing,
                priority: RecommendationPriority::Medium,
                description: "Low throughput detected - consider increasing batch size".to_string(),
                estimated_improvement: T::from(25.0).unwrap(),
                implementation_difficulty: DifficultyLevel::Easy,
            })
        } else {
            None
        }
    }
    
    fn analyze_compute_optimization(&self, stats: &TrainingPerformanceStats) -> Option<OptimizationRecommendation<T>> {
        if stats.gpu_operations == 0 && stats.samples_processed > 1000 {
            Some(OptimizationRecommendation {
                category: OptimizationCategory::ComputeOptimization,
                priority: RecommendationPriority::High,
                description: "No GPU operations detected - consider GPU acceleration for large datasets".to_string(),
                estimated_improvement: T::from(200.0).unwrap(),
                implementation_difficulty: DifficultyLevel::Hard,
            })
        } else {
            None
        }
    }
    
    fn analyze_memory_optimization(&self, stats: &TrainingPerformanceStats) -> Option<OptimizationRecommendation<T>> {
        if stats.cache_hit_rate < 0.6 {
            Some(OptimizationRecommendation {
                category: OptimizationCategory::MemoryOptimization,
                priority: RecommendationPriority::Medium,
                description: format!("Low cache hit rate ({:.1}%) - optimize data access patterns", stats.cache_hit_rate * 100.0),
                estimated_improvement: T::from(20.0).unwrap(),
                implementation_difficulty: DifficultyLevel::Medium,
            })
        } else {
            None
        }
    }
    
    pub fn reset(&mut self) {
        // Nothing to reset for now
    }
}

/// Trend analyzer for convergence detection
#[derive(Debug, Default)]
pub struct TrendAnalyzer<T: Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Clone + Default + Send + Sync> TrendAnalyzer<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
    
    pub fn analyze_metric_trend(
        &self,
        history: &[HashMap<MetricType, T>],
        metric_type: MetricType,
        config: &MetricsConfig<T>,
    ) -> ConvergenceTrend<T> {
        let values: Vec<T> = history.iter()
            .rev()
            .take(config.trend_window_size)
            .filter_map(|metrics| metrics.get(&metric_type).copied())
            .collect();
        
        if values.len() < 3 {
            return ConvergenceTrend {
                trend_direction: TrendDirection::Stable,
                convergence_rate: T::zero(),
                stability: T::from(0.5).unwrap(),
                predicted_epochs_to_convergence: None,
                confidence: T::from(0.1).unwrap(),
            };
        }
        
        let trend_direction = self.calculate_trend_direction(&values);
        let convergence_rate = self.calculate_convergence_rate(&values);
        let stability = self.calculate_stability(&values);
        let predicted_epochs = self.predict_convergence_epochs(&values, config);
        let confidence = self.calculate_confidence(&values, config);
        
        ConvergenceTrend {
            trend_direction,
            convergence_rate,
            stability,
            predicted_epochs_to_convergence: predicted_epochs,
            confidence,
        }
    }
    
    fn calculate_trend_direction(&self, values: &[T]) -> TrendDirection {
        if values.len() < 3 {
            return TrendDirection::Stable;
        }
        
        let first_third = &values[0..values.len()/3];
        let last_third = &values[(2*values.len()/3)..];
        
        let first_avg = first_third.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(first_third.len()).unwrap();
        let last_avg = last_third.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(last_third.len()).unwrap();
        
        let diff = last_avg - first_avg;
        let threshold = T::from(0.01).unwrap();
        
        if diff > threshold {
            TrendDirection::Deteriorating
        } else if diff < -threshold {
            TrendDirection::Improving
        } else {
            TrendDirection::Stable
        }
    }
    
    fn calculate_convergence_rate(&self, values: &[T]) -> T {
        if values.len() < 2 {
            return T::zero();
        }
        
        let first = values.last().unwrap();
        let last = values.first().unwrap();
        let rate = (*first - *last).abs() / T::from(values.len()).unwrap();
        rate
    }
    
    fn calculate_stability(&self, values: &[T]) -> T {
        if values.len() < 2 {
            return T::from(0.5).unwrap();
        }
        
        let mean = values.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(values.len()).unwrap();
        let variance = values.iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x) / T::from(values.len()).unwrap();
        
        // Convert variance to stability score (0-1)
        let stability = T::one() / (T::one() + variance);
        stability
    }
    
    fn predict_convergence_epochs(&self, values: &[T], config: &MetricsConfig<T>) -> Option<u64> {
        if values.len() < 5 {
            return None;
        }
        
        let convergence_rate = self.calculate_convergence_rate(values);
        let current_value = *values.first().unwrap();
        
        if convergence_rate > T::zero() {
            let epochs_to_convergence = current_value / convergence_rate;
            if epochs_to_convergence < T::from(1000.0).unwrap() {
                Some(epochs_to_convergence.to_u64().unwrap_or(u64::MAX))
            } else {
                None
            }
        } else {
            None
        }
    }
    
    fn calculate_confidence(&self, values: &[T], _config: &MetricsConfig<T>) -> T {
        let stability = self.calculate_stability(values);
        let data_completeness = T::from(values.len() as f64 / 20.0).unwrap().min(T::one());
        
        (stability + data_completeness) / T::from(2.0).unwrap()
    }
    
    pub fn reset(&mut self) {
        // Nothing to reset for now
    }
}

// Helper trait implementations for numeric conversions
trait NumericConversions<T> {
    fn to_u64(self) -> Option<u64>;
}

impl NumericConversions<f32> for f32 {
    fn to_u64(self) -> Option<u64> {
        if self >= 0.0 && self <= u64::MAX as f32 {
            Some(self as u64)
        } else {
            None
        }
    }
}

impl NumericConversions<f64> for f64 {
    fn to_u64(self) -> Option<u64> {
        if self >= 0.0 && self <= u64::MAX as f64 {
            Some(self as u64)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_collector() {
        let metrics = vec![MetricType::Loss, MetricType::Accuracy];
        let mut collector = ZenMetricsCollector::<f32>::new(metrics);
        
        collector.set_loss_value(0.5);
        
        let predictions = vec![0.8, 0.3, 0.9, 0.1];
        let targets = vec![1.0, 0.0, 1.0, 0.0];
        
        let computed_metrics = collector
            .compute_metrics(&predictions, &targets, TaskType::BinaryClassification)
            .unwrap();
        
        assert!(computed_metrics.contains_key(&MetricType::Loss));
        assert!(computed_metrics.contains_key(&MetricType::Accuracy));
        assert_eq!(computed_metrics[&MetricType::Loss], 0.5);
        
        // Accuracy should be 1.0 (all predictions correct)
        assert!((computed_metrics[&MetricType::Accuracy] - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_confusion_matrix() {
        let collector = ZenMetricsCollector::<f32>::new(vec![]);
        
        let predictions = vec![0.8, 0.3, 0.7, 0.1];
        let targets = vec![1.0, 0.0, 1.0, 0.0];
        
        let (tp, fp, tn, fn_val) = collector
            .compute_confusion_matrix(&predictions, &targets)
            .unwrap();
        
        assert_eq!(tp, 2.0); // True positives
        assert_eq!(fp, 0.0); // False positives
        assert_eq!(tn, 2.0); // True negatives
        assert_eq!(fn_val, 0.0); // False negatives
    }
    
    #[test]
    fn test_mae_computation() {
        let collector = ZenMetricsCollector::<f32>::new(vec![]);
        
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.1, 1.8, 3.2];
        
        let mae = collector.compute_mae(&predictions, &targets).unwrap();
        let expected_mae = (0.1 + 0.2 + 0.2) / 3.0;
        
        assert!((mae - expected_mae).abs() < 1e-6);
    }
    
    #[test]
    fn test_rmse_computation() {
        let collector = ZenMetricsCollector::<f32>::new(vec![]);
        
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.1, 1.8, 3.2];
        
        let rmse = collector.compute_rmse(&predictions, &targets).unwrap();
        let expected_rmse = ((0.01 + 0.04 + 0.04) / 3.0_f32).sqrt();
        
        assert!((rmse - expected_rmse).abs() < 1e-6);
    }
    
    #[test]
    fn test_r_squared_computation() {
        let collector = ZenMetricsCollector::<f32>::new(vec![]);
        
        let predictions = vec![2.0, 4.0, 6.0];
        let targets = vec![2.0, 4.0, 6.0]; // Perfect predictions
        
        let r2 = collector.compute_r_squared(&predictions, &targets).unwrap();
        assert!((r2 - 1.0).abs() < 1e-6); // Should be 1.0 for perfect fit
    }
    
    #[test]
    fn test_running_averages() {
        let mut collector = ZenMetricsCollector::<f32>::new(vec![MetricType::Accuracy]);
        
        let mut metrics1 = HashMap::new();
        metrics1.insert(MetricType::Accuracy, 0.8);
        collector.update_running_averages(&metrics1);
        
        let mut metrics2 = HashMap::new();
        metrics2.insert(MetricType::Accuracy, 0.9);
        collector.update_running_averages(&metrics2);
        
        let avg = collector.get_running_average(MetricType::Accuracy).unwrap();
        // Should be weighted average: 0.1 * 0.9 + 0.9 * 0.8 = 0.81
        assert!((avg - 0.81).abs() < 1e-6);
    }
    
    #[test]
    fn test_metrics_dashboard() {
        let mut dashboard = MetricsDashboard::<f32>::new(Duration::from_millis(100));
        
        let collector = ZenMetricsCollector::new(vec![MetricType::Accuracy]);
        dashboard.add_collector("test".to_string(), collector);
        
        let predictions = vec![0.8, 0.3];
        let targets = vec![1.0, 0.0];
        
        dashboard.update_metrics(&predictions, &targets, TaskType::BinaryClassification).unwrap();
        
        let summary = dashboard.get_metrics_summary();
        assert!(summary.contains_key("test"));
    }
}