//! Comprehensive training metrics collection and analysis
//!
//! This module provides extensive metrics collection capabilities for monitoring
//! neural network training progress, performance, and convergence. Includes
//! standard metrics like accuracy, precision, recall, as well as specialized
//! metrics for different types of learning tasks.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use num_traits::Float;

use crate::TrainingError;
use super::{MetricType, EpochMetrics};

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
        }
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
}

impl<T: Float + Default> Default for MetricsConfig<T> {
    fn default() -> Self {
        Self {
            running_average_alpha: T::from(0.1).unwrap(),
            max_history_size: Some(1000),
            enable_caching: true,
            binary_threshold: T::from(0.5).unwrap(),
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