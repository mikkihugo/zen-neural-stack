//! Main forecasting interface for neuro-divergent models
//!
//! This module provides the main NeuralForecast class that manages
//! multiple models and provides a unified interface for training and prediction.

use std::collections::HashMap;
use num_traits::Float;
use crate::errors::{NeuroDivergentError, NeuroDivergentResult};
use crate::foundation::{BaseModel, TrainingMetrics, ValidationMetrics};

/// Comprehensive training metrics utilities
mod training_metrics_utils {
    use super::*;
    
    /// Use TrainingMetrics for comprehensive model performance tracking
    pub fn create_comprehensive_training_metrics<T: Float>(
        initial_loss: T,
        final_loss: T,
        training_duration: std::time::Duration,
        epochs: usize,
        convergence_reason: &str,
    ) -> TrainingMetrics<T> {
        TrainingMetrics {
            initial_loss,
            final_loss,
            training_duration,
            epochs_completed: epochs,
            convergence_reason: convergence_reason.to_string(),
            loss_history: Vec::new(),
            best_epoch: 0,
            early_stopping_triggered: false,
            gradient_norm_history: Vec::new(),
            learning_rate_schedule: Vec::new(),
        }
    }
    
    /// Use TrainingMetrics for model comparison and selection
    pub fn compare_training_metrics<T: Float>(
        metrics_a: &TrainingMetrics<T>,
        metrics_b: &TrainingMetrics<T>,
    ) -> NeuroDivergentResult<String> {
        let improvement = metrics_a.final_loss - metrics_b.final_loss;
        let efficiency_a = metrics_a.final_loss.to_f64().unwrap_or(0.0) / metrics_a.training_duration.as_secs_f64();
        let efficiency_b = metrics_b.final_loss.to_f64().unwrap_or(0.0) / metrics_b.training_duration.as_secs_f64();
        
        let comparison = if improvement > T::zero() {
            format!("Model B performed better with {:.6} lower loss", improvement.to_f64().unwrap_or(0.0))
        } else if improvement < T::zero() {
            format!("Model A performed better with {:.6} lower loss", (-improvement).to_f64().unwrap_or(0.0))
        } else {
            "Models performed equally".to_string()
        };
        
        Ok(format!(
            "{} | Efficiency A: {:.6}, Efficiency B: {:.6} | Epochs A: {}, Epochs B: {}",
            comparison, efficiency_a, efficiency_b, metrics_a.epochs_completed, metrics_b.epochs_completed
        ))
    }
    
    /// Use TrainingMetrics for comprehensive performance analysis
    pub fn analyze_training_performance<T: Float>(
        metrics: &TrainingMetrics<T>,
        models: &[Box<dyn BaseModel<T>>],
    ) -> HashMap<String, String> {
        let mut analysis = HashMap::new();
        
        analysis.insert("final_loss".to_string(), metrics.final_loss.to_f64().unwrap_or(0.0).to_string());
        analysis.insert("training_duration_secs".to_string(), metrics.training_duration.as_secs().to_string());
        analysis.insert("epochs_completed".to_string(), metrics.epochs_completed.to_string());
        analysis.insert("convergence_reason".to_string(), metrics.convergence_reason.clone());
        analysis.insert("early_stopping".to_string(), metrics.early_stopping_triggered.to_string());
        analysis.insert("models_count".to_string(), models.len().to_string());
        
        if !metrics.loss_history.is_empty() {
            let improvement = metrics.loss_history.first().unwrap() - metrics.final_loss;
            analysis.insert("total_improvement".to_string(), improvement.to_f64().unwrap_or(0.0).to_string());
        }
        
        analysis
    }
    
    /// Use ValidationMetrics for model evaluation tracking
    pub fn track_validation_metrics<T: Float>(
        validation_metrics: &ValidationMetrics<T>,
        training_metrics: &TrainingMetrics<T>,
    ) -> NeuroDivergentResult<()> {
        let overfitting_indicator = training_metrics.final_loss.to_f64().unwrap_or(0.0) - 
                                   validation_metrics.validation_loss.to_f64().unwrap_or(0.0);
        
        if overfitting_indicator > 0.1 {
            log::warn!("Potential overfitting detected: training_loss={:.6}, validation_loss={:.6}",
                      training_metrics.final_loss.to_f64().unwrap_or(0.0),
                      validation_metrics.validation_loss.to_f64().unwrap_or(0.0));
        }
        
        log::info!("Validation metrics: accuracy={:.4}, precision={:.4}, recall={:.4}",
                  validation_metrics.accuracy,
                  validation_metrics.precision,
                  validation_metrics.recall);
        
        Ok(())
    }
}
use crate::data::{TimeSeriesDataFrame, ForecastDataFrame};
use crate::config::{TrainingConfig, PredictionConfig, CrossValidationConfig};

/// Main entry point for neural forecasting
pub struct NeuralForecast<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    models: Vec<Box<dyn BaseModel<T>>>,
    frequency: Option<String>,
    local_scaler_type: Option<String>,
    num_threads: Option<usize>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> NeuralForecast<T> {
    /// Create new NeuralForecast instance builder
    pub fn new() -> NeuralForecastBuilder<T> {
        NeuralForecastBuilder::new()
    }
    
    /// Fit models to the provided time series data
    pub fn fit(&mut self, data: TimeSeriesDataFrame<T>) -> NeuroDivergentResult<()> {
        self.fit_with_config(data, TrainingConfig::default())
    }
    
    /// Fit with custom training configuration
    pub fn fit_with_config(
        &mut self, 
        data: TimeSeriesDataFrame<T>,
        config: TrainingConfig<T>
    ) -> NeuroDivergentResult<()> {
        let dataset = data.to_dataset()?;
        
        // Use comprehensive training metrics tracking
        let mut all_training_metrics = Vec::new();
        
        for model in &mut self.models {
            let start_time = std::time::Instant::now();
            let metrics = model.fit(&dataset)?;
            let training_duration = start_time.elapsed();
            
            // Use TrainingMetrics utilities for comprehensive tracking
            let comprehensive_metrics = training_metrics_utils::create_comprehensive_training_metrics(
                T::from(1.0).unwrap(), // Initial loss placeholder
                metrics.final_loss,
                training_duration,
                metrics.epochs_completed,
                &metrics.convergence_reason,
            );
            
            // Store metrics for analysis
            all_training_metrics.push(comprehensive_metrics.clone());
            
            // Use training metrics for performance analysis
            let analysis = training_metrics_utils::analyze_training_performance(&comprehensive_metrics, &self.models);
            
            log::info!("Model {} training complete:", model.name());
            log::debug!("  Final loss: {:?}", metrics.final_loss);
            log::debug!("  Duration: {:?}", training_duration);
            log::debug!("  Epochs: {}", metrics.epochs_completed);
            log::debug!("  Convergence: {}", metrics.convergence_reason);
            
            // Log detailed analysis
            for (key, value) in analysis {
                log::trace!("  {}: {}", key, value);
            }
        }
        
        // Compare models if multiple were trained
        if all_training_metrics.len() > 1 {
            for i in 0..all_training_metrics.len() - 1 {
                if let Ok(comparison) = training_metrics_utils::compare_training_metrics(
                    &all_training_metrics[i],
                    &all_training_metrics[i + 1],
                ) {
                    log::info!("Model comparison {}-{}: {}", i, i + 1, comparison);
                }
            }
        }
        
        Ok(())
    }
    
    /// Generate forecasts for all fitted models
    pub fn predict(&self) -> NeuroDivergentResult<ForecastDataFrame<T>> {
        self.predict_with_config(PredictionConfig::default())
    }
    
    /// Generate forecasts with custom configuration
    pub fn predict_with_config(
        &self, 
        _config: PredictionConfig
    ) -> NeuroDivergentResult<ForecastDataFrame<T>> {
        // This is a placeholder implementation
        let model_names: Vec<String> = self.models.iter().map(|m| m.name().to_string()).collect();
        Ok(ForecastDataFrame::new(model_names, 24))
    }
    
    /// Cross-validation for model evaluation
    pub fn cross_validation(
        &mut self,
        data: TimeSeriesDataFrame<T>,
        config: CrossValidationConfig
    ) -> NeuroDivergentResult<HashMap<String, ValidationMetrics<T>>> {
        let mut results = HashMap::new();
        let dataset = data.to_dataset()?;
        
        for model in &mut self.models {
            let metrics = model.validate(&dataset)?;
            results.insert(model.name().to_string(), metrics);
        }
        
        Ok(results)
    }
    
    /// Get model by name
    pub fn get_model(&self, name: &str) -> Option<&dyn BaseModel<T>> {
        self.models.iter()
            .find(|m| m.name() == name)
            .map(|m| m.as_ref())
    }
    
    /// List all model names
    pub fn model_names(&self) -> Vec<String> {
        self.models.iter().map(|m| m.name().to_string()).collect()
    }
}

/// Builder for constructing NeuralForecast instances
pub struct NeuralForecastBuilder<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    models: Vec<Box<dyn BaseModel<T>>>,
    frequency: Option<String>,
    local_scaler_type: Option<String>,
    num_threads: Option<usize>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> NeuralForecastBuilder<T> {
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            frequency: None,
            local_scaler_type: None,
            num_threads: None,
        }
    }
    
    /// Add models to the forecasting ensemble
    pub fn with_models(mut self, models: Vec<Box<dyn BaseModel<T>>>) -> Self {
        self.models = models;
        self
    }
    
    /// Add a single model
    pub fn with_model(mut self, model: Box<dyn BaseModel<T>>) -> Self {
        self.models.push(model);
        self
    }
    
    /// Set the frequency of the time series
    pub fn with_frequency<S: Into<String>>(mut self, frequency: S) -> Self {
        self.frequency = Some(frequency.into());
        self
    }
    
    /// Set local scaler type for data preprocessing
    pub fn with_local_scaler<S: Into<String>>(mut self, scaler_type: S) -> Self {
        self.local_scaler_type = Some(scaler_type.into());
        self
    }
    
    /// Set number of threads for parallel processing
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }
    
    /// Build the NeuralForecast instance
    pub fn build(self) -> NeuroDivergentResult<NeuralForecast<T>> {
        if self.models.is_empty() {
            return Err(NeuroDivergentError::config("At least one model is required"));
        }
        
        Ok(NeuralForecast {
            models: self.models,
            frequency: self.frequency,
            local_scaler_type: self.local_scaler_type,
            num_threads: self.num_threads,
        })
    }
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> Default for NeuralForecastBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}