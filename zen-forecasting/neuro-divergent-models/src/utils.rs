//! Utility functions for neural forecasting models
//!
//! This module provides common utility functions used across different models,
//! including mathematical operations, data preprocessing, and validation helpers.

use num_traits::Float;
use std::collections::HashMap;
use crate::errors::{NeuroDivergentError, NeuroDivergentResult};

/// Configuration and metadata utilities
mod config_utils {
    use super::*;
    
    /// Use HashMap for model configuration and hyperparameter storage
    pub fn create_model_config<T: Float>() -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("model_type".to_string(), "neuro_divergent".to_string());
        config.insert("precision".to_string(), std::any::type_name::<T>().to_string());
        config.insert("version".to_string(), env!("CARGO_PKG_VERSION").to_string());
        config.insert("features".to_string(), "math_utils,validation,preprocessing".to_string());
        config
    }
    
    /// Use HashMap for validation metadata and error tracking
    pub fn create_validation_metadata(
        total_samples: usize,
        validation_errors: Vec<String>,
        performance_metrics: Vec<f64>,
    ) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("total_samples".to_string(), total_samples.to_string());
        metadata.insert("error_count".to_string(), validation_errors.len().to_string());
        metadata.insert("errors".to_string(), validation_errors.join("; "));
        
        if !performance_metrics.is_empty() {
            let avg_performance: f64 = performance_metrics.iter().sum::<f64>() / performance_metrics.len() as f64;
            metadata.insert("avg_performance".to_string(), avg_performance.to_string());
            metadata.insert("min_performance".to_string(), performance_metrics.iter().fold(f64::INFINITY, |a, &b| a.min(b)).to_string());
            metadata.insert("max_performance".to_string(), performance_metrics.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)).to_string());
        }
        
        metadata
    }
    
    /// Use HashMap for hyperparameter tuning results
    pub fn store_hyperparameter_results<T: Float>(
        hyperparams: &[(String, f64)],
        performance_score: T,
        training_time_ms: u64,
    ) -> HashMap<String, String> {
        let mut results = HashMap::new();
        
        // Store hyperparameters
        for (param, value) in hyperparams {
            results.insert(format!("hyperparam_{}", param), value.to_string());
        }
        
        // Store performance metrics
        results.insert("performance_score".to_string(), performance_score.to_string());
        results.insert("training_time_ms".to_string(), training_time_ms.to_string());
        results.insert("timestamp".to_string(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs().to_string());
        
        results
    }
}

/// Mathematical utility functions
pub mod math {
    use super::*;
    
    /// Calculate mean squared error
    pub fn mse<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(actual: &[T], predicted: &[T]) -> NeuroDivergentResult<T> {
        if actual.len() != predicted.len() {
            return Err(NeuroDivergentError::dimension_mismatch(actual.len(), predicted.len()));
        }
        
        let sum = actual.iter()
            .zip(predicted.iter())
            .map(|(&a, &p)| {
                let diff = a - p;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x);
            
        Ok(sum / T::from(actual.len()).unwrap())
    }
    
    /// Calculate mean absolute error
    pub fn mae<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(actual: &[T], predicted: &[T]) -> NeuroDivergentResult<T> {
        if actual.len() != predicted.len() {
            return Err(NeuroDivergentError::dimension_mismatch(actual.len(), predicted.len()));
        }
        
        let sum = actual.iter()
            .zip(predicted.iter())
            .map(|(&a, &p)| (a - p).abs())
            .fold(T::zero(), |acc, x| acc + x);
            
        Ok(sum / T::from(actual.len()).unwrap())
    }
    
    /// Calculate root mean squared error
    pub fn rmse<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(actual: &[T], predicted: &[T]) -> NeuroDivergentResult<T> {
        let mse_value = mse(actual, predicted)?;
        Ok(mse_value.sqrt())
    }
    
    /// Apply softmax function
    pub fn softmax<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(inputs: &[T]) -> Vec<T> {
        let max_val = inputs.iter().fold(inputs[0], |max, &x| max.max(x));
        let exp_values: Vec<T> = inputs.iter()
            .map(|&x| (x - max_val).exp())
            .collect();
        let sum_exp: T = exp_values.iter().copied().sum();
        
        exp_values.iter()
            .map(|&exp_val| exp_val / sum_exp)
            .collect()
    }
    
    /// Calculate dot product of two vectors
    pub fn dot_product<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(a: &[T], b: &[T]) -> NeuroDivergentResult<T> {
        if a.len() != b.len() {
            return Err(NeuroDivergentError::dimension_mismatch(a.len(), b.len()));
        }
        
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum())
    }
    
    /// Matrix multiplication for 2D vectors
    pub fn matrix_multiply<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(a: &[Vec<T>], b: &[Vec<T>]) -> NeuroDivergentResult<Vec<Vec<T>>> {
        if a.is_empty() || b.is_empty() {
            return Err(NeuroDivergentError::data("Cannot multiply empty matrices"));
        }
        
        let rows_a = a.len();
        let cols_a = a[0].len();
        let rows_b = b.len();
        let cols_b = b[0].len();
        
        if cols_a != rows_b {
            return Err(NeuroDivergentError::dimension_mismatch(cols_a, rows_b));
        }
        
        let mut result = vec![vec![T::zero(); cols_b]; rows_a];
        
        for i in 0..rows_a {
            for j in 0..cols_b {
                for k in 0..cols_a {
                    result[i][j] = result[i][j] + a[i][k] * b[k][j];
                }
            }
        }
        
        Ok(result)
    }
}

/// Data preprocessing utilities
pub mod preprocessing {
    use super::*;
    
    /// Min-Max scaler
    #[derive(Debug, Clone)]
    pub struct MinMaxScaler<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
        min_val: Option<T>,
        max_val: Option<T>,
        range: (T, T), // (min_range, max_range)
    }
    
    impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> MinMaxScaler<T> {
        pub fn new() -> Self {
            Self {
                min_val: None,
                max_val: None,
                range: (T::zero(), T::one()),
            }
        }
        
        pub fn with_range(mut self, min_range: T, max_range: T) -> Self {
            self.range = (min_range, max_range);
            self
        }
        
        pub fn fit(&mut self, data: &[T]) -> NeuroDivergentResult<()> {
            if data.is_empty() {
                return Err(NeuroDivergentError::data("Cannot fit scaler on empty data"));
            }
            
            self.min_val = Some(data.iter().fold(data[0], |min, &x| min.min(x)));
            self.max_val = Some(data.iter().fold(data[0], |max, &x| max.max(x)));
            
            Ok(())
        }
        
        pub fn transform(&self, data: &[T]) -> NeuroDivergentResult<Vec<T>> {
            let min_val = self.min_val.ok_or_else(|| 
                NeuroDivergentError::state("Scaler has not been fitted"))?;
            let max_val = self.max_val.ok_or_else(|| 
                NeuroDivergentError::state("Scaler has not been fitted"))?;
            
            if max_val == min_val {
                return Ok(vec![self.range.0; data.len()]);
            }
            
            let scale = (self.range.1 - self.range.0) / (max_val - min_val);
            let transformed = data.iter()
                .map(|&x| (x - min_val) * scale + self.range.0)
                .collect();
                
            Ok(transformed)
        }
        
        pub fn inverse_transform(&self, data: &[T]) -> NeuroDivergentResult<Vec<T>> {
            let min_val = self.min_val.ok_or_else(|| 
                NeuroDivergentError::state("Scaler has not been fitted"))?;
            let max_val = self.max_val.ok_or_else(|| 
                NeuroDivergentError::state("Scaler has not been fitted"))?;
            
            if max_val == min_val {
                return Ok(vec![min_val; data.len()]);
            }
            
            let scale = (max_val - min_val) / (self.range.1 - self.range.0);
            let transformed = data.iter()
                .map(|&x| (x - self.range.0) * scale + min_val)
                .collect();
                
            Ok(transformed)
        }
        
        pub fn fit_transform(&mut self, data: &[T]) -> NeuroDivergentResult<Vec<T>> {
            self.fit(data)?;
            self.transform(data)
        }
    }
    
    /// Standard scaler (Z-score normalization)
    #[derive(Debug, Clone)]
    pub struct StandardScaler<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
        mean: Option<T>,
        std: Option<T>,
    }
    
    impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> StandardScaler<T> {
        pub fn new() -> Self {
            Self {
                mean: None,
                std: None,
            }
        }
        
        pub fn fit(&mut self, data: &[T]) -> NeuroDivergentResult<()> {
            if data.is_empty() {
                return Err(NeuroDivergentError::data("Cannot fit scaler on empty data"));
            }
            
            let n = T::from(data.len()).unwrap();
            let mean = data.iter().copied().sum::<T>() / n;
            
            let variance = data.iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum::<T>() / n;
                
            let std = variance.sqrt();
            
            self.mean = Some(mean);
            self.std = Some(std);
            
            Ok(())
        }
        
        pub fn transform(&self, data: &[T]) -> NeuroDivergentResult<Vec<T>> {
            let mean = self.mean.ok_or_else(|| 
                NeuroDivergentError::state("Scaler has not been fitted"))?;
            let std = self.std.ok_or_else(|| 
                NeuroDivergentError::state("Scaler has not been fitted"))?;
            
            if std == T::zero() {
                return Ok(vec![T::zero(); data.len()]);
            }
            
            let transformed = data.iter()
                .map(|&x| (x - mean) / std)
                .collect();
                
            Ok(transformed)
        }
        
        pub fn inverse_transform(&self, data: &[T]) -> NeuroDivergentResult<Vec<T>> {
            let mean = self.mean.ok_or_else(|| 
                NeuroDivergentError::state("Scaler has not been fitted"))?;
            let std = self.std.ok_or_else(|| 
                NeuroDivergentError::state("Scaler has not been fitted"))?;
            
            let transformed = data.iter()
                .map(|&x| x * std + mean)
                .collect();
                
            Ok(transformed)
        }
        
        pub fn fit_transform(&mut self, data: &[T]) -> NeuroDivergentResult<Vec<T>> {
            self.fit(data)?;
            self.transform(data)
        }
    }
}

/// Validation utilities
pub mod validation {
    use super::*;
    
    /// Validate input dimensions
    pub fn validate_dimensions<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(
        input: &[T], 
        expected_size: usize, 
        name: &str
    ) -> NeuroDivergentResult<()> {
        if input.len() != expected_size {
            return Err(NeuroDivergentError::dimension_mismatch(expected_size, input.len()));
        }
        Ok(())
    }
    
    /// Validate that vectors have the same length
    pub fn validate_same_length<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(
        a: &[T], 
        b: &[T], 
        name_a: &str, 
        name_b: &str
    ) -> NeuroDivergentResult<()> {
        if a.len() != b.len() {
            return Err(NeuroDivergentError::dimension_mismatch(a.len(), b.len()));
        }
        Ok(())
    }
    
    /// Validate that all values are finite
    pub fn validate_finite<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(data: &[T], name: &str) -> NeuroDivergentResult<()> {
        for (i, &value) in data.iter().enumerate() {
            if !value.is_finite() {
                return Err(NeuroDivergentError::data(
                    format!("Non-finite value at index {} in {}: {:?}", i, name, value)
                ));
            }
        }
        Ok(())
    }
    
    /// Validate that values are in a specific range
    pub fn validate_range<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(
        data: &[T], 
        min_val: T, 
        max_val: T, 
        name: &str
    ) -> NeuroDivergentResult<()> {
        for (i, &value) in data.iter().enumerate() {
            if value < min_val || value > max_val {
                return Err(NeuroDivergentError::data(
                    format!("Value at index {} in {} out of range [{:?}, {:?}]: {:?}", 
                           i, name, min_val, max_val, value)
                ));
            }
        }
        Ok(())
    }
}

/// Time series specific utilities
pub mod timeseries {
    use super::*;
    
    /// Create sliding windows from a time series
    pub fn create_sliding_windows<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(
        data: &[T], 
        window_size: usize, 
        step_size: usize
    ) -> Vec<Vec<T>> {
        let mut windows = Vec::new();
        
        for i in (0..data.len()).step_by(step_size) {
            if i + window_size <= data.len() {
                windows.push(data[i..i + window_size].to_vec());
            }
        }
        
        windows
    }
    
    /// Split time series into train/validation sets
    pub fn train_validation_split<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(
        data: &[T], 
        validation_split: f64
    ) -> NeuroDivergentResult<(Vec<T>, Vec<T>)> {
        if validation_split < 0.0 || validation_split >= 1.0 {
            return Err(NeuroDivergentError::config(
                "Validation split must be in range [0, 1)"
            ));
        }
        
        let split_index = ((1.0 - validation_split) * data.len() as f64) as usize;
        
        Ok((
            data[..split_index].to_vec(),
            data[split_index..].to_vec(),
        ))
    }
    
    /// Calculate seasonal decomposition (simple moving average)
    pub fn simple_seasonal_decompose<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(
        data: &[T], 
        period: usize
    ) -> NeuroDivergentResult<(Vec<T>, Vec<T>)> {
        if period == 0 || period > data.len() {
            return Err(NeuroDivergentError::config(
                format!("Invalid period {} for data length {}", period, data.len())
            ));
        }
        
        let mut trend = Vec::with_capacity(data.len());
        let mut seasonal = Vec::with_capacity(data.len());
        
        // Simple moving average for trend
        for i in 0..data.len() {
            let start = i.saturating_sub(period / 2);
            let end = (i + period / 2 + 1).min(data.len());
            
            let sum: T = data[start..end].iter().copied().sum();
            let count = T::from(end - start).unwrap();
            let avg = sum / count;
            
            trend.push(avg);
            seasonal.push(data[i] - avg);
        }
        
        Ok((trend, seasonal))
    }
}