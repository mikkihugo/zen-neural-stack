//! Data structures for time series handling
//!
//! This module provides data structures for handling time series data,
//! including DataFrame-like structures for time series operations.

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use num_traits::Float;
use std::marker::PhantomData;
use crate::errors::{NeuroDivergentError, NeuroDivergentResult};
use crate::foundation::{TimeSeriesDataset, TimeSeriesSample, TimeSeriesInput};

/// Comprehensive time series sample and input utilities
mod time_series_utils {
    use super::*;
    
    /// Use TimeSeriesSample for comprehensive data point representation
    pub fn create_time_series_sample<T: Float>(
        timestamp: DateTime<Utc>,
        target_value: T,
        features: Vec<T>,
        metadata: HashMap<String, String>,
    ) -> TimeSeriesSample<T> {
        TimeSeriesSample {
            timestamp,
            target: target_value,
            features,
            metadata,
            sequence_length: features.len(),
            is_training: true,
            weight: T::one(),
        }
    }
    
    /// Use TimeSeriesInput for batch processing operations
    pub fn create_batch_time_series_input<T: Float>(
        samples: Vec<TimeSeriesSample<T>>,
        window_size: usize,
        horizon: usize,
    ) -> NeuroDivergentResult<TimeSeriesInput<T>> {
        if samples.is_empty() {
            return Err(NeuroDivergentError::ValidationError {
                message: "Cannot create input from empty samples".to_string(),
                details: HashMap::new(),
            });
        }
        
        let input = TimeSeriesInput {
            sequences: samples.iter().map(|s| s.features.clone()).collect(),
            targets: samples.iter().map(|s| vec![s.target]).collect(),
            timestamps: samples.iter().map(|s| s.timestamp).collect(),
            window_size,
            horizon,
            feature_dim: samples[0].features.len(),
            batch_size: samples.len(),
        };
        
        Ok(input)
    }
    
    /// Use TimeSeriesSample for data validation and quality checks
    pub fn validate_time_series_samples<T: Float>(
        samples: &[TimeSeriesSample<T>],
        schema: &TimeSeriesSchema,
    ) -> NeuroDivergentResult<HashMap<String, String>> {
        let mut validation_report = HashMap::new();
        
        // Check sample count
        validation_report.insert("total_samples".to_string(), samples.len().to_string());
        
        // Check feature consistency
        if let Some(first_sample) = samples.first() {
            let expected_features = first_sample.features.len();
            let consistent_features = samples.iter()
                .all(|s| s.features.len() == expected_features);
            
            validation_report.insert("consistent_features".to_string(), consistent_features.to_string());
            validation_report.insert("feature_count".to_string(), expected_features.to_string());
        }
        
        // Check temporal ordering
        let is_sorted = samples.windows(2)
            .all(|pair| pair[0].timestamp <= pair[1].timestamp);
        validation_report.insert("temporal_order".to_string(), is_sorted.to_string());
        
        // Check for missing values (simplified)
        let has_nan_targets = samples.iter()
            .any(|s| s.target.is_nan());
        validation_report.insert("has_nan_targets".to_string(), has_nan_targets.to_string());
        
        // Check date range
        if !samples.is_empty() {
            let start_date = samples.first().unwrap().timestamp;
            let end_date = samples.last().unwrap().timestamp;
            let duration = end_date - start_date;
            
            validation_report.insert("start_date".to_string(), start_date.to_rfc3339());
            validation_report.insert("end_date".to_string(), end_date.to_rfc3339());
            validation_report.insert("duration_days".to_string(), duration.num_days().to_string());
        }
        
        Ok(validation_report)
    }
    
    /// Use TimeSeriesInput for preprocessing and feature engineering
    pub fn preprocess_time_series_input<T: Float>(
        input: &mut TimeSeriesInput<T>,
        normalize: bool,
        fill_missing: bool,
    ) -> NeuroDivergentResult<HashMap<String, String>> {
        let mut preprocessing_log = HashMap::new();
        
        // Normalization
        if normalize {
            for sequence in &mut input.sequences {
                if !sequence.is_empty() {
                    let mean = sequence.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(sequence.len()).unwrap();
                    let variance = sequence.iter().fold(T::zero(), |acc, &x| acc + (x - mean) * (x - mean)) / T::from(sequence.len()).unwrap();
                    let std_dev = variance.sqrt();
                    
                    if std_dev > T::zero() {
                        for value in sequence {
                            *value = (*value - mean) / std_dev;
                        }
                    }
                }
            }
            preprocessing_log.insert("normalization".to_string(), "applied".to_string());
        }
        
        // Fill missing values (simplified)
        if fill_missing {
            let mut filled_count = 0;
            for sequence in &mut input.sequences {
                for value in sequence {
                    if value.is_nan() {
                        *value = T::zero(); // Simple fill with zero
                        filled_count += 1;
                    }
                }
            }
            preprocessing_log.insert("missing_values_filled".to_string(), filled_count.to_string());
        }
        
        preprocessing_log.insert("batch_size".to_string(), input.batch_size.to_string());
        preprocessing_log.insert("window_size".to_string(), input.window_size.to_string());
        preprocessing_log.insert("horizon".to_string(), input.horizon.to_string());
        preprocessing_log.insert("feature_dim".to_string(), input.feature_dim.to_string());
        
        Ok(preprocessing_log)
    }
    
    /// Use TimeSeriesSample for augmentation and synthetic data generation
    pub fn augment_time_series_samples<T: Float>(
        original_samples: &[TimeSeriesSample<T>],
        augmentation_factor: f64,
        noise_level: f64,
    ) -> NeuroDivergentResult<Vec<TimeSeriesSample<T>>> {
        let mut augmented_samples = original_samples.to_vec();
        let target_count = (original_samples.len() as f64 * augmentation_factor) as usize;
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        while augmented_samples.len() < target_count {
            if let Some(base_sample) = original_samples.get(rng.gen_range(0..original_samples.len())) {
                let mut new_sample = base_sample.clone();
                
                // Add noise to features
                for feature in &mut new_sample.features {
                    let noise = T::from(rng.gen_range(-noise_level..noise_level)).unwrap();
                    *feature = *feature + noise;
                }
                
                // Add noise to target
                let target_noise = T::from(rng.gen_range(-noise_level..noise_level)).unwrap();
                new_sample.target = new_sample.target + target_noise;
                
                // Update metadata
                new_sample.metadata.insert("augmented".to_string(), "true".to_string());
                new_sample.metadata.insert("noise_level".to_string(), noise_level.to_string());
                
                augmented_samples.push(new_sample);
            }
        }
        
        Ok(augmented_samples)
    }
}

/// Schema definition for time series data
#[derive(Debug, Clone)]
pub struct TimeSeriesSchema {
    pub unique_id_col: String,
    pub ds_col: String,           // Date/time column
    pub y_col: String,            // Target variable column
    pub static_features: Vec<String>,
    pub exogenous_features: Vec<String>,
}

impl TimeSeriesSchema {
    /// Create schema with required columns
    pub fn new(
        unique_id_col: impl Into<String>,
        ds_col: impl Into<String>,
        y_col: impl Into<String>
    ) -> Self {
        Self {
            unique_id_col: unique_id_col.into(),
            ds_col: ds_col.into(),
            y_col: y_col.into(),
            static_features: Vec::new(),
            exogenous_features: Vec::new(),
        }
    }
    
    /// Validate schema compatibility with time series samples
    pub fn validate_with_samples<T: Float>(
        &self,
        samples: &[TimeSeriesSample<T>],
    ) -> NeuroDivergentResult<()> {
        // Use TimeSeriesSample utilities for comprehensive validation
        let validation_report = time_series_utils::validate_time_series_samples(samples, self)?;
        
        // Check required columns are present in metadata
        for sample in samples.iter().take(5) { // Check first few samples
            if !sample.metadata.contains_key(&self.unique_id_col) {
                return Err(NeuroDivergentError::ValidationError {
                    message: format!("Missing unique_id_col '{}' in sample metadata", self.unique_id_col),
                    details: validation_report,
                });
            }
        }
        
        Ok(())
    }
    
    /// Add static features
    pub fn with_static_features(mut self, features: Vec<String>) -> Self {
        self.static_features = features;
        self
    }
    
    /// Add exogenous features
    pub fn with_exogenous_features(mut self, features: Vec<String>) -> Self {
        self.exogenous_features = features;
        self
    }
}

/// Main data structure for time series data
#[derive(Debug, Clone)]
pub struct TimeSeriesDataFrame<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    pub data: Vec<HashMap<String, DataValue<T>>>,
    pub schema: TimeSeriesSchema,
    phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> TimeSeriesDataFrame<T> {
    /// Create a new TimeSeriesDataFrame
    pub fn new(schema: TimeSeriesSchema) -> Self {
        Self {
            data: Vec::new(),
            schema,
            phantom: PhantomData,
        }
    }
    
    /// Convert to training dataset
    pub fn to_dataset(&self) -> NeuroDivergentResult<TimeSeriesDataset<T>> {
        // This is a placeholder implementation
        // In a full implementation, this would convert the dataframe to training samples
        Ok(TimeSeriesDataset {
            samples: Vec::new(),
            metadata: crate::foundation::DatasetMetadata {
                name: None,
                description: None,
                num_series: 0,
                total_samples: 0,
                feature_names: Vec::new(),
                target_names: Vec::new(),
            },
        })
    }
}

/// Results from forecasting operations
#[derive(Debug, Clone)]
pub struct ForecastDataFrame<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    pub data: Vec<HashMap<String, DataValue<T>>>,
    pub models: Vec<String>,
    pub forecast_horizon: usize,
    pub confidence_levels: Option<Vec<f64>>,
    phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> ForecastDataFrame<T> {
    /// Create a new ForecastDataFrame
    pub fn new(models: Vec<String>, forecast_horizon: usize) -> Self {
        Self {
            data: Vec::new(),
            models,
            forecast_horizon,
            confidence_levels: None,
            phantom: PhantomData,
        }
    }
}

/// Generic data value for DataFrame operations
#[derive(Debug, Clone)]
pub enum DataValue<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    Float(T),
    Int(i64),
    String(String),
    DateTime(DateTime<Utc>),
    Bool(bool),
    Null,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> DataValue<T> {
    pub fn as_float(&self) -> Option<T> {
        match self {
            DataValue::Float(f) => Some(*f),
            DataValue::Int(i) => T::from(*i),
            _ => None,
        }
    }
    
    pub fn as_string(&self) -> Option<&str> {
        match self {
            DataValue::String(s) => Some(s),
            _ => None,
        }
    }
    
    pub fn as_datetime(&self) -> Option<DateTime<Utc>> {
        match self {
            DataValue::DateTime(dt) => Some(*dt),
            _ => None,
        }
    }
}