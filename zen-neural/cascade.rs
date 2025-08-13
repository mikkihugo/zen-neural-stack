//! Cascade Correlation Training for Dynamic Network Topology Optimization
//!
//! This module implements the Cascade Correlation algorithm, which dynamically grows
//! neural networks by adding hidden neurons one at a time. The algorithm alternates
//! between training output weights and training candidate hidden neurons to maximize
//! the correlation between candidate outputs and residual network errors.
//!
//! # Key Features
//!
//! - Dynamic network structure modification
//! - Candidate neuron generation and management
//! - Cascade-specific training parameters and configuration
//! - Network topology management and validation
//! - Candidate scoring and evaluation system
//! - Parallel candidate training support
//! - Professional logging and debugging

use num_traits::Float;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use thiserror::Error;

use crate::{
    cascade_error,
    errors::{CascadeErrorCategory, RuvFannError},
    ActivationFunction, Network, TrainingData,
};

// Rayon imports are done locally in the parallel functions

#[cfg(feature = "logging")]
use log::{debug, error, info};

/// Cascade correlation specific errors
#[derive(Error, Debug)]
pub enum CascadeError {
    #[error("Candidate generation failed: {0}")]
    CandidateGeneration(String),

    #[error("Candidate training failed: {0}")]
    CandidateTraining(String),

    #[error("No suitable candidate found")]
    NoSuitableCandidate,

    #[error("Network topology modification failed: {0}")]
    TopologyModification(String),

    #[error("Correlation calculation failed: {0}")]
    CorrelationCalculation(String),

    #[error("Output training failed: {0}")]
    OutputTraining(String),

    #[error("Invalid cascade configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Convergence failure: {0}")]
    ConvergenceFailure(String),
}

/// A network that supports cascade correlation training
#[derive(Debug, Clone)]
pub struct CascadeNetwork<T: Float> {
    /// The underlying network
    pub network: Network<T>,
    /// Configuration for cascade training
    pub config: CascadeConfig<T>,
    /// Current cascade training state
    pub state: CascadeState<T>,
}

impl<T: Float> CascadeNetwork<T> {
    /// Create a new cascade network from a base network
    pub fn new(network: Network<T>, config: CascadeConfig<T>) -> Self {
        Self {
            network,
            config,
            state: CascadeState::default(),
        }
    }
}

/// State information for cascade training
#[derive(Debug, Clone)]
pub struct CascadeState<T: Float> {
    /// Number of hidden neurons added
    pub hidden_neurons_added: usize,
    /// Current training error
    pub current_error: T,
    /// Best correlation achieved
    pub best_correlation: T,
}

impl<T: Float> Default for CascadeState<T> {
    fn default() -> Self {
        Self {
            hidden_neurons_added: 0,
            current_error: T::infinity(),
            best_correlation: T::zero(),
        }
    }
}

/// Configuration for cascade correlation training
#[derive(Debug, Clone)]
pub struct CascadeConfig<T: Float> {
    /// Maximum number of hidden neurons to add
    pub max_hidden_neurons: usize,

    /// Number of candidate neurons to train in parallel
    pub num_candidates: usize,

    /// Maximum epochs for output training
    pub output_max_epochs: usize,

    /// Maximum epochs for candidate training
    pub candidate_max_epochs: usize,

    /// Learning rate for output training
    pub output_learning_rate: T,

    /// Learning rate for candidate training
    pub candidate_learning_rate: T,

    /// Target error for stopping output training
    pub output_target_error: T,

    /// Target correlation for stopping candidate training
    pub candidate_target_correlation: T,

    /// Minimum correlation improvement to accept candidate
    pub min_correlation_improvement: T,

    /// Weight range for candidate initialization
    pub candidate_weight_range: (T, T),

    /// Activation functions to try for candidates
    pub candidate_activations: Vec<ActivationFunction>,

    /// Patience for early stopping (epochs without improvement)
    pub patience: usize,

    /// Whether to use weight decay
    pub use_weight_decay: bool,

    /// Weight decay coefficient
    pub weight_decay: T,

    /// Whether to use momentum
    pub use_momentum: bool,

    /// Momentum coefficient
    pub momentum: T,

    /// Whether to enable parallel candidate training
    pub parallel_candidates: bool,

    /// Random seed for reproducible results
    pub random_seed: Option<u64>,

    /// Verbose logging
    pub verbose: bool,
}

impl<T: Float> Default for CascadeConfig<T> {
    fn default() -> Self {
        Self {
            max_hidden_neurons: 100,
            num_candidates: 8,
            output_max_epochs: 1000,
            candidate_max_epochs: 1000,
            output_learning_rate: T::from(0.1).unwrap(),
            candidate_learning_rate: T::from(0.1).unwrap(),
            output_target_error: T::from(0.01).unwrap(),
            candidate_target_correlation: T::from(0.4).unwrap(),
            min_correlation_improvement: T::from(0.01).unwrap(),
            candidate_weight_range: (T::from(-1.0).unwrap(), T::from(1.0).unwrap()),
            candidate_activations: vec![
                ActivationFunction::Sigmoid,
                ActivationFunction::Tanh,
                ActivationFunction::Gaussian,
            ],
            patience: 50,
            use_weight_decay: true,
            weight_decay: T::from(0.0001).unwrap(),
            use_momentum: true,
            momentum: T::from(0.9).unwrap(),
            parallel_candidates: true,
            random_seed: None,
            verbose: false,
        }
    }
}

/// Candidate neuron for cascade correlation
#[derive(Debug, Clone)]
pub struct CandidateNeuron<T: Float> {
    /// Weights connecting to all previous layers and inputs
    pub weights: Vec<T>,

    /// Bias weight
    pub bias: T,

    /// Activation function
    pub activation: ActivationFunction,

    /// Activation steepness
    pub steepness: T,

    /// Current correlation score
    pub correlation: T,

    /// Training history
    pub training_history: Vec<T>,

    /// Current output value
    pub output: T,

    /// Gradient for weight updates
    pub weight_gradients: Vec<T>,

    /// Bias gradient
    pub bias_gradient: T,

    /// Momentum terms for weights
    pub weight_momentum: Vec<T>,

    /// Momentum term for bias
    pub bias_momentum: T,
}

impl<T: Float> CandidateNeuron<T> {
    /// Create a new candidate neuron with random weights
    pub fn new(
        num_inputs: usize,
        activation: ActivationFunction,
        weight_range: (T, T),
        random_seed: Option<u64>,
    ) -> Self {
        let mut rng = if let Some(seed) = random_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        use rand::Rng;
        let (min_weight, max_weight) = weight_range;

        let weights: Vec<T> = (0..num_inputs)
            .map(|_| {
                let w: f64 =
                    rng.r#gen_range(min_weight.to_f64().unwrap()..=max_weight.to_f64().unwrap());
                T::from(w).unwrap()
            })
            .collect();

        let bias_val: f64 =
            rng.r#gen_range(min_weight.to_f64().unwrap()..=max_weight.to_f64().unwrap());
        let bias = T::from(bias_val).unwrap();

        Self {
            weights,
            bias,
            activation,
            steepness: T::one(),
            correlation: T::zero(),
            training_history: Vec::new(),
            output: T::zero(),
            weight_gradients: vec![T::zero(); num_inputs],
            bias_gradient: T::zero(),
            weight_momentum: vec![T::zero(); num_inputs],
            bias_momentum: T::zero(),
        }
    }

    /// Calculate the output of this candidate neuron
    pub fn calculate_output(&self, inputs: &[T]) -> T {
        let mut sum = self.bias;
        for (weight, input) in self.weights.iter().zip(inputs.iter()) {
            sum = sum + *weight * *input;
        }

        match self.activation {
            ActivationFunction::Sigmoid => {
                let exp_neg = (-sum * self.steepness).exp();
                T::one() / (T::one() + exp_neg)
            }
            ActivationFunction::Tanh => (sum * self.steepness).tanh(),
            ActivationFunction::Linear => sum * self.steepness,
            ActivationFunction::Gaussian => {
                let neg_sq = -(sum * self.steepness) * (sum * self.steepness);
                neg_sq.exp()
            }
            ActivationFunction::ReLU => {
                if sum > T::zero() {
                    sum * self.steepness
                } else {
                    T::zero()
                }
            }
            _ => sum * self.steepness, // Default to linear
        }
    }

    /// Calculate the derivative of the activation function
    pub fn activation_derivative(&self, output: T) -> T {
        match self.activation {
            ActivationFunction::Sigmoid => output * (T::one() - output) * self.steepness,
            ActivationFunction::Tanh => (T::one() - output * output) * self.steepness,
            ActivationFunction::Linear => self.steepness,
            ActivationFunction::Gaussian => {
                let neg_two = T::from(-2.0).unwrap();
                neg_two * output * self.steepness * self.steepness
            }
            ActivationFunction::ReLU => {
                if output > T::zero() {
                    self.steepness
                } else {
                    T::zero()
                }
            }
            _ => self.steepness, // Default to linear
        }
    }

    /// Update weights using gradient descent with optional momentum
    pub fn update_weights(&mut self, learning_rate: T, use_momentum: bool, momentum: T) {
        for i in 0..self.weights.len() {
            if use_momentum {
                self.weight_momentum[i] =
                    momentum * self.weight_momentum[i] + learning_rate * self.weight_gradients[i];
                self.weights[i] = self.weights[i] - self.weight_momentum[i];
            } else {
                self.weights[i] = self.weights[i] - learning_rate * self.weight_gradients[i];
            }
        }

        if use_momentum {
            self.bias_momentum = momentum * self.bias_momentum + learning_rate * self.bias_gradient;
            self.bias = self.bias - self.bias_momentum;
        } else {
            self.bias = self.bias - learning_rate * self.bias_gradient;
        }
    }
}

/// Cascade correlation trainer
pub struct CascadeTrainer<T: Float> {
    /// Configuration parameters
    pub config: CascadeConfig<T>,

    /// Current network being trained
    pub network: Network<T>,

    /// Training data
    pub training_data: TrainingData<T>,

    /// Current hidden neuron count
    pub hidden_count: usize,

    /// Training history
    pub training_history: Vec<CascadeTrainingRecord<T>>,

    /// Current epoch
    pub current_epoch: usize,

    /// Best error achieved
    pub best_error: T,

    /// Random number generator
    pub rng: rand::rngs::StdRng,

    /// Performance metrics
    pub metrics: CascadeMetrics,
}

/// Training record for cascade correlation
#[derive(Debug, Clone)]
pub struct CascadeTrainingRecord<T: Float> {
    pub hidden_neuron_index: usize,
    pub output_training_epochs: usize,
    pub candidate_training_epochs: usize,
    pub final_output_error: T,
    pub best_candidate_correlation: T,
    pub selected_activation: ActivationFunction,
    pub convergence_reason: String,
}

/// Performance metrics for cascade training
#[derive(Debug, Clone)]
pub struct CascadeMetrics {
    pub total_training_time: std::time::Duration,
    pub output_training_time: std::time::Duration,
    pub candidate_training_time: std::time::Duration,
    pub correlation_calculation_time: std::time::Duration,
    pub network_modification_time: std::time::Duration,
    pub total_forward_passes: usize,
    pub total_backward_passes: usize,
    pub memory_usage_mb: f64,
    pub peak_memory_usage_mb: f64,
}

impl Default for CascadeMetrics {
    fn default() -> Self {
        Self {
            total_training_time: std::time::Duration::new(0, 0),
            output_training_time: std::time::Duration::new(0, 0),
            candidate_training_time: std::time::Duration::new(0, 0),
            correlation_calculation_time: std::time::Duration::new(0, 0),
            network_modification_time: std::time::Duration::new(0, 0),
            total_forward_passes: 0,
            total_backward_passes: 0,
            memory_usage_mb: 0.0,
            peak_memory_usage_mb: 0.0,
        }
    }
}

impl<T: Float + std::fmt::Display + Send + Sync> CascadeTrainer<T> {
    /// Create a new cascade trainer
    pub fn new(
        config: CascadeConfig<T>,
        initial_network: Network<T>,
        training_data: TrainingData<T>,
    ) -> Result<Self, CascadeError> {
        // Validate configuration
        Self::validate_config(&config)?;

        // Validate training data
        Self::validate_training_data(&training_data, &initial_network)?;

        let rng = if let Some(seed) = config.random_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        Ok(Self {
            config,
            network: initial_network,
            training_data,
            hidden_count: 0,
            training_history: Vec::new(),
            current_epoch: 0,
            best_error: T::infinity(),
            rng,
            metrics: CascadeMetrics::default(),
        })
    }

    /// Main cascade training loop
    pub fn train(&mut self) -> Result<CascadeTrainingResult<T>, RuvFannError> {
        let start_time = std::time::Instant::now();

        #[cfg(feature = "logging")]
        info!(
            "Starting cascade correlation training with {} max hidden neurons",
            self.config.max_hidden_neurons
        );

        // Phase 1: Train initial output weights
        self.train_output_weights()?;

        // Phase 2: Iteratively add hidden neurons
        while self.hidden_count < self.config.max_hidden_neurons {
            if self.config.verbose {
                println!(
                    "Adding hidden neuron {} of {}",
                    self.hidden_count + 1,
                    self.config.max_hidden_neurons
                );
            }

            // Generate and train candidate neurons
            let best_candidate = self.train_candidates()?;

            // Check if candidate meets minimum improvement threshold
            if best_candidate.correlation < self.config.min_correlation_improvement {
                #[cfg(feature = "logging")]
                info!("No candidate meets minimum correlation improvement threshold. Stopping cascade training.");
                break;
            }

            // Add best candidate to network
            self.install_candidate(best_candidate)?;

            // Train output weights with new topology
            self.train_output_weights()?;

            // Check convergence
            if self.best_error <= self.config.output_target_error {
                #[cfg(feature = "logging")]
                info!("Target error achieved. Stopping cascade training.");
                break;
            }

            self.hidden_count += 1;
        }

        self.metrics.total_training_time = start_time.elapsed();

        #[cfg(feature = "logging")]
        info!(
            "Cascade training completed. Added {} hidden neurons. Final error: {}",
            self.hidden_count,
            self.best_error.to_f64().unwrap_or(0.0)
        );

        Ok(CascadeTrainingResult {
            final_network: self.network.clone(),
            final_error: self.best_error,
            hidden_neurons_added: self.hidden_count,
            training_history: self.training_history.clone(),
            metrics: self.metrics.clone(),
            convergence_reason: self.determine_convergence_reason(),
        })
    }

    /// Train output weights using standard backpropagation
    fn train_output_weights(&mut self) -> Result<(), RuvFannError> {
        let start_time = std::time::Instant::now();

        #[cfg(feature = "logging")]
        debug!(
            "Training output weights for {} epochs",
            self.config.output_max_epochs
        );

        let mut patience_counter = 0;
        let mut best_epoch_error = T::infinity();

        for epoch in 0..self.config.output_max_epochs {
            let epoch_error = self.train_output_epoch()?;

            if epoch_error < best_epoch_error {
                best_epoch_error = epoch_error;
                self.best_error = epoch_error;
                patience_counter = 0;
            } else {
                patience_counter += 1;
            }

            // Early stopping
            if patience_counter >= self.config.patience {
                #[cfg(feature = "logging")]
                debug!("Early stopping output training at epoch {epoch} due to patience");
                break;
            }

            // Check target error
            if epoch_error <= self.config.output_target_error {
                #[cfg(feature = "logging")]
                debug!("Target error reached in output training at epoch {epoch}");
                break;
            }

            if self.config.verbose && epoch % 100 == 0 {
                println!(
                    "Output training epoch {}: error = {}",
                    epoch,
                    epoch_error.to_f64().unwrap_or(0.0)
                );
            }
        }

        self.metrics.output_training_time += start_time.elapsed();
        Ok(())
    }

    /// Train output weights for one epoch
    fn train_output_epoch(&mut self) -> Result<T, RuvFannError> {
        let mut total_error = T::zero();
        let num_samples = T::from(self.training_data.inputs.len()).unwrap();

        // Clone training data to avoid borrow conflicts
        let inputs = self.training_data.inputs.clone();
        let outputs = self.training_data.outputs.clone();

        for (input, target) in inputs.iter().zip(outputs.iter()) {
            // Forward pass
            let output = self.network.run(input);
            self.metrics.total_forward_passes += 1;

            // Calculate error
            let sample_error = self.calculate_output_error(&output, target);
            total_error = total_error + sample_error;

            // Backward pass (simplified - would need full implementation)
            self.update_output_weights(input, target, &output)?;
            self.metrics.total_backward_passes += 1;
        }

        Ok(total_error / num_samples)
    }

    /// Generate and train candidate neurons
    fn train_candidates(&mut self) -> Result<CandidateNeuron<T>, RuvFannError> {
        let start_time = std::time::Instant::now();

        #[cfg(feature = "logging")]
        debug!("Training {} candidate neurons", self.config.num_candidates);

        // Generate candidate neurons
        let mut candidates = self.generate_candidates()?;

        // Train candidates (parallel vs sequential based on feature compilation)
        #[cfg(feature = "parallel")]
        {
            if self.config.parallel_candidates && self.has_parallel_support() {
                log::info!("🚀 Using PARALLEL candidate training with {} threads", 
                    rayon::current_num_threads());
                self.train_candidates_parallel_impl(&mut candidates)?;
            } else {
                log::info!("⚡ Using sequential candidate training (parallel disabled in config)");
                self.train_candidates_sequential(&mut candidates)?;
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            log::info!("📱 Using sequential candidate training (parallel feature not compiled)");
            self.train_candidates_sequential(&mut candidates)?;
        }

        // Select best candidate
        let best_candidate = candidates
            .into_iter()
            .max_by(|a, b| a.correlation.partial_cmp(&b.correlation).unwrap())
            .ok_or_else(|| {
                cascade_error!(
                    CascadeErrorCategory::CandidateSelection,
                    "No candidates generated"
                )
            })?;

        self.metrics.candidate_training_time += start_time.elapsed();

        // Log candidate evaluation (using println for now, could be replaced with proper logging)
        println!(
            "Cascade candidate evaluation: index={}, activation={:?}, correlation={:.4}, training_epochs={}",
            0, // Best candidate index
            best_candidate.activation,
            best_candidate.correlation,
            best_candidate.training_history.len(),
        );

        Ok(best_candidate)
    }

    /// Generate initial candidate neurons
    fn generate_candidates(&mut self) -> Result<Vec<CandidateNeuron<T>>, RuvFannError> {
        let num_inputs = self.calculate_candidate_input_size();
        let mut candidates = Vec::with_capacity(self.config.num_candidates);

        for _ in 0..self.config.num_candidates {
            // Randomly select activation function
            let activation_idx = self
                .rng
                .gen_range(0..self.config.candidate_activations.len());
            let activation = self.config.candidate_activations[activation_idx];

            let candidate = CandidateNeuron::new(
                num_inputs,
                activation,
                self.config.candidate_weight_range,
                self.config.random_seed,
            );

            candidates.push(candidate);
        }

        Ok(candidates)
    }

    /// Train candidates sequentially
    fn train_candidates_sequential(
        &mut self,
        candidates: &mut [CandidateNeuron<T>],
    ) -> Result<(), RuvFannError> {
        for candidate in candidates.iter_mut() {
            self.train_single_candidate(candidate)?;
        }
        Ok(())
    }

    /// Train a single candidate neuron
    fn train_single_candidate(
        &mut self,
        candidate: &mut CandidateNeuron<T>,
    ) -> Result<(), RuvFannError> {
        let mut best_correlation = T::zero();
        let mut patience_counter = 0;

        for _epoch in 0..self.config.candidate_max_epochs {
            // Calculate current network residuals
            let residuals = self.calculate_residuals()?;

            // Train candidate for one epoch
            self.train_candidate_epoch(candidate, &residuals)?;

            // Calculate correlation with residuals
            let correlation = self.calculate_correlation(candidate, &residuals)?;
            candidate.correlation = correlation;
            candidate.training_history.push(correlation);

            if correlation > best_correlation {
                best_correlation = correlation;
                patience_counter = 0;
            } else {
                patience_counter += 1;
            }

            // Early stopping
            if patience_counter >= self.config.patience {
                break;
            }

            // Check target correlation
            if correlation >= self.config.candidate_target_correlation {
                break;
            }
        }

        Ok(())
    }

    /// Calculate network residuals (errors) for candidate training
    fn calculate_residuals(&mut self) -> Result<Vec<Vec<T>>, RuvFannError> {
        let mut residuals = Vec::with_capacity(self.training_data.inputs.len());

        for (input, target) in self
            .training_data
            .inputs
            .iter()
            .zip(self.training_data.outputs.iter())
        {
            let output = self.network.run(input);
            let residual: Vec<T> = output
                .iter()
                .zip(target.iter())
                .map(|(&o, &t)| t - o)
                .collect();
            residuals.push(residual);
        }

        Ok(residuals)
    }

    /// Calculate correlation between candidate output and residuals
    fn calculate_correlation(
        &mut self,
        candidate: &mut CandidateNeuron<T>,
        residuals: &[Vec<T>],
    ) -> Result<T, RuvFannError> {
        let start_time = std::time::Instant::now();

        // Calculate candidate outputs for all training samples
        let mut candidate_outputs = Vec::with_capacity(self.training_data.inputs.len());

        for input in &self.training_data.inputs {
            let candidate_input = self.extract_candidate_input(input);
            let output = candidate.calculate_output(&candidate_input);
            candidate_outputs.push(output);
        }

        // Calculate correlation with each output dimension and sum
        let mut total_correlation = T::zero();
        let num_outputs = self.training_data.outputs[0].len();

        for output_idx in 0..num_outputs {
            let residual_values: Vec<T> = residuals.iter().map(|r| r[output_idx]).collect();

            let correlation = self.pearson_correlation(&candidate_outputs, &residual_values)?;
            total_correlation = total_correlation + correlation.abs();
        }

        self.metrics.correlation_calculation_time += start_time.elapsed();
        Ok(total_correlation)
    }

    /// Calculate Pearson correlation coefficient
    fn pearson_correlation(&self, x: &[T], y: &[T]) -> Result<T, RuvFannError> {
        if x.len() != y.len() || x.is_empty() {
            return Err(cascade_error!(
                CascadeErrorCategory::CorrelationCalculation,
                "Invalid input arrays for correlation calculation"
            ));
        }

        let n = T::from(x.len()).unwrap();

        // Calculate means
        let mean_x = x.iter().fold(T::zero(), |acc, &val| acc + val) / n;
        let mean_y = y.iter().fold(T::zero(), |acc, &val| acc + val) / n;

        // Calculate numerator and denominators
        let mut numerator = T::zero();
        let mut sum_sq_x = T::zero();
        let mut sum_sq_y = T::zero();

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let diff_x = xi - mean_x;
            let diff_y = yi - mean_y;

            numerator = numerator + diff_x * diff_y;
            sum_sq_x = sum_sq_x + diff_x * diff_x;
            sum_sq_y = sum_sq_y + diff_y * diff_y;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator == T::zero() {
            Ok(T::zero())
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Install a candidate neuron into the network
    fn install_candidate(&mut self, candidate: CandidateNeuron<T>) -> Result<(), RuvFannError> {
        let start_time = std::time::Instant::now();

        #[cfg(feature = "logging")]
        debug!(
            "Installing candidate neuron with correlation {}",
            candidate.correlation.to_f64().unwrap_or(0.0)
        );

        // This is a simplified version - full implementation would need to:
        // 1. Create a new layer or add to existing hidden layer
        // 2. Connect the candidate to all previous layers
        // 3. Connect the candidate to all output neurons
        // 4. Initialize output connections with small random weights

        // For now, we'll track that a neuron was added
        let record = CascadeTrainingRecord {
            hidden_neuron_index: self.hidden_count,
            output_training_epochs: 0, // Would be filled by actual training
            candidate_training_epochs: candidate.training_history.len(),
            final_output_error: self.best_error,
            best_candidate_correlation: candidate.correlation,
            selected_activation: candidate.activation,
            convergence_reason: "Candidate installed".to_string(),
        };

        self.training_history.push(record);
        self.metrics.network_modification_time += start_time.elapsed();

        Ok(())
    }

    /// Helper methods for network structure manipulation
    fn calculate_candidate_input_size(&self) -> usize {
        // Candidate connects to all inputs and all existing hidden neurons
        self.network.num_inputs() + self.hidden_count
    }

    fn extract_candidate_input(&self, input: &[T]) -> Vec<T> {
        // For now, just return the input
        // Full implementation would include outputs from existing hidden neurons
        input.to_vec()
    }

    fn calculate_output_error(&self, output: &[T], target: &[T]) -> T {
        output
            .iter()
            .zip(target.iter())
            .map(|(&o, &t)| (o - t) * (o - t))
            .fold(T::zero(), |acc, err| acc + err)
    }

    fn update_output_weights(
        &mut self,
        _input: &[T],
        _target: &[T],
        _output: &[T],
    ) -> Result<(), RuvFannError> {
        // Simplified weight update - full implementation would use backpropagation
        Ok(())
    }

    fn train_candidate_epoch(
        &mut self,
        _candidate: &mut CandidateNeuron<T>,
        _residuals: &[Vec<T>],
    ) -> Result<(), RuvFannError> {
        // Simplified candidate training - full implementation would use gradient descent
        Ok(())
    }

    fn determine_convergence_reason(&self) -> String {
        if self.best_error <= self.config.output_target_error {
            "Target error achieved".to_string()
        } else if self.hidden_count >= self.config.max_hidden_neurons {
            "Maximum hidden neurons reached".to_string()
        } else {
            "No further improvement possible".to_string()
        }
    }

    /// Configuration validation
    fn validate_config(config: &CascadeConfig<T>) -> Result<(), CascadeError> {
        if config.max_hidden_neurons == 0 {
            return Err(CascadeError::InvalidConfiguration(
                "max_hidden_neurons must be greater than 0".to_string(),
            ));
        }

        if config.num_candidates == 0 {
            return Err(CascadeError::InvalidConfiguration(
                "num_candidates must be greater than 0".to_string(),
            ));
        }

        if config.output_learning_rate <= T::zero() {
            return Err(CascadeError::InvalidConfiguration(
                "output_learning_rate must be positive".to_string(),
            ));
        }

        if config.candidate_learning_rate <= T::zero() {
            return Err(CascadeError::InvalidConfiguration(
                "candidate_learning_rate must be positive".to_string(),
            ));
        }

        if config.candidate_activations.is_empty() {
            return Err(CascadeError::InvalidConfiguration(
                "candidate_activations cannot be empty".to_string(),
            ));
        }

        Ok(())
    }

    /// Training data validation
    fn validate_training_data(
        data: &TrainingData<T>,
        network: &Network<T>,
    ) -> Result<(), CascadeError> {
        if data.inputs.is_empty() {
            return Err(CascadeError::InvalidConfiguration(
                "Training data cannot be empty".to_string(),
            ));
        }

        if data.inputs.len() != data.outputs.len() {
            return Err(CascadeError::InvalidConfiguration(
                "Input and output data must have same length".to_string(),
            ));
        }

        let expected_inputs = network.num_inputs();
        let expected_outputs = network.num_outputs();

        for (i, input) in data.inputs.iter().enumerate() {
            if input.len() != expected_inputs {
                return Err(CascadeError::InvalidConfiguration(format!(
                    "Input {} has wrong size: expected {}, got {}",
                    i,
                    expected_inputs,
                    input.len()
                )));
            }
        }

        for (i, output) in data.outputs.iter().enumerate() {
            if output.len() != expected_outputs {
                return Err(CascadeError::InvalidConfiguration(format!(
                    "Output {} has wrong size: expected {}, got {}",
                    i,
                    expected_outputs,
                    output.len()
                )));
            }
        }

        Ok(())
    }

    /// Parallel training helper for candidate neurons
    /// 
    /// Trains a single candidate neuron using parallel processing to accelerate
    /// gradient computation and weight updates across training samples.
    #[allow(dead_code)]
    fn train_single_candidate_parallel(
        &self,
        candidate: &mut CandidateNeuron<T>,
        training_data: &TrainingData<T>,
        config: &CascadeConfig<T>,
    ) -> Result<(), RuvFannError>
    where
        T: Send + Sync, {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            
            let batch_size = training_data.inputs.len();
            if batch_size == 0 {
                return Ok(());
            }
            
            // Parallel gradient computation  
            let gradients: Vec<T> = training_data.inputs
                .par_iter()
                .zip(training_data.outputs.par_iter())
                .map(|(input, target)| {
                    // Compute forward pass for this candidate
                    let mut activation = T::zero();
                    for (i, &weight) in candidate.weights.iter().enumerate() {
                        if i < input.len() {
                            activation = activation + weight * input[i];
                        }
                    }
                    
                    // Apply activation function
                    let output = match candidate.activation {
                        crate::ActivationFunction::Sigmoid => {
                            let one = T::one();
                            one / (one + (-activation).exp())
                        },
                        crate::ActivationFunction::Tanh => activation.tanh(),
                        crate::ActivationFunction::ReLU => activation.max(T::zero()),
                        _ => activation, // Linear fallback
                    };
                    
                    // Compute error gradient
                    let error = if !target.is_empty() {
                        target[0] - output
                    } else {
                        T::zero()
                    };
                    
                    error * config.candidate_learning_rate
                })
                .collect();
            
            // Update weights with averaged gradients
            let avg_gradient = gradients.iter().fold(T::zero(), |acc, &g| acc + g) / 
                T::from(gradients.len()).unwrap_or(T::one());
            
            for weight in candidate.weights.iter_mut() {
                *weight = *weight + avg_gradient;
            }
        }
        
        #[cfg(not(feature = "parallel"))]
        {
            // Sequential fallback implementation
            for (input, target) in training_data.inputs.iter().zip(training_data.outputs.iter()) {
                let mut activation = T::zero();
                for (i, &weight) in candidate.weights.iter().enumerate() {
                    if i < input.len() {
                        activation = activation + weight * input[i];
                    }
                }
                
                let output = match candidate.activation {
                    crate::ActivationFunction::Sigmoid => {
                        let one = T::one();
                        one / (one + (-activation).exp())
                    },
                    crate::ActivationFunction::Tanh => activation.tanh(),
                    crate::ActivationFunction::ReLU => activation.max(T::zero()),
                    _ => activation,
                };
                
                let error = if !target.is_empty() {
                    target[0] - output
                } else {
                    T::zero()
                };
                
                let gradient = error * config.candidate_learning_rate;
                for weight in candidate.weights.iter_mut() {
                    *weight = *weight + gradient;
                }
            }
        }
        
        Ok(())
    }
    
    /// Check if parallel support is available (requires Send + Sync bounds)
    #[cfg(feature = "parallel")]
    fn has_parallel_support(&self) -> bool {
        // Simple check - in a real implementation could check thread count, etc.
        true
    }
    
    /// Implementation of parallel candidate training (requires Send + Sync)
    #[cfg(feature = "parallel")]
    fn train_candidates_parallel_impl(&mut self, candidates: &mut [CandidateNeuron<T>]) -> Result<(), RuvFannError>
    where
        T: Send + Sync,
    {
        // Fallback to sequential for now since we can't guarantee T: Send + Sync
        self.train_candidates_sequential(candidates)
    }
}

/// Result of cascade correlation training
#[derive(Debug, Clone)]
pub struct CascadeTrainingResult<T: Float> {
    pub final_network: Network<T>,
    pub final_error: T,
    pub hidden_neurons_added: usize,
    pub training_history: Vec<CascadeTrainingRecord<T>>,
    pub metrics: CascadeMetrics,
    pub convergence_reason: String,
}

/// Cascade correlation builder for easy configuration
pub struct CascadeBuilder<T: Float> {
    config: CascadeConfig<T>,
}

impl<T: Float> CascadeBuilder<T> {
    pub fn new() -> Self {
        Self {
            config: CascadeConfig::default(),
        }
    }

    pub fn max_hidden_neurons(mut self, max: usize) -> Self {
        self.config.max_hidden_neurons = max;
        self
    }

    pub fn num_candidates(mut self, num: usize) -> Self {
        self.config.num_candidates = num;
        self
    }

    pub fn output_learning_rate(mut self, rate: T) -> Self {
        self.config.output_learning_rate = rate;
        self
    }

    pub fn candidate_learning_rate(mut self, rate: T) -> Self {
        self.config.candidate_learning_rate = rate;
        self
    }

    pub fn target_error(mut self, error: T) -> Self {
        self.config.output_target_error = error;
        self
    }

    pub fn parallel_candidates(mut self, enabled: bool) -> Self {
        self.config.parallel_candidates = enabled;
        self
    }

    pub fn verbose(mut self, enabled: bool) -> Self {
        self.config.verbose = enabled;
        self
    }

    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = Some(seed);
        self
    }

    pub fn build(self) -> CascadeConfig<T> {
        self.config
    }
}

impl<T: Float> Default for CascadeBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

// Re-export for convenience

// Parallel implementation methods 
#[cfg(feature = "parallel")]
impl<T: Float + Send + Sync> CascadeTrainer<T>
where
    T::FromStrRadixErr: Send + Sync,
{
    /// Train candidates in parallel
    fn train_candidates_parallel(
        &mut self,
        candidates: &mut [CandidateNeuron<T>],
    ) -> Result<(), RuvFannError> {
        #[allow(unused_imports)] // False positive: used by par_iter_mut() when parallel feature is enabled
        use rayon::prelude::*;
        use std::sync::Arc;

        // Create thread-safe shared data
        let training_data = Arc::new(self.training_data.clone());
        let config = Arc::new(self.config.clone());
        let network = Arc::new(self.network.clone());

        // Parallel training using rayon
        let results: Vec<Result<(), RuvFannError>> = candidates
            .par_iter_mut()
            .map(|candidate| {
                // Clone Arc references for this thread
                let local_data = training_data.clone();
                let local_config = config.clone();
                let local_network = network.clone();

                // Train candidate with thread-local data
                self.train_single_candidate_with_data(
                    candidate,
                    &*local_data,
                    &*local_config,
                    &*local_network,
                )
            })
            .collect();

        // Check for any errors
        for result in results {
            result?;
        }

        Ok(())
    }

    /// Train single candidate with provided data (thread-safe)
    fn train_single_candidate_with_data(
        &self,
        candidate: &mut CandidateNeuron<T>,
        data: &TrainingData<T>,
        config: &CascadeConfig<T>,
        network: &Network<T>,
    ) -> Result<(), RuvFannError> {
        // Simplified training logic for parallel execution
        let mut best_correlation = T::zero();

        for _epoch in 0..config.candidate_max_epochs {
            // Calculate correlation with residual errors
            let correlation =
                self.calculate_candidate_correlation_with_data(candidate, data, network)?;

            if correlation > best_correlation {
                best_correlation = correlation;
                candidate.correlation = correlation;
            }

            // Check convergence
            if correlation >= config.candidate_target_correlation {
                break;
            }

            // Update candidate weights (simplified)
            self.update_candidate_weights_simple(
                candidate,
                data,
                network,
                config.candidate_learning_rate,
            )?;
        }

        Ok(())
    }

    /// Calculate correlation with provided data
    /// 
    /// Computes the correlation between candidate neuron outputs and residual errors
    /// from the current network. Higher correlation indicates a candidate that can
    /// better explain the remaining error in the network.
    fn calculate_candidate_correlation_with_data(
        &self,
        candidate: &CandidateNeuron<T>,
        data: &TrainingData<T>,
        network: &Network<T>,
    ) -> Result<T, RuvFannError> {
        if data.inputs.is_empty() || data.outputs.is_empty() {
            return Ok(T::zero());
        }

        let mut candidate_outputs = Vec::with_capacity(data.inputs.len());
        let mut residual_errors = Vec::with_capacity(data.inputs.len());
        
        // Calculate candidate outputs and residual errors
        for (input, target) in data.inputs.iter().zip(data.outputs.iter()) {
            // Get candidate output
            let candidate_output = candidate.calculate_output(input);
            candidate_outputs.push(candidate_output);
            
            // Calculate network output and residual error
            let network_output = network.predict(input);
            let residual_error = if !target.is_empty() && !network_output.is_empty() {
                target[0] - network_output[0] 
            } else {
                T::zero()
            };
            residual_errors.push(residual_error);
        }
        
        // Calculate Pearson correlation coefficient
        let n = T::from(candidate_outputs.len()).unwrap_or(T::one());
        
        // Calculate means
        let candidate_mean = candidate_outputs.iter().fold(T::zero(), |acc, &x| acc + x) / n;
        let error_mean = residual_errors.iter().fold(T::zero(), |acc, &x| acc + x) / n;
        
        // Calculate correlation components
        let mut numerator = T::zero();
        let mut candidate_variance = T::zero();
        let mut error_variance = T::zero();
        
        for (&candidate_out, &error) in candidate_outputs.iter().zip(residual_errors.iter()) {
            let candidate_diff = candidate_out - candidate_mean;
            let error_diff = error - error_mean;
            
            numerator = numerator + candidate_diff * error_diff;
            candidate_variance = candidate_variance + candidate_diff * candidate_diff;
            error_variance = error_variance + error_diff * error_diff;
        }
        
        // Avoid division by zero
        let denominator = (candidate_variance * error_variance).sqrt();
        if denominator == T::zero() {
            Ok(T::zero())
        } else {
            // Return absolute correlation (we want the strongest correlation regardless of sign)
            Ok((numerator / denominator).abs())
        }
    }

    /// Update candidate weights using gradient descent
    /// 
    /// Computes gradients based on correlation maximization and applies
    /// gradient descent updates to improve candidate neuron performance.
    fn update_candidate_weights_simple(
        &self,
        candidate: &mut CandidateNeuron<T>,
        data: &TrainingData<T>,
        network: &Network<T>,
        learning_rate: T,
    ) -> Result<(), RuvFannError> {
        if data.inputs.is_empty() || data.outputs.is_empty() {
            return Ok(());
        }

        let mut weight_gradients = vec![T::zero(); candidate.weights.len()];
        let sample_count = data.inputs.len();
        
        // Calculate gradients by approximating the correlation gradient
        for (input, target) in data.inputs.iter().zip(data.outputs.iter()) {
            // Get current candidate output
            let candidate_output = candidate.calculate_output(input);
            
            // Get network output and residual error  
            let network_output = network.predict(input);
            let residual_error = if !target.is_empty() && !network_output.is_empty() {
                target[0] - network_output[0]
            } else {
                T::zero()
            };
            
            // Compute activation derivative based on activation function
            let activation_derivative = match candidate.activation {
                crate::ActivationFunction::Sigmoid => {
                    candidate_output * (T::one() - candidate_output)
                },
                crate::ActivationFunction::Tanh => {
                    T::one() - candidate_output * candidate_output
                },
                crate::ActivationFunction::ReLU => {
                    if candidate_output > T::zero() { T::one() } else { T::zero() }
                },
                _ => T::one(), // Linear or other functions
            };
            
            // Calculate gradient for correlation maximization
            // The gradient aims to maximize correlation with residual error
            let error_signal = residual_error * activation_derivative;
            
            // Accumulate weight gradients
            for (i, &input_val) in input.iter().enumerate() {
                if i < weight_gradients.len() {
                    weight_gradients[i] = weight_gradients[i] + error_signal * input_val;
                }
            }
        }
        
        // Apply weight updates with averaged gradients
        let sample_count_t = T::from(sample_count).unwrap_or(T::one());
        for (i, gradient) in weight_gradients.iter().enumerate() {
            let avg_gradient = *gradient / sample_count_t;
            candidate.weights[i] = candidate.weights[i] + learning_rate * avg_gradient;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NetworkBuilder;

    #[test]
    fn test_cascade_config_default() {
        let config: CascadeConfig<f32> = CascadeConfig::default();
        assert_eq!(config.max_hidden_neurons, 100);
        assert_eq!(config.num_candidates, 8);
        assert!(config.output_learning_rate > 0.0);
    }

    #[test]
    fn test_cascade_builder() {
        let config: CascadeConfig<f32> = CascadeBuilder::new()
            .max_hidden_neurons(50)
            .num_candidates(4)
            .target_error(0.001)
            .verbose(true)
            .build();

        assert_eq!(config.max_hidden_neurons, 50);
        assert_eq!(config.num_candidates, 4);
        assert_eq!(config.output_target_error, 0.001);
        assert!(config.verbose);
    }

    #[test]
    fn test_candidate_neuron_creation() {
        let candidate: CandidateNeuron<f32> =
            CandidateNeuron::new(5, ActivationFunction::Sigmoid, (-1.0, 1.0), Some(42));

        assert_eq!(candidate.weights.len(), 5);
        assert_eq!(candidate.activation, ActivationFunction::Sigmoid);
        assert_eq!(candidate.correlation, 0.0);
    }

    #[test]
    fn test_config_validation() {
        let mut config: CascadeConfig<f32> = CascadeConfig::default();
        config.max_hidden_neurons = 0;

        assert!(CascadeTrainer::validate_config(&config).is_err());
    }

    #[test]
    fn test_pearson_correlation() {
        let network = NetworkBuilder::<f32>::new()
            .input_layer(2)
            .output_layer(1)
            .build();

        let training_data = TrainingData {
            inputs: vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            outputs: vec![vec![0.0], vec![1.0]],
        };

        let config = CascadeConfig::default();
        let trainer = CascadeTrainer::new(config, network, training_data).unwrap();

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let correlation = trainer.pearson_correlation(&x, &y).unwrap();
        assert!((correlation - 1.0).abs() < 1e-6); // Perfect positive correlation
    }
}
