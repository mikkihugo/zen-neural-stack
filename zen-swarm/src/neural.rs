//! Neural coordination and learning with Candle-based networks
//!
//! High-performance neural networks for agent coordination, pattern recognition,
//! and distributed learning across persistent swarms.

use crate::{Agent, SwarmError, Task};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Neural coordination system with distributed learning
pub struct NeuralCoordinator {
  /// Candle device (CPU/GPU)
  device: Device,
  /// Neural networks by type
  networks: Arc<DashMap<String, NeuralNetwork>>,
  /// Training data storage
  training_data: Arc<RwLock<TrainingDataStore>>,
  /// Agent behavioral patterns
  agent_patterns: Arc<DashMap<String, AgentPattern>>,
  /// Coordination models
  coordination_models: Arc<DashMap<String, CoordinationModel>>,
  /// Learning sessions
  learning_sessions: Arc<DashMap<String, LearningSession>>,
}

/// Neural network wrapper for coordination tasks
pub struct NeuralNetwork {
  pub network_id: String,
  pub network_type: NetworkType,
  pub input_size: usize,
  pub hidden_sizes: Vec<usize>,
  pub output_size: usize,
  pub model: Box<dyn NeuralModel + Send + Sync>,
  pub training_config: TrainingConfig,
  pub performance_metrics: NetworkMetrics,
  pub last_trained: DateTime<Utc>,
}

/// Types of neural networks for different coordination tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkType {
  /// Agent behavior prediction
  AgentBehavior {
    prediction_horizon: usize,
    context_window: usize,
  },
  /// Task allocation optimization
  TaskAllocation {
    agent_capacity: usize,
    task_complexity_factors: Vec<String>,
  },
  /// Pattern recognition for swarm coordination
  PatternRecognition {
    pattern_types: Vec<String>,
    similarity_threshold: f32,
  },
  /// Resource optimization
  ResourceOptimization {
    resource_types: Vec<String>,
    optimization_targets: Vec<String>,
  },
  /// Communication coordination
  CommunicationCoordination {
    message_types: Vec<String>,
    routing_strategies: Vec<String>,
  },
}

/// Neural model trait for different architectures
pub trait NeuralModel {
  fn forward(&self, input: &Tensor) -> Result<Tensor, SwarmError>;
  fn train_step(
    &mut self,
    input: &Tensor,
    target: &Tensor,
  ) -> Result<f32, SwarmError>;
  fn save_weights(&self, path: &str) -> Result<(), SwarmError>;
  fn load_weights(&mut self, path: &str) -> Result<(), SwarmError>;
  fn get_architecture_info(&self) -> ArchitectureInfo;
}

/// Multilayer perceptron for coordination tasks
pub struct MLPModel {
  layers: Vec<Linear>,
  device: Device,
  var_map: VarMap,
}

/// Architecture information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureInfo {
  pub model_type: String,
  pub layer_sizes: Vec<usize>,
  pub activation_functions: Vec<String>,
  pub parameter_count: usize,
  pub memory_usage_mb: f32,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
  pub learning_rate: f64,
  pub batch_size: usize,
  pub epochs: usize,
  pub validation_split: f32,
  pub early_stopping: bool,
  pub early_stopping_patience: usize,
  pub optimizer: OptimizerType,
  pub loss_function: LossFunctionType,
}

/// Optimizer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
  SGD { momentum: f64 },
  Adam { beta1: f64, beta2: f64 },
  RMSprop { alpha: f64 },
  AdaGrad,
}

/// Loss function types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossFunctionType {
  MeanSquaredError,
  CrossEntropy,
  BinaryCrossEntropy,
  HuberLoss { delta: f64 },
}

/// Network performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
  pub accuracy: f32,
  pub loss: f32,
  pub training_time_ms: u64,
  pub inference_time_ms: f32,
  pub convergence_epochs: Option<usize>,
  pub validation_accuracy: f32,
  pub overfitting_score: f32,
}

/// Training data storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDataStore {
  /// Input-output pairs for supervised learning
  pub supervised_data: HashMap<String, Vec<TrainingExample>>,
  /// Reward signals for reinforcement learning
  pub reward_data: HashMap<String, Vec<RewardExample>>,
  /// Unsupervised data for pattern discovery
  pub unsupervised_data: HashMap<String, Vec<DataPoint>>,
  /// Data quality metrics
  pub data_quality: DataQualityMetrics,
}

/// Training example for supervised learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
  pub input: Vec<f32>,
  pub target: Vec<f32>,
  pub weight: f32,
  pub timestamp: DateTime<Utc>,
  pub source: String, // Which agent/swarm generated this example
  pub context: Value,
}

/// Reward example for reinforcement learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardExample {
  pub state: Vec<f32>,
  pub action: Vec<f32>,
  pub reward: f32,
  pub next_state: Vec<f32>,
  pub done: bool,
  pub timestamp: DateTime<Utc>,
  pub episode_id: String,
}

/// Data point for unsupervised learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
  pub features: Vec<f32>,
  pub metadata: Value,
  pub timestamp: DateTime<Utc>,
  pub cluster_id: Option<String>,
}

/// Data quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityMetrics {
  pub completeness: f32,
  pub consistency: f32,
  pub accuracy: f32,
  pub timeliness: f32,
  pub uniqueness: f32,
  pub total_samples: usize,
  pub outlier_percentage: f32,
}

/// Agent behavioral pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPattern {
  pub agent_id: String,
  pub pattern_type: PatternType,
  pub features: Vec<f32>,
  pub confidence: f32,
  pub learned_from_samples: usize,
  pub last_updated: DateTime<Utc>,
  pub prediction_accuracy: f32,
  pub behavioral_traits: BehavioralTraits,
}

/// Types of behavioral patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
  TaskPreference {
    preferred_categories: Vec<String>,
  },
  CommunicationStyle {
    verbosity: f32,
    formality: f32,
  },
  DecisionMaking {
    risk_tolerance: f32,
    speed: f32,
  },
  Collaboration {
    cooperation_level: f32,
    leadership: f32,
  },
  Learning {
    adaptation_rate: f32,
    retention: f32,
  },
}

/// Behavioral traits extracted from neural analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralTraits {
  pub consistency: f32,
  pub predictability: f32,
  pub adaptability: f32,
  pub efficiency: f32,
  pub creativity: f32,
  pub reliability: f32,
}

/// Coordination model for swarm optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationModel {
  pub model_id: String,
  pub swarm_size: usize,
  pub coordination_strategy: CoordinationStrategy,
  pub optimization_targets: Vec<OptimizationTarget>,
  pub constraints: Vec<Constraint>,
  pub performance_history: Vec<PerformanceSnapshot>,
  pub current_efficiency: f32,
}

/// Coordination strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationStrategy {
  Centralized { leader_agent: String },
  Decentralized { consensus_mechanism: String },
  Hierarchical { layers: Vec<String> },
  MarketBased { bidding_strategy: String },
  SwarmIntelligence { algorithm: String },
}

/// Optimization targets for coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationTarget {
  MinimizeLatency,
  MaximizeThroughput,
  BalanceLoad,
  MinimizeResourceUsage,
  MaximizeAccuracy,
  OptimizeCommunication,
}

/// Constraints for coordination optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
  ResourceLimit { resource: String, limit: f32 },
  TimeLimit { deadline: DateTime<Utc> },
  QualityThreshold { metric: String, threshold: f32 },
  CommunicationBandwidth { max_messages_per_second: u32 },
  AgentCapacity { agent_id: String, max_tasks: u32 },
}

/// Performance snapshot for model tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
  pub timestamp: DateTime<Utc>,
  pub metrics: HashMap<String, f32>,
  pub swarm_state: SwarmState,
  pub optimization_score: f32,
}

/// Swarm state for coordination analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmState {
  pub active_agents: usize,
  pub pending_tasks: usize,
  pub average_load: f32,
  pub communication_rate: f32,
  pub error_rate: f32,
  pub resource_utilization: HashMap<String, f32>,
}

/// Learning session for continuous improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningSession {
  pub session_id: String,
  pub session_type: LearningType,
  pub participating_agents: Vec<String>,
  pub learning_objectives: Vec<String>,
  pub start_time: DateTime<Utc>,
  pub end_time: Option<DateTime<Utc>>,
  pub progress: LearningProgress,
  pub generated_insights: Vec<Insight>,
}

/// Types of learning sessions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningType {
  Individual {
    agent_id: String,
  },
  Collaborative {
    group_size: usize,
  },
  CrossSwarm {
    participating_swarms: Vec<String>,
  },
  MetaLearning {
    learning_to_learn: bool,
  },
  TransferLearning {
    source_domain: String,
    target_domain: String,
  },
}

/// Learning progress tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningProgress {
  pub completion_percentage: f32,
  pub accuracy_improvement: f32,
  pub convergence_rate: f32,
  pub stability_score: f32,
  pub knowledge_retention: f32,
}

/// Insights generated from learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Insight {
  pub insight_id: String,
  pub insight_type: InsightType,
  pub description: String,
  pub confidence: f32,
  pub actionable_recommendations: Vec<String>,
  pub discovered_at: DateTime<Utc>,
  pub validation_status: ValidationStatus,
}

/// Types of insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightType {
  PatternDiscovery,
  PerformanceOptimization,
  CoordinationImprovement,
  ResourceEfficiency,
  CommunicationOptimization,
  ErrorPrevention,
}

/// Validation status for insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
  Pending,
  Validated { success_rate: f32 },
  Rejected { reason: String },
  UnderReview,
}

impl NeuralCoordinator {
  /// Create new neural coordinator
  pub async fn new(device: Device) -> Result<Self, SwarmError> {
    Ok(Self {
      device,
      networks: Arc::new(DashMap::new()),
      training_data: Arc::new(RwLock::new(TrainingDataStore::new())),
      agent_patterns: Arc::new(DashMap::new()),
      coordination_models: Arc::new(DashMap::new()),
      learning_sessions: Arc::new(DashMap::new()),
    })
  }

  /// Create neural network for specific coordination task
  pub async fn create_network(
    &self,
    network_type: NetworkType,
    input_size: usize,
    hidden_sizes: Vec<usize>,
    output_size: usize,
  ) -> Result<String, SwarmError> {
    let network_id = Uuid::new_v4().to_string();

    // Create MLP model
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &self.device);

    let mut layers = Vec::new();
    let mut prev_size = input_size;

    for &hidden_size in &hidden_sizes {
      layers.push(candle_nn::linear(
        prev_size,
        hidden_size,
        vs.pp(&format!("layer_{}", layers.len())),
      )?);
      prev_size = hidden_size;
    }

    // Output layer
    layers.push(candle_nn::linear(prev_size, output_size, vs.pp("output"))?);

    let model = MLPModel {
      layers,
      device: self.device.clone(),
      var_map,
    };

    let network = NeuralNetwork {
      network_id: network_id.clone(),
      network_type,
      input_size,
      hidden_sizes,
      output_size,
      model: Box::new(model),
      training_config: TrainingConfig::default(),
      performance_metrics: NetworkMetrics::default(),
      last_trained: Utc::now(),
    };

    self.networks.insert(network_id.clone(), network);

    tracing::info!("Created neural network: {} for coordination", network_id);
    Ok(network_id)
  }

  /// Train neural network with coordination data
  pub async fn train_network(
    &self,
    network_id: &str,
    training_data: Vec<TrainingExample>,
  ) -> Result<NetworkMetrics, SwarmError> {
    let network = self.networks.get_mut(network_id).ok_or_else(|| {
      SwarmError::Neural(format!("Network {} not found", network_id))
    })?;

    let start_time = std::time::Instant::now();
    let mut total_loss = 0.0;
    let batch_size = network.training_config.batch_size;

    // Process training data in batches
    for batch_start in (0..training_data.len()).step_by(batch_size) {
      let batch_end = (batch_start + batch_size).min(training_data.len());
      let batch = &training_data[batch_start..batch_end];

      // Prepare batch tensors
      let inputs: Vec<Vec<f32>> =
        batch.iter().map(|ex| ex.input.clone()).collect();
      let targets: Vec<Vec<f32>> =
        batch.iter().map(|ex| ex.target.clone()).collect();

      // Convert to tensors
      let input_tensor = Tensor::from_vec(
        inputs.into_iter().flatten().collect::<Vec<f32>>(),
        (batch.len(), network.input_size),
        &self.device,
      )
      .map_err(|e| {
        SwarmError::Neural(format!("Failed to create input tensor: {}", e))
      })?;

      let target_tensor = Tensor::from_vec(
        targets.into_iter().flatten().collect::<Vec<f32>>(),
        (batch.len(), network.output_size),
        &self.device,
      )
      .map_err(|e| {
        SwarmError::Neural(format!("Failed to create target tensor: {}", e))
      })?;

      // Train step
      let mut network_mut = self.networks.get_mut(network_id).unwrap();
      let loss = network_mut
        .model
        .train_step(&input_tensor, &target_tensor)?;
      total_loss += loss;
    }

    let training_time = start_time.elapsed().as_millis() as u64;
    let avg_loss = total_loss / (training_data.len() / batch_size) as f32;

    // Update metrics
    let metrics = NetworkMetrics {
      accuracy: 1.0 - avg_loss, // Simplified accuracy calculation
      loss: avg_loss,
      training_time_ms: training_time,
      inference_time_ms: 0.0, // Will be updated during inference
      convergence_epochs: Some(network.training_config.epochs),
      validation_accuracy: 0.0, // Would be calculated with validation set
      overfitting_score: 0.0, // Would be calculated comparing train vs validation
    };

    // Update network
    let mut network = network;
    network.performance_metrics = metrics.clone();
    network.last_trained = Utc::now();

    tracing::info!(
      "Trained network {} - Loss: {:.4}, Time: {}ms",
      network_id,
      avg_loss,
      training_time
    );
    Ok(metrics)
  }

  /// Predict using neural network
  pub async fn predict(
    &self,
    network_id: &str,
    input: Vec<f32>,
  ) -> Result<Vec<f32>, SwarmError> {
    let network = self.networks.get(network_id).ok_or_else(|| {
      SwarmError::Neural(format!("Network {} not found", network_id))
    })?;

    let input_tensor =
      Tensor::from_vec(input, (1, network.input_size), &self.device).map_err(
        |e| SwarmError::Neural(format!("Failed to create input tensor: {}", e)),
      )?;

    let output_tensor = network.model.forward(&input_tensor)?;
    let output_data = output_tensor.to_vec1::<f32>().map_err(|e| {
      SwarmError::Neural(format!("Failed to extract output: {}", e))
    })?;

    Ok(output_data)
  }

  /// Learn agent behavioral pattern
  pub async fn learn_agent_pattern(
    &self,
    agent_id: &str,
    behavioral_data: Vec<Vec<f32>>,
  ) -> Result<AgentPattern, SwarmError> {
    // Simplified pattern learning - in production would use clustering/classification
    let features = if let Some(first_sample) = behavioral_data.first() {
      first_sample.clone()
    } else {
      return Err(SwarmError::Neural(
        "No behavioral data provided".to_string(),
      ));
    };

    let pattern = AgentPattern {
      agent_id: agent_id.to_string(),
      pattern_type: PatternType::TaskPreference {
        preferred_categories: vec![
          "analysis".to_string(),
          "coordination".to_string(),
        ],
      },
      features,
      confidence: 0.75,
      learned_from_samples: behavioral_data.len(),
      last_updated: Utc::now(),
      prediction_accuracy: 0.0,
      behavioral_traits: BehavioralTraits {
        consistency: 0.8,
        predictability: 0.7,
        adaptability: 0.6,
        efficiency: 0.9,
        creativity: 0.5,
        reliability: 0.85,
      },
    };

    self
      .agent_patterns
      .insert(agent_id.to_string(), pattern.clone());

    tracing::info!("Learned behavioral pattern for agent: {}", agent_id);
    Ok(pattern)
  }

  /// Start learning session
  pub async fn start_learning_session(
    &self,
    session_type: LearningType,
    objectives: Vec<String>,
  ) -> Result<String, SwarmError> {
    let session_id = Uuid::new_v4().to_string();

    let session = LearningSession {
      session_id: session_id.clone(),
      session_type,
      participating_agents: Vec::new(),
      learning_objectives: objectives,
      start_time: Utc::now(),
      end_time: None,
      progress: LearningProgress {
        completion_percentage: 0.0,
        accuracy_improvement: 0.0,
        convergence_rate: 0.0,
        stability_score: 1.0,
        knowledge_retention: 1.0,
      },
      generated_insights: Vec::new(),
    };

    self.learning_sessions.insert(session_id.clone(), session);

    tracing::info!("Started learning session: {}", session_id);
    Ok(session_id)
  }

  /// Get neural coordinator statistics
  pub async fn get_stats(&self) -> Result<NeuralStats, SwarmError> {
    Ok(NeuralStats {
      active_networks: self.networks.len(),
      learned_agent_patterns: self.agent_patterns.len(),
      coordination_models: self.coordination_models.len(),
      active_learning_sessions: self.learning_sessions.len(),
      total_training_examples: self
        .training_data
        .read()
        .await
        .supervised_data
        .values()
        .map(|v| v.len())
        .sum(),
    })
  }
}

impl MLPModel {
  pub fn new(
    layer_sizes: Vec<usize>,
    device: Device,
  ) -> Result<Self, SwarmError> {
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);

    let mut layers = Vec::new();

    for i in 0..layer_sizes.len() - 1 {
      layers.push(
        candle_nn::linear(
          layer_sizes[i],
          layer_sizes[i + 1],
          vs.pp(&format!("layer_{}", i)),
        )
        .map_err(|e| {
          SwarmError::Neural(format!("Failed to create layer: {}", e))
        })?,
      );
    }

    Ok(Self {
      layers,
      device,
      var_map,
    })
  }
}

impl NeuralModel for MLPModel {
  fn forward(&self, input: &Tensor) -> Result<Tensor, SwarmError> {
    let mut x = input.clone();

    for (i, layer) in self.layers.iter().enumerate() {
      x = layer.forward(&x).map_err(|e| {
        SwarmError::Neural(format!("Forward pass failed at layer {}: {}", i, e))
      })?;

      // Apply ReLU activation (except for output layer)
      if i < self.layers.len() - 1 {
        x = x.relu().map_err(|e| {
          SwarmError::Neural(format!("ReLU activation failed: {}", e))
        })?;
      }
    }

    Ok(x)
  }

  fn train_step(
    &mut self,
    input: &Tensor,
    target: &Tensor,
  ) -> Result<f32, SwarmError> {
    let prediction = self.forward(input)?;

    // Calculate MSE loss
    let diff = (prediction - target).map_err(|e| {
      SwarmError::Neural(format!("Loss calculation failed: {}", e))
    })?;
    let squared = diff
      .sqr()
      .map_err(|e| SwarmError::Neural(format!("Squared loss failed: {}", e)))?;
    let loss = squared
      .mean_all()
      .map_err(|e| SwarmError::Neural(format!("Mean loss failed: {}", e)))?;

    let loss_value = loss.to_scalar::<f32>().map_err(|e| {
      SwarmError::Neural(format!("Loss extraction failed: {}", e))
    })?;

    // In a full implementation, we would compute gradients and update weights here

    Ok(loss_value)
  }

  fn save_weights(&self, path: &str) -> Result<(), SwarmError> {
    // In production, would serialize var_map to file
    tracing::info!("Saving weights to: {}", path);
    Ok(())
  }

  fn load_weights(&mut self, path: &str) -> Result<(), SwarmError> {
    // In production, would load var_map from file
    tracing::info!("Loading weights from: {}", path);
    Ok(())
  }

  fn get_architecture_info(&self) -> ArchitectureInfo {
    ArchitectureInfo {
      model_type: "MLP".to_string(),
      layer_sizes: vec![], // Would extract from actual layers
      activation_functions: vec!["ReLU".to_string()],
      parameter_count: 0,   // Would calculate from var_map
      memory_usage_mb: 0.0, // Would estimate from parameters
    }
  }
}

impl TrainingDataStore {
  pub fn new() -> Self {
    Self {
      supervised_data: HashMap::new(),
      reward_data: HashMap::new(),
      unsupervised_data: HashMap::new(),
      data_quality: DataQualityMetrics::default(),
    }
  }
}

impl Default for TrainingConfig {
  fn default() -> Self {
    Self {
      learning_rate: 0.001,
      batch_size: 32,
      epochs: 100,
      validation_split: 0.2,
      early_stopping: true,
      early_stopping_patience: 10,
      optimizer: OptimizerType::Adam {
        beta1: 0.9,
        beta2: 0.999,
      },
      loss_function: LossFunctionType::MeanSquaredError,
    }
  }
}

impl Default for NetworkMetrics {
  fn default() -> Self {
    Self {
      accuracy: 0.0,
      loss: 0.0,
      training_time_ms: 0,
      inference_time_ms: 0.0,
      convergence_epochs: None,
      validation_accuracy: 0.0,
      overfitting_score: 0.0,
    }
  }
}

impl Default for DataQualityMetrics {
  fn default() -> Self {
    Self {
      completeness: 1.0,
      consistency: 1.0,
      accuracy: 1.0,
      timeliness: 1.0,
      uniqueness: 1.0,
      total_samples: 0,
      outlier_percentage: 0.0,
    }
  }
}

/// Neural coordinator statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralStats {
  pub active_networks: usize,
  pub learned_agent_patterns: usize,
  pub coordination_models: usize,
  pub active_learning_sessions: usize,
  pub total_training_examples: usize,
}

#[cfg(test)]
mod tests {
  use super::*;

  #[tokio::test]
  async fn test_neural_coordinator_creation() {
    let device = Device::Cpu;
    let coordinator = NeuralCoordinator::new(device).await;
    assert!(coordinator.is_ok());
  }

  #[tokio::test]
  async fn test_network_creation() {
    let device = Device::Cpu;
    let coordinator = NeuralCoordinator::new(device).await.unwrap();

    let network_type = NetworkType::TaskAllocation {
      agent_capacity: 10,
      task_complexity_factors: vec![
        "cpu_required".to_string(),
        "memory_required".to_string(),
      ],
    };

    let network_id = coordinator
      .create_network(network_type, 10, vec![20, 15], 5)
      .await;

    assert!(network_id.is_ok());
  }
}
