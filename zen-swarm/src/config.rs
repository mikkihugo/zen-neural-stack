//! Configuration for zen-swarm

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main swarm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
  /// Maximum number of agents in the swarm
  pub max_agents: usize,
  /// Default coordination strategy
  pub strategy: CoordinationStrategy,
  /// Storage configuration
  pub storage: StorageConfig,
  /// Network configuration
  pub network: NetworkConfig,

  #[cfg(feature = "vector")]
  /// Vector database configuration
  pub vector: Option<VectorConfig>,

  #[cfg(feature = "runtime")]
  /// Node.js runtime configuration
  pub runtime: Option<RuntimeConfig>,

  #[cfg(feature = "graph")]
  /// Graph database configuration  
  pub graph: Option<GraphConfig>,

  #[cfg(feature = "neural")]
  /// Neural network configuration
  pub neural: Option<NeuralConfig>,
}

/// Coordination strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationStrategy {
  Parallel,
  Sequential,
  Adaptive,
  Hierarchical,
}

/// Storage configuration with .zen directory structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
  /// Base .zen directory (shared across all AI systems)
  pub zen_base_dir: PathBuf,
  /// Swarm-specific data directory (.zen/swarm)
  pub swarm_data_dir: PathBuf,
  /// Claude Code specific directory (.zen/claude)
  pub claude_data_dir: PathBuf,
  /// Gemini specific directory (.zen/gemini)
  pub gemini_data_dir: PathBuf,
  /// Central collective directory (.zen/collective)
  pub collective_data_dir: PathBuf,
  pub max_size_gb: u64,
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
  pub host: String,
  pub port: u16,
  pub max_connections: usize,
}

#[cfg(feature = "vector")]
/// Vector database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorConfig {
  pub storage_path: PathBuf,
  pub embedding_model: String,
  pub dimension: usize,
}

#[cfg(feature = "runtime")]
/// Node.js runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
  pub temp_dir: Option<PathBuf>,
  pub max_execution_time_seconds: u64,
  pub max_memory_mb: u64,
}

#[cfg(feature = "graph")]
/// Graph database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
  pub database_path: PathBuf,
  pub max_nodes: usize,
}

#[cfg(feature = "neural")]
/// Neural network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
  pub model_path: PathBuf,
  pub device: String, // "cpu", "cuda", "metal"
}

impl Default for SwarmConfig {
  fn default() -> Self {
    Self {
      max_agents: 100,
      strategy: CoordinationStrategy::Adaptive,
      storage: StorageConfig {
        zen_base_dir: "./.zen".into(),
        swarm_data_dir: "./.zen/swarm".into(),
        claude_data_dir: "./.zen/claude".into(),
        gemini_data_dir: "./.zen/gemini".into(),
        collective_data_dir: "./.zen/collective".into(),
        max_size_gb: 10,
      },
      network: NetworkConfig {
        host: "127.0.0.1".to_string(),
        port: 3000,
        max_connections: 1000,
      },

      #[cfg(feature = "vector")]
      vector: Some(VectorConfig {
        storage_path: "./.zen/swarm/vector_data".into(),
        embedding_model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        dimension: 384,
      }),

      #[cfg(feature = "runtime")]
      runtime: Some(RuntimeConfig {
        temp_dir: None,
        max_execution_time_seconds: 60,
        max_memory_mb: 512,
      }),

      #[cfg(feature = "graph")]
      graph: Some(GraphConfig {
        database_path: "./.zen/swarm/graph_data".into(),
        max_nodes: 1_000_000,
      }),

      #[cfg(feature = "neural")]
      neural: Some(NeuralConfig {
        model_path: "./.zen/swarm/models".into(),
        device: "cpu".to_string(),
      }),
    }
  }
}

impl SwarmConfig {
  /// Builder pattern for vector database
  #[cfg(feature = "vector")]
  pub fn with_vector_db(mut self, path: &str) -> Self {
    self.vector = Some(VectorConfig {
      storage_path: path.into(),
      embedding_model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
      dimension: 384,
    });
    self
  }

  /// Builder pattern for graph database
  #[cfg(feature = "graph")]
  pub fn with_graph_db(mut self, path: &str) -> Self {
    self.graph = Some(GraphConfig {
      database_path: path.into(),
      max_nodes: 1_000_000,
    });
    self
  }

  /// Builder pattern for Node.js runtime
  #[cfg(feature = "runtime")]
  pub fn with_node_runtime(mut self) -> Self {
    self.runtime = Some(RuntimeConfig {
      temp_dir: None,
      max_execution_time_seconds: 60,
      max_memory_mb: 512,
    });
    self
  }
}
