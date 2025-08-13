//! # zen-swarm
//!
//! Unified high-performance swarm intelligence system combining neural networks,
//! vector databases, AI coordination, and distributed computing capabilities.
//!
//! This crate consolidates multiple specialized components into a single,
//! high-performance package optimized for AI agent coordination and swarm intelligence.
//!
//! ## Features
//!
//! - **🧠 Neural Networks**: Candle-based neural coordination and learning
//! - **🔍 Vector Database**: LanceDB v0.20 with semantic search and RAG
//! - **📊 Graph Analysis**: Kuzu graph database for complex relationship analysis  
//! - **🚀 Node.js Runtime**: External Node.js v24+ processes for AI plugins
//! - **💾 Persistence**: LibSQL storage with ACID guarantees
//! - **🌐 MCP Protocol**: Model Context Protocol server integration
//! - **📈 Performance**: 1M+ operations/second with SIMD acceleration
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    zen-swarm                            │
//! ├─────────────────────────────────────────────────────────┤
//! │  🧠 Neural Coord   🔍 Vector DB   📊 Graph Analysis    │
//! │  🚀 Node.js v24+   💾 Persistence  🌐 MCP Server       │
//! ├─────────────────────────────────────────────────────────┤
//! │              High-Performance Rust Core                │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use zen_swarm::{Swarm, SwarmConfig, Agent};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Initialize swarm with all capabilities
//!     let config = SwarmConfig::default()
//!         .with_vector_db("./vector_data")
//!         .with_graph_db("./graph_data")
//!         .with_node_runtime();
//!     
//!     let swarm = Swarm::new(config).await?;
//!     
//!     // Spawn AI coordination agents
//!     let researcher = swarm.spawn_agent("researcher", Agent::researcher()).await?;
//!     let analyst = swarm.spawn_agent("analyst", Agent::analyst()).await?;
//!     let coordinator = swarm.spawn_agent("coordinator", Agent::coordinator()).await?;
//!     
//!     // Execute coordinated AI tasks
//!     let results = swarm.orchestrate_task(
//!         "Analyze complex system architecture and provide recommendations",
//!         vec![researcher, analyst, coordinator]
//!     ).await?;
//!     
//!     println!("🎯 Swarm Results: {:#?}", results);
//!     Ok(())
//! }
//! ```

// Core modules - always available
pub mod agent;
pub mod config;
pub mod core;
pub mod error;
pub mod task;

// Feature-gated modules
#[cfg(feature = "neural")]
pub mod neural;

#[cfg(feature = "vector")]
pub mod vector;

#[cfg(feature = "runtime")]
pub mod runtime;

#[cfg(feature = "persistence")]
pub mod persistence;

#[cfg(feature = "graph")]
pub mod graph;

// Unified database system combining Cozo + LanceDB
#[cfg(all(feature = "graph", feature = "vector"))]
pub mod unified_database;

// Type-secured event logging system
pub mod events;

// Comprehensive type safety system
pub mod types;

#[cfg(feature = "mcp")]
pub mod mcp;

#[cfg(feature = "mcp")]
pub mod openapi;

// Daemon system for repository-scoped coordination
pub mod daemon;

// A2A (Agent-to-Agent) coordination system  
pub mod a2a;

// Additional production modules
pub mod lifecycle;
pub mod plugins;

// Type aliases for convenience
pub type SwarmResult<T> = Result<T, error::SwarmError>;

// Re-exports for easy access - specific exports to avoid conflicts
pub use agent::{Agent, AgentType};
pub use config::{NetworkConfig, StorageConfig, SwarmConfig};
pub use core::{Swarm, SwarmCommand, SwarmStats};
pub use error::SwarmError;
pub use task::{Task, TaskResult, TaskStatus};

#[cfg(feature = "neural")]
pub use neural::{NetworkType, NeuralCoordinator, NeuralNetwork};

#[cfg(feature = "vector")]
pub use vector::{VectorDb, VectorQuery, VectorResult};

#[cfg(feature = "runtime")]
pub use runtime::{NodeProcess, NodeRuntime};

#[cfg(feature = "persistence")]
pub use persistence::{
  BackupInfo as PersistenceBackupInfo,
  BackupManager as PersistenceBackupManager,
  BackupType as PersistenceBackupType, PersistenceManager,
};

#[cfg(feature = "graph")]
pub use graph::{
  PersistentAgent, PersistentSwarm, PersistentTask, SwarmServiceType,
};

#[cfg(feature = "mcp")]
pub use mcp::{McpResource, McpServer, McpTool};

// A2A client for zen-orchestrator communication
pub mod a2a_client;

// Daemon and A2A system re-exports
pub use daemon::{SwarmDaemon, SwarmDaemonConfig, SwarmDaemonType, SwarmDaemonCli};
pub use a2a::{A2ACoordinator, A2AMessage, A2AChannelType, A2AConfig};
pub use a2a_client::{A2AClient, A2AClientConfig, ConnectionStatus};

// Production module re-exports - specific to avoid conflicts
pub use lifecycle::{
  BackupInfo as LifecycleBackupInfo, BackupManager as LifecycleBackupManager,
  BackupType as LifecycleBackupType, CleanupPolicy,
  ResourceRequirements as LifecycleResourceRequirements,
  RetentionPolicy as LifecycleRetentionPolicy, SwarmLifecycleManager,
  SwarmLifecycleState, ZenDirectoryConfig,
};
pub use plugins::{
  Plugin, PluginManager, PluginType, QualityGate,
  ResourceRequirements as PluginResourceRequirements,
  RetentionPolicy as PluginRetentionPolicy, SwarmPluginConfig,
};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build information
pub const BUILD_INFO: &str = concat!(
  "zen-swarm v",
  env!("CARGO_PKG_VERSION"),
  " (development built)"
);

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_version() {
    assert!(!VERSION.is_empty());
    println!("zen-swarm version: {}", VERSION);
  }

  #[tokio::test]
  async fn test_swarm_creation() {
    let config = SwarmConfig::default();
    let swarm = Swarm::new(config).await;
    assert!(swarm.is_ok(), "Failed to create swarm: {:?}", swarm.err());
  }
}
