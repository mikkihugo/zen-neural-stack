//! Pure Rust Cozo-based persistence layer
//!
//! Replacement for LibSQL using pure Rust Cozo database.
//! No FFI dependencies, faster compilation, unified database system.

#[cfg(feature = "graph")]
use crate::PersistentSwarm;
use crate::{SwarmError, SwarmResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Pure Rust persistence manager using Cozo
#[derive(Debug)]
pub struct CozoPersistenceManager {
  /// Cozo database instance
  #[cfg(feature = "persistence")]
  db: Arc<RwLock<cozo::DbInstance>>,
  /// Database configuration
  config: CozoPersistenceConfig,
}

/// Cozo persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CozoPersistenceConfig {
  /// Database file path
  pub db_path: String,
  /// Enable WAL mode
  pub enable_wal: bool,
}

/// Persistence statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct CozoPersistenceStats {
  pub total_swarms: usize,
  pub total_agents: usize,
  pub total_tasks: usize,
  pub total_memories: usize,
  pub database_size_mb: f64,
}

impl CozoPersistenceManager {
  /// Create new persistence manager with pure Rust Cozo
  #[cfg(feature = "persistence")]
  pub async fn new(config: CozoPersistenceConfig) -> SwarmResult<Self> {
    let db =
      cozo::DbInstance::new("sqlite", &config.db_path, "").map_err(|e| {
        SwarmError::Persistence(format!(
          "Failed to create Cozo database: {}",
          e
        ))
      })?;

    let manager = Self {
      db: Arc::new(RwLock::new(db)),
      config,
    };

    // Initialize schema
    manager.initialize_schema().await?;

    tracing::info!(
      "Pure Rust Cozo persistence manager initialized: {}",
      config.db_path
    );
    Ok(manager)
  }

  /// Initialize database schema using CozoScript
  #[cfg(feature = "persistence")]
  async fn initialize_schema(&self) -> SwarmResult<()> {
    let db = self.db.read().await;

    // Create swarm states table
    let swarm_schema = r#"
            {
                :create swarm_states {
                    swarm_id: String,
                    domain: String,
                    service_type: String,
                    creation_time: String,
                    last_activity: String,
                    state_data: String,
                    version: Int,
                    =>
                    swarm_id
                }
            }
        "#;

    db.run_script(
      swarm_schema,
      Default::default(),
      cozo::ScriptMutability::Mutable,
    )
    .map_err(|e| {
      SwarmError::Persistence(format!(
        "Failed to create swarm_states table: {}",
        e
      ))
    })?;

    // Create agent states table
    let agent_schema = r#"
            {
                :create agent_states {
                    agent_id: String,
                    swarm_id: String,
                    agent_type: String,
                    current_state: String,
                    memory_data: String,
                    performance_metrics: String,
                    created_at: String,
                    last_updated: String,
                    =>
                    agent_id
                }
            }
        "#;

    db.run_script(
      agent_schema,
      Default::default(),
      cozo::ScriptMutability::Mutable,
    )
    .map_err(|e| {
      SwarmError::Persistence(format!(
        "Failed to create agent_states table: {}",
        e
      ))
    })?;

    // Create tasks table
    let task_schema = r#"
            {
                :create tasks {
                    task_id: String,
                    swarm_id: String,
                    description: String,
                    status: String,
                    priority: String,
                    assigned_agents: String,
                    created_at: String,
                    updated_at: String,
                    metadata: String,
                    =>
                    task_id
                }
            }
        "#;

    db.run_script(
      task_schema,
      Default::default(),
      cozo::ScriptMutability::Mutable,
    )
    .map_err(|e| {
      SwarmError::Persistence(format!("Failed to create tasks table: {}", e))
    })?;

    // Create memories table
    let memory_schema = r#"
            {
                :create memories {
                    memory_id: String,
                    agent_id: String,
                    swarm_id: String,
                    memory_key: String,
                    memory_value: String,
                    memory_type: String,
                    created_at: String,
                    last_accessed: String,
                    access_count: Int,
                    =>
                    memory_id
                }
            }
        "#;

    db.run_script(
      memory_schema,
      Default::default(),
      cozo::ScriptMutability::Mutable,
    )
    .map_err(|e| {
      SwarmError::Persistence(format!("Failed to create memories table: {}", e))
    })?;

    tracing::info!("Cozo database schema initialized successfully");
    Ok(())
  }

  /// Save swarm state using CozoScript
  #[cfg(all(feature = "persistence", feature = "graph"))]
  pub async fn save_swarm_state(
    &self,
    swarm: &PersistentSwarm,
  ) -> SwarmResult<()> {
    let db = self.db.read().await;

    let state_json = serde_json::to_string(swarm).map_err(|e| {
      SwarmError::Persistence(format!("Failed to serialize swarm state: {}", e))
    })?;

    let query = format!(
      r#"
            {{
                ?[swarm_id, domain, service_type, creation_time, last_activity, state_data, version] <- 
                [['{}', '{}', '{}', '{}', '{}', '{}', 1]]
                
                :put swarm_states {{swarm_id, domain, service_type, creation_time, last_activity, state_data, version}}
            }}
        "#,
      swarm.swarm_id,
      swarm.domain,
      serde_json::to_string(&swarm.service_type).unwrap_or_default(),
      swarm.creation_time.to_rfc3339(),
      swarm.last_activity.to_rfc3339(),
      state_json
    );

    db.run_script(&query, Default::default(), cozo::ScriptMutability::Mutable)
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to save swarm state: {}", e))
      })?;

    tracing::debug!("Saved swarm state for: {}", swarm.swarm_id);
    Ok(())
  }

  /// Load swarm state using CozoScript
  #[cfg(all(feature = "persistence", feature = "graph"))]
  pub async fn load_swarm_state(
    &self,
    swarm_id: &str,
  ) -> SwarmResult<Option<PersistentSwarm>> {
    let db = self.db.read().await;

    let query = format!(
      r#"
            {{
                ?[state_data] := *swarm_states[swarm_id, domain, service_type, creation_time, last_activity, state_data, version],
                                swarm_id == '{}'
            }}
        "#,
      swarm_id
    );

    let result = db
      .run_script(
        &query,
        Default::default(),
        cozo::ScriptMutability::Immutable,
      )
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to query swarm state: {}", e))
      })?;

    if let Some(rows) = result.rows.first() {
      if let Some(cozo::DataValue::Str(state_json)) = rows.first() {
        let swarm: PersistentSwarm =
          serde_json::from_str(state_json).map_err(|e| {
            SwarmError::Persistence(format!(
              "Failed to deserialize swarm state: {}",
              e
            ))
          })?;

        tracing::debug!("Loaded swarm state for: {}", swarm_id);
        return Ok(Some(swarm));
      }
    }

    Ok(None)
  }

  /// Get persistence statistics
  #[cfg(feature = "persistence")]
  pub async fn get_stats(&self) -> SwarmResult<CozoPersistenceStats> {
    let db = self.db.read().await;

    // Count records using CozoScript
    let swarm_count_query = r#"
            {
                ?[count] := count(swarm_id), *swarm_states[swarm_id]
            }
        "#;

    let result = db
      .run_script(
        swarm_count_query,
        Default::default(),
        cozo::ScriptMutability::Immutable,
      )
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to count swarms: {}", e))
      })?;

    let swarm_count = result
      .rows
      .first()
      .and_then(|row| row.first())
      .and_then(|val| match val {
        cozo::DataValue::Int(n) => Some(*n as usize),
        _ => None,
      })
      .unwrap_or(0);

    Ok(CozoPersistenceStats {
      total_swarms: swarm_count,
      total_agents: 0, // Would implement similar queries
      total_tasks: 0,
      total_memories: 0,
      database_size_mb: 0.0,
    })
  }
}

impl Default for CozoPersistenceConfig {
  fn default() -> Self {
    Self {
      db_path: ".zen/swarm/cozo.db".to_string(),
      enable_wal: true,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::TempDir;

  #[tokio::test]
  #[cfg(feature = "persistence")]
  async fn test_cozo_persistence_manager_creation() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test_cozo.db");

    let mut config = CozoPersistenceConfig::default();
    config.db_path = db_path.to_string_lossy().to_string();

    let manager = CozoPersistenceManager::new(config).await;
    assert!(manager.is_ok());
  }
}
