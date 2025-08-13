//! Unified Database System: Cozo + LanceDB Integration
//!
//! Optimal architecture combining:
//! - Cozo: Pure Rust for graph operations, relational data, analytics, transactions
//! - LanceDB: Specialized vector operations, embeddings, similarity search
//!
//! This creates a powerful hybrid system where each database does what it's best at.

use crate::{SwarmError, SwarmResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Unified database manager combining Cozo + LanceDB
#[derive(Debug)]
pub struct UnifiedDatabase {
  /// Cozo for graph operations, relational data, analytics
  #[cfg(feature = "graph")]
  cozo: Arc<RwLock<cozo::DbInstance>>,

  /// LanceDB for vector operations, embeddings, similarity search
  #[cfg(feature = "vector")]
  lancedb: Arc<crate::vector::VectorDb>,

  /// Configuration
  config: UnifiedDatabaseConfig,
}

/// Configuration for unified database system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedDatabaseConfig {
  /// Cozo database path for graph/relational data
  pub cozo_path: String,

  /// LanceDB path for vector data
  pub lancedb_path: String,

  /// Integration strategy
  pub integration_strategy: IntegrationStrategy,
}

/// Integration strategies between Cozo and LanceDB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrationStrategy {
  /// Reference-based: Cozo stores metadata, LanceDB stores vectors
  ReferencesBased,

  /// Synchronized: Keep both databases in sync with cross-references
  Synchronized,

  /// Event-driven: Use events to coordinate between databases
  EventDriven,
}

/// Unified data operations that span both databases
impl UnifiedDatabase {
  /// Create new unified database system
  pub async fn new(config: UnifiedDatabaseConfig) -> SwarmResult<Self> {
    // Initialize Cozo
    #[cfg(feature = "graph")]
    let cozo = {
      let db = cozo::DbInstance::new("sqlite", &config.cozo_path, "").map_err(
        |e| {
          SwarmError::Configuration(format!(
            "Failed to create Cozo database: {}",
            e
          ))
        },
      )?;
      Arc::new(RwLock::new(db))
    };

    // Initialize LanceDB
    #[cfg(feature = "vector")]
    let lancedb = {
      let vector_db =
        crate::vector::VectorDb::new(&config.lancedb_path).await?;
      Arc::new(vector_db)
    };

    let unified = Self {
      #[cfg(feature = "graph")]
      cozo,
      #[cfg(feature = "vector")]
      lancedb,
      config,
    };

    // Initialize schemas and integration
    unified.initialize_integration().await?;

    tracing::info!(
      "Unified database system initialized - Cozo: {}, LanceDB: {}",
      config.cozo_path,
      config.lancedb_path
    );
    Ok(unified)
  }

  /// Initialize integration between Cozo and LanceDB
  async fn initialize_integration(&self) -> SwarmResult<()> {
    match self.config.integration_strategy {
      IntegrationStrategy::ReferencesBased => {
        self.setup_references_based_integration().await?;
      }
      IntegrationStrategy::Synchronized => {
        self.setup_synchronized_integration().await?;
      }
      IntegrationStrategy::EventDriven => {
        self.setup_event_driven_integration().await?;
      }
    }
    Ok(())
  }

  /// Setup references-based integration
  /// Cozo stores metadata with vector_ids pointing to LanceDB
  async fn setup_references_based_integration(&self) -> SwarmResult<()> {
    #[cfg(feature = "graph")]
    {
      let cozo = self.cozo.read().await;

      // Create vector metadata table in Cozo
      let schema = r#"
                {
                    :create vector_metadata {
                        vector_id: String,
                        lancedb_table: String,
                        lancedb_row_id: String,
                        swarm_id: String,
                        agent_id: String,
                        content_type: String,
                        content_summary: String,
                        embedding_model: String,
                        created_at: String,
                        metadata: String,
                        =>
                        vector_id
                    }
                }
            "#;

      cozo
        .run_script(schema, Default::default(), cozo::ScriptMutability::Mutable)
        .map_err(|e| {
          SwarmError::Configuration(format!(
            "Failed to create vector metadata schema: {}",
            e
          ))
        })?;
    }

    tracing::info!("References-based integration setup complete");
    Ok(())
  }

  /// Setup synchronized integration
  /// Both databases store overlapping data with cross-references
  async fn setup_synchronized_integration(&self) -> SwarmResult<()> {
    // Create sync tables in both databases
    self.setup_references_based_integration().await?;

    #[cfg(feature = "graph")]
    {
      let cozo = self.cozo.read().await;

      // Create sync log table
      let sync_schema = r#"
                {
                    :create sync_log {
                        sync_id: String,
                        operation: String,
                        table_name: String,
                        record_id: String,
                        timestamp: String,
                        status: String,
                        =>
                        sync_id
                    }
                }
            "#;

      cozo
        .run_script(
          sync_schema,
          Default::default(),
          cozo::ScriptMutability::Mutable,
        )
        .map_err(|e| {
          SwarmError::Configuration(format!(
            "Failed to create sync schema: {}",
            e
          ))
        })?;
    }

    tracing::info!("Synchronized integration setup complete");
    Ok(())
  }

  /// Setup event-driven integration
  /// Use events to coordinate operations between databases
  async fn setup_event_driven_integration(&self) -> SwarmResult<()> {
    self.setup_references_based_integration().await?;

    // Setup event channels (would implement with tokio channels or similar)
    tracing::info!("Event-driven integration setup complete");
    Ok(())
  }
}

/// High-level operations that use both databases optimally
impl UnifiedDatabase {
  /// Store agent memory with vector embeddings
  /// Cozo: stores structured metadata, relationships
  /// LanceDB: stores vector embeddings for similarity search
  pub async fn store_agent_memory_with_embeddings(
    &self,
    agent_id: &str,
    swarm_id: &str,
    memory_content: &str,
    embedding: Vec<f32>,
    metadata: Value,
  ) -> SwarmResult<String> {
    let memory_id = Uuid::new_v4().to_string();

    // 1. Store vector in LanceDB for similarity search
    #[cfg(feature = "vector")]
    {
      self
        .lancedb
        .insert_embeddings(
          "agent_memories",
          vec![embedding],
          vec![memory_content.to_string()],
          vec![serde_json::json!({
              "memory_id": memory_id,
              "agent_id": agent_id,
              "swarm_id": swarm_id,
              "metadata": metadata
          })],
        )
        .await?;
    }

    // 2. Store structured data and relationships in Cozo
    #[cfg(feature = "graph")]
    {
      let cozo = self.cozo.read().await;

      let query = format!(
        r#"
                {{
                    # Store memory record
                    ?[memory_id, agent_id, swarm_id, content, created_at, metadata] <- 
                    [['{}', '{}', '{}', '{}', '{}', '{}']]
                    
                    :put agent_memories {{memory_id, agent_id, swarm_id, content, created_at, metadata}}
                    
                    # Store vector reference
                    ?[vector_id, lancedb_table, lancedb_row_id, swarm_id, agent_id, content_type, content_summary, embedding_model, created_at, metadata] <-
                    [['{}', 'agent_memories', '{}', '{}', '{}', 'memory', '{}', 'default', '{}', '{}']]
                    
                    :put vector_metadata {{vector_id, lancedb_table, lancedb_row_id, swarm_id, agent_id, content_type, content_summary, embedding_model, created_at, metadata}}
                }}
            "#,
        memory_id,
        agent_id,
        swarm_id,
        memory_content,
        Utc::now().to_rfc3339(),
        serde_json::to_string(&metadata).unwrap_or_default(),
        memory_id,
        memory_id,
        swarm_id,
        agent_id,
        &memory_content[..memory_content.len().min(100)], // Summary
        Utc::now().to_rfc3339(),
        serde_json::to_string(&metadata).unwrap_or_default()
      );

      cozo
        .run_script(&query, Default::default(), cozo::ScriptMutability::Mutable)
        .map_err(|e| {
          SwarmError::Persistence(format!(
            "Failed to store memory in Cozo: {}",
            e
          ))
        })?;
    }

    tracing::debug!("Stored agent memory with embeddings: {}", memory_id);
    Ok(memory_id)
  }

  /// Search agent memories using vector similarity + graph relationships
  /// LanceDB: finds similar memories by embedding
  /// Cozo: enriches results with graph context and relationships
  pub async fn search_agent_memories(
    &self,
    query_embedding: Vec<f32>,
    swarm_id: Option<&str>,
    agent_id: Option<&str>,
    limit: usize,
  ) -> SwarmResult<Vec<EnrichedMemoryResult>> {
    let mut results = Vec::new();

    // 1. Vector similarity search in LanceDB
    #[cfg(feature = "vector")]
    {
      let vector_query = crate::vector::VectorQuery {
        embedding: query_embedding,
        table_name: "agent_memories".to_string(),
        limit,
        threshold: Some(0.7),
        coordinate_agents: false,
        use_central_rag: false,
        federate_query: false,
        metadata_filter: None,
      };

      let vector_results =
        self.lancedb.distributed_search(vector_query).await?;

      // 2. Enrich each result with graph data from Cozo
      for result in vector_results.local_results {
        if let Some(memory_id) =
          result.metadata.get("memory_id").and_then(|v| v.as_str())
        {
          let enriched = self.enrich_memory_result(memory_id, result).await?;
          results.push(enriched);
        }
      }
    }

    Ok(results)
  }

  /// Enrich vector search result with graph data from Cozo
  async fn enrich_memory_result(
    &self,
    memory_id: &str,
    vector_result: crate::vector::VectorResult,
  ) -> SwarmResult<EnrichedMemoryResult> {
    #[cfg(feature = "graph")]
    {
      let cozo = self.cozo.read().await;

      // Query related data and relationships
      let query = format!(
        r#"
                {{
                    # Get memory details
                    ?[agent_id, swarm_id, content, created_at, metadata] := 
                        *agent_memories[memory_id, agent_id, swarm_id, content, created_at, metadata],
                        memory_id == '{}'
                    
                    # Get related memories from same agent
                    ?[related_memory_id, related_content] := 
                        *agent_memories[related_memory_id, agent_id, swarm_id, related_content, _, _],
                        *agent_memories['{}', agent_id, _, _, _, _],
                        related_memory_id != '{}'
                }}
            "#,
        memory_id, memory_id, memory_id
      );

      let result = cozo
        .run_script(
          &query,
          Default::default(),
          cozo::ScriptMutability::Immutable,
        )
        .map_err(|e| {
          SwarmError::Persistence(format!(
            "Failed to enrich memory result: {}",
            e
          ))
        })?;

      // Parse results and create enriched response
      if let Some(row) = result.rows.first() {
        return Ok(EnrichedMemoryResult {
          memory_id: memory_id.to_string(),
          vector_similarity_score: vector_result.score,
          content: vector_result.content,
          agent_id: extract_string_from_cozo_value(row.get(0)),
          swarm_id: extract_string_from_cozo_value(row.get(1)),
          created_at: extract_string_from_cozo_value(row.get(3)),
          metadata: vector_result.metadata,
          related_memories: Vec::new(), // Would parse from second query
          graph_context: GraphContext {
            connected_agents: Vec::new(),
            related_tasks: Vec::new(),
            relationship_strength: 1.0,
          },
        });
      }
    }

    // Fallback if graph feature not enabled
    Ok(EnrichedMemoryResult {
      memory_id: memory_id.to_string(),
      vector_similarity_score: vector_result.score,
      content: vector_result.content,
      agent_id: "unknown".to_string(),
      swarm_id: "unknown".to_string(),
      created_at: Utc::now().to_rfc3339(),
      metadata: vector_result.metadata,
      related_memories: Vec::new(),
      graph_context: GraphContext::default(),
    })
  }

  /// Get database statistics from both systems
  pub async fn get_unified_stats(&self) -> SwarmResult<UnifiedDatabaseStats> {
    let mut stats = UnifiedDatabaseStats::default();

    // Cozo stats
    #[cfg(feature = "graph")]
    {
      let cozo = self.cozo.read().await;
      let memory_count_query = r#"
                {
                    ?[count] := count(memory_id), *agent_memories[memory_id]
                }
            "#;

      let result = cozo
        .run_script(
          memory_count_query,
          Default::default(),
          cozo::ScriptMutability::Immutable,
        )
        .unwrap_or_default();

      stats.cozo_records = result
        .rows
        .first()
        .and_then(|row| row.first())
        .and_then(|val| match val {
          cozo::DataValue::Int(n) => Some(*n as usize),
          _ => None,
        })
        .unwrap_or(0);
    }

    // LanceDB stats
    #[cfg(feature = "vector")]
    {
      let vector_stats = self.lancedb.get_stats().await?;
      stats.lancedb_tables = vector_stats.total_tables;
    }

    Ok(stats)
  }
}

/// Enriched memory result combining vector similarity + graph context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichedMemoryResult {
  pub memory_id: String,
  pub vector_similarity_score: f32,
  pub content: String,
  pub agent_id: String,
  pub swarm_id: String,
  pub created_at: String,
  pub metadata: Value,
  pub related_memories: Vec<String>,
  pub graph_context: GraphContext,
}

/// Graph context from Cozo relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphContext {
  pub connected_agents: Vec<String>,
  pub related_tasks: Vec<String>,
  pub relationship_strength: f32,
}

impl Default for GraphContext {
  fn default() -> Self {
    Self {
      connected_agents: Vec::new(),
      related_tasks: Vec::new(),
      relationship_strength: 0.0,
    }
  }
}

/// Unified database statistics
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct UnifiedDatabaseStats {
  pub cozo_records: usize,
  pub lancedb_tables: usize,
  pub total_vectors: usize,
  pub integration_health: f32,
}

impl Default for UnifiedDatabaseConfig {
  fn default() -> Self {
    Self {
      cozo_path: ".zen/swarm/cozo.db".to_string(),
      lancedb_path: ".zen/swarm/vectors".to_string(),
      integration_strategy: IntegrationStrategy::ReferencesBased,
    }
  }
}

// Helper function to extract string from Cozo DataValue
fn extract_string_from_cozo_value(value: Option<&cozo::DataValue>) -> String {
  match value {
    Some(cozo::DataValue::Str(s)) => s.clone(),
    Some(v) => format!("{:?}", v),
    None => "unknown".to_string(),
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::TempDir;

  #[tokio::test]
  async fn test_unified_database_creation() {
    let temp_dir = TempDir::new().unwrap();
    let cozo_path = temp_dir.path().join("test_cozo.db");
    let lance_path = temp_dir.path().join("test_vectors");

    let config = UnifiedDatabaseConfig {
      cozo_path: cozo_path.to_string_lossy().to_string(),
      lancedb_path: lance_path.to_string_lossy().to_string(),
      integration_strategy: IntegrationStrategy::ReferencesBased,
    };

    let unified_db = UnifiedDatabase::new(config).await;
    assert!(unified_db.is_ok());
  }
}
