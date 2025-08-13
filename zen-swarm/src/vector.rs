//! Vector database and embeddings module with distributed agent coordination
//!
//! High-performance vector search using LanceDB v0.20 with agent-to-agent (A2A) communication,
//! central RAG systems, and FACT (Federated Agent Coordination Technology) integration.

use crate::{Agent, SwarmError};
use anyhow::Result;
use arrow::{
  array::{Float32Array, StringArray},
  datatypes::{DataType, Field, Schema},
  record_batch::RecordBatch,
};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use data_encoding;
use futures_util;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Table};
use reqwest;
use ring;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_stream::StreamExt;
use tokio_tungstenite;
use uuid::Uuid;

/// Vector database manager with distributed agent coordination  
pub struct VectorDb {
  /// LanceDB connection
  connection: Arc<Connection>,
  /// Active tables by name
  tables: Arc<DashMap<String, Arc<Table>>>,
  /// Central RAG system endpoints
  rag_endpoints: Arc<RwLock<Vec<RagEndpoint>>>,
  /// FACT coordination nodes
  fact_nodes: Arc<RwLock<Vec<FactNode>>>,
  /// Agent-to-Agent communication channels
  a2a_channels: Arc<DashMap<String, AgentChannel>>,
}

/// Central RAG system endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagEndpoint {
  pub id: String,
  pub url: String,
  pub auth_token: Option<String>,
  pub capabilities: Vec<String>,
  pub priority: u8,
  pub health_check_url: String,
}

/// FACT (Federated Agent Coordination Technology) node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactNode {
  pub node_id: String,
  pub address: String,
  pub port: u16,
  pub public_key: String,
  pub capabilities: Vec<String>,
  pub trust_level: f32, // 0.0 to 1.0
  pub last_seen: chrono::DateTime<chrono::Utc>,
}

/// Agent-to-Agent communication channel
#[derive(Debug, Clone)]
pub struct AgentChannel {
  pub agent_id: String,
  pub channel_type: ChannelType,
  pub endpoint: String,
  pub encryption_key: Option<String>,
  pub last_activity: chrono::DateTime<chrono::Utc>,
}

/// Communication channel types for distributed agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelType {
  /// Direct TCP connection
  DirectTcp,
  /// WebSocket for real-time updates
  WebSocket,
  /// HTTP for request-response
  Http,
  /// MCP federation for standardized protocol
  McpFederation,
  /// Encrypted P2P channel
  EncryptedP2P,
}

/// Vector search query with distributed coordination
#[derive(Debug, Serialize, Deserialize)]
pub struct VectorQuery {
  pub embedding: Vec<f32>,
  pub table_name: String,
  pub limit: usize,
  pub threshold: Option<f32>,
  /// Request coordination from other agents
  pub coordinate_agents: bool,
  /// Query central RAG systems
  pub use_central_rag: bool,
  /// Federate across FACT network
  pub federate_query: bool,
  pub metadata_filter: Option<Value>,
}

/// Vector search result with provenance tracking
#[derive(Debug, Serialize, Deserialize)]
pub struct VectorResult {
  pub id: String,
  pub score: f32,
  pub content: String,
  pub metadata: Value,
  /// Source agent that provided this result
  pub source_agent: Option<String>,
  /// Central RAG source if applicable
  pub rag_source: Option<String>,
  /// FACT network provenance
  pub fact_provenance: Option<FactProvenance>,
}

/// FACT network provenance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactProvenance {
  pub origin_node: String,
  pub hop_count: u32,
  pub trust_chain: Vec<String>,
  pub verification_signatures: Vec<String>,
}

/// Distributed vector search results
#[derive(Debug, Serialize, Deserialize)]
pub struct DistributedVectorResults {
  pub local_results: Vec<VectorResult>,
  pub agent_results: Vec<(String, Vec<VectorResult>)>,
  pub rag_results: Vec<(String, Vec<VectorResult>)>,
  pub fact_results: Vec<(String, Vec<VectorResult>)>,
  pub total_nodes_queried: usize,
  pub query_time_ms: u64,
  pub federation_success: bool,
}

impl VectorDb {
  /// Create new vector database with distributed coordination
  pub async fn new(db_path: &str) -> Result<Self, SwarmError> {
    let connection =
      lancedb::connect(db_path).execute().await.map_err(|e| {
        SwarmError::Vector(format!("Failed to connect to LanceDB: {}", e))
      })?;

    Ok(Self {
      connection: Arc::new(connection),
      tables: Arc::new(DashMap::new()),
      rag_endpoints: Arc::new(RwLock::new(Vec::new())),
      fact_nodes: Arc::new(RwLock::new(Vec::new())),
      a2a_channels: Arc::new(DashMap::new()),
    })
  }

  /// Register central RAG system endpoint
  pub async fn register_rag_endpoint(
    &self,
    endpoint: RagEndpoint,
  ) -> Result<(), SwarmError> {
    let mut endpoints = self.rag_endpoints.write().await;
    endpoints.push(endpoint);
    Ok(())
  }

  /// Register FACT network node
  pub async fn register_fact_node(
    &self,
    node: FactNode,
  ) -> Result<(), SwarmError> {
    let mut nodes = self.fact_nodes.write().await;
    nodes.push(node);
    Ok(())
  }

  /// Establish agent-to-agent communication channel
  pub async fn establish_a2a_channel(
    &self,
    agent_id: String,
    channel_type: ChannelType,
    endpoint: String,
    encryption_key: Option<String>,
  ) -> Result<(), SwarmError> {
    let channel = AgentChannel {
      agent_id: agent_id.clone(),
      channel_type,
      endpoint,
      encryption_key,
      last_activity: chrono::Utc::now(),
    };

    self.a2a_channels.insert(agent_id, channel);
    Ok(())
  }

  /// Create vector table with schema
  pub async fn create_table(
    &self,
    table_name: &str,
    schema: Arc<Schema>,
  ) -> Result<(), SwarmError> {
    let table = self
      .connection
      .create_empty_table(table_name, schema)
      .execute()
      .await
      .map_err(|e| {
        SwarmError::Vector(format!("Failed to create table: {}", e))
      })?;

    self.tables.insert(table_name.to_string(), Arc::new(table));
    Ok(())
  }

  /// Insert embeddings with metadata
  pub async fn insert_embeddings(
    &self,
    table_name: &str,
    embeddings: Vec<Vec<f32>>,
    contents: Vec<String>,
    metadata: Vec<Value>,
  ) -> Result<(), SwarmError> {
    let table = self.tables.get(table_name).ok_or_else(|| {
      SwarmError::Vector(format!("Table {} not found", table_name))
    })?;

    // Convert to Arrow format
    let embedding_array =
      Float32Array::from_iter_values(embeddings.into_iter().flatten());
    let content_array = StringArray::from(contents);
    let metadata_array = StringArray::from(
      metadata
        .into_iter()
        .map(|m| m.to_string())
        .collect::<Vec<_>>(),
    );

    let schema = Arc::new(Schema::new(vec![
      Field::new("embedding", DataType::Float32, false),
      Field::new("content", DataType::Utf8, false),
      Field::new("metadata", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
      schema,
      vec![
        Arc::new(embedding_array),
        Arc::new(content_array),
        Arc::new(metadata_array),
      ],
    )
    .map_err(|e| {
      SwarmError::Vector(format!("Failed to create batch: {}", e))
    })?;

    // Use LanceDB v0.20.0 API for batch insertion
    let table_ref = table.value();
    let schema = batch.schema();
    let batch_reader =
      arrow::record_batch::RecordBatchIterator::new(vec![Ok(batch)], schema);
    table_ref.add(batch_reader).execute().await.map_err(|e| {
      SwarmError::Vector(format!("Failed to insert data: {}", e))
    })?;

    Ok(())
  }

  /// Distributed vector search with A2A coordination and central RAG integration
  pub async fn distributed_search(
    &self,
    query: VectorQuery,
  ) -> Result<DistributedVectorResults, SwarmError> {
    let start_time = std::time::Instant::now();
    let mut total_nodes_queried = 1; // Start with local node

    // 1. Local vector search
    let local_results = self.local_search(&query).await?;

    // 2. Agent-to-Agent coordination if requested
    let agent_results = if query.coordinate_agents {
      self.coordinate_with_agents(&query).await?
    } else {
      Vec::new()
    };

    // 3. Central RAG system queries if requested
    let rag_results = if query.use_central_rag {
      self.query_central_rag_systems(&query).await?
    } else {
      Vec::new()
    };

    // 4. FACT network federation if requested
    let fact_results = if query.federate_query {
      self.federate_across_fact_network(&query).await?
    } else {
      Vec::new()
    };

    total_nodes_queried +=
      agent_results.len() + rag_results.len() + fact_results.len();

    let query_time_ms = start_time.elapsed().as_millis() as u64;
    let federation_success = !agent_results.is_empty()
      || !rag_results.is_empty()
      || !fact_results.is_empty();

    Ok(DistributedVectorResults {
      local_results,
      agent_results,
      rag_results,
      fact_results,
      total_nodes_queried,
      query_time_ms,
      federation_success,
    })
  }

  /// Local vector search
  async fn local_search(
    &self,
    query: &VectorQuery,
  ) -> Result<Vec<VectorResult>, SwarmError> {
    let table = self.tables.get(&query.table_name).ok_or_else(|| {
      SwarmError::Vector(format!("Table {} not found", query.table_name))
    })?;

    // Perform vector similarity search using LanceDB v0.20.0 API
    let table_ref = table.value();
    let mut search_query = table_ref.query().limit(query.limit);

    // Note: distance_type may not be available in v0.20.0 API
    // if let Some(_threshold) = query.threshold {
    //     search_query = search_query.distance_type(lancedb::DistanceType::Cosine);
    // }

    let results = search_query
      .nearest_to(query.embedding.clone())
      .map_err(|e| {
        SwarmError::Vector(format!("Failed to create search query: {}", e))
      })?
      .execute()
      .await
      .map_err(|e| SwarmError::Vector(format!("Search failed: {}", e)))?;

    let mut vector_results = Vec::new();
    let mut stream = results;
    tokio::pin!(stream);

    while let Some(batch) = stream
      .try_next()
      .await
      .map_err(|e| SwarmError::Vector(format!("Stream error: {}", e)))?
    {
      // Process batch results with production similarity scoring
      for row in 0..batch.num_rows() {
        let id = self
          .extract_id_from_batch(&batch, row)
          .unwrap_or_else(|| Uuid::new_v4().to_string());
        let score = self
          .calculate_similarity_score(&batch, row, &query.embedding)
          .unwrap_or(0.0);
        let content = self
          .extract_content_from_batch(&batch, row)
          .unwrap_or_else(|| format!("Content row {}", row));
        let metadata = self
          .extract_metadata_from_batch(&batch, row)
          .unwrap_or_else(|| serde_json::json!({}));

        // Apply threshold filtering
        if let Some(threshold) = query.threshold {
          if score < threshold {
            continue;
          }
        }

        vector_results.push(VectorResult {
          id,
          score,
          content,
          metadata,
          source_agent: None,
          rag_source: None,
          fact_provenance: None,
        });
      }
    }

    Ok(vector_results)
  }

  /// Coordinate with other agents via A2A channels
  async fn coordinate_with_agents(
    &self,
    query: &VectorQuery,
  ) -> Result<Vec<(String, Vec<VectorResult>)>, SwarmError> {
    let mut agent_results = Vec::new();

    for channel_ref in self.a2a_channels.iter() {
      let (agent_id, channel) = channel_ref.pair();

      match channel.channel_type {
        ChannelType::McpFederation => {
          // Use MCP federation protocol for standardized agent communication
          let results = self
            .query_agent_via_mcp(agent_id, &channel.endpoint, query)
            .await?;
          agent_results.push((agent_id.clone(), results));
        }
        ChannelType::WebSocket => {
          // Real-time WebSocket communication
          let results = self
            .query_agent_via_websocket(agent_id, &channel.endpoint, query)
            .await?;
          agent_results.push((agent_id.clone(), results));
        }
        ChannelType::Http => {
          // HTTP request-response
          let results = self
            .query_agent_via_http(agent_id, &channel.endpoint, query)
            .await?;
          agent_results.push((agent_id.clone(), results));
        }
        _ => {
          // Other channel types - implement as needed
          continue;
        }
      }
    }

    Ok(agent_results)
  }

  /// Query central RAG systems
  async fn query_central_rag_systems(
    &self,
    query: &VectorQuery,
  ) -> Result<Vec<(String, Vec<VectorResult>)>, SwarmError> {
    let endpoints = self.rag_endpoints.read().await;
    let mut rag_results = Vec::new();

    for endpoint in endpoints.iter() {
      // Query each RAG endpoint
      let results = self.query_rag_endpoint(endpoint, query).await?;
      rag_results.push((endpoint.id.clone(), results));
    }

    Ok(rag_results)
  }

  /// Federate across FACT network
  async fn federate_across_fact_network(
    &self,
    query: &VectorQuery,
  ) -> Result<Vec<(String, Vec<VectorResult>)>, SwarmError> {
    let nodes = self.fact_nodes.read().await;
    let mut fact_results = Vec::new();

    for node in nodes.iter() {
      // Only query trusted nodes
      if node.trust_level >= 0.7 {
        let results = self.query_fact_node(node, query).await?;
        fact_results.push((node.node_id.clone(), results));
      }
    }

    Ok(fact_results)
  }

  /// Query agent via MCP federation protocol
  async fn query_agent_via_mcp(
    &self,
    agent_id: &str,
    endpoint: &str,
    query: &VectorQuery,
  ) -> Result<Vec<VectorResult>, SwarmError> {
    // MCP federation implementation
    // This would use the MCP JSON-RPC protocol to communicate with other agents
    tracing::debug!(
      "Querying agent {} via MCP federation at {}",
      agent_id,
      endpoint
    );

    // Placeholder implementation
    Ok(vec![VectorResult {
      id: Uuid::new_v4().to_string(),
      score: 0.75,
      content: format!("Federated result from agent {}", agent_id),
      metadata: serde_json::json!({"source": "mcp_federation"}),
      source_agent: Some(agent_id.to_string()),
      rag_source: None,
      fact_provenance: None,
    }])
  }

  /// Query agent via WebSocket
  async fn query_agent_via_websocket(
    &self,
    agent_id: &str,
    endpoint: &str,
    query: &VectorQuery,
  ) -> Result<Vec<VectorResult>, SwarmError> {
    tracing::debug!(
      "Querying agent {} via WebSocket at {}",
      agent_id,
      endpoint
    );

    // Placeholder implementation
    Ok(vec![VectorResult {
      id: Uuid::new_v4().to_string(),
      score: 0.80,
      content: format!("WebSocket result from agent {}", agent_id),
      metadata: serde_json::json!({"source": "websocket", "realtime": true}),
      source_agent: Some(agent_id.to_string()),
      rag_source: None,
      fact_provenance: None,
    }])
  }

  /// Query agent via HTTP
  async fn query_agent_via_http(
    &self,
    agent_id: &str,
    endpoint: &str,
    query: &VectorQuery,
  ) -> Result<Vec<VectorResult>, SwarmError> {
    tracing::debug!("Querying agent {} via HTTP at {}", agent_id, endpoint);

    // Placeholder implementation
    Ok(vec![VectorResult {
      id: Uuid::new_v4().to_string(),
      score: 0.70,
      content: format!("HTTP result from agent {}", agent_id),
      metadata: serde_json::json!({"source": "http"}),
      source_agent: Some(agent_id.to_string()),
      rag_source: None,
      fact_provenance: None,
    }])
  }

  /// Query RAG endpoint
  async fn query_rag_endpoint(
    &self,
    endpoint: &RagEndpoint,
    query: &VectorQuery,
  ) -> Result<Vec<VectorResult>, SwarmError> {
    tracing::debug!(
      "Querying RAG endpoint {} at {}",
      endpoint.id,
      endpoint.url
    );

    // Placeholder implementation for RAG system integration
    Ok(vec![VectorResult {
      id: Uuid::new_v4().to_string(),
      score: 0.90,
      content: format!("Central RAG result from {}", endpoint.id),
      metadata: serde_json::json!({"rag_endpoint": endpoint.id, "priority": endpoint.priority}),
      source_agent: None,
      rag_source: Some(endpoint.id.clone()),
      fact_provenance: None,
    }])
  }

  /// Query FACT network node
  async fn query_fact_node(
    &self,
    node: &FactNode,
    query: &VectorQuery,
  ) -> Result<Vec<VectorResult>, SwarmError> {
    tracing::debug!(
      "Querying FACT node {} at {}:{}",
      node.node_id,
      node.address,
      node.port
    );

    // Placeholder implementation for FACT network federation
    let provenance = FactProvenance {
      origin_node: node.node_id.clone(),
      hop_count: 1,
      trust_chain: vec![node.node_id.clone()],
      verification_signatures: vec!["sig_placeholder".to_string()],
    };

    Ok(vec![VectorResult {
      id: Uuid::new_v4().to_string(),
      score: 0.85,
      content: format!("FACT network result from {}", node.node_id),
      metadata: serde_json::json!({
          "fact_node": node.node_id,
          "trust_level": node.trust_level,
          "capabilities": node.capabilities
      }),
      source_agent: None,
      rag_source: None,
      fact_provenance: Some(provenance),
    }])
  }

  /// Get vector database statistics
  pub async fn get_stats(&self) -> Result<VectorDbStats, SwarmError> {
    Ok(VectorDbStats {
      total_tables: self.tables.len(),
      active_rag_endpoints: self.rag_endpoints.read().await.len(),
      fact_network_nodes: self.fact_nodes.read().await.len(),
      a2a_channels: self.a2a_channels.len(),
    })
  }

  /// Production-level similarity scoring
  fn calculate_similarity_score(
    &self,
    batch: &RecordBatch,
    row: usize,
    query_embedding: &[f32],
  ) -> Option<f32> {
    // Extract embedding vector from batch
    let embedding_column = batch.column_by_name("embedding")?;

    // Handle different embedding column types
    match embedding_column.data_type() {
      DataType::FixedSizeList(_, size) => {
        let list_array = embedding_column
          .as_any()
          .downcast_ref::<arrow::array::FixedSizeListArray>()?;
        let values = list_array.value(row);
        let float_values = values.as_any().downcast_ref::<Float32Array>()?;

        let embedding: Vec<f32> =
          (0..*size as usize).map(|i| float_values.value(i)).collect();

        Some(self.cosine_similarity(query_embedding, &embedding))
      }
      DataType::Float32 => {
        // Single dimension case - simple difference
        let float_array =
          embedding_column.as_any().downcast_ref::<Float32Array>()?;
        let value = float_array.value(row);
        let query_val = query_embedding.get(0).copied().unwrap_or(0.0);
        Some(1.0 - (value - query_val).abs().min(1.0))
      }
      _ => {
        tracing::warn!(
          "Unsupported embedding data type: {:?}",
          embedding_column.data_type()
        );
        None
      }
    }
  }

  /// Cosine similarity calculation
  fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
      return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
      return 0.0;
    }

    (dot_product / (norm_a * norm_b)).clamp(-1.0, 1.0)
  }

  /// Extract ID from batch result
  fn extract_id_from_batch(
    &self,
    batch: &RecordBatch,
    row: usize,
  ) -> Option<String> {
    if let Some(id_column) = batch.column_by_name("id") {
      if let Some(string_array) =
        id_column.as_any().downcast_ref::<StringArray>()
      {
        return string_array.value(row).to_string().into();
      }
    }
    None
  }

  /// Extract content from batch result
  fn extract_content_from_batch(
    &self,
    batch: &RecordBatch,
    row: usize,
  ) -> Option<String> {
    if let Some(content_column) = batch.column_by_name("content") {
      if let Some(string_array) =
        content_column.as_any().downcast_ref::<StringArray>()
      {
        return string_array.value(row).to_string().into();
      }
    }
    None
  }

  /// Extract metadata from batch result
  fn extract_metadata_from_batch(
    &self,
    batch: &RecordBatch,
    row: usize,
  ) -> Option<serde_json::Value> {
    if let Some(metadata_column) = batch.column_by_name("metadata") {
      if let Some(string_array) =
        metadata_column.as_any().downcast_ref::<StringArray>()
      {
        let metadata_str = string_array.value(row);
        return serde_json::from_str(metadata_str).ok();
      }
    }
    None
  }

  /// Sign FACT network request
  fn sign_fact_request(
    &self,
    request: &str,
    public_key: &str,
  ) -> Result<String, SwarmError> {
    use data_encoding::HEXUPPER;
    use ring::digest;

    // Simple signing using SHA-256 hash with public key
    let mut context = digest::Context::new(&digest::SHA256);
    context.update(request.as_bytes());
    context.update(public_key.as_bytes());
    let digest_bytes = context.finish();

    Ok(HEXUPPER.encode(digest_bytes.as_ref()))
  }

  /// Verify FACT network signature
  fn verify_fact_signature(
    &self,
    data: &str,
    signature: &str,
    public_key: &str,
  ) -> Result<bool, SwarmError> {
    let expected_signature = self.sign_fact_request(data, public_key)?;
    Ok(expected_signature == signature)
  }
}

/// Vector database statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct VectorDbStats {
  pub total_tables: usize,
  pub active_rag_endpoints: usize,
  pub fact_network_nodes: usize,
  pub a2a_channels: usize,
}

/// Helper function to create default vector schema
pub fn create_default_vector_schema(embedding_dim: usize) -> Arc<Schema> {
  Arc::new(Schema::new(vec![
    Field::new("id", DataType::Utf8, false),
    Field::new(
      "embedding",
      DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        embedding_dim as i32,
      ),
      false,
    ),
    Field::new("content", DataType::Utf8, false),
    Field::new("metadata", DataType::Utf8, true),
    Field::new(
      "timestamp",
      DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None),
      false,
    ),
  ]))
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::TempDir;

  #[tokio::test]
  async fn test_vector_db_creation() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test_vector.db");

    let vector_db = VectorDb::new(db_path.to_str().unwrap()).await;
    assert!(vector_db.is_ok());
  }

  #[tokio::test]
  async fn test_distributed_search() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test_dist_vector.db");

    let vector_db = VectorDb::new(db_path.to_str().unwrap()).await.unwrap();

    let query = VectorQuery {
      embedding: vec![0.1, 0.2, 0.3, 0.4],
      table_name: "test_table".to_string(),
      limit: 10,
      threshold: Some(0.5),
      coordinate_agents: false,
      use_central_rag: false,
      federate_query: false,
      metadata_filter: None,
    };

    // This will fail because table doesn't exist, but tests the interface
    let result = vector_db.distributed_search(query).await;
    assert!(result.is_err()); // Expected since no table created
  }
}
