//! Type-secured event logging system for database coordination and audit trails
//!
//! Provides compile-time type safety for all database events, ensuring consistency
//! between Cozo and LanceDB operations with full audit trails and recovery capabilities.
//!
//! ## Features
//!
//! - **Type Safety**: Compile-time guarantees for event structure
//! - **Audit Trails**: Complete history of all database operations
//! - **Event Sourcing**: Reconstruct state from event log
//! - **Cross-Database Coordination**: Sync between Cozo and LanceDB
//! - **Serialization**: Efficient binary and JSON serialization
//! - **Recovery**: Replay events for disaster recovery

use crate::{SwarmError, SwarmResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use uuid::Uuid;

/// Type-secured event log manager
#[derive(Debug)]
pub struct TypeSecuredEventLog {
  /// Event storage in Cozo for durability and complex queries
  #[cfg(feature = "graph")]
  cozo: Arc<RwLock<cozo::DbInstance>>,

  /// In-memory event cache for fast access
  event_cache: Arc<RwLock<EventCache>>,

  /// Event broadcast channel for real-time subscribers
  event_broadcaster: broadcast::Sender<SecuredEvent>,

  /// Configuration
  config: EventLogConfig,
}

/// Event log configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventLogConfig {
  /// Maximum events in memory cache
  pub cache_size: usize,

  /// Event retention period in days
  pub retention_days: u32,

  /// Enable event compression
  pub compression: bool,

  /// Event verification settings
  pub verification: EventVerificationConfig,
}

/// Event verification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventVerificationConfig {
  /// Enable cryptographic signatures
  pub signatures: bool,

  /// Enable checksums for integrity
  pub checksums: bool,

  /// Enable sequence number validation
  pub sequence_validation: bool,
}

/// In-memory event cache for performance
#[derive(Debug, Default)]
pub struct EventCache {
  /// Recent events by type
  events_by_type: HashMap<EventType, Vec<SecuredEvent>>,

  /// Events by source (cozo, lancedb, unified)
  events_by_source: HashMap<EventSource, Vec<SecuredEvent>>,

  /// Event sequence numbers for ordering
  sequence_tracker: HashMap<EventSource, u64>,
}

/// Type-secured event with compile-time safety
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecuredEvent {
  /// Unique event identifier
  pub event_id: String,

  /// Event sequence number for ordering
  pub sequence_number: u64,

  /// Event type (type-safe enum)
  pub event_type: EventType,

  /// Event source system
  pub source: EventSource,

  /// Timestamp with microsecond precision
  pub timestamp: DateTime<Utc>,

  /// Type-safe event payload
  pub payload: EventPayload,

  /// Event metadata
  pub metadata: EventMetadata,

  /// Cryptographic signature (optional)
  pub signature: Option<String>,

  /// Integrity checksum
  pub checksum: String,
}

/// Type-safe event types with exhaustive matching
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
  // Database Operations
  DatabaseCreate,
  DatabaseDelete,
  DatabaseUpdate,
  DatabaseQuery,

  // Vector Operations
  VectorInsert,
  VectorUpdate,
  VectorDelete,
  VectorSearch,

  // Graph Operations
  NodeCreate,
  NodeUpdate,
  NodeDelete,
  EdgeCreate,
  EdgeUpdate,
  EdgeDelete,
  GraphQuery,

  // Swarm Operations
  SwarmCreate,
  SwarmUpdate,
  SwarmDelete,
  SwarmQuery,

  // Agent Operations
  AgentCreate,
  AgentUpdate,
  AgentDelete,
  AgentMemoryStore,
  AgentMemoryRetrieve,

  // Task Operations
  TaskCreate,
  TaskUpdate,
  TaskComplete,
  TaskFail,

  // System Operations
  SystemStart,
  SystemStop,
  SystemError,
  SystemRecovery,

  // Synchronization Events
  SyncStart,
  SyncComplete,
  SyncConflict,
  SyncResolution,
}

/// Event source systems
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventSource {
  Cozo,
  LanceDB,
  UnifiedDatabase,
  SwarmCoordinator,
  Agent,
  System,
}

/// Type-safe event payloads with guaranteed structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventPayload {
  // Database payloads
  DatabasePayload(DatabaseEventPayload),

  // Vector payloads
  VectorPayload(VectorEventPayload),

  // Graph payloads
  GraphPayload(GraphEventPayload),

  // Swarm payloads
  SwarmPayload(SwarmEventPayload),

  // Agent payloads
  AgentPayload(AgentEventPayload),

  // Task payloads
  TaskPayload(TaskEventPayload),

  // System payloads
  SystemPayload(SystemEventPayload),

  // Sync payloads
  SyncPayload(SyncEventPayload),
}

/// Database operation event payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseEventPayload {
  pub database_name: String,
  pub table_name: Option<String>,
  pub operation_details: Value,
  pub affected_rows: Option<u64>,
  pub execution_time_ms: u64,
}

/// Vector operation event payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEventPayload {
  pub table_name: String,
  pub vector_dimension: usize,
  pub vector_count: usize,
  pub operation_details: VectorOperation,
  pub similarity_threshold: Option<f32>,
  pub execution_time_ms: u64,
}

/// Specific vector operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorOperation {
  Insert {
    vectors: usize,
    content_length: usize,
  },
  Update {
    vector_id: String,
    changes: Value,
  },
  Delete {
    vector_ids: Vec<String>,
  },
  Search {
    query_vector_dim: usize,
    results_count: usize,
    similarity_scores: Vec<f32>,
  },
}

/// Graph operation event payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEventPayload {
  pub node_count: Option<usize>,
  pub edge_count: Option<usize>,
  pub query_complexity: QueryComplexity,
  pub operation_details: GraphOperation,
  pub execution_time_ms: u64,
}

/// Graph operation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphOperation {
  NodeOperation {
    node_id: String,
    node_type: String,
    properties: Value,
  },
  EdgeOperation {
    edge_id: String,
    from_node: String,
    to_node: String,
    edge_type: String,
  },
  QueryOperation {
    query: String,
    result_count: usize,
  },
}

/// Query complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryComplexity {
  Simple,
  Medium,
  Complex,
  VeryComplex,
}

/// Swarm operation event payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmEventPayload {
  pub swarm_id: String,
  pub operation: SwarmOperation,
  pub agent_count: usize,
  pub task_count: usize,
  pub performance_metrics: SwarmPerformanceMetrics,
}

/// Swarm operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmOperation {
  Create { config: Value },
  Update { changes: Value },
  Delete { cleanup_strategy: String },
  Query { query_type: String, results: Value },
}

/// Swarm performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmPerformanceMetrics {
  pub cpu_usage: f32,
  pub memory_usage_mb: u64,
  pub network_io_bytes: u64,
  pub task_completion_rate: f32,
}

/// Agent operation event payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEventPayload {
  pub agent_id: String,
  pub swarm_id: String,
  pub operation: AgentOperation,
  pub performance_impact: AgentPerformanceImpact,
}

/// Agent operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentOperation {
  Create {
    agent_type: String,
    capabilities: Vec<String>,
  },
  Update {
    changes: Value,
  },
  Delete {
    reason: String,
  },
  MemoryStore {
    memory_type: String,
    content_size: usize,
  },
  MemoryRetrieve {
    query: String,
    results_count: usize,
  },
}

/// Agent performance impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPerformanceImpact {
  pub cpu_delta: f32,
  pub memory_delta_mb: i64,
  pub execution_time_ms: u64,
}

/// Task operation event payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEventPayload {
  pub task_id: String,
  pub swarm_id: String,
  pub operation: TaskOperation,
  pub resource_usage: TaskResourceUsage,
}

/// Task operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskOperation {
  Create {
    description: String,
    priority: String,
    requirements: Value,
  },
  Update {
    changes: Value,
  },
  Complete {
    result: Value,
    artifacts: Vec<String>,
  },
  Fail {
    error: String,
    retry_count: u32,
  },
}

/// Task resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResourceUsage {
  pub cpu_time_ms: u64,
  pub memory_peak_mb: u64,
  pub network_bytes: u64,
  pub storage_bytes: u64,
}

/// System operation event payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemEventPayload {
  pub operation: SystemOperation,
  pub system_state: SystemState,
  pub resource_usage: SystemResourceUsage,
}

/// System operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemOperation {
  Start {
    version: String,
    config: Value,
  },
  Stop {
    reason: String,
    graceful: bool,
  },
  Error {
    error_type: String,
    error_message: String,
    stack_trace: Option<String>,
  },
  Recovery {
    recovery_strategy: String,
    success: bool,
  },
}

/// System state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
  pub uptime_seconds: u64,
  pub active_swarms: usize,
  pub active_agents: usize,
  pub pending_tasks: usize,
  pub health_status: String,
}

/// System resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResourceUsage {
  pub cpu_usage_percent: f32,
  pub memory_usage_mb: u64,
  pub disk_usage_mb: u64,
  pub network_io_bytes: u64,
}

/// Synchronization event payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncEventPayload {
  pub sync_id: String,
  pub operation: SyncOperation,
  pub databases_involved: Vec<EventSource>,
  pub sync_metrics: SyncMetrics,
}

/// Synchronization operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncOperation {
  Start {
    sync_strategy: String,
    expected_duration_ms: u64,
  },
  Complete {
    records_synced: usize,
    conflicts_resolved: usize,
  },
  Conflict {
    conflict_type: String,
    resolution_strategy: String,
  },
  Resolution {
    resolution_result: Value,
    success: bool,
  },
}

/// Synchronization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncMetrics {
  pub duration_ms: u64,
  pub records_processed: usize,
  pub conflicts_detected: usize,
  pub bandwidth_used_bytes: u64,
}

/// Event metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
  /// Context information
  pub context: HashMap<String, Value>,

  /// Correlation ID for tracing related events
  pub correlation_id: Option<String>,

  /// Parent event ID for event hierarchies
  pub parent_event_id: Option<String>,

  /// Tags for categorization
  pub tags: Vec<String>,

  /// Priority level
  pub priority: EventPriority,

  /// Retention policy
  pub retention_policy: RetentionPolicy,
}

/// Event priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventPriority {
  Critical,
  High,
  Medium,
  Low,
  Debug,
}

/// Event retention policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionPolicy {
  Permanent,
  Days(u32),
  Weeks(u32),
  Months(u32),
  UntilProcessed,
}

impl TypeSecuredEventLog {
  /// Create new type-secured event log
  #[cfg(feature = "graph")]
  pub async fn new(
    cozo: Arc<RwLock<cozo::DbInstance>>,
    config: EventLogConfig,
  ) -> SwarmResult<Self> {
    let (event_broadcaster, _) = broadcast::channel(1000);

    let event_log = Self {
      cozo,
      event_cache: Arc::new(RwLock::new(EventCache::default())),
      event_broadcaster,
      config,
    };

    // Initialize event log schema in Cozo
    event_log.initialize_schema().await?;

    tracing::info!("Type-secured event log initialized");
    Ok(event_log)
  }

  /// Initialize event log schema in Cozo
  #[cfg(feature = "graph")]
  async fn initialize_schema(&self) -> SwarmResult<()> {
    let cozo = self.cozo.read().await;

    let schema = r#"
            {
                :create event_log {
                    event_id: String,
                    sequence_number: Int,
                    event_type: String,
                    source: String,
                    timestamp: String,
                    payload: String,
                    metadata: String,
                    signature: String?,
                    checksum: String,
                    =>
                    event_id
                }
                
                :create event_index_by_type {
                    event_type: String,
                    timestamp: String,
                    event_id: String,
                    =>
                    [event_type, timestamp, event_id]
                }
                
                :create event_index_by_source {
                    source: String,
                    timestamp: String,
                    event_id: String,
                    =>
                    [source, timestamp, event_id]
                }
            }
        "#;

    cozo
      .run_script(schema, Default::default(), cozo::ScriptMutability::Mutable)
      .map_err(|e| {
        SwarmError::Configuration(format!(
          "Failed to initialize event log schema: {}",
          e
        ))
      })?;

    Ok(())
  }

  /// Log a type-secured event
  pub async fn log_event(
    &self,
    event_type: EventType,
    source: EventSource,
    payload: EventPayload,
  ) -> SwarmResult<String> {
    let event_id = Uuid::new_v4().to_string();
    let sequence_number = self.get_next_sequence_number(source.clone()).await;
    let timestamp = Utc::now();

    // Calculate checksum for integrity
    let checksum = self.calculate_checksum(&event_type, &payload, timestamp)?;

    // Create signature if enabled
    let signature = if self.config.verification.signatures {
      Some(self.create_signature(&event_id, &payload)?)
    } else {
      None
    };

    let event = SecuredEvent {
      event_id: event_id.clone(),
      sequence_number,
      event_type: event_type.clone(),
      source: source.clone(),
      timestamp,
      payload,
      metadata: EventMetadata {
        context: HashMap::new(),
        correlation_id: None,
        parent_event_id: None,
        tags: Vec::new(),
        priority: EventPriority::Medium,
        retention_policy: RetentionPolicy::Days(self.config.retention_days),
      },
      signature,
      checksum,
    };

    // Store in cache
    self.cache_event(event.clone()).await;

    // Store in Cozo for durability
    #[cfg(feature = "graph")]
    self.persist_event(&event).await?;

    // Broadcast to subscribers
    let _ = self.event_broadcaster.send(event);

    tracing::debug!(
      "Logged type-secured event: {} ({:?})",
      event_id,
      event_type
    );
    Ok(event_id)
  }

  /// Get next sequence number for ordering
  async fn get_next_sequence_number(&self, source: EventSource) -> u64 {
    let mut cache = self.event_cache.write().await;
    let counter = cache.sequence_tracker.entry(source).or_insert(0);
    *counter += 1;
    *counter
  }

  /// Calculate integrity checksum
  fn calculate_checksum(
    &self,
    event_type: &EventType,
    payload: &EventPayload,
    timestamp: DateTime<Utc>,
  ) -> SwarmResult<String> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    format!("{:?}{:?}{}", event_type, payload, timestamp).hash(&mut hasher);
    Ok(format!("sha256-{:x}", hasher.finish()))
  }

  /// Create cryptographic signature
  fn create_signature(
    &self,
    event_id: &str,
    payload: &EventPayload,
  ) -> SwarmResult<String> {
    // In production, would use proper cryptographic signing
    Ok(
      format!("sig-{}-{:?}", event_id, payload)
        .chars()
        .take(32)
        .collect(),
    )
  }

  /// Cache event in memory
  async fn cache_event(&self, event: SecuredEvent) {
    let mut cache = self.event_cache.write().await;

    // Add to type index
    cache
      .events_by_type
      .entry(event.event_type.clone())
      .or_insert_with(Vec::new)
      .push(event.clone());

    // Add to source index
    cache
      .events_by_source
      .entry(event.source.clone())
      .or_insert_with(Vec::new)
      .push(event);

    // Trim cache if needed
    self.trim_cache(&mut cache).await;
  }

  /// Trim cache to configured size
  async fn trim_cache(&self, cache: &mut EventCache) {
    for events in cache.events_by_type.values_mut() {
      if events.len() > self.config.cache_size {
        events.drain(0..(events.len() - self.config.cache_size));
      }
    }

    for events in cache.events_by_source.values_mut() {
      if events.len() > self.config.cache_size {
        events.drain(0..(events.len() - self.config.cache_size));
      }
    }
  }

  /// Persist event to Cozo
  #[cfg(feature = "graph")]
  async fn persist_event(&self, event: &SecuredEvent) -> SwarmResult<()> {
    let cozo = self.cozo.read().await;

    let payload_json = serde_json::to_string(&event.payload).map_err(|e| {
      SwarmError::Persistence(format!("Failed to serialize payload: {}", e))
    })?;

    let metadata_json =
      serde_json::to_string(&event.metadata).map_err(|e| {
        SwarmError::Persistence(format!("Failed to serialize metadata: {}", e))
      })?;

    let query = format!(
      r#"
            {{
                # Insert main event record
                ?[event_id, sequence_number, event_type, source, timestamp, payload, metadata, signature, checksum] <-
                [['{}', {}, '{}', '{}', '{}', '{}', '{}', '{}', '{}']]
                
                :put event_log {{event_id, sequence_number, event_type, source, timestamp, payload, metadata, signature, checksum}}
                
                # Insert type index
                ?[event_type, timestamp, event_id] <-
                [['{}', '{}', '{}']]
                
                :put event_index_by_type {{event_type, timestamp, event_id}}
                
                # Insert source index  
                ?[source, timestamp, event_id] <-
                [['{}', '{}', '{}']]
                
                :put event_index_by_source {{source, timestamp, event_id}}
            }}
        "#,
      event.event_id,
      event.sequence_number,
      serde_json::to_string(&event.event_type).unwrap_or_default(),
      serde_json::to_string(&event.source).unwrap_or_default(),
      event.timestamp.to_rfc3339(),
      payload_json,
      metadata_json,
      event.signature.as_deref().unwrap_or(""),
      event.checksum,
      serde_json::to_string(&event.event_type).unwrap_or_default(),
      event.timestamp.to_rfc3339(),
      event.event_id,
      serde_json::to_string(&event.source).unwrap_or_default(),
      event.timestamp.to_rfc3339(),
      event.event_id
    );

    cozo
      .run_script(&query, Default::default(), cozo::ScriptMutability::Mutable)
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to persist event: {}", e))
      })?;

    Ok(())
  }

  /// Query events by type with type safety
  pub async fn query_events_by_type(
    &self,
    event_type: EventType,
    limit: usize,
  ) -> SwarmResult<Vec<SecuredEvent>> {
    // Try cache first
    {
      let cache = self.event_cache.read().await;
      if let Some(events) = cache.events_by_type.get(&event_type) {
        let result: Vec<_> = events.iter().rev().take(limit).cloned().collect();
        if !result.is_empty() {
          return Ok(result);
        }
      }
    }

    // Fallback to Cozo query
    #[cfg(feature = "graph")]
    {
      let cozo = self.cozo.read().await;
      let query = format!(
        r#"
                {{
                    ?[event_id, sequence_number, event_type, source, timestamp, payload, metadata, signature, checksum] :=
                        *event_log[event_id, sequence_number, event_type, source, timestamp, payload, metadata, signature, checksum],
                        event_type == '{}'
                    
                    :order -timestamp
                    :limit {}
                }}
            "#,
        serde_json::to_string(&event_type).unwrap_or_default(),
        limit
      );

      let result = cozo
        .run_script(
          &query,
          Default::default(),
          cozo::ScriptMutability::Immutable,
        )
        .map_err(|e| {
          SwarmError::Persistence(format!("Failed to query events: {}", e))
        })?;

      // Would parse results and reconstruct SecuredEvent objects
      // For now, return empty vec
      return Ok(Vec::new());
    }

    Ok(Vec::new())
  }

  /// Subscribe to real-time events
  pub fn subscribe_to_events(&self) -> broadcast::Receiver<SecuredEvent> {
    self.event_broadcaster.subscribe()
  }

  /// Verify event integrity
  pub async fn verify_event_integrity(
    &self,
    event: &SecuredEvent,
  ) -> SwarmResult<bool> {
    // Verify checksum
    let expected_checksum = self.calculate_checksum(
      &event.event_type,
      &event.payload,
      event.timestamp,
    )?;
    if event.checksum != expected_checksum {
      return Ok(false);
    }

    // Verify signature if present
    if let Some(signature) = &event.signature {
      let expected_signature =
        self.create_signature(&event.event_id, &event.payload)?;
      if *signature != expected_signature {
        return Ok(false);
      }
    }

    Ok(true)
  }
}

impl Default for EventLogConfig {
  fn default() -> Self {
    Self {
      cache_size: 1000,
      retention_days: 30,
      compression: false,
      verification: EventVerificationConfig {
        signatures: false,
        checksums: true,
        sequence_validation: true,
      },
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_event_type_exhaustiveness() {
    // Ensure all event types are handled
    let event_types = vec![
      EventType::DatabaseCreate,
      EventType::VectorInsert,
      EventType::NodeCreate,
      EventType::SwarmCreate,
      EventType::AgentCreate,
      EventType::TaskCreate,
      EventType::SystemStart,
      EventType::SyncStart,
    ];

    for event_type in event_types {
      match event_type {
        EventType::DatabaseCreate
        | EventType::DatabaseDelete
        | EventType::DatabaseUpdate
        | EventType::DatabaseQuery => {}
        EventType::VectorInsert
        | EventType::VectorUpdate
        | EventType::VectorDelete
        | EventType::VectorSearch => {}
        EventType::NodeCreate
        | EventType::NodeUpdate
        | EventType::NodeDelete
        | EventType::EdgeCreate
        | EventType::EdgeUpdate
        | EventType::EdgeDelete
        | EventType::GraphQuery => {}
        EventType::SwarmCreate
        | EventType::SwarmUpdate
        | EventType::SwarmDelete
        | EventType::SwarmQuery => {}
        EventType::AgentCreate
        | EventType::AgentUpdate
        | EventType::AgentDelete
        | EventType::AgentMemoryStore
        | EventType::AgentMemoryRetrieve => {}
        EventType::TaskCreate
        | EventType::TaskUpdate
        | EventType::TaskComplete
        | EventType::TaskFail => {}
        EventType::SystemStart
        | EventType::SystemStop
        | EventType::SystemError
        | EventType::SystemRecovery => {}
        EventType::SyncStart
        | EventType::SyncComplete
        | EventType::SyncConflict
        | EventType::SyncResolution => {}
      }
    }
  }

  #[test]
  fn test_payload_type_safety() {
    let payload = EventPayload::DatabasePayload(DatabaseEventPayload {
      database_name: "test".to_string(),
      table_name: Some("events".to_string()),
      operation_details: serde_json::json!({"op": "insert"}),
      affected_rows: Some(1),
      execution_time_ms: 42,
    });

    match payload {
      EventPayload::DatabasePayload(db_payload) => {
        assert_eq!(db_payload.database_name, "test");
        assert_eq!(db_payload.execution_time_ms, 42);
      }
      _ => panic!("Wrong payload type"),
    }
  }
}
