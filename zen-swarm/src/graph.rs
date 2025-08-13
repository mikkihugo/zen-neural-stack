//! Graph database and analysis with persistent swarm coordination
//!
//! Pure Rust graph database implementation using Cozo for complex relationship
//! analysis with persistent swarm states, central task coordination, and domain-specific
//! swarm federation. No CPU instruction dependencies.

use crate::SwarmError;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// High-performance graph database manager with Cozo (pure analytical database)
pub struct GraphDb {
  /// Cozo for analytical queries and persistence - handles all graph operations
  #[cfg(feature = "graph")]
  cozo: Option<Arc<RwLock<CozoInstance>>>,

  /// Persistent swarm states
  persistent_swarms: Arc<DashMap<String, PersistentSwarm>>,

  /// Central task coordination hub
  central_tasks: Arc<RwLock<CentralTaskHub>>,

  /// Domain-specific swarm registry
  domain_swarms: Arc<DashMap<String, DomainSwarm>>,
}

/// Cozo instance wrapper for analytical queries
#[cfg(feature = "graph")]
pub struct CozoInstance {
  pub db: cozo::DbInstance,
  pub session_id: String,
}

/// Graph node representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
  pub id: String,
  pub node_type: GraphNodeType,
  pub properties: HashMap<String, Value>,
  pub created_at: DateTime<Utc>,
  pub updated_at: DateTime<Utc>,
}

/// Graph edge representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
  pub id: String,
  pub edge_type: GraphEdgeType,
  pub properties: HashMap<String, Value>,
  pub weight: f64,
  pub created_at: DateTime<Utc>,
}

/// Types of graph nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphNodeType {
  SwarmNode,
  AgentNode,
  TaskNode,
  ServiceNode,
  DomainNode,
  CentralHub,
}

/// Types of graph edges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphEdgeType {
  SwarmMembership,
  TaskAssignment,
  ServiceDependency,
  Communication,
  Coordination,
  DataFlow,
}

/// Persistent swarm state for graph database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentSwarm {
  /// Unique swarm identifier
  pub swarm_id: String,

  /// Service type (service vs domain swarm)
  pub service_type: SwarmServiceType,

  /// Agents in the swarm
  pub agents: HashMap<String, PersistentAgent>,

  /// Persistent tasks
  pub persistent_tasks: Vec<PersistentTask>,

  /// Central coordination endpoints
  pub central_endpoints: Vec<CentralEndpoint>,

  /// Domain this swarm operates in
  pub domain: String,

  /// Swarm creation time
  pub creation_time: DateTime<Utc>,

  /// Last activity timestamp
  pub last_activity: DateTime<Utc>,

  /// Swarm status
  pub status: SwarmStatus,

  /// Performance metrics
  pub metrics: SwarmMetrics,
}

/// Service vs Domain swarm types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmServiceType {
  /// Service-specific swarm (handles one type of service)
  Service {
    service_name: String,
    version: String,
  },

  /// Domain-specific swarm (handles entire domain/business area)
  Domain {
    domain_name: String,
    capabilities: Vec<String>,
  },

  /// Hybrid swarm (can handle both service and domain tasks)
  Hybrid {
    primary_service: String,
    secondary_domains: Vec<String>,
  },
}

/// Persistent agent state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentAgent {
  /// Agent unique identifier
  pub agent_id: String,

  /// Agent type and capabilities
  pub agent_type: String,
  pub capabilities: Vec<String>,

  /// Current status
  pub status: AgentStatus,

  /// Performance history
  pub performance_history: Vec<AgentPerformanceRecord>,

  /// Learned patterns
  pub learned_patterns: HashMap<String, Value>,

  /// Communication endpoints
  pub endpoints: Vec<AgentEndpoint>,
}

/// Persistent task state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentTask {
  /// Task unique identifier
  pub task_id: String,

  /// Task description and requirements
  pub description: String,
  pub requirements: TaskRequirements,

  /// Task status and progress
  pub status: TaskStatus,
  pub progress_percentage: f32,

  /// Assigned agents
  pub assigned_agents: Vec<String>,

  /// Task dependencies
  pub dependencies: Vec<TaskDependency>,

  /// Results and outputs
  pub results: Option<TaskResults>,

  /// Timestamps
  pub created_at: DateTime<Utc>,
  pub started_at: Option<DateTime<Utc>>,
  pub completed_at: Option<DateTime<Utc>>,
}

/// Central coordination endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralEndpoint {
  /// Endpoint identifier
  pub endpoint_id: String,

  /// Endpoint type (RAG, FACT, coordination, etc.)
  pub endpoint_type: CentralEndpointType,

  /// Access URL or connection string
  pub url: String,

  /// Authentication requirements
  pub auth_config: Option<AuthConfig>,

  /// Capabilities provided
  pub capabilities: Vec<String>,

  /// Health status
  pub health_status: EndpointHealthStatus,
}

/// Agent status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStatus {
  Active,
  Idle,
  Busy,
  Learning,
  Offline,
  Error { message: String },
}

/// Task status enumeration  
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
  Pending,
  Running,
  Completed,
  Failed { error: String },
  Cancelled,
}

/// Swarm status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmStatus {
  Active,
  Idle,
  Scaling,
  Degraded,
  Offline,
}

/// Central endpoint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CentralEndpointType {
  Rag,
  Fact,
  Coordination,
  Storage,
  Analytics,
  Monitoring,
}

/// Endpoint health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EndpointHealthStatus {
  Healthy,
  Degraded,
  Unhealthy,
  Unknown,
}

/// Additional supporting types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPerformanceRecord {
  pub timestamp: DateTime<Utc>,
  pub task_id: String,
  pub execution_time_ms: u64,
  pub success_rate: f32,
  pub resource_usage: ResourceUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequirements {
  pub cpu_cores: u32,
  pub memory_mb: u32,
  pub storage_gb: u32,
  pub network_bandwidth_mbps: u32,
  pub special_capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDependency {
  pub dependency_type: DependencyType,
  pub target_task_id: String,
  pub condition: DependencyCondition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
  Sequential, // Must complete before this task starts
  Parallel,   // Must run in parallel
  Resource,   // Shares resources
  Data,       // Requires data output
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyCondition {
  Completed,
  Running,
  OutputAvailable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResults {
  pub output: Value,
  pub metrics: TaskExecutionMetrics,
  pub artifacts: Vec<TaskArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEndpoint {
  pub endpoint_type: AgentEndpointType,
  pub url: String,
  pub protocol: CommunicationProtocol,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentEndpointType {
  Http,
  WebSocket,
  Mcp,
  Custom { protocol: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationProtocol {
  Http,
  WebSocket,
  Grpc,
  JsonRpc,
  Custom { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
  pub auth_type: AuthType,
  pub credentials: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthType {
  None,
  ApiKey,
  Bearer,
  Basic,
  Custom { scheme: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
  pub cpu_percentage: f32,
  pub memory_mb: u32,
  pub network_bytes: u64,
  pub disk_io_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskExecutionMetrics {
  pub execution_time_ms: u64,
  pub cpu_time_ms: u64,
  pub memory_peak_mb: u32,
  pub network_bytes: u64,
  pub cache_hit_ratio: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskArtifact {
  pub artifact_type: ArtifactType,
  pub location: String,
  pub size_bytes: u64,
  pub checksum: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactType {
  File,
  Directory,
  Database,
  Model,
  Report,
  Visualization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmMetrics {
  pub total_tasks: u32,
  pub completed_tasks: u32,
  pub failed_tasks: u32,
  pub average_execution_time_ms: u64,
  pub success_rate: f32,
  pub resource_utilization: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralTaskHub {
  pub hub_id: String,
  pub active_coordinators: Vec<TaskCoordinator>,
  pub task_queues: HashMap<String, TaskQueue>,
  pub routing_rules: Vec<RoutingRule>,
  pub load_balancer: LoadBalancerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSwarm {
  pub domain_id: String,
  pub domain_name: String,
  pub swarm_ids: Vec<String>,
  pub coordination_strategy: CoordinationStrategy,
  pub performance_metrics: DomainMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskCoordinator {
  pub coordinator_id: String,
  pub specialization: Vec<String>,
  pub load_capacity: u32,
  pub current_load: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskQueue {
  pub queue_id: String,
  pub priority_level: TaskPriority,
  pub pending_tasks: Vec<String>,
  pub processing_tasks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskPriority {
  Low,
  Normal,
  High,
  Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRule {
  pub rule_id: String,
  pub condition: RoutingCondition,
  pub action: RoutingAction,
  pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingCondition {
  TaskType { task_type: String },
  SwarmCapacity { min_agents: u32 },
  Performance { min_success_rate: f32 },
  Geographic { region: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingAction {
  RouteToSwarm { swarm_id: String },
  LoadBalance { swarm_ids: Vec<String> },
  Replicate { replication_factor: u32 },
  Escalate { escalation_level: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerConfig {
  pub algorithm: LoadBalancingAlgorithm,
  pub health_check_interval_secs: u64,
  pub failure_threshold: u32,
  pub recovery_threshold: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
  RoundRobin,
  WeightedRoundRobin,
  LeastConnections,
  ResourceBased,
  PerformanceBased,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationStrategy {
  Centralized,
  Decentralized,
  Hierarchical,
  Federated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainMetrics {
  pub total_swarms: u32,
  pub active_swarms: u32,
  pub total_tasks_processed: u64,
  pub average_response_time_ms: u64,
  pub domain_success_rate: f32,
}

impl GraphDb {
  /// Create new graph database with pure Cozo implementation
  pub async fn new(config: GraphConfig) -> Result<Self, SwarmError> {
    Ok(Self {
      #[cfg(feature = "graph")]
      cozo: if config.enable_cozo {
        // Initialize Cozo database for graph analytics
        Some(Arc::new(RwLock::new(CozoInstance {
          db: cozo::DbInstance::new("mem", "", "").map_err(|e| {
            SwarmError::Graph(format!("Failed to initialize Cozo: {}", e))
          })?,
          session_id: Uuid::new_v4().to_string(),
        })))
      } else {
        None
      },
      persistent_swarms: Arc::new(DashMap::new()),
      central_tasks: Arc::new(RwLock::new(CentralTaskHub {
        hub_id: Uuid::new_v4().to_string(),
        active_coordinators: Vec::new(),
        task_queues: HashMap::new(),
        routing_rules: Vec::new(),
        load_balancer: LoadBalancerConfig {
          algorithm: LoadBalancingAlgorithm::ResourceBased,
          health_check_interval_secs: 30,
          failure_threshold: 3,
          recovery_threshold: 2,
        },
      })),
      domain_swarms: Arc::new(DashMap::new()),
    })
  }

  /// Add a node to the graph (stored in Cozo)
  #[cfg(feature = "graph")]
  pub async fn add_node(&self, node: GraphNode) -> Result<String, SwarmError> {
    if let Some(cozo) = &self.cozo {
      let cozo_guard = cozo.read().await;
      let node_data = serde_json::to_value(&node).map_err(|e| {
        SwarmError::Graph(format!("Failed to serialize node: {}", e))
      })?;

      // Store node in Cozo using CozoScript
      let query = format!(
        "?[id, type, properties, created_at] <- [['{}', '{}', '{}', '{}']]
                 :put nodes {{id, type, properties, created_at}}",
        node.id,
        serde_json::to_string(&node.node_type).unwrap_or_default(),
        serde_json::to_string(&node.properties).unwrap_or_default(),
        node.created_at.to_rfc3339()
      );

      cozo_guard
        .db
        .run_script(&query, Default::default(), cozo::ScriptMutability::Mutable)
        .map_err(|e| {
          SwarmError::Graph(format!("Failed to add node to Cozo: {}", e))
        })?;
    }
    Ok(node.id)
  }

  /// Add an edge to the graph (stored in Cozo)  
  #[cfg(feature = "graph")]
  pub async fn add_edge(
    &self,
    from_id: &str,
    to_id: &str,
    edge: GraphEdge,
  ) -> Result<String, SwarmError> {
    if let Some(cozo) = &self.cozo {
      let cozo_guard = cozo.read().await;

      let query = format!(
                "?[from_id, to_id, edge_id, edge_type, weight] <- [['{}', '{}', '{}', '{}', {}]]
                 :put edges {{from_id, to_id, edge_id, edge_type, weight}}",
                from_id, to_id, edge.id,
                serde_json::to_string(&edge.edge_type).unwrap_or_default(),
                edge.weight
            );

      cozo_guard
        .db
        .run_script(&query, Default::default(), cozo::ScriptMutability::Mutable)
        .map_err(|e| {
          SwarmError::Graph(format!("Failed to add edge to Cozo: {}", e))
        })?;
    }
    Ok(edge.id)
  }

  /// Create a persistent swarm
  pub async fn create_persistent_swarm(
    &self,
    swarm: PersistentSwarm,
  ) -> Result<String, SwarmError> {
    let swarm_id = swarm.swarm_id.clone();
    self.persistent_swarms.insert(swarm_id.clone(), swarm);

    // Add swarm as a graph node in Cozo
    let swarm_node = GraphNode {
      id: swarm_id.clone(),
      node_type: GraphNodeType::SwarmNode,
      properties: HashMap::new(),
      created_at: chrono::Utc::now(),
      updated_at: chrono::Utc::now(),
    };

    #[cfg(feature = "graph")]
    self.add_node(swarm_node).await?;

    Ok(swarm_id)
  }

  /// Get graph statistics
  pub async fn get_stats(&self) -> GraphStats {
    #[cfg(feature = "graph")]
    {
      if let Some(cozo) = &self.cozo {
        let cozo_guard = cozo.read().await;

        // Count nodes and edges using Cozo queries
        let node_count_result = cozo_guard
          .db
          .run_script(
            "?[count] := *nodes[id]; ?[count] := count(count)",
            Default::default(),
            cozo::ScriptMutability::Immutable,
          )
          .unwrap_or_default();

        let edge_count_result = cozo_guard
          .db
          .run_script(
            "?[count] := *edges[from_id]; ?[count] := count(count)",
            Default::default(),
            cozo::ScriptMutability::Immutable,
          )
          .unwrap_or_default();

        // Parse results (simplified for now)
        let total_nodes = 0; // Would parse from node_count_result
        let total_edges = 0; // Would parse from edge_count_result

        return GraphStats {
          total_nodes,
          total_edges,
          persistent_swarms: self.persistent_swarms.len(),
          domain_swarms: self.domain_swarms.len(),
        };
      }
    }

    // Fallback when graph feature is disabled
    GraphStats {
      total_nodes: 0,
      total_edges: 0,
      persistent_swarms: self.persistent_swarms.len(),
      domain_swarms: self.domain_swarms.len(),
    }
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
  pub database_path: String,
  pub enable_cozo: bool,
  pub max_nodes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
  pub total_nodes: usize,
  pub total_edges: usize,
  pub persistent_swarms: usize,
  pub domain_swarms: usize,
}

impl Default for GraphConfig {
  fn default() -> Self {
    Self {
      database_path: "./.zen/swarm/graph_data".to_string(),
      enable_cozo: true, // Enable by default - pure Rust, no CPU issues
      max_nodes: 1_000_000,
    }
  }
}
