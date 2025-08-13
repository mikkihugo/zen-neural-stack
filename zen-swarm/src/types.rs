//! Comprehensive type safety system for zen-swarm
//!
//! This module provides strongly-typed alternatives to generic types like
//! `serde_json::Value`, `String`, and `HashMap` throughout the codebase.
//! Eliminates runtime errors by catching type mismatches at compile time.

use crate::events::SwarmPerformanceMetrics;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

// ============================================================================
// ID TYPES - Replace generic String IDs with type-safe alternatives
// ============================================================================

/// Type-safe swarm identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SwarmId(String);

/// Type-safe agent identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(String);

/// Type-safe task identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(String);

/// Type-safe memory identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryId(String);

/// Type-safe node identifier for graphs
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(String);

/// Type-safe edge identifier for graphs
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeId(String);

/// Type-safe vector identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VectorId(String);

/// Type-safe session identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(String);

/// Type-safe correlation identifier for tracing
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CorrelationId(String);

// ID Generation and Validation
macro_rules! impl_id_type {
  ($id_type:ty) => {
    impl $id_type {
      /// Create a new random ID
      pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
      }

      /// Create from existing string (with validation)
      pub fn from_string(s: String) -> Result<Self, TypedError> {
        if s.is_empty() {
          return Err(TypedError::InvalidId("ID cannot be empty".to_string()));
        }
        if s.len() > 255 {
          return Err(TypedError::InvalidId("ID too long".to_string()));
        }
        Ok(Self(s))
      }

      /// Get the underlying string
      pub fn as_str(&self) -> &str {
        &self.0
      }

      /// Convert to string
      pub fn into_string(self) -> String {
        self.0
      }

      /// Validate ID format
      pub fn is_valid(&self) -> bool {
        !self.0.is_empty() && self.0.len() <= 255
      }
    }

    impl fmt::Display for $id_type {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
      }
    }

    impl Default for $id_type {
      fn default() -> Self {
        Self::new()
      }
    }

    impl From<Uuid> for $id_type {
      fn from(uuid: Uuid) -> Self {
        Self(uuid.to_string())
      }
    }
  };
}

impl_id_type!(SwarmId);
impl_id_type!(AgentId);
impl_id_type!(TaskId);
impl_id_type!(MemoryId);
impl_id_type!(NodeId);
impl_id_type!(EdgeId);
impl_id_type!(VectorId);
impl_id_type!(SessionId);
impl_id_type!(CorrelationId);

// ============================================================================
// CONFIGURATION TYPES - Replace generic Value/HashMap with structured types
// ============================================================================

/// Type-safe swarm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfiguration {
  pub topology: SwarmTopology,
  pub max_agents: AgentCount,
  pub coordination_strategy: CoordinationStrategy,
  pub performance_settings: PerformanceSettings,
  pub security_settings: SecuritySettings,
  pub resource_limits: ResourceLimits,
}

/// Swarm topology with compile-time validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmTopology {
  Mesh { max_connections_per_agent: u32 },
  Hierarchical { levels: u32, branching_factor: u32 },
  Ring { bidirectional: bool },
  Star { central_agent_capacity: u32 },
  Custom { configuration: CustomTopologyConfig },
}

/// Agent count with validation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AgentCount(u32);

impl AgentCount {
  pub fn new(count: u32) -> Result<Self, TypedError> {
    if count == 0 {
      return Err(TypedError::InvalidConfiguration(
        "Agent count cannot be zero".to_string(),
      ));
    }
    if count > 10000 {
      return Err(TypedError::InvalidConfiguration(
        "Agent count too high".to_string(),
      ));
    }
    Ok(Self(count))
  }

  pub fn get(&self) -> u32 {
    self.0
  }
}

/// Coordination strategies with specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationStrategy {
  Centralized {
    coordinator_agent: AgentId,
    heartbeat_interval_ms: u64,
  },
  Decentralized {
    consensus_threshold: f32,
    election_timeout_ms: u64,
  },
  Hierarchical {
    layer_count: u32,
    delegation_rules: DelegationRules,
  },
  Hybrid {
    primary_strategy: Box<CoordinationStrategy>,
    fallback_strategy: Box<CoordinationStrategy>,
    switch_threshold: f32,
  },
}

/// Performance settings with typed constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSettings {
  pub max_concurrent_tasks: TaskCount,
  pub task_timeout: Duration,
  pub memory_limit: MemoryLimit,
  pub cpu_limit: CpuLimit,
  pub batch_size: BatchSize,
  pub queue_size: QueueSize,
}

/// Security settings with type safety
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecuritySettings {
  pub authentication: AuthenticationConfig,
  pub authorization: AuthorizationConfig,
  pub encryption: EncryptionConfig,
  pub audit_level: AuditLevel,
}

/// Resource limits with validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
  pub max_memory_mb: MemoryLimit,
  pub max_cpu_percentage: CpuLimit,
  pub max_network_bandwidth_mbps: NetworkLimit,
  pub max_storage_gb: StorageLimit,
  pub max_file_handles: FileHandleLimit,
}

// ============================================================================
// MEASUREMENT TYPES - Replace primitive numbers with domain-specific types
// ============================================================================

/// Memory measurement with units
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct MemoryLimit(u64); // MB

impl MemoryLimit {
  pub fn megabytes(mb: u64) -> Result<Self, TypedError> {
    if mb > 1024 * 1024 {
      // 1TB limit
      return Err(TypedError::InvalidConfiguration(
        "Memory limit too high".to_string(),
      ));
    }
    Ok(Self(mb))
  }

  pub fn gigabytes(gb: u64) -> Result<Self, TypedError> {
    Self::megabytes(gb * 1024)
  }

  pub fn mb(&self) -> u64 {
    self.0
  }
  pub fn bytes(&self) -> u64 {
    self.0 * 1024 * 1024
  }
}

/// CPU limit as percentage
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CpuLimit(f32); // 0.0 to 100.0

impl CpuLimit {
  pub fn percentage(pct: f32) -> Result<Self, TypedError> {
    if pct < 0.0 || pct > 100.0 {
      return Err(TypedError::InvalidConfiguration(
        "CPU percentage must be 0-100".to_string(),
      ));
    }
    Ok(Self(pct))
  }

  pub fn get(&self) -> f32 {
    self.0
  }
}

/// Duration with validation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Duration {
  milliseconds: u64,
}

impl Duration {
  pub fn milliseconds(ms: u64) -> Self {
    Self { milliseconds: ms }
  }

  pub fn seconds(s: u64) -> Self {
    Self {
      milliseconds: s * 1000,
    }
  }

  pub fn minutes(m: u64) -> Self {
    Self {
      milliseconds: m * 60 * 1000,
    }
  }

  pub fn ms(&self) -> u64 {
    self.milliseconds
  }
  pub fn as_chrono_duration(&self) -> chrono::Duration {
    chrono::Duration::milliseconds(self.milliseconds as i64)
  }
}

/// Task count with validation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TaskCount(u32);

impl TaskCount {
  pub fn new(count: u32) -> Result<Self, TypedError> {
    if count > 100000 {
      return Err(TypedError::InvalidConfiguration(
        "Task count too high".to_string(),
      ));
    }
    Ok(Self(count))
  }

  pub fn get(&self) -> u32 {
    self.0
  }
}

/// Batch size with validation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BatchSize(u32);

impl BatchSize {
  pub fn new(size: u32) -> Result<Self, TypedError> {
    if size == 0 {
      return Err(TypedError::InvalidConfiguration(
        "Batch size cannot be zero".to_string(),
      ));
    }
    if size > 10000 {
      return Err(TypedError::InvalidConfiguration(
        "Batch size too large".to_string(),
      ));
    }
    Ok(Self(size))
  }

  pub fn get(&self) -> u32 {
    self.0
  }
}

/// Queue size with validation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QueueSize(u32);

impl QueueSize {
  pub fn new(size: u32) -> Result<Self, TypedError> {
    if size > 1000000 {
      return Err(TypedError::InvalidConfiguration(
        "Queue size too large".to_string(),
      ));
    }
    Ok(Self(size))
  }

  pub fn get(&self) -> u32 {
    self.0
  }
}

/// Network bandwidth limit
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NetworkLimit(u32); // Mbps

/// Storage limit
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StorageLimit(u64); // GB

/// File handle limit
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FileHandleLimit(u32);

// ============================================================================
// STATUS AND STATE TYPES - Replace string-based status with typed enums
// ============================================================================

/// Comprehensive agent status with detailed substates
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TypedAgentStatus {
  Initializing {
    progress_percentage: u8,
    current_step: InitializationStep,
  },
  Active {
    current_task: Option<TaskId>,
    utilization_percentage: u8,
    last_heartbeat: DateTime<Utc>,
  },
  Idle {
    idle_since: DateTime<Utc>,
    ready_for_tasks: bool,
  },
  Busy {
    task_id: TaskId,
    estimated_completion: DateTime<Utc>,
    can_interrupt: bool,
  },
  Learning {
    learning_type: LearningType,
    progress_percentage: u8,
    estimated_duration: Duration,
  },
  Suspended {
    reason: SuspensionReason,
    suspended_at: DateTime<Utc>,
    resume_conditions: Vec<ResumeCondition>,
  },
  Error {
    error_type: AgentErrorType,
    error_message: String,
    recovery_actions: Vec<RecoveryAction>,
    is_recoverable: bool,
  },
  Terminating {
    reason: TerminationReason,
    cleanup_progress: u8,
  },
  Terminated {
    termination_time: DateTime<Utc>,
    final_statistics: AgentFinalStats,
  },
}

/// Task status with progress tracking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TypedTaskStatus {
  Created {
    created_at: DateTime<Utc>,
    requirements_met: bool,
  },
  Queued {
    queue_position: u32,
    estimated_start_time: DateTime<Utc>,
  },
  Assigned {
    agent_id: AgentId,
    assigned_at: DateTime<Utc>,
  },
  Running {
    agent_id: AgentId,
    progress_percentage: u8,
    current_step: TaskStep,
    estimated_completion: DateTime<Utc>,
  },
  Suspended {
    reason: TaskSuspensionReason,
    suspended_at: DateTime<Utc>,
  },
  Completed {
    result: TaskResult,
    completed_at: DateTime<Utc>,
    execution_statistics: TaskExecutionStats,
  },
  Failed {
    failure_reason: TaskFailureReason,
    failed_at: DateTime<Utc>,
    retry_count: u32,
    is_retryable: bool,
  },
  Cancelled {
    cancelled_at: DateTime<Utc>,
    cancellation_reason: CancellationReason,
  },
}

/// Swarm status with health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypedSwarmStatus {
  Initializing {
    agents_initialized: u32,
    total_agents: u32,
    initialization_phase: InitializationPhase,
  },
  Active {
    healthy_agents: u32,
    total_agents: u32,
    active_tasks: u32,
    performance_score: f32,
    last_health_check: DateTime<Utc>,
  },
  Degraded {
    healthy_agents: u32,
    total_agents: u32,
    degradation_reasons: Vec<DegradationReason>,
    recovery_in_progress: bool,
  },
  Scaling {
    current_agents: u32,
    target_agents: u32,
    scaling_direction: ScalingDirection,
    estimated_completion: DateTime<Utc>,
  },
  Maintenance {
    maintenance_type: MaintenanceType,
    affected_components: Vec<String>,
    estimated_duration: Duration,
  },
  Terminating {
    termination_reason: SwarmTerminationReason,
    agents_terminated: u32,
    total_agents: u32,
  },
  Terminated {
    terminated_at: DateTime<Utc>,
    final_statistics: SwarmFinalStats,
  },
}

// ============================================================================
// SUPPORTING ENUMS AND STRUCTS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InitializationStep {
  LoadingConfiguration,
  EstablishingConnections,
  InitializingMemory,
  RunningHealthChecks,
  RegisteringWithSwarm,
  Ready,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LearningType {
  SkillAcquisition { skill_name: String },
  PatternRecognition { domain: String },
  PerformanceOptimization,
  AdaptiveStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SuspensionReason {
  ResourceStarvation { resource_type: String },
  SystemMaintenance,
  UserRequest,
  AutomaticOptimization,
  ErrorRecovery,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResumeCondition {
  ResourceAvailable { resource_type: String },
  MaintenanceComplete,
  UserApproval,
  ErrorResolved,
  TimeElapsed { duration: Duration },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentErrorType {
  CommunicationFailure,
  ResourceExhaustion,
  TaskExecutionError,
  ConfigurationError,
  NetworkPartition,
  DatabaseConnectionLost,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecoveryAction {
  Restart,
  ResetConfiguration,
  ReestablishConnections,
  ClearCache,
  RequestUserIntervention,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TerminationReason {
  UserRequest,
  SystemShutdown,
  ResourceExhaustion,
  UnrecoverableError,
  TaskCompletion,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentFinalStats {
  pub total_runtime: Duration,
  pub tasks_completed: u32,
  pub tasks_failed: u32,
  pub avg_task_duration: Duration,
  pub peak_memory_usage: MemoryLimit,
  pub total_cpu_time: Duration,
}

// Task-related types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskStep {
  pub step_name: String,
  pub step_number: u32,
  pub total_steps: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskResult {
  Success {
    output: TaskOutput,
  },
  PartialSuccess {
    output: TaskOutput,
    warnings: Vec<String>,
  },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskOutput {
  TextOutput(String),
  StructuredData(serde_json::Value),
  FileOutput { path: String, size_bytes: u64 },
  MultipleOutputs(Vec<TaskOutput>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskExecutionStats {
  pub execution_time: Duration,
  pub cpu_time: Duration,
  pub memory_used: MemoryLimit,
  pub network_io_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskFailureReason {
  TimeoutExceeded,
  ResourceUnavailable { resource: String },
  DependencyFailure { dependency: TaskId },
  ValidationError { field: String, message: String },
  ExternalServiceFailure { service: String },
  InternalError { error: String },
}

// Configuration types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomTopologyConfig {
  pub node_connections: HashMap<AgentId, Vec<AgentId>>,
  pub connection_weights: HashMap<(AgentId, AgentId), f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegationRules {
  pub max_delegation_depth: u32,
  pub delegation_criteria: Vec<DelegationCriterion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DelegationCriterion {
  TaskComplexity { threshold: f32 },
  AgentCapability { required_capability: String },
  ResourceAvailability { resource_type: String, minimum: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
  pub method: AuthMethod,
  pub token_expiry: Duration,
  pub multi_factor_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthMethod {
  None,
  Token { algorithm: String },
  Certificate { ca_path: String },
  OAuth { provider: String, client_id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationConfig {
  pub model: AuthorizationModel,
  pub default_permissions: Vec<Permission>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthorizationModel {
  RoleBased {
    roles: HashMap<String, Vec<Permission>>,
  },
  AttributeBased {
    policies: Vec<AbacPolicy>,
  },
  Simple {
    allowed_operations: Vec<String>,
  },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Permission {
  Read { resource: String },
  Write { resource: String },
  Execute { operation: String },
  Admin,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbacPolicy {
  pub name: String,
  pub subject_attributes: HashMap<String, String>,
  pub resource_attributes: HashMap<String, String>,
  pub action: String,
  pub effect: PolicyEffect,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyEffect {
  Allow,
  Deny,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
  pub at_rest: bool,
  pub in_transit: bool,
  pub algorithm: EncryptionAlgorithm,
  pub key_rotation_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
  Aes256,
  ChaCha20Poly1305,
  Custom {
    name: String,
    parameters: HashMap<String, String>,
  },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditLevel {
  None,
  Minimal,       // Errors and security events only
  Standard,      // Standard operations
  Comprehensive, // All operations
  Debug,         // Everything including debug info
}

// Additional swarm types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InitializationPhase {
  CreatingAgents,
  EstablishingTopology,
  ConfiguringCommunication,
  RunningHealthChecks,
  Ready,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DegradationReason {
  AgentFailure { agent_id: AgentId },
  ResourceExhaustion { resource: String },
  NetworkPartition,
  HighLatency { threshold_exceeded_ms: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingDirection {
  Up { reason: String },
  Down { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaintenanceType {
  Planned { scheduled_at: DateTime<Utc> },
  Emergency { severity: MaintenanceSeverity },
  Automatic { trigger: AutoMaintenanceTrigger },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaintenanceSeverity {
  Low,
  Medium,
  High,
  Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutoMaintenanceTrigger {
  PerformanceDegradation,
  ResourceLeakage,
  ConfigurationDrift,
  SecurityUpdate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmTerminationReason {
  UserRequest,
  TaskCompletion,
  ResourceExhaustion,
  CriticalFailure,
  ScheduledShutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmFinalStats {
  pub total_runtime: Duration,
  pub peak_agent_count: u32,
  pub total_tasks_processed: u64,
  pub success_rate: f32,
  pub resource_efficiency: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskSuspensionReason {
  ResourceWait,
  DependencyWait { dependency: TaskId },
  UserPause,
  SystemMaintenance,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CancellationReason {
  UserRequest,
  Timeout,
  ResourceUnavailable,
  SystemShutdown,
  SupersededBy { task_id: TaskId },
}

// ============================================================================
// ERROR TYPES - Comprehensive error handling
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypedError {
  InvalidId(String),
  InvalidConfiguration(String),
  ValidationFailed {
    field: String,
    reason: String,
  },
  OutOfRange {
    value: String,
    min: String,
    max: String,
  },
  TypeMismatch {
    expected: String,
    actual: String,
  },
}

impl fmt::Display for TypedError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      TypedError::InvalidId(msg) => write!(f, "Invalid ID: {}", msg),
      TypedError::InvalidConfiguration(msg) => {
        write!(f, "Invalid configuration: {}", msg)
      }
      TypedError::ValidationFailed { field, reason } => {
        write!(f, "Validation failed for {}: {}", field, reason)
      }
      TypedError::OutOfRange { value, min, max } => {
        write!(f, "Value {} out of range [{}, {}]", value, min, max)
      }
      TypedError::TypeMismatch { expected, actual } => {
        write!(f, "Type mismatch: expected {}, got {}", expected, actual)
      }
    }
  }
}

impl std::error::Error for TypedError {}

// ============================================================================
// TYPE ALIASES FOR COMMON PATTERNS
// ============================================================================

/// Type-safe result with comprehensive error information
pub type TypedResult<T> = Result<T, TypedError>;

/// Type-safe collections with domain-specific keys
pub type AgentCollection<T> = HashMap<AgentId, T>;
pub type TaskCollection<T> = HashMap<TaskId, T>;
pub type SwarmCollection<T> = HashMap<SwarmId, T>;

/// Type-safe metadata that replaces generic Value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypedMetadata {
  AgentMetadata(AgentMetadata),
  TaskMetadata(TaskMetadata),
  SwarmMetadata(SwarmMetadata),
  SystemMetadata(SystemMetadata),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetadata {
  pub capabilities: Vec<String>,
  pub performance_metrics: AgentPerformanceMetrics,
  pub configuration: AgentSpecificConfig,
  pub relationships: Vec<AgentRelationship>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetadata {
  pub priority: TaskPriority,
  pub requirements: TaskRequirements,
  pub constraints: TaskConstraints,
  pub dependencies: Vec<TaskDependency>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmMetadata {
  pub creation_context: SwarmCreationContext,
  pub performance_history: Vec<SwarmPerformanceSnapshot>,
  pub configuration_history: Vec<ConfigurationChange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetadata {
  pub version: String,
  pub build_info: BuildInfo,
  pub runtime_info: RuntimeInfo,
  pub feature_flags: HashMap<String, bool>,
}

// Supporting metadata types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPerformanceMetrics {
  pub cpu_usage_history: Vec<(DateTime<Utc>, f32)>,
  pub memory_usage_history: Vec<(DateTime<Utc>, u64)>,
  pub task_completion_rate: f32,
  pub error_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSpecificConfig {
  pub max_concurrent_tasks: TaskCount,
  pub specializations: Vec<String>,
  pub communication_preferences: CommunicationPreferences,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRelationship {
  pub related_agent: AgentId,
  pub relationship_type: RelationshipType,
  pub strength: f32,
  pub established_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
  Peer,
  Subordinate,
  Supervisor,
  Collaborator,
  Competitor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationPreferences {
  pub preferred_protocols: Vec<String>,
  pub max_message_size_bytes: u32,
  pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskPriority {
  Critical { deadline: DateTime<Utc> },
  High { target_completion: DateTime<Utc> },
  Normal,
  Low { can_defer_until: DateTime<Utc> },
  Background,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequirements {
  pub minimum_agent_capabilities: Vec<String>,
  pub resource_requirements: ResourceRequirements,
  pub isolation_level: IsolationLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
  pub cpu_cores: Option<u32>,
  pub memory_mb: Option<MemoryLimit>,
  pub storage_gb: Option<StorageLimit>,
  pub network_bandwidth_mbps: Option<NetworkLimit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationLevel {
  None,
  Process,
  Container,
  VirtualMachine,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConstraints {
  pub max_duration: Duration,
  pub allowed_failure_rate: f32,
  pub required_completion_percentage: f32,
  pub geographic_constraints: Option<GeographicConstraints>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeographicConstraints {
  pub allowed_regions: Vec<String>,
  pub preferred_region: Option<String>,
  pub data_residency_requirements: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDependency {
  pub dependency_task: TaskId,
  pub dependency_type: DependencyType,
  pub condition: DependencyCondition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
  Sequential,
  Parallel,
  Resource,
  Data,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyCondition {
  MustComplete,
  MustStart,
  ResourceAvailable,
  DataAvailable { data_key: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmCreationContext {
  pub created_by: String,
  pub creation_reason: String,
  pub initial_configuration: SwarmConfiguration,
  pub creation_timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmPerformanceSnapshot {
  pub timestamp: DateTime<Utc>,
  pub metrics: SwarmPerformanceMetrics,
  pub health_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationChange {
  pub timestamp: DateTime<Utc>,
  pub changed_by: String,
  pub change_type: ConfigurationChangeType,
  pub old_value: Option<String>,
  pub new_value: String,
  pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigurationChangeType {
  AgentAdded,
  AgentRemoved,
  AgentModified,
  TopologyChanged,
  PerformanceSettingsChanged,
  SecuritySettingsChanged,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInfo {
  pub version: String,
  pub build_timestamp: DateTime<Utc>,
  pub git_commit: String,
  pub build_environment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeInfo {
  pub start_time: DateTime<Utc>,
  pub rust_version: String,
  pub platform: String,
  pub available_features: Vec<String>,
}

// ============================================================================
// UTILITY FUNCTIONS AND TRAITS
// ============================================================================

/// Trait for types that can be validated
pub trait Validatable {
  type Error;

  fn validate(&self) -> Result<(), Self::Error>;
  fn is_valid(&self) -> bool {
    self.validate().is_ok()
  }
}

/// Trait for types that have type-safe conversion
pub trait TypedConversion<T> {
  type Error;

  fn try_from_untyped(value: T) -> Result<Self, Self::Error>
  where
    Self: Sized;
  fn into_untyped(self) -> T;
}

// Implement validation for key types
impl Validatable for SwarmConfiguration {
  type Error = TypedError;

  fn validate(&self) -> Result<(), Self::Error> {
    // Validate agent count
    if self.max_agents.get() == 0 {
      return Err(TypedError::InvalidConfiguration(
        "Max agents cannot be zero".to_string(),
      ));
    }

    // Validate topology constraints
    match &self.topology {
      SwarmTopology::Hierarchical {
        levels,
        branching_factor,
      } => {
        if *levels == 0 || *branching_factor == 0 {
          return Err(TypedError::InvalidConfiguration("Hierarchical topology requires positive levels and branching factor".to_string()));
        }
      }
      _ => {} // Other topologies validated in their constructors
    }

    Ok(())
  }
}

impl Validatable for PerformanceSettings {
  type Error = TypedError;

  fn validate(&self) -> Result<(), Self::Error> {
    if self.max_concurrent_tasks.get() == 0 {
      return Err(TypedError::InvalidConfiguration(
        "Max concurrent tasks cannot be zero".to_string(),
      ));
    }

    if self.task_timeout.ms() == 0 {
      return Err(TypedError::InvalidConfiguration(
        "Task timeout cannot be zero".to_string(),
      ));
    }

    Ok(())
  }
}
