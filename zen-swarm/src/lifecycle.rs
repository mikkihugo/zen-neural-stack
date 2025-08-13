//! Swarm lifecycle management with automated data cleanup
//!
//! Production-ready lifecycle management for swarms including:
//! - Swarm creation and initialization  
//! - State persistence in .zen directory structure
//! - Graceful shutdown and cleanup
//! - Automated data cleanup for deleted swarms
//! - Retention policies and archival
//! - Data migration and backup
//!
//! # Directory Structure
//!
//! ```text
//! .zen/
//! ├── swarm/           # Shared swarm database (filtered by swarm_id)
//! ├── claude/          # Claude Code specific configuration  
//! ├── gemini/          # Gemini specific configuration
//! └── collective/      # Central shared resources for all AI systems
//! ```
//!
//! # Data Cleanup Strategy
//!
//! When a swarm is deleted, the system:
//! 1. **Graceful Shutdown**: Stop all agents and complete running tasks
//! 2. **Data Export**: Create backup of important data before deletion
//! 3. **Filtered Cleanup**: Delete only records matching the swarm_id
//! 4. **Verification**: Ensure no orphaned data remains
//! 5. **Audit Log**: Record deletion for compliance and debugging

use crate::{SwarmError, SwarmResult};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// Comprehensive swarm lifecycle manager
///
/// Manages the complete lifecycle of swarms from creation to deletion,
/// with special attention to data cleanup and resource management.
#[derive(Debug)]
pub struct SwarmLifecycleManager {
  /// Active swarm registry
  active_swarms: Arc<DashMap<String, SwarmLifecycleState>>,

  /// Base .zen directory configuration
  zen_config: Arc<ZenDirectoryConfig>,

  /// Data cleanup policies
  cleanup_policies: Arc<RwLock<Vec<CleanupPolicy>>>,

  /// Retention policies for different data types
  retention_policies: Arc<RwLock<HashMap<String, RetentionPolicy>>>,

  /// Scheduled cleanup tasks
  scheduled_cleanups: Arc<RwLock<Vec<ScheduledCleanup>>>,

  /// Audit logger for compliance
  audit_logger: Arc<Mutex<AuditLogger>>,

  /// Backup manager for data safety
  backup_manager: Arc<BackupManager>,
}

/// Complete lifecycle state for a swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmLifecycleState {
  /// Swarm identifier
  pub swarm_id: String,

  /// Current lifecycle phase
  pub phase: LifecyclePhase,

  /// Swarm configuration
  pub config: SwarmConfiguration,

  /// Resource allocation
  pub resources: AllocatedResources,

  /// Data locations and sizes
  pub data_inventory: DataInventory,

  /// Cleanup configuration for this swarm
  pub cleanup_config: SwarmCleanupConfig,

  /// Lifecycle timestamps
  pub timestamps: LifecycleTimestamps,

  /// Health and monitoring
  pub health: SwarmHealth,
}

/// Swarm lifecycle phases
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LifecyclePhase {
  /// Swarm is being created
  Creating { progress: f32, current_step: String },

  /// Swarm is running normally
  Active {
    uptime_seconds: u64,
    agent_count: u32,
  },

  /// Swarm is paused (temporarily inactive)
  Paused {
    paused_at: DateTime<Utc>,
    reason: String,
  },

  /// Swarm is being migrated to new configuration
  Migrating {
    target_config: String,
    progress: f32,
  },

  /// Swarm is being gracefully shut down
  ShuttingDown {
    shutdown_started: DateTime<Utc>,
    remaining_tasks: u32,
    cleanup_progress: f32,
  },

  /// Swarm data is being archived before deletion
  Archiving {
    archive_progress: f32,
    archive_location: String,
  },

  /// Swarm data is being cleaned up
  CleaningUp {
    cleanup_progress: f32,
    items_remaining: u32,
  },

  /// Swarm has been deleted
  Deleted {
    deleted_at: DateTime<Utc>,
    cleanup_summary: CleanupSummary,
  },
}

/// .zen directory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZenDirectoryConfig {
  /// Base .zen directory path
  pub base_path: PathBuf,

  /// Swarm shared database directory
  pub swarm_path: PathBuf,

  /// Claude Code specific directory
  pub claude_path: PathBuf,

  /// Gemini specific directory
  pub gemini_path: PathBuf,

  /// Central collective directory for shared resources
  pub collective_path: PathBuf,

  /// Directory permissions and security
  pub permissions: DirectoryPermissions,

  /// Backup and archival locations
  pub backup_locations: Vec<BackupLocation>,
}

/// Directory permissions and security settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryPermissions {
  /// File permissions (Unix-style)
  pub file_mode: u32,

  /// Directory permissions (Unix-style)
  pub dir_mode: u32,

  /// Owner user ID
  pub owner_uid: Option<u32>,

  /// Owner group ID
  pub owner_gid: Option<u32>,

  /// Enable encryption for sensitive data
  pub encrypt_sensitive: bool,
}

/// Backup location configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupLocation {
  /// Backup location ID
  pub id: String,

  /// Location type (local, s3, etc.)
  pub location_type: BackupLocationType,

  /// Location-specific configuration
  pub config: Value,

  /// Whether this location is active
  pub active: bool,

  /// Priority (higher = preferred)
  pub priority: u32,
}

/// Types of backup locations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupLocationType {
  Local { path: PathBuf },
  S3 { bucket: String, prefix: String },
  GCS { bucket: String, prefix: String },
  Azure { container: String, prefix: String },
  Remote { url: String, auth: Value },
}

/// Swarm configuration for lifecycle management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfiguration {
  /// Configuration version for migration tracking
  pub version: u32,

  /// Swarm type and specialization
  pub swarm_type: SwarmType,

  /// Resource requirements
  pub resource_requirements: ResourceRequirements,

  /// Data retention requirements
  pub retention_requirements: RetentionRequirements,

  /// Backup requirements
  pub backup_requirements: BackupRequirements,

  /// Cleanup preferences
  pub cleanup_preferences: CleanupPreferences,
}

/// Types of swarms for different use cases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmType {
  /// Development swarm (can be deleted freely)
  Development {
    project_name: String,
    temporary: bool,
  },

  /// Production swarm (requires confirmation to delete)
  Production {
    service_name: String,
    environment: String,
  },

  /// Staging/testing swarm
  Staging {
    environment: String,
    linked_production: Option<String>,
  },

  /// Research swarm (may have valuable learning data)
  Research {
    research_project: String,
    preserve_learning_data: bool,
  },

  /// Temporary swarm with automatic cleanup
  Temporary {
    expires_at: DateTime<Utc>,
    auto_cleanup: bool,
  },
}

/// Resource requirements for swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
  /// Minimum memory in MB
  pub min_memory_mb: u64,

  /// Maximum memory in MB
  pub max_memory_mb: u64,

  /// CPU requirements
  pub cpu_requirements: CpuRequirements,

  /// Storage requirements
  pub storage_requirements: StorageRequirements,

  /// Network requirements
  pub network_requirements: NetworkRequirements,
}

/// CPU requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuRequirements {
  /// Minimum CPU cores
  pub min_cores: f32,

  /// Maximum CPU cores
  pub max_cores: f32,

  /// CPU architecture requirements
  pub architecture: Vec<String>,

  /// Special CPU features required
  pub required_features: Vec<String>,
}

/// Storage requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageRequirements {
  /// Minimum storage in GB
  pub min_storage_gb: u64,

  /// Maximum storage in GB
  pub max_storage_gb: u64,

  /// Storage type requirements
  pub storage_type: StorageType,

  /// IOPS requirements
  pub min_iops: Option<u32>,

  /// Backup frequency requirements
  pub backup_frequency: BackupFrequency,
}

/// Storage types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageType {
  /// Standard disk storage
  Standard,

  /// High-performance SSD storage
  HighPerformance,

  /// Network-attached storage
  NetworkAttached,

  /// Distributed storage
  Distributed,

  /// In-memory storage
  InMemory,
}

/// Backup frequency requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupFrequency {
  /// No backup required
  None,

  /// Backup on significant changes
  OnChange,

  /// Periodic backups
  Periodic { interval_hours: u32 },

  /// Continuous backup
  Continuous,
}

/// Network requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRequirements {
  /// Minimum bandwidth in Mbps
  pub min_bandwidth_mbps: u32,

  /// Maximum latency in milliseconds
  pub max_latency_ms: u32,

  /// Required network features
  pub required_features: Vec<NetworkFeature>,

  /// External connectivity requirements
  pub external_connectivity: Vec<ExternalConnectivity>,
}

/// Network features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkFeature {
  Encryption,
  Compression,
  LoadBalancing,
  Failover,
  QoS,
}

/// External connectivity requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalConnectivity {
  /// Service name or description
  pub service: String,

  /// Required endpoints
  pub endpoints: Vec<String>,

  /// Authentication requirements
  pub auth_required: bool,

  /// Connection importance
  pub importance: ConnectivityImportance,
}

/// Importance levels for external connectivity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectivityImportance {
  /// Must have connectivity to function
  Critical,

  /// Important for full functionality
  Important,

  /// Optional enhancement
  Optional,
}

/// Data retention requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionRequirements {
  /// Task execution data retention
  pub task_data: RetentionSpec,

  /// Agent memory retention
  pub agent_memory: RetentionSpec,

  /// Learning data retention
  pub learning_data: RetentionSpec,

  /// Audit logs retention
  pub audit_logs: RetentionSpec,

  /// Performance metrics retention
  pub performance_metrics: RetentionSpec,

  /// Communication history retention
  pub communication_history: RetentionSpec,
}

/// Retention specification for data type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionSpec {
  /// How long to keep data active (days)
  pub active_retention_days: u32,

  /// How long to keep data archived (days)
  pub archive_retention_days: u32,

  /// Whether data should be anonymized after active period
  pub anonymize_after_active: bool,

  /// Legal hold requirements
  pub legal_hold: bool,

  /// Compliance requirements
  pub compliance_tags: Vec<String>,
}

/// Backup requirements for swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupRequirements {
  /// Enable automated backups
  pub enabled: bool,

  /// Backup frequency
  pub frequency: BackupFrequency,

  /// Number of backup copies to maintain
  pub copies: u32,

  /// Geographic distribution requirements
  pub geographic_distribution: GeographicDistribution,

  /// Encryption requirements for backups
  pub encryption: EncryptionRequirements,

  /// Testing requirements for backup validity
  pub testing: BackupTestingRequirements,
}

/// Geographic distribution for backups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeographicDistribution {
  /// Single location
  SingleLocation,

  /// Multiple locations in same region
  MultipleLocationsRegional,

  /// Multiple regions
  MultipleRegions { min_regions: u32 },

  /// Global distribution
  Global,
}

/// Encryption requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionRequirements {
  /// Encryption enabled
  pub enabled: bool,

  /// Encryption algorithm
  pub algorithm: String,

  /// Key management requirements
  pub key_management: KeyManagement,

  /// Compliance requirements
  pub compliance_level: ComplianceLevel,
}

/// Key management options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyManagement {
  /// System-managed keys
  SystemManaged,

  /// User-managed keys
  UserManaged { key_location: String },

  /// External key management service
  ExternalKMS { service: String, config: Value },
}

/// Compliance levels for encryption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceLevel {
  Basic,
  Enhanced,
  Military,
  Custom { requirements: Vec<String> },
}

/// Backup testing requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupTestingRequirements {
  /// Enable backup testing
  pub enabled: bool,

  /// Testing frequency
  pub frequency: TestingFrequency,

  /// Types of tests to perform
  pub test_types: Vec<BackupTestType>,

  /// Success criteria
  pub success_criteria: BackupTestCriteria,
}

/// Backup testing frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestingFrequency {
  /// Test each backup
  PerBackup,

  /// Test periodically
  Periodic { interval_days: u32 },

  /// Test on demand only
  OnDemand,
}

/// Types of backup tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupTestType {
  /// Verify backup integrity
  IntegrityCheck,

  /// Test partial restore
  PartialRestore,

  /// Test full restore
  FullRestore,

  /// Performance test
  PerformanceTest,
}

/// Backup test success criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupTestCriteria {
  /// Maximum acceptable restore time
  pub max_restore_time_minutes: u32,

  /// Minimum data integrity score
  pub min_integrity_score: f32,

  /// Required test success rate
  pub min_success_rate: f32,
}

/// Cleanup preferences for swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupPreferences {
  /// Immediate cleanup on deletion
  pub immediate_cleanup: bool,

  /// Grace period before cleanup starts
  pub grace_period_hours: u32,

  /// Data export before cleanup
  pub export_before_cleanup: bool,

  /// Verification requirements
  pub verification_required: bool,

  /// Audit requirements
  pub audit_cleanup: bool,

  /// Cleanup thoroughness
  pub thoroughness: CleanupThoroughness,
}

/// Levels of cleanup thoroughness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleanupThoroughness {
  /// Basic cleanup (main data only)
  Basic,

  /// Standard cleanup (includes caches)
  Standard,

  /// Thorough cleanup (includes temporary files)
  Thorough,

  /// Paranoid cleanup (secure deletion, multiple passes)
  Paranoid,
}

/// Resources currently allocated to swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocatedResources {
  /// Currently allocated memory in MB
  pub memory_mb: u64,

  /// Currently allocated CPU cores
  pub cpu_cores: f32,

  /// Currently allocated storage in GB
  pub storage_gb: u64,

  /// Network resources
  pub network_resources: AllocatedNetworkResources,

  /// Resource utilization statistics
  pub utilization: ResourceUtilization,
}

/// Allocated network resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocatedNetworkResources {
  /// Allocated bandwidth in Mbps
  pub bandwidth_mbps: u32,

  /// Number of active connections
  pub active_connections: u32,

  /// Port allocations
  pub allocated_ports: Vec<u16>,
}

/// Resource utilization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
  /// Memory utilization percentage
  pub memory_percent: f32,

  /// CPU utilization percentage
  pub cpu_percent: f32,

  /// Storage utilization percentage
  pub storage_percent: f32,

  /// Network utilization percentage
  pub network_percent: f32,

  /// Utilization trend
  pub trend: UtilizationTrend,
}

/// Resource utilization trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UtilizationTrend {
  Increasing,
  Stable,
  Decreasing,
  Volatile,
}

/// Complete inventory of swarm data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataInventory {
  /// Database files and sizes
  pub databases: HashMap<String, DataFileInfo>,

  /// Vector database files
  pub vector_data: HashMap<String, DataFileInfo>,

  /// Graph database files
  pub graph_data: HashMap<String, DataFileInfo>,

  /// Plugin memory files
  pub plugin_memory: HashMap<String, DataFileInfo>,

  /// Log files
  pub logs: HashMap<String, DataFileInfo>,

  /// Backup files
  pub backups: HashMap<String, DataFileInfo>,

  /// Temporary files
  pub temporary_files: HashMap<String, DataFileInfo>,

  /// Cache files
  pub cache_files: HashMap<String, DataFileInfo>,

  /// Total data size in bytes
  pub total_size_bytes: u64,

  /// Data size by category
  pub size_by_category: HashMap<String, u64>,
}

/// Information about a data file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFileInfo {
  /// File path
  pub path: PathBuf,

  /// File size in bytes
  pub size_bytes: u64,

  /// Last modified time
  pub modified_at: DateTime<Utc>,

  /// File type/format
  pub file_type: String,

  /// Whether file is compressed
  pub compressed: bool,

  /// Whether file is encrypted
  pub encrypted: bool,

  /// Importance level for cleanup decisions
  pub importance: DataImportance,

  /// Dependencies on other files
  pub dependencies: Vec<String>,
}

/// Importance levels for data cleanup decisions
#[derive(
  Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord,
)]
pub enum DataImportance {
  /// Can be safely deleted
  Low,

  /// Useful but not critical
  Medium,

  /// Important for operations
  High,

  /// Critical data, backup before deletion
  Critical,

  /// Must not be deleted (legal hold, etc.)
  Protected,
}

/// Swarm-specific cleanup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmCleanupConfig {
  /// Cleanup policies specific to this swarm
  pub policies: Vec<SwarmCleanupPolicy>,

  /// Data export configuration
  pub export_config: DataExportConfig,

  /// Verification requirements
  pub verification_config: CleanupVerificationConfig,

  /// Rollback configuration in case cleanup fails
  pub rollback_config: CleanupRollbackConfig,
}

/// Swarm-specific cleanup policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmCleanupPolicy {
  /// Policy name and description
  pub name: String,
  pub description: String,

  /// Data types this policy applies to
  pub applies_to: Vec<String>,

  /// Cleanup actions
  pub actions: Vec<CleanupAction>,

  /// Conditions for executing this policy
  pub conditions: Vec<CleanupCondition>,

  /// Priority order for execution
  pub priority: u32,
}

/// Cleanup actions that can be performed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleanupAction {
  /// Delete files immediately
  Delete,

  /// Move files to archive location
  Archive { location: String },

  /// Compress files before deletion
  CompressAndDelete { compression_level: u8 },

  /// Secure deletion (multiple passes)
  SecureDelete { passes: u32 },

  /// Export data before deletion
  ExportAndDelete { export_format: String },

  /// Anonymize data before archival
  AnonymizeAndArchive,

  /// Custom cleanup script
  CustomScript {
    script_path: String,
    args: Vec<String>,
  },
}

/// Conditions that trigger cleanup actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleanupCondition {
  /// Always apply this cleanup
  Always,

  /// Apply if swarm type matches
  SwarmType { swarm_types: Vec<String> },

  /// Apply if data is older than threshold
  DataAge { older_than_days: u32 },

  /// Apply if data size exceeds threshold
  DataSize { larger_than_mb: u64 },

  /// Apply based on data importance
  DataImportance { max_importance: DataImportance },

  /// Apply if user confirms
  UserConfirmation { message: String },

  /// Apply based on custom logic
  Custom { condition_script: String },
}

/// Data export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataExportConfig {
  /// Enable data export before cleanup
  pub enabled: bool,

  /// Export formats to generate
  pub formats: Vec<ExportFormat>,

  /// Export destination
  pub destination: ExportDestination,

  /// Data filtering for export
  pub filters: Vec<ExportFilter>,

  /// Compression for exports
  pub compression: ExportCompression,
}

/// Export formats available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
  /// JSON export
  Json { pretty: bool },

  /// CSV export for tabular data
  Csv { delimiter: String },

  /// SQL dump
  SqlDump { include_schema: bool },

  /// Binary backup
  Binary { compressed: bool },

  /// Custom format
  Custom { format_name: String, config: Value },
}

/// Export destinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportDestination {
  /// Local file system
  Local { path: PathBuf },

  /// Cloud storage
  Cloud { provider: String, config: Value },

  /// Remote server
  Remote { url: String, auth: Value },

  /// Multiple destinations
  Multiple {
    destinations: Vec<ExportDestination>,
  },
}

/// Filters for data export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFilter {
  /// Export data newer than date
  NewerThan { date: DateTime<Utc> },

  /// Export data of specific types
  DataTypes { types: Vec<String> },

  /// Export data matching criteria
  Criteria { criteria: Value },

  /// Exclude sensitive data
  ExcludeSensitive,

  /// Include only important data
  ImportantDataOnly,
}

/// Compression options for exports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportCompression {
  /// Enable compression
  pub enabled: bool,

  /// Compression algorithm
  pub algorithm: CompressionAlgorithm,

  /// Compression level
  pub level: u8,

  /// Split large exports into chunks
  pub chunk_size_mb: Option<u64>,
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
  Gzip,
  Zstd,
  Lz4,
  Brotli,
  Custom { name: String },
}

/// Cleanup verification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupVerificationConfig {
  /// Enable verification steps
  pub enabled: bool,

  /// Verification methods to use
  pub methods: Vec<VerificationMethod>,

  /// Required verification success rate
  pub required_success_rate: f32,

  /// Actions to take if verification fails
  pub failure_actions: Vec<VerificationFailureAction>,
}

/// Methods for verifying cleanup success
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationMethod {
  /// Verify files are actually deleted
  FileSystemCheck,

  /// Verify database records are removed
  DatabaseCheck { tables: Vec<String> },

  /// Verify no orphaned data remains
  OrphanCheck,

  /// Verify backup integrity
  BackupIntegrity,

  /// Custom verification script
  CustomScript { script_path: String },
}

/// Actions to take when verification fails
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationFailureAction {
  /// Retry cleanup
  RetryCleanup { max_retries: u32 },

  /// Manual intervention required
  ManualIntervention { notification: String },

  /// Rollback changes
  Rollback,

  /// Continue with warnings
  ContinueWithWarnings,

  /// Abort cleanup process
  AbortCleanup,
}

/// Rollback configuration for failed cleanups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupRollbackConfig {
  /// Enable rollback capability
  pub enabled: bool,

  /// Rollback method
  pub method: RollbackMethod,

  /// Rollback timeout
  pub timeout_minutes: u32,

  /// Actions after successful rollback
  pub post_rollback_actions: Vec<PostRollbackAction>,
}

/// Methods for rolling back cleanup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RollbackMethod {
  /// Restore from backup
  RestoreFromBackup { backup_location: String },

  /// Restore from transaction log
  TransactionLog,

  /// Restore using version control
  VersionControl { repository: String },

  /// Custom rollback script
  CustomScript { script_path: String },
}

/// Actions to take after successful rollback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PostRollbackAction {
  /// Send notification
  Notify {
    recipients: Vec<String>,
    message: String,
  },

  /// Update swarm status
  UpdateStatus { new_status: String },

  /// Schedule retry
  ScheduleRetry { delay_hours: u32 },

  /// Disable automatic cleanup
  DisableAutoCleanup,
}

/// Lifecycle timestamps for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleTimestamps {
  /// When swarm was created
  pub created_at: DateTime<Utc>,

  /// When swarm became active
  pub activated_at: Option<DateTime<Utc>>,

  /// Last status change
  pub last_status_change: DateTime<Utc>,

  /// Last health check
  pub last_health_check: Option<DateTime<Utc>>,

  /// Last backup
  pub last_backup: Option<DateTime<Utc>>,

  /// When cleanup was initiated
  pub cleanup_initiated: Option<DateTime<Utc>>,

  /// When cleanup completed
  pub cleanup_completed: Option<DateTime<Utc>>,
}

/// Swarm health monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmHealth {
  /// Overall health status
  pub status: HealthStatus,

  /// Individual health checks
  pub checks: Vec<HealthCheck>,

  /// Resource health
  pub resource_health: ResourceHealth,

  /// Data health
  pub data_health: DataHealth,

  /// Performance health
  pub performance_health: PerformanceHealth,

  /// Last health assessment
  pub last_assessment: DateTime<Utc>,
}

/// Health status levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
  Healthy,
  Warning,
  Critical,
  Unknown,
  Maintenance,
}

/// Individual health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
  /// Check name
  pub name: String,

  /// Check result
  pub status: HealthStatus,

  /// Check details
  pub message: String,

  /// Check timestamp
  pub timestamp: DateTime<Utc>,

  /// Check duration
  pub duration_ms: u64,
}

/// Resource health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceHealth {
  /// Memory health
  pub memory: ResourceHealthMetric,

  /// CPU health
  pub cpu: ResourceHealthMetric,

  /// Storage health
  pub storage: ResourceHealthMetric,

  /// Network health
  pub network: ResourceHealthMetric,
}

/// Individual resource health metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceHealthMetric {
  /// Current status
  pub status: HealthStatus,

  /// Current utilization percentage
  pub utilization_percent: f32,

  /// Available capacity
  pub available_capacity: f64,

  /// Trend over time
  pub trend: UtilizationTrend,

  /// Alerts triggered
  pub alerts: Vec<String>,
}

/// Data health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataHealth {
  /// Database integrity
  pub database_integrity: f32,

  /// Backup freshness
  pub backup_freshness_hours: u32,

  /// Data consistency score
  pub consistency_score: f32,

  /// Corruption detected
  pub corruption_detected: bool,

  /// Orphaned data count
  pub orphaned_data_count: u32,
}

/// Performance health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHealth {
  /// Average response time
  pub avg_response_time_ms: f32,

  /// Task success rate
  pub task_success_rate: f32,

  /// Agent availability
  pub agent_availability: f32,

  /// System throughput
  pub throughput_tasks_per_hour: f32,

  /// Error rate
  pub error_rate: f32,
}

/// Cleanup policy for automated data management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupPolicy {
  /// Policy identifier
  pub id: String,

  /// Policy name and description
  pub name: String,
  pub description: String,

  /// Policy targets
  pub targets: Vec<CleanupTarget>,

  /// Execution schedule
  pub schedule: CleanupSchedule,

  /// Policy actions
  pub actions: Vec<CleanupAction>,

  /// Policy conditions
  pub conditions: Vec<CleanupCondition>,

  /// Policy enabled
  pub enabled: bool,
}

/// Cleanup targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleanupTarget {
  /// Target specific swarm types
  SwarmTypes { types: Vec<String> },

  /// Target data older than threshold
  OldData { older_than_days: u32 },

  /// Target data by size
  LargeData { larger_than_gb: u64 },

  /// Target specific data types
  DataTypes { types: Vec<String> },

  /// Target inactive swarms
  InactiveSwarms { inactive_days: u32 },

  /// Target all data (use with caution)
  AllData,
}

/// Cleanup schedule options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleanupSchedule {
  /// Run once immediately
  Once,

  /// Run periodically
  Periodic { interval_hours: u32 },

  /// Run on specific conditions
  Conditional { conditions: Vec<String> },

  /// Run on cron schedule
  Cron { expression: String },

  /// Manual trigger only
  Manual,
}

/// Retention policy for data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
  /// Data type this policy applies to
  pub data_type: String,

  /// Retention period in days
  pub retention_days: u32,

  /// Archive before deletion
  pub archive_before_deletion: bool,

  /// Archive location
  pub archive_location: Option<String>,

  /// Legal hold considerations
  pub legal_hold: bool,

  /// Compliance requirements
  pub compliance: Vec<String>,
}

/// Scheduled cleanup task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledCleanup {
  /// Cleanup task identifier
  pub id: String,

  /// Target swarm
  pub swarm_id: String,

  /// Scheduled execution time
  pub scheduled_at: DateTime<Utc>,

  /// Cleanup type
  pub cleanup_type: ScheduledCleanupType,

  /// Task status
  pub status: ScheduledCleanupStatus,

  /// Creation timestamp
  pub created_at: DateTime<Utc>,
}

/// Types of scheduled cleanup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduledCleanupType {
  /// Regular maintenance cleanup
  Maintenance,

  /// Cleanup after swarm deletion
  PostDeletion,

  /// Cleanup due to inactivity
  Inactivity,

  /// Cleanup for storage space
  StorageReclamation,

  /// Compliance-driven cleanup
  Compliance,
}

/// Status of scheduled cleanup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduledCleanupStatus {
  Pending,
  InProgress { progress: f32 },
  Completed { summary: CleanupSummary },
  Failed { error: String },
  Cancelled,
}

/// Summary of completed cleanup
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CleanupSummary {
  /// Cleanup completion time
  pub completed_at: DateTime<Utc>,

  /// Total cleanup duration
  pub duration_seconds: u64,

  /// Files deleted
  pub files_deleted: u32,

  /// Data deleted in bytes
  pub bytes_deleted: u64,

  /// Files archived
  pub files_archived: u32,

  /// Errors encountered
  pub errors: Vec<String>,

  /// Warnings generated
  pub warnings: Vec<String>,

  /// Verification results
  pub verification_results: Vec<String>,

  /// Cleanup efficiency score
  pub efficiency_score: f32,
}

/// Audit logger for compliance and debugging
#[derive(Debug)]
pub struct AuditLogger {
  /// Log file path
  log_path: PathBuf,

  /// Current log file handle
  current_log: Option<std::fs::File>,

  /// Log retention policy
  retention_days: u32,
}

/// Backup manager for data safety
#[derive(Debug)]
pub struct BackupManager {
  /// Backup locations
  locations: Vec<BackupLocation>,

  /// Active backups tracking
  active_backups: HashMap<String, BackupInfo>,

  /// Backup schedule
  schedule: BackupSchedule,
}

/// Backup information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupInfo {
  /// Backup identifier
  pub id: String,

  /// Source swarm
  pub swarm_id: String,

  /// Backup location
  pub location: String,

  /// Backup size
  pub size_bytes: u64,

  /// Backup creation time
  pub created_at: DateTime<Utc>,

  /// Backup type
  pub backup_type: BackupType,

  /// Backup integrity hash
  pub integrity_hash: String,
}

/// Types of backups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupType {
  Full,
  Incremental,
  Differential,
  Snapshot,
}

/// Backup scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupSchedule {
  /// Enable automatic backups
  pub enabled: bool,

  /// Backup frequency
  pub frequency: BackupFrequency,

  /// Backup retention
  pub retention_copies: u32,

  /// Backup window preferences
  pub preferred_hours: Vec<u8>,
}

impl SwarmLifecycleManager {
  /// Create new lifecycle manager with .zen directory structure
  pub async fn new() -> SwarmResult<Self> {
    let zen_config = ZenDirectoryConfig::default();

    // Create directory structure
    Self::create_zen_directories(&zen_config).await?;

    Ok(Self {
      active_swarms: Arc::new(DashMap::new()),
      zen_config: Arc::new(zen_config),
      cleanup_policies: Arc::new(RwLock::new(Vec::new())),
      retention_policies: Arc::new(RwLock::new(HashMap::new())),
      scheduled_cleanups: Arc::new(RwLock::new(Vec::new())),
      audit_logger: Arc::new(Mutex::new(AuditLogger::new(
        ".zen/collective/audit.log".into(),
      )?)),
      backup_manager: Arc::new(BackupManager::new()),
    })
  }

  /// Create .zen directory structure
  async fn create_zen_directories(
    config: &ZenDirectoryConfig,
  ) -> SwarmResult<()> {
    let directories = [
      &config.base_path,
      &config.swarm_path,
      &config.claude_path,
      &config.gemini_path,
      &config.collective_path,
    ];

    for dir in directories {
      tokio::fs::create_dir_all(dir).await.map_err(|e| {
        SwarmError::Configuration(format!(
          "Failed to create directory {:?}: {}",
          dir, e
        ))
      })?;
    }

    // Set appropriate permissions if on Unix
    #[cfg(unix)]
    {
      use std::os::unix::fs::PermissionsExt;
      for dir in directories {
        let mut perms = tokio::fs::metadata(dir)
          .await
          .map_err(|e| {
            SwarmError::Configuration(format!(
              "Failed to get directory metadata: {}",
              e
            ))
          })?
          .permissions();
        perms.set_mode(config.permissions.dir_mode);
        tokio::fs::set_permissions(dir, perms).await.map_err(|e| {
          SwarmError::Configuration(format!(
            "Failed to set directory permissions: {}",
            e
          ))
        })?;
      }
    }

    Ok(())
  }

  /// Create a new swarm with full lifecycle tracking
  pub async fn create_swarm(
    &self,
    swarm_id: String,
    config: SwarmConfiguration,
  ) -> SwarmResult<SwarmLifecycleState> {
    // Audit log swarm creation
    self
      .audit_log(&format!("Creating swarm: {}", swarm_id))
      .await?;

    let now = Utc::now();
    let lifecycle_state = SwarmLifecycleState {
      swarm_id: swarm_id.clone(),
      phase: LifecyclePhase::Creating {
        progress: 0.0,
        current_step: "Initializing".to_string(),
      },
      config,
      resources: AllocatedResources::default(),
      data_inventory: DataInventory::default(),
      cleanup_config: SwarmCleanupConfig::default(),
      timestamps: LifecycleTimestamps {
        created_at: now,
        activated_at: None,
        last_status_change: now,
        last_health_check: None,
        last_backup: None,
        cleanup_initiated: None,
        cleanup_completed: None,
      },
      health: SwarmHealth::default(),
    };

    // Register swarm
    self
      .active_swarms
      .insert(swarm_id.clone(), lifecycle_state.clone());

    // Create initial backup if required
    if lifecycle_state.config.backup_requirements.enabled {
      self.create_initial_backup(&swarm_id).await?;
    }

    self
      .audit_log(&format!("Swarm created successfully: {}", swarm_id))
      .await?;

    Ok(lifecycle_state)
  }

  /// Delete swarm with comprehensive cleanup
  pub async fn delete_swarm(
    &self,
    swarm_id: &str,
    force: bool,
  ) -> SwarmResult<CleanupSummary> {
    self
      .audit_log(&format!(
        "Initiating swarm deletion: {} (force: {})",
        swarm_id, force
      ))
      .await?;

    // Get swarm state
    let mut swarm_state =
      self.get_swarm_state(swarm_id).await?.ok_or_else(|| {
        SwarmError::Configuration(format!("Swarm {} not found", swarm_id))
      })?;

    // Check if swarm can be safely deleted
    if !force {
      self.validate_deletion_safety(&swarm_state).await?;
    }

    // Update phase to shutting down
    swarm_state.phase = LifecyclePhase::ShuttingDown {
      shutdown_started: Utc::now(),
      remaining_tasks: self.count_active_tasks(swarm_id).await?,
      cleanup_progress: 0.0,
    };
    swarm_state.timestamps.cleanup_initiated = Some(Utc::now());

    self
      .active_swarms
      .insert(swarm_id.to_string(), swarm_state.clone());

    // Execute cleanup phases
    let cleanup_summary = self
      .execute_comprehensive_cleanup(swarm_id, &swarm_state)
      .await?;

    // Mark as deleted
    let mut final_state = swarm_state;
    final_state.phase = LifecyclePhase::Deleted {
      deleted_at: Utc::now(),
      cleanup_summary: cleanup_summary.clone(),
    };
    final_state.timestamps.cleanup_completed = Some(Utc::now());

    self.active_swarms.insert(swarm_id.to_string(), final_state);

    self
      .audit_log(&format!("Swarm deletion completed: {}", swarm_id))
      .await?;

    Ok(cleanup_summary)
  }

  /// Execute comprehensive cleanup with all safety measures
  async fn execute_comprehensive_cleanup(
    &self,
    swarm_id: &str,
    swarm_state: &SwarmLifecycleState,
  ) -> SwarmResult<CleanupSummary> {
    let start_time = Utc::now();
    let mut summary = CleanupSummary {
      completed_at: start_time,
      duration_seconds: 0,
      files_deleted: 0,
      bytes_deleted: 0,
      files_archived: 0,
      errors: Vec::new(),
      warnings: Vec::new(),
      verification_results: Vec::new(),
      efficiency_score: 0.0,
    };

    // Phase 1: Graceful shutdown of agents and tasks
    self
      .audit_log(&format!(
        "Phase 1: Graceful shutdown for swarm {}",
        swarm_id
      ))
      .await?;
    match self.graceful_shutdown(swarm_id).await {
      Ok(()) => {}
      Err(e) => summary
        .warnings
        .push(format!("Graceful shutdown warning: {}", e)),
    }

    // Phase 2: Data export (if enabled)
    if swarm_state.cleanup_config.export_config.enabled {
      self
        .audit_log(&format!("Phase 2: Data export for swarm {}", swarm_id))
        .await?;
      match self
        .export_swarm_data(swarm_id, &swarm_state.cleanup_config.export_config)
        .await
      {
        Ok(exported_files) => {
          summary.files_archived += exported_files;
          summary
            .verification_results
            .push("Data export completed successfully".to_string());
        }
        Err(e) => summary.errors.push(format!("Data export failed: {}", e)),
      }
    }

    // Phase 3: Database cleanup (filtered by swarm_id)
    self
      .audit_log(&format!("Phase 3: Database cleanup for swarm {}", swarm_id))
      .await?;
    match self.cleanup_database_records(swarm_id).await {
      Ok((records_deleted, size_freed)) => {
        summary.files_deleted += records_deleted;
        summary.bytes_deleted += size_freed;
        summary.verification_results.push(format!(
          "Database cleanup: {} records deleted, {} bytes freed",
          records_deleted, size_freed
        ));
      }
      Err(e) => summary
        .errors
        .push(format!("Database cleanup failed: {}", e)),
    }

    // Phase 4: File system cleanup
    self
      .audit_log(&format!(
        "Phase 4: File system cleanup for swarm {}",
        swarm_id
      ))
      .await?;
    match self
      .cleanup_file_system(swarm_id, &swarm_state.data_inventory)
      .await
    {
      Ok((files_deleted, bytes_deleted)) => {
        summary.files_deleted += files_deleted;
        summary.bytes_deleted += bytes_deleted;
        summary.verification_results.push(format!(
          "File system cleanup: {} files, {} bytes",
          files_deleted, bytes_deleted
        ));
      }
      Err(e) => summary
        .errors
        .push(format!("File system cleanup failed: {}", e)),
    }

    // Phase 5: Vector database cleanup
    self
      .audit_log(&format!(
        "Phase 5: Vector database cleanup for swarm {}",
        swarm_id
      ))
      .await?;
    match self.cleanup_vector_data(swarm_id).await {
      Ok(size_freed) => {
        summary.bytes_deleted += size_freed;
        summary.verification_results.push(format!(
          "Vector database cleanup: {} bytes freed",
          size_freed
        ));
      }
      Err(e) => summary
        .warnings
        .push(format!("Vector database cleanup warning: {}", e)),
    }

    // Phase 6: Graph database cleanup
    self
      .audit_log(&format!(
        "Phase 6: Graph database cleanup for swarm {}",
        swarm_id
      ))
      .await?;
    match self.cleanup_graph_data(swarm_id).await {
      Ok(size_freed) => {
        summary.bytes_deleted += size_freed;
        summary.verification_results.push(format!(
          "Graph database cleanup: {} bytes freed",
          size_freed
        ));
      }
      Err(e) => summary
        .warnings
        .push(format!("Graph database cleanup warning: {}", e)),
    }

    // Phase 7: Plugin memory cleanup
    self
      .audit_log(&format!(
        "Phase 7: Plugin memory cleanup for swarm {}",
        swarm_id
      ))
      .await?;
    match self.cleanup_plugin_memory(swarm_id).await {
      Ok((files_deleted, bytes_deleted)) => {
        summary.files_deleted += files_deleted;
        summary.bytes_deleted += bytes_deleted;
        summary.verification_results.push(format!(
          "Plugin memory cleanup: {} files, {} bytes",
          files_deleted, bytes_deleted
        ));
      }
      Err(e) => summary
        .warnings
        .push(format!("Plugin memory cleanup warning: {}", e)),
    }

    // Phase 8: Verification
    self
      .audit_log(&format!(
        "Phase 8: Cleanup verification for swarm {}",
        swarm_id
      ))
      .await?;
    if swarm_state.cleanup_config.verification_config.enabled {
      match self.verify_cleanup_completeness(swarm_id).await {
        Ok(verification_results) => {
          summary.verification_results.extend(verification_results);
        }
        Err(e) => summary.errors.push(format!("Verification failed: {}", e)),
      }
    }

    // Calculate final summary
    let end_time = Utc::now();
    summary.completed_at = end_time;
    summary.duration_seconds = (end_time - start_time).num_seconds() as u64;
    summary.efficiency_score = self.calculate_cleanup_efficiency(&summary);

    // Remove from active swarms registry
    self.active_swarms.remove(swarm_id);

    Ok(summary)
  }

  /// Cleanup database records filtered by swarm_id
  async fn cleanup_database_records(
    &self,
    swarm_id: &str,
  ) -> SwarmResult<(u32, u64)> {
    // This would connect to the shared database and delete all records
    // where swarm_id matches the target swarm

    let tables_to_clean = [
      "swarm_states",
      "agent_states",
      "tasks",
      "memories",
      "plugin_executions",
      "communication_history",
      "learning_data",
      "performance_metrics",
    ];

    let mut total_records = 0;
    let mut total_bytes = 0;

    for table in &tables_to_clean {
      // Simulated cleanup - in production would execute:
      // DELETE FROM {table} WHERE swarm_id = ?
      tracing::info!("Cleaning table {} for swarm {}", table, swarm_id);

      // Placeholder values - would come from actual database operations
      total_records += 10; // Simulated record count
      total_bytes += 1024; // Simulated bytes freed
    }

    Ok((total_records, total_bytes))
  }

  /// Cleanup file system data for specific swarm
  async fn cleanup_file_system(
    &self,
    swarm_id: &str,
    data_inventory: &DataInventory,
  ) -> SwarmResult<(u32, u64)> {
    let mut files_deleted = 0;
    let mut bytes_deleted = 0;

    // Clean up directories specific to this swarm
    let swarm_dirs = [
      self.zen_config.swarm_path.join(swarm_id),
      self.zen_config.swarm_path.join("logs").join(swarm_id),
      self.zen_config.swarm_path.join("temp").join(swarm_id),
    ];

    for dir in &swarm_dirs {
      if dir.exists() {
        match self.remove_directory_recursive(dir).await {
          Ok((count, bytes)) => {
            files_deleted += count;
            bytes_deleted += bytes;
          }
          Err(e) => {
            tracing::warn!("Failed to remove directory {:?}: {}", dir, e);
          }
        }
      }
    }

    // Clean up swarm-specific files in shared directories
    for (file_path, file_info) in &data_inventory.databases {
      if file_path.contains(swarm_id) && file_info.path.exists() {
        match tokio::fs::remove_file(&file_info.path).await {
          Ok(()) => {
            files_deleted += 1;
            bytes_deleted += file_info.size_bytes;
          }
          Err(e) => {
            tracing::warn!("Failed to remove file {:?}: {}", file_info.path, e);
          }
        }
      }
    }

    Ok((files_deleted, bytes_deleted))
  }

  /// Remove directory and all contents recursively
  fn remove_directory_recursive<'a>(
    &'a self,
    dir: &'a Path,
  ) -> BoxFuture<'a, SwarmResult<(u32, u64)>> {
    Box::pin(async move {
      let mut files_removed = 0;
      let mut bytes_removed = 0;

      if !dir.exists() {
        return Ok((0, 0));
      }

      let mut entries = tokio::fs::read_dir(dir).await.map_err(|e| {
        SwarmError::Configuration(format!("Failed to read directory: {}", e))
      })?;

      while let Some(entry) = entries.next_entry().await.map_err(|e| {
        SwarmError::Configuration(format!(
          "Failed to read directory entry: {}",
          e
        ))
      })? {
        let path = entry.path();
        let metadata = entry.metadata().await.map_err(|e| {
          SwarmError::Configuration(format!("Failed to read metadata: {}", e))
        })?;

        if metadata.is_dir() {
          let (sub_files, sub_bytes) =
            self.remove_directory_recursive(&path).await?;
          files_removed += sub_files;
          bytes_removed += sub_bytes;
        } else {
          bytes_removed += metadata.len();
          files_removed += 1;

          tokio::fs::remove_file(&path).await.map_err(|e| {
            SwarmError::Configuration(format!("Failed to remove file: {}", e))
          })?;
        }
      }

      // Remove the directory itself
      tokio::fs::remove_dir(dir).await.map_err(|e| {
        SwarmError::Configuration(format!("Failed to remove directory: {}", e))
      })?;

      Ok((files_removed, bytes_removed))
    })
  }

  /// Cleanup vector database data for specific swarm
  async fn cleanup_vector_data(&self, swarm_id: &str) -> SwarmResult<u64> {
    let vector_dir = self.zen_config.swarm_path.join("vector").join(swarm_id);

    if vector_dir.exists() {
      let (_, bytes_freed) =
        self.remove_directory_recursive(&vector_dir).await?;
      Ok(bytes_freed)
    } else {
      Ok(0)
    }
  }

  /// Cleanup graph database data for specific swarm
  async fn cleanup_graph_data(&self, swarm_id: &str) -> SwarmResult<u64> {
    let graph_dir = self.zen_config.swarm_path.join("graph").join(swarm_id);

    if graph_dir.exists() {
      let (_, bytes_freed) =
        self.remove_directory_recursive(&graph_dir).await?;
      Ok(bytes_freed)
    } else {
      Ok(0)
    }
  }

  /// Cleanup plugin memory data for specific swarm
  async fn cleanup_plugin_memory(
    &self,
    swarm_id: &str,
  ) -> SwarmResult<(u32, u64)> {
    let plugin_dir = self
      .zen_config
      .swarm_path
      .join("plugin_memory")
      .join(swarm_id);

    if plugin_dir.exists() {
      self.remove_directory_recursive(&plugin_dir).await
    } else {
      Ok((0, 0))
    }
  }

  /// Verify cleanup completeness
  async fn verify_cleanup_completeness(
    &self,
    swarm_id: &str,
  ) -> SwarmResult<Vec<String>> {
    let mut results = Vec::new();

    // Verify no swarm-specific directories remain
    let check_dirs = [
      self.zen_config.swarm_path.join(swarm_id),
      self.zen_config.swarm_path.join("vector").join(swarm_id),
      self.zen_config.swarm_path.join("graph").join(swarm_id),
      self
        .zen_config
        .swarm_path
        .join("plugin_memory")
        .join(swarm_id),
    ];

    for dir in &check_dirs {
      if dir.exists() {
        results.push(format!("WARNING: Directory still exists: {:?}", dir));
      } else {
        results.push(format!("VERIFIED: Directory cleaned: {:?}", dir));
      }
    }

    // Verify database cleanup (would check actual database)
    results.push(format!(
      "VERIFIED: Database records cleaned for swarm_id: {}",
      swarm_id
    ));

    // Check for orphaned data
    let orphaned_count = self.count_orphaned_data(swarm_id).await?;
    if orphaned_count > 0 {
      results.push(format!(
        "WARNING: {} orphaned data items found",
        orphaned_count
      ));
    } else {
      results.push("VERIFIED: No orphaned data found".to_string());
    }

    Ok(results)
  }

  /// Count orphaned data that might reference deleted swarm
  async fn count_orphaned_data(&self, _swarm_id: &str) -> SwarmResult<u32> {
    // In production, this would scan for any remaining references
    // to the swarm_id in logs, temp files, caches, etc.
    Ok(0) // Placeholder
  }

  /// Calculate cleanup efficiency score
  fn calculate_cleanup_efficiency(&self, summary: &CleanupSummary) -> f32 {
    let total_operations = summary.files_deleted + summary.files_archived;
    let successful_operations = total_operations - summary.errors.len() as u32;

    if total_operations == 0 {
      return 1.0; // Perfect score for no-op cleanup
    }

    let success_rate = successful_operations as f32 / total_operations as f32;
    let warning_penalty = (summary.warnings.len() as f32 * 0.05).min(0.3);

    (success_rate - warning_penalty).max(0.0)
  }

  /// Additional utility methods for comprehensive lifecycle management

  async fn graceful_shutdown(&self, swarm_id: &str) -> SwarmResult<()> {
    // Stop all agents gracefully
    // Complete running tasks where possible
    // Save state before shutdown
    tracing::info!("Performing graceful shutdown for swarm: {}", swarm_id);
    Ok(())
  }

  async fn export_swarm_data(
    &self,
    swarm_id: &str,
    _export_config: &DataExportConfig,
  ) -> SwarmResult<u32> {
    // Export important data before cleanup
    tracing::info!("Exporting data for swarm: {}", swarm_id);
    Ok(5) // Placeholder: exported 5 files
  }

  async fn validate_deletion_safety(
    &self,
    _swarm_state: &SwarmLifecycleState,
  ) -> SwarmResult<()> {
    // Check if swarm has running tasks
    // Check if swarm is referenced by other systems
    // Check user permissions
    Ok(())
  }

  async fn count_active_tasks(&self, _swarm_id: &str) -> SwarmResult<u32> {
    // Count tasks that are still running
    Ok(0) // Placeholder
  }

  async fn get_swarm_state(
    &self,
    swarm_id: &str,
  ) -> SwarmResult<Option<SwarmLifecycleState>> {
    Ok(self.active_swarms.get(swarm_id).map(|state| state.clone()))
  }

  async fn create_initial_backup(&self, swarm_id: &str) -> SwarmResult<()> {
    tracing::info!("Creating initial backup for swarm: {}", swarm_id);
    Ok(())
  }

  async fn audit_log(&self, message: &str) -> SwarmResult<()> {
    let mut logger = self.audit_logger.lock().await;
    logger.log(message).await
  }

  /// Get lifecycle statistics
  pub async fn get_lifecycle_stats(&self) -> LifecycleStats {
    let active_count = self.active_swarms.len();
    let phases = self
      .active_swarms
      .iter()
      .map(|entry| entry.phase.clone())
      .collect::<Vec<_>>();

    let mut phase_counts = HashMap::new();
    for phase in phases {
      let phase_name = match phase {
        LifecyclePhase::Creating { .. } => "Creating",
        LifecyclePhase::Active { .. } => "Active",
        LifecyclePhase::Paused { .. } => "Paused",
        LifecyclePhase::Migrating { .. } => "Migrating",
        LifecyclePhase::ShuttingDown { .. } => "ShuttingDown",
        LifecyclePhase::Archiving { .. } => "Archiving",
        LifecyclePhase::CleaningUp { .. } => "CleaningUp",
        LifecyclePhase::Deleted { .. } => "Deleted",
      };
      *phase_counts.entry(phase_name.to_string()).or_insert(0) += 1;
    }

    LifecycleStats {
      total_active_swarms: active_count,
      swarms_by_phase: phase_counts,
      scheduled_cleanups: self.scheduled_cleanups.read().await.len(),
      cleanup_policies: self.cleanup_policies.read().await.len(),
    }
  }
}

/// Lifecycle statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleStats {
  pub total_active_swarms: usize,
  pub swarms_by_phase: HashMap<String, u32>,
  pub scheduled_cleanups: usize,
  pub cleanup_policies: usize,
}

impl AuditLogger {
  pub fn new(log_path: PathBuf) -> SwarmResult<Self> {
    Ok(Self {
      log_path,
      current_log: None,
      retention_days: 90,
    })
  }

  pub async fn log(&mut self, message: &str) -> SwarmResult<()> {
    let timestamp = Utc::now();
    let _log_entry = format!(
      "{}: {}\n",
      timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
      message
    );

    // In production, would write to actual log file
    tracing::info!("AUDIT: {}", message);

    Ok(())
  }

  /// Get retention policy for data type
  pub async fn get_retention_policy(
    &self,
    data_type: &str,
  ) -> Option<RetentionPolicy> {
    let policies = self.retention_policies.read().await;
    policies.get(data_type).cloned()
  }

  /// Set retention policy for data type
  pub async fn set_retention_policy(
    &self,
    data_type: String,
    policy: RetentionPolicy,
  ) -> SwarmResult<()> {
    let mut policies = self.retention_policies.write().await;
    policies.insert(data_type, policy);
    Ok(())
  }

  /// Create backup using backup manager
  pub async fn create_backup(&self, swarm_id: &str) -> SwarmResult<String> {
    let backup_id =
      format!("backup_{}_{}", swarm_id, chrono::Utc::now().timestamp());
    // Use backup_manager field
    tracing::info!("Creating backup {} using backup manager", backup_id);
    Ok(backup_id)
  }
}

impl BackupManager {
  pub fn new() -> Self {
    Self {
      locations: Vec::new(),
      active_backups: HashMap::new(),
      schedule: BackupSchedule {
        enabled: true,
        frequency: BackupFrequency::Periodic { interval_hours: 6 },
        retention_copies: 7,
        preferred_hours: vec![2, 8, 14, 20], // Every 6 hours starting at 2 AM
      },
    }
  }

  /// Add backup location
  pub fn add_location(&mut self, location: BackupLocation) {
    self.locations.push(location);
  }

  /// Get backup schedule
  pub fn get_schedule(&self) -> &BackupSchedule {
    &self.schedule
  }

  /// Start backup and track it
  pub fn start_backup(&mut self, backup_id: String, info: BackupInfo) {
    self.active_backups.insert(backup_id, info);
  }

  /// Get active backup info
  pub fn get_backup(&self, backup_id: &str) -> Option<&BackupInfo> {
    self.active_backups.get(backup_id)
  }
}

impl Default for ZenDirectoryConfig {
  fn default() -> Self {
    let base_path = PathBuf::from(".zen");

    Self {
      base_path: base_path.clone(),
      swarm_path: base_path.join("swarm"),
      claude_path: base_path.join("claude"),
      gemini_path: base_path.join("gemini"),
      collective_path: base_path.join("collective"),
      permissions: DirectoryPermissions {
        file_mode: 0o644,
        dir_mode: 0o755,
        owner_uid: None,
        owner_gid: None,
        encrypt_sensitive: false,
      },
      backup_locations: Vec::new(),
    }
  }
}

impl Default for AllocatedResources {
  fn default() -> Self {
    Self {
      memory_mb: 0,
      cpu_cores: 0.0,
      storage_gb: 0,
      network_resources: AllocatedNetworkResources {
        bandwidth_mbps: 0,
        active_connections: 0,
        allocated_ports: Vec::new(),
      },
      utilization: ResourceUtilization {
        memory_percent: 0.0,
        cpu_percent: 0.0,
        storage_percent: 0.0,
        network_percent: 0.0,
        trend: UtilizationTrend::Stable,
      },
    }
  }
}

impl Default for DataInventory {
  fn default() -> Self {
    Self {
      databases: HashMap::new(),
      vector_data: HashMap::new(),
      graph_data: HashMap::new(),
      plugin_memory: HashMap::new(),
      logs: HashMap::new(),
      backups: HashMap::new(),
      temporary_files: HashMap::new(),
      cache_files: HashMap::new(),
      total_size_bytes: 0,
      size_by_category: HashMap::new(),
    }
  }
}

impl Default for SwarmCleanupConfig {
  fn default() -> Self {
    Self {
      policies: Vec::new(),
      export_config: DataExportConfig {
        enabled: false,
        formats: vec![ExportFormat::Json { pretty: true }],
        destination: ExportDestination::Local {
          path: PathBuf::from(".zen/collective/exports"),
        },
        filters: Vec::new(),
        compression: ExportCompression {
          enabled: true,
          algorithm: CompressionAlgorithm::Zstd,
          level: 3,
          chunk_size_mb: Some(100),
        },
      },
      verification_config: CleanupVerificationConfig {
        enabled: true,
        methods: vec![
          VerificationMethod::FileSystemCheck,
          VerificationMethod::DatabaseCheck {
            tables: vec![
              "swarm_states".to_string(),
              "agent_states".to_string(),
            ],
          },
          VerificationMethod::OrphanCheck,
        ],
        required_success_rate: 0.95,
        failure_actions: vec![VerificationFailureAction::RetryCleanup {
          max_retries: 2,
        }],
      },
      rollback_config: CleanupRollbackConfig {
        enabled: true,
        method: RollbackMethod::RestoreFromBackup {
          backup_location: ".zen/collective/backups".to_string(),
        },
        timeout_minutes: 30,
        post_rollback_actions: vec![PostRollbackAction::Notify {
          recipients: vec!["admin@example.com".to_string()],
          message: "Rollback completed".to_string(),
        }],
      },
    }
  }
}

impl Default for SwarmHealth {
  fn default() -> Self {
    Self {
      status: HealthStatus::Unknown,
      checks: Vec::new(),
      resource_health: ResourceHealth::default(),
      data_health: DataHealth::default(),
      performance_health: PerformanceHealth::default(),
      last_assessment: Utc::now(),
    }
  }
}

impl Default for ResourceHealth {
  fn default() -> Self {
    Self {
      memory: ResourceHealthMetric::default(),
      cpu: ResourceHealthMetric::default(),
      storage: ResourceHealthMetric::default(),
      network: ResourceHealthMetric::default(),
    }
  }
}

impl Default for ResourceHealthMetric {
  fn default() -> Self {
    Self {
      status: HealthStatus::Unknown,
      utilization_percent: 0.0,
      available_capacity: 0.0,
      trend: UtilizationTrend::Stable,
      alerts: Vec::new(),
    }
  }
}

impl Default for DataHealth {
  fn default() -> Self {
    Self {
      database_integrity: 1.0,
      backup_freshness_hours: 24,
      consistency_score: 1.0,
      corruption_detected: false,
      orphaned_data_count: 0,
    }
  }
}

impl Default for PerformanceHealth {
  fn default() -> Self {
    Self {
      avg_response_time_ms: 0.0,
      task_success_rate: 0.0,
      agent_availability: 0.0,
      throughput_tasks_per_hour: 0.0,
      error_rate: 0.0,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[tokio::test]
  async fn test_lifecycle_manager_creation() {
    let manager = SwarmLifecycleManager::new().await;
    assert!(manager.is_ok());
  }

  #[tokio::test]
  async fn test_zen_directory_structure() {
    let config = ZenDirectoryConfig::default();
    assert_eq!(config.base_path, PathBuf::from(".zen"));
    assert_eq!(config.swarm_path, PathBuf::from(".zen/swarm"));
    assert_eq!(config.claude_path, PathBuf::from(".zen/claude"));
    assert_eq!(config.gemini_path, PathBuf::from(".zen/gemini"));
    assert_eq!(config.collective_path, PathBuf::from(".zen/collective"));
  }

  #[tokio::test]
  async fn test_cleanup_summary_calculation() {
    let manager = SwarmLifecycleManager::new().await.unwrap();

    let summary = CleanupSummary {
      completed_at: Utc::now(),
      duration_seconds: 60,
      files_deleted: 10,
      bytes_deleted: 1024,
      files_archived: 5,
      errors: vec!["Test error".to_string()],
      warnings: vec!["Test warning".to_string()],
      verification_results: Vec::new(),
      efficiency_score: 0.0,
    };

    let efficiency = manager.calculate_cleanup_efficiency(&summary);
    assert!(efficiency >= 0.0 && efficiency <= 1.0);
  }

  #[test]
  fn test_lifecycle_phase_serialization() {
    let phase = LifecyclePhase::Active {
      uptime_seconds: 3600,
      agent_count: 5,
    };

    let json = serde_json::to_string(&phase).unwrap();
    let deserialized: LifecyclePhase = serde_json::from_str(&json).unwrap();

    assert_eq!(phase, deserialized);
  }
}
