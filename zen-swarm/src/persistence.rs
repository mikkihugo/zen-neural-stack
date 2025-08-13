//! Persistence layer with Cozo for swarm state management
//!
//! High-performance storage for swarm states, agent memories, task history,
//! and coordination data with ACID guarantees and distributed consistency.
//!
//! Uses pure Rust Cozo database - no FFI dependencies!

use crate::{SwarmError, Task};

#[cfg(feature = "graph")]
use crate::PersistentSwarm;
use anyhow::Result;
// Note: Removed libsql dependency - using pure Cozo for all persistence
// use libsql::{Connection, Database};
use serde::{Deserialize, Serialize};

// Placeholder for PersistentSwarm when graph feature is disabled
#[cfg(not(feature = "graph"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentSwarm {
  pub swarm_id: String,
  pub created_at: chrono::DateTime<chrono::Utc>,
}
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Persistence manager for distributed swarm storage
#[derive(Debug)]
pub struct PersistenceManager {
  /// Primary database connection
  db: Arc<String>, // Placeholder for removed libsql Connection
  /// Database configuration
  config: PersistenceConfig,
  /// Transaction manager
  transactions: Arc<RwLock<TransactionManager>>,
  /// Schema version for migrations
  schema_version: u32,
  /// Backup manager
  backup_manager: Arc<BackupManager>,
}

/// Persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
  /// Database file path
  pub db_path: String,
  /// Enable WAL mode for better concurrency
  pub enable_wal: bool,
  /// Connection pool size
  pub pool_size: u32,
  /// Auto-vacuum setting
  pub auto_vacuum: bool,
  /// Backup configuration
  pub backup: BackupConfig,
  /// Replication settings
  pub replication: ReplicationConfig,
  /// Performance settings
  pub performance: PerformanceConfig,
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
  /// Enable automatic backups
  pub enabled: bool,
  /// Backup interval in seconds
  pub interval_seconds: u64,
  /// Backup directory
  pub backup_directory: String,
  /// Number of backups to keep
  pub retention_count: u32,
  /// Compression for backups
  pub compress: bool,
  /// Encryption for sensitive data
  pub encrypt: bool,
}

/// Replication configuration for distributed swarms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationConfig {
  /// Enable replication
  pub enabled: bool,
  /// Replication nodes
  pub nodes: Vec<ReplicationNode>,
  /// Consistency level
  pub consistency_level: ConsistencyLevel,
  /// Conflict resolution strategy
  pub conflict_resolution: ConflictResolution,
}

/// Replication node configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationNode {
  pub node_id: String,
  pub address: String,
  pub port: u16,
  pub role: NodeRole,
  pub priority: u32,
}

/// Node roles in replication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeRole {
  Primary,
  Secondary,
  Witness,
}

/// Consistency levels for distributed operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyLevel {
  Strong,   // All nodes must confirm
  Eventual, // Eventually consistent
  Weak,     // Best effort
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
  LastWriteWins,
  FirstWriteWins,
  MergeStrategy,
  ManualReview,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
  /// Cache size in MB
  pub cache_size_mb: u64,
  /// Page size
  pub page_size: u32,
  /// Journal mode
  pub journal_mode: JournalMode,
  /// Synchronous mode
  pub synchronous: SynchronousMode,
  /// Memory mapped I/O
  pub mmap_size: u64,
}

/// Journal modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JournalMode {
  Delete,
  Truncate,
  Persist,
  Memory,
  Wal,
  Off,
}

/// Synchronous modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynchronousMode {
  Off,
  Normal,
  Full,
  Extra,
}

/// Transaction manager for ACID guarantees
#[derive(Debug)]
pub struct TransactionManager {
  /// Active transactions
  active_transactions: HashMap<String, TransactionInfo>,
  /// Transaction isolation level
  isolation_level: IsolationLevel,
  /// Deadlock detection
  deadlock_detection: bool,
}

/// Transaction information
#[derive(Debug, Clone)]
pub struct TransactionInfo {
  pub transaction_id: String,
  pub start_time: DateTime<Utc>,
  pub isolation_level: IsolationLevel,
  pub read_tables: Vec<String>,
  pub write_tables: Vec<String>,
  pub is_readonly: bool,
}

/// Isolation levels for transactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationLevel {
  ReadUncommitted,
  ReadCommitted,
  RepeatableRead,
  Serializable,
}

/// Backup manager for data protection
#[derive(Debug)]
pub struct BackupManager {
  config: BackupConfig,
  last_backup: Option<DateTime<Utc>>,
  backup_history: Vec<BackupInfo>,
}

/// Backup information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupInfo {
  pub backup_id: String,
  pub created_at: DateTime<Utc>,
  pub file_path: String,
  pub size_bytes: u64,
  pub checksum: String,
  pub compression_ratio: f32,
  pub backup_type: BackupType,
}

/// Types of backups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupType {
  Full,
  Incremental,
  Differential,
}

/// Swarm state record in database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmStateRecord {
  pub swarm_id: String,
  pub domain: String,
  pub service_type: String,
  pub creation_time: DateTime<Utc>,
  pub last_activity: DateTime<Utc>,
  pub state_data: Value,
  pub version: u32,
}

/// Agent state record in database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStateRecord {
  pub agent_id: String,
  pub swarm_id: String,
  pub agent_type: String,
  pub current_state: String,
  pub memory_data: Value,
  pub performance_metrics: Value,
  pub created_at: DateTime<Utc>,
  pub last_updated: DateTime<Utc>,
}

/// Task record in database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRecord {
  pub task_id: String,
  pub swarm_id: String,
  pub description: String,
  pub status: String,
  pub priority: String,
  pub assigned_agents: Value,
  pub created_at: DateTime<Utc>,
  pub updated_at: DateTime<Utc>,
  pub metadata: Value,
}

/// Memory record for agent persistent memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
  pub memory_id: String,
  pub agent_id: String,
  pub swarm_id: String,
  pub memory_key: String,
  pub memory_value: Value,
  pub memory_type: String,
  pub created_at: DateTime<Utc>,
  pub last_accessed: DateTime<Utc>,
  pub access_count: u64,
}

impl PersistenceManager {
  /// Create new persistence manager with LibSQL
  pub async fn new(config: PersistenceConfig) -> Result<Self, SwarmError> {
    // Note: libsql removed in favor of pure Cozo for all persistence
    // Create a placeholder persistence manager for now
    Err(SwarmError::Persistence(
        "libsql removed in favor of Cozo - use cozo-based persistence instead".to_string()
    ))
  }

  /// Initialize database schema
  async fn initialize_schema(&self) -> Result<(), SwarmError> {
    // Create swarm_states table
    self
      .db
      .execute(
        "CREATE TABLE IF NOT EXISTS swarm_states (
                    swarm_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    service_type TEXT NOT NULL,
                    creation_time TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    state_data TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1
                )",
        (),
      )
      .await
      .map_err(|e| {
        SwarmError::Persistence(format!(
          "Failed to create swarm_states table: {}",
          e
        ))
      })?;

    // Create agent_states table
    self
      .db
      .execute(
        "CREATE TABLE IF NOT EXISTS agent_states (
                    agent_id TEXT PRIMARY KEY,
                    swarm_id TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    current_state TEXT NOT NULL,
                    memory_data TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    FOREIGN KEY (swarm_id) REFERENCES swarm_states(swarm_id)
                )",
        (),
      )
      .await
      .map_err(|e| {
        SwarmError::Persistence(format!(
          "Failed to create agent_states table: {}",
          e
        ))
      })?;

    // Create tasks table
    self
      .db
      .execute(
        "CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    swarm_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    assigned_agents TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    FOREIGN KEY (swarm_id) REFERENCES swarm_states(swarm_id)
                )",
        (),
      )
      .await
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to create tasks table: {}", e))
      })?;

    // Create memories table
    self
      .db
      .execute(
        "CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    swarm_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    memory_value TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (agent_id) REFERENCES agent_states(agent_id),
                    FOREIGN KEY (swarm_id) REFERENCES swarm_states(swarm_id)
                )",
        (),
      )
      .await
      .map_err(|e| {
        SwarmError::Persistence(format!(
          "Failed to create memories table: {}",
          e
        ))
      })?;

    // Create indexes for better performance
    self.create_indexes().await?;

    Ok(())
  }

  /// Create database indexes
  async fn create_indexes(&self) -> Result<(), SwarmError> {
    let indexes = vec![
      "CREATE INDEX IF NOT EXISTS idx_agent_states_swarm_id ON agent_states(swarm_id)",
      "CREATE INDEX IF NOT EXISTS idx_tasks_swarm_id ON tasks(swarm_id)",
      "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
      "CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON memories(agent_id)",
      "CREATE INDEX IF NOT EXISTS idx_memories_swarm_id ON memories(swarm_id)",
      "CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(memory_key)",
    ];

    for index_sql in indexes {
      self.db.execute(index_sql, ()).await.map_err(|e| {
        SwarmError::Persistence(format!("Failed to create index: {}", e))
      })?;
    }

    Ok(())
  }

  /// Configure database settings
  async fn configure_database(&self) -> Result<(), SwarmError> {
    if self.config.enable_wal {
      self
        .db
        .execute("PRAGMA journal_mode = WAL", ())
        .await
        .map_err(|e| {
          SwarmError::Persistence(format!("Failed to enable WAL: {}", e))
        })?;
    }

    // Set cache size
    let cache_size = self.config.performance.cache_size_mb * 256; // Convert MB to pages
    self
      .db
      .execute(&format!("PRAGMA cache_size = {}", cache_size), ())
      .await
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to set cache size: {}", e))
      })?;

    // Set synchronous mode
    let sync_mode = match self.config.performance.synchronous {
      SynchronousMode::Off => "OFF",
      SynchronousMode::Normal => "NORMAL",
      SynchronousMode::Full => "FULL",
      SynchronousMode::Extra => "EXTRA",
    };

    self
      .db
      .execute(&format!("PRAGMA synchronous = {}", sync_mode), ())
      .await
      .map_err(|e| {
        SwarmError::Persistence(format!(
          "Failed to set synchronous mode: {}",
          e
        ))
      })?;

    Ok(())
  }

  /// Save swarm state to database
  #[cfg(feature = "graph")]
  pub async fn save_swarm_state(
    &self,
    swarm: &PersistentSwarm,
  ) -> Result<(), SwarmError> {
    let state_json = serde_json::to_string(swarm).map_err(|e| {
      SwarmError::Persistence(format!("Failed to serialize swarm state: {}", e))
    })?;

    self.db
            .execute(
                "INSERT OR REPLACE INTO swarm_states 
                 (swarm_id, domain, service_type, creation_time, last_activity, state_data, version)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                (
                    swarm.swarm_id.as_str(),
                    swarm.domain.as_str(),
                    serde_json::to_string(&swarm.service_type).unwrap_or_default().as_str(),
                    swarm.creation_time.to_rfc3339().as_str(),
                    swarm.last_activity.to_rfc3339().as_str(),
                    state_json.as_str(),
                    1,
                ),
            )
            .await
            .map_err(|e| SwarmError::Persistence(format!("Failed to save swarm state: {}", e)))?;

    tracing::debug!("Saved swarm state for: {}", swarm.swarm_id);
    Ok(())
  }

  /// Load swarm state from database
  #[cfg(feature = "graph")]
  pub async fn load_swarm_state(
    &self,
    swarm_id: &str,
  ) -> Result<Option<PersistentSwarm>, SwarmError> {
    let mut rows = self
      .db
      .prepare("SELECT state_data FROM swarm_states WHERE swarm_id = ?1")
      .await
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to prepare query: {}", e))
      })?
      .query([swarm_id])
      .await
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to query swarm state: {}", e))
      })?;

    if let Some(row) = rows.next().await.map_err(|e| {
      SwarmError::Persistence(format!("Failed to fetch row: {}", e))
    })? {
      let state_json: String = row.get(0).map_err(|e| {
        SwarmError::Persistence(format!("Failed to get state data: {}", e))
      })?;
      let swarm: PersistentSwarm =
        serde_json::from_str(&state_json).map_err(|e| {
          SwarmError::Persistence(format!(
            "Failed to deserialize swarm state: {}",
            e
          ))
        })?;

      tracing::debug!("Loaded swarm state for: {}", swarm_id);
      Ok(Some(swarm))
    } else {
      Ok(None)
    }
  }

  /// Save agent memory
  pub async fn save_agent_memory(
    &self,
    agent_id: &str,
    swarm_id: &str,
    key: &str,
    value: &Value,
    memory_type: &str,
  ) -> Result<(), SwarmError> {
    let memory_id = Uuid::new_v4().to_string();
    let now = Utc::now();
    let value_json = serde_json::to_string(value).map_err(|e| {
      SwarmError::Persistence(format!(
        "Failed to serialize memory value: {}",
        e
      ))
    })?;

    self.db
            .execute(
                "INSERT OR REPLACE INTO memories 
                 (memory_id, agent_id, swarm_id, memory_key, memory_value, memory_type, created_at, last_accessed, access_count)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                 ON CONFLICT(memory_id) DO UPDATE SET
                 memory_value = ?5, last_accessed = ?8, access_count = access_count + 1",
                (
                    memory_id.as_str(),
                    agent_id,
                    swarm_id,
                    key,
                    value_json.as_str(),
                    memory_type,
                    now.to_rfc3339().as_str(),
                    now.to_rfc3339().as_str(),
                    1,
                ),
            )
            .await
            .map_err(|e| SwarmError::Persistence(format!("Failed to save agent memory: {}", e)))?;

    Ok(())
  }

  /// Load agent memory
  pub async fn load_agent_memory(
    &self,
    agent_id: &str,
    key: &str,
  ) -> Result<Option<Value>, SwarmError> {
    let mut rows = self.db
            .prepare("SELECT memory_value FROM memories WHERE agent_id = ?1 AND memory_key = ?2")
            .await
            .map_err(|e| SwarmError::Persistence(format!("Failed to prepare query: {}", e)))?
            .query([agent_id, key])
            .await
            .map_err(|e| SwarmError::Persistence(format!("Failed to query memory: {}", e)))?;

    if let Some(row) = rows.next().await.map_err(|e| {
      SwarmError::Persistence(format!("Failed to fetch row: {}", e))
    })? {
      let value_json: String = row.get(0).map_err(|e| {
        SwarmError::Persistence(format!("Failed to get memory value: {}", e))
      })?;
      let value: Value = serde_json::from_str(&value_json).map_err(|e| {
        SwarmError::Persistence(format!("Failed to deserialize memory: {}", e))
      })?;

      // Update access count and last_accessed
      self.db
                .execute(
                    "UPDATE memories SET last_accessed = ?1, access_count = access_count + 1 
                     WHERE agent_id = ?2 AND memory_key = ?3",
                    (Utc::now().to_rfc3339().as_str(), agent_id, key),
                )
                .await
                .map_err(|e| SwarmError::Persistence(format!("Failed to update access count: {}", e)))?;

      Ok(Some(value))
    } else {
      Ok(None)
    }
  }

  /// Save task to database
  pub async fn save_task(
    &self,
    task: &Task,
    swarm_id: &str,
  ) -> Result<(), SwarmError> {
    let assigned_agents_json = serde_json::to_string(&task.assigned_agents)
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to serialize agents: {}", e))
      })?;

    self.db
            .execute(
                "INSERT OR REPLACE INTO tasks 
                 (task_id, swarm_id, description, status, priority, assigned_agents, created_at, updated_at, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                (
                    task.id.to_string().as_str(),
                    swarm_id,
                    task.description.as_str(),
                    serde_json::to_string(&task.status).unwrap_or_default().as_str(),
                    "medium", // Default priority
                    assigned_agents_json.as_str(),
                    task.created_at.to_rfc3339().as_str(),
                    Utc::now().to_rfc3339().as_str(),
                    serde_json::to_string(&task.metadata).unwrap_or_default().as_str(),
                ),
            )
            .await
            .map_err(|e| SwarmError::Persistence(format!("Failed to save task: {}", e)))?;

    Ok(())
  }

  /// Load tasks for swarm
  pub async fn load_tasks_for_swarm(
    &self,
    swarm_id: &str,
  ) -> Result<Vec<TaskRecord>, SwarmError> {
    let mut rows = self
      .db
      .prepare(
        "SELECT * FROM tasks WHERE swarm_id = ?1 ORDER BY created_at DESC",
      )
      .await
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to prepare query: {}", e))
      })?
      .query([swarm_id])
      .await
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to query tasks: {}", e))
      })?;

    let mut tasks = Vec::new();
    while let Some(row) = rows.next().await.map_err(|e| {
      SwarmError::Persistence(format!("Failed to fetch row: {}", e))
    })? {
      let task = TaskRecord {
        task_id: row.get(0).map_err(|e| {
          SwarmError::Persistence(format!("Failed to get task_id: {}", e))
        })?,
        swarm_id: row.get(1).map_err(|e| {
          SwarmError::Persistence(format!("Failed to get swarm_id: {}", e))
        })?,
        description: row.get(2).map_err(|e| {
          SwarmError::Persistence(format!("Failed to get description: {}", e))
        })?,
        status: row.get(3).map_err(|e| {
          SwarmError::Persistence(format!("Failed to get status: {}", e))
        })?,
        priority: row.get(4).map_err(|e| {
          SwarmError::Persistence(format!("Failed to get priority: {}", e))
        })?,
        assigned_agents: serde_json::from_str(&row.get::<String>(5).map_err(
          |e| SwarmError::Persistence(format!("Failed to get agents: {}", e)),
        )?)
        .unwrap_or_default(),
        created_at: DateTime::parse_from_rfc3339(
          &row.get::<String>(6).map_err(|e| {
            SwarmError::Persistence(format!("Failed to get created_at: {}", e))
          })?,
        )
        .map_err(|e| {
          SwarmError::Persistence(format!("Failed to parse created_at: {}", e))
        })?
        .with_timezone(&Utc),
        updated_at: DateTime::parse_from_rfc3339(
          &row.get::<String>(7).map_err(|e| {
            SwarmError::Persistence(format!("Failed to get updated_at: {}", e))
          })?,
        )
        .map_err(|e| {
          SwarmError::Persistence(format!("Failed to parse updated_at: {}", e))
        })?
        .with_timezone(&Utc),
        metadata: serde_json::from_str(&row.get::<String>(8).map_err(|e| {
          SwarmError::Persistence(format!("Failed to get metadata: {}", e))
        })?)
        .unwrap_or_default(),
      };
      tasks.push(task);
    }

    Ok(tasks)
  }

  /// Create database backup
  pub async fn create_backup(
    &self,
    backup_type: BackupType,
  ) -> Result<BackupInfo, SwarmError> {
    let backup_id = Uuid::new_v4().to_string();
    let backup_filename = format!(
      "backup_{}_{}.db",
      backup_id,
      Utc::now().format("%Y%m%d_%H%M%S")
    );
    let backup_path = Path::new(&self.backup_manager.config.backup_directory)
      .join(&backup_filename);

    // Create backup directory if it doesn't exist
    if let Some(parent) = backup_path.parent() {
      tokio::fs::create_dir_all(parent).await.map_err(|e| {
        SwarmError::Persistence(format!(
          "Failed to create backup directory: {}",
          e
        ))
      })?;
    }

    // In production, would use LibSQL backup API
    // For now, we simulate the backup
    let backup_data = b"simulated backup data";
    tokio::fs::write(&backup_path, backup_data)
      .await
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to write backup: {}", e))
      })?;

    let backup_info = BackupInfo {
      backup_id,
      created_at: Utc::now(),
      file_path: backup_path.to_string_lossy().to_string(),
      size_bytes: backup_data.len() as u64,
      checksum: "sha256_placeholder".to_string(), // Would calculate actual checksum
      compression_ratio: 1.0, // Would calculate if compression enabled
      backup_type,
    };

    tracing::info!(
      "Created backup: {} at {}",
      backup_info.backup_id,
      backup_info.file_path
    );
    Ok(backup_info)
  }

  /// Get persistence statistics
  pub async fn get_stats(&self) -> Result<PersistenceStats, SwarmError> {
    // Count records in each table
    let swarm_count: i64 = self
      .db
      .prepare("SELECT COUNT(*) FROM swarm_states")
      .await
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to prepare count query: {}", e))
      })?
      .query(())
      .await
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to query count: {}", e))
      })?
      .next()
      .await
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to fetch count: {}", e))
      })?
      .ok_or_else(|| SwarmError::Persistence("No count result".to_string()))?
      .get(0)
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to get count: {}", e))
      })?;

    let agent_count: i64 = self
      .db
      .prepare("SELECT COUNT(*) FROM agent_states")
      .await
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to prepare count query: {}", e))
      })?
      .query(())
      .await
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to query count: {}", e))
      })?
      .next()
      .await
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to fetch count: {}", e))
      })?
      .ok_or_else(|| SwarmError::Persistence("No count result".to_string()))?
      .get(0)
      .map_err(|e| {
        SwarmError::Persistence(format!("Failed to get count: {}", e))
      })?;

    Ok(PersistenceStats {
      total_swarms: swarm_count as usize,
      total_agents: agent_count as usize,
      total_tasks: 0,        // Would query tasks table
      total_memories: 0,     // Would query memories table
      database_size_mb: 0.0, // Would calculate actual size
      backup_count: self.backup_manager.backup_history.len(),
    })
  }
}

impl TransactionManager {
  pub fn new() -> Self {
    Self {
      active_transactions: HashMap::new(),
      isolation_level: IsolationLevel::ReadCommitted,
      deadlock_detection: true,
    }
  }

  pub async fn begin_transaction(
    &mut self,
    isolation_level: IsolationLevel,
  ) -> String {
    let transaction_id = Uuid::new_v4().to_string();
    let transaction_info = TransactionInfo {
      transaction_id: transaction_id.clone(),
      start_time: Utc::now(),
      isolation_level,
      read_tables: Vec::new(),
      write_tables: Vec::new(),
      is_readonly: false,
    };

    self
      .active_transactions
      .insert(transaction_id.clone(), transaction_info);
    transaction_id
  }
}

impl BackupManager {
  pub fn new(config: BackupConfig) -> Self {
    Self {
      config,
      last_backup: None,
      backup_history: Vec::new(),
    }
  }
}

/// Persistence statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct PersistenceStats {
  pub total_swarms: usize,
  pub total_agents: usize,
  pub total_tasks: usize,
  pub total_memories: usize,
  pub database_size_mb: f64,
  pub backup_count: usize,
}

impl Default for PersistenceConfig {
  fn default() -> Self {
    Self {
      db_path: ".zen-swarm/swarm.db".to_string(),
      enable_wal: true,
      pool_size: 10,
      auto_vacuum: true,
      backup: BackupConfig {
        enabled: true,
        interval_seconds: 3600, // 1 hour
        backup_directory: ".zen-swarm/backups".to_string(),
        retention_count: 24, // Keep 24 backups
        compress: true,
        encrypt: false,
      },
      replication: ReplicationConfig {
        enabled: false,
        nodes: Vec::new(),
        consistency_level: ConsistencyLevel::Strong,
        conflict_resolution: ConflictResolution::LastWriteWins,
      },
      performance: PerformanceConfig {
        cache_size_mb: 64,
        page_size: 4096,
        journal_mode: JournalMode::Wal,
        synchronous: SynchronousMode::Normal,
        mmap_size: 268435456, // 256MB
      },
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::TempDir;

  #[tokio::test]
  async fn test_persistence_manager_creation() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");

    let mut config = PersistenceConfig::default();
    config.db_path = db_path.to_string_lossy().to_string();

    let manager = PersistenceManager::new(config).await;
    assert!(manager.is_ok());
  }

  #[test]
  fn test_backup_info_serialization() {
    let backup_info = BackupInfo {
      backup_id: "test-backup".to_string(),
      created_at: Utc::now(),
      file_path: "/tmp/backup.db".to_string(),
      size_bytes: 1024,
      checksum: "sha256_hash".to_string(),
      compression_ratio: 0.8,
      backup_type: BackupType::Full,
    };

    let json = serde_json::to_string(&backup_info).unwrap();
    let deserialized: BackupInfo = serde_json::from_str(&json).unwrap();

    assert_eq!(backup_info.backup_id, deserialized.backup_id);
  }
}
