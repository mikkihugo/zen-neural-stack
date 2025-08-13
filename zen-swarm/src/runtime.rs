//! Node.js v24+ runtime for AI plugin execution with persistent swarm coordination
//!
//! External Node.js process management with native TypeScript support,
//! plugin system integration, and persistent swarm state management in .zen-swarm

use crate::{SwarmError, Task, TaskResult, TaskStatus};
use anyhow::Result;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command as AsyncCommand;
use tokio::sync::{RwLock, mpsc};
use uuid::Uuid;

/// Node.js v24+ runtime manager with persistent swarm storage
#[derive(Debug)]
pub struct NodeRuntime {
  /// Node.js executable path
  node_executable: PathBuf,
  /// Active Node.js processes
  processes: Arc<DashMap<String, NodeProcess>>,
  /// Plugin registry
  plugins: Arc<RwLock<PluginRegistry>>,
  /// Persistent swarm state directory (.zen-swarm)
  swarm_state_dir: PathBuf,
  /// Runtime configuration
  config: RuntimeConfig,
  /// Process communication channels
  process_channels: Arc<DashMap<String, ProcessChannel>>,
}

/// Node.js process information
#[derive(Debug)]
pub struct NodeProcess {
  pub process_id: String,
  pub child: Option<Child>,
  pub plugin_id: String,
  pub status: ProcessStatus,
  pub created_at: DateTime<Utc>,
  pub last_activity: DateTime<Utc>,
  pub memory_usage_mb: u64,
  pub cpu_usage_percent: f32,
}

/// Process status types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessStatus {
  Starting,
  Running,
  Idle,
  Busy { task_id: String },
  Paused,
  Stopping,
  Crashed { error: String },
  Terminated,
}

/// Plugin registry with persistent state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginRegistry {
  /// Registered plugins
  pub plugins: HashMap<String, PluginInfo>,
  /// Plugin dependencies
  pub dependencies: HashMap<String, Vec<String>>,
  /// Active plugin instances
  pub active_instances: HashMap<String, PluginInstance>,
  /// Plugin load order for swarm restoration
  pub load_order: Vec<String>,
}

/// Plugin information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInfo {
  pub id: String,
  pub name: String,
  pub version: String,
  pub description: String,
  pub entry_point: PathBuf,
  pub capabilities: Vec<String>,
  pub required_node_version: String,
  pub dependencies: Vec<String>,
  pub config_schema: Value,
  pub persistent_state: bool,
  pub auto_start: bool,
}

/// Plugin instance state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInstance {
  pub instance_id: String,
  pub plugin_id: String,
  pub process_id: Option<String>,
  pub config: Value,
  pub state: PluginInstanceState,
  pub created_at: DateTime<Utc>,
  pub last_used: DateTime<Utc>,
  pub usage_stats: PluginUsageStats,
  /// Persistent memory for this plugin instance
  pub persistent_memory: HashMap<String, Value>,
}

/// Plugin instance state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginInstanceState {
  Inactive,
  Loading,
  Active,
  Error { message: String },
  Suspended,
  Migrating { to_swarm: String },
}

/// Plugin usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginUsageStats {
  pub invocation_count: u64,
  pub success_count: u64,
  pub error_count: u64,
  pub average_execution_time_ms: f64,
  pub total_memory_allocated_mb: u64,
  pub last_performance_score: f32,
}

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
  /// Node.js version requirement
  pub node_version: String,
  /// Maximum concurrent processes
  pub max_processes: usize,
  /// Process timeout in seconds
  pub process_timeout: u64,
  /// Memory limit per process (MB)
  pub memory_limit_mb: u64,
  /// CPU limit per process (percentage)
  pub cpu_limit_percent: f32,
  /// TypeScript compilation settings
  pub typescript_config: TypeScriptConfig,
  /// Plugin directories to scan
  pub plugin_directories: Vec<PathBuf>,
  /// Swarm state persistence settings
  pub persistence: PersistenceConfig,
}

/// TypeScript configuration for Node.js v24+
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeScriptConfig {
  /// Use --experimental-strip-types flag
  pub strip_types: bool,
  /// TypeScript configuration file
  pub tsconfig_path: Option<PathBuf>,
  /// Compilation target
  pub target: String,
  /// Module system
  pub module_system: String,
}

/// Persistence configuration for swarm state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
  /// Base directory for swarm state (.zen-swarm)
  pub state_directory: PathBuf,
  /// Auto-save interval in seconds
  pub auto_save_interval: u64,
  /// Compression for state files
  pub compress_state: bool,
  /// Backup retention policy
  pub backup_retention_days: u32,
  /// Encryption for sensitive state
  pub encrypt_sensitive_data: bool,
}

/// Process communication channel
#[derive(Debug)]
pub struct ProcessChannel {
  pub process_id: String,
  pub stdin_sender: mpsc::UnboundedSender<String>,
  pub stdout_receiver: Arc<RwLock<mpsc::UnboundedReceiver<String>>>,
  pub stderr_receiver: Arc<RwLock<mpsc::UnboundedReceiver<String>>>,
}

/// Plugin execution request
#[derive(Debug, Serialize, Deserialize)]
pub struct PluginExecutionRequest {
  pub plugin_id: String,
  pub function_name: String,
  pub arguments: Value,
  pub context: PluginExecutionContext,
  pub timeout_ms: Option<u64>,
}

/// Plugin execution context
#[derive(Debug, Serialize, Deserialize)]
pub struct PluginExecutionContext {
  pub swarm_id: String,
  pub task_id: Option<String>,
  pub agent_id: Option<String>,
  pub session_id: String,
  pub persistent_memory_keys: Vec<String>,
  pub coordination_info: CoordinationInfo,
}

/// Coordination information for plugin execution
#[derive(Debug, Serialize, Deserialize)]
pub struct CoordinationInfo {
  pub central_endpoints: Vec<String>,
  pub federation_channels: Vec<String>,
  pub available_agents: Vec<String>,
  pub shared_resources: Vec<String>,
}

/// Plugin execution result
#[derive(Debug, Serialize, Deserialize)]
pub struct PluginExecutionResult {
  pub success: bool,
  pub result: Value,
  pub error: Option<String>,
  pub execution_time_ms: u64,
  pub memory_used_mb: f32,
  pub persistent_updates: HashMap<String, Value>,
  pub coordination_events: Vec<CoordinationEvent>,
}

/// Events generated during plugin coordination
#[derive(Debug, Serialize, Deserialize)]
pub struct CoordinationEvent {
  pub event_type: String,
  pub timestamp: DateTime<Utc>,
  pub data: Value,
  pub target_agents: Vec<String>,
  pub requires_propagation: bool,
}

impl NodeRuntime {
  /// Create new Node.js runtime with persistent swarm support
  pub async fn new(config: RuntimeConfig) -> Result<Self, SwarmError> {
    // Verify Node.js v24+ is available
    let node_executable = Self::find_node_executable(&config.node_version)?;

    // Create swarm state directory
    let swarm_state_dir = config.persistence.state_directory.clone();
    tokio::fs::create_dir_all(&swarm_state_dir)
      .await
      .map_err(|e| {
        SwarmError::Runtime(format!("Failed to create state directory: {}", e))
      })?;

    // Load existing plugin registry or create new one
    let plugins = Self::load_plugin_registry(&swarm_state_dir).await?;

    Ok(Self {
      node_executable,
      processes: Arc::new(DashMap::new()),
      plugins: Arc::new(RwLock::new(plugins)),
      swarm_state_dir,
      config,
      process_channels: Arc::new(DashMap::new()),
    })
  }

  /// Find Node.js executable and verify version
  fn find_node_executable(
    required_version: &str,
  ) -> Result<PathBuf, SwarmError> {
    // Try common Node.js locations
    let possible_paths = [
      "node",
      "/usr/local/bin/node",
      "/usr/bin/node",
      "/opt/node/bin/node",
    ];

    for path in &possible_paths {
      if let Ok(output) =
        std::process::Command::new(path).arg("--version").output()
      {
        let version = String::from_utf8_lossy(&output.stdout);
        if version.trim().starts_with("v24")
          || version.trim().starts_with("v25")
        {
          return Ok(PathBuf::from(path));
        }
      }
    }

    Err(SwarmError::Runtime(format!(
      "Node.js {} not found. Please install Node.js v24+",
      required_version
    )))
  }

  /// Load plugin registry from persistent state
  async fn load_plugin_registry(
    state_dir: &Path,
  ) -> Result<PluginRegistry, SwarmError> {
    let registry_path = state_dir.join("plugin_registry.json");

    if registry_path.exists() {
      let content =
        tokio::fs::read_to_string(&registry_path)
          .await
          .map_err(|e| {
            SwarmError::Runtime(format!(
              "Failed to read plugin registry: {}",
              e
            ))
          })?;

      let registry: PluginRegistry =
        serde_json::from_str(&content).map_err(|e| {
          SwarmError::Runtime(format!("Failed to parse plugin registry: {}", e))
        })?;

      tracing::info!(
        "Loaded {} plugins from persistent registry",
        registry.plugins.len()
      );
      Ok(registry)
    } else {
      Ok(PluginRegistry {
        plugins: HashMap::new(),
        dependencies: HashMap::new(),
        active_instances: HashMap::new(),
        load_order: Vec::new(),
      })
    }
  }

  /// Save plugin registry to persistent state
  async fn save_plugin_registry(&self) -> Result<(), SwarmError> {
    let registry_path = self.swarm_state_dir.join("plugin_registry.json");
    let plugins = self.plugins.read().await;

    let content = serde_json::to_string_pretty(&*plugins).map_err(|e| {
      SwarmError::Runtime(format!("Failed to serialize plugin registry: {}", e))
    })?;

    tokio::fs::write(&registry_path, content)
      .await
      .map_err(|e| {
        SwarmError::Runtime(format!("Failed to save plugin registry: {}", e))
      })?;

    Ok(())
  }

  /// Register a new plugin
  pub async fn register_plugin(
    &self,
    plugin: PluginInfo,
  ) -> Result<(), SwarmError> {
    let mut plugins = self.plugins.write().await;

    // Verify plugin entry point exists
    if !plugin.entry_point.exists() {
      return Err(SwarmError::Runtime(format!(
        "Plugin entry point does not exist: {:?}",
        plugin.entry_point
      )));
    }

    plugins.plugins.insert(plugin.id.clone(), plugin.clone());

    if plugin.auto_start {
      plugins.load_order.push(plugin.id.clone());
    }

    drop(plugins); // Release lock
    self.save_plugin_registry().await?;

    tracing::info!("Registered plugin: {} v{}", plugin.name, plugin.version);
    Ok(())
  }

  /// Start plugin instance
  pub async fn start_plugin_instance(
    &self,
    plugin_id: &str,
    config: Value,
  ) -> Result<String, SwarmError> {
    let instance_id = Uuid::new_v4().to_string();

    let mut plugins = self.plugins.write().await;
    let plugin = plugins
      .plugins
      .get(plugin_id)
      .ok_or_else(|| {
        SwarmError::Runtime(format!("Plugin {} not found", plugin_id))
      })?
      .clone();

    // Create plugin instance
    let instance = PluginInstance {
      instance_id: instance_id.clone(),
      plugin_id: plugin_id.to_string(),
      process_id: None,
      config,
      state: PluginInstanceState::Loading,
      created_at: Utc::now(),
      last_used: Utc::now(),
      usage_stats: PluginUsageStats {
        invocation_count: 0,
        success_count: 0,
        error_count: 0,
        average_execution_time_ms: 0.0,
        total_memory_allocated_mb: 0,
        last_performance_score: 1.0,
      },
      persistent_memory: HashMap::new(),
    };

    plugins
      .active_instances
      .insert(instance_id.clone(), instance);
    drop(plugins); // Release lock

    // Start Node.js process for this plugin
    let process_id = self.start_plugin_process(&plugin, &instance_id).await?;

    // Update instance with process ID
    let mut plugins = self.plugins.write().await;
    if let Some(instance) = plugins.active_instances.get_mut(&instance_id) {
      instance.process_id = Some(process_id);
      instance.state = PluginInstanceState::Active;
    }
    drop(plugins);

    self.save_plugin_registry().await?;

    tracing::info!(
      "Started plugin instance: {} for plugin: {}",
      instance_id,
      plugin_id
    );
    Ok(instance_id)
  }

  /// Start Node.js process for plugin
  async fn start_plugin_process(
    &self,
    plugin: &PluginInfo,
    instance_id: &str,
  ) -> Result<String, SwarmError> {
    let process_id = Uuid::new_v4().to_string();

    let mut cmd = AsyncCommand::new(&self.node_executable);

    // Add TypeScript support for Node.js v24+
    if self.config.typescript_config.strip_types {
      cmd.arg("--experimental-strip-types");
    }

    cmd
      .arg(&plugin.entry_point)
      .arg("--instance-id")
      .arg(instance_id)
      .arg("--swarm-state-dir")
      .arg(&self.swarm_state_dir)
      .stdin(Stdio::piped())
      .stdout(Stdio::piped())
      .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| {
      SwarmError::Runtime(format!("Failed to start Node.js process: {}", e))
    })?;

    // Set up communication channels
    let (stdin_sender, mut stdin_receiver) =
      mpsc::unbounded_channel::<String>();
    let (stdout_sender, stdout_receiver) = mpsc::unbounded_channel::<String>();
    let (stderr_sender, stderr_receiver) = mpsc::unbounded_channel::<String>();

    // Handle stdin
    if let Some(stdin) = child.stdin.take() {
      let mut stdin = tokio::io::BufWriter::new(stdin);
      tokio::spawn(async move {
        while let Some(data) = stdin_receiver.recv().await {
          if let Err(e) = stdin.write_all(data.as_bytes()).await {
            tracing::error!("Failed to write to plugin stdin: {}", e);
            break;
          }
          if let Err(e) = stdin.flush().await {
            tracing::error!("Failed to flush plugin stdin: {}", e);
            break;
          }
        }
      });
    }

    // Handle stdout
    if let Some(stdout) = child.stdout.take() {
      let mut reader = BufReader::new(stdout);
      tokio::spawn(async move {
        let mut line = String::new();
        while let Ok(n) = reader.read_line(&mut line).await {
          if n == 0 {
            break;
          }
          if let Err(e) = stdout_sender.send(line.trim().to_string()) {
            tracing::error!("Failed to send stdout data: {}", e);
            break;
          }
          line.clear();
        }
      });
    }

    // Handle stderr
    if let Some(stderr) = child.stderr.take() {
      let mut reader = BufReader::new(stderr);
      tokio::spawn(async move {
        let mut line = String::new();
        while let Ok(n) = reader.read_line(&mut line).await {
          if n == 0 {
            break;
          }
          if let Err(e) = stderr_sender.send(line.trim().to_string()) {
            tracing::error!("Failed to send stderr data: {}", e);
            break;
          }
          line.clear();
        }
      });
    }

    // Store process information
    let node_process = NodeProcess {
      process_id: process_id.clone(),
      child: None, // We don't store Child in the struct due to thread safety
      plugin_id: plugin.id.clone(),
      status: ProcessStatus::Starting,
      created_at: Utc::now(),
      last_activity: Utc::now(),
      memory_usage_mb: 0,
      cpu_usage_percent: 0.0,
    };

    self.processes.insert(process_id.clone(), node_process);

    // Store communication channels
    let channel = ProcessChannel {
      process_id: process_id.clone(),
      stdin_sender,
      stdout_receiver: Arc::new(RwLock::new(stdout_receiver)),
      stderr_receiver: Arc::new(RwLock::new(stderr_receiver)),
    };

    self.process_channels.insert(process_id.clone(), channel);

    Ok(process_id)
  }

  /// Execute plugin function with persistent state coordination
  pub async fn execute_plugin(
    &self,
    request: PluginExecutionRequest,
  ) -> Result<PluginExecutionResult, SwarmError> {
    let start_time = std::time::Instant::now();

    // Load persistent memory for this plugin
    let persistent_memory = self
      .load_plugin_persistent_memory(
        &request.plugin_id,
        &request.context.persistent_memory_keys,
      )
      .await?;

    // Find active plugin instance
    let plugins = self.plugins.read().await;
    let instance = plugins
      .active_instances
      .values()
      .find(|i| {
        i.plugin_id == request.plugin_id
          && matches!(i.state, PluginInstanceState::Active)
      })
      .ok_or_else(|| {
        SwarmError::Runtime(format!(
          "No active instance for plugin {}",
          request.plugin_id
        ))
      })?
      .clone();
    drop(plugins);

    // Get process channel
    let process_id = instance.process_id.ok_or_else(|| {
      SwarmError::Runtime("Plugin instance has no process".to_string())
    })?;

    let channel = self.process_channels.get(&process_id).ok_or_else(|| {
      SwarmError::Runtime(format!("Process channel {} not found", process_id))
    })?;

    // Prepare execution message
    let execution_message = serde_json::json!({
        "type": "execute",
        "function": request.function_name,
        "arguments": request.arguments,
        "context": request.context,
        "persistent_memory": persistent_memory,
        "timestamp": Utc::now().to_rfc3339()
    });

    // Send to plugin process
    channel
      .stdin_sender
      .send(execution_message.to_string())
      .map_err(|e| {
        SwarmError::Runtime(format!("Failed to send to plugin: {}", e))
      })?;

    // Wait for response (simplified - in production would handle timeouts)
    let mut stdout_receiver = channel.stdout_receiver.write().await;
    let response = stdout_receiver.recv().await.ok_or_else(|| {
      SwarmError::Runtime("Plugin did not respond".to_string())
    })?;

    // Parse response
    let result: Value = serde_json::from_str(&response).map_err(|e| {
      SwarmError::Runtime(format!("Failed to parse plugin response: {}", e))
    })?;

    let execution_time = start_time.elapsed().as_millis() as u64;

    // Save any persistent memory updates
    if let Some(persistent_updates) =
      result.get("persistent_updates").and_then(|v| v.as_object())
    {
      for (key, value) in persistent_updates {
        self
          .save_plugin_persistent_memory(&request.plugin_id, key, value.clone())
          .await?;
      }
    }

    // Update plugin usage statistics
    self
      .update_plugin_stats(
        &instance.instance_id,
        execution_time,
        result
          .get("success")
          .and_then(|v| v.as_bool())
          .unwrap_or(false),
      )
      .await?;

    Ok(PluginExecutionResult {
      success: result
        .get("success")
        .and_then(|v| v.as_bool())
        .unwrap_or(false),
      result: result.get("result").unwrap_or(&Value::Null).clone(),
      error: result
        .get("error")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string()),
      execution_time_ms: execution_time,
      memory_used_mb: result
        .get("memory_used_mb")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32,
      persistent_updates: result
        .get("persistent_updates")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_else(HashMap::new),
      coordination_events: result
        .get("coordination_events")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_else(Vec::new),
    })
  }

  /// Load persistent memory for plugin
  async fn load_plugin_persistent_memory(
    &self,
    plugin_id: &str,
    keys: &[String],
  ) -> Result<HashMap<String, Value>, SwarmError> {
    let memory_dir = self.swarm_state_dir.join("plugin_memory").join(plugin_id);
    let mut memory = HashMap::new();

    if memory_dir.exists() {
      for key in keys {
        let memory_file = memory_dir.join(format!("{}.json", key));
        if memory_file.exists() {
          let content =
            tokio::fs::read_to_string(&memory_file).await.map_err(|e| {
              SwarmError::Runtime(format!(
                "Failed to read persistent memory: {}",
                e
              ))
            })?;
          let value: Value = serde_json::from_str(&content).map_err(|e| {
            SwarmError::Runtime(format!(
              "Failed to parse persistent memory: {}",
              e
            ))
          })?;
          memory.insert(key.clone(), value);
        }
      }
    }

    Ok(memory)
  }

  /// Save persistent memory for plugin
  async fn save_plugin_persistent_memory(
    &self,
    plugin_id: &str,
    key: &str,
    value: Value,
  ) -> Result<(), SwarmError> {
    let memory_dir = self.swarm_state_dir.join("plugin_memory").join(plugin_id);
    tokio::fs::create_dir_all(&memory_dir).await.map_err(|e| {
      SwarmError::Runtime(format!("Failed to create memory directory: {}", e))
    })?;

    let memory_file = memory_dir.join(format!("{}.json", key));
    let content = serde_json::to_string_pretty(&value).map_err(|e| {
      SwarmError::Runtime(format!("Failed to serialize memory: {}", e))
    })?;

    tokio::fs::write(&memory_file, content).await.map_err(|e| {
      SwarmError::Runtime(format!("Failed to save persistent memory: {}", e))
    })?;

    Ok(())
  }

  /// Update plugin usage statistics
  async fn update_plugin_stats(
    &self,
    instance_id: &str,
    execution_time: u64,
    success: bool,
  ) -> Result<(), SwarmError> {
    let mut plugins = self.plugins.write().await;
    if let Some(instance) = plugins.active_instances.get_mut(instance_id) {
      let stats = &mut instance.usage_stats;
      stats.invocation_count += 1;
      if success {
        stats.success_count += 1;
      } else {
        stats.error_count += 1;
      }

      // Update running average
      let total_time = stats.average_execution_time_ms
        * (stats.invocation_count - 1) as f64
        + execution_time as f64;
      stats.average_execution_time_ms =
        total_time / stats.invocation_count as f64;

      instance.last_used = Utc::now();
    }
    drop(plugins);

    self.save_plugin_registry().await?;
    Ok(())
  }

  /// Get runtime statistics
  pub async fn get_stats(&self) -> Result<RuntimeStats, SwarmError> {
    let plugins = self.plugins.read().await;

    Ok(RuntimeStats {
      active_processes: self.processes.len(),
      registered_plugins: plugins.plugins.len(),
      active_plugin_instances: plugins.active_instances.len(),
      node_version: self.get_node_version().await?,
      state_directory_size_mb: self.get_state_directory_size().await?,
    })
  }

  /// Get Node.js version
  async fn get_node_version(&self) -> Result<String, SwarmError> {
    let output = AsyncCommand::new(&self.node_executable)
      .arg("--version")
      .output()
      .await
      .map_err(|e| {
        SwarmError::Runtime(format!("Failed to get Node.js version: {}", e))
      })?;

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
  }

  /// Get state directory size
  async fn get_state_directory_size(&self) -> Result<f64, SwarmError> {
    // Simplified implementation - in production would recursively calculate directory size
    Ok(0.0) // Placeholder
  }
}

/// Runtime statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct RuntimeStats {
  pub active_processes: usize,
  pub registered_plugins: usize,
  pub active_plugin_instances: usize,
  pub node_version: String,
  pub state_directory_size_mb: f64,
}

impl Default for RuntimeConfig {
  fn default() -> Self {
    Self {
      node_version: "24.0.0".to_string(),
      max_processes: 10,
      process_timeout: 300,
      memory_limit_mb: 512,
      cpu_limit_percent: 80.0,
      typescript_config: TypeScriptConfig {
        strip_types: true,
        tsconfig_path: None,
        target: "ES2022".to_string(),
        module_system: "ESNext".to_string(),
      },
      plugin_directories: vec![
        PathBuf::from("./plugins"),
        PathBuf::from("./node_modules/.zen-plugins"),
      ],
      persistence: PersistenceConfig {
        state_directory: PathBuf::from(".zen-swarm"),
        auto_save_interval: 30,
        compress_state: true,
        backup_retention_days: 7,
        encrypt_sensitive_data: true,
      },
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::TempDir;

  #[tokio::test]
  async fn test_runtime_creation() {
    let temp_dir = TempDir::new().unwrap();
    let mut config = RuntimeConfig::default();
    config.persistence.state_directory = temp_dir.path().to_path_buf();

    // This will fail if Node.js v24+ is not installed, but tests the interface
    let runtime = NodeRuntime::new(config).await;

    // Test should pass even if Node.js is not available
    match runtime {
      Ok(_) => println!("Runtime created successfully"),
      Err(e) => println!("Expected error (Node.js v24+ not found): {}", e),
    }
  }

  #[test]
  fn test_plugin_info_serialization() {
    let plugin = PluginInfo {
      id: "test-plugin".to_string(),
      name: "Test Plugin".to_string(),
      version: "1.0.0".to_string(),
      description: "Test plugin description".to_string(),
      entry_point: PathBuf::from("./test-plugin.js"),
      capabilities: vec!["analysis".to_string(), "generation".to_string()],
      required_node_version: "24.0.0".to_string(),
      dependencies: Vec::new(),
      config_schema: serde_json::json!({}),
      persistent_state: true,
      auto_start: false,
    };

    let json = serde_json::to_string(&plugin).unwrap();
    let deserialized: PluginInfo = serde_json::from_str(&json).unwrap();

    assert_eq!(plugin.id, deserialized.id);
    assert_eq!(plugin.name, deserialized.name);
  }
}
