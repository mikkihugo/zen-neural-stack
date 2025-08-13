//! Plugin system for linting, formatting, and code quality automation
//!
//! Extensible plugin architecture for language-specific tooling with automatic
//! task completion when all quality gates pass. Each swarm shares the same database
//! but filters by swarm_id for performance and data consistency.

use crate::{SwarmError, SwarmResult};
use anyhow::Result;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Command as AsyncCommand;
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

/// Plugin manager for code quality automation
#[derive(Debug)]
pub struct PluginManager {
  /// Registered plugins by type
  plugins: Arc<DashMap<PluginType, Vec<Plugin>>>,
  /// Plugin execution history
  execution_history: Arc<RwLock<Vec<PluginExecution>>>,
  /// Quality gates configuration
  quality_gates: Arc<RwLock<Vec<QualityGate>>>,
  /// Shared database connection pool (all swarms use same DB, filtered by swarm_id)
  db_pool: Arc<Mutex<()>>, // Placeholder for actual database pool
  /// Plugin configuration per swarm
  swarm_plugin_configs: Arc<DashMap<String, SwarmPluginConfig>>,
}

/// Types of plugins for different quality aspects
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum PluginType {
  /// Linting plugins (credo, dialyzer, gleam format --check)
  Linting,
  /// Code formatting plugins (mix format, gleam format)
  Formatting,
  /// Testing plugins (mix test, gleam test)
  Testing,
  /// Build validation plugins
  BuildValidation,
  /// Security scanning plugins
  SecurityScanning,
  /// Performance benchmarking plugins
  PerformanceBenchmarking,
  /// Documentation generation plugins
  DocumentationGeneration,
  /// Dependency auditing plugins
  DependencyAuditing,
  /// Custom user-defined plugins
  Custom(String),
}

/// Plugin definition with execution capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plugin {
  pub plugin_id: String,
  pub name: String,
  pub plugin_type: PluginType,
  pub version: String,
  pub description: String,
  /// Programming languages this plugin supports
  pub supported_languages: Vec<String>,
  /// Command to execute the plugin
  pub executable: PluginExecutable,
  /// Configuration schema for this plugin
  pub config_schema: Value,
  /// Default configuration
  pub default_config: Value,
  /// Success criteria for plugin execution
  pub success_criteria: Vec<SuccessCriterion>,
  /// Whether this plugin blocks task completion on failure
  pub blocking: bool,
  /// Auto-retry configuration
  pub retry_config: RetryConfig,
  /// Plugin dependencies
  pub dependencies: Vec<String>,
  /// Resource requirements
  pub resource_requirements: ResourceRequirements,
  /// Learning capabilities
  pub learning_enabled: bool,
}

/// Plugin executable configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginExecutable {
  /// System command
  Command {
    command: String,
    args: Vec<String>,
    working_dir: Option<PathBuf>,
    environment: HashMap<String, String>,
  },
  /// Node.js plugin script
  NodeScript {
    script_path: PathBuf,
    args: Vec<String>,
    node_version: Option<String>,
  },
  /// Python plugin script
  PythonScript {
    script_path: PathBuf,
    args: Vec<String>,
    python_version: Option<String>,
  },
  /// Native binary
  Binary {
    binary_path: PathBuf,
    args: Vec<String>,
  },
  /// HTTP API call
  HttpApi {
    base_url: String,
    endpoint: String,
    method: String,
    headers: HashMap<String, String>,
  },
}

/// Success criteria for plugin execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuccessCriterion {
  /// Exit code must be zero
  ExitCodeZero,
  /// Exit code must match specific value
  ExitCodeEquals(i32),
  /// Stdout must contain specific text
  StdoutContains(String),
  /// Stdout must match regex pattern
  StdoutMatches(String),
  /// Stderr must be empty
  StderrEmpty,
  /// JSON output must match schema
  JsonOutputValid(Value),
  /// Performance metric must be below threshold
  PerformanceThreshold { metric: String, threshold: f64 },
  /// File must exist after execution
  FileExists(PathBuf),
  /// File content must match criteria
  FileContentMatches { file: PathBuf, pattern: String },
}

/// Retry configuration for plugin execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
  pub enabled: bool,
  pub max_attempts: u32,
  pub delay_seconds: u32,
  pub exponential_backoff: bool,
  pub retry_on_exit_codes: Vec<i32>,
}

/// Resource requirements for plugin execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
  pub max_memory_mb: Option<u64>,
  pub max_cpu_percent: Option<f32>,
  pub timeout_seconds: u32,
  pub requires_network: bool,
  pub requires_filesystem_access: bool,
}

/// Quality gate that must pass before task completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGate {
  pub gate_id: String,
  pub name: String,
  pub description: String,
  /// Languages this gate applies to
  pub applicable_languages: Vec<String>,
  /// Plugins that must pass for this gate
  pub required_plugins: Vec<String>,
  /// Minimum success rate (0.0 to 1.0)
  pub min_success_rate: f32,
  /// Whether this gate blocks task completion
  pub blocking: bool,
  /// Gate execution order
  pub execution_order: u32,
  /// Conditions for gate activation
  pub activation_conditions: Vec<ActivationCondition>,
}

/// Conditions that activate quality gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationCondition {
  /// Always activate
  Always,
  /// Activate for specific file patterns
  FilePattern(String),
  /// Activate for specific languages
  Language(String),
  /// Activate based on task metadata
  TaskMetadata { key: String, value: Value },
  /// Activate based on swarm configuration
  SwarmConfig { key: String, value: Value },
  /// Activate based on time/schedule
  Schedule { cron_expression: String },
}

/// Plugin execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginExecution {
  pub execution_id: String,
  pub plugin_id: String,
  pub swarm_id: String,
  pub task_id: Option<String>,
  pub agent_id: Option<String>,
  pub started_at: DateTime<Utc>,
  pub completed_at: Option<DateTime<Utc>>,
  pub status: ExecutionStatus,
  pub exit_code: Option<i32>,
  pub stdout: String,
  pub stderr: String,
  pub resource_usage: ResourceUsage,
  pub success_criteria_results: Vec<CriterionResult>,
  pub learning_data: Option<LearningData>,
}

/// Status of plugin execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExecutionStatus {
  Pending,
  Running,
  Completed,
  Failed { error: String },
  Timeout,
  Cancelled,
  Retrying { attempt: u32 },
}

/// Resource usage during plugin execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
  pub peak_memory_mb: f64,
  pub peak_cpu_percent: f32,
  pub execution_time_ms: u64,
  pub disk_io_mb: f64,
  pub network_io_mb: f64,
}

/// Result of checking a success criterion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriterionResult {
  pub criterion: SuccessCriterion,
  pub passed: bool,
  pub details: String,
  pub confidence: f32,
}

/// Learning data extracted from plugin execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningData {
  pub patterns_discovered: Vec<String>,
  pub performance_insights: Vec<String>,
  pub error_patterns: Vec<String>,
  pub optimization_suggestions: Vec<String>,
  pub language_specific_insights: HashMap<String, Vec<String>>,
}

/// Plugin configuration per swarm (all swarms share DB, filtered by swarm_id)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmPluginConfig {
  pub swarm_id: String,
  /// Enabled plugins for this swarm
  pub enabled_plugins: Vec<String>,
  /// Plugin-specific configurations
  pub plugin_configs: HashMap<String, Value>,
  /// Quality gates enabled for this swarm
  pub enabled_quality_gates: Vec<String>,
  /// Auto-completion settings
  pub auto_completion: AutoCompletionConfig,
  /// Learning settings
  pub learning_config: LearningConfig,
}

/// Auto-completion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoCompletionConfig {
  /// Enable automatic task completion when all gates pass
  pub enabled: bool,
  /// Require manual approval even if gates pass
  pub require_manual_approval: bool,
  /// Timeout for waiting for manual approval (seconds)
  pub approval_timeout_seconds: u32,
  /// Agents that can provide approval
  pub approval_agents: Vec<String>,
  /// Actions to take on completion
  pub completion_actions: Vec<CompletionAction>,
}

/// Actions taken when task completion conditions are met
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompletionAction {
  /// Mark task as completed in database
  MarkTaskCompleted,
  /// Notify specific agents
  NotifyAgents { agent_ids: Vec<String> },
  /// Trigger next phase of work
  TriggerNextPhase { phase_name: String },
  /// Create Git commit with results
  CreateGitCommit { message: String },
  /// Deploy code to environment
  Deploy { environment: String },
  /// Generate summary report
  GenerateReport { template: String },
  /// Archive task artifacts
  ArchiveArtifacts { storage_location: String },
  /// Update learning models
  UpdateLearningModels,
  /// Send webhook notification
  SendWebhook { url: String, payload: Value },
}

/// Learning configuration for plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
  /// Enable learning from plugin executions
  pub enabled: bool,
  /// Languages to focus learning on
  pub focus_languages: Vec<String>,
  /// Learning objectives
  pub objectives: Vec<LearningObjective>,
  /// Minimum confidence threshold for applying learned patterns
  pub confidence_threshold: f32,
  /// Maximum learning data retention (days)
  pub retention_days: u32,
}

/// Learning objectives for plugin system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningObjective {
  /// Learn to write better code in specific languages
  ImproveCodeQuality { language: String },
  /// Learn common error patterns and solutions
  ErrorPatternRecognition,
  /// Learn performance optimization techniques
  PerformanceOptimization,
  /// Learn code style preferences
  StyleConsistency,
  /// Learn testing best practices
  TestingBestPractices,
  /// Learn security patterns
  SecurityPatterns,
}

/// Shared database schema considerations (all swarms use same DB)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmDatabaseConfig {
  /// Single shared database path (e.g., .zen-swarm/shared.db)
  pub database_path: String,
  /// Whether to use separate schemas per swarm
  pub use_schema_per_swarm: bool,
  /// Table prefix strategy
  pub table_prefix_strategy: TablePrefixStrategy,
  /// Indexing strategy for swarm_id filtering
  pub indexing_strategy: IndexingStrategy,
  /// Data retention policies per swarm
  pub retention_policies: HashMap<String, RetentionPolicy>,
}

/// Table prefix strategies for shared database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TablePrefixStrategy {
  /// No prefixes, use swarm_id column filtering
  NoPrefix,
  /// Prefix tables with swarm_id (e.g., swarm_123_tasks)
  SwarmIdPrefix,
  /// Use separate schema per swarm
  SeparateSchema,
}

/// Indexing strategies for efficient swarm_id filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexingStrategy {
  /// Single column index on swarm_id
  SingleColumn,
  /// Composite indexes with swarm_id as first column
  CompositeIndexes,
  /// Partitioned tables by swarm_id
  Partitioned,
}

/// Data retention policy per swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
  pub plugin_executions_days: u32,
  pub learning_data_days: u32,
  pub error_logs_days: u32,
  pub performance_data_days: u32,
  pub auto_cleanup: bool,
}

impl PluginManager {
  /// Create new plugin manager with shared database
  pub async fn new() -> SwarmResult<Self> {
    Ok(Self {
      plugins: Arc::new(DashMap::new()),
      execution_history: Arc::new(RwLock::new(Vec::new())),
      quality_gates: Arc::new(RwLock::new(Vec::new())),
      db_pool: Arc::new(Mutex::new(())), // Placeholder
      swarm_plugin_configs: Arc::new(DashMap::new()),
    })
  }

  /// Register default language plugins (Elixir, Gleam, etc.)
  pub async fn register_default_plugins(&self) -> SwarmResult<()> {
    // Elixir linting with Credo
    let credo_plugin = Plugin {
      plugin_id: "elixir-credo".to_string(),
      name: "Credo Linter".to_string(),
      plugin_type: PluginType::Linting,
      version: "1.7.0".to_string(),
      description: "Static code analysis for Elixir".to_string(),
      supported_languages: vec!["elixir".to_string()],
      executable: PluginExecutable::Command {
        command: "mix".to_string(),
        args: vec![
          "credo".to_string(),
          "--format".to_string(),
          "json".to_string(),
        ],
        working_dir: None,
        environment: HashMap::new(),
      },
      config_schema: serde_json::json!({
          "type": "object",
          "properties": {
              "strict": {"type": "boolean", "default": false},
              "checks": {"type": "array", "items": {"type": "string"}}
          }
      }),
      default_config: serde_json::json!({"strict": false}),
      success_criteria: vec![SuccessCriterion::ExitCodeZero],
      blocking: true,
      retry_config: RetryConfig {
        enabled: true,
        max_attempts: 2,
        delay_seconds: 5,
        exponential_backoff: false,
        retry_on_exit_codes: vec![1],
      },
      dependencies: vec!["elixir".to_string(), "mix".to_string()],
      resource_requirements: ResourceRequirements {
        max_memory_mb: Some(512),
        max_cpu_percent: Some(80.0),
        timeout_seconds: 300,
        requires_network: false,
        requires_filesystem_access: true,
      },
      learning_enabled: true,
    };

    // Elixir formatting
    let mix_format_plugin = Plugin {
      plugin_id: "elixir-format".to_string(),
      name: "Mix Format".to_string(),
      plugin_type: PluginType::Formatting,
      version: "1.15.0".to_string(),
      description: "Code formatting for Elixir".to_string(),
      supported_languages: vec!["elixir".to_string()],
      executable: PluginExecutable::Command {
        command: "mix".to_string(),
        args: vec!["format".to_string(), "--check-formatted".to_string()],
        working_dir: None,
        environment: HashMap::new(),
      },
      config_schema: serde_json::json!({}),
      default_config: serde_json::json!({}),
      success_criteria: vec![SuccessCriterion::ExitCodeZero],
      blocking: true,
      retry_config: RetryConfig {
        enabled: false,
        max_attempts: 1,
        delay_seconds: 0,
        exponential_backoff: false,
        retry_on_exit_codes: vec![],
      },
      dependencies: vec!["elixir".to_string(), "mix".to_string()],
      resource_requirements: ResourceRequirements {
        max_memory_mb: Some(256),
        max_cpu_percent: Some(50.0),
        timeout_seconds: 60,
        requires_network: false,
        requires_filesystem_access: true,
      },
      learning_enabled: true,
    };

    // Gleam format plugin
    let gleam_format_plugin = Plugin {
      plugin_id: "gleam-format".to_string(),
      name: "Gleam Format".to_string(),
      plugin_type: PluginType::Formatting,
      version: "1.0.0".to_string(),
      description: "Code formatting for Gleam".to_string(),
      supported_languages: vec!["gleam".to_string()],
      executable: PluginExecutable::Command {
        command: "gleam".to_string(),
        args: vec!["format".to_string(), "--check".to_string()],
        working_dir: None,
        environment: HashMap::new(),
      },
      config_schema: serde_json::json!({}),
      default_config: serde_json::json!({}),
      success_criteria: vec![SuccessCriterion::ExitCodeZero],
      blocking: true,
      retry_config: RetryConfig {
        enabled: false,
        max_attempts: 1,
        delay_seconds: 0,
        exponential_backoff: false,
        retry_on_exit_codes: vec![],
      },
      dependencies: vec!["gleam".to_string()],
      resource_requirements: ResourceRequirements {
        max_memory_mb: Some(256),
        max_cpu_percent: Some(50.0),
        timeout_seconds: 60,
        requires_network: false,
        requires_filesystem_access: true,
      },
      learning_enabled: true,
    };

    // Gleam test plugin
    let gleam_test_plugin = Plugin {
      plugin_id: "gleam-test".to_string(),
      name: "Gleam Test".to_string(),
      plugin_type: PluginType::Testing,
      version: "1.0.0".to_string(),
      description: "Test runner for Gleam".to_string(),
      supported_languages: vec!["gleam".to_string()],
      executable: PluginExecutable::Command {
        command: "gleam".to_string(),
        args: vec!["test".to_string()],
        working_dir: None,
        environment: HashMap::new(),
      },
      config_schema: serde_json::json!({}),
      default_config: serde_json::json!({}),
      success_criteria: vec![SuccessCriterion::ExitCodeZero],
      blocking: true,
      retry_config: RetryConfig {
        enabled: true,
        max_attempts: 2,
        delay_seconds: 10,
        exponential_backoff: false,
        retry_on_exit_codes: vec![1],
      },
      dependencies: vec!["gleam".to_string()],
      resource_requirements: ResourceRequirements {
        max_memory_mb: Some(512),
        max_cpu_percent: Some(80.0),
        timeout_seconds: 600,
        requires_network: true, // May need to download dependencies
        requires_filesystem_access: true,
      },
      learning_enabled: true,
    };

    // Register all plugins
    self.register_plugin(credo_plugin).await?;
    self.register_plugin(mix_format_plugin).await?;
    self.register_plugin(gleam_format_plugin).await?;
    self.register_plugin(gleam_test_plugin).await?;

    // Create default quality gates
    self.create_default_quality_gates().await?;

    tracing::info!("Registered default plugins for Elixir and Gleam");
    Ok(())
  }

  /// Register a plugin
  pub async fn register_plugin(&self, plugin: Plugin) -> SwarmResult<()> {
    let mut plugins = self
      .plugins
      .entry(plugin.plugin_type.clone())
      .or_insert_with(Vec::new);
    plugins.push(plugin.clone());

    tracing::info!("Registered plugin: {} ({})", plugin.name, plugin.plugin_id);
    Ok(())
  }

  /// Execute plugins for quality gates and auto-complete task when all pass
  pub async fn execute_quality_gates(
    &self,
    swarm_id: &str,
    task_id: &str,
    language: &str,
    files: &[PathBuf],
  ) -> SwarmResult<QualityGateResults> {
    let swarm_config = self.get_swarm_config(swarm_id).await?;
    let quality_gates = self.quality_gates.read().await;

    let mut gate_results = Vec::new();
    let mut all_passed = true;

    // Execute applicable quality gates in order
    let mut applicable_gates: Vec<_> = quality_gates
      .iter()
      .filter(|gate| {
        gate.applicable_languages.contains(&language.to_string())
          && swarm_config.enabled_quality_gates.contains(&gate.gate_id)
      })
      .collect();

    applicable_gates.sort_by_key(|gate| gate.execution_order);

    for gate in applicable_gates {
      tracing::info!(
        "Executing quality gate: {} for swarm: {}",
        gate.name,
        swarm_id
      );

      let gate_result = self
        .execute_single_quality_gate(gate, swarm_id, task_id, files)
        .await?;
      let gate_passed = gate_result.success_rate >= gate.min_success_rate;

      if !gate_passed {
        all_passed = false;

        if gate.blocking {
          tracing::warn!(
            "Blocking quality gate failed: {} ({}%)",
            gate.name,
            gate_result.success_rate * 100.0
          );
          break; // Don't continue with remaining gates if blocking gate fails
        }
      }

      gate_results.push(gate_result);
    }

    let results = QualityGateResults {
      swarm_id: swarm_id.to_string(),
      task_id: task_id.to_string(),
      language: language.to_string(),
      all_gates_passed: all_passed,
      gate_results,
      execution_time_ms: 0, // Would be calculated
      auto_completed: false,
    };

    // Auto-complete task if all gates passed and auto-completion is enabled
    if all_passed && swarm_config.auto_completion.enabled {
      self.auto_complete_task(swarm_id, task_id, &results).await?;
    }

    Ok(results)
  }

  /// Execute a single quality gate
  async fn execute_single_quality_gate(
    &self,
    gate: &QualityGate,
    swarm_id: &str,
    task_id: &str,
    files: &[PathBuf],
  ) -> SwarmResult<QualityGateResult> {
    let mut plugin_results = Vec::new();
    let mut total_success = 0;
    let mut total_executed = 0;

    for plugin_id in &gate.required_plugins {
      if let Some(plugin) = self.find_plugin_by_id(plugin_id) {
        let execution_result = self
          .execute_plugin(&plugin, swarm_id, Some(task_id), files)
          .await?;

        let success = execution_result.status == ExecutionStatus::Completed
          && execution_result
            .success_criteria_results
            .iter()
            .all(|r| r.passed);

        if success {
          total_success += 1;
        }
        total_executed += 1;

        plugin_results.push(execution_result);
      }
    }

    let success_rate = if total_executed > 0 {
      total_success as f32 / total_executed as f32
    } else {
      0.0
    };

    Ok(QualityGateResult {
      gate_id: gate.gate_id.clone(),
      gate_name: gate.name.clone(),
      success_rate,
      plugin_results,
      passed: success_rate >= gate.min_success_rate,
      execution_time_ms: 0, // Would be calculated
    })
  }

  /// Execute a specific plugin
  async fn execute_plugin(
    &self,
    plugin: &Plugin,
    swarm_id: &str,
    task_id: Option<&str>,
    files: &[PathBuf],
  ) -> SwarmResult<PluginExecution> {
    let execution_id = Uuid::new_v4().to_string();
    let start_time = Utc::now();

    tracing::debug!(
      "Executing plugin: {} for swarm: {}",
      plugin.name,
      swarm_id
    );

    let mut execution = PluginExecution {
      execution_id: execution_id.clone(),
      plugin_id: plugin.plugin_id.clone(),
      swarm_id: swarm_id.to_string(),
      task_id: task_id.map(|s| s.to_string()),
      agent_id: None, // Could be filled in if agent context available
      started_at: start_time,
      completed_at: None,
      status: ExecutionStatus::Running,
      exit_code: None,
      stdout: String::new(),
      stderr: String::new(),
      resource_usage: ResourceUsage {
        peak_memory_mb: 0.0,
        peak_cpu_percent: 0.0,
        execution_time_ms: 0,
        disk_io_mb: 0.0,
        network_io_mb: 0.0,
      },
      success_criteria_results: Vec::new(),
      learning_data: None,
    };

    // Execute based on plugin type
    let result = match &plugin.executable {
      PluginExecutable::Command {
        command,
        args,
        working_dir,
        environment,
      } => {
        self
          .execute_command_plugin(
            command,
            args,
            working_dir,
            environment,
            files,
          )
          .await
      }
      PluginExecutable::NodeScript {
        script_path, args, ..
      } => self.execute_node_plugin(script_path, args, files).await,
      _ => {
        #[cfg(feature = "runtime")]
        return Err(SwarmError::Runtime(
          "Plugin execution type not implemented".to_string(),
        ));
        #[cfg(not(feature = "runtime"))]
        return Err(SwarmError::Internal(
          "Plugin execution type not implemented".to_string(),
        ));
      }
    };

    // Update execution record
    match result {
      Ok((exit_code, stdout, stderr)) => {
        execution.completed_at = Some(Utc::now());
        execution.status = ExecutionStatus::Completed;
        execution.exit_code = Some(exit_code);
        execution.stdout = stdout;
        execution.stderr = stderr;

        // Check success criteria
        execution.success_criteria_results = self
          .check_success_criteria(&plugin.success_criteria, &execution)
          .await;

        // Generate learning data if enabled
        if plugin.learning_enabled {
          execution.learning_data =
            self.generate_learning_data(&execution, plugin).await;
        }
      }
      Err(error) => {
        execution.completed_at = Some(Utc::now());
        execution.status = ExecutionStatus::Failed {
          error: error.to_string(),
        };
      }
    }

    // Store execution in history (filtered by swarm_id in database)
    let mut history = self.execution_history.write().await;
    history.push(execution.clone());

    Ok(execution)
  }

  /// Execute command-based plugin
  async fn execute_command_plugin(
    &self,
    command: &str,
    args: &[String],
    working_dir: &Option<PathBuf>,
    environment: &HashMap<String, String>,
    _files: &[PathBuf],
  ) -> Result<(i32, String, String), SwarmError> {
    let mut cmd = AsyncCommand::new(command);
    cmd.args(args);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    if let Some(dir) = working_dir {
      cmd.current_dir(dir);
    }

    for (key, value) in environment {
      cmd.env(key, value);
    }

    let output = cmd.output().await.map_err(|e| {
      #[cfg(feature = "runtime")]
      return SwarmError::Runtime(format!(
        "Failed to execute plugin command: {}",
        e
      ));
      #[cfg(not(feature = "runtime"))]
      return SwarmError::Internal(format!(
        "Failed to execute plugin command: {}",
        e
      ));
    })?;

    let exit_code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    Ok((exit_code, stdout, stderr))
  }

  /// Execute Node.js-based plugin
  async fn execute_node_plugin(
    &self,
    script_path: &PathBuf,
    args: &[String],
    _files: &[PathBuf],
  ) -> Result<(i32, String, String), SwarmError> {
    let mut cmd = AsyncCommand::new("node");
    cmd.arg(script_path);
    cmd.args(args);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let output = cmd.output().await.map_err(|e| {
      #[cfg(feature = "runtime")]
      return SwarmError::Runtime(format!(
        "Failed to execute Node.js plugin: {}",
        e
      ));
      #[cfg(not(feature = "runtime"))]
      return SwarmError::Internal(format!(
        "Failed to execute Node.js plugin: {}",
        e
      ));
    })?;

    let exit_code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    Ok((exit_code, stdout, stderr))
  }

  /// Check success criteria for plugin execution
  async fn check_success_criteria(
    &self,
    criteria: &[SuccessCriterion],
    execution: &PluginExecution,
  ) -> Vec<CriterionResult> {
    let mut results = Vec::new();

    for criterion in criteria {
      let result = match criterion {
        SuccessCriterion::ExitCodeZero => CriterionResult {
          criterion: criterion.clone(),
          passed: execution.exit_code == Some(0),
          details: format!("Exit code: {:?}", execution.exit_code),
          confidence: 1.0,
        },
        SuccessCriterion::StdoutContains(text) => CriterionResult {
          criterion: criterion.clone(),
          passed: execution.stdout.contains(text),
          details: format!(
            "Stdout contains '{}': {}",
            text,
            execution.stdout.contains(text)
          ),
          confidence: 0.9,
        },
        SuccessCriterion::StderrEmpty => CriterionResult {
          criterion: criterion.clone(),
          passed: execution.stderr.trim().is_empty(),
          details: format!(
            "Stderr empty: {}",
            execution.stderr.trim().is_empty()
          ),
          confidence: 1.0,
        },
        _ => CriterionResult {
          criterion: criterion.clone(),
          passed: false,
          details: "Criterion not implemented".to_string(),
          confidence: 0.0,
        },
      };

      results.push(result);
    }

    results
  }

  /// Generate learning data from plugin execution
  async fn generate_learning_data(
    &self,
    execution: &PluginExecution,
    plugin: &Plugin,
  ) -> Option<LearningData> {
    if !plugin.learning_enabled {
      return None;
    }

    let mut patterns_discovered = Vec::new();
    let mut error_patterns = Vec::new();
    let mut optimization_suggestions = Vec::new();

    // Analyze execution results for patterns
    if execution.exit_code == Some(0) {
      patterns_discovered.push("successful_execution".to_string());
    } else {
      error_patterns
        .push(format!("exit_code_{}", execution.exit_code.unwrap_or(-1)));
    }

    // Analyze stderr for common error patterns
    if !execution.stderr.is_empty() {
      if execution.stderr.contains("warning") {
        error_patterns.push("contains_warnings".to_string());
      }
      if execution.stderr.contains("error") {
        error_patterns.push("contains_errors".to_string());
      }
    }

    // Generate performance optimizations
    if execution.resource_usage.execution_time_ms > 30000 {
      // 30 seconds
      optimization_suggestions
        .push("consider_performance_optimization".to_string());
    }

    Some(LearningData {
      patterns_discovered,
      performance_insights: Vec::new(),
      error_patterns,
      optimization_suggestions,
      language_specific_insights: HashMap::new(),
    })
  }

  /// Auto-complete task when all quality gates pass
  async fn auto_complete_task(
    &self,
    swarm_id: &str,
    task_id: &str,
    _results: &QualityGateResults,
  ) -> SwarmResult<()> {
    let swarm_config = self.get_swarm_config(swarm_id).await?;

    tracing::info!(
      "Auto-completing task {} for swarm {} - all quality gates passed",
      task_id,
      swarm_id
    );

    // Execute completion actions
    for action in &swarm_config.auto_completion.completion_actions {
      match action {
        CompletionAction::MarkTaskCompleted => {
          // Update task status in database (filtered by swarm_id)
          tracing::info!("Marking task {} as completed in database", task_id);
          // Implementation would update database record
        }
        CompletionAction::NotifyAgents { agent_ids } => {
          for agent_id in agent_ids {
            tracing::info!("Notifying agent {} of task completion", agent_id);
            // Implementation would send notification
          }
        }
        CompletionAction::GenerateReport { template } => {
          tracing::info!(
            "Generating completion report using template: {}",
            template
          );
          // Implementation would generate report
        }
        CompletionAction::UpdateLearningModels => {
          tracing::info!("Updating learning models with completion data");
          // Implementation would update neural networks
        }
        _ => {
          tracing::debug!("Completion action not implemented: {:?}", action);
        }
      }
    }

    Ok(())
  }

  /// Find plugin by ID
  fn find_plugin_by_id(&self, plugin_id: &str) -> Option<Plugin> {
    for plugins_by_type in self.plugins.iter() {
      for plugin in plugins_by_type.value() {
        if plugin.plugin_id == plugin_id {
          return Some(plugin.clone());
        }
      }
    }
    None
  }

  /// Get swarm configuration (database query filtered by swarm_id)
  async fn get_swarm_config(
    &self,
    swarm_id: &str,
  ) -> SwarmResult<SwarmPluginConfig> {
    // Use database connection for configuration retrieval
    let _db_lock = self.db_pool.lock().await;

    if let Some(config) = self.swarm_plugin_configs.get(swarm_id) {
      Ok(config.clone())
    } else {
      // Load from database or create default
      let default_config = SwarmPluginConfig {
        swarm_id: swarm_id.to_string(),
        enabled_plugins: vec![
          "elixir-credo".to_string(),
          "elixir-format".to_string(),
          "gleam-format".to_string(),
          "gleam-test".to_string(),
        ],
        plugin_configs: HashMap::new(),
        enabled_quality_gates: vec![
          "code-quality".to_string(),
          "test-quality".to_string(),
        ],
        auto_completion: AutoCompletionConfig {
          enabled: true,
          require_manual_approval: false,
          approval_timeout_seconds: 300,
          approval_agents: Vec::new(),
          completion_actions: vec![
            CompletionAction::MarkTaskCompleted,
            CompletionAction::UpdateLearningModels,
            CompletionAction::GenerateReport {
              template: "standard".to_string(),
            },
          ],
        },
        learning_config: LearningConfig {
          enabled: true,
          focus_languages: vec!["elixir".to_string(), "gleam".to_string()],
          objectives: vec![
            LearningObjective::ImproveCodeQuality {
              language: "elixir".to_string(),
            },
            LearningObjective::ImproveCodeQuality {
              language: "gleam".to_string(),
            },
            LearningObjective::ErrorPatternRecognition,
          ],
          confidence_threshold: 0.8,
          retention_days: 90,
        },
      };

      self
        .swarm_plugin_configs
        .insert(swarm_id.to_string(), default_config.clone());
      Ok(default_config)
    }
  }

  /// Create default quality gates
  async fn create_default_quality_gates(&self) -> SwarmResult<()> {
    let mut gates = self.quality_gates.write().await;

    // Code quality gate
    gates.push(QualityGate {
      gate_id: "code-quality".to_string(),
      name: "Code Quality Gate".to_string(),
      description: "Ensures code meets linting and formatting standards"
        .to_string(),
      applicable_languages: vec!["elixir".to_string(), "gleam".to_string()],
      required_plugins: vec![
        "elixir-credo".to_string(),
        "elixir-format".to_string(),
        "gleam-format".to_string(),
      ],
      min_success_rate: 1.0, // 100% must pass
      blocking: true,
      execution_order: 1,
      activation_conditions: vec![ActivationCondition::Always],
    });

    // Test quality gate
    gates.push(QualityGate {
      gate_id: "test-quality".to_string(),
      name: "Test Quality Gate".to_string(),
      description: "Ensures all tests pass".to_string(),
      applicable_languages: vec!["elixir".to_string(), "gleam".to_string()],
      required_plugins: vec!["gleam-test".to_string()],
      min_success_rate: 1.0, // 100% must pass
      blocking: true,
      execution_order: 2,
      activation_conditions: vec![ActivationCondition::Always],
    });

    Ok(())
  }

  /// Get plugin statistics (filtered by swarm_id)
  pub async fn get_stats(
    &self,
    swarm_id: Option<&str>,
  ) -> SwarmResult<PluginStats> {
    let history = self.execution_history.read().await;

    let filtered_executions: Vec<_> = if let Some(swarm_id) = swarm_id {
      history.iter().filter(|e| e.swarm_id == swarm_id).collect()
    } else {
      history.iter().collect()
    };

    let total_executions = filtered_executions.len();
    let successful_executions = filtered_executions
      .iter()
      .filter(|e| e.status == ExecutionStatus::Completed)
      .count();

    Ok(PluginStats {
      total_plugins_registered: self
        .plugins
        .iter()
        .map(|p| p.value().len())
        .sum(),
      total_executions,
      successful_executions,
      failed_executions: total_executions - successful_executions,
      average_execution_time_ms: 0.0, // Would be calculated
      swarm_specific: swarm_id.is_some(),
    })
  }
}

/// Results of quality gate execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGateResults {
  pub swarm_id: String,
  pub task_id: String,
  pub language: String,
  pub all_gates_passed: bool,
  pub gate_results: Vec<QualityGateResult>,
  pub execution_time_ms: u64,
  pub auto_completed: bool,
}

/// Result of a single quality gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGateResult {
  pub gate_id: String,
  pub gate_name: String,
  pub success_rate: f32,
  pub plugin_results: Vec<PluginExecution>,
  pub passed: bool,
  pub execution_time_ms: u64,
}

/// Plugin system statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginStats {
  pub total_plugins_registered: usize,
  pub total_executions: usize,
  pub successful_executions: usize,
  pub failed_executions: usize,
  pub average_execution_time_ms: f64,
  pub swarm_specific: bool,
}

#[cfg(test)]
mod tests {
  use super::*;

  #[tokio::test]
  async fn test_plugin_manager_creation() {
    let manager = PluginManager::new().await;
    assert!(manager.is_ok());
  }

  #[tokio::test]
  async fn test_register_default_plugins() {
    let manager = PluginManager::new().await.unwrap();
    let result = manager.register_default_plugins().await;
    assert!(result.is_ok());

    // Check that plugins were registered
    assert!(!manager.plugins.is_empty());
  }

  #[test]
  fn test_success_criterion_serialization() {
    let criterion = SuccessCriterion::ExitCodeZero;
    let json = serde_json::to_string(&criterion).unwrap();
    let deserialized: SuccessCriterion = serde_json::from_str(&json).unwrap();

    match deserialized {
      SuccessCriterion::ExitCodeZero => {}
      _ => panic!("Deserialization failed"),
    }
  }
}
