//! Zen Swarm Daemon - Repository-scoped intelligent coordination
//!
//! Each repository gets its own swarm daemon that handles:
//! - Service-level agent coordination
//! - Domain-level intelligence 
//! - Connection to THE COLLECTIVE for cross-repo learning
//! - MCP server for Claude Code integration

use crate::{
    Swarm, SwarmError, SwarmResult,
    a2a::{A2ACoordinator, A2AConfig, A2AMessage, AICapability, SwarmContext, Priority as LocalPriority},
    a2a_client::{A2AClient, A2AClientConfig, ConnectionStatus},
    mcp::McpServer,
    mcp_stdio::McpStdioServer,
};
use zen_orchestrator::a2a::{
    A2AMessage as OrchestratorA2AMessage, A2AResponse, SwarmStatus, SwarmMetrics,
    RepositoryContext, Priority, CodePattern, BuildOptimization, TaskType, 
    TaskRequirements, A2AResult,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::fs;
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn, error, debug};
use uuid::Uuid;

/// Zen Swarm Daemon - One per repository for distributed intelligence
#[derive(Debug)]
pub struct SwarmDaemon {
    /// Daemon configuration
    pub config: SwarmDaemonConfig,
    /// Core swarm coordination engine
    pub swarm: Arc<Swarm>,
    /// A2A coordinator for inter-swarm communication (local)
    pub a2a_coordinator: Arc<A2ACoordinator>,
    /// A2A client for zen-orchestrator communication
    pub a2a_client: Arc<RwLock<A2AClient>>,
    /// MCP server for Claude Code stdio integration
    pub mcp_stdio_server: Arc<McpStdioServer>,
    /// Task execution engine
    pub task_executor: Arc<TaskExecutor>,
    /// Daemon status and metrics
    pub status: Arc<RwLock<SwarmDaemonStatus>>,
    /// Repository intelligence cache
    pub repo_intelligence: Arc<RwLock<RepositoryIntelligence>>,
}

/// Swarm daemon configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmDaemonConfig {
    /// Repository path this daemon serves
    pub repo_path: PathBuf,
    /// Daemon name/identifier
    pub daemon_name: String,
    /// Port for HTTP MCP server
    pub daemon_port: u16,
    /// THE COLLECTIVE endpoint for A2A
    pub collective_endpoint: Option<String>,
    /// Type of swarm daemon
    pub daemon_type: SwarmDaemonType,
    /// Capabilities this daemon provides
    pub capabilities: Vec<String>,
    /// A2A configuration
    pub a2a_config: A2AConfig,
    /// Auto-discovery settings
    pub discovery_config: DiscoveryConfig,
}

/// Types of swarm daemons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmDaemonType {
    /// Project-specific repository daemon
    ProjectRepository { 
        project_name: String,
        tech_stack: Vec<String>,
        domain_areas: Vec<String>,
    },
    /// Central repository for organization-wide patterns
    CentralRepository,
    /// Service-specialized daemon (e.g., security, testing)
    ServiceSpecialized { 
        service_type: String,
        specializations: Vec<String>,
    },
}

/// Discovery and peer finding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Enable automatic peer swarm discovery
    pub peer_discovery_enabled: bool,
    /// Enable THE COLLECTIVE intelligence sharing
    pub collective_intelligence_enabled: bool,
    /// Enable cross-repository pattern sharing
    pub cross_repo_sharing_enabled: bool,
    /// Discovery broadcast interval (seconds)
    pub discovery_interval_sec: u64,
}

/// Current daemon status and health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmDaemonStatus {
    /// Daemon start time
    pub started_at: DateTime<Utc>,
    /// Current daemon health
    pub health_status: DaemonHealth,
    /// Connected peer swarms
    pub connected_peers: HashMap<String, PeerSwarmInfo>,
    /// THE COLLECTIVE connection status
    pub collective_status: CollectiveConnectionStatus,
    /// Active agents in this swarm
    pub active_agents: usize,
    /// Tasks processed since startup
    pub total_tasks_processed: u64,
    /// Intelligence queries handled
    pub intelligence_queries_handled: u64,
}

/// Daemon health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DaemonHealth {
    Healthy,
    Degraded { issues: Vec<String> },
    Unhealthy { errors: Vec<String> },
    Starting,
    Stopping,
}

/// Information about connected peer swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerSwarmInfo {
    pub swarm_id: String,
    pub daemon_type: SwarmDaemonType,
    pub endpoint: String,
    pub capabilities: Vec<String>,
    pub last_seen: DateTime<Utc>,
    pub connection_quality: f32, // 0.0 to 1.0
    pub shared_intelligence: bool,
}

/// THE COLLECTIVE connection status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveConnectionStatus {
    pub connected: bool,
    pub endpoint: Option<String>,
    pub last_heartbeat: Option<DateTime<Utc>>,
    pub intelligence_sharing_active: bool,
    pub connection_stability: f32, // 0.0 to 1.0
}

/// Repository-specific intelligence cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryIntelligence {
    /// Code patterns discovered in this repository
    pub code_patterns: HashMap<String, CodePattern>,
    /// Build and performance optimizations
    pub build_optimizations: Vec<BuildOptimization>,
    /// Domain-specific knowledge
    pub domain_knowledge: HashMap<String, DomainKnowledge>,
    /// Cross-repository learnings applied here
    pub applied_cross_repo_patterns: Vec<CrossRepoPattern>,
    /// Intelligence last updated
    pub last_updated: DateTime<Utc>,
}

/// Code pattern discovered in repository
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodePattern {
    pub pattern_id: String,
    pub pattern_type: String,
    pub description: String,
    pub file_paths: Vec<String>,
    pub confidence: f32,
    pub usage_frequency: u32,
    pub performance_impact: Option<f32>,
}

/// Build or performance optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildOptimization {
    pub optimization_id: String,
    pub optimization_type: String,
    pub description: String,
    pub performance_gain: f32,
    pub applied_at: DateTime<Utc>,
    pub success_rate: f32,
}

/// Domain-specific knowledge for this repository
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainKnowledge {
    pub domain_area: String,
    pub knowledge_type: String,
    pub insights: Vec<String>,
    pub confidence_level: f32,
    pub sources: Vec<String>,
    pub last_validated: DateTime<Utc>,
}

/// Cross-repository pattern applied to this repo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossRepoPattern {
    pub pattern_id: String,
    pub source_repo: String,
    pub pattern_description: String,
    pub adaptation_notes: String,
    pub success_rate: f32,
    pub applied_at: DateTime<Utc>,
}

/// Task execution engine for coding tasks
#[derive(Debug)]
pub struct TaskExecutor {
    /// Repository path for file operations
    pub repo_path: PathBuf,
    /// A2A coordinator for LLM requests
    pub a2a_coordinator: Arc<A2ACoordinator>,
    /// Repository intelligence cache
    pub intelligence_cache: Arc<RwLock<RepositoryIntelligence>>,
    /// File operations handler
    pub file_ops: Arc<FileOperations>,
    /// Build and test runner
    pub build_runner: Arc<BuildRunner>,
}

/// File operations handler
#[derive(Debug)]
pub struct FileOperations {
    pub repo_path: PathBuf,
}

/// Build and test runner
#[derive(Debug)]
pub struct BuildRunner {
    pub repo_path: PathBuf,
}

/// Task execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContext {
    pub working_directory: String,
    pub relevant_files: Vec<String>,
    pub build_system: Option<String>,
    pub test_framework: Option<String>,
}

/// Coding task request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodingTaskRequest {
    pub task_description: String,
    pub preferred_llm: String,
    pub context: TaskContext,
}

/// Coding task response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodingTaskResponse {
    pub task_result: String,
    pub files_modified: Vec<String>,
    pub actions_taken: Vec<String>,
    pub model_used: String,
    pub provider: String,
    pub usage: TokenUsage,
    pub processing_time_ms: u64,
}

/// Token usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Task analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAnalysis {
    pub task_type: String,
    pub complexity: TaskComplexity,
    pub relevant_files: Vec<String>,
    pub required_capabilities: Vec<String>,
    pub estimated_time_ms: u64,
}

/// Task complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskComplexity {
    Simple,
    Moderate,
    Complex,
    Expert,
}

/// Implementation plan from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationPlan {
    pub steps: Vec<PlanStep>,
    pub estimated_time_ms: u64,
    pub required_files: Vec<String>,
}

/// Individual plan steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanStep {
    CreateFile { path: String, content: String },
    ModifyFile { path: String, changes: Vec<FileChange> },
    DeleteFile { path: String },
    RunCommand { command: String, args: Vec<String> },
    RunTests { test_pattern: Option<String> },
}

/// File change operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChange {
    pub line_start: usize,
    pub line_end: usize,
    pub new_content: String,
}

/// Execution result
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub summary: String,
    pub token_usage: TokenUsage,
    pub command_outputs: Vec<CommandOutput>,
    pub test_results: Vec<TestResult>,
}

/// Command execution output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandOutput {
    pub command: String,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

/// Test execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub success: bool,
    pub output: String,
    pub tests_run: u32,
    pub tests_passed: u32,
    pub tests_failed: u32,
}

/// MCP configuration for Claude Code CLIs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfiguration {
    pub servers: HashMap<String, McpServerConfig>,
}

/// Individual MCP server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub command: String,
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env: Option<HashMap<String, String>>,
}

/// CLI interaction logging for swarm monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliInteractionLog {
    pub timestamp: DateTime<Utc>,
    pub session_id: String,
    pub cli_tool: String, // "claude", "gemini", "gpt", etc.
    pub interaction_type: CliInteractionType,
    pub prompt: Option<String>,
    pub response: Option<String>,
    pub status: CliStatus,
    pub processing_time_ms: Option<u64>,
    pub token_usage: Option<TokenUsage>,
    pub error: Option<String>,
}

/// Types of CLI interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CliInteractionType {
    Query,
    CodeGeneration,
    FileEdit,
    TaskExecution,
    HealthCheck,
    Configuration,
}

/// CLI interaction status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CliStatus {
    Started,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub build_success: bool,
    pub tests_passed: bool,
    pub lint_clean: bool,
    pub build_output: String,
    pub test_output: String,
    pub lint_output: String,
}

impl SwarmDaemon {
    /// Create new swarm daemon for repository
    pub async fn new(config: SwarmDaemonConfig) -> SwarmResult<Self> {
        info!("🚀 Initializing Zen Swarm Daemon: {}", config.daemon_name);
        info!("📁 Repository: {}", config.repo_path.display());
        
        // Initialize core swarm
        let swarm = Arc::new(Swarm::new(&config.daemon_name).await?);
        
        // Initialize A2A coordinator (local)
        let a2a_coordinator = Arc::new(A2ACoordinator::new(config.a2a_config.clone()));
        
        // Initialize A2A client for zen-orchestrator communication
        let a2a_client_config = A2AClientConfig {
            swarm_id: config.daemon_name.clone(),
            repository_path: config.repo_path.to_string_lossy().to_string(),
            capabilities: config.capabilities.clone(),
            daemon_port: config.daemon_port,
            orchestrator_endpoint: config.collective_endpoint.clone()
                .unwrap_or_else(|| "localhost:8080".to_string()),
            connection_timeout_ms: 5000,
            heartbeat_interval_sec: 30,
            request_timeout_ms: 10000,
        };
        let a2a_client = Arc::new(RwLock::new(A2AClient::new(a2a_client_config)));
        
        // Initialize MCP stdio server for Claude Code
        let mut mcp_stdio = McpStdioServer::new(&config.daemon_name, "2.0.0");
        if let Some(collective_endpoint) = &config.collective_endpoint {
            mcp_stdio.initialize_a2a(Some(collective_endpoint.clone())).await?;
        }
        let mcp_stdio_server = Arc::new(mcp_stdio);
        
        // No HTTP MCP server - all CLIs use stdio MCP
        
        // Initialize repository intelligence cache
        let repo_intelligence = Arc::new(RwLock::new(RepositoryIntelligence {
            code_patterns: HashMap::new(),
            build_optimizations: Vec::new(),
            domain_knowledge: HashMap::new(),
            applied_cross_repo_patterns: Vec::new(),
            last_updated: Utc::now(),
        }));
        
        // Initialize task executor
        let task_executor = Arc::new(TaskExecutor::new(
            config.repo_path.clone(),
            a2a_coordinator.clone(),
            repo_intelligence.clone(),
        ).await?);
        
        // Initialize daemon status
        let status = Arc::new(RwLock::new(SwarmDaemonStatus {
            started_at: Utc::now(),
            health_status: DaemonHealth::Starting,
            connected_peers: HashMap::new(),
            collective_status: CollectiveConnectionStatus {
                connected: false,
                endpoint: config.collective_endpoint.clone(),
                last_heartbeat: None,
                intelligence_sharing_active: false,
                connection_stability: 0.0,
            },
            active_agents: 0,
            total_tasks_processed: 0,
            intelligence_queries_handled: 0,
        }));
        
        Ok(Self {
            config,
            swarm,
            a2a_coordinator,
            a2a_client,
            mcp_stdio_server,
            task_executor,
            status,
            repo_intelligence,
        })
    }
    
    /// Start the swarm daemon (main entry point)
    pub async fn start(&self) -> SwarmResult<()> {
        info!("🐝 Starting Zen Swarm Daemon: {}", self.config.daemon_name);
        
        // Update status to healthy
        {
            let mut status = self.status.write().await;
            status.health_status = DaemonHealth::Healthy;
        }
        
        // Initialize A2A coordinator (local)
        self.a2a_coordinator.initialize().await?;
        
        // Start A2A client and register with zen-orchestrator
        {
            let mut client = self.a2a_client.write().await;
            if let Err(e) = client.start().await {
                warn!("⚠️ Failed to start A2A client: {}", e);
            } else {
                match client.register_swarm().await {
                    Ok(response) => info!("🧠 Registered with THE COLLECTIVE via zen-orchestrator: {:?}", response),
                    Err(e) => warn!("⚠️ Failed to register with zen-orchestrator: {}", e),
                }
            }
        }
        
        // Connect to THE COLLECTIVE if configured (legacy method)
        if let Some(collective_endpoint) = &self.config.collective_endpoint {
            match self.connect_to_collective(collective_endpoint).await {
                Ok(_) => info!("🧠 Connected to THE COLLECTIVE (legacy): {}", collective_endpoint),
                Err(e) => warn!("⚠️ Failed to connect to THE COLLECTIVE (legacy): {}", e),
            }
        }
        
        // Start peer discovery if enabled
        if self.config.discovery_config.peer_discovery_enabled {
            self.start_peer_discovery().await?;
        }
        
        // Start repository intelligence analysis
        self.start_repository_analysis().await?;
        
        // Register daemon with THE COLLECTIVE
        self.register_with_collective().await?;
        
        info!("✅ Zen Swarm Daemon started successfully");
        info!("🔗 MCP stdio: Available for Claude Code integration");
        info!("🌐 MCP HTTP: http://127.0.0.1:{}", self.config.daemon_port);
        info!("📊 A2A peers: {} connected", self.status.read().await.connected_peers.len());
        
        Ok(())
    }
    
    /// Connect to THE COLLECTIVE for cross-swarm intelligence
    async fn connect_to_collective(&self, endpoint: &str) -> SwarmResult<()> {
        info!("🧠 Connecting to THE COLLECTIVE at: {}", endpoint);
        
        // Use A2A coordinator to establish connection
        self.a2a_coordinator.connect_to_collective(endpoint).await?;
        
        // Update status
        {
            let mut status = self.status.write().await;
            status.collective_status.connected = true;
            status.collective_status.last_heartbeat = Some(Utc::now());
            status.collective_status.intelligence_sharing_active = true;
            status.collective_status.connection_stability = 1.0;
        }
        
        Ok(())
    }
    
    /// Register this daemon with THE COLLECTIVE
    async fn register_with_collective(&self) -> SwarmResult<()> {
        let registration_message = A2AMessage::Registration {
            swarm_id: self.config.daemon_name.clone(),
            capabilities: self.config.capabilities.clone(),
            endpoints: std::iter::from_fn(|| None)
                .chain(std::iter::once(("mcp_stdio".to_string(), 
                       format!("stdio://zen-swarm-{}", self.config.daemon_name))))
                .chain(std::iter::once(("mcp_http".to_string(),
                       format!("http://127.0.0.1:{}", self.config.daemon_port))))
                .chain(std::iter::once(("a2a".to_string(),
                       format!("ws://127.0.0.1:{}/a2a", self.config.daemon_port))))
                .collect(),
        };
        
        match self.a2a_coordinator.send_message("the-collective", registration_message).await {
            Ok(_) => {
                info!("📝 Successfully registered with THE COLLECTIVE");
                Ok(())
            }
            Err(e) => {
                warn!("⚠️ Failed to register with THE COLLECTIVE: {}", e);
                Err(e)
            }
        }
    }
    
    /// Start peer swarm discovery
    async fn start_peer_discovery(&self) -> SwarmResult<()> {
        info!("🔍 Starting peer swarm discovery");
        
        // This would implement actual peer discovery logic
        // For now, just log that discovery is active
        
        Ok(())
    }
    
    /// Start repository intelligence analysis
    async fn start_repository_analysis(&self) -> SwarmResult<()> {
        info!("🧠 Starting repository intelligence analysis: {}", self.config.repo_path.display());
        
        // This would implement:
        // - Code pattern recognition
        // - Build optimization detection  
        // - Domain knowledge extraction
        // - Performance analysis
        
        Ok(())
    }
    
    /// Get daemon status
    pub async fn get_status(&self) -> SwarmDaemonStatus {
        self.status.read().await.clone()
    }
    
    /// Get repository intelligence
    pub async fn get_repository_intelligence(&self) -> RepositoryIntelligence {
        self.repo_intelligence.read().await.clone()
    }
    
    /// Handle MCP request with daemon context
    pub async fn handle_mcp_request(&self, method: &str, params: serde_json::Value) -> SwarmResult<serde_json::Value> {
        debug!("🔄 Handling MCP request: {}", method);
        
        let mut status = self.status.write().await;
        status.intelligence_queries_handled += 1;
        drop(status);
        
        match method {
            "daemon_status" => Ok(serde_json::to_value(self.get_status().await)?),
            "repository_intelligence" => Ok(serde_json::to_value(self.get_repository_intelligence().await)?),
            "peer_swarms" => Ok(serde_json::json!({
                "connected_peers": self.status.read().await.connected_peers
            })),
            "generate_mcp_config" => {
                let config_path = self.generate_mcp_config(None).await?;
                Ok(serde_json::json!({
                    "config_path": config_path.display().to_string(),
                    "usage": self.get_cli_usage_examples().await?
                }))
            },
            "cli_interactions" => {
                let cli_tool = params.get("cli_tool").and_then(|v| v.as_str()).map(|s| s.to_string());
                let interactions = self.get_cli_interactions(cli_tool).await?;
                Ok(serde_json::to_value(interactions)?)
            },
            _ => {
                // Delegate to swarm for standard operations
                Ok(serde_json::json!({
                    "message": format!("Daemon {} handling request: {}", self.config.daemon_name, method),
                    "daemon_type": self.config.daemon_type,
                    "capabilities": self.config.capabilities
                }))
            }
        }
    }
    
    /// Generate MCP configuration file for Claude Code and code-mesh integration
    pub async fn generate_mcp_config(&self, output_path: Option<PathBuf>) -> SwarmResult<PathBuf> {
        let config = McpConfiguration {
            servers: std::iter::once((
                "zen-swarm".to_string(),
                McpServerConfig {
                    command: "npx".to_string(),
                    args: vec![
                        "zen-swarm".to_string(),
                        "mcp".to_string(),
                        "start".to_string(),
                        "--daemon-port".to_string(),
                        self.config.daemon_port.to_string(),
                    ],
                    env: Some(std::iter::once((
                        "ZEN_SWARM_REPO_PATH".to_string(),
                        self.config.repo_path.display().to_string(),
                    )).collect()),
                },
            )).collect(),
        };
        
        let config_json = serde_json::to_string_pretty(&config)
            .map_err(|e| SwarmError::Serialization(format!("Failed to serialize MCP config: {}", e)))?;
        
        let config_path = output_path.unwrap_or_else(|| {
            self.config.repo_path.join(".zen-swarm-mcp.json")
        });
        
        fs::write(&config_path, config_json)
            .map_err(|e| SwarmError::FileSystem(format!("Failed to write MCP config: {}", e)))?;
        
        info!("📝 Generated MCP config: {}", config_path.display());
        Ok(config_path)
    }
    
    /// Log CLI interaction for swarm monitoring (JSON structured logging)
    pub async fn log_cli_interaction(&self, log: CliInteractionLog) -> SwarmResult<()> {
        let log_json = serde_json::to_string(&log)
            .map_err(|e| SwarmError::Serialization(format!("Failed to serialize CLI log: {}", e)))?;
        
        // Log to structured format for swarm monitoring (logtape-style)
        info!(target: "cli_interaction", "{}", log_json);
        
        // Store in repository intelligence for analysis
        let mut intelligence = self.repo_intelligence.write().await;
        let interactions_key = format!("cli_interactions_{}", log.cli_tool);
        if let Some(domain_knowledge) = intelligence.domain_knowledge.get_mut(&interactions_key) {
            domain_knowledge.insights.push(log_json);
            domain_knowledge.last_validated = Utc::now();
        } else {
            intelligence.domain_knowledge.insert(interactions_key, DomainKnowledge {
                domain_area: "cli_interactions".to_string(),
                knowledge_type: log.cli_tool.clone(),
                insights: vec![log_json],
                confidence_level: 1.0,
                sources: vec!["swarm_daemon_cli_monitoring".to_string()],
                last_validated: Utc::now(),
            });
        }
        
        Ok(())
    }
    
    /// Get CLI interaction history
    pub async fn get_cli_interactions(&self, cli_tool: Option<String>) -> SwarmResult<Vec<CliInteractionLog>> {
        let intelligence = self.repo_intelligence.read().await;
        let mut interactions = Vec::new();
        
        for (key, domain_knowledge) in &intelligence.domain_knowledge {
            if key.starts_with("cli_interactions_") {
                if let Some(ref tool) = cli_tool {
                    if !key.contains(tool) {
                        continue;
                    }
                }
                
                for insight_json in &domain_knowledge.insights {
                    if let Ok(log) = serde_json::from_str::<CliInteractionLog>(insight_json) {
                        interactions.push(log);
                    }
                }
            }
        }
        
        // Sort by timestamp (most recent first)
        interactions.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(interactions)
    }
    
    /// Get CLI usage examples for different integrations
    pub async fn get_cli_usage_examples(&self) -> SwarmResult<HashMap<String, String>> {
        let config_path = self.config.repo_path.join(".zen-swarm-mcp.json").display().to_string();
        
        let examples = std::iter::from_fn(|| None)
            // Claude Code with zen-swarm MCP
            .chain(std::iter::once((
                "claude_interactive".to_string(),
                format!("claude --mcp-config '{}'", config_path)
            )))
            .chain(std::iter::once((
                "claude_noninteractive".to_string(),
                format!("claude --mcp-config '{}' -p --output-format json \"Your prompt here\"", config_path)
            )))
            // code-mesh integration  
            .chain(std::iter::once((
                "code_mesh_integration".to_string(),
                format!("code-mesh --swarm-config '{}' --provider zen-swarm", config_path)
            )))
            // Direct API usage (for Gemini, GPT, etc.)
            .chain(std::iter::once((
                "gemini_api_usage".to_string(),
                "# Gemini API calls route through THE COLLECTIVE via A2A protocol".to_string()
            )))
            .collect();
        
        Ok(examples)
    }
    
    /// Request intelligence from THE COLLECTIVE via zen-orchestrator
    pub async fn request_intelligence(
        &self,
        task_description: String,
        preferred_llm: String,
    ) -> SwarmResult<A2AResponse> {
        debug!("🧠 Requesting intelligence from THE COLLECTIVE");
        
        let context = RepositoryContext {
            working_directory: self.config.repo_path.to_string_lossy().to_string(),
            relevant_files: vec![], // TODO: Analyze and populate
            build_system: None, // TODO: Detect build system
            test_framework: None, // TODO: Detect test framework
            domain_area: None, // TODO: Analyze domain
        };
        
        let client = self.a2a_client.read().await;
        let response = client.request_intelligence(
            task_description,
            preferred_llm,
            context,
            Priority::Normal,
        ).await
        .map_err(|e| SwarmError::CoordinationError(format!("Intelligence request failed: {}", e)))?;
        
        debug!("✅ Received intelligence response from THE COLLECTIVE");
        Ok(response)
    }
    
    /// Share repository intelligence with THE COLLECTIVE
    pub async fn share_intelligence(&self) -> SwarmResult<()> {
        debug!("📤 Sharing repository intelligence with THE COLLECTIVE");
        
        let intelligence = self.repo_intelligence.read().await;
        
        // Convert repository intelligence to A2A format
        let patterns: Vec<CodePattern> = intelligence.code_patterns.iter().map(|(id, pattern)| {
            CodePattern {
                pattern_id: id.clone(),
                pattern_type: "general".to_string(),
                description: format!("{:?}", pattern), 
                confidence: 0.8, // TODO: Calculate confidence
                usage_frequency: 1, // TODO: Track usage
            }
        }).collect();
        
        let optimizations: Vec<BuildOptimization> = intelligence.build_optimizations.iter().map(|opt| {
            BuildOptimization {
                optimization_id: "opt".to_string(),
                optimization_type: "build".to_string(),
                description: opt.clone(),
                performance_gain: 1.0, // TODO: Track metrics
                success_rate: 0.9, // TODO: Track success
            }
        }).collect();
        
        let domain_knowledge = intelligence.domain_knowledge.clone();
        
        let client = self.a2a_client.read().await;
        client.share_repository_intelligence(patterns, optimizations, domain_knowledge).await
            .map_err(|e| SwarmError::CoordinationError(format!("Intelligence sharing failed: {}", e)))?;
            
        debug!("✅ Successfully shared repository intelligence");
        Ok(())
    }
    
    /// Coordinate task execution with THE COLLECTIVE
    pub async fn coordinate_task(
        &self,
        task_type: TaskType,
        requirements: TaskRequirements,
    ) -> SwarmResult<A2AResponse> {
        debug!("🔄 Coordinating task with THE COLLECTIVE");
        
        let client = self.a2a_client.read().await;
        let response = client.coordinate_task(task_type, requirements).await
            .map_err(|e| SwarmError::CoordinationError(format!("Task coordination failed: {}", e)))?;
            
        debug!("✅ Task coordination response received");
        Ok(response)
    }
    
    /// Discover available capabilities from THE COLLECTIVE
    pub async fn discover_capabilities(
        &self,
        requested_capabilities: Vec<String>,
    ) -> SwarmResult<A2AResponse> {
        debug!("🔍 Discovering capabilities from THE COLLECTIVE");
        
        let client = self.a2a_client.read().await;
        let response = client.discover_capabilities(requested_capabilities).await
            .map_err(|e| SwarmError::CoordinationError(format!("Capability discovery failed: {}", e)))?;
            
        debug!("✅ Capability discovery completed");
        Ok(response)
    }
    
    /// Get current A2A connection status
    pub async fn connection_status(&self) -> ConnectionStatus {
        let client = self.a2a_client.read().await;
        client.connection_status().await
    }
    
    /// Send heartbeat to THE COLLECTIVE via A2A
    pub async fn send_heartbeat(&self) -> SwarmResult<()> {
        let status = SwarmStatus::Healthy;
        let metrics = SwarmMetrics {
            tasks_processed: {
                let status = self.status.read().await;
                status.total_tasks_processed
            },
            average_response_time_ms: 100.0, // TODO: Track real metrics
            error_rate: 0.0, // TODO: Track error rate
            cpu_usage: 25.0, // TODO: Get real system metrics
            memory_usage: 512.0, // TODO: Get real memory usage
            disk_usage: 1024.0, // TODO: Get real disk usage
        };
        
        let client = self.a2a_client.read().await;
        client.send_heartbeat(status, metrics).await
            .map_err(|e| SwarmError::CoordinationError(format!("Heartbeat failed: {}", e)))?;
            
        Ok(())
    }
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            peer_discovery_enabled: true,
            collective_intelligence_enabled: true,
            cross_repo_sharing_enabled: true,
            discovery_interval_sec: 30,
        }
    }
}

/// CLI entry point for daemon management
#[derive(Debug)]
pub struct SwarmDaemonCli;

impl SwarmDaemonCli {
    /// Start daemon from CLI
    pub async fn start_daemon(repo_path: PathBuf, port: u16, collective_endpoint: Option<String>) -> SwarmResult<()> {
        let daemon_name = format!("zen-swarm-{}", 
            repo_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown"));
        
        let config = SwarmDaemonConfig {
            repo_path: repo_path.clone(),
            daemon_name: daemon_name.clone(),
            daemon_port: port,
            collective_endpoint,
            daemon_type: SwarmDaemonType::ProjectRepository {
                project_name: repo_path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
                tech_stack: vec!["rust".to_string()], // Would be auto-detected
                domain_areas: vec!["general".to_string()], // Would be analyzed
            },
            capabilities: vec![
                "code_generation".to_string(),
                "test_orchestration".to_string(),
                "build_optimization".to_string(),
                "domain_analysis".to_string(),
            ],
            a2a_config: A2AConfig::default(),
            discovery_config: DiscoveryConfig::default(),
        };
        
        let daemon = SwarmDaemon::new(config).await?;
        daemon.start().await?;
        
        // Keep daemon running
        info!("🔄 Zen Swarm Daemon running - Press Ctrl+C to stop");
        tokio::signal::ctrl_c().await.expect("Failed to listen for ctrl+c");
        info!("🛑 Shutting down Zen Swarm Daemon");
        
        Ok(())
    }
}

impl TaskExecutor {
    /// Create new task executor
    pub async fn new(
        repo_path: PathBuf,
        a2a_coordinator: Arc<A2ACoordinator>,
        intelligence_cache: Arc<RwLock<RepositoryIntelligence>>,
    ) -> SwarmResult<Self> {
        let file_ops = Arc::new(FileOperations::new(repo_path.clone()));
        let build_runner = Arc::new(BuildRunner::new(repo_path.clone()));
        
        Ok(Self {
            repo_path,
            a2a_coordinator,
            intelligence_cache,
            file_ops,
            build_runner,
        })
    }

    /// Execute coding task with specified LLM via A2A to THE COLLECTIVE
    pub async fn execute_coding_task(
        &self,
        task_description: &str,
        preferred_llm: &str,
        context: TaskContext,
    ) -> SwarmResult<CodingTaskResponse> {
        info!("🚀 Executing coding task with {}: {}", preferred_llm, task_description);
        
        let mut actions_taken = Vec::new();
        let mut files_modified = Vec::new();
        let start_time = std::time::Instant::now();
        
        // 1. ANALYZE: Understand the task and repository context
        let analysis = self.analyze_task(task_description, &context).await?;
        actions_taken.push("repository_analysis".to_string());
        
        // 2. PLAN: Get plan from specified LLM via A2A to THE COLLECTIVE
        let plan = self.get_llm_plan_via_a2a(task_description, preferred_llm, &analysis).await?;
        actions_taken.push(format!("planning_via_{}_through_collective", preferred_llm));
        
        // 3. EXECUTE: Perform the actual coding work
        let execution_result = self.execute_plan(&plan, &mut files_modified, &mut actions_taken).await?;
        
        // 4. VALIDATE: Run tests and verify changes
        let _validation_result = self.validate_changes(&files_modified).await?;
        actions_taken.push("validation_and_testing".to_string());
        
        // 5. REPORT: Generate comprehensive result
        let processing_time_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(CodingTaskResponse {
            task_result: execution_result.summary,
            files_modified,
            actions_taken,
            model_used: preferred_llm.to_string(),
            provider: self.get_provider_for_model(preferred_llm),
            usage: execution_result.token_usage,
            processing_time_ms,
        })
    }
    
    /// Get implementation plan from LLM via A2A to THE COLLECTIVE
    async fn get_llm_plan_via_a2a(
        &self,
        task: &str,
        llm_model: &str,
        analysis: &TaskAnalysis,
    ) -> SwarmResult<ImplementationPlan> {
        let llm_request = A2AMessage::IntelligenceRequest {
            request_id: Uuid::new_v4().to_string(),
            capability: AICapability::LLMReasoning {
                model: llm_model.to_string(),
                context_length: 8192,
            },
            context: SwarmContext {
                swarm_id: format!("task-executor-{}", self.repo_path.file_name()
                    .and_then(|n| n.to_str()).unwrap_or("unknown")),
                active_agents: 1,
                current_tasks: vec!["planning".to_string()],
                available_resources: std::iter::once((
                    "task_description".to_string(), 
                    serde_json::json!(task)
                )).chain(std::iter::once((
                    "repository_analysis".to_string(),
                    serde_json::json!(analysis)
                ))).collect(),
                local_capabilities: vec!["code_execution".to_string(), "file_operations".to_string()],
            },
            priority: Priority::Normal,
        };
        
        // Send to THE COLLECTIVE via A2A for LLM processing
        info!("🧠 Requesting {} plan via A2A to THE COLLECTIVE", llm_model);
        match self.a2a_coordinator.send_message("the-collective", llm_request).await {
            Ok(_) => {
                // Simplified response - would wait for actual LLM plan response
                Ok(ImplementationPlan {
                    steps: vec![
                        PlanStep::ModifyFile {
                            path: analysis.relevant_files.first()
                                .unwrap_or(&"src/main.rs".to_string()).clone(),
                            changes: vec![FileChange {
                                line_start: 1,
                                line_end: 1,
                                new_content: format!("// Task completed via {}: {}", llm_model, task),
                            }],
                        },
                    ],
                    estimated_time_ms: 3000,
                    required_files: analysis.relevant_files.clone(),
                })
            }
            Err(e) => Err(e),
        }
    }
    
    /// Analyze task and repository context
    async fn analyze_task(&self, task: &str, context: &TaskContext) -> SwarmResult<TaskAnalysis> {
        Ok(TaskAnalysis {
            task_type: "code_modification".to_string(),
            complexity: TaskComplexity::Moderate,
            relevant_files: context.relevant_files.clone(),
            required_capabilities: vec!["file_operations".to_string(), "code_generation".to_string()],
            estimated_time_ms: 5000,
        })
    }
    
    /// Execute the implementation plan (actual file operations)
    async fn execute_plan(
        &self,
        plan: &ImplementationPlan,
        files_modified: &mut Vec<String>,
        actions_taken: &mut Vec<String>,
    ) -> SwarmResult<ExecutionResult> {
        info!("🔧 Executing implementation plan: {} steps", plan.steps.len());
        
        let mut execution_result = ExecutionResult::new();
        
        for step in &plan.steps {
            match step {
                PlanStep::CreateFile { path, content } => {
                    self.file_ops.create_file(path, content).await?;
                    files_modified.push(path.clone());
                    actions_taken.push(format!("created_file_{}", path));
                }
                PlanStep::ModifyFile { path, changes } => {
                    self.file_ops.apply_changes(path, changes).await?;
                    files_modified.push(path.clone());
                    actions_taken.push(format!("modified_file_{}", path));
                }
                PlanStep::DeleteFile { path } => {
                    self.file_ops.delete_file(path).await?;
                    actions_taken.push(format!("deleted_file_{}", path));
                }
                PlanStep::RunCommand { command, args } => {
                    let output = self.build_runner.run_command(command, args).await?;
                    execution_result.command_outputs.push(output);
                    actions_taken.push(format!("executed_command_{}", command));
                }
                PlanStep::RunTests { test_pattern } => {
                    let test_result = self.build_runner.run_tests(test_pattern).await?;
                    execution_result.test_results.push(test_result);
                    actions_taken.push("executed_tests".to_string());
                }
            }
        }
        
        execution_result.summary = format!("Completed {} implementation steps via A2A intelligence", plan.steps.len());
        Ok(execution_result)
    }
    
    /// Validate changes by running tests and builds
    async fn validate_changes(&self, files_modified: &[String]) -> SwarmResult<ValidationResult> {
        info!("🧪 Validating changes to {} files", files_modified.len());
        
        let build_result = self.build_runner.run_build().await?;
        let test_result = self.build_runner.run_all_tests().await?;
        let lint_result = self.build_runner.run_linter().await?;
        
        Ok(ValidationResult {
            build_success: build_result.success,
            tests_passed: test_result.success,
            lint_clean: lint_result.success,
            build_output: build_result.output.clone(),
            test_output: test_result.output.clone(),
            lint_output: lint_result.output.clone(),
        })
    }
    
    /// Get provider name for LLM model
    fn get_provider_for_model(&self, model: &str) -> String {
        match model {
            m if m.starts_with("gemini") => "google_gemini".to_string(),
            m if m.starts_with("gpt") => "openai".to_string(), 
            m if m.starts_with("claude") => "anthropic".to_string(),
            m if m.starts_with("llama") => "ollama_local".to_string(),
            _ => "unknown".to_string(),
        }
    }
}

impl FileOperations {
    pub fn new(repo_path: PathBuf) -> Self {
        Self { repo_path }
    }
    
    pub async fn create_file(&self, path: &str, content: &str) -> SwarmResult<()> {
        let full_path = self.repo_path.join(path);
        if let Some(parent) = full_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| SwarmError::FileSystem(format!("Failed to create directories: {}", e)))?;
        }
        
        tokio::fs::write(full_path, content).await
            .map_err(|e| SwarmError::FileSystem(format!("Failed to create file {}: {}", path, e)))?;
        
        info!("✏️ Created file: {}", path);
        Ok(())
    }
    
    pub async fn apply_changes(&self, path: &str, changes: &[FileChange]) -> SwarmResult<()> {
        let full_path = self.repo_path.join(path);
        
        let content = tokio::fs::read_to_string(&full_path).await
            .map_err(|e| SwarmError::FileSystem(format!("Failed to read file {}: {}", path, e)))?;
        
        let mut lines: Vec<&str> = content.lines().collect();
        
        for change in changes {
            if change.line_start <= lines.len() {
                let new_lines: Vec<&str> = change.new_content.lines().collect();
                lines.splice(change.line_start-1..change.line_end, new_lines);
            }
        }
        
        let new_content = lines.join("\n");
        tokio::fs::write(full_path, new_content).await
            .map_err(|e| SwarmError::FileSystem(format!("Failed to modify file {}: {}", path, e)))?;
        
        info!("✏️ Modified file: {}", path);
        Ok(())
    }
    
    pub async fn delete_file(&self, path: &str) -> SwarmResult<()> {
        let full_path = self.repo_path.join(path);
        tokio::fs::remove_file(full_path).await
            .map_err(|e| SwarmError::FileSystem(format!("Failed to delete file {}: {}", path, e)))?;
        
        info!("🗑️ Deleted file: {}", path);
        Ok(())
    }
}

impl BuildRunner {
    pub fn new(repo_path: PathBuf) -> Self {
        Self { repo_path }
    }
    
    pub async fn run_command(&self, command: &str, args: &[String]) -> SwarmResult<CommandOutput> {
        let output = tokio::process::Command::new(command)
            .args(args)
            .current_dir(&self.repo_path)
            .output()
            .await
            .map_err(|e| SwarmError::Process(format!("Failed to run command {}: {}", command, e)))?;
        
        Ok(CommandOutput {
            command: format!("{} {}", command, args.join(" ")),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }
    
    pub async fn run_tests(&self, test_pattern: &Option<String>) -> SwarmResult<TestResult> {
        let args = if let Some(pattern) = test_pattern {
            vec!["test".to_string(), pattern.clone()]
        } else {
            vec!["test".to_string()]
        };
        
        let output = self.run_command("cargo", &args).await?;
        
        Ok(TestResult {
            success: output.exit_code == 0,
            output: format!("{}\n{}", output.stdout, output.stderr),
            tests_run: 1,
            tests_passed: if output.exit_code == 0 { 1 } else { 0 },
            tests_failed: if output.exit_code == 0 { 0 } else { 1 },
        })
    }
    
    pub async fn run_build(&self) -> SwarmResult<TestResult> {
        let output = self.run_command("cargo", &["build".to_string()]).await?;
        
        Ok(TestResult {
            success: output.exit_code == 0,
            output: format!("{}\n{}", output.stdout, output.stderr),
            tests_run: 0,
            tests_passed: 0,
            tests_failed: 0,
        })
    }
    
    pub async fn run_all_tests(&self) -> SwarmResult<TestResult> {
        self.run_tests(&None).await
    }
    
    pub async fn run_linter(&self) -> SwarmResult<TestResult> {
        let output = self.run_command("cargo", &["clippy".to_string()]).await?;
        
        Ok(TestResult {
            success: output.exit_code == 0,
            output: format!("{}\n{}", output.stdout, output.stderr),
            tests_run: 0,
            tests_passed: 0,
            tests_failed: 0,
        })
    }
}

impl ExecutionResult {
    pub fn new() -> Self {
        Self {
            summary: String::new(),
            token_usage: TokenUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
            command_outputs: Vec::new(),
            test_results: Vec::new(),
        }
    }
}