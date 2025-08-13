//! A2A Client for zen-swarm daemons to communicate with zen-orchestrator
//!
//! This module provides the client-side implementation of the A2A protocol
//! that allows zen-swarm repository daemons to communicate with zen-orchestrator
//! running inside THE COLLECTIVE.

use zen_orchestrator::a2a::{
    A2AMessage, A2AResponse, SwarmStatus, SwarmMetrics, RepositoryContext,
    Priority, A2AError, A2AResult, CodePattern, BuildOptimization,
    TaskType, TaskRequirements, ComputeRequirements, RepositoryIntelligence,
    TaskCoordination, CapabilityDiscovery, SwarmHeartbeat, IntelligenceRequest,
    SwarmRegistration,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, mpsc};
use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{info, warn, error, debug};
use uuid::Uuid;

/// A2A client for communicating with zen-orchestrator
#[derive(Debug)]
pub struct A2AClient {
    /// Client configuration
    config: A2AClientConfig,
    
    /// Connection status
    connection_status: Arc<RwLock<ConnectionStatus>>,
    
    /// Pending requests tracking
    pending_requests: Arc<RwLock<HashMap<String, PendingRequest>>>,
    
    /// Message channels
    message_sender: mpsc::UnboundedSender<A2AMessage>,
    response_receiver: Arc<RwLock<mpsc::UnboundedReceiver<A2AResponse>>>,
    
    /// Connection handle
    connection_handle: Option<tokio::task::JoinHandle<()>>,
}

/// A2A client configuration
#[derive(Debug, Clone)]
pub struct A2AClientConfig {
    /// Unique swarm identifier
    pub swarm_id: String,
    
    /// Repository path for this swarm
    pub repository_path: String,
    
    /// Swarm capabilities
    pub capabilities: Vec<String>,
    
    /// Local daemon port
    pub daemon_port: u16,
    
    /// zen-orchestrator server endpoint
    pub orchestrator_endpoint: String,
    
    /// Connection timeout
    pub connection_timeout_ms: u64,
    
    /// Heartbeat interval
    pub heartbeat_interval_sec: u64,
    
    /// Request timeout
    pub request_timeout_ms: u64,
}

/// Connection status
#[derive(Debug, Clone)]
pub enum ConnectionStatus {
    Disconnected,
    Connecting,
    Connected {
        connected_at: DateTime<Utc>,
        orchestrator_info: OrchestratorInfo,
    },
    Reconnecting {
        attempt: u32,
        last_error: String,
    },
    Failed {
        error: String,
        retry_at: Option<DateTime<Utc>>,
    },
}

/// Information about connected orchestrator
#[derive(Debug, Clone)]
pub struct OrchestratorInfo {
    pub server_id: String,
    pub version: String,
    pub capabilities: Vec<String>,
    pub endpoints: HashMap<String, String>,
}

/// Pending request information
#[derive(Debug)]
struct PendingRequest {
    request_id: String,
    sent_at: DateTime<Utc>,
    request_type: String,
    response_sender: mpsc::UnboundedSender<A2AResponse>,
}

impl A2AClient {
    /// Create new A2A client
    pub fn new(config: A2AClientConfig) -> Self {
        let (message_sender, _message_receiver) = mpsc::unbounded_channel();
        let (_response_sender, response_receiver) = mpsc::unbounded_channel();
        
        Self {
            config,
            connection_status: Arc::new(RwLock::new(ConnectionStatus::Disconnected)),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            message_sender,
            response_receiver: Arc::new(RwLock::new(response_receiver)),
            connection_handle: None,
        }
    }
    
    /// Start the A2A client
    pub async fn start(&mut self) -> A2AResult<()> {
        info!("🌐 Starting A2A client for swarm {}", self.config.swarm_id);
        
        // Update status to connecting
        *self.connection_status.write().await = ConnectionStatus::Connecting;
        
        // Start connection management task
        self.start_connection_manager().await;
        
        // Start heartbeat task
        self.start_heartbeat_task().await;
        
        info!("✅ A2A client started successfully");
        Ok(())
    }
    
    /// Register this swarm with zen-orchestrator
    pub async fn register_swarm(&self) -> A2AResult<A2AResponse> {
        let request_id = Uuid::new_v4().to_string();
        
        let message = A2AMessage::SwarmRegistration {
            swarm_id: self.config.swarm_id.clone(),
            repository_path: self.config.repository_path.clone(),
            capabilities: self.config.capabilities.clone(),
            daemon_port: self.config.daemon_port,
        };
        
        self.send_message_with_response(request_id, message).await
    }
    
    /// Request LLM intelligence from THE COLLECTIVE
    pub async fn request_intelligence(
        &self,
        task_description: String,
        preferred_llm: String,
        context: RepositoryContext,
        priority: Priority,
    ) -> A2AResult<A2AResponse> {
        let request_id = Uuid::new_v4().to_string();
        
        let message = A2AMessage::IntelligenceRequest {
            request_id: request_id.clone(),
            swarm_id: self.config.swarm_id.clone(),
            task_description,
            preferred_llm,
            context,
            priority,
        };
        
        self.send_message_with_response(request_id, message).await
    }
    
    /// Share repository intelligence with THE COLLECTIVE
    pub async fn share_repository_intelligence(
        &self,
        patterns: Vec<CodePattern>,
        optimizations: Vec<BuildOptimization>,
        domain_knowledge: HashMap<String, String>,
    ) -> A2AResult<()> {
        let message = A2AMessage::RepositoryIntelligence {
            swarm_id: self.config.swarm_id.clone(),
            patterns,
            optimizations,
            domain_knowledge,
        };
        
        self.send_message(message).await?;
        Ok(())
    }
    
    /// Request task coordination from THE COLLECTIVE
    pub async fn coordinate_task(
        &self,
        task_type: TaskType,
        requirements: TaskRequirements,
    ) -> A2AResult<A2AResponse> {
        let task_id = Uuid::new_v4().to_string();
        let request_id = Uuid::new_v4().to_string();
        
        let message = A2AMessage::TaskCoordination {
            task_id,
            requesting_swarm: self.config.swarm_id.clone(),
            task_type,
            requirements,
        };
        
        self.send_message_with_response(request_id, message).await
    }
    
    /// Discover available capabilities from THE COLLECTIVE
    pub async fn discover_capabilities(
        &self,
        requested_capabilities: Vec<String>,
    ) -> A2AResult<A2AResponse> {
        let request_id = Uuid::new_v4().to_string();
        
        let message = A2AMessage::CapabilityDiscovery {
            request_id: request_id.clone(),
            requesting_swarm: self.config.swarm_id.clone(),
            requested_capabilities,
        };
        
        self.send_message_with_response(request_id, message).await
    }
    
    /// Send heartbeat to maintain connection
    pub async fn send_heartbeat(
        &self,
        status: SwarmStatus,
        metrics: SwarmMetrics,
    ) -> A2AResult<()> {
        let message = A2AMessage::SwarmHeartbeat {
            swarm_id: self.config.swarm_id.clone(),
            timestamp: Utc::now(),
            status,
            metrics,
        };
        
        self.send_message(message).await?;
        Ok(())
    }
    
    /// Get current connection status
    pub async fn connection_status(&self) -> ConnectionStatus {
        self.connection_status.read().await.clone()
    }
    
    /// Send message without expecting response
    async fn send_message(&self, message: A2AMessage) -> A2AResult<()> {
        debug!("📤 Sending A2A message: {:?}", message);
        
        self.message_sender.send(message)
            .map_err(|e| A2AError::InternalError(format!("Failed to send message: {}", e)))?;
            
        Ok(())
    }
    
    /// Send message and wait for response
    async fn send_message_with_response(
        &self,
        request_id: String,
        message: A2AMessage,
    ) -> A2AResult<A2AResponse> {
        debug!("📤 Sending A2A message with response: {:?}", message);
        
        // Create response channel
        let (response_sender, mut response_receiver) = mpsc::unbounded_channel();
        
        // Track pending request
        let pending_request = PendingRequest {
            request_id: request_id.clone(),
            sent_at: Utc::now(),
            request_type: format!("{:?}", message),
            response_sender,
        };
        
        self.pending_requests.write().await.insert(request_id.clone(), pending_request);
        
        // Send message
        self.message_sender.send(message)
            .map_err(|e| A2AError::InternalError(format!("Failed to send message: {}", e)))?;
        
        // Wait for response with timeout
        let timeout_duration = Duration::from_millis(self.config.request_timeout_ms);
        
        match tokio::time::timeout(timeout_duration, response_receiver.recv()).await {
            Ok(Some(response)) => {
                // Clean up pending request
                self.pending_requests.write().await.remove(&request_id);
                Ok(response)
            }
            Ok(None) => {
                self.pending_requests.write().await.remove(&request_id);
                Err(A2AError::InternalError("Response channel closed".to_string()))
            }
            Err(_) => {
                self.pending_requests.write().await.remove(&request_id);
                Err(A2AError::Timeout("Request timed out".to_string()))
            }
        }
    }
    
    /// Start connection management background task
    async fn start_connection_manager(&self) {
        let config = self.config.clone();
        let connection_status = self.connection_status.clone();
        let message_sender = self.message_sender.clone();
        
        tokio::spawn(async move {
            let mut retry_count = 0;
            let max_retries = 5;
            let mut retry_delay = Duration::from_secs(1);
            
            loop {
                match Self::establish_connection(&config).await {
                    Ok(stream) => {
                        info!("✅ A2A connection established to {}", config.orchestrator_endpoint);
                        
                        // Update connection status
                        *connection_status.write().await = ConnectionStatus::Connected {
                            connected_at: Utc::now(),
                            orchestrator_info: OrchestratorInfo {
                                server_id: "zen-orchestrator".to_string(),
                                version: "1.0.0".to_string(),
                                capabilities: vec!["llm_routing".to_string(), "neural_services".to_string()],
                                endpoints: HashMap::new(),
                            },
                        };
                        
                        retry_count = 0;
                        retry_delay = Duration::from_secs(1);
                        
                        // Handle connection until it fails
                        if let Err(e) = Self::handle_connection(stream, message_sender.clone()).await {
                            error!("💔 A2A connection lost: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("❌ Failed to establish A2A connection: {}", e);
                        
                        retry_count += 1;
                        if retry_count >= max_retries {
                            *connection_status.write().await = ConnectionStatus::Failed {
                                error: format!("Max retries exceeded: {}", e),
                                retry_at: Some(Utc::now() + chrono::Duration::minutes(5)),
                            };
                            tokio::time::sleep(Duration::from_secs(300)).await; // 5 minutes
                            retry_count = 0;
                        } else {
                            *connection_status.write().await = ConnectionStatus::Reconnecting {
                                attempt: retry_count,
                                last_error: e.to_string(),
                            };
                            tokio::time::sleep(retry_delay).await;
                            retry_delay = std::cmp::min(retry_delay * 2, Duration::from_secs(30));
                        }
                    }
                }
            }
        });
    }
    
    /// Start heartbeat background task
    async fn start_heartbeat_task(&self) {
        let config = self.config.clone();
        let client = self.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(config.heartbeat_interval_sec));
            
            loop {
                interval.tick().await;
                
                let status = SwarmStatus::Healthy;
                let metrics = SwarmMetrics {
                    tasks_processed: 0,
                    average_response_time_ms: 100.0,
                    error_rate: 0.0,
                    cpu_usage: 25.0,
                    memory_usage: 512.0,
                    disk_usage: 1024.0,
                };
                
                if let Err(e) = client.send_heartbeat(status, metrics).await {
                    warn!("💔 Heartbeat failed: {}", e);
                }
            }
        });
    }
    
    /// Establish TCP connection to zen-orchestrator
    async fn establish_connection(config: &A2AClientConfig) -> A2AResult<TcpStream> {
        let timeout_duration = Duration::from_millis(config.connection_timeout_ms);
        
        tokio::time::timeout(timeout_duration, TcpStream::connect(&config.orchestrator_endpoint))
            .await
            .map_err(|_| A2AError::Timeout("Connection timeout".to_string()))?
            .map_err(|e| A2AError::InternalError(format!("Connection failed: {}", e)))
    }
    
    /// Handle established connection
    async fn handle_connection(
        mut stream: TcpStream,
        message_sender: mpsc::UnboundedSender<A2AMessage>,
    ) -> A2AResult<()> {
        let mut buffer = vec![0; 4096];
        
        loop {
            match stream.read(&mut buffer).await {
                Ok(0) => {
                    // Connection closed
                    return Err(A2AError::InternalError("Connection closed by remote".to_string()));
                }
                Ok(n) => {
                    // Process received data
                    let data = &buffer[..n];
                    debug!("📥 Received {} bytes from orchestrator", n);
                    
                    // TODO: Implement proper message framing and parsing
                    // For now, just log the data
                    if let Ok(text) = std::str::from_utf8(data) {
                        debug!("📥 Received text: {}", text);
                    }
                }
                Err(e) => {
                    return Err(A2AError::InternalError(format!("Read error: {}", e)));
                }
            }
        }
    }
}

impl Clone for A2AClient {
    fn clone(&self) -> Self {
        let (message_sender, _) = mpsc::unbounded_channel();
        let (_, response_receiver) = mpsc::unbounded_channel();
        
        Self {
            config: self.config.clone(),
            connection_status: self.connection_status.clone(),
            pending_requests: self.pending_requests.clone(),
            message_sender,
            response_receiver: Arc::new(RwLock::new(response_receiver)),
            connection_handle: None,
        }
    }
}

impl Default for A2AClientConfig {
    fn default() -> Self {
        Self {
            swarm_id: Uuid::new_v4().to_string(),
            repository_path: ".".to_string(),
            capabilities: vec!["basic_coordination".to_string()],
            daemon_port: 0,
            orchestrator_endpoint: "localhost:8080".to_string(),
            connection_timeout_ms: 5000,
            heartbeat_interval_sec: 30,
            request_timeout_ms: 10000,
        }
    }
}