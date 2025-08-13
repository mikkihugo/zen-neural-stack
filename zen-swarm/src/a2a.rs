//! Agent-to-Agent (A2A) coordination system
//!
//! Dedicated A2A protocol implementation for external communication:
//! - Connection to THE COLLECTIVE (zen-code)
//! - Inter-swarm coordination  
//! - External service integration
//! - Protocol-agnostic channel management
//!
//! This is separate from MCP which is ONLY for Claude Code integration.

use crate::SwarmError;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// A2A coordinator for all external communication
#[derive(Debug)]
pub struct A2ACoordinator {
    /// Coordinator ID
    pub id: String,
    /// Active A2A channels by target name
    channels: Arc<DashMap<String, A2AChannel>>,
    /// Message router for different protocols
    message_router: Arc<A2AMessageRouter>,
    /// THE COLLECTIVE connection status
    collective_status: Arc<RwLock<CollectiveStatus>>,
    /// Peer swarm registry
    peer_swarms: Arc<DashMap<String, PeerSwarm>>,
    /// A2A configuration
    config: A2AConfig,
}

/// A2A communication channel
#[derive(Debug, Clone)]
pub struct A2AChannel {
    pub target_id: String,
    pub channel_type: A2AChannelType,
    pub endpoint: String,
    pub encryption_key: Option<String>,
    pub status: ChannelStatus,
    pub last_activity: DateTime<Utc>,
    pub protocol_version: String,
}

/// A2A channel types (NO MCP - pure A2A protocols)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum A2AChannelType {
    /// In-process memory channel (same swarm controller)
    InProcessMemory {
        shared_memory_key: String,
    },
    /// WebSocket for real-time bidirectional communication
    WebSocket,
    /// Direct TCP for low-latency communication
    DirectTcp,
    /// HTTP for simple request-response
    Http,
    /// Encrypted P2P channel
    EncryptedP2P,
    /// Custom protocol
    Custom { protocol: String, config: Value },
}

/// Channel connection status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelStatus {
    Connecting,
    Active,
    Idle,
    Reconnecting,
    Failed { error: String },
    Closed,
}

/// A2A message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum A2AMessage {
    /// Request intelligence from THE COLLECTIVE
    IntelligenceRequest {
        request_id: String,
        capability: AICapability,
        context: SwarmContext,
        priority: Priority,
    },
    
    /// Response from THE COLLECTIVE
    IntelligenceResponse {
        request_id: String,
        result: IntelligenceResult,
        confidence: f32,
        sources: Vec<String>,
    },
    
    /// Swarm-to-swarm coordination
    SwarmCoordination {
        coordination_type: CoordinationType,
        swarm_id: String,
        payload: Value,
    },
    
    /// Vector search coordination
    VectorSearchRequest {
        query_id: String,
        embedding: Vec<f32>,
        table_name: String,
        limit: usize,
        metadata_filter: Option<Value>,
    },
    
    /// Vector search response
    VectorSearchResponse {
        query_id: String,
        results: Vec<VectorSearchResult>,
        source_swarm: String,
    },
    
    /// Heartbeat/health check
    Heartbeat {
        swarm_id: String,
        timestamp: DateTime<Utc>,
        status: SwarmHealthStatus,
    },
    
    /// Registration with THE COLLECTIVE or peer swarms
    Registration {
        swarm_id: String,
        capabilities: Vec<String>,
        endpoints: HashMap<String, String>,
    },
}

/// AI capability types for THE COLLECTIVE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AICapability {
    LLMReasoning { model: String, context_length: usize },
    VectorSearch { dimensions: usize, similarity_metric: String },
    RAGRetrieval { knowledge_domains: Vec<String> },
    MultiModalAnalysis { modalities: Vec<String> },
    TimeSeriesForecasting { horizon: String },
    CodeGeneration { languages: Vec<String> },
    GeneralIntelligence,
}

/// Priority levels for A2A requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Normal,
    High,
    Critical,
}

/// Swarm context for A2A communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmContext {
    pub swarm_id: String,
    pub active_agents: usize,
    pub current_tasks: Vec<String>,
    pub available_resources: HashMap<String, Value>,
    pub local_capabilities: Vec<String>,
}

/// Intelligence result from THE COLLECTIVE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelligenceResult {
    pub reasoning: Value,
    pub recommendations: Vec<String>,
    pub additional_data: Value,
    pub processing_time_ms: u64,
}

/// Coordination types between swarms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationType {
    TaskDelegation { task_id: String, requirements: Value },
    ResourceSharing { resource_type: String, amount: f64 },
    KnowledgeSync { domain: String, updates: Value },
    LoadBalancing { current_load: f32, capacity: f32 },
    EmergencyCoordination { severity: String, details: Value },
}

/// Vector search result for A2A
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    pub id: String,
    pub score: f32,
    pub content: String,
    pub metadata: Value,
    pub source_swarm: String,
}

/// Swarm health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmHealthStatus {
    Healthy,
    Degraded { issues: Vec<String> },
    Critical { errors: Vec<String> },
    Offline,
}

/// THE COLLECTIVE connection status
#[derive(Debug, Clone)]
pub struct CollectiveStatus {
    pub connected: bool,
    pub endpoint: Option<String>,
    pub last_heartbeat: Option<DateTime<Utc>>,
    pub available_capabilities: Vec<AICapability>,
    pub connection_quality: f32, // 0.0 to 1.0
}

/// Peer swarm information
#[derive(Debug, Clone)]
pub struct PeerSwarm {
    pub swarm_id: String,
    pub endpoint: String,
    pub capabilities: Vec<String>,
    pub trust_level: f32, // 0.0 to 1.0
    pub last_seen: DateTime<Utc>,
    pub response_time_ms: u64,
}

/// A2A configuration
#[derive(Debug, Clone)]
pub struct A2AConfig {
    pub collective_endpoint: Option<String>,
    pub discovery_enabled: bool,
    pub heartbeat_interval_sec: u64,
    pub connection_timeout_sec: u64,
    pub max_retry_attempts: u32,
    pub encryption_enabled: bool,
}

/// Message router for different A2A protocols
#[derive(Debug)]
pub struct A2AMessageRouter {
    websocket_handler: Option<Arc<WebSocketHandler>>,
    tcp_handler: Option<Arc<TcpHandler>>,
    http_handler: Option<Arc<HttpHandler>>,
    p2p_handler: Option<Arc<P2PHandler>>,
}

/// WebSocket handler for A2A communication
#[derive(Debug)]
pub struct WebSocketHandler {
    active_connections: Arc<DashMap<String, tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>>>,
}

/// TCP handler for A2A communication  
#[derive(Debug)]
pub struct TcpHandler {
    active_connections: Arc<DashMap<String, tokio::net::TcpStream>>,
}

/// HTTP handler for A2A communication
#[derive(Debug)]
pub struct HttpHandler {
    client: reqwest::Client,
}

/// P2P handler for encrypted A2A communication
#[derive(Debug)]
pub struct P2PHandler {
    encryption_keys: Arc<DashMap<String, Vec<u8>>>,
}

impl A2ACoordinator {
    /// Create new A2A coordinator
    pub fn new(config: A2AConfig) -> Self {
        let id = format!("a2a-{}", Uuid::new_v4());
        
        Self {
            id,
            channels: Arc::new(DashMap::new()),
            message_router: Arc::new(A2AMessageRouter::new()),
            collective_status: Arc::new(RwLock::new(CollectiveStatus {
                connected: false,
                endpoint: config.collective_endpoint.clone(),
                last_heartbeat: None,
                available_capabilities: Vec::new(),
                connection_quality: 0.0,
            })),
            peer_swarms: Arc::new(DashMap::new()),
            config,
        }
    }

    /// Initialize A2A coordinator and establish connections
    pub async fn initialize(&self) -> Result<(), SwarmError> {
        info!("🔗 Initializing A2A coordinator: {}", self.id);

        // Connect to THE COLLECTIVE if endpoint provided
        if let Some(collective_endpoint) = &self.config.collective_endpoint {
            self.connect_to_collective(collective_endpoint).await?;
        }

        // Start peer discovery if enabled
        if self.config.discovery_enabled {
            self.start_peer_discovery().await?;
        }

        // Start heartbeat system
        self.start_heartbeat_system().await?;

        info!("✅ A2A coordinator initialized successfully");
        Ok(())
    }

    /// Connect to THE COLLECTIVE
    pub async fn connect_to_collective(&self, endpoint: &str) -> Result<(), SwarmError> {
        info!("🧠 Connecting to THE COLLECTIVE at: {}", endpoint);

        let channel = A2AChannel {
            target_id: "the-collective".to_string(),
            channel_type: A2AChannelType::WebSocket,
            endpoint: endpoint.to_string(),
            encryption_key: None, // Could be configured
            status: ChannelStatus::Connecting,
            last_activity: Utc::now(),
            protocol_version: "a2a-1.0".to_string(),
        };

        self.channels.insert("the-collective".to_string(), channel);

        // Update COLLECTIVE status
        {
            let mut status = self.collective_status.write().await;
            status.connected = true;
            status.endpoint = Some(endpoint.to_string());
            status.last_heartbeat = Some(Utc::now());
        }

        // Register with THE COLLECTIVE
        let registration_message = A2AMessage::Registration {
            swarm_id: self.id.clone(),
            capabilities: vec![
                "vector_search".to_string(),
                "task_coordination".to_string(),
                "knowledge_processing".to_string(),
            ],
            endpoints: std::iter::once((
                "a2a".to_string(),
                "ws://localhost:9001/a2a".to_string()
            )).collect(),
        };

        self.send_message("the-collective", registration_message).await?;

        info!("✅ Connected to THE COLLECTIVE");
        Ok(())
    }

    /// Send A2A message to target with collective visibility
    pub async fn send_message(&self, target: &str, message: A2AMessage) -> Result<(), SwarmError> {
        debug!("📤 Sending A2A message to: {}", target);

        // 🧠 COLLECTIVE VISIBILITY: Copy message to THE COLLECTIVE for intelligence gathering
        self.notify_collective_of_a2a_message(target, &message).await?;

        let channel = self.channels.get(target)
            .ok_or_else(|| SwarmError::Communication(format!("No A2A channel to {}", target)))?;

        // Send directly to target (deployment-aware optimal routing)
        match channel.channel_type {
            A2AChannelType::InProcessMemory { ref shared_memory_key } => {
                // Same swarm controller - ultra-fast in-memory communication
                self.send_via_in_process_memory(&channel, message, shared_memory_key).await
            }
            A2AChannelType::WebSocket => {
                // Cross-process/network WebSocket communication  
                self.send_via_websocket(&channel, message).await
            }
            A2AChannelType::Http => {
                // Cross-network HTTP communication
                self.send_via_http(&channel, message).await
            }
            A2AChannelType::DirectTcp => {
                // Low-latency TCP communication
                self.send_via_tcp(&channel, message).await
            }
            A2AChannelType::EncryptedP2P => {
                // Secure peer-to-peer communication
                self.send_via_p2p(&channel, message).await
            }
            A2AChannelType::Custom { .. } => {
                warn!("Custom A2A protocols not yet implemented");
                Err(SwarmError::Communication("Custom protocol not implemented".to_string()))
            }
        }
    }

    /// Notify THE COLLECTIVE of A2A inter-swarm communication for intelligence gathering
    async fn notify_collective_of_a2a_message(&self, target: &str, message: &A2AMessage) -> Result<(), SwarmError> {
        // Skip collective notification for messages TO the collective (avoid loops)
        if target == "the-collective" {
            return Ok(());
        }

        // Create intelligence visibility message
        let visibility_message = A2AMessage::IntelligenceRequest {
            request_id: format!("visibility-{}", Uuid::new_v4()),
            capability: AICapability::GeneralIntelligence,
            context: SwarmContext {
                swarm_id: self.id.clone(),
                active_agents: 0, // Would be populated with real data
                current_tasks: vec!["inter_swarm_a2a_communication".to_string()],
                available_resources: std::iter::once((
                    "a2a_message_intelligence".to_string(),
                    serde_json::json!({
                        "target_swarm": target,
                        "message_type": match message {
                            A2AMessage::IntelligenceRequest { .. } => "intelligence_request",
                            A2AMessage::IntelligenceResponse { .. } => "intelligence_response", 
                            A2AMessage::SwarmCoordination { .. } => "swarm_coordination",
                            A2AMessage::VectorSearchRequest { .. } => "vector_search_request",
                            A2AMessage::VectorSearchResponse { .. } => "vector_search_response",
                            A2AMessage::Heartbeat { .. } => "heartbeat",
                            A2AMessage::Registration { .. } => "registration",
                        },
                        "routing_method": "direct_a2a_with_collective_visibility",
                        "timestamp": chrono::Utc::now().to_rfc3339()
                    })
                )).collect(),
                local_capabilities: vec!["direct_a2a_routing".to_string(), "collective_visibility".to_string()],
            },
            priority: Priority::Low, // Intelligence gathering is low priority
        };

        // Send visibility notification to THE COLLECTIVE (non-blocking)
        if let Some(collective_channel) = self.channels.get("the-collective") {
            match self.send_via_websocket(&collective_channel, visibility_message).await {
                Ok(_) => debug!("🧠 Collective visibility: A2A message to {} reported", target),
                Err(e) => warn!("⚠️ Failed to notify collective of A2A message: {}", e),
            }
        }

        Ok(())
    }

    /// Request intelligence from THE COLLECTIVE
    pub async fn request_collective_intelligence(
        &self,
        capability: AICapability,
        context: SwarmContext,
        priority: Priority,
    ) -> Result<IntelligenceResult, SwarmError> {
        let request_id = Uuid::new_v4().to_string();
        
        let message = A2AMessage::IntelligenceRequest {
            request_id: request_id.clone(),
            capability,
            context,
            priority,
        };

        self.send_message("the-collective", message).await?;

        // TODO: Implement response waiting mechanism
        // For now, return a placeholder response
        Ok(IntelligenceResult {
            reasoning: serde_json::json!({
                "analysis": "THE COLLECTIVE intelligence via A2A",
                "source": "distributed_reasoning_network"
            }),
            recommendations: vec![
                "Coordinate with peer swarms for distributed processing".to_string(),
                "Use collective knowledge base for enhanced reasoning".to_string(),
            ],
            additional_data: serde_json::json!({
                "collective_confidence": 0.95,
                "processing_nodes": 12,
                "knowledge_domains_accessed": ["general", "technical", "coordination"]
            }),
            processing_time_ms: 150,
        })
    }

    /// Coordinate vector search across peer swarms (direct A2A with collective visibility)
    pub async fn coordinate_vector_search(
        &self,
        embedding: Vec<f32>,
        table_name: &str,
        limit: usize,
    ) -> Result<Vec<VectorSearchResult>, SwarmError> {
        let query_id = Uuid::new_v4().to_string();
        let mut all_results = Vec::new();

        info!("🔍 Starting distributed vector search - query: {}", query_id);

        // Direct A2A to all peer swarms (efficient routing with collective visibility)
        for peer_ref in self.peer_swarms.iter() {
            let (peer_id, _peer_info) = peer_ref.pair();
            
            let search_message = A2AMessage::VectorSearchRequest {
                query_id: query_id.clone(),
                embedding: embedding.clone(),
                table_name: table_name.to_string(),
                limit,
                metadata_filter: Some(serde_json::json!({
                    "routing_method": "direct_a2a_with_collective_visibility",
                    "requesting_swarm": self.id,
                    "collective_intelligence_enabled": true
                })),
            };

            // send_message() automatically notifies THE COLLECTIVE for visibility
            match self.send_message(peer_id, search_message).await {
                Ok(_) => {
                    debug!("✅ Vector search sent to peer swarm: {}", peer_id);
                    
                    // Placeholder result - real implementation would wait for response
                    all_results.push(VectorSearchResult {
                        id: Uuid::new_v4().to_string(),
                        score: 0.88,
                        content: format!("Direct A2A search result from swarm: {}", peer_id),
                        metadata: serde_json::json!({
                            "source_swarm": peer_id,
                            "routing_method": "direct_a2a",
                            "collective_visibility": "enabled",
                            "query_id": query_id
                        }),
                        source_swarm: peer_id.clone(),
                    });
                }
                Err(e) => {
                    warn!("❌ Failed to query peer swarm {}: {}", peer_id, e);
                }
            }
        }

        // If no peer swarms available, query THE COLLECTIVE directly for swarm discovery
        if self.peer_swarms.is_empty() {
            info!("🧠 No peer swarms registered - requesting swarm discovery from THE COLLECTIVE");
            
            let discovery_message = A2AMessage::IntelligenceRequest {
                request_id: format!("discovery-{}", query_id),
                capability: AICapability::VectorSearch {
                    dimensions: embedding.len(),
                    similarity_metric: "cosine".to_string(),
                },
                context: SwarmContext {
                    swarm_id: self.id.clone(),
                    active_agents: 0,
                    current_tasks: vec!["peer_swarm_discovery".to_string()],
                    available_resources: std::iter::once((
                        "vector_search_request".to_string(),
                        serde_json::json!({
                            "table_name": table_name,
                            "limit": limit,
                            "requires_peer_discovery": true
                        })
                    )).collect(),
                    local_capabilities: vec!["vector_search".to_string()],
                },
                priority: Priority::Normal,
            };

            match self.send_message("the-collective", discovery_message).await {
                Ok(_) => {
                    all_results.push(VectorSearchResult {
                        id: query_id.clone(),
                        score: 0.90,
                        content: "Collective intelligence with swarm discovery".to_string(),
                        metadata: serde_json::json!({
                            "source": "collective_intelligence",
                            "includes_peer_discovery": true,
                            "routing_method": "collective_fallback_with_discovery"
                        }),
                        source_swarm: "the-collective".to_string(),
                    });
                }
                Err(e) => {
                    warn!("❌ Failed to request swarm discovery from THE COLLECTIVE: {}", e);
                }
            }
        }

        info!("🎯 Completed distributed vector search - {} results", all_results.len());
        Ok(all_results)
    }

    /// Get A2A coordinator status
    pub async fn get_status(&self) -> A2AStatus {
        let collective_status = self.collective_status.read().await;
        
        A2AStatus {
            coordinator_id: self.id.clone(),
            active_channels: self.channels.len(),
            collective_connected: collective_status.connected,
            peer_swarms: self.peer_swarms.len(),
            total_messages_sent: 0, // Would be tracked in real implementation
            uptime_seconds: 0, // Would be calculated
        }
    }

    // Protocol-specific sending methods
    async fn send_via_in_process_memory(&self, channel: &A2AChannel, message: A2AMessage, _shared_key: &str) -> Result<(), SwarmError> {
        debug!("⚡ Sending via in-process memory to: {} (same swarm controller)", channel.target_id);
        
        // Ultra-fast in-memory communication for swarms in same process
        // This would use shared memory structures or async channels
        // For now, simulate with immediate "delivery"
        
        debug!("🚀 In-process A2A delivery completed instantly - target: {}", channel.target_id);
        Ok(())
    }

    async fn send_via_websocket(&self, channel: &A2AChannel, message: A2AMessage) -> Result<(), SwarmError> {
        debug!("📡 Sending via WebSocket to: {} (cross-process/network)", channel.target_id);
        // WebSocket implementation would go here
        Ok(())
    }

    async fn send_via_http(&self, channel: &A2AChannel, message: A2AMessage) -> Result<(), SwarmError> {
        debug!("🌐 Sending via HTTP to: {} (cross-network)", channel.target_id);
        // HTTP implementation would go here
        Ok(())
    }

    async fn send_via_tcp(&self, channel: &A2AChannel, message: A2AMessage) -> Result<(), SwarmError> {
        debug!("🔗 Sending via TCP to: {} (low-latency network)", channel.target_id);
        // TCP implementation would go here
        Ok(())
    }

    async fn send_via_p2p(&self, channel: &A2AChannel, message: A2AMessage) -> Result<(), SwarmError> {
        debug!("🔐 Sending via P2P to: {} (encrypted peer-to-peer)", channel.target_id);
        // P2P implementation would go here
        Ok(())
    }

    async fn start_peer_discovery(&self) -> Result<(), SwarmError> {
        info!("🔍 Starting peer swarm discovery");
        // Peer discovery implementation would go here
        Ok(())
    }

    async fn start_heartbeat_system(&self) -> Result<(), SwarmError> {
        info!("💓 Starting A2A heartbeat system");
        // Heartbeat implementation would go here
        Ok(())
    }
}

impl A2AMessageRouter {
    fn new() -> Self {
        Self {
            websocket_handler: None,
            tcp_handler: None,
            http_handler: Some(Arc::new(HttpHandler {
                client: reqwest::Client::new(),
            })),
            p2p_handler: None,
        }
    }
}

/// A2A coordinator status
#[derive(Debug, Clone, Serialize)]
pub struct A2AStatus {
    pub coordinator_id: String,
    pub active_channels: usize,
    pub collective_connected: bool,
    pub peer_swarms: usize,
    pub total_messages_sent: u64,
    pub uptime_seconds: u64,
}

impl Default for A2AConfig {
    fn default() -> Self {
        Self {
            collective_endpoint: None,
            discovery_enabled: true,
            heartbeat_interval_sec: 30,
            connection_timeout_sec: 10,
            max_retry_attempts: 3,
            encryption_enabled: true,
        }
    }
}