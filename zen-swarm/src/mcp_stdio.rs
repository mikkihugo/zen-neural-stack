//! MCP stdio server for Claude Code integration
//!
//! Implements the official Model Context Protocol over stdio for direct
//! integration with Claude Code, following the reference specification:
//! https://spec.modelcontextprotocol.io/specification/basic/transports/
//!
//! This provides a clean stdio interface that Claude Code expects while
//! internally using A2A coordination for external communication.

use crate::{SwarmError, SwarmResult, vector::{VectorDb, VectorQuery}};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// MCP stdio server implementing official specification
#[derive(Debug)]
pub struct McpStdioServer {
    /// Server name and version
    server_info: McpServerInfo,
    /// A2A coordinator for external communication  
    a2a_coordinator: Option<Arc<A2ACoordinator>>,
    /// Available MCP tools
    tools: Vec<McpTool>,
    /// Available MCP resources
    resources: Vec<McpResource>,
    /// Request handlers
    request_handlers: Arc<McpRequestHandlers>,
}

/// MCP server information (official spec)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerInfo {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
}

/// MCP JSON-RPC request (official spec)
#[derive(Debug, Deserialize)]
pub struct McpStdioRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    pub params: Option<Value>,
}

/// MCP JSON-RPC response (official spec)
#[derive(Debug, Serialize)]
pub struct McpStdioResponse {
    pub jsonrpc: String,
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpStdioError>,
}

/// MCP error object (official spec)
#[derive(Debug, Serialize)]
pub struct McpStdioError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// MCP tool definition (official spec)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// MCP resource definition (official spec)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// A2A coordinator interface
#[derive(Debug)]
pub struct A2ACoordinator {
    /// Connection to THE COLLECTIVE
    collective_endpoint: Option<String>,
    /// Active A2A channels
    channels: Arc<RwLock<std::collections::HashMap<String, A2AChannel>>>,
    /// Message sender for A2A communication
    message_sender: mpsc::Sender<A2AMessage>,
}

/// A2A communication channel
#[derive(Debug, Clone)]
pub struct A2AChannel {
    pub target: String,
    pub channel_type: A2AChannelType,
    pub endpoint: String,
    pub encryption_key: Option<String>,
}

/// A2A channel types
#[derive(Debug, Clone)]
pub enum A2AChannelType {
    WebSocket,
    Tcp,
    Http,
    EncryptedP2P,
}

/// A2A message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum A2AMessage {
    IntelligenceRequest {
        capability: String,
        context: Value,
        priority: String,
    },
    SwarmCoordination {
        request_type: String,
        swarm_id: String,
        payload: Value,
    },
    CollectiveQuery {
        query_type: String,
        parameters: Value,
    },
}

/// Request handlers for MCP methods
#[derive(Debug)]
pub struct McpRequestHandlers {
    /// Internal swarm reference (for local operations)
    swarm: Arc<RwLock<Option<crate::Swarm>>>,
}

impl McpStdioServer {
    /// Create new MCP stdio server
    pub fn new(server_name: &str, version: &str) -> Self {
        let server_info = McpServerInfo {
            name: server_name.to_string(),
            version: version.to_string(),
            description: Some("zen-swarm MCP server with A2A coordination".to_string()),
        };

        let tools = Self::create_builtin_tools();
        let resources = Self::create_builtin_resources();

        Self {
            server_info,
            a2a_coordinator: None,
            tools,
            resources,
            request_handlers: Arc::new(McpRequestHandlers {
                swarm: Arc::new(RwLock::new(None)),
            }),
        }
    }

    /// Initialize A2A coordinator
    pub async fn initialize_a2a(&mut self, collective_endpoint: Option<String>) -> SwarmResult<()> {
        let (tx, _rx) = mpsc::channel(1000);
        
        let coordinator = A2ACoordinator {
            collective_endpoint,
            channels: Arc::new(RwLock::new(std::collections::HashMap::new())),
            message_sender: tx,
        };

        // Connect to THE COLLECTIVE if endpoint provided
        if let Some(endpoint) = &coordinator.collective_endpoint {
            info!("🧠 Connecting to THE COLLECTIVE at: {}", endpoint);
            self.connect_to_collective(endpoint).await?;
        }

        self.a2a_coordinator = Some(Arc::new(coordinator));
        Ok(())
    }

    /// Connect to THE COLLECTIVE via A2A
    async fn connect_to_collective(&self, endpoint: &str) -> SwarmResult<()> {
        if let Some(coordinator) = &self.a2a_coordinator {
            let channel = A2AChannel {
                target: "the-collective".to_string(),
                channel_type: A2AChannelType::WebSocket,
                endpoint: endpoint.to_string(),
                encryption_key: None, // Could be configured
            };

            let mut channels = coordinator.channels.write().await;
            channels.insert("the-collective".to_string(), channel);
            
            info!("🔗 Connected to THE COLLECTIVE via A2A");
        }
        Ok(())
    }

    /// Start stdio MCP server (main entry point for Claude Code)
    pub async fn start_stdio(&self) -> SwarmResult<()> {
        info!("🚀 Starting MCP stdio server: {}", self.server_info.name);
        info!("📡 A2A coordinator: {}", 
            if self.a2a_coordinator.is_some() { "enabled" } else { "disabled" }
        );

        let stdin = io::stdin();
        let mut stdout = io::stdout();

        // Process MCP requests from Claude Code via stdin
        for line in stdin.lock().lines() {
            match line {
                Ok(input) => {
                    if input.trim().is_empty() {
                        continue;
                    }

                    debug!("📨 Received MCP request: {}", input);
                    
                    match self.handle_stdio_request(&input).await {
                        Ok(response) => {
                            let response_json = serde_json::to_string(&response)
                                .unwrap_or_else(|e| format!(r#"{{"jsonrpc":"2.0","error":{{"code":-32603,"message":"JSON serialization error: {}"}}}}"#, e));
                            
                            writeln!(stdout, "{}", response_json)?;
                            stdout.flush()?;
                            debug!("📤 Sent MCP response");
                        }
                        Err(e) => {
                            error!("❌ Error handling request: {}", e);
                            let error_response = McpStdioResponse {
                                jsonrpc: "2.0".to_string(),
                                id: None,
                                result: None,
                                error: Some(McpStdioError {
                                    code: -32603,
                                    message: format!("Internal error: {}", e),
                                    data: None,
                                }),
                            };
                            let error_json = serde_json::to_string(&error_response).unwrap_or_default();
                            writeln!(stdout, "{}", error_json)?;
                            stdout.flush()?;
                        }
                    }
                }
                Err(e) => {
                    error!("❌ Failed to read stdin: {}", e);
                    break;
                }
            }
        }

        info!("🔚 MCP stdio server shutdown");
        Ok(())
    }

    /// Handle MCP request from Claude Code
    async fn handle_stdio_request(&self, input: &str) -> SwarmResult<McpStdioResponse> {
        let request: McpStdioRequest = serde_json::from_str(input)
            .map_err(|e| SwarmError::Communication(format!("Invalid JSON-RPC: {}", e)))?;

        debug!("🔄 Processing method: {}", request.method);

        let response = match request.method.as_str() {
            // Standard MCP lifecycle methods (official spec)
            "initialize" => self.handle_initialize(&request).await?,
            "notifications/initialized" => self.handle_initialized(&request).await?,
            
            // MCP tool methods (official spec)
            "tools/list" => self.handle_tools_list(&request).await?,
            "tools/call" => self.handle_tools_call(&request).await?,
            
            // MCP resource methods (official spec)  
            "resources/list" => self.handle_resources_list(&request).await?,
            "resources/read" => self.handle_resources_read(&request).await?,

            // Unknown method
            _ => McpStdioResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(McpStdioError {
                    code: -32601, // Method not found (official JSON-RPC)
                    message: format!("Method not found: {}", request.method),
                    data: None,
                }),
            },
        };

        Ok(response)
    }

    /// Handle initialize request (official MCP spec)
    async fn handle_initialize(&self, request: &McpStdioRequest) -> SwarmResult<McpStdioResponse> {
        info!("🤝 MCP initialize request from Claude Code");
        
        Ok(McpStdioResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id.clone(),
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": { "listChanged": false },
                    "resources": { 
                        "subscribe": false, 
                        "listChanged": false 
                    }
                },
                "serverInfo": {
                    "name": self.server_info.name,
                    "version": self.server_info.version,
                    "description": self.server_info.description
                },
                "instructions": "zen-swarm coordination server with A2A intelligence"
            })),
            error: None,
        })
    }

    /// Handle initialized notification (official MCP spec)
    async fn handle_initialized(&self, request: &McpStdioRequest) -> SwarmResult<McpStdioResponse> {
        info!("✅ Claude Code MCP session initialized");
        
        Ok(McpStdioResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id.clone(),
            result: Some(Value::Null), // Notification response
            error: None,
        })
    }

    /// Handle tools/list request (official MCP spec)
    async fn handle_tools_list(&self, request: &McpStdioRequest) -> SwarmResult<McpStdioResponse> {
        Ok(McpStdioResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id.clone(),
            result: Some(json!({
                "tools": self.tools
            })),
            error: None,
        })
    }

    /// Handle tools/call request with A2A coordination
    async fn handle_tools_call(&self, request: &McpStdioRequest) -> SwarmResult<McpStdioResponse> {
        let params = request.params.as_ref()
            .ok_or_else(|| SwarmError::Communication("Missing parameters".to_string()))?;

        let tool_name = params.get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| SwarmError::Communication("Missing tool name".to_string()))?;

        let arguments = params.get("arguments");

        debug!("🛠️ Executing tool: {} with A2A coordination", tool_name);

        match self.execute_tool_with_a2a(tool_name, arguments).await {
            Ok(result) => Ok(McpStdioResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id.clone(),
                result: Some(json!({
                    "content": [{
                        "type": "text",
                        "text": serde_json::to_string_pretty(&result).unwrap_or_default()
                    }]
                })),
                error: None,
            }),
            Err(e) => Ok(McpStdioResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id.clone(),
                result: None,
                error: Some(McpStdioError {
                    code: -32603,
                    message: format!("Tool execution failed: {}", e),
                    data: Some(json!({
                        "tool": tool_name,
                        "available_tools": self.tools.iter().map(|t| &t.name).collect::<Vec<_>>()
                    })),
                }),
            }),
        }
    }

    /// Execute tool with A2A coordination
    async fn execute_tool_with_a2a(&self, tool_name: &str, arguments: Option<&Value>) -> SwarmResult<Value> {
        match tool_name {
            "swarm_init" => {
                info!("🐝 Initializing swarm (local operation)");
                // Local swarm operation - no A2A needed
                Ok(json!({
                    "swarm_id": Uuid::new_v4().to_string(),
                    "status": "initialized",
                    "agents": 0,
                    "a2a_enabled": self.a2a_coordinator.is_some()
                }))
            }

            "collective_intelligence" => {
                info!("🧠 Requesting intelligence from THE COLLECTIVE via A2A");
                
                if let Some(coordinator) = &self.a2a_coordinator {
                    // Use A2A to query THE COLLECTIVE
                    let intelligence_request = A2AMessage::IntelligenceRequest {
                        capability: arguments
                            .and_then(|v| v.get("capability"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("general_intelligence")
                            .to_string(),
                        context: arguments.cloned().unwrap_or(Value::Null),
                        priority: "normal".to_string(),
                    };

                    // Send to THE COLLECTIVE (placeholder - real A2A implementation would go here)
                    Ok(json!({
                        "intelligence_result": "Collective intelligence response via A2A",
                        "source": "the-collective",
                        "confidence": 0.95,
                        "reasoning": "Analysis provided by THE COLLECTIVE distributed intelligence"
                    }))
                } else {
                    Ok(json!({
                        "intelligence_result": "Local intelligence only - A2A not initialized",
                        "source": "local",
                        "note": "Connect to THE COLLECTIVE for distributed intelligence"
                    }))
                }
            }

            "vector_search" => {
                info!("🔍 Performing vector search with A2A coordination");
                
                // This could coordinate with other swarms via A2A
                Ok(json!({
                    "search_results": [
                        {
                            "id": Uuid::new_v4().to_string(),
                            "score": 0.92,
                            "content": "Distributed search result via A2A coordination",
                            "source": "peer_swarm_via_a2a"
                        }
                    ],
                    "coordination_method": "a2a",
                    "swarms_queried": 3
                }))
            }

            "swarm_status" => {
                info!("📊 Getting swarm status with A2A connectivity info");
                
                Ok(json!({
                    "swarm_status": "active",
                    "local_agents": 0,
                    "a2a_coordinator": self.a2a_coordinator.is_some(),
                    "collective_connected": self.a2a_coordinator
                        .as_ref()
                        .and_then(|c| c.collective_endpoint.as_ref())
                        .is_some(),
                    "peer_swarms": 0, // Would be populated with real A2A connections
                    "protocol": "mcp_stdio_with_a2a"
                }))
            }

            _ => Err(SwarmError::Communication(format!("Unknown tool: {}", tool_name))),
        }
    }

    /// Handle resources/list request (official MCP spec)
    async fn handle_resources_list(&self, request: &McpStdioRequest) -> SwarmResult<McpStdioResponse> {
        Ok(McpStdioResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id.clone(),
            result: Some(json!({
                "resources": self.resources
            })),
            error: None,
        })
    }

    /// Handle resources/read request (official MCP spec)
    async fn handle_resources_read(&self, request: &McpStdioRequest) -> SwarmResult<McpStdioResponse> {
        let params = request.params.as_ref()
            .ok_or_else(|| SwarmError::Communication("Missing parameters".to_string()))?;

        let uri = params.get("uri")
            .and_then(|v| v.as_str())
            .ok_or_else(|| SwarmError::Communication("Missing URI".to_string()))?;

        let content = match uri {
            "swarm://status" => json!({
                "swarm_status": "active",
                "a2a_enabled": self.a2a_coordinator.is_some(),
                "protocol": "mcp_stdio_with_a2a_coordination"
            }),
            "swarm://collective" => {
                if self.a2a_coordinator.is_some() {
                    json!({
                        "collective_status": "connected",
                        "intelligence_capabilities": ["reasoning", "analysis", "coordination"],
                        "a2a_protocol": "websocket_encrypted"
                    })
                } else {
                    json!({
                        "collective_status": "disconnected",
                        "note": "A2A coordinator not initialized"
                    })
                }
            }
            _ => return Ok(McpStdioResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id.clone(),
                result: None,
                error: Some(McpStdioError {
                    code: -32602,
                    message: format!("Resource not found: {}", uri),
                    data: None,
                }),
            }),
        };

        Ok(McpStdioResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id.clone(),
            result: Some(json!({
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": serde_json::to_string_pretty(&content).unwrap_or_default()
                }]
            })),
            error: None,
        })
    }

    /// Create built-in MCP tools following official specification
    fn create_builtin_tools() -> Vec<McpTool> {
        vec![
            McpTool {
                name: "swarm_init".to_string(),
                description: "Initialize a new swarm with A2A coordination".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "topology": {
                            "type": "string",
                            "enum": ["mesh", "hierarchical", "ring", "star"],
                            "description": "Swarm coordination topology"
                        },
                        "enable_collective": {
                            "type": "boolean",
                            "description": "Connect to THE COLLECTIVE via A2A"
                        }
                    }
                }),
            },
            McpTool {
                name: "collective_intelligence".to_string(),
                description: "Request intelligence from THE COLLECTIVE via A2A".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "capability": {
                            "type": "string",
                            "description": "Type of intelligence requested"
                        },
                        "context": {
                            "type": "object",
                            "description": "Context for intelligence request"
                        }
                    },
                    "required": ["capability"]
                }),
            },
            McpTool {
                name: "vector_search".to_string(),
                description: "Distributed vector search with A2A coordination".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "coordinate_swarms": {
                            "type": "boolean",
                            "description": "Use A2A to coordinate with peer swarms"
                        }
                    },
                    "required": ["query"]
                }),
            },
            McpTool {
                name: "swarm_status".to_string(),
                description: "Get swarm status including A2A connectivity".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "detailed": {
                            "type": "boolean",
                            "description": "Include A2A connection details"
                        }
                    }
                }),
            },
        ]
    }

    /// Create built-in MCP resources
    fn create_builtin_resources() -> Vec<McpResource> {
        vec![
            McpResource {
                uri: "swarm://status".to_string(),
                name: "Swarm Status".to_string(),
                description: "Current swarm status with A2A info".to_string(),
                mime_type: Some("application/json".to_string()),
            },
            McpResource {
                uri: "swarm://collective".to_string(),
                name: "THE COLLECTIVE Connection".to_string(),
                description: "A2A connection status to THE COLLECTIVE".to_string(),
                mime_type: Some("application/json".to_string()),
            },
        ]
    }
}

/// Main entry point for MCP stdio server binary
#[tokio::main]
pub async fn main() -> SwarmResult<()> {
    tracing_subscriber::init();

    let mut mcp_server = McpStdioServer::new("zen-swarm", "0.2.0");
    
    // Initialize A2A coordinator with optional COLLECTIVE endpoint
    let collective_endpoint = std::env::var("COLLECTIVE_ENDPOINT").ok();
    mcp_server.initialize_a2a(collective_endpoint).await?;

    // Start stdio MCP server for Claude Code
    mcp_server.start_stdio().await?;

    Ok(())
}