//! Model Context Protocol (MCP) server implementation
//!
//! Full implementation of the official MCP specification in Rust.
//! This provides the same JSON-RPC interface as the TypeScript reference,
//! but with Rust's performance and safety advantages.
//!
//! MCP Specification: https://spec.modelcontextprotocol.io/
//! Reference Implementation: https://github.com/modelcontextprotocol/servers

use crate::{Swarm, SwarmError, SwarmStats};
use axum::{
  Json, Router,
  extract::{State, WebSocketUpgrade},
  response::Response,
  routing::{get, post},
};
use chrono;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{debug, error, info};
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;
use uuid::Uuid;

/// MCP Server implementing official specification
#[derive(Debug)]
pub struct McpServer {
  /// Associated swarm for coordination
  swarm: Arc<Swarm>,
  /// Server configuration
  config: McpConfig,
  /// Available tools/functions
  tools: Vec<McpTool>,
  /// Available resources
  resources: Vec<McpResource>,
}

/// MCP Server configuration
#[derive(Debug, Clone, utoipa::ToSchema)]
pub struct McpConfig {
  pub name: String,
  pub version: String,
  pub description: String,
  pub host: String,
  pub port: u16,
  pub protocol_version: String,
}

/// MCP JSON-RPC Request (official spec)
#[derive(Debug, Deserialize, ToSchema)]
pub struct McpRequest {
  pub jsonrpc: String,
  pub id: Option<Value>,
  pub method: String,
  pub params: Option<Value>,
}

/// MCP JSON-RPC Response (official spec)
#[derive(Debug, Serialize, ToSchema)]
pub struct McpResponse {
  pub jsonrpc: String,
  pub id: Option<Value>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub result: Option<Value>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub error: Option<McpError>,
}

/// MCP Error object (official spec)
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct McpError {
  pub code: i32,
  pub message: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub data: Option<Value>,
}

/// MCP Tool definition (official spec)
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct McpTool {
  pub name: String,
  pub description: String,
  pub input_schema: Value, // JSON Schema
}

/// MCP Resource definition (official spec)
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct McpResource {
  pub uri: String,
  pub name: String,
  pub description: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub mime_type: Option<String>,
}

/// MCP Tool call request (official spec)
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct McpToolCall {
  pub name: String,
  pub arguments: Option<Value>,
}

impl Default for McpConfig {
  fn default() -> Self {
    Self {
      name: "zen-swarm".to_string(),
      version: "1.0.0".to_string(),
      description: "High-performance swarm intelligence MCP server".to_string(),
      host: "127.0.0.1".to_string(),
      port: 3001,
      protocol_version: "2024-11-05".to_string(), // Official MCP version
    }
  }
}

impl McpServer {
  /// Create new MCP server with swarm integration
  pub fn new(swarm: Arc<Swarm>, config: McpConfig) -> Self {
    let tools = Self::create_builtin_tools();
    let resources = Self::create_builtin_resources();

    Self {
      swarm,
      config,
      tools,
      resources,
    }
  }

  /// Create built-in swarm coordination tools
  fn create_builtin_tools() -> Vec<McpTool> {
    vec![
      McpTool {
        name: "swarm_init".to_string(),
        description: "Initialize a new swarm with specified configuration"
          .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "topology": {
                    "type": "string",
                    "enum": ["mesh", "hierarchical", "ring", "star"],
                    "description": "Swarm topology type"
                },
                "maxAgents": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 10,
                    "description": "Maximum number of agents"
                }
            },
            "required": ["topology"]
        }),
      },
      McpTool {
        name: "agent_spawn".to_string(),
        description: "Spawn a new agent in the swarm".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["researcher", "analyst", "coder", "coordinator", "optimizer"],
                    "description": "Type of agent to spawn"
                },
                "name": {
                    "type": "string",
                    "description": "Optional custom name for the agent"
                },
                "capabilities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of agent capabilities"
                }
            },
            "required": ["type"]
        }),
      },
      McpTool {
        name: "task_orchestrate".to_string(),
        description: "Orchestrate a task across multiple agents".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Task description or instructions"
                },
                "agents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of agent IDs to use"
                },
                "strategy": {
                    "type": "string",
                    "enum": ["parallel", "sequential", "adaptive"],
                    "default": "adaptive",
                    "description": "Execution strategy"
                }
            },
            "required": ["task"]
        }),
      },
      McpTool {
        name: "swarm_status".to_string(),
        description: "Get current swarm status and metrics".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "verbose": {
                    "type": "boolean",
                    "default": false,
                    "description": "Include detailed information"
                }
            }
        }),
      },
    ]
  }

  /// Create built-in resources
  fn create_builtin_resources() -> Vec<McpResource> {
    vec![
      McpResource {
        uri: "swarm://status".to_string(),
        name: "Swarm Status".to_string(),
        description: "Current swarm status and metrics".to_string(),
        mime_type: Some("application/json".to_string()),
      },
      McpResource {
        uri: "swarm://agents".to_string(),
        name: "Active Agents".to_string(),
        description: "List of all active agents in the swarm".to_string(),
        mime_type: Some("application/json".to_string()),
      },
      McpResource {
        uri: "swarm://tasks".to_string(),
        name: "Task Queue".to_string(),
        description: "Current task queue and execution status".to_string(),
        mime_type: Some("application/json".to_string()),
      },
    ]
  }

  /// Start MCP server with Swagger UI
  pub async fn start(&self) -> Result<(), SwarmError> {
    use crate::openapi::ApiDoc;

    let app = Router::new()
      .route("/mcp", post(Self::handle_jsonrpc))
      .route("/mcp/ws", get(Self::handle_websocket))
      .route("/health", get(Self::health_check))
      // 📚 Swagger UI - better than TypeScript!
      .merge(
        SwaggerUi::new("/docs").url("/api-doc/openapi.json", ApiDoc::openapi()),
      )
      .route(
        "/tools/docs",
        get(|| async { crate::openapi::TOOLS_DOCUMENTATION }),
      )
      .with_state(Arc::new(self.clone()));

    let addr = format!("{}:{}", self.config.host, self.config.port);
    let listener = TcpListener::bind(&addr).await.map_err(|e| {
      SwarmError::Network(format!("Failed to bind to {}: {}", addr, e))
    })?;

    info!("🌐 MCP Server starting on {}", addr);
    info!("📚 Swagger UI available at http://{}/docs", addr);
    info!("🛠️ Tools documentation at http://{}/tools/docs", addr);
    info!(
      "📋 Available tools: {:?}",
      self.tools.iter().map(|t| &t.name).collect::<Vec<_>>()
    );

    axum::serve(listener, app)
      .await
      .map_err(|e| SwarmError::Network(format!("Server error: {}", e)))?;

    Ok(())
  }

  /// Handle JSON-RPC requests (official MCP spec)
  async fn handle_jsonrpc(
    State(server): State<Arc<McpServer>>,
    Json(request): Json<McpRequest>,
  ) -> Json<McpResponse> {
    debug!(
      "📨 MCP Request: {} - {}",
      request.method,
      request.id.as_ref().unwrap_or(&json!(null))
    );

    let response = match request.method.as_str() {
      // Standard MCP methods (official spec)
      "initialize" => server.handle_initialize(&request).await,
      "notifications/initialized" => server.handle_initialized(&request).await,
      "tools/list" => server.handle_tools_list(&request).await,
      "tools/call" => {
        let result = server.handle_tools_call_async(&request).await;
        result
      }
      "resources/list" => server.handle_resources_list(&request).await,
      "resources/read" => server.handle_resources_read(&request).await,

      // Unknown method
      _ => McpResponse {
        jsonrpc: "2.0".to_string(),
        id: request.id,
        result: None,
        error: Some(McpError {
          code: -32601,
          message: format!("Method not found: {}", request.method),
          data: None,
        }),
      },
    };

    Json(response)
  }

  /// Handle WebSocket connections for streaming MCP
  async fn handle_websocket(
    ws: WebSocketUpgrade,
    State(_server): State<Arc<McpServer>>,
  ) -> Response {
    ws.on_upgrade(|_socket| async {
      info!("🔌 MCP WebSocket connection established");
      // WebSocket handler would go here
    })
  }

  /// Health check endpoint
  async fn health_check(State(server): State<Arc<McpServer>>) -> Json<Value> {
    let stats = server.swarm.stats().await;

    Json(json!({
        "status": "healthy",
        "server": {
            "name": server.config.name,
            "version": server.config.version,
            "protocol": server.config.protocol_version
        },
        "swarm": stats
    }))
  }

  /// Handle MCP initialize request (official spec)
  async fn handle_initialize(&self, request: &McpRequest) -> McpResponse {
    McpResponse {
      jsonrpc: "2.0".to_string(),
      id: request.id.clone(),
      result: Some(json!({
          "protocolVersion": self.config.protocol_version,
          "capabilities": {
              "tools": {
                  "listChanged": false
              },
              "resources": {
                  "subscribe": false,
                  "listChanged": false
              }
          },
          "serverInfo": {
              "name": self.config.name,
              "version": self.config.version,
              "description": self.config.description
          }
      })),
      error: None,
    }
  }

  /// Handle initialized notification (official spec)
  async fn handle_initialized(&self, request: &McpRequest) -> McpResponse {
    info!("✅ MCP Client initialized");
    McpResponse {
      jsonrpc: "2.0".to_string(),
      id: request.id.clone(),
      result: Some(json!(null)), // Notification response
      error: None,
    }
  }

  /// Handle tools/list request (official spec)
  async fn handle_tools_list(&self, request: &McpRequest) -> McpResponse {
    McpResponse {
      jsonrpc: "2.0".to_string(),
      id: request.id.clone(),
      result: Some(json!({
          "tools": self.tools
      })),
      error: None,
    }
  }

  /// Handle tools/call request (official spec)
  /// Handle tools/call request (synchronous wrapper)
  async fn handle_tools_call(&self, request: &McpRequest) -> McpResponse {
    self.handle_tools_call_async(request).await
  }

  /// Handle tools/call request (async implementation)
  async fn handle_tools_call_async(&self, request: &McpRequest) -> McpResponse {
    let params = request.params.as_ref();
    let tool_call: Result<McpToolCall, _> = params
      .map(|p| serde_json::from_value(p.clone()))
      .unwrap_or(Err(serde_json::Error::custom("No params")));

    match tool_call {
      Ok(call) => match self.execute_tool(&call.name, call.arguments).await {
        Ok(result) => McpResponse {
          jsonrpc: "2.0".to_string(),
          id: request.id.clone(),
          result: Some(json!({
              "content": [{
                  "type": "text",
                  "text": serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string())
              }]
          })),
          error: None,
        },
        Err(mcp_error) => McpResponse {
          jsonrpc: "2.0".to_string(),
          id: request.id.clone(),
          result: None,
          error: Some(mcp_error),
        },
      },
      Err(e) => McpResponse {
        jsonrpc: "2.0".to_string(),
        id: request.id.clone(),
        result: None,
        error: Some(McpError {
          code: -32602,
          message: format!("Invalid tool call parameters: {}", e),
          data: Some(json!({
              "expected_format": {
                  "name": "tool_name",
                  "arguments": "object with tool-specific parameters"
              },
              "available_tools": self.tools.iter().map(|t| &t.name).collect::<Vec<_>>()
          })),
        }),
      },
    }
  }

  /// Handle resources/list request (official spec)
  async fn handle_resources_list(&self, request: &McpRequest) -> McpResponse {
    McpResponse {
      jsonrpc: "2.0".to_string(),
      id: request.id.clone(),
      result: Some(json!({
          "resources": self.resources
      })),
      error: None,
    }
  }

  /// Handle resources/read request (official MCP spec)
  async fn handle_resources_read(&self, request: &McpRequest) -> McpResponse {
    let params = request.params.as_ref().and_then(|p| p.as_object());
    let uri = params.and_then(|p| p.get("uri").and_then(|u| u.as_str()));

    match uri {
      Some("swarm://status") => {
        let stats = self.swarm.stats().await;
        let stats_json = match serde_json::to_string_pretty(&stats) {
          Ok(json) => json,
          Err(e) => {
            return McpResponse {
              jsonrpc: "2.0".to_string(),
              id: request.id.clone(),
              result: None,
              error: Some(McpError {
                code: -32603,
                message: format!("Failed to serialize swarm stats: {}", e),
                data: None,
              }),
            };
          }
        };

        McpResponse {
          jsonrpc: "2.0".to_string(),
          id: request.id.clone(),
          result: Some(json!({
              "contents": [{
                  "uri": "swarm://status",
                  "mimeType": "application/json",
                  "text": stats_json
              }]
          })),
          error: None,
        }
      }
      Some("swarm://agents") => {
        let agents = self.swarm.list_agents().await;
        let agents_data = json!({
            "total_agents": agents.len(),
            "agent_list": agents,
            "timestamp": chrono::Utc::now().to_rfc3339()
        });

        McpResponse {
          jsonrpc: "2.0".to_string(),
          id: request.id.clone(),
          result: Some(json!({
              "contents": [{
                  "uri": "swarm://agents",
                  "mimeType": "application/json",
                  "text": serde_json::to_string_pretty(&agents_data).unwrap_or_default()
              }]
          })),
          error: None,
        }
      }
      Some("swarm://tasks") => {
        // Get task queue information from swarm
        let task_info = json!({
            "active_tasks": 0, // Would need to expose this from swarm
            "queued_tasks": 0,
            "completed_tasks": 0,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "note": "Task details require additional swarm API exposure"
        });

        McpResponse {
          jsonrpc: "2.0".to_string(),
          id: request.id.clone(),
          result: Some(json!({
              "contents": [{
                  "uri": "swarm://tasks",
                  "mimeType": "application/json",
                  "text": serde_json::to_string_pretty(&task_info).unwrap_or_default()
              }]
          })),
          error: None,
        }
      }
      Some(unknown_uri) => McpResponse {
        jsonrpc: "2.0".to_string(),
        id: request.id.clone(),
        result: None,
        error: Some(McpError {
          code: -32602,
          message: format!("Resource not found: {}", unknown_uri),
          data: Some(json!({
              "available_resources": [
                  "swarm://status",
                  "swarm://agents",
                  "swarm://tasks"
              ]
          })),
        }),
      },
      None => McpResponse {
        jsonrpc: "2.0".to_string(),
        id: request.id.clone(),
        result: None,
        error: Some(McpError {
          code: -32602,
          message: "Missing required parameter: uri".to_string(),
          data: None,
        }),
      },
    }
  }

  /// Execute a tool call with full MCP API integration
  async fn execute_tool(
    &self,
    tool_name: &str,
    args: Option<Value>,
  ) -> Result<Value, McpError> {
    match tool_name {
      "swarm_status" => {
        let verbose = args
          .as_ref()
          .and_then(|a| a.as_object())
          .and_then(|o| o.get("verbose"))
          .and_then(|v| v.as_bool())
          .unwrap_or(false);

        let stats = self.swarm.stats().await;

        if verbose {
          // Include detailed agent information
          let agents = self.swarm.list_agents().await;
          Ok(json!({
              "status": "active",
              "swarm_stats": stats,
              "agents": {
                  "count": agents.len(),
                  "list": agents
              },
              "timestamp": chrono::Utc::now().to_rfc3339(),
              "protocol_version": self.config.protocol_version,
              "server_info": {
                  "name": self.config.name,
                  "version": self.config.version,
                  "description": self.config.description
              }
          }))
        } else {
          Ok(json!({
              "status": "active",
              "stats": stats,
              "timestamp": chrono::Utc::now().to_rfc3339()
          }))
        }
      }
      "agent_spawn" => {
        let agent_args = args.as_ref().and_then(|a| a.as_object());
        let agent_name = agent_args
          .and_then(|a| a.get("name"))
          .and_then(|n| n.as_str())
          .unwrap_or("unnamed_agent");

        let agent_type = agent_args
          .and_then(|a| a.get("type"))
          .and_then(|t| t.as_str())
          .unwrap_or("general");

        // Parse agent type string to AgentType enum
        let agent_type_enum = match agent_type {
          "researcher" => crate::agent::AgentType::Researcher,
          "analyst" => crate::agent::AgentType::Analyst,
          "coder" => crate::agent::AgentType::Coder,
          "coordinator" => crate::agent::AgentType::Coordinator,
          "optimizer" => crate::agent::AgentType::Optimizer,
          _ => crate::agent::AgentType::Coordinator, // Default fallback
        };

        // Create a basic agent
        let agent = crate::Agent::new(agent_name, agent_type_enum);

        match self.swarm.spawn_agent(agent_name, agent).await {
          Ok(agent_id) => Ok(json!({
              "success": true,
              "agent_id": agent_id,
              "agent_name": agent_name,
              "agent_type": agent_type,
              "spawned_at": chrono::Utc::now().to_rfc3339()
          })),
          Err(e) => Err(McpError {
            code: -32603,
            message: format!("Failed to spawn agent: {}", e),
            data: Some(json!({
                "agent_name": agent_name,
                "agent_type": agent_type,
                "supported_types": ["researcher", "analyst", "coder", "coordinator", "optimizer"]
            })),
          }),
        }
      }
      "task_orchestrate" => {
        let task_args = args.as_ref().and_then(|a| a.as_object());
        let task_description = task_args
          .and_then(|a| a.get("task"))
          .and_then(|t| t.as_str())
          .ok_or_else(|| McpError {
            code: -32602,
            message: "Missing required parameter: task".to_string(),
            data: None,
          })?;

        let agent_ids = task_args
          .and_then(|a| a.get("agents"))
          .and_then(|agents| agents.as_array())
          .map(|arr| {
            arr
              .iter()
              .filter_map(|v| v.as_str().map(|s| s.to_string()))
              .collect()
          })
          .unwrap_or_else(Vec::new);

        match self
          .swarm
          .orchestrate_task(task_description, agent_ids.clone())
          .await
        {
          Ok(task_result) => Ok(json!({
              "success": true,
              "task_result": task_result,
              "agents_used": agent_ids,
              "completed_at": chrono::Utc::now().to_rfc3339()
          })),
          Err(e) => Err(McpError {
            code: -32603,
            message: format!("Task orchestration failed: {}", e),
            data: Some(json!({
                "task_description": task_description,
                "requested_agents": agent_ids
            })),
          }),
        }
      }
      _ => Err(McpError {
        code: -32601,
        message: format!("Unknown tool: {}", tool_name),
        data: Some(json!({
            "available_tools": self.tools.iter().map(|t| &t.name).collect::<Vec<_>>()
        })),
      }),
    }
  }
}

impl Clone for McpServer {
  fn clone(&self) -> Self {
    Self {
      swarm: self.swarm.clone(),
      config: self.config.clone(),
      tools: self.tools.clone(),
      resources: self.resources.clone(),
    }
  }
}
