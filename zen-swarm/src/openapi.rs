//! OpenAPI/Swagger documentation for zen-swarm MCP server
//!
//! Auto-generated Swagger UI and OpenAPI specs for the MCP JSON-RPC API.
//! This provides the same level of API documentation as TypeScript servers,
//! but generated automatically from Rust code.

use crate::{SwarmStats, mcp::*};
use utoipa::OpenApi;

/// OpenAPI documentation for zen-swarm MCP server
#[derive(OpenApi)]
#[openapi(
    paths(
        handle_jsonrpc,
        health_check,
    ),
    components(
        schemas(
            McpRequest,
            McpResponse,
            McpError,
            McpTool,
            McpResource,
            McpToolCall,
            SwarmStats,
            McpConfig,
        )
    ),
    tags(
        (name = "MCP", description = "Model Context Protocol JSON-RPC API"),
        (name = "Health", description = "Health check and monitoring endpoints"),
        (name = "Tools", description = "MCP tool execution"),
        (name = "Resources", description = "MCP resource management"),
    ),
    info(
        title = "zen-swarm MCP Server",
        version = "1.0.0",
        description = "High-performance Model Context Protocol server for swarm intelligence coordination",
        contact(
            name = "zen-swarm Team",
            url = "https://github.com/zen-neural-stack/zen-swarm",
        ),
        license(
            name = "MIT OR Apache-2.0",
            url = "https://github.com/zen-neural-stack/zen-swarm/blob/main/LICENSE"
        )
    ),
    servers(
        (url = "http://localhost:3001", description = "Local development server"),
        (url = "https://api.zen-swarm.com", description = "Production server")
    )
)]
pub struct ApiDoc;

/// JSON-RPC endpoint for MCP protocol
#[utoipa::path(
    post,
    path = "/mcp",
    tag = "MCP",
    summary = "Process MCP JSON-RPC requests",
    description = "Handle Model Context Protocol JSON-RPC requests according to official specification",
    request_body(
        content = McpRequest,
        description = "MCP JSON-RPC request",
        content_type = "application/json"
    ),
    responses(
        (
            status = 200,
            description = "MCP JSON-RPC response",
            body = McpResponse,
            content_type = "application/json",
            example = json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": { "listChanged": false },
                        "resources": { "subscribe": false, "listChanged": false }
                    },
                    "serverInfo": {
                        "name": "zen-swarm",
                        "version": "1.0.0",
                        "description": "High-performance swarm intelligence MCP server"
                    }
                }
            })
        ),
        (
            status = 400,
            description = "Invalid JSON-RPC request",
            body = McpResponse
        ),
        (
            status = 500,
            description = "Internal server error",
            body = McpResponse
        )
    )
)]
pub async fn handle_jsonrpc() {}

/// Health check endpoint
#[utoipa::path(
    get,
    path = "/health",
    tag = "Health",
    summary = "Server health check", 
    description = "Check server health and get current swarm statistics",
    responses(
        (
            status = 200,
            description = "Server is healthy",
            body = serde_json::Value,
            content_type = "application/json",
            example = json!({
                "status": "healthy",
                "server": {
                    "name": "zen-swarm",
                    "version": "1.0.0",
                    "protocol": "2024-11-05"
                },
                "swarm": {
                    "swarm_id": "123e4567-e89b-12d3-a456-426614174000",
                    "active_agents": 5,
                    "total_tasks": 10,
                    "completed_tasks": 8,
                    "failed_tasks": 0,
                    "uptime_seconds": 3600,
                    "memory_usage_mb": 128.5
                }
            })
        )
    )
)]
pub async fn health_check() {}

/// Available MCP Tools documentation
pub const TOOLS_DOCUMENTATION: &str = r#"
# zen-swarm MCP Tools

The following tools are available via the MCP protocol:

## swarm_init
Initialize a new swarm with specified topology.

**Parameters:**
- `topology`: "mesh" | "hierarchical" | "ring" | "star" 
- `maxAgents`: integer (1-1000, default: 10)

**Example:**
```json
{
  "name": "swarm_init",
  "arguments": {
    "topology": "hierarchical", 
    "maxAgents": 50
  }
}
```

## agent_spawn
Spawn a new agent in the swarm.

**Parameters:**
- `type`: "researcher" | "analyst" | "coder" | "coordinator" | "optimizer"
- `name`: string (optional custom name)
- `capabilities`: array of strings (optional)

**Example:**
```json
{
  "name": "agent_spawn",
  "arguments": {
    "type": "researcher",
    "name": "data-researcher-01", 
    "capabilities": ["web_search", "document_analysis"]
  }
}
```

## task_orchestrate
Orchestrate a task across multiple agents.

**Parameters:**
- `task`: string (task description)
- `agents`: array of agent IDs (optional)
- `strategy`: "parallel" | "sequential" | "adaptive" (default: "adaptive")

**Example:**
```json
{
  "name": "task_orchestrate",
  "arguments": {
    "task": "Analyze market trends for Q4 2024",
    "agents": ["researcher-01", "analyst-02"],
    "strategy": "parallel"
  }
}
```

## swarm_status
Get current swarm status and metrics.

**Parameters:**
- `verbose`: boolean (default: false)

**Example:**
```json
{
  "name": "swarm_status", 
  "arguments": {
    "verbose": true
  }
}
```
"#;
