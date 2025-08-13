# 🖥️ Direct CLI LLM Integration via Swarm Daemons

## 🎯 **DIRECT CLI ARCHITECTURE: No Plugins Needed**

### **Each LLM gets its own CLI that connects directly to swarm daemon**

```
gemini-cli ←→ zen-swarm daemon ←→ THE COLLECTIVE
claude-cli ←→ zen-swarm daemon ←→ THE COLLECTIVE  
gpt-cli    ←→ zen-swarm daemon ←→ THE COLLECTIVE
llama-cli  ←→ zen-swarm daemon ←→ THE COLLECTIVE
```

**No plugins required!** Each CLI is a standalone tool that uses the same swarm infrastructure.

## 🛠️ **CLI IMPLEMENTATIONS**

### **Gemini CLI (Direct Connection)**
```rust
// gemini-cli/src/main.rs
use clap::{Arg, Command};
use zen_swarm::{SwarmDaemonClient, A2AMessage, AICapability};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("gemini-cli")
        .version("1.0.0")
        .about("Google Gemini CLI via zen-swarm")
        .arg(Arg::new("prompt")
            .help("Prompt for Gemini")
            .required(true)
            .index(1))
        .arg(Arg::new("model")
            .long("model")
            .short('m')
            .help("Gemini model to use")
            .default_value("gemini-1.5-pro"))
        .arg(Arg::new("daemon-port")
            .long("daemon-port")
            .help("Swarm daemon port")
            .default_value("9001"))
        .get_matches();

    let prompt = matches.get_one::<String>("prompt").unwrap();
    let model = matches.get_one::<String>("model").unwrap();
    let port: u16 = matches.get_one::<String>("daemon-port").unwrap().parse()?;

    // Connect directly to local swarm daemon
    let client = SwarmDaemonClient::new(format!("http://localhost:{}", port));
    
    // Send coding task to swarm daemon (swarm executes the task)
    let response = client.execute_coding_task(A2AMessage::IntelligenceRequest {
        request_id: uuid::Uuid::new_v4().to_string(),
        capability: AICapability::LLMReasoning {
            model: model.clone(),
            context_length: 8192,
        },
        context: zen_swarm::SwarmContext {
            swarm_id: "gemini-cli".to_string(),
            active_agents: 0,
            current_tasks: vec!["llm_request".to_string()],
            available_resources: std::iter::once((
                "prompt".to_string(),
                serde_json::json!(prompt)
            )).collect(),
            local_capabilities: vec!["gemini_models".to_string()],
        },
        priority: zen_swarm::Priority::Normal,
    }).await?;

    // Output response
    println!("{}", response.content);
    
    Ok(())
}
```

### **GPT CLI (Direct Connection)**
```rust
// gpt-cli/src/main.rs
use clap::{Arg, Command};
use zen_swarm::{SwarmDaemonClient, A2AMessage, AICapability};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("gpt-cli")
        .version("1.0.0")
        .about("OpenAI GPT CLI via zen-swarm")
        .arg(Arg::new("prompt")
            .help("Prompt for GPT")
            .required(true)
            .index(1))
        .arg(Arg::new("model")
            .long("model")
            .short('m')
            .help("GPT model to use")
            .default_value("gpt-4"))
        .get_matches();

    let prompt = matches.get_one::<String>("prompt").unwrap();
    let model = matches.get_one::<String>("model").unwrap();

    // Same pattern - connect to swarm daemon, route to THE COLLECTIVE
    let client = SwarmDaemonClient::new("http://localhost:9001".to_string());
    
    let response = client.execute_llm_request(A2AMessage::IntelligenceRequest {
        request_id: uuid::Uuid::new_v4().to_string(),
        capability: AICapability::LLMReasoning {
            model: model.clone(),
            context_length: 8192,
        },
        context: zen_swarm::SwarmContext {
            swarm_id: "gpt-cli".to_string(),
            active_agents: 0,
            current_tasks: vec!["llm_request".to_string()],
            available_resources: std::iter::once((
                "prompt".to_string(),
                serde_json::json!(prompt)
            )).collect(),
            local_capabilities: vec!["openai_models".to_string()],
        },
        priority: zen_swarm::Priority::Normal,
    }).await?;

    println!("{}", response.content);
    Ok(())
}
```

### **Llama CLI (Direct Connection)**
```rust
// llama-cli/src/main.rs - Same pattern
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Same architecture, different model targeting
    let client = SwarmDaemonClient::new("http://localhost:9001".to_string());
    
    let response = client.execute_llm_request(A2AMessage::IntelligenceRequest {
        capability: AICapability::LLMReasoning {
            model: "llama-3.1-8b".to_string(),
            context_length: 4096,
        },
        // ... same pattern
    }).await?;

    println!("{}", response.content);
    Ok(())
}
```

## 🔧 **SWARM DAEMON CLIENT**

### **Shared Client Library**
```rust
// zen-swarm/src/client.rs
pub struct SwarmDaemonClient {
    base_url: String,
    http_client: reqwest::Client,
}

impl SwarmDaemonClient {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            http_client: reqwest::Client::new(),
        }
    }

    /// Execute coding task via swarm daemon (swarm does the work)
    pub async fn execute_coding_task(&self, request: A2AMessage) -> Result<CodingTaskResponse, SwarmError> {
        let response = self.http_client
            .post(&format!("{}/tasks/execute", self.base_url))
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let task_response: CodingTaskResponse = response.json().await?;
            Ok(task_response)
        } else {
            Err(SwarmError::Communication(format!("HTTP error: {}", response.status())))
        }
    }

    /// Execute simple LLM query (for non-coding tasks)
    pub async fn execute_llm_query(&self, request: A2AMessage) -> Result<LLMResponse, SwarmError> {
        let response = self.http_client
            .post(&format!("{}/a2a/llm", self.base_url))
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let llm_response: LLMResponse = response.json().await?;
            Ok(llm_response)
        } else {
            Err(SwarmError::Communication(format!("HTTP error: {}", response.status())))
        }
    }

    /// Get swarm daemon status
    pub async fn get_status(&self) -> Result<DaemonStatus, SwarmError> {
        let response = self.http_client
            .get(&format!("{}/status", self.base_url))
            .send()
            .await?;

        Ok(response.json().await?)
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct CodingTaskResponse {
    pub task_result: String,          // Final result/output
    pub files_modified: Vec<String>,  // Files that were changed
    pub actions_taken: Vec<String>,   // What the swarm did
    pub model_used: String,           // Which LLM was used
    pub provider: String,             // LLM provider
    pub usage: TokenUsage,            // Token usage
    pub processing_time_ms: u64,      // Total task time
}

#[derive(Debug, serde::Deserialize)]
pub struct LLMResponse {
    pub content: String,
    pub model: String,
    pub provider: String,
    pub usage: TokenUsage,
    pub processing_time_ms: u64,
}

#[derive(Debug, serde::Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}
```

## 📦 **CLI DISTRIBUTION**

### **Separate CLI Packages**
```toml
# gemini-cli/Cargo.toml
[package]
name = "gemini-cli"
version = "1.0.0"
description = "Google Gemini CLI via zen-swarm"

[dependencies]
zen-swarm = { path = "../zen-swarm" }
clap = "4.0"
tokio = { version = "1.0", features = ["full"] }
serde_json = "1.0"
uuid = { version = "1.0", features = ["v4"] }

[[bin]]
name = "gemini"
path = "src/main.rs"
```

```toml
# gpt-cli/Cargo.toml
[package]
name = "gpt-cli"
version = "1.0.0"
description = "OpenAI GPT CLI via zen-swarm"

[dependencies]
zen-swarm = { path = "../zen-swarm" }
# ... same deps

[[bin]]
name = "gpt"
path = "src/main.rs"
```

## 🚀 **USAGE EXAMPLES**

### **Using Different LLM CLIs**
```bash
# Start swarm daemon for repository
zen-swarm-daemon start --port 9001 --collective ws://collective.company.com:8080

# Use Gemini directly
gemini "Analyze this Rust code for performance issues" --model gemini-1.5-pro

# Use GPT-4 directly  
gpt "Generate unit tests for this function" --model gpt-4

# Use Llama locally
llama "Explain this algorithm" --model llama-3.1-8b

# Use Claude Code (existing)
claude "Review this code architecture"
```

### **All CLIs Share Same Infrastructure**
```bash
# All these commands route through the same swarm daemon
gemini "What is Rust borrowing?" 
gpt "What is Rust borrowing?"
llama "What is Rust borrowing?"
claude "What is Rust borrowing?"

# Same repository context, same intelligence, different LLMs
```

## 🎯 **BENEFITS OF DIRECT CLI APPROACH**

### **✅ Consistency**
- All LLM CLIs use identical architecture
- Same repository context and intelligence
- Unified error handling and logging

### **✅ Simplicity**
- No plugin system complexity
- Direct connection to swarm daemon
- Standard CLI argument patterns

### **✅ Performance**
- No plugin loading overhead
- Direct HTTP/A2A communication
- Efficient request routing

### **✅ Maintainability**
- Each CLI is independent
- Shared client library reduces duplication
- Easy to add new LLM CLIs

## 🔧 **SWARM DAEMON HTTP ENDPOINTS**

### **LLM Request Endpoint**
```rust
// zen-swarm/src/daemon.rs - Add HTTP routes
async fn setup_http_routes(&self) -> Router {
    Router::new()
        .route("/a2a/llm", post(Self::handle_llm_request))
        .route("/status", get(Self::handle_status))
        .route("/health", get(Self::handle_health))
        .with_state(Arc::new(self.clone()))
}

async fn handle_llm_request(
    State(daemon): State<Arc<SwarmDaemon>>,
    Json(request): Json<A2AMessage>,
) -> Result<Json<LLMResponse>, StatusCode> {
    match daemon.a2a_coordinator.send_message("the-collective", request).await {
        Ok(response) => {
            // Convert A2A response to LLM response format
            let llm_response = LLMResponse::from_a2a_response(response);
            Ok(Json(llm_response))
        }
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}
```

Perfect! Now we have:
- **claude-cli** (existing) ←→ zen-swarm 
- **gemini-cli** (new) ←→ zen-swarm
- **gpt-cli** (new) ←→ zen-swarm  
- **llama-cli** (new) ←→ zen-swarm

All sharing the same swarm intelligence infrastructure! 🎯