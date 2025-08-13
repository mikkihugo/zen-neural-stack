//! Zen-Swarm Provider for code-mesh
//!
//! This provider integrates zen-swarm coordination with code-mesh's LLM system.
//! It routes requests through zen-swarm daemons to THE COLLECTIVE for multi-LLM support
//! while providing repository-scoped intelligence and coordination.

use async_trait::async_trait;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;
use chrono::Utc;

use code_mesh_core::llm::{
    Provider, Model, ProviderConfig, ProviderHealth, RateLimitInfo, ModelInfo,
    GenerateOptions, GenerateResult, Message, MessageRole, MessageContent,
    Usage, FinishReason
};

/// Zen-swarm provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZenSwarmProviderConfig {
    pub swarm_daemon_endpoint: String,
    pub collective_endpoint: Option<String>,
    pub repo_path: String,
    pub supported_models: Vec<String>,
}

impl Default for ZenSwarmProviderConfig {
    fn default() -> Self {
        Self {
            swarm_daemon_endpoint: "http://localhost:9001".to_string(),
            collective_endpoint: Some("ws://localhost:8080".to_string()),
            repo_path: std::env::current_dir()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            supported_models: vec![
                // Multi-LLM support through THE COLLECTIVE
                "claude-3-sonnet".to_string(), "claude-3-opus".to_string(), "claude-3-haiku".to_string(),
                "gemini-1.5-pro".to_string(), "gemini-1.5-flash".to_string(), "gemini-1.0-pro".to_string(),
                "gpt-4".to_string(), "gpt-4-turbo".to_string(), "gpt-3.5-turbo".to_string(),
                "llama-3.1-8b".to_string(), "llama-3.1-70b".to_string(), "codellama".to_string(),
            ],
        }
    }
}

/// Zen-swarm provider implementation
pub struct ZenSwarmProvider {
    config: ZenSwarmProviderConfig,
    client: reqwest::Client,
    models: HashMap<String, Arc<ZenSwarmModel>>,
}

impl ZenSwarmProvider {
    pub fn new(config: ZenSwarmProviderConfig) -> Self {
        let client = reqwest::Client::new();
        let mut models = HashMap::new();
        
        // Create model instances for all supported models
        for model_id in &config.supported_models {
            let model = Arc::new(ZenSwarmModel::new(
                model_id.clone(),
                config.swarm_daemon_endpoint.clone(),
                client.clone(),
            ));
            models.insert(model_id.clone(), model);
        }
        
        Self {
            config,
            client,
            models,
        }
    }
}

#[async_trait]
impl Provider for ZenSwarmProvider {
    fn id(&self) -> &str {
        "zen-swarm"
    }
    
    fn name(&self) -> &str {
        "Zen-Swarm Distributed AI Coordination"
    }
    
    fn base_url(&self) -> &str {
        &self.config.swarm_daemon_endpoint
    }
    
    fn api_version(&self) -> &str {
        "1.0"
    }
    
    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        tracing::info!("🐝 Listing models from zen-swarm daemon...");
        
        // Try to get dynamic model list from swarm daemon
        let models_response = self.client
            .get(&format!("{}/models/available", self.config.swarm_daemon_endpoint))
            .send()
            .await;
            
        let model_ids = if let Ok(response) = models_response {
            if let Ok(data) = response.json::<serde_json::Value>().await {
                data.get("models")
                    .and_then(|m| m.as_array())
                    .map(|arr| arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect())
                    .unwrap_or_else(|| self.config.supported_models.clone())
            } else {
                self.config.supported_models.clone()
            }
        } else {
            self.config.supported_models.clone()
        };
        
        let models = model_ids.into_iter().map(|id| {
            ModelInfo {
                id: id.clone(),
                name: format!("{} via zen-swarm", id),
                provider_id: "zen-swarm".to_string(),
                context_length: self.get_context_length(&id),
                supports_streaming: true,
                supports_tools: true,
                supports_vision: self.supports_vision(&id),
            }
        }).collect();
        
        Ok(models)
    }
    
    async fn get_model(&self, model_id: &str) -> Result<Arc<dyn Model>> {
        self.models
            .get(model_id)
            .cloned()
            .map(|m| m as Arc<dyn Model>)
            .ok_or_else(|| anyhow!("Model not found: {}", model_id))
    }
    
    async fn health_check(&self) -> Result<ProviderHealth> {
        tracing::info!("🏥 Checking zen-swarm daemon health...");
        
        let response = self.client
            .get(&format!("{}/health", self.config.swarm_daemon_endpoint))
            .send()
            .await?;
            
        let is_healthy = response.status().is_success();
        let status_code = response.status().as_u16();
        
        if is_healthy {
            // Also check THE COLLECTIVE connection if configured
            let collective_healthy = if self.config.collective_endpoint.is_some() {
                let status_response = self.client
                    .get(&format!("{}/daemon_status", self.config.swarm_daemon_endpoint))
                    .send()
                    .await;
                    
                if let Ok(resp) = status_response {
                    if let Ok(data) = resp.json::<serde_json::Value>().await {
                        data.get("collective_status")
                            .and_then(|cs| cs.get("connected"))
                            .and_then(|c| c.as_bool())
                            .unwrap_or(false)
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                true
            };
            
            Ok(ProviderHealth {
                is_healthy: collective_healthy,
                message: if collective_healthy {
                    "zen-swarm daemon healthy, connected to THE COLLECTIVE".to_string()
                } else {
                    "zen-swarm daemon healthy, THE COLLECTIVE disconnected".to_string()
                },
                status_code: Some(status_code),
                response_time_ms: None,
            })
        } else {
            Ok(ProviderHealth {
                is_healthy: false,
                message: format!("zen-swarm daemon unhealthy: HTTP {}", status_code),
                status_code: Some(status_code),
                response_time_ms: None,
            })
        }
    }
    
    fn get_config(&self) -> &ProviderConfig {
        // Convert our config to the base ProviderConfig
        // This is a simplified mapping - in practice you'd want a proper conversion
        &ProviderConfig {
            api_key: None,
            base_url: Some(self.config.swarm_daemon_endpoint.clone()),
            organization: None,
            project: None,
            custom_headers: None,
        }
    }
    
    async fn update_config(&mut self, _config: ProviderConfig) -> Result<()> {
        // For now, zen-swarm config is immutable after creation
        // In practice, you might want to support dynamic reconfiguration
        Ok(())
    }
    
    async fn get_rate_limits(&self) -> Result<RateLimitInfo> {
        // zen-swarm coordinates rate limiting through THE COLLECTIVE
        Ok(RateLimitInfo {
            requests_per_minute: Some(1000), // High limit due to intelligent routing
            tokens_per_minute: None,
            requests_remaining: None,
            tokens_remaining: None,
            reset_time: None,
        })
    }
}

impl ZenSwarmProvider {
    fn get_context_length(&self, model_id: &str) -> u32 {
        match model_id {
            id if id.starts_with("claude-3") => 200000,
            id if id.starts_with("gemini-1.5") => 128000,
            id if id.starts_with("gpt-4") => 8192,
            id if id.starts_with("llama-3.1") => 4096,
            _ => 4096,
        }
    }
    
    fn supports_vision(&self, model_id: &str) -> bool {
        matches!(model_id, 
            id if id.starts_with("claude-3") ||
                  id.starts_with("gemini-1.5") ||
                  id.starts_with("gpt-4")
        )
    }
}

/// Individual model implementation that routes through zen-swarm
pub struct ZenSwarmModel {
    id: String,
    swarm_endpoint: String,
    client: reqwest::Client,
}

impl ZenSwarmModel {
    pub fn new(id: String, swarm_endpoint: String, client: reqwest::Client) -> Self {
        Self {
            id,
            swarm_endpoint,
            client,
        }
    }
}

#[async_trait]
impl Model for ZenSwarmModel {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn provider_id(&self) -> &str {
        "zen-swarm"
    }
    
    async fn generate(&self, messages: &[Message], options: &GenerateOptions) -> Result<GenerateResult> {
        tracing::info!("🧠 Generating with {} via zen-swarm...", self.id);
        
        // Convert messages to prompt string
        let prompt = messages.iter()
            .map(|msg| match msg.role {
                MessageRole::System => format!("System: {}", self.extract_text_content(&msg.content)),
                MessageRole::User => format!("User: {}", self.extract_text_content(&msg.content)),
                MessageRole::Assistant => format!("Assistant: {}", self.extract_text_content(&msg.content)),
            })
            .collect::<Vec<_>>()
            .join("\n");
        
        // Create zen-swarm task request
        let request_payload = serde_json::json!({
            "task_description": prompt,
            "preferred_llm": self.id,
            "context": {
                "working_directory": std::env::current_dir()?.to_string_lossy(),
                "relevant_files": [],
                "temperature": options.temperature,
                "max_tokens": options.max_tokens,
                "top_p": options.top_p,
            }
        });
        
        let start_time = std::time::Instant::now();
        
        // Send request to zen-swarm daemon
        let response = self.client
            .post(&format!("{}/tasks/execute", self.swarm_endpoint))
            .header("Content-Type", "application/json")
            .header("X-Request-ID", Uuid::new_v4().to_string())
            .json(&request_payload)
            .send()
            .await?;
            
        let processing_time = start_time.elapsed();
        
        if !response.status().is_success() {
            return Err(anyhow!("zen-swarm execution failed: HTTP {}", response.status()));
        }
        
        let result: serde_json::Value = response.json().await?;
        
        // Log interaction
        self.log_interaction(&prompt, &result, processing_time.as_millis() as u64).await?;
        
        // Convert result to GenerateResult
        let content = result.get("task_result")
            .and_then(|r| r.as_str())
            .unwrap_or("")
            .to_string();
            
        let usage = result.get("usage")
            .and_then(|u| serde_json::from_value::<Usage>(u.clone()).ok())
            .unwrap_or_default();
            
        Ok(GenerateResult {
            content: MessageContent::Text(content),
            usage: Some(usage),
            finish_reason: Some(FinishReason::Stop),
            model_id: self.id.clone(),
            provider_id: "zen-swarm".to_string(),
        })
    }
    
    async fn generate_stream(
        &self,
        messages: &[Message],
        options: &GenerateOptions,
    ) -> Result<Box<dyn futures::Stream<Item = Result<StreamChunk>> + Unpin + Send>> {
        // For now, use non-streaming and return as single chunk
        // In practice, you'd implement actual streaming through zen-swarm
        let result = self.generate(messages, options).await?;
        
        let chunk = StreamChunk {
            content: result.content,
            usage: result.usage,
            finish_reason: result.finish_reason,
            model_id: result.model_id,
            provider_id: result.provider_id,
        };
        
        let stream = futures::stream::once(async { Ok(chunk) });
        Ok(Box::new(Box::pin(stream)))
    }
}

impl ZenSwarmModel {
    fn extract_text_content(&self, content: &MessageContent) -> &str {
        match content {
            MessageContent::Text(text) => text,
            MessageContent::MultiModal(parts) => {
                // For simplicity, just extract the first text part
                parts.iter()
                    .find_map(|part| match part {
                        MessageContentPart::Text(text) => Some(text.as_str()),
                        _ => None,
                    })
                    .unwrap_or("")
            }
        }
    }
    
    async fn log_interaction(&self, prompt: &str, result: &serde_json::Value, processing_time_ms: u64) -> Result<()> {
        let log_payload = serde_json::json!({
            "timestamp": Utc::now().to_rfc3339(),
            "session_id": Uuid::new_v4().to_string(),
            "cli_tool": "code-mesh",
            "interaction_type": "CodeGeneration",
            "prompt": prompt,
            "response": result.get("task_result").and_then(|r| r.as_str()),
            "status": "Completed",
            "processing_time_ms": processing_time_ms,
            "token_usage": result.get("usage"),
        });
        
        // Send to swarm daemon for monitoring
        let _ = self.client
            .post(&format!("{}/cli/log-interaction", self.swarm_endpoint))
            .json(&log_payload)
            .send()
            .await; // Ignore errors for logging
            
        Ok(())
    }
}