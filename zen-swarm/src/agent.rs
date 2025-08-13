//! Agent definitions and management

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Agent types with specific capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentType {
  Researcher,
  Analyst,
  Coder,
  Coordinator,
  Optimizer,
}

/// Agent status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStatus {
  Idle,
  Working,
  Blocked,
  Error(String),
}

/// Agent definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
  pub id: Uuid,
  pub name: String,
  pub agent_type: AgentType,
  pub status: AgentStatus,
  pub capabilities: Vec<String>,
  pub metadata: HashMap<String, String>,
}

impl Agent {
  /// Create a new agent
  pub fn new(name: &str, agent_type: AgentType) -> Self {
    Self {
      id: Uuid::new_v4(),
      name: name.to_string(),
      agent_type,
      status: AgentStatus::Idle,
      capabilities: Vec::new(),
      metadata: HashMap::new(),
    }
  }

  /// Create a researcher agent
  pub fn researcher() -> Self {
    let mut agent = Self::new("researcher", AgentType::Researcher);
    agent.capabilities = vec![
      "web_search".to_string(),
      "document_analysis".to_string(),
      "data_gathering".to_string(),
    ];
    agent
  }

  /// Create an analyst agent
  pub fn analyst() -> Self {
    let mut agent = Self::new("analyst", AgentType::Analyst);
    agent.capabilities = vec![
      "data_analysis".to_string(),
      "pattern_recognition".to_string(),
      "statistical_analysis".to_string(),
    ];
    agent
  }

  /// Create a coordinator agent
  pub fn coordinator() -> Self {
    let mut agent = Self::new("coordinator", AgentType::Coordinator);
    agent.capabilities = vec![
      "task_coordination".to_string(),
      "resource_management".to_string(),
      "decision_making".to_string(),
    ];
    agent
  }

  /// Create a coder agent
  pub fn coder() -> Self {
    let mut agent = Self::new("coder", AgentType::Coder);
    agent.capabilities = vec![
      "code_generation".to_string(),
      "debugging".to_string(),
      "testing".to_string(),
    ];
    agent
  }

  /// Create an optimizer agent
  pub fn optimizer() -> Self {
    let mut agent = Self::new("optimizer", AgentType::Optimizer);
    agent.capabilities = vec![
      "performance_optimization".to_string(),
      "resource_optimization".to_string(),
      "algorithm_tuning".to_string(),
    ];
    agent
  }
}
