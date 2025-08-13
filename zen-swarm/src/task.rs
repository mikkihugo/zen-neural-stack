//! Task definition and execution

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Task status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
  Pending,
  Running,
  Completed,
  Failed(String),
  Cancelled,
}

/// Task definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
  pub id: Uuid,
  pub description: String,
  pub assigned_agents: Vec<String>,
  pub status: TaskStatus,
  pub created_at: chrono::DateTime<chrono::Utc>,
  pub metadata: Value,
}

/// Task execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
  pub task_id: Uuid,
  pub status: TaskStatus,
  pub result: Value,
  pub agents_used: Vec<String>,
  pub execution_time_ms: u64,
}

impl Task {
  /// Create a new task
  pub fn new(description: &str, assigned_agents: Vec<String>) -> Self {
    Self {
      id: Uuid::new_v4(),
      description: description.to_string(),
      assigned_agents,
      status: TaskStatus::Pending,
      created_at: chrono::Utc::now(),
      metadata: Value::Null,
    }
  }

  /// Create task with metadata
  pub fn with_metadata(
    description: &str,
    assigned_agents: Vec<String>,
    metadata: Value,
  ) -> Self {
    let mut task = Self::new(description, assigned_agents);
    task.metadata = metadata;
    task
  }
}
