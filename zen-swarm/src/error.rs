//! Error types for zen-swarm

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SwarmError {
  #[error("Configuration error: {0}")]
  Config(String),

  #[error("Agent error: {0}")]
  Agent(String),

  #[error("Task execution error: {0}")]
  Task(String),

  #[error("Communication error: {0}")]
  Communication(String),

  #[error("Network error: {0}")]
  Network(String),

  #[error("Storage error: {0}")]
  Storage(String),

  #[error("Internal error: {0}")]
  Internal(String),

  #[cfg(feature = "neural")]
  #[error("Neural network error: {0}")]
  Neural(String),

  #[cfg(feature = "vector")]
  #[error("Vector database error: {0}")]
  Vector(String),

  #[cfg(feature = "runtime")]
  #[error("Runtime error: {0}")]
  Runtime(String),

  #[cfg(feature = "graph")]
  #[error("Graph database error: {0}")]
  Graph(String),

  #[cfg(feature = "persistence")]
  #[error("Persistence error: {0}")]
  Persistence(String),

  #[cfg(feature = "neural")]
  #[error("Language learning error in {language}: {message}")]
  LanguageLearning { language: String, message: String },

  #[error("Code quality error: {0}")]
  CodeQuality(String),

  #[error("Configuration error: {0}")]
  Configuration(String),
}

/// Error conversions for external crates
#[cfg(feature = "neural")]
impl From<candle_core::Error> for SwarmError {
  /// Convert a Candle neural network error into SwarmError
  fn from(error: candle_core::Error) -> Self {
    SwarmError::Neural(format!("Candle error: {}", error))
  }
}

impl SwarmError {
  /// Determine if this error type is recoverable through retry or other means
  ///
  /// # Returns
  ///
  /// - `true` for transient errors like network issues that may resolve
  /// - `false` for permanent errors like configuration problems
  pub fn is_recoverable(&self) -> bool {
    match self {
      SwarmError::Communication(_) | SwarmError::Network(_) => true,
      SwarmError::Config(_) | SwarmError::Internal(_) => false,
      _ => true,
    }
  }
}
