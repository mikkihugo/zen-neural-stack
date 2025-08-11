/**
 * Zen Neural Unified Storage System
 * 
 * Multi-model storage using SurrealDB for all neural network needs:
 * - Graph data (GNN)
 * - Training metadata
 * - Model weights and checkpoints
 * - Performance metrics
 * - Configuration data
 * - Distributed coordination
 */

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[cfg(feature = "zen-storage")]
use surrealdb::{Surreal, engine::any::Any, sql::Thing};

pub mod cache;
pub mod distributed;
pub mod migrations;
pub mod queries;

/// Unified storage manager for all zen-neural data
pub struct ZenUnifiedStorage {
    #[cfg(feature = "zen-storage")]
    db: Surreal<Any>,
    
    cache: cache::ZenStorageCache,
    distributed: bool,
    node_id: Option<String>,
}

impl ZenUnifiedStorage {
    /// Create new unified storage instance
    #[cfg(feature = "zen-storage")]
    pub async fn new(connection_string: &str, distributed: bool) -> Result<Self, StorageError> {
        let db = if distributed {
            // Connect to distributed SurrealDB cluster
            Surreal::new::<surrealdb::engine::any::Any>(connection_string).await?
        } else {
            // Local embedded storage
            Surreal::new::<surrealdb::engine::local::RocksDb>(connection_string).await?
        };
        
        db.use_ns("zen_neural").use_db("main").await?;
        
        let storage = Self {
            db,
            cache: cache::ZenStorageCache::new(),
            distributed,
            node_id: if distributed { Some(uuid::Uuid::new_v4().to_string()) } else { None },
        };
        
        storage.setup_schema().await?;
        
        Ok(storage)
    }
    
    /// Setup database schema for zen-neural
    #[cfg(feature = "zen-storage")]
    async fn setup_schema(&self) -> Result<(), StorageError> {
        self.db.query("
            -- Graph data for GNN
            DEFINE TABLE graph_nodes SCHEMALESS;
            DEFINE TABLE graph_edges SCHEMALESS;
            DEFINE INDEX graph_traversal ON graph_nodes FIELDS in, out;
            DEFINE INDEX edge_lookup ON graph_edges FIELDS from, to;
            
            -- Neural network models
            DEFINE TABLE neural_models SCHEMALESS;
            DEFINE TABLE model_weights SCHEMALESS;
            DEFINE TABLE model_checkpoints SCHEMALESS;
            DEFINE INDEX model_lookup ON neural_models FIELDS model_id, version;
            
            -- Training data and metadata
            DEFINE TABLE training_runs SCHEMALESS;
            DEFINE TABLE training_metrics SCHEMALESS;
            DEFINE TABLE validation_results SCHEMALESS;
            DEFINE INDEX training_lookup ON training_runs FIELDS model_id, timestamp;
            
            -- Performance and monitoring
            DEFINE TABLE performance_metrics SCHEMALESS;
            DEFINE TABLE system_health SCHEMALESS;
            DEFINE TABLE benchmark_results SCHEMALESS;
            
            -- Configuration and settings
            DEFINE TABLE configurations SCHEMALESS;
            DEFINE TABLE hyperparameters SCHEMALESS;
            
            -- Distributed coordination (if distributed)
            DEFINE TABLE distributed_nodes SCHEMALESS;
            DEFINE TABLE consensus_log SCHEMALESS;
            DEFINE TABLE shard_assignments SCHEMALESS;
            
            -- Caching and optimization
            DEFINE TABLE query_cache SCHEMALESS;
            DEFINE TABLE embedding_cache SCHEMALESS;
        ").await?;
        
        Ok(())
    }
    
    // === GRAPH OPERATIONS ===
    
    /// Store GNN graph data
    #[cfg(feature = "zen-storage")]
    pub async fn store_graph(&self, graph_id: &str, nodes: Vec<GNNNode>, edges: Vec<GNNEdge>) -> Result<(), StorageError> {
        let _result = self.db.query("
            BEGIN TRANSACTION;
            
            FOR $node IN $nodes {
                CREATE graph_nodes CONTENT {
                    id: $node.id,
                    graph_id: $graph_id,
                    features: $node.features,
                    node_type: $node.node_type,
                    metadata: $node.metadata,
                    created_at: time::now()
                };
            };
            
            FOR $edge IN $edges {
                CREATE graph_edges CONTENT {
                    id: $edge.id,
                    graph_id: $graph_id,
                    from: $edge.from,
                    to: $edge.to,
                    weight: $edge.weight,
                    edge_type: $edge.edge_type,
                    created_at: time::now()
                };
            };
            
            COMMIT TRANSACTION;
        ")
        .bind(("graph_id", graph_id))
        .bind(("nodes", nodes))
        .bind(("edges", edges))
        .await?;
        
        Ok(())
    }
    
    /// Load GNN subgraph for message passing
    #[cfg(feature = "zen-storage")]
    pub async fn load_subgraph(&self, center_node: &str, depth: u32) -> Result<Graph, StorageError> {
        let result = self.db.query("
            SELECT * FROM graph_nodes WHERE id INSIDE (
                SELECT ->graph_edges->to FROM graph_nodes WHERE id = $center
                UNION
                SELECT <-graph_edges<-from FROM graph_nodes WHERE id = $center
                -- Recursive traversal up to depth
            )
        ")
        .bind(("center", center_node))
        .bind(("depth", depth))
        .await?;
        
        let graph = self.parse_graph_result(result)?;
        
        // Cache for future use
        self.cache.store_subgraph(center_node, depth, &graph).await;
        
        Ok(graph)
    }
    
    // === NEURAL MODEL OPERATIONS ===
    
    /// Store trained neural model
    #[cfg(feature = "zen-storage")]
    pub async fn store_model(&self, model: &ZenNeuralModel) -> Result<String, StorageError> {
        let model_id = uuid::Uuid::new_v4().to_string();
        
        let _result = self.db.query("
            BEGIN TRANSACTION;
            
            CREATE neural_models CONTENT {
                model_id: $model_id,
                architecture: $architecture,
                hyperparameters: $hyperparams,
                training_config: $config,
                performance_metrics: $metrics,
                created_at: time::now(),
                status: 'active'
            };
            
            CREATE model_weights CONTENT {
                model_id: $model_id,
                weights: $weights,
                biases: $biases,
                layer_config: $layers,
                version: 1,
                created_at: time::now()
            };
            
            COMMIT TRANSACTION;
        ")
        .bind(("model_id", &model_id))
        .bind(("architecture", model.architecture.clone()))
        .bind(("hyperparams", model.hyperparameters.clone()))
        .bind(("config", model.config.clone()))
        .bind(("metrics", model.metrics.clone()))
        .bind(("weights", model.weights.clone()))
        .bind(("biases", model.biases.clone()))
        .bind(("layers", model.layers.clone()))
        .await?;
        
        Ok(model_id)
    }
    
    /// Load neural model by ID
    #[cfg(feature = "zen-storage")]
    pub async fn load_model(&self, model_id: &str) -> Result<ZenNeuralModel, StorageError> {
        // Check cache first
        if let Some(model) = self.cache.get_model(model_id).await {
            return Ok(model);
        }
        
        let result = self.db.query("
            SELECT * FROM neural_models, model_weights 
            WHERE neural_models.model_id = $model_id 
            AND model_weights.model_id = $model_id
        ")
        .bind(("model_id", model_id))
        .await?;
        
        let model = self.parse_model_result(result)?;
        
        // Cache for future use
        self.cache.store_model(model_id, &model).await;
        
        Ok(model)
    }
    
    // === TRAINING OPERATIONS ===
    
    /// Store training run metadata and results
    #[cfg(feature = "zen-storage")]
    pub async fn store_training_run(
        &self,
        run: &TrainingRun,
        metrics: Vec<TrainingMetric>,
    ) -> Result<String, StorageError> {
        let run_id = uuid::Uuid::new_v4().to_string();
        
        let _result = self.db.query("
            BEGIN TRANSACTION;
            
            CREATE training_runs CONTENT {
                run_id: $run_id,
                model_id: $model_id,
                dataset_info: $dataset,
                hyperparameters: $hyperparams,
                start_time: $start_time,
                end_time: $end_time,
                final_loss: $final_loss,
                epochs_completed: $epochs,
                status: $status,
                node_id: $node_id
            };
            
            FOR $metric IN $metrics {
                CREATE training_metrics CONTENT {
                    run_id: $run_id,
                    epoch: $metric.epoch,
                    metric_name: $metric.name,
                    value: $metric.value,
                    timestamp: $metric.timestamp
                };
            };
            
            COMMIT TRANSACTION;
        ")
        .bind(("run_id", &run_id))
        .bind(("model_id", &run.model_id))
        .bind(("dataset", &run.dataset_info))
        .bind(("hyperparams", &run.hyperparameters))
        .bind(("start_time", run.start_time))
        .bind(("end_time", run.end_time))
        .bind(("final_loss", run.final_loss))
        .bind(("epochs", run.epochs_completed))
        .bind(("status", &run.status))
        .bind(("node_id", &self.node_id))
        .bind(("metrics", metrics))
        .await?;
        
        Ok(run_id)
    }
    
    // === DISTRIBUTED OPERATIONS ===
    
    /// Register this node in distributed network
    #[cfg(feature = "zen-storage")]
    pub async fn register_distributed_node(&self, node_info: &DistributedNodeInfo) -> Result<(), StorageError> {
        if !self.distributed {
            return Err(StorageError::NotDistributed);
        }
        
        let _result = self.db.query("
            UPSERT distributed_nodes CONTENT {
                node_id: $node_id,
                node_type: $node_type,
                capabilities: $capabilities,
                endpoint: $endpoint,
                status: 'active',
                last_heartbeat: time::now(),
                system_info: $system_info
            }
        ")
        .bind(("node_id", &node_info.node_id))
        .bind(("node_type", &node_info.node_type))
        .bind(("capabilities", &node_info.capabilities))
        .bind(("endpoint", &node_info.endpoint))
        .bind(("system_info", &node_info.system_info))
        .await?;
        
        Ok(())
    }
    
    /// Get active nodes in distributed network
    #[cfg(feature = "zen-storage")]
    pub async fn get_active_nodes(&self) -> Result<Vec<DistributedNodeInfo>, StorageError> {
        let result = self.db.query("
            SELECT * FROM distributed_nodes 
            WHERE status = 'active' 
            AND last_heartbeat > time::now() - 60s
        ").await?;
        
        let nodes = self.parse_nodes_result(result)?;
        Ok(nodes)
    }
}

// === DATA STRUCTURES ===

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GNNNode {
    pub id: String,
    pub features: Vec<f32>,
    pub node_type: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GNNEdge {
    pub id: String,
    pub from: String,
    pub to: String,
    pub weight: f32,
    pub edge_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<GNNNode>,
    pub edges: Vec<GNNEdge>,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZenNeuralModel {
    pub architecture: String,
    pub hyperparameters: HashMap<String, serde_json::Value>,
    pub config: HashMap<String, serde_json::Value>,
    pub metrics: HashMap<String, f32>,
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,
    pub layers: Vec<LayerConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    pub layer_type: String,
    pub input_size: usize,
    pub output_size: usize,
    pub activation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRun {
    pub model_id: String,
    pub dataset_info: HashMap<String, serde_json::Value>,
    pub hyperparameters: HashMap<String, serde_json::Value>,
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub final_loss: Option<f32>,
    pub epochs_completed: u32,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetric {
    pub epoch: u32,
    pub name: String,
    pub value: f32,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedNodeInfo {
    pub node_id: String,
    pub node_type: String,
    pub capabilities: Vec<String>,
    pub endpoint: String,
    pub system_info: HashMap<String, serde_json::Value>,
}

/// Storage operation errors
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("Database error: {0}")]
    #[cfg(feature = "zen-storage")]
    DatabaseError(#[from] surrealdb::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Not configured for distributed operation")]
    NotDistributed,
    
    #[error("Cache miss: {0}")]
    CacheMiss(String),
    
    #[error("Invalid query: {0}")]
    InvalidQuery(String),
}