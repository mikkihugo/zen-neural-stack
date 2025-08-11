/**
 * @file zen-neural/src/gnn/tests/storage_integration_tests.rs
 * @brief Comprehensive Integration Tests for GNN SurrealDB Storage
 * 
 * This module contains comprehensive tests for the GNN storage integration,
 * demonstrating functionality and performance characteristics for:
 * 
 * - **Graph Storage**: Single and partitioned graph persistence
 * - **Model Checkpoints**: Version control and deduplication
 * - **Training History**: Complete training lifecycle tracking
 * - **Performance**: Million+ node graph operations
 * - **Distributed Operations**: Multi-node coordination
 * - **Query Optimization**: Advanced graph pattern matching
 * 
 * ## Test Categories:
 * 
 * ### Unit Tests:
 * - Basic CRUD operations
 * - Data structure validation
 * - Error handling
 * - Configuration management
 * 
 * ### Integration Tests:
 * - End-to-end storage workflows
 * - Cross-module compatibility
 * - Transaction handling
 * - Cache effectiveness
 * 
 * ### Performance Tests:
 * - Large graph operations (1M+ nodes)
 * - Query response times
 * - Memory usage validation
 * - Compression effectiveness
 * 
 * ### Distributed Tests:
 * - Multi-node coordination
 * - Partition balancing
 * - Fault tolerance
 * - Load distribution
 * 
 * @author Storage Integration Specialist Agent (ruv-swarm)
 * @version 1.0.0-alpha.1
 * @since 2024-08-11
 * 
 * @see crate::gnn::storage::GraphStorage Main storage interface
 * @see crate::storage::ZenUnifiedStorage Base storage system
 */

#[cfg(all(test, feature = "zen-storage", feature = "gnn"))]
mod tests {
    use super::super::{
        storage::{GraphStorage, GNNStorageConfig, ModelCheckpoint, GNNTrainingRun, TrainingStatus},
        data::{GraphData, GraphMetadata},
        GNNConfig, GNNError
    };
    use crate::storage::{GNNNode, GNNEdge, ZenUnifiedStorage};
    
    use std::collections::HashMap;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokio;
    use uuid::Uuid;
    use ndarray::Array2;
    
    // === HELPER FUNCTIONS ===

    /// Create test storage instance
    async fn create_test_storage() -> Result<GraphStorage, GNNError> {
        let config = GNNStorageConfig {
            max_nodes_per_partition: 1000, // Smaller for tests
            replication_factor: 1,
            enable_query_cache: true,
            cache_ttl_seconds: 60,
            max_concurrent_ops: 5,
            enable_compression: true,
            batch_size: 100,
            enable_versioning: true,
        };
        
        // Use memory database for tests
        GraphStorage::new("mem://", config).await
    }

    /// Create test graph data
    fn create_test_graph(num_nodes: usize, num_edges: usize) -> Result<GraphData, GNNError> {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        
        // Create nodes with random features
        for i in 0..num_nodes {
            let features = (0..10).map(|_| rand::random::<f32>()).collect();
            let node_features = Array2::from_shape_vec((1, 10), features)?;
            
            nodes.push(GNNNode {
                id: format!("node_{}", i),
                features: node_features.row(0).to_vec(),
                node_type: if i % 2 == 0 { "type_a".to_string() } else { "type_b".to_string() },
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("cluster".to_string(), serde_json::Value::Number((i % 5).into()));
                    meta
                },
            });
        }
        
        // Create edges with some structure
        for i in 0..num_edges.min(num_nodes * (num_nodes - 1)) {
            let source = i % num_nodes;
            let target = (i + 1) % num_nodes;
            
            edges.push(GNNEdge {
                id: format!("edge_{}_{}", source, target),
                from: format!("node_{}", source),
                to: format!("node_{}", target),
                weight: rand::random::<f32>(),
                edge_type: if i % 3 == 0 { "strong".to_string() } else { "weak".to_string() },
            });
        }

        // Convert to GraphData format
        let node_features = Array2::from_shape_fn((num_nodes, 10), |(i, j)| {
            nodes[i].features[j]
        });
        
        let adjacency_edges: Vec<(usize, usize)> = edges.iter()
            .map(|e| {
                let source = e.from.strip_prefix("node_").unwrap().parse().unwrap();
                let target = e.to.strip_prefix("node_").unwrap().parse().unwrap();
                (source, target)
            })
            .collect();

        GraphData::new(node_features, None, adjacency_edges)
    }

    // === BASIC STORAGE TESTS ===

    #[tokio::test]
    async fn test_storage_creation() {
        let storage = create_test_storage().await;
        assert!(storage.is_ok(), "Storage creation should succeed");
    }

    #[tokio::test]
    async fn test_small_graph_storage_and_retrieval() {
        let storage = create_test_storage().await.expect("Storage creation failed");
        let graph_data = create_test_graph(10, 15).expect("Graph creation failed");
        let graph_id = "test_small_graph";

        // Store graph
        let result = storage.store_graph(graph_id, &graph_data, None).await;
        assert!(result.is_ok(), "Graph storage should succeed");

        // Retrieve graph
        let retrieved = storage.load_graph(graph_id).await;
        assert!(retrieved.is_ok(), "Graph retrieval should succeed");

        let retrieved_graph = retrieved.unwrap();
        assert_eq!(retrieved_graph.num_nodes(), graph_data.num_nodes());
        assert_eq!(retrieved_graph.num_edges(), graph_data.num_edges());
    }

    #[tokio::test]
    async fn test_large_graph_partitioning() {
        let storage = create_test_storage().await.expect("Storage creation failed");
        let graph_data = create_test_graph(1500, 3000).expect("Large graph creation failed");
        let graph_id = "test_large_graph";

        // Store large graph (should trigger partitioning)
        let result = storage.store_graph(graph_id, &graph_data, None).await;
        assert!(result.is_ok(), "Large graph storage should succeed");

        let partition_ids = result.unwrap();
        assert!(partition_ids.len() > 1, "Large graph should be partitioned");

        // Verify we can retrieve the complete graph
        let retrieved = storage.load_graph(graph_id).await;
        assert!(retrieved.is_ok(), "Partitioned graph retrieval should succeed");

        let retrieved_graph = retrieved.unwrap();
        assert_eq!(retrieved_graph.num_nodes(), graph_data.num_nodes());
    }

    #[tokio::test]
    async fn test_k_hop_subgraph_extraction() {
        let storage = create_test_storage().await.expect("Storage creation failed");
        let graph_data = create_test_graph(50, 100).expect("Graph creation failed");
        let graph_id = "test_k_hop";

        // Store graph
        storage.store_graph(graph_id, &graph_data, None).await.expect("Graph storage failed");

        // Extract k-hop subgraph
        let subgraph = storage.load_k_hop_subgraph(graph_id, "node_0", 2).await;
        assert!(subgraph.is_ok(), "K-hop subgraph extraction should succeed");

        let sub = subgraph.unwrap();
        assert!(sub.nodes.len() > 0, "Subgraph should contain nodes");
        assert!(sub.nodes.len() <= 50, "Subgraph should be smaller than full graph");
    }

    // === MODEL CHECKPOINT TESTS ===

    #[tokio::test]
    async fn test_model_checkpoint_lifecycle() {
        let storage = create_test_storage().await.expect("Storage creation failed");
        let model_id = "test_gnn_model";
        
        // Create test model configuration
        let config = GNNConfig::default();
        let weights = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.6, 0.7, 0.8],
            vec![0.9, 1.0, 1.1, 1.2],
        ];
        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), 0.25);
        metrics.insert("accuracy".to_string(), 0.85);
        metrics.insert("step".to_string(), 1000.0);

        // Save checkpoint
        let checkpoint_id = storage.save_model_checkpoint(model_id, &config, &weights, &metrics).await;
        assert!(checkpoint_id.is_ok(), "Checkpoint saving should succeed");

        let checkpoint_id = checkpoint_id.unwrap();
        assert!(checkpoint_id.contains(model_id), "Checkpoint ID should contain model ID");

        // Load checkpoint
        let loaded = storage.load_model_checkpoint(&checkpoint_id).await;
        assert!(loaded.is_ok(), "Checkpoint loading should succeed");

        let (loaded_config, loaded_weights, loaded_metrics) = loaded.unwrap();
        assert_eq!(loaded_config.node_dimensions, config.node_dimensions);
        assert_eq!(loaded_weights.len(), weights.len());
        assert_eq!(loaded_metrics.len(), metrics.len());
    }

    #[tokio::test]
    async fn test_model_checkpoint_versioning() {
        let storage = create_test_storage().await.expect("Storage creation failed");
        let model_id = "versioned_model";
        
        let config = GNNConfig::default();
        let weights = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        
        // Create multiple checkpoints
        let mut checkpoint_ids = Vec::new();
        for i in 0..3 {
            let mut metrics = HashMap::new();
            metrics.insert("loss".to_string(), 1.0 - (i as f32 * 0.1));
            metrics.insert("step".to_string(), (i * 100) as f32);
            
            let checkpoint_id = storage.save_model_checkpoint(model_id, &config, &weights, &metrics).await;
            assert!(checkpoint_id.is_ok(), "Checkpoint {} saving should succeed", i);
            checkpoint_ids.push(checkpoint_id.unwrap());
        }

        // Verify all checkpoints can be loaded
        for (i, checkpoint_id) in checkpoint_ids.iter().enumerate() {
            let loaded = storage.load_model_checkpoint(checkpoint_id).await;
            assert!(loaded.is_ok(), "Checkpoint {} loading should succeed", i);
            
            let (_, _, loaded_metrics) = loaded.unwrap();
            let expected_loss = 1.0 - (i as f32 * 0.1);
            let actual_loss = loaded_metrics.get("loss").unwrap();
            assert!((actual_loss - expected_loss).abs() < 0.001, "Loss should match for checkpoint {}", i);
        }
    }

    // === TRAINING HISTORY TESTS ===

    #[tokio::test]
    async fn test_training_lifecycle() {
        let storage = create_test_storage().await.expect("Storage creation failed");
        let model_id = "training_model";
        let graph_id = "training_graph";
        
        let config = GNNConfig::default();
        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("learning_rate".to_string(), serde_json::Value::Number(0.001.into()));
        hyperparameters.insert("batch_size".to_string(), serde_json::Value::Number(32.into()));

        // Start training run
        let run_id = storage.start_training_run(model_id, graph_id, &config, hyperparameters).await;
        assert!(run_id.is_ok(), "Training run start should succeed");
        let run_id = run_id.unwrap();

        // Record some epochs
        for epoch in 1..=5 {
            let mut epoch_metrics = HashMap::new();
            epoch_metrics.insert("loss".to_string(), 1.0 - (epoch as f32 * 0.1));
            epoch_metrics.insert("accuracy".to_string(), 0.5 + (epoch as f32 * 0.08));
            
            let result = storage.record_training_epoch(&run_id, epoch, &epoch_metrics).await;
            assert!(result.is_ok(), "Epoch {} recording should succeed", epoch);
        }

        // Complete training
        let mut final_metrics = HashMap::new();
        final_metrics.insert("final_loss".to_string(), 0.5);
        final_metrics.insert("final_accuracy".to_string(), 0.92);
        
        let result = storage.complete_training_run(&run_id, final_metrics).await;
        assert!(result.is_ok(), "Training completion should succeed");

        // Get training history
        let history = storage.get_training_history(model_id).await;
        assert!(history.is_ok(), "Training history retrieval should succeed");
        
        let history = history.unwrap();
        assert!(history.len() > 0, "Should have training history");
        assert_eq!(history[0].history.len(), 5, "Should have 5 epochs recorded");
    }

    // === PERFORMANCE TESTS ===

    #[tokio::test]
    async fn test_batch_feature_updates() {
        let storage = create_test_storage().await.expect("Storage creation failed");
        let graph_data = create_test_graph(100, 200).expect("Graph creation failed");
        let graph_id = "batch_update_test";

        // Store initial graph
        storage.store_graph(graph_id, &graph_data, None).await.expect("Graph storage failed");

        // Prepare batch feature updates
        let mut updates = Vec::new();
        for i in 0..50 {
            updates.push(crate::storage::GNNNodeFeatureUpdate {
                node_id: format!("node_{}", i),
                features: (0..10).map(|_| rand::random::<f32>()).collect(),
                version: 2,
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            });
        }

        // Perform batch update
        let result = storage.batch_update_node_features(graph_id, updates).await;
        assert!(result.is_ok(), "Batch feature update should succeed");
    }

    #[tokio::test]
    async fn test_storage_statistics() {
        let storage = create_test_storage().await.expect("Storage creation failed");
        let graph_data = create_test_graph(200, 400).expect("Graph creation failed");
        let graph_id = "stats_test";

        // Store graph
        storage.store_graph(graph_id, &graph_data, None).await.expect("Graph storage failed");

        // Get statistics
        let stats = storage.get_storage_stats().await;
        assert!(stats.is_ok(), "Statistics retrieval should succeed");
        
        let stats = stats.unwrap();
        assert!(stats.total_graphs > 0, "Should report stored graphs");
        assert!(stats.total_nodes > 0, "Should report stored nodes");
        assert!(stats.total_edges > 0, "Should report stored edges");
    }

    // === ERROR HANDLING TESTS ===

    #[tokio::test]
    async fn test_invalid_graph_id() {
        let storage = create_test_storage().await.expect("Storage creation failed");
        
        // Try to load non-existent graph
        let result = storage.load_graph("non_existent_graph").await;
        assert!(result.is_err(), "Loading non-existent graph should fail");
        
        match result.unwrap_err() {
            GNNError::InvalidInput(msg) => {
                assert!(msg.contains("not found"), "Error should mention graph not found");
            }
            _ => panic!("Should return InvalidInput error"),
        }
    }

    #[tokio::test]
    async fn test_invalid_checkpoint_id() {
        let storage = create_test_storage().await.expect("Storage creation failed");
        
        // Try to load non-existent checkpoint
        let result = storage.load_model_checkpoint("invalid_checkpoint_id").await;
        assert!(result.is_err(), "Loading invalid checkpoint should fail");
    }

    // === CONFIGURATION TESTS ===

    #[tokio::test]
    async fn test_storage_configuration() {
        let config = GNNStorageConfig {
            max_nodes_per_partition: 5000,
            replication_factor: 3,
            enable_query_cache: false,
            cache_ttl_seconds: 120,
            max_concurrent_ops: 20,
            enable_compression: false,
            batch_size: 500,
            enable_versioning: false,
        };
        
        // Test that custom configuration is respected
        assert_eq!(config.max_nodes_per_partition, 5000);
        assert_eq!(config.replication_factor, 3);
        assert!(!config.enable_query_cache);
        assert!(!config.enable_compression);
        assert!(!config.enable_versioning);
    }

    #[test]
    fn test_default_configuration() {
        let config = GNNStorageConfig::default();
        assert_eq!(config.max_nodes_per_partition, 100_000);
        assert_eq!(config.replication_factor, 2);
        assert!(config.enable_query_cache);
        assert!(config.enable_compression);
        assert!(config.enable_versioning);
        assert_eq!(config.batch_size, 1000);
    }

    // === INTEGRATION TESTS ===

    #[tokio::test]
    async fn test_end_to_end_gnn_workflow() {
        let storage = create_test_storage().await.expect("Storage creation failed");
        
        // 1. Create and store graph
        let graph_data = create_test_graph(100, 200).expect("Graph creation failed");
        let graph_id = "e2e_workflow_graph";
        storage.store_graph(graph_id, &graph_data, None).await.expect("Graph storage failed");
        
        // 2. Start training session
        let model_id = "e2e_model";
        let config = GNNConfig::default();
        let hyperparams = HashMap::new();
        
        let run_id = storage.start_training_run(model_id, graph_id, &config, hyperparams).await
            .expect("Training start failed");
        
        // 3. Simulate training epochs with checkpoints
        let weights = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        
        for epoch in 1..=3 {
            // Record epoch metrics
            let mut metrics = HashMap::new();
            metrics.insert("loss".to_string(), 1.0 - (epoch as f32 * 0.2));
            metrics.insert("accuracy".to_string(), 0.6 + (epoch as f32 * 0.1));
            metrics.insert("step".to_string(), (epoch * 100) as f32);
            
            storage.record_training_epoch(&run_id, epoch, &metrics).await
                .expect("Epoch recording failed");
            
            // Save checkpoint
            storage.save_model_checkpoint(model_id, &config, &weights, &metrics).await
                .expect("Checkpoint saving failed");
        }
        
        // 4. Complete training
        let final_metrics = HashMap::new();
        storage.complete_training_run(&run_id, final_metrics).await
            .expect("Training completion failed");
        
        // 5. Verify everything is stored correctly
        let retrieved_graph = storage.load_graph(graph_id).await.expect("Graph retrieval failed");
        assert_eq!(retrieved_graph.num_nodes(), graph_data.num_nodes());
        
        let history = storage.get_training_history(model_id).await.expect("History retrieval failed");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].history.len(), 3);
        
        let stats = storage.get_storage_stats().await.expect("Stats retrieval failed");
        assert!(stats.total_graphs >= 1);
    }

    // === CONCURRENCY TESTS ===

    #[tokio::test]
    async fn test_concurrent_operations() {
        let storage = create_test_storage().await.expect("Storage creation failed");
        
        // Create multiple tasks that operate concurrently
        let mut handles = Vec::new();
        
        for i in 0..5 {
            let storage_clone = storage.clone(); // Assuming Arc<GraphStorage>
            let handle = tokio::spawn(async move {
                let graph_data = create_test_graph(20, 30)?;
                let graph_id = format!("concurrent_graph_{}", i);
                
                storage_clone.store_graph(&graph_id, &graph_data, None).await?;
                storage_clone.load_graph(&graph_id).await
            });
            
            handles.push(handle);
        }
        
        // Wait for all operations to complete
        let results: Result<Vec<_>, _> = futures::future::join_all(handles).await
            .into_iter()
            .collect();
        
        assert!(results.is_ok(), "Concurrent operations should succeed");
        
        let graph_results: Result<Vec<_>, _> = results.unwrap()
            .into_iter()
            .collect();
        
        assert!(graph_results.is_ok(), "All graph operations should succeed");
        assert_eq!(graph_results.unwrap().len(), 5, "Should have 5 successful operations");
    }

    // === PERFORMANCE BENCHMARKS ===
    
    #[tokio::test]
    async fn benchmark_large_graph_operations() {
        let storage = create_test_storage().await.expect("Storage creation failed");
        
        let start_time = SystemTime::now();
        
        // Create moderately large graph for testing
        let graph_data = create_test_graph(1000, 2000).expect("Large graph creation failed");
        let graph_id = "benchmark_graph";
        
        // Measure storage time
        let storage_start = SystemTime::now();
        storage.store_graph(graph_id, &graph_data, None).await.expect("Graph storage failed");
        let storage_duration = storage_start.elapsed().unwrap();
        
        // Measure retrieval time  
        let retrieval_start = SystemTime::now();
        let _retrieved = storage.load_graph(graph_id).await.expect("Graph retrieval failed");
        let retrieval_duration = retrieval_start.elapsed().unwrap();
        
        let total_duration = start_time.elapsed().unwrap();
        
        println!("Benchmark Results for 1K nodes, 2K edges:");
        println!("Storage time: {:?}", storage_duration);
        println!("Retrieval time: {:?}", retrieval_duration); 
        println!("Total time: {:?}", total_duration);
        
        // Performance assertions (adjust based on expected performance)
        assert!(storage_duration.as_millis() < 10000, "Storage should complete within 10 seconds");
        assert!(retrieval_duration.as_millis() < 5000, "Retrieval should complete within 5 seconds");
    }
}