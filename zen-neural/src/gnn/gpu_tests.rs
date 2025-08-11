/**
 * @file zen-neural/src/gnn/gpu_tests.rs
 * @brief Comprehensive Tests for GPU-Accelerated Graph Neural Networks
 * 
 * This module contains extensive tests and benchmarks for the GPU acceleration
 * system, demonstrating performance improvements and correctness validation
 * against CPU implementations.
 * 
 * ## Test Categories
 * 
 * - **Correctness Tests**: Verify GPU results match CPU implementations
 * - **Performance Benchmarks**: Measure speedup on various graph sizes
 * - **Memory Tests**: Validate efficient GPU memory usage
 * - **Edge Case Tests**: Handle unusual graph structures gracefully
 * - **Integration Tests**: End-to-end GNN pipeline validation
 * 
 * ## Performance Targets
 * 
 * - 10x speedup on graphs with 1,000+ nodes
 * - 100x speedup on graphs with 10,000+ nodes
 * - <1ms latency for small graphs (<100 nodes)
 * - <10ms latency for medium graphs (1,000 nodes)
 * - <100ms latency for large graphs (10,000 nodes)
 * 
 * @author GPU Acceleration Expert (ruv-swarm)
 * @version 1.0.0
 * @since 2024-08-11
 */

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, Array1};
    
    use crate::gnn::{
        GraphData, GNNConfig, GNNModel, AggregationMethod, ActivationFunction,
        data::generate_random_graph
    };
    
    #[cfg(feature = "gpu")]
    use crate::gnn::gpu::GPUGraphProcessor;
    
    #[cfg(feature = "gpu")]
    use crate::webgpu::WebGPUBackend;
    
    // === TEST UTILITIES ===
    
    /// Create a simple triangle graph for testing
    fn create_test_graph() -> GraphData {
        let node_features = Array2::from_shape_vec((3, 4), vec![
            1.0, 0.5, 0.2, 0.1,  // Node 0
            0.8, 1.0, 0.1, 0.3,  // Node 1  
            0.3, 0.7, 0.9, 0.5   // Node 2
        ]).unwrap();
        
        let edge_features = Some(Array2::from_shape_vec((3, 2), vec![
            0.1, 0.9,  // Edge 0->1
            0.2, 0.8,  // Edge 1->2
            0.3, 0.7   // Edge 2->0
        ]).unwrap());
        
        let adjacency = vec![(0, 1), (1, 2), (2, 0)];
        
        GraphData::new(node_features, edge_features, adjacency).unwrap()
    }
    
    /// Create a larger graph for performance testing
    fn create_large_graph(num_nodes: usize, num_edges: usize) -> GraphData {
        generate_random_graph(
            num_nodes, 
            num_edges, 
            64,  // node_feature_dim
            32,  // edge_feature_dim  
            Some(42) // seed for reproducibility
        ).unwrap()
    }
    
    /// Compare two node feature matrices with tolerance
    fn assert_features_close(actual: &Array2<f32>, expected: &Array2<f32>, tolerance: f32) {
        assert_eq!(actual.shape(), expected.shape());
        
        for ((i, j), &actual_val) in actual.indexed_iter() {
            let expected_val = expected[[i, j]];
            assert_abs_diff_eq!(
                actual_val, 
                expected_val, 
                epsilon = tolerance,
                "Mismatch at position ({}, {}): {} vs {}",
                i, j, actual_val, expected_val
            );
        }
    }
    
    // === BASIC FUNCTIONALITY TESTS ===
    
    #[tokio::test]
    #[cfg(feature = "gpu")]
    async fn test_gpu_processor_initialization() {
        if let Ok(backend) = WebGPUBackend::new().await {
            let config = GNNConfig::default();
            let processor = GPUGraphProcessor::new(
                std::sync::Arc::new(backend),
                &config
            ).await;
            
            assert!(processor.is_ok(), "GPU processor should initialize successfully");
            
            let processor = processor.unwrap();
            let stats = processor.get_performance_stats();
            assert_eq!(stats.graphs_processed, 0);
        } else {
            println!("GPU not available, skipping GPU initialization test");
        }
    }
    
    #[tokio::test]
    #[cfg(feature = "gpu")]
    async fn test_small_graph_processing() {
        if let Ok(backend) = WebGPUBackend::new().await {
            let config = GNNConfig {
                node_dimensions: 4,
                edge_dimensions: 2,
                hidden_dimensions: 8,
                output_dimensions: 8,
                num_layers: 2,
                aggregation: AggregationMethod::Mean,
                activation: ActivationFunction::ReLU,
                dropout_rate: 0.0, // Disable dropout for testing
                ..Default::default()
            };
            
            let mut processor = GPUGraphProcessor::new(
                std::sync::Arc::new(backend),
                &config
            ).await.unwrap();
            
            let graph = create_test_graph();
            let result = processor.process_graph(&graph).await;
            
            assert!(result.is_ok(), "Small graph processing should succeed");
            
            let embeddings = result.unwrap();
            assert_eq!(embeddings.nrows(), 3); // 3 nodes
            assert_eq!(embeddings.ncols(), config.hidden_dimensions); // hidden dim
            
            // Check that embeddings are not all zeros
            let sum = embeddings.sum();
            assert!(sum.abs() > 1e-6, "Embeddings should not be all zeros");
        } else {
            println!("GPU not available, skipping small graph test");
        }
    }
    
    // === CORRECTNESS TESTS ===
    
    #[tokio::test]
    #[cfg(feature = "gpu")]
    async fn test_gpu_vs_cpu_consistency() {
        if let Ok(backend) = WebGPUBackend::new().await {
            let config = GNNConfig {
                node_dimensions: 16,
                edge_dimensions: 8,
                hidden_dimensions: 32,
                output_dimensions: 32,
                num_layers: 3,
                aggregation: AggregationMethod::Mean,
                activation: ActivationFunction::ReLU,
                dropout_rate: 0.0, // Disable randomness for consistency
                ..Default::default()
            };
            
            // Create identical graphs
            let graph = create_large_graph(50, 100);
            
            // Process on CPU
            let cpu_model = GNNModel::with_config(config.clone()).unwrap();
            let cpu_embeddings = cpu_model.forward(
                &graph, 
                crate::gnn::TrainingMode::Inference
            ).await.unwrap();
            
            // Process on GPU
            let mut gpu_processor = GPUGraphProcessor::new(
                std::sync::Arc::new(backend),
                &config
            ).await.unwrap();
            let gpu_embeddings = gpu_processor.process_graph(&graph).await.unwrap();
            
            // Compare results (allow for some numerical differences due to GPU precision)
            assert_features_close(&gpu_embeddings, &cpu_embeddings, 1e-3);
        } else {
            println!("GPU not available, skipping consistency test");
        }
    }
    
    #[tokio::test]
    #[cfg(feature = "gpu")]
    async fn test_different_aggregation_methods() {
        if let Ok(backend) = WebGPUBackend::new().await {
            let graph = create_test_graph();
            
            let aggregation_methods = vec![
                AggregationMethod::Mean,
                AggregationMethod::Max,
                AggregationMethod::Sum,
            ];
            
            for &agg_method in &aggregation_methods {
                let config = GNNConfig {
                    node_dimensions: 4,
                    edge_dimensions: 2,
                    hidden_dimensions: 8,
                    aggregation: agg_method,
                    dropout_rate: 0.0,
                    ..Default::default()
                };
                
                let mut processor = GPUGraphProcessor::new(
                    std::sync::Arc::new(backend.clone()),
                    &config
                ).await.unwrap();
                
                let result = processor.process_graph(&graph).await;
                assert!(result.is_ok(), "Processing should succeed for {:?}", agg_method);
                
                let embeddings = result.unwrap();
                assert!(embeddings.sum().abs() > 1e-6, "Non-zero embeddings for {:?}", agg_method);
            }
        } else {
            println!("GPU not available, skipping aggregation methods test");
        }
    }
    
    // === PERFORMANCE BENCHMARKS ===
    
    #[tokio::test]
    #[cfg(feature = "gpu")]
    async fn test_performance_small_graphs() {
        if let Ok(backend) = WebGPUBackend::new().await {
            let config = GNNConfig::default();
            let mut processor = GPUGraphProcessor::new(
                std::sync::Arc::new(backend),
                &config
            ).await.unwrap();
            
            // Test on small graphs (should complete quickly)
            let graph = create_large_graph(100, 200);
            
            let start = Instant::now();
            let result = processor.process_graph(&graph).await;
            let elapsed = start.elapsed();
            
            assert!(result.is_ok(), "Small graph processing should succeed");
            assert!(elapsed.as_millis() < 10, "Small graph should process in <10ms, took {}ms", elapsed.as_millis());
            
            println!("Small graph (100 nodes) processed in: {:?}", elapsed);
        } else {
            println!("GPU not available, skipping small graph performance test");
        }
    }
    
    #[tokio::test]
    #[cfg(feature = "gpu")]
    async fn test_performance_medium_graphs() {
        if let Ok(backend) = WebGPUBackend::new().await {
            let config = GNNConfig::default();
            let mut processor = GPUGraphProcessor::new(
                std::sync::Arc::new(backend),
                &config
            ).await.unwrap();
            
            // Test on medium graphs
            let graph = create_large_graph(1000, 2000);
            
            let start = Instant::now();
            let result = processor.process_graph(&graph).await;
            let elapsed = start.elapsed();
            
            assert!(result.is_ok(), "Medium graph processing should succeed");
            assert!(elapsed.as_millis() < 50, "Medium graph should process in <50ms, took {}ms", elapsed.as_millis());
            
            println!("Medium graph (1000 nodes) processed in: {:?}", elapsed);
        } else {
            println!("GPU not available, skipping medium graph performance test");
        }
    }
    
    #[tokio::test]
    #[cfg(feature = "gpu")]
    async fn test_performance_large_graphs() {
        if let Ok(backend) = WebGPUBackend::new().await {
            let config = GNNConfig::default();
            let mut processor = GPUGraphProcessor::new(
                std::sync::Arc::new(backend),
                &config
            ).await.unwrap();
            
            // Test on large graphs (this is where GPU should really shine)
            let graph = create_large_graph(5000, 10000);
            
            let start = Instant::now();
            let result = processor.process_graph(&graph).await;
            let elapsed = start.elapsed();
            
            assert!(result.is_ok(), "Large graph processing should succeed");
            assert!(elapsed.as_millis() < 200, "Large graph should process in <200ms, took {}ms", elapsed.as_millis());
            
            println!("Large graph (5000 nodes) processed in: {:?}", elapsed);
            
            let stats = processor.get_performance_stats();
            println!("GPU Performance Stats: {:?}", stats);
        } else {
            println!("GPU not available, skipping large graph performance test");
        }
    }
    
    #[tokio::test]
    #[cfg(feature = "gpu")]
    async fn test_batch_processing_performance() {
        if let Ok(backend) = WebGPUBackend::new().await {
            let config = GNNConfig::default();
            let mut processor = GPUGraphProcessor::new(
                std::sync::Arc::new(backend),
                &config
            ).await.unwrap();
            
            // Create a batch of graphs
            let graphs: Vec<_> = (0..5)
                .map(|i| create_large_graph(500, 1000))
                .collect();
            
            let start = Instant::now();
            let results = processor.process_batch(&graphs).await;
            let elapsed = start.elapsed();
            
            assert!(results.is_ok(), "Batch processing should succeed");
            let embeddings = results.unwrap();
            assert_eq!(embeddings.len(), 5, "Should process all 5 graphs");
            
            println!("Batch of 5 graphs (500 nodes each) processed in: {:?}", elapsed);
            
            let avg_per_graph = elapsed.as_millis() / 5;
            assert!(avg_per_graph < 20, "Average per graph should be <20ms, was {}ms", avg_per_graph);
        } else {
            println!("GPU not available, skipping batch processing test");
        }
    }
    
    // === MEMORY EFFICIENCY TESTS ===
    
    #[tokio::test]
    #[cfg(feature = "gpu")]
    async fn test_memory_usage() {
        if let Ok(backend) = WebGPUBackend::new().await {
            let config = GNNConfig::default();
            let mut processor = GPUGraphProcessor::new(
                std::sync::Arc::new(backend),
                &config
            ).await.unwrap();
            
            // Process several graphs to test memory management
            for size in [100, 500, 1000].iter() {
                let graph = create_large_graph(*size, size * 2);
                let result = processor.process_graph(&graph).await;
                
                assert!(result.is_ok(), "Graph processing should succeed for size {}", size);
                
                // Check performance stats for memory usage tracking
                let stats = processor.get_performance_stats();
                println!("Memory stats after {} node graph: {:?}", size, stats.memory_stats);
            }
        } else {
            println!("GPU not available, skipping memory usage test");
        }
    }
    
    // === EDGE CASE TESTS ===
    
    #[tokio::test]
    #[cfg(feature = "gpu")]
    async fn test_empty_graph_handling() {
        if let Ok(backend) = WebGPUBackend::new().await {
            let config = GNNConfig::default();
            let mut processor = GPUGraphProcessor::new(
                std::sync::Arc::new(backend),
                &config
            ).await.unwrap();
            
            // Test with minimal graph
            let node_features = Array2::from_shape_vec((1, 128), vec![0.0; 128]).unwrap();
            let empty_graph = GraphData::new(node_features, None, vec![]).unwrap();
            
            let result = processor.process_graph(&empty_graph).await;
            
            // Should handle gracefully (might return error or zero embeddings)
            if let Ok(embeddings) = result {
                assert_eq!(embeddings.nrows(), 1);
                println!("Empty graph processed successfully");
            } else {
                println!("Empty graph correctly rejected: {:?}", result);
            }
        } else {
            println!("GPU not available, skipping empty graph test");
        }
    }
    
    #[tokio::test]
    #[cfg(feature = "gpu")]
    async fn test_disconnected_graph() {
        if let Ok(backend) = WebGPUBackend::new().await {
            let config = GNNConfig {
                node_dimensions: 8,
                hidden_dimensions: 16,
                ..Default::default()
            };
            
            let mut processor = GPUGraphProcessor::new(
                std::sync::Arc::new(backend),
                &config
            ).await.unwrap();
            
            // Create graph with isolated nodes
            let node_features = Array2::from_shape_vec((4, 8), vec![1.0; 32]).unwrap();
            let adjacency = vec![(0, 1)]; // Only one edge, nodes 2 and 3 are isolated
            let disconnected_graph = GraphData::new(node_features, None, adjacency).unwrap();
            
            let result = processor.process_graph(&disconnected_graph).await;
            assert!(result.is_ok(), "Disconnected graph should be handled");
            
            let embeddings = result.unwrap();
            assert_eq!(embeddings.nrows(), 4);
            println!("Disconnected graph processed successfully");
        } else {
            println!("GPU not available, skipping disconnected graph test");
        }
    }
    
    // === STRESS TESTS ===
    
    #[tokio::test]
    #[cfg(feature = "gpu")]
    async fn test_stress_many_small_graphs() {
        if let Ok(backend) = WebGPUBackend::new().await {
            let config = GNNConfig::default();
            let mut processor = GPUGraphProcessor::new(
                std::sync::Arc::new(backend),
                &config
            ).await.unwrap();
            
            // Process many small graphs rapidly
            let num_graphs = 20;
            let start = Instant::now();
            
            for i in 0..num_graphs {
                let graph = create_large_graph(50, 100);
                let result = processor.process_graph(&graph).await;
                assert!(result.is_ok(), "Graph {} should process successfully", i);
            }
            
            let elapsed = start.elapsed();
            let avg_time = elapsed.as_millis() / num_graphs;
            
            println!("Processed {} small graphs in {:?} (avg: {}ms/graph)", 
                     num_graphs, elapsed, avg_time);
            
            assert!(avg_time < 5, "Average processing time should be <5ms per small graph");
        } else {
            println!("GPU not available, skipping stress test");
        }
    }
    
    // === INTEGRATION TESTS ===
    
    #[tokio::test]  
    #[cfg(feature = "gpu")]
    async fn test_end_to_end_training_simulation() {
        if let Ok(backend) = WebGPUBackend::new().await {
            let config = GNNConfig {
                node_dimensions: 32,
                hidden_dimensions: 64,
                output_dimensions: 16,
                num_layers: 3,
                dropout_rate: 0.1,
                ..Default::default()
            };
            
            let mut processor = GPUGraphProcessor::new(
                std::sync::Arc::new(backend),
                &config
            ).await.unwrap();
            
            // Simulate training loop with multiple forward passes
            let train_graphs: Vec<_> = (0..10)
                .map(|_| create_large_graph(200, 400))
                .collect();
            
            let start = Instant::now();
            
            // Simulate multiple epochs
            for epoch in 0..3 {
                for (i, graph) in train_graphs.iter().enumerate() {
                    let result = processor.process_graph(graph).await;
                    assert!(result.is_ok(), "Training step {}.{} should succeed", epoch, i);
                }
            }
            
            let elapsed = start.elapsed();
            println!("Simulated training (3 epochs, 10 graphs each) completed in: {:?}", elapsed);
            
            let stats = processor.get_performance_stats();
            println!("Final performance stats: {:?}", stats);
            
            // Should have processed 30 graphs total
            assert_eq!(stats.graphs_processed, 30);
            assert!(stats.throughput > 0.0, "Should have positive throughput");
        } else {
            println!("GPU not available, skipping training simulation");
        }
    }
    
    // === FEATURE-SPECIFIC TESTS ===
    
    #[tokio::test]
    #[cfg(feature = "gpu")]
    async fn test_different_activation_functions() {
        if let Ok(backend) = WebGPUBackend::new().await {
            let graph = create_test_graph();
            
            let activations = vec![
                ActivationFunction::ReLU,
                ActivationFunction::Tanh,
                ActivationFunction::Sigmoid,
                ActivationFunction::LeakyReLU,
                ActivationFunction::GELU,
            ];
            
            for &activation in &activations {
                let config = GNNConfig {
                    node_dimensions: 4,
                    edge_dimensions: 2,
                    hidden_dimensions: 8,
                    activation,
                    dropout_rate: 0.0,
                    ..Default::default()
                };
                
                let mut processor = GPUGraphProcessor::new(
                    std::sync::Arc::new(backend.clone()),
                    &config
                ).await.unwrap();
                
                let result = processor.process_graph(&graph).await;
                assert!(result.is_ok(), "Processing should succeed for {:?}", activation);
                
                let embeddings = result.unwrap();
                
                // Verify activation function effects
                match activation {
                    ActivationFunction::ReLU => {
                        // All values should be non-negative
                        assert!(embeddings.iter().all(|&x| x >= 0.0), "ReLU should produce non-negative values");
                    },
                    ActivationFunction::Tanh => {
                        // All values should be in [-1, 1]
                        assert!(embeddings.iter().all(|&x| x >= -1.0 && x <= 1.0), "Tanh should be in [-1, 1]");
                    },
                    ActivationFunction::Sigmoid => {
                        // All values should be in [0, 1]
                        assert!(embeddings.iter().all(|&x| x >= 0.0 && x <= 1.0), "Sigmoid should be in [0, 1]");
                    },
                    _ => {} // Other activations have different ranges
                }
                
                println!("Activation {:?} processed successfully", activation);
            }
        } else {
            println!("GPU not available, skipping activation function test");
        }
    }
    
    // === BENCHMARK COMPARISON FUNCTION ===
    
    /// Compare GPU vs CPU performance across different graph sizes
    #[tokio::test]
    #[cfg(feature = "gpu")]
    async fn benchmark_gpu_vs_cpu() {
        if let Ok(backend) = WebGPUBackend::new().await {
            let config = GNNConfig {
                node_dimensions: 64,
                hidden_dimensions: 128,
                output_dimensions: 64,
                num_layers: 3,
                dropout_rate: 0.0,
                ..Default::default()
            };
            
            let graph_sizes = vec![100, 500, 1000, 2000];
            
            println!("=== GPU vs CPU Performance Benchmark ===");
            println!("{:<12} {:<15} {:<15} {:<10}", "Graph Size", "GPU Time (ms)", "CPU Time (ms)", "Speedup");
            println!("{:-<52}", "");
            
            for &size in &graph_sizes {
                let graph = create_large_graph(size, size * 2);
                
                // GPU timing
                let mut gpu_processor = GPUGraphProcessor::new(
                    std::sync::Arc::new(backend.clone()),
                    &config
                ).await.unwrap();
                
                let gpu_start = Instant::now();
                let gpu_result = gpu_processor.process_graph(&graph).await;
                let gpu_elapsed = gpu_start.elapsed();
                
                // CPU timing
                let cpu_model = GNNModel::with_config(config.clone()).unwrap();
                
                let cpu_start = Instant::now();
                let cpu_result = cpu_model.forward(
                    &graph, 
                    crate::gnn::TrainingMode::Inference
                ).await;
                let cpu_elapsed = cpu_start.elapsed();
                
                if gpu_result.is_ok() && cpu_result.is_ok() {
                    let speedup = cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64();
                    
                    println!("{:<12} {:<15.2} {:<15.2} {:<10.2}x", 
                             size, 
                             gpu_elapsed.as_secs_f64() * 1000.0,
                             cpu_elapsed.as_secs_f64() * 1000.0,
                             speedup);
                    
                    // For larger graphs, we expect significant speedup
                    if size >= 1000 {
                        assert!(speedup > 1.0, "GPU should be faster than CPU for large graphs (size: {})", size);
                    }
                } else {
                    println!("{:<12} ERROR", size);
                }
            }
        } else {
            println!("GPU not available, skipping benchmark");
        }
    }
}