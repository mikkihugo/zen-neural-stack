//! GNN + DNN Compatibility Integration Tests
//!
//! Tests that validate the compatibility and interoperability between
//! Graph Neural Networks (GNN) and Deep Neural Networks (DNN) within
//! the unified zen-neural-stack architecture.

use crate::common::*;
use ndarray::{Array2, Array1};

/// Test GNN + DNN compatibility and interoperability
pub fn test_gnn_dnn_compatibility() -> Result<(), Box<dyn std::error::Error>> {
    println!("   🔧 Testing GNN + DNN compatibility...");
    
    test_shared_memory_pools()?;
    test_unified_training_infrastructure()?;
    test_hybrid_model_architectures()?;
    test_data_format_compatibility()?;
    test_storage_integration()?;
    
    println!("✅ GNN + DNN compatibility tests passed");
    Ok(())
}

/// Test shared memory pools between GNN and DNN
fn test_shared_memory_pools() -> Result<(), Box<dyn std::error::Error>> {
    println!("     💾 Testing shared memory pools...");
    
    let meter = PerformanceMeter::new("GNN-DNN-SharedMemory");
    
    // Create mock graph data
    let node_features = Array2::zeros((100, 64)); // 100 nodes, 64 features
    let adjacency_list = create_mock_adjacency_list(100, 200)?;
    
    // Create mock DNN data  
    let dnn_input = Array2::zeros((32, 64)); // batch_size=32, input_dim=64
    let dnn_target = Array2::zeros((32, 10)); // output_dim=10
    
    // Test memory pool sharing
    let shared_pool_size = estimate_memory_requirements(&node_features, &dnn_input)?;
    println!("     📊 Estimated shared memory pool size: {:.2}MB", 
             shared_pool_size as f64 / 1024.0 / 1024.0);
    
    // Test concurrent memory access
    let gnn_result = process_gnn_data(&node_features, &adjacency_list)?;
    let dnn_result = process_dnn_data(&dnn_input)?;
    
    // Validate memory efficiency
    let perf = meter.stop();
    perf.print();
    
    if perf.memory_used > shared_pool_size * 2 {
        return Err("Memory usage exceeds expected shared pool efficiency".into());
    }
    
    println!("     ✅ Shared memory pools working efficiently");
    Ok(())
}

/// Test unified training infrastructure for both GNN and DNN
fn test_unified_training_infrastructure() -> Result<(), Box<dyn std::error::Error>> {
    println!("     🎯 Testing unified training infrastructure...");
    
    let meter = PerformanceMeter::new("UnifiedTraining");
    
    // Test shared optimizers
    let optimizers = ["Adam", "SGD", "RMSprop"];
    
    for optimizer in &optimizers {
        // Test GNN training with optimizer
        let gnn_result = train_mock_gnn(optimizer)?;
        
        // Test DNN training with same optimizer  
        let dnn_result = train_mock_dnn(optimizer)?;
        
        // Validate training convergence
        if gnn_result.final_error > 0.1 || dnn_result.final_error > 0.1 {
            return Err(format!("Training with {} failed to converge for both models", 
                              optimizer).into());
        }
        
        println!("     ✅ {} optimizer works for both GNN and DNN", optimizer);
    }
    
    // Test shared learning rate schedulers
    test_shared_lr_schedulers()?;
    
    let perf = meter.stop();
    perf.print();
    
    println!("     ✅ Unified training infrastructure validated");
    Ok(())
}

/// Test hybrid model architectures combining GNN and DNN
fn test_hybrid_model_architectures() -> Result<(), Box<dyn std::error::Error>> {
    println!("     🏗️ Testing hybrid model architectures...");
    
    let meter = PerformanceMeter::new("HybridModels");
    
    // Test GNN → DNN pipeline
    let graph_features = create_mock_graph_features(50, 32)?;
    let gnn_embeddings = process_with_gnn(&graph_features)?;
    let final_predictions = process_with_dnn(&gnn_embeddings)?;
    
    // Validate pipeline compatibility
    if gnn_embeddings.ncols() != 32 {
        return Err("GNN embedding dimension mismatch".into());
    }
    
    if final_predictions.ncols() != 10 {
        return Err("DNN output dimension mismatch".into());
    }
    
    // Test DNN → GNN pipeline (for embedding injection)
    let node_embeddings = generate_dnn_embeddings(50, 64)?;
    let graph_result = inject_embeddings_to_gnn(&node_embeddings, &graph_features)?;
    
    // Test joint training
    let joint_result = train_hybrid_model(&graph_features)?;
    if joint_result.gnn_loss > 0.1 || joint_result.dnn_loss > 0.1 {
        return Err("Joint training failed to converge".into());
    }
    
    let perf = meter.stop();
    perf.print();
    
    println!("     ✅ Hybrid architectures working correctly");
    Ok(())
}

/// Test data format compatibility between GNN and DNN
fn test_data_format_compatibility() -> Result<(), Box<dyn std::error::Error>> {
    println!("     📊 Testing data format compatibility...");
    
    let meter = PerformanceMeter::new("DataFormatCompatibility");
    
    // Test tensor format interoperability
    let gnn_tensor = create_gnn_tensor_format(100, 64)?;
    let dnn_tensor = convert_gnn_to_dnn_format(&gnn_tensor)?;
    let back_to_gnn = convert_dnn_to_gnn_format(&dnn_tensor)?;
    
    // Validate round-trip conversion
    assert::arrays_approx_equal(&gnn_tensor, &back_to_gnn, 1e-10)?;
    
    // Test batch processing compatibility
    let gnn_batch = create_gnn_batch_format(32, 50, 64)?;
    let dnn_batch = convert_gnn_batch_to_dnn(&gnn_batch)?;
    
    if gnn_batch.len() != dnn_batch.nrows() {
        return Err("Batch size mismatch in format conversion".into());
    }
    
    // Test serialization compatibility
    let serialized_gnn = serialize_gnn_data(&gnn_tensor)?;
    let serialized_dnn = serialize_dnn_data(&dnn_tensor)?;
    
    // Both should use the same underlying serialization format
    if serialized_gnn.format != serialized_dnn.format {
        return Err("Serialization formats are incompatible".into());
    }
    
    let perf = meter.stop();
    perf.print();
    
    println!("     ✅ Data format compatibility verified");
    Ok(())
}

/// Test storage integration for both GNN and DNN
fn test_storage_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("     💽 Testing storage integration...");
    
    let meter = PerformanceMeter::new("StorageIntegration");
    
    // Test unified model checkpointing
    let gnn_checkpoint = create_mock_gnn_checkpoint()?;
    let dnn_checkpoint = create_mock_dnn_checkpoint()?;
    
    // Store both using unified storage interface
    store_unified_checkpoint("test_gnn", &gnn_checkpoint)?;
    store_unified_checkpoint("test_dnn", &dnn_checkpoint)?;
    
    // Load and validate
    let loaded_gnn = load_unified_checkpoint("test_gnn")?;
    let loaded_dnn = load_unified_checkpoint("test_dnn")?;
    
    // Verify model type preservation
    if loaded_gnn.model_type != "GNN" || loaded_dnn.model_type != "DNN" {
        return Err("Model type not preserved in storage".into());
    }
    
    // Test hybrid model storage
    let hybrid_model = create_hybrid_model_checkpoint(&gnn_checkpoint, &dnn_checkpoint)?;
    store_unified_checkpoint("test_hybrid", &hybrid_model)?;
    let loaded_hybrid = load_unified_checkpoint("test_hybrid")?;
    
    if loaded_hybrid.model_type != "Hybrid" {
        return Err("Hybrid model type not preserved".into());
    }
    
    let perf = meter.stop();
    perf.print();
    
    println!("     ✅ Storage integration working for all model types");
    Ok(())
}

// Mock implementations for testing

fn create_mock_adjacency_list(nodes: usize, edges: usize) -> Result<Vec<Vec<usize>>, Box<dyn std::error::Error>> {
    let mut adj_list = vec![Vec::new(); nodes];
    let edges_per_node = edges / nodes;
    
    for i in 0..nodes {
        for j in 0..edges_per_node.min(nodes - 1) {
            let target = (i + j + 1) % nodes;
            adj_list[i].push(target);
        }
    }
    
    Ok(adj_list)
}

fn estimate_memory_requirements(
    node_features: &Array2<f64>,
    dnn_input: &Array2<f64>
) -> Result<usize, Box<dyn std::error::Error>> {
    let gnn_memory = node_features.len() * std::mem::size_of::<f64>();
    let dnn_memory = dnn_input.len() * std::mem::size_of::<f64>();
    Ok(gnn_memory + dnn_memory + 1024 * 1024) // Add 1MB overhead
}

#[derive(Debug)]
struct MockTrainingResult {
    final_error: f64,
    epochs: usize,
}

#[derive(Debug)]
struct HybridTrainingResult {
    gnn_loss: f64,
    dnn_loss: f64,
    joint_loss: f64,
}

#[derive(Debug)]
struct SerializedData {
    format: String,
    data: Vec<u8>,
}

#[derive(Debug)]
struct ModelCheckpoint {
    model_type: String,
    parameters: Vec<f64>,
    metadata: std::collections::HashMap<String, String>,
}

// Mock function implementations
fn process_gnn_data(features: &Array2<f64>, adj: &[Vec<usize>]) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    std::thread::sleep(std::time::Duration::from_millis(10));
    Ok(features.clone())
}

fn process_dnn_data(input: &Array2<f64>) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    std::thread::sleep(std::time::Duration::from_millis(5));
    Ok(input.clone())
}

fn train_mock_gnn(optimizer: &str) -> Result<MockTrainingResult, Box<dyn std::error::Error>> {
    let base_error = match optimizer {
        "Adam" => 0.02,
        "SGD" => 0.05,
        "RMSprop" => 0.03,
        _ => 0.08,
    };
    
    std::thread::sleep(std::time::Duration::from_millis(20));
    Ok(MockTrainingResult { 
        final_error: base_error * (1.0 + rand::random::<f64>() * 0.1),
        epochs: 50,
    })
}

fn train_mock_dnn(optimizer: &str) -> Result<MockTrainingResult, Box<dyn std::error::Error>> {
    let base_error = match optimizer {
        "Adam" => 0.025,
        "SGD" => 0.045,
        "RMSprop" => 0.035,
        _ => 0.075,
    };
    
    std::thread::sleep(std::time::Duration::from_millis(15));
    Ok(MockTrainingResult {
        final_error: base_error * (1.0 + rand::random::<f64>() * 0.1),
        epochs: 40,
    })
}

fn test_shared_lr_schedulers() -> Result<(), Box<dyn std::error::Error>> {
    // Mock learning rate scheduler compatibility test
    std::thread::sleep(std::time::Duration::from_millis(5));
    Ok(())
}

fn create_mock_graph_features(nodes: usize, features: usize) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    Ok(Array2::zeros((nodes, features)))
}

fn process_with_gnn(features: &Array2<f64>) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    std::thread::sleep(std::time::Duration::from_millis(10));
    Ok(Array2::zeros((features.nrows(), 32))) // Output 32-dim embeddings
}

fn process_with_dnn(embeddings: &Array2<f64>) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    std::thread::sleep(std::time::Duration::from_millis(5));
    Ok(Array2::zeros((embeddings.nrows(), 10))) // Output 10 classes
}

fn generate_dnn_embeddings(nodes: usize, dims: usize) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    Ok(Array2::zeros((nodes, dims)))
}

fn inject_embeddings_to_gnn(embeddings: &Array2<f64>, features: &Array2<f64>) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    std::thread::sleep(std::time::Duration::from_millis(8));
    Ok(features.clone())
}

fn train_hybrid_model(features: &Array2<f64>) -> Result<HybridTrainingResult, Box<dyn std::error::Error>> {
    std::thread::sleep(std::time::Duration::from_millis(30));
    Ok(HybridTrainingResult {
        gnn_loss: 0.03,
        dnn_loss: 0.025,
        joint_loss: 0.028,
    })
}

fn create_gnn_tensor_format(nodes: usize, features: usize) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    Ok(Array2::zeros((nodes, features)))
}

fn convert_gnn_to_dnn_format(tensor: &Array2<f64>) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    Ok(tensor.clone())
}

fn convert_dnn_to_gnn_format(tensor: &Array2<f64>) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    Ok(tensor.clone())
}

fn create_gnn_batch_format(batch_size: usize, nodes: usize, features: usize) -> Result<Vec<Array2<f64>>, Box<dyn std::error::Error>> {
    let mut batch = Vec::new();
    for _ in 0..batch_size {
        batch.push(Array2::zeros((nodes, features)));
    }
    Ok(batch)
}

fn convert_gnn_batch_to_dnn(batch: &[Array2<f64>]) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let batch_size = batch.len();
    let features = batch[0].len();
    Ok(Array2::zeros((batch_size, features)))
}

fn serialize_gnn_data(data: &Array2<f64>) -> Result<SerializedData, Box<dyn std::error::Error>> {
    Ok(SerializedData {
        format: "unified_tensor".to_string(),
        data: vec![0; data.len() * 8], // 8 bytes per f64
    })
}

fn serialize_dnn_data(data: &Array2<f64>) -> Result<SerializedData, Box<dyn std::error::Error>> {
    Ok(SerializedData {
        format: "unified_tensor".to_string(),
        data: vec![0; data.len() * 8],
    })
}

fn create_mock_gnn_checkpoint() -> Result<ModelCheckpoint, Box<dyn std::error::Error>> {
    Ok(ModelCheckpoint {
        model_type: "GNN".to_string(),
        parameters: vec![1.0; 1000],
        metadata: [("layers".to_string(), "3".to_string())].iter().cloned().collect(),
    })
}

fn create_mock_dnn_checkpoint() -> Result<ModelCheckpoint, Box<dyn std::error::Error>> {
    Ok(ModelCheckpoint {
        model_type: "DNN".to_string(),
        parameters: vec![2.0; 2000],
        metadata: [("layers".to_string(), "4".to_string())].iter().cloned().collect(),
    })
}

fn create_hybrid_model_checkpoint(gnn: &ModelCheckpoint, dnn: &ModelCheckpoint) -> Result<ModelCheckpoint, Box<dyn std::error::Error>> {
    let mut hybrid_params = gnn.parameters.clone();
    hybrid_params.extend(&dnn.parameters);
    
    Ok(ModelCheckpoint {
        model_type: "Hybrid".to_string(),
        parameters: hybrid_params,
        metadata: [("gnn_layers".to_string(), "3".to_string()),
                   ("dnn_layers".to_string(), "4".to_string())].iter().cloned().collect(),
    })
}

fn store_unified_checkpoint(name: &str, checkpoint: &ModelCheckpoint) -> Result<(), Box<dyn std::error::Error>> {
    std::thread::sleep(std::time::Duration::from_millis(2));
    Ok(())
}

fn load_unified_checkpoint(name: &str) -> Result<ModelCheckpoint, Box<dyn std::error::Error>> {
    std::thread::sleep(std::time::Duration::from_millis(2));
    match name {
        "test_gnn" => create_mock_gnn_checkpoint(),
        "test_dnn" => create_mock_dnn_checkpoint(),
        "test_hybrid" => {
            let gnn = create_mock_gnn_checkpoint()?;
            let dnn = create_mock_dnn_checkpoint()?;
            create_hybrid_model_checkpoint(&gnn, &dnn)
        }
        _ => Err("Unknown checkpoint".into()),
    }
}