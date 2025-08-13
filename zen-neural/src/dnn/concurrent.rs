/**
 * @file zen-neural/src/dnn/concurrent.rs
 * @brief Concurrent Neural Network Operations with Arc/RwLock
 * 
 * This module implements thread-safe concurrent neural network operations using Arc and RwLock
 * for high-performance parallel processing. Enables safe sharing of neural network models
 * across multiple threads while maintaining data integrity and performance.
 * 
 * ## Concurrency Architecture:
 * - **Arc<RwLock<T>>**: Thread-safe reference-counted smart pointers with reader-writer locks
 * - **Parallel Inference**: Multiple threads can perform forward passes simultaneously
 * - **Concurrent Training**: Thread-safe parameter updates and gradient accumulation
 * - **Lock-Free Reads**: Multiple concurrent readers for inference operations
 * - **Exclusive Writes**: Single writer for parameter updates during training
 * 
 * ## Performance Benefits:
 * - **Multi-threaded Inference**: Scale inference across CPU cores
 * - **Concurrent Model Serving**: Handle multiple requests simultaneously
 * - **Parallel Batch Processing**: Distribute batches across threads
 * - **Lock Granularity**: Fine-grained locking for minimal contention
 * - **Memory Efficiency**: Share model parameters across threads
 * 
 * @author Concurrent Neural Systems Agent
 * @version 1.0.0-alpha.1
 * @since 2025-01-14
 */

use std::sync::{Arc, RwLock, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use num_traits::Float;
use ndarray::{Array2, Array1, Axis};

#[cfg(feature = "concurrent")]
use crossbeam_channel::{Sender, Receiver, bounded, unbounded};

#[cfg(feature = "concurrent")]
use uuid;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "parallel")]
#[allow(unused_imports)] // False positive: used by parallel iterators when parallel feature is enabled
use rayon::prelude::*;

use super::{ZenDNNModel, DNNError, DNNTensor, DNNTrainingMode, DNNLayer};
use super::data::{TensorShape, TensorOps};

// === CONCURRENT DNN MODEL ===

/**
 * Thread-safe concurrent DNN model using Arc and RwLock.
 * 
 * Wraps the standard ZenDNNModel in Arc<RwLock<T>> to enable safe concurrent access
 * across multiple threads. Provides methods for parallel inference, concurrent training,
 * and thread-safe parameter management.
 */
#[derive(Debug, Clone)]
pub struct ConcurrentDNNModel {
    /// Thread-safe model wrapped in Arc<RwLock>
    model: Arc<RwLock<ZenDNNModel>>,
    
    /// Concurrent inference configuration
    inference_config: ConcurrentInferenceConfig,
    
    /// Thread pool for parallel operations
    thread_pool: Arc<ConcurrentThreadPool>,
    
    /// Performance metrics for concurrent operations
    metrics: Arc<Mutex<ConcurrentMetrics>>,
}

/**
 * Configuration for concurrent inference operations.
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct ConcurrentInferenceConfig {
    /// Maximum number of concurrent inference threads
    pub max_inference_threads: usize,
    
    /// Batch size for parallel processing
    pub parallel_batch_size: usize,
    
    /// Timeout for inference operations (milliseconds)
    pub inference_timeout_ms: u64,
    
    /// Whether to enable concurrent model serving
    pub enable_concurrent_serving: bool,
    
    /// Buffer size for concurrent request queue
    pub request_queue_size: usize,
    
    /// Enable performance monitoring
    pub enable_metrics: bool,
}

impl Default for ConcurrentInferenceConfig {
    fn default() -> Self {
        Self {
            max_inference_threads: num_cpus::get(),
            parallel_batch_size: 32,
            inference_timeout_ms: 5000, // 5 second timeout
            enable_concurrent_serving: true,
            request_queue_size: 1000,
            enable_metrics: true,
        }
    }
}

/**
 * Thread pool for concurrent neural network operations.
 */
#[derive(Debug)]
pub struct ConcurrentThreadPool {
    /// Worker threads for inference
    inference_workers: Vec<thread::JoinHandle<()>>,
    
    /// Request sender channel
    request_sender: Sender<InferenceRequest>,
    
    /// Response receiver channel
    response_receiver: Receiver<InferenceResponse>,
    
    /// Shutdown signal
    shutdown_sender: Sender<bool>,
    
    /// Pool status
    is_running: Arc<RwLock<bool>>,
}

/**
 * Concurrent inference request.
 */
#[derive(Debug)]
pub struct InferenceRequest {
    /// Input tensor for inference
    pub input: DNNTensor,
    
    /// Inference mode
    pub mode: DNNTrainingMode,
    
    /// Request ID for tracking
    pub request_id: String,
    
    /// Response sender for this request
    pub response_sender: Sender<InferenceResponse>,
    
    /// Request timestamp
    pub timestamp: Instant,
}

/**
 * Concurrent inference response.
 */
#[derive(Debug)]
pub struct InferenceResponse {
    /// Output tensor from inference
    pub output: Result<DNNTensor, DNNError>,
    
    /// Request ID that generated this response
    pub request_id: String,
    
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    
    /// Thread ID that processed the request
    pub thread_id: String,
    
    /// Response timestamp
    pub timestamp: Instant,
}

/**
 * Performance metrics for concurrent operations.
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct ConcurrentMetrics {
    /// Total number of concurrent inference requests
    pub total_inference_requests: u64,
    
    /// Number of successful inferences
    pub successful_inferences: u64,
    
    /// Number of failed inferences
    pub failed_inferences: u64,
    
    /// Average processing time per request (ms)
    pub avg_processing_time_ms: f64,
    
    /// Peak concurrent requests handled
    pub peak_concurrent_requests: u64,
    
    /// Total thread contention time (ms)
    pub total_contention_time_ms: u64,
    
    /// Lock acquisition statistics
    pub lock_stats: LockStatistics,
    
    /// Thread utilization metrics
    pub thread_utilization: ThreadUtilization,
    
    /// Per-thread performance metrics tracking
    pub thread_performance: HashMap<String, ThreadPerformanceData>,
    
    /// Layer-specific timing and memory metrics
    pub layer_metrics: HashMap<String, LayerTimingData>,
}

/// Performance data for individual threads
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct ThreadPerformanceData {
    /// Thread identifier
    pub thread_id: String,
    /// Number of tasks processed
    pub tasks_completed: u64,
    /// Total processing time (ms)
    pub total_processing_time_ms: f64,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
}

/// Timing data for individual layers
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct LayerTimingData {
    /// Layer name
    pub layer_name: String,
    /// Total forward pass time (ms)
    pub total_forward_time_ms: f64,
    /// Number of forward passes
    pub forward_pass_count: u64,
    /// Average time per forward pass (ms)
    pub avg_forward_time_ms: f64,
    /// Memory allocation count
    pub memory_allocations: u64,
}

/**
 * Lock acquisition and contention statistics.
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct LockStatistics {
    /// Total read lock acquisitions
    pub read_locks_acquired: u64,
    
    /// Total write lock acquisitions
    pub write_locks_acquired: u64,
    
    /// Average time to acquire read lock (ms)
    pub avg_read_lock_time_ms: f64,
    
    /// Average time to acquire write lock (ms)
    pub avg_write_lock_time_ms: f64,
    
    /// Maximum contention time observed (ms)
    pub max_contention_time_ms: u64,
    
    /// Lock contention events
    pub lock_contentions: u64,
}

/**
 * Thread utilization and performance metrics.
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct ThreadUtilization {
    /// Number of active threads
    pub active_threads: usize,
    
    /// Average CPU utilization per thread (0.0 - 1.0)
    pub avg_cpu_utilization: f64,
    
    /// Total thread idle time (ms)
    pub total_idle_time_ms: u64,
    
    /// Thread efficiency score (0.0 - 1.0)
    pub thread_efficiency: f64,
    
    /// Load balancing effectiveness (0.0 - 1.0)
    pub load_balance_score: f64,
}

impl ConcurrentDNNModel {
    /// Create a new concurrent DNN model from an existing model
    pub fn new(model: ZenDNNModel) -> Result<Self, DNNError> {
        Self::with_config(model, ConcurrentInferenceConfig::default())
    }
    
    /// Create a concurrent DNN model with custom configuration
    pub fn with_config(
        model: ZenDNNModel,
        config: ConcurrentInferenceConfig,
    ) -> Result<Self, DNNError> {
        if !model.compiled {
            return Err(DNNError::ModelNotCompiled(
                "Model must be compiled before creating concurrent wrapper".to_string()
            ));
        }
        
        let model = Arc::new(RwLock::new(model));
        let metrics = Arc::new(Mutex::new(ConcurrentMetrics::default()));
        let thread_pool = Arc::new(Self::create_thread_pool(&config, model.clone(), metrics.clone())?);
        
        Ok(Self {
            model,
            inference_config: config,
            thread_pool,
            metrics,
        })
    }
    
    /**
     * Perform concurrent forward pass with automatic load balancing.
     * 
     * Distributes the input batch across available threads for parallel processing,
     * then aggregates the results. Uses read locks to allow multiple concurrent
     * inferences without blocking.
     */
    pub async fn concurrent_forward(
        &self,
        input: &DNNTensor,
        mode: DNNTrainingMode,
    ) -> Result<DNNTensor, DNNError> {
        let start_time = Instant::now();
        
        // Track metrics if enabled
        if self.inference_config.enable_metrics {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.total_inference_requests += 1;
        }
        
        // Check if batch should be split for parallel processing
        let batch_size = input.batch_size();
        if batch_size <= self.inference_config.parallel_batch_size {
            // Small batch - process directly
            return self.single_forward(input, mode).await;
        }
        
        // Large batch - split across threads
        self.parallel_batch_forward(input, mode).await
    }
    
    /**
     * Process a single forward pass with read lock.
     */
    async fn single_forward(
        &self,
        input: &DNNTensor,
        mode: DNNTrainingMode,
    ) -> Result<DNNTensor, DNNError> {
        let lock_start = Instant::now();
        
        // Acquire read lock (allows concurrent reads)
        let model = self.model.read().map_err(|_| {
            DNNError::ComputationError("Failed to acquire read lock on model".to_string())
        })?;
        
        let lock_time = lock_start.elapsed().as_millis() as u64;
        
        // Update lock statistics
        if self.inference_config.enable_metrics {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.lock_stats.read_locks_acquired += 1;
            metrics.lock_stats.avg_read_lock_time_ms = 
                (metrics.lock_stats.avg_read_lock_time_ms * (metrics.lock_stats.read_locks_acquired - 1) as f64 + lock_time as f64) 
                / metrics.lock_stats.read_locks_acquired as f64;
        }
        
        // Perform forward pass
        let result = model.forward(input, mode).await;
        
        // Update success/failure metrics
        if self.inference_config.enable_metrics {
            let mut metrics = self.metrics.lock().unwrap();
            match &result {
                Ok(_) => metrics.successful_inferences += 1,
                Err(_) => metrics.failed_inferences += 1,
            }
        }
        
        result
    }
    
    /**
     * Process large batch with parallel processing across threads.
     */
    async fn parallel_batch_forward(
        &self,
        input: &DNNTensor,
        mode: DNNTrainingMode,
    ) -> Result<DNNTensor, DNNError> {
        let batch_size = input.batch_size();
        let chunk_size = self.inference_config.parallel_batch_size;
        let num_chunks = (batch_size + chunk_size - 1) / chunk_size;
        
        // Split input into chunks
        let input_chunks = self.split_tensor_into_chunks(input, chunk_size)?;
        
        // Process chunks in parallel
        #[cfg(feature = "parallel")]
        {
            let results: Result<Vec<DNNTensor>, DNNError> = input_chunks
                .par_iter()
                .map(|chunk| {
                    // Each thread gets its own runtime for async operations
                    let runtime = tokio::runtime::Runtime::new().unwrap();
                    runtime.block_on(self.single_forward(chunk, mode))
                })
                .collect();
            
            let chunk_results = results?;
            
            // Concatenate results
            self.concatenate_tensor_results(&chunk_results)
        }
        
        #[cfg(not(feature = "parallel"))]
        {
            // Fallback to sequential processing
            let mut chunk_results = Vec::new();
            for chunk in input_chunks {
                let result = self.single_forward(&chunk, mode).await?;
                chunk_results.push(result);
            }
            
            self.concatenate_tensor_results(&chunk_results)
        }
    }
    
    /**
     * Concurrent parameter update with write lock.
     * 
     * Acquires exclusive write lock to safely update model parameters during training.
     * Blocks all concurrent operations until parameter update is complete.
     */
    pub async fn concurrent_parameter_update<F>(
        &self,
        update_fn: F,
    ) -> Result<(), DNNError>
    where
        F: FnOnce(&mut ZenDNNModel) -> Result<(), DNNError>,
    {
        let lock_start = Instant::now();
        
        // Acquire write lock (exclusive access)
        let mut model = self.model.write().map_err(|_| {
            DNNError::ComputationError("Failed to acquire write lock on model".to_string())
        })?;
        
        let lock_time = lock_start.elapsed().as_millis() as u64;
        
        // Update lock statistics
        if self.inference_config.enable_metrics {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.lock_stats.write_locks_acquired += 1;
            metrics.lock_stats.avg_write_lock_time_ms = 
                (metrics.lock_stats.avg_write_lock_time_ms * (metrics.lock_stats.write_locks_acquired - 1) as f64 + lock_time as f64) 
                / metrics.lock_stats.write_locks_acquired as f64;
        }
        
        // Apply parameter update
        let result = update_fn(&mut *model);
        
        // Release write lock before returning
        drop(model);
        
        result
    }
    
    /**
     * Get concurrent performance metrics.
     */
    pub fn get_concurrent_metrics(&self) -> ConcurrentMetrics {
        self.metrics.lock().unwrap().clone()
    }
    
    /**
     * Reset concurrent performance metrics.
     */
    pub fn reset_metrics(&self) {
        let mut metrics = self.metrics.lock().unwrap();
        *metrics = ConcurrentMetrics::default();
    }
    
    /**
     * Clone the underlying model for safe sharing.
     * 
     * Creates a new Arc reference to the same underlying model,
     * allowing safe sharing across threads.
     */
    pub fn share_model(&self) -> Arc<RwLock<ZenDNNModel>> {
        Arc::clone(&self.model)
    }
    
    /**
     * Perform concurrent model inference with request queuing.
     */
    pub async fn queue_inference_request(
        &self,
        input: DNNTensor,
        mode: DNNTrainingMode,
    ) -> Result<InferenceResponse, DNNError> {
        let request_id = format!("req_{}", uuid::Uuid::new_v4());
        let (response_sender, response_receiver) = bounded(1);
        
        let request = InferenceRequest {
            input,
            mode,
            request_id: request_id.clone(),
            response_sender,
            timestamp: Instant::now(),
        };
        
        // Send request to thread pool
        self.thread_pool.request_sender
            .send(request)
            .map_err(|_| DNNError::ComputationError("Failed to queue inference request".to_string()))?;
        
        // Wait for response with timeout
        let timeout = Duration::from_millis(self.inference_config.inference_timeout_ms);
        response_receiver
            .recv_timeout(timeout)
            .map_err(|_| DNNError::ComputationError("Inference request timed out".to_string()))
    }
    
    // === PRIVATE HELPER METHODS ===
    
    /// Create concurrent thread pool for inference processing
    fn create_thread_pool(
        config: &ConcurrentInferenceConfig,
        model: Arc<RwLock<ZenDNNModel>>,
        metrics: Arc<Mutex<ConcurrentMetrics>>,
    ) -> Result<ConcurrentThreadPool, DNNError> {
        let (request_sender, request_receiver) = bounded(config.request_queue_size);
        let (response_sender, response_receiver) = unbounded();
        let (shutdown_sender, shutdown_receiver) = bounded(1);
        let is_running = Arc::new(RwLock::new(true));
        
        let mut inference_workers = Vec::new();
        
        // Spawn worker threads
        for thread_id in 0..config.max_inference_threads {
            let worker_model = Arc::clone(&model);
            let worker_request_receiver = request_receiver.clone();
            let worker_response_sender = response_sender.clone();
            let worker_shutdown_receiver = shutdown_receiver.clone();
            let worker_metrics = Arc::clone(&metrics);
            let worker_is_running = Arc::clone(&is_running);
            let thread_name = format!("dnn_inference_worker_{}", thread_id);
            
            let worker = thread::Builder::new()
                .name(thread_name.clone())
                .spawn(move || {
                    Self::inference_worker_loop(
                        thread_name,
                        worker_model,
                        worker_request_receiver,
                        worker_response_sender,
                        worker_shutdown_receiver,
                        worker_metrics,
                        worker_is_running,
                    );
                })
                .map_err(|e| DNNError::ComputationError(format!("Failed to spawn inference worker: {}", e)))?;
            
            inference_workers.push(worker);
        }
        
        Ok(ConcurrentThreadPool {
            inference_workers,
            request_sender,
            response_receiver,
            shutdown_sender,
            is_running,
        })
    }
    
    /// Worker thread loop for processing inference requests
    fn inference_worker_loop(
        thread_name: String,
        model: Arc<RwLock<ZenDNNModel>>,
        request_receiver: Receiver<InferenceRequest>,
        response_sender: Sender<InferenceResponse>,
        shutdown_receiver: Receiver<bool>,
        metrics: Arc<Mutex<ConcurrentMetrics>>,
        is_running: Arc<RwLock<bool>>,
    ) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        
        loop {
            // Check for shutdown signal
            if let Ok(_) = shutdown_receiver.try_recv() {
                break;
            }
            
            if !*is_running.read().unwrap() {
                break;
            }
            
            // Wait for inference request
            match request_receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(request) => {
                    let processing_start = Instant::now();
                    
                    // Process inference request
                    let output = runtime.block_on(async {
                        let model_guard = model.read().unwrap();
                        model_guard.forward(&request.input, request.mode).await
                    });
                    
                    let processing_time = processing_start.elapsed().as_millis() as u64;
                    
                    // Create response
                    let response = InferenceResponse {
                        output,
                        request_id: request.request_id,
                        processing_time_ms: processing_time,
                        thread_id: thread_name.clone(),
                        timestamp: Instant::now(),
                    };
                    
                    // Send response
                    if let Err(_) = request.response_sender.send(response) {
                        // Client may have timed out, continue processing
                    }
                    
                    // Update metrics
                    {
                        let mut metrics_guard = metrics.lock().unwrap();
                        metrics_guard.avg_processing_time_ms = 
                            (metrics_guard.avg_processing_time_ms * (metrics_guard.total_inference_requests - 1) as f64 + processing_time as f64) 
                            / metrics_guard.total_inference_requests as f64;
                    }
                }
                Err(_) => {
                    // Timeout - continue loop to check for shutdown
                    continue;
                }
            }
        }
    }
    
    /// Split tensor into chunks for parallel processing
    fn split_tensor_into_chunks(&self, input: &DNNTensor, chunk_size: usize) -> Result<Vec<DNNTensor>, DNNError> {
        let batch_size = input.batch_size();
        let feature_dim = input.feature_dim();
        let mut chunks = Vec::new();
        
        let mut start_idx = 0;
        while start_idx < batch_size {
            let end_idx = (start_idx + chunk_size).min(batch_size);
            let chunk_batch_size = end_idx - start_idx;
            
            // Extract chunk data
            let chunk_data: Vec<f32> = (start_idx..end_idx)
                .flat_map(|batch_idx| {
                    (0..feature_dim).map(move |feature_idx| {
                        input.data[[batch_idx, feature_idx]]
                    })
                })
                .collect();
            
            // Create chunk tensor
            let chunk_shape = TensorShape::new_2d(chunk_batch_size, feature_dim);
            let chunk_tensor = DNNTensor::from_vec(chunk_data, &chunk_shape)?;
            chunks.push(chunk_tensor);
            
            start_idx = end_idx;
        }
        
        Ok(chunks)
    }
    
    /// Concatenate tensor results from parallel processing
    fn concatenate_tensor_results(&self, results: &[DNNTensor]) -> Result<DNNTensor, DNNError> {
        if results.is_empty() {
            return Err(DNNError::InvalidInput("No results to concatenate".to_string()));
        }
        
        let feature_dim = results[0].feature_dim();
        let total_batch_size: usize = results.iter().map(|t| t.batch_size()).sum();
        
        // Concatenate data
        let mut concatenated_data = Vec::with_capacity(total_batch_size * feature_dim);
        for result in results {
            for batch_idx in 0..result.batch_size() {
                for feature_idx in 0..feature_dim {
                    concatenated_data.push(result.data[[batch_idx, feature_idx]]);
                }
            }
        }
        
        // Create concatenated tensor
        let concatenated_shape = TensorShape::new_2d(total_batch_size, feature_dim);
        DNNTensor::from_vec(concatenated_data, &concatenated_shape)
    }
}

impl Drop for ConcurrentDNNModel {
    /// Cleanup concurrent resources on drop
    fn drop(&mut self) {
        // Signal shutdown to worker threads
        let _ = self.thread_pool.shutdown_sender.send(true);
        
        // Update running status
        if let Ok(mut is_running) = self.thread_pool.is_running.write() {
            *is_running = false;
        }
    }
}

// === CONCURRENT LAYER WRAPPER ===

/**
 * Thread-safe wrapper for individual DNN layers.
 * 
 * Enables concurrent access to layer parameters and forward/backward operations
 * while maintaining thread safety through Arc and RwLock.
 */
#[derive(Debug, Clone)]
pub struct ConcurrentLayer {
    /// Thread-safe layer wrapped in Arc<RwLock>
    layer: Arc<RwLock<Box<dyn DNNLayer>>>,
    
    /// Layer metadata and performance tracking
    metadata: Arc<Mutex<LayerMetadata>>,
}

/**
 * Metadata and performance tracking for concurrent layers.
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct LayerMetadata {
    /// Layer name/identifier
    pub name: String,
    
    /// Number of concurrent forward passes
    pub concurrent_forwards: u64,
    
    /// Number of concurrent backward passes
    pub concurrent_backwards: u64,
    
    /// Average forward pass time (ms)
    pub avg_forward_time_ms: f64,
    
    /// Average backward pass time (ms)
    pub avg_backward_time_ms: f64,
    
    /// Layer contention metrics
    pub contention_events: u64,
    
    /// Thread safety violations (should be 0)
    pub safety_violations: u64,
}

impl ConcurrentLayer {
    /// Wrap a layer in thread-safe concurrent wrapper
    pub fn new(layer: Box<dyn DNNLayer>, name: String) -> Self {
        Self {
            layer: Arc::new(RwLock::new(layer)),
            metadata: Arc::new(Mutex::new(LayerMetadata {
                name,
                ..Default::default()
            })),
        }
    }
    
    /// Perform concurrent forward pass
    pub async fn concurrent_forward(
        &self,
        input: &DNNTensor,
        mode: DNNTrainingMode,
    ) -> Result<DNNTensor, DNNError> {
        let start_time = Instant::now();
        
        // Acquire read lock for forward pass
        let layer = self.layer.read().map_err(|_| {
            DNNError::ComputationError("Failed to acquire read lock on layer".to_string())
        })?;
        
        // Perform forward pass
        let result = layer.forward(input, mode).await;
        
        // Update metadata
        let processing_time = start_time.elapsed().as_millis() as u64;
        {
            let mut metadata = self.metadata.lock().unwrap();
            metadata.concurrent_forwards += 1;
            metadata.avg_forward_time_ms = 
                (metadata.avg_forward_time_ms * (metadata.concurrent_forwards - 1) as f64 + processing_time as f64) 
                / metadata.concurrent_forwards as f64;
        }
        
        result
    }
    
    /// Perform concurrent backward pass (requires write lock)
    pub async fn concurrent_backward(
        &self,
        input: &DNNTensor,
        grad_output: &DNNTensor,
    ) -> Result<DNNTensor, DNNError> {
        let start_time = Instant::now();
        
        // Acquire write lock for backward pass (modifies layer state)
        let mut layer = self.layer.write().map_err(|_| {
            DNNError::ComputationError("Failed to acquire write lock on layer".to_string())
        })?;
        
        // Perform backward pass
        let result = layer.backward(input, grad_output).await;
        
        // Update metadata
        let processing_time = start_time.elapsed().as_millis() as u64;
        {
            let mut metadata = self.metadata.lock().unwrap();
            metadata.concurrent_backwards += 1;
            metadata.avg_backward_time_ms = 
                (metadata.avg_backward_time_ms * (metadata.concurrent_backwards - 1) as f64 + processing_time as f64) 
                / metadata.concurrent_backwards as f64;
        }
        
        result
    }
    
    /// Get layer metadata and performance metrics
    pub fn get_metadata(&self) -> LayerMetadata {
        self.metadata.lock().unwrap().clone()
    }
    
    /// Share layer across threads safely
    pub fn share_layer(&self) -> Arc<RwLock<Box<dyn DNNLayer>>> {
        Arc::clone(&self.layer)
    }
}

// === CONCURRENT UTILITIES ===

/**
 * Utilities for managing concurrent neural network operations.
 */
pub struct ConcurrentUtils;

/// Operations that can be performed along tensor axes concurrently
#[derive(Debug, Clone, Copy)]
pub enum ConcurrentAxisOp {
    Sum,
    Mean,
    Max,
}

impl ConcurrentUtils {
    /// Create thread-safe tensor arrays for concurrent access
    pub fn create_concurrent_tensor_arrays(
        shapes: Vec<(usize, usize)>
    ) -> Result<HashMap<String, Arc<RwLock<Array2<f32>>>>, DNNError> {
        let mut tensor_arrays = HashMap::new();
        
        for (idx, (rows, cols)) in shapes.iter().enumerate() {
            let array = Array2::<f32>::zeros((*rows, *cols));
            let key = format!("tensor_{}", idx);
            tensor_arrays.insert(key, Arc::new(RwLock::new(array)));
        }
        
        Ok(tensor_arrays)
    }
    
    /// Perform concurrent axis-wise operations on tensor data
    pub fn concurrent_axis_operation(
        data: Arc<RwLock<Array2<f32>>>,
        axis: usize,
        operation: ConcurrentAxisOp
    ) -> Result<Array1<f32>, DNNError> {
        let data_lock = data.read().unwrap();
        
        let result = match operation {
            ConcurrentAxisOp::Sum => {
                match axis {
                    0 => data_lock.sum_axis(Axis(0)),
                    1 => data_lock.sum_axis(Axis(1)),
                    _ => return Err(DNNError::InvalidConfiguration(
                        format!("Unsupported axis: {}", axis)
                    )),
                }
            }
            ConcurrentAxisOp::Mean => {
                match axis {
                    0 => data_lock.mean_axis(Axis(0)).ok_or_else(|| DNNError::ComputationError(
                        "Failed to compute mean along axis 0".to_string()
                    ))?,
                    1 => data_lock.mean_axis(Axis(1)).ok_or_else(|| DNNError::ComputationError(
                        "Failed to compute mean along axis 1".to_string()
                    ))?,
                    _ => return Err(DNNError::InvalidConfiguration(
                        format!("Unsupported axis: {}", axis)
                    )),
                }
            }
            ConcurrentAxisOp::Max => {
                let result_shape = match axis {
                    0 => data_lock.ncols(),
                    1 => data_lock.nrows(),
                    _ => return Err(DNNError::InvalidConfiguration(
                        format!("Unsupported axis: {}", axis)
                    )),
                };
                
                let mut result = Array1::<f32>::zeros(result_shape);
                
                match axis {
                    0 => {
                        for (col_idx, mut col) in result.iter_mut().enumerate() {
                            *col = data_lock.column(col_idx).fold(Float::neg_infinity(), |acc, &x| Float::max(acc, x));
                        }
                    }
                    1 => {
                        for (row_idx, mut row) in result.iter_mut().enumerate() {
                            *row = data_lock.row(row_idx).fold(Float::neg_infinity(), |acc, &x| Float::max(acc, x));
                        }
                    }
                    _ => unreachable!(),
                }
                
                result
            }
        };
        
        Ok(result)
    }
    
    /// Create multiple concurrent model instances for load balancing
    pub fn create_model_pool(
        base_model: &ZenDNNModel,
        pool_size: usize,
    ) -> Result<Vec<ConcurrentDNNModel>, DNNError> {
        let mut model_pool = Vec::with_capacity(pool_size);
        
        for _ in 0..pool_size {
            let model_clone = base_model.clone();
            let concurrent_model = ConcurrentDNNModel::new(model_clone)?;
            model_pool.push(concurrent_model);
        }
        
        Ok(model_pool)
    }
    
    /// Distribute inference workload across model pool
    pub async fn distributed_inference(
        model_pool: &[ConcurrentDNNModel],
        inputs: Vec<DNNTensor>,
        mode: DNNTrainingMode,
    ) -> Result<Vec<DNNTensor>, DNNError> {
        if model_pool.is_empty() {
            return Err(DNNError::InvalidConfiguration("Model pool is empty".to_string()));
        }
        
        let pool_size = model_pool.len();
        let mut results = Vec::with_capacity(inputs.len());
        
        #[cfg(feature = "parallel")]
        {
            use std::sync::Arc;
            
            let results_arc = Arc::new(Mutex::new(vec![None; inputs.len()]));
            let model_pool_arc = Arc::new(model_pool);
            
            inputs
                .par_iter()
                .enumerate()
                .for_each(|(idx, input)| {
                    let model_idx = idx % pool_size;
                    let model = &model_pool_arc[model_idx];
                    let results_ref = Arc::clone(&results_arc);
                    
                    // Each thread gets its own runtime for async operations
                    let runtime = tokio::runtime::Runtime::new().unwrap();
                    let result = runtime.block_on(model.concurrent_forward(input, mode));
                    
                    let mut results_guard = results_ref.lock().unwrap();
                    results_guard[idx] = Some(result);
                });
            
            // Collect results
            let results_guard = results_arc.lock().unwrap();
            for result_opt in results_guard.iter() {
                match result_opt {
                    Some(Ok(tensor)) => results.push(tensor.clone()),
                    Some(Err(e)) => return Err(e.clone()),
                    None => return Err(DNNError::ComputationError("Missing result".to_string())),
                }
            }
        }
        
        #[cfg(not(feature = "parallel"))]
        {
            // Sequential fallback
            for (idx, input) in inputs.iter().enumerate() {
                let model_idx = idx % pool_size;
                let model = &model_pool[model_idx];
                let result = model.concurrent_forward(input, mode).await?;
                results.push(result);
            }
        }
        
        Ok(results)
    }
    
    /// Monitor concurrent performance across model pool
    pub fn aggregate_pool_metrics(model_pool: &[ConcurrentDNNModel]) -> ConcurrentMetrics {
        let mut aggregated = ConcurrentMetrics::default();
        
        for model in model_pool {
            let metrics = model.get_concurrent_metrics();
            
            aggregated.total_inference_requests += metrics.total_inference_requests;
            aggregated.successful_inferences += metrics.successful_inferences;
            aggregated.failed_inferences += metrics.failed_inferences;
            aggregated.peak_concurrent_requests = aggregated.peak_concurrent_requests.max(metrics.peak_concurrent_requests);
            aggregated.total_contention_time_ms += metrics.total_contention_time_ms;
            
            // Aggregate lock statistics
            aggregated.lock_stats.read_locks_acquired += metrics.lock_stats.read_locks_acquired;
            aggregated.lock_stats.write_locks_acquired += metrics.lock_stats.write_locks_acquired;
            aggregated.lock_stats.lock_contentions += metrics.lock_stats.lock_contentions;
        }
        
        // Compute averages
        let model_count = model_pool.len() as f64;
        if model_count > 0.0 {
            aggregated.avg_processing_time_ms = model_pool.iter()
                .map(|m| m.get_concurrent_metrics().avg_processing_time_ms)
                .sum::<f64>() / model_count;
                
            aggregated.lock_stats.avg_read_lock_time_ms = model_pool.iter()
                .map(|m| m.get_concurrent_metrics().lock_stats.avg_read_lock_time_ms)
                .sum::<f64>() / model_count;
                
            aggregated.lock_stats.avg_write_lock_time_ms = model_pool.iter()
                .map(|m| m.get_concurrent_metrics().lock_stats.avg_write_lock_time_ms)
                .sum::<f64>() / model_count;
        }
        
        aggregated
    }
}

// === TESTS ===

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dnn::{ZenDNNModel, DNNConfig, ActivationType};
    use crate::dnn::data::{TensorShape, DNNTensor};
    use std::time::Duration;
    
    #[tokio::test]
    async fn test_concurrent_model_creation() {
        let config = DNNConfig {
            input_dim: 10,
            output_dim: 5,
            hidden_layers: vec![20, 15],
            activation: ActivationType::ReLU,
            ..Default::default()
        };
        
        let mut model = ZenDNNModel::with_config(config).unwrap();
        model.compile().unwrap();
        
        let concurrent_model = ConcurrentDNNModel::new(model).unwrap();
        
        // Test basic functionality
        assert!(concurrent_model.inference_config.enable_concurrent_serving);
        assert!(concurrent_model.inference_config.max_inference_threads > 0);
    }
    
    #[tokio::test]
    async fn test_concurrent_forward_pass() {
        let config = DNNConfig {
            input_dim: 4,
            output_dim: 2,
            hidden_layers: vec![8],
            activation: ActivationType::Linear,
            ..Default::default()
        };
        
        let mut model = ZenDNNModel::with_config(config).unwrap();
        model.compile().unwrap();
        
        let concurrent_model = ConcurrentDNNModel::new(model).unwrap();
        
        // Create test input
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5];
        let input_shape = TensorShape::new_2d(2, 4);
        let input = DNNTensor::from_vec(input_data, &input_shape).unwrap();
        
        // Test concurrent forward pass
        let output = concurrent_model.concurrent_forward(&input, DNNTrainingMode::Inference).await.unwrap();
        
        assert_eq!(output.batch_size(), 2);
        assert_eq!(output.feature_dim(), 2);
        assert!(!output.has_invalid_values());
    }
    
    #[tokio::test]
    async fn test_parallel_batch_processing() {
        let config = DNNConfig {
            input_dim: 3,
            output_dim: 1,
            hidden_layers: vec![5],
            activation: ActivationType::ReLU,
            ..Default::default()
        };
        
        let mut model = ZenDNNModel::with_config(config).unwrap();
        model.compile().unwrap();
        
        let concurrent_config = ConcurrentInferenceConfig {
            parallel_batch_size: 4, // Force parallel processing for batch > 4
            ..Default::default()
        };
        
        let concurrent_model = ConcurrentDNNModel::with_config(model, concurrent_config).unwrap();
        
        // Create large batch (8 samples)
        let large_batch_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let input_shape = TensorShape::new_2d(8, 3);
        let large_input = DNNTensor::from_vec(large_batch_data, &input_shape).unwrap();
        
        // Test parallel batch processing
        let output = concurrent_model.concurrent_forward(&large_input, DNNTrainingMode::Inference).await.unwrap();
        
        assert_eq!(output.batch_size(), 8);
        assert_eq!(output.feature_dim(), 1);
        assert!(!output.has_invalid_values());
    }
    
    #[tokio::test]
    async fn test_concurrent_parameter_update() {
        let config = DNNConfig {
            input_dim: 2,
            output_dim: 1,
            hidden_layers: vec![3],
            activation: ActivationType::Linear,
            ..Default::default()
        };
        
        let mut model = ZenDNNModel::with_config(config).unwrap();
        model.compile().unwrap();
        
        let concurrent_model = ConcurrentDNNModel::new(model).unwrap();
        
        // Test parameter update with write lock
        let result = concurrent_model.concurrent_parameter_update(|model| {
            // Simulate parameter update
            model.metadata.insert("test_update".to_string(), "completed".to_string());
            Ok(())
        }).await;
        
        assert!(result.is_ok());
        
        // Verify update was applied
        {
            let model_guard = concurrent_model.model.read().unwrap();
            assert_eq!(model_guard.metadata.get("test_update"), Some(&"completed".to_string()));
        }
    }
    
    #[test]
    fn test_model_pool_creation() {
        let config = DNNConfig {
            input_dim: 5,
            output_dim: 3,
            hidden_layers: vec![10],
            activation: ActivationType::ReLU,
            ..Default::default()
        };
        
        let mut base_model = ZenDNNModel::with_config(config).unwrap();
        base_model.compile().unwrap();
        
        let pool_size = 4;
        let model_pool = ConcurrentUtils::create_model_pool(&base_model, pool_size).unwrap();
        
        assert_eq!(model_pool.len(), pool_size);
    }
    
    #[test]
    fn test_concurrent_metrics() {
        let config = DNNConfig {
            input_dim: 2,
            output_dim: 1,
            hidden_layers: vec![4],
            activation: ActivationType::Linear,
            ..Default::default()
        };
        
        let mut model = ZenDNNModel::with_config(config).unwrap();
        model.compile().unwrap();
        
        let concurrent_model = ConcurrentDNNModel::new(model).unwrap();
        
        // Get initial metrics
        let initial_metrics = concurrent_model.get_concurrent_metrics();
        assert_eq!(initial_metrics.total_inference_requests, 0);
        assert_eq!(initial_metrics.successful_inferences, 0);
        
        // Reset metrics
        concurrent_model.reset_metrics();
        let reset_metrics = concurrent_model.get_concurrent_metrics();
        assert_eq!(reset_metrics.total_inference_requests, 0);
    }
    
    #[tokio::test]
    async fn test_concurrent_layer_wrapper() {
        use crate::dnn::layers::{DenseLayer, DNNLayer};
        use crate::dnn::ActivationType;
        
        let layer = DenseLayer::new(5, ActivationType::Linear, true).unwrap();
        let mut boxed_layer: Box<dyn DNNLayer> = Box::new(layer);
        boxed_layer.compile(3).unwrap();
        
        let concurrent_layer = ConcurrentLayer::new(boxed_layer, "test_layer".to_string());
        
        // Create test input
        let input_data = vec![1.0, 2.0, 3.0];
        let input_shape = TensorShape::new_2d(1, 3);
        let input = DNNTensor::from_vec(input_data, &input_shape).unwrap();
        
        // Test concurrent forward pass
        let output = concurrent_layer.concurrent_forward(&input, DNNTrainingMode::Inference).await.unwrap();
        
        assert_eq!(output.batch_size(), 1);
        assert_eq!(output.feature_dim(), 5);
        
        // Check metadata
        let metadata = concurrent_layer.get_metadata();
        assert_eq!(metadata.name, "test_layer");
        assert_eq!(metadata.concurrent_forwards, 1);
    }
}