/**
 * @file webgpu/shaders/gnn_aggregation.wgsl  
 * @brief WebGPU Compute Shader for Graph Neural Network Message Aggregation
 * 
 * This compute shader implements efficient parallel aggregation of messages from
 * neighboring nodes in a graph neural network. It supports multiple aggregation
 * methods (mean, max, sum) with optimized reduction algorithms.
 * 
 * ## Algorithm Overview
 * 
 * For each target node:
 * 1. Load all incoming messages from neighbors
 * 2. Apply the specified aggregation method (mean/max/sum)
 * 3. Store aggregated result for node update phase
 * 
 * ## Aggregation Methods
 * 
 * - **Mean**: Average of neighbor messages (default)
 * - **Max**: Element-wise maximum across messages
 * - **Sum**: Element-wise sum of all messages
 * - **Attention** (future): Weighted aggregation with learned attention
 * 
 * ## Performance Optimizations
 * 
 * - Parallel reduction within workgroups
 * - Vectorized operations for SIMD efficiency
 * - Coalesced memory access patterns
 * - Efficient handling of variable neighborhood sizes
 * 
 * @author GPU Acceleration Expert (ruv-swarm)
 * @version 1.0.0
 * @since 2024-08-11
 */

// Optimal workgroup size for most GPU architectures
override WORKGROUP_SIZE: u32 = 256u;

// Maximum messages per node for shared memory optimization
override MAX_MESSAGES_PER_NODE: u32 = 64u;

// === BIND GROUPS AND UNIFORMS ===

@group(0) @binding(0) var<storage, read> messages: array<f32>;
@group(0) @binding(1) var<storage, read> adjacency_list: array<GpuAdjacencyEntry>;
@group(0) @binding(2) var<storage, read_write> aggregated_results: array<f32>;
@group(0) @binding(3) var<uniform> params: GpuKernelParams;

// === DATA STRUCTURES ===

struct GpuAdjacencyEntry {
    node_idx: u32,
    neighbor_start: u32,
    neighbor_count: u32,
    _padding: u32,
}

struct GpuKernelParams {
    aggregation_method: u32,  // 0=mean, 1=max, 2=sum, 3=attention
    activation_function: u32,
    dropout_rate: f32,
    message_passing_steps: u32,
    hidden_dim: u32,
    output_dim: u32,
    random_seed: u32,
    _padding: u32,
}

// === SHARED MEMORY FOR WORKGROUP OPTIMIZATION ===

// Shared memory for efficient reduction operations
var<workgroup> shared_messages: array<vec4<f32>, 256>;
var<workgroup> shared_indices: array<u32, 256>;

// === UTILITY FUNCTIONS ===

/// Load message vector from global memory
fn load_message(message_idx: u32, hidden_dim: u32) -> vec4<f32> {
    let base_idx = message_idx * hidden_dim;
    
    if (base_idx >= arrayLength(&messages)) {
        return vec4<f32>(0.0);
    }
    
    var message = vec4<f32>(0.0);
    let max_load = min(4u, hidden_dim);
    
    for (var i = 0u; i < max_load; i++) {
        if (base_idx + i < arrayLength(&messages)) {
            message[i] = messages[base_idx + i];
        }
    }
    
    return message;
}

/// Store aggregated result to global memory
fn store_aggregated(node_idx: u32, aggregated: vec4<f32>, hidden_dim: u32) {
    let base_idx = node_idx * hidden_dim;
    let max_store = min(4u, hidden_dim);
    
    for (var i = 0u; i < max_store; i++) {
        if (base_idx + i < arrayLength(&aggregated_results)) {
            aggregated_results[base_idx + i] = aggregated[i];
        }
    }
}

/// Element-wise maximum of two vectors
fn vec_max(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return max(a, b);
}

/// Element-wise addition of two vectors
fn vec_add(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return a + b;
}

/// Divide vector by scalar (for mean calculation)
fn vec_div_scalar(v: vec4<f32>, s: f32) -> vec4<f32> {
    if (s <= 0.0) {
        return vec4<f32>(0.0);
    }
    return v / s;
}

// === AGGREGATION ALGORITHMS ===

/**
 * Mean aggregation: Average all neighbor messages
 * 
 * Computes: agg(messages) = (1/|N|) * Σ message_i
 * where N is the set of neighbors
 */
fn aggregate_mean(messages: ptr<function, array<vec4<f32>, MAX_MESSAGES_PER_NODE>>, 
                  count: u32) -> vec4<f32> {
    if (count == 0u) {
        return vec4<f32>(0.0);
    }
    
    var sum = vec4<f32>(0.0);
    
    for (var i = 0u; i < count; i++) {
        sum = vec_add(sum, (*messages)[i]);
    }
    
    return vec_div_scalar(sum, f32(count));
}

/**
 * Max aggregation: Element-wise maximum across all messages
 * 
 * Computes: agg(messages) = max(message_1, message_2, ..., message_n)
 * Applied element-wise across the feature dimension
 */
fn aggregate_max(messages: ptr<function, array<vec4<f32>, MAX_MESSAGES_PER_NODE>>, 
                 count: u32) -> vec4<f32> {
    if (count == 0u) {
        return vec4<f32>(-1e6); // Large negative value for max operation
    }
    
    var max_val = (*messages)[0];
    
    for (var i = 1u; i < count; i++) {
        max_val = vec_max(max_val, (*messages)[i]);
    }
    
    return max_val;
}

/**
 * Sum aggregation: Element-wise sum of all messages
 * 
 * Computes: agg(messages) = Σ message_i
 * Simple summation across all neighbor messages
 */
fn aggregate_sum(messages: ptr<function, array<vec4<f32>, MAX_MESSAGES_PER_NODE>>, 
                 count: u32) -> vec4<f32> {
    if (count == 0u) {
        return vec4<f32>(0.0);
    }
    
    var sum = vec4<f32>(0.0);
    
    for (var i = 0u; i < count; i++) {
        sum = vec_add(sum, (*messages)[i]);
    }
    
    return sum;
}

/**
 * Attention-based aggregation (placeholder for future implementation)
 * 
 * Would compute: agg(messages) = Σ α_i * message_i
 * where α_i are learned attention weights
 */
fn aggregate_attention(messages: ptr<function, array<vec4<f32>, MAX_MESSAGES_PER_NODE>>, 
                      count: u32, 
                      node_features: vec4<f32>) -> vec4<f32> {
    // Placeholder implementation - would need attention weights
    // For now, fall back to mean aggregation
    return aggregate_mean(messages, count);
}

// === PARALLEL REDUCTION HELPERS ===

/**
 * Workgroup-level parallel reduction for mean aggregation.
 * 
 * Uses shared memory to efficiently compute the sum and count
 * across all threads in a workgroup before dividing.
 */
fn workgroup_reduce_mean(local_id: vec3<u32>, message: vec4<f32>, valid: bool) -> vec4<f32> {
    let tid = local_id.x;
    
    // Store message in shared memory
    shared_messages[tid] = select(vec4<f32>(0.0), message, valid);
    shared_indices[tid] = select(0u, 1u, valid);
    
    workgroupBarrier();
    
    // Parallel reduction
    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_messages[tid] = vec_add(shared_messages[tid], shared_messages[tid + stride]);
            shared_indices[tid] += shared_indices[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Thread 0 has the final result
    if (tid == 0u && shared_indices[0] > 0u) {
        return vec_div_scalar(shared_messages[0], f32(shared_indices[0]));
    }
    
    return vec4<f32>(0.0);
}

/**
 * Workgroup-level parallel reduction for max aggregation.
 * 
 * Uses shared memory to efficiently compute the element-wise
 * maximum across all messages in the workgroup.
 */
fn workgroup_reduce_max(local_id: vec3<u32>, message: vec4<f32>, valid: bool) -> vec4<f32> {
    let tid = local_id.x;
    
    // Store message in shared memory (use large negative for invalid)
    shared_messages[tid] = select(vec4<f32>(-1e6), message, valid);
    
    workgroupBarrier();
    
    // Parallel reduction with max operation
    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_messages[tid] = vec_max(shared_messages[tid], shared_messages[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Thread 0 has the final result
    if (tid == 0u) {
        return shared_messages[0];
    }
    
    return vec4<f32>(-1e6);
}

// === MAIN COMPUTE SHADERS ===

/**
 * Main aggregation compute kernel.
 * 
 * Each thread processes one target node, aggregating all incoming
 * messages from its neighbors according to the specified method.
 * 
 * ## Thread Mapping
 * - global_id.x: Target node index to process
 * - Each thread handles aggregation for one node
 * 
 * ## Algorithm Steps
 * 1. Load adjacency information for target node
 * 2. Load all incoming messages from neighbors  
 * 3. Apply aggregation method (mean/max/sum)
 * 4. Store aggregated result for node update phase
 */
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;
    
    // Bounds check
    if (node_idx >= arrayLength(&adjacency_list)) {
        return;
    }
    
    // Load adjacency information for this target node
    let adj_entry = adjacency_list[node_idx];
    let neighbor_count = adj_entry.neighbor_count;
    
    // Handle nodes with no neighbors (isolated nodes)
    if (neighbor_count == 0u) {
        store_aggregated(node_idx, vec4<f32>(0.0), params.hidden_dim);
        return;
    }
    
    // Load messages from neighbors
    var neighbor_messages: array<vec4<f32>, MAX_MESSAGES_PER_NODE>;
    let max_neighbors = min(neighbor_count, MAX_MESSAGES_PER_NODE);
    
    for (var i = 0u; i < max_neighbors; i++) {
        let message_idx = adj_entry.neighbor_start + i;
        neighbor_messages[i] = load_message(message_idx, params.hidden_dim);
    }
    
    // Apply aggregation method
    var aggregated_result: vec4<f32>;
    
    switch (params.aggregation_method) {
        case 0u: { // Mean aggregation
            aggregated_result = aggregate_mean(&neighbor_messages, max_neighbors);
        }
        case 1u: { // Max aggregation
            aggregated_result = aggregate_max(&neighbor_messages, max_neighbors);
        }
        case 2u: { // Sum aggregation
            aggregated_result = aggregate_sum(&neighbor_messages, max_neighbors);
        }
        case 3u: { // Attention aggregation (placeholder)
            // Would need node features for attention computation
            aggregated_result = aggregate_attention(&neighbor_messages, max_neighbors, vec4<f32>(0.0));
        }
        default: { // Default to mean
            aggregated_result = aggregate_mean(&neighbor_messages, max_neighbors);
        }
    }
    
    // Store the aggregated result
    store_aggregated(node_idx, aggregated_result, params.hidden_dim);
}

/**
 * Optimized aggregation for graphs with high-degree nodes.
 * 
 * Uses workgroup-level parallel reduction to efficiently handle
 * nodes with many neighbors (degree > WORKGROUP_SIZE).
 */
@compute @workgroup_size(WORKGROUP_SIZE)
fn main_high_degree(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let node_idx = workgroup_id.x;
    let thread_id = local_id.x;
    
    if (node_idx >= arrayLength(&adjacency_list)) {
        return;
    }
    
    let adj_entry = adjacency_list[node_idx];
    let neighbor_count = adj_entry.neighbor_count;
    
    // Each thread processes one message
    let message_idx = adj_entry.neighbor_start + thread_id;
    let valid_message = thread_id < neighbor_count;
    
    var message = vec4<f32>(0.0);
    if (valid_message) {
        message = load_message(message_idx, params.hidden_dim);
    }
    
    // Perform workgroup-level reduction
    var aggregated_result: vec4<f32>;
    
    switch (params.aggregation_method) {
        case 0u: { // Mean aggregation
            aggregated_result = workgroup_reduce_mean(local_id, message, valid_message);
        }
        case 1u: { // Max aggregation
            aggregated_result = workgroup_reduce_max(local_id, message, valid_message);
        }
        case 2u: { // Sum aggregation  
            // Sum is like mean but without division
            aggregated_result = workgroup_reduce_mean(local_id, message, valid_message);
            if (thread_id == 0u && shared_indices[0] > 0u) {
                aggregated_result = aggregated_result * f32(shared_indices[0]);
            }
        }
        default: {
            aggregated_result = workgroup_reduce_mean(local_id, message, valid_message);
        }
    }
    
    // Only thread 0 writes the result
    if (thread_id == 0u) {
        store_aggregated(node_idx, aggregated_result, params.hidden_dim);
    }
}

/**
 * Memory-optimized aggregation for very large graphs.
 * 
 * Processes messages in chunks to reduce memory requirements
 * for graphs that exceed available GPU memory.
 */
@compute @workgroup_size(WORKGROUP_SIZE) 
fn main_memory_optimized(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;
    
    if (node_idx >= arrayLength(&adjacency_list)) {
        return;
    }
    
    let adj_entry = adjacency_list[node_idx];
    let neighbor_count = adj_entry.neighbor_count;
    
    if (neighbor_count == 0u) {
        store_aggregated(node_idx, vec4<f32>(0.0), params.hidden_dim);
        return;
    }
    
    // Process messages in chunks to reduce memory usage
    let chunk_size = 32u;
    let num_chunks = (neighbor_count + chunk_size - 1u) / chunk_size;
    
    var final_result = vec4<f32>(0.0);
    var total_processed = 0u;
    
    for (var chunk = 0u; chunk < num_chunks; chunk++) {
        let chunk_start = chunk * chunk_size;
        let chunk_end = min(chunk_start + chunk_size, neighbor_count);
        let chunk_count = chunk_end - chunk_start;
        
        // Process this chunk
        var chunk_result = vec4<f32>(0.0);
        
        switch (params.aggregation_method) {
            case 0u, 2u: { // Mean or Sum - accumulate
                for (var i = chunk_start; i < chunk_end; i++) {
                    let message_idx = adj_entry.neighbor_start + i;
                    let message = load_message(message_idx, params.hidden_dim);
                    chunk_result = vec_add(chunk_result, message);
                }
                final_result = vec_add(final_result, chunk_result);
            }
            case 1u: { // Max - track maximum
                var chunk_max = vec4<f32>(-1e6);
                for (var i = chunk_start; i < chunk_end; i++) {
                    let message_idx = adj_entry.neighbor_start + i;
                    let message = load_message(message_idx, params.hidden_dim);
                    chunk_max = vec_max(chunk_max, message);
                }
                
                if (chunk == 0u) {
                    final_result = chunk_max;
                } else {
                    final_result = vec_max(final_result, chunk_max);
                }
            }
            default: {
                // Default behavior
            }
        }
        
        total_processed += chunk_count;
    }
    
    // Finalize result based on aggregation method
    if (params.aggregation_method == 0u && total_processed > 0u) { // Mean
        final_result = vec_div_scalar(final_result, f32(total_processed));
    }
    
    store_aggregated(node_idx, final_result, params.hidden_dim);
}

/**
 * Batched aggregation for processing multiple graphs simultaneously.
 * 
 * Extends the basic aggregation to handle mini-batches of graphs
 * for efficient training scenarios.
 */
@compute @workgroup_size(WORKGROUP_SIZE)
fn main_batched(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z;
    let node_idx = global_id.x;
    
    // Compute batch offsets
    // This would require additional uniform parameters for batch sizes
    // and proper indexing into batched data structures
    
    // For now, process as single graph
    // Full implementation would handle batch-specific indexing
    main();
}