/**
 * @file webgpu/shaders/gnn_node_update.wgsl
 * @brief WebGPU Compute Shader for Graph Neural Network Node Updates
 * 
 * This compute shader implements efficient parallel node update operations for
 * graph neural networks, including GRU-style gated updates, residual connections,
 * and various activation functions.
 * 
 * ## Algorithm Overview
 * 
 * For each node:
 * 1. Load current node representation
 * 2. Load aggregated messages from neighbors
 * 3. Apply gated update mechanism (GRU-style)
 * 4. Compute new node representation
 * 5. Apply activation function
 * 
 * ## Update Mechanisms
 * 
 * - **GRU Update**: Gated recurrent unit style updates with reset and update gates
 * - **Residual Update**: Skip connections for gradient flow
 * - **Simple Update**: Direct replacement or addition
 * - **Attention Update**: Context-aware updates (future extension)
 * 
 * ## Performance Features
 * 
 * - Vectorized operations for SIMD efficiency
 * - Fused activation function application
 * - Memory-efficient in-place updates where possible
 * - Support for different hidden dimensions
 * 
 * @author GPU Acceleration Expert (ruv-swarm)
 * @version 1.0.0
 * @since 2024-08-11
 */

// Optimal workgroup size for node update operations
override WORKGROUP_SIZE: u32 = 256u;

// === BIND GROUPS AND UNIFORMS ===

@group(0) @binding(0) var<storage, read> current_features: array<f32>;
@group(0) @binding(1) var<storage, read> aggregated_messages: array<f32>;
@group(0) @binding(2) var<storage, read_write> updated_features: array<f32>;
@group(0) @binding(3) var<uniform> params: GpuKernelParams;

// === DATA STRUCTURES ===

struct GpuKernelParams {
    aggregation_method: u32,
    activation_function: u32,
    dropout_rate: f32,
    message_passing_steps: u32,
    hidden_dim: u32,
    output_dim: u32,
    random_seed: u32,
    _padding: u32,
}

// === LEARNED PARAMETERS (WOULD BE LOADED FROM BUFFERS) ===

// For demonstration, we'll use compile-time constants
// In practice, these would be loaded from additional storage buffers
const WEIGHT_SCALE: f32 = 0.1;
const BIAS_INIT: f32 = 0.0;

// === UTILITY FUNCTIONS ===

/// Load node features as vec4 for SIMD processing
fn load_node_features(node_idx: u32, feature_dim: u32) -> vec4<f32> {
    let base_idx = node_idx * feature_dim;
    
    if (base_idx >= arrayLength(&current_features)) {
        return vec4<f32>(0.0);
    }
    
    var features = vec4<f32>(0.0);
    let max_load = min(4u, feature_dim);
    
    for (var i = 0u; i < max_load; i++) {
        if (base_idx + i < arrayLength(&current_features)) {
            features[i] = current_features[base_idx + i];
        }
    }
    
    return features;
}

/// Load aggregated messages as vec4
fn load_aggregated_message(node_idx: u32, hidden_dim: u32) -> vec4<f32> {
    let base_idx = node_idx * hidden_dim;
    
    if (base_idx >= arrayLength(&aggregated_messages)) {
        return vec4<f32>(0.0);
    }
    
    var message = vec4<f32>(0.0);
    let max_load = min(4u, hidden_dim);
    
    for (var i = 0u; i < max_load; i++) {
        if (base_idx + i < arrayLength(&aggregated_messages)) {
            message[i] = aggregated_messages[base_idx + i];
        }
    }
    
    return message;
}

/// Store updated features to global memory
fn store_updated_features(node_idx: u32, features: vec4<f32>, feature_dim: u32) {
    let base_idx = node_idx * feature_dim;
    let max_store = min(4u, feature_dim);
    
    for (var i = 0u; i < max_store; i++) {
        if (base_idx + i < arrayLength(&updated_features)) {
            updated_features[base_idx + i] = features[i];
        }
    }
}

// === ACTIVATION FUNCTIONS ===

/// Apply activation function with optimized implementations
fn apply_activation(x: vec4<f32>, activation_type: u32) -> vec4<f32> {
    switch (activation_type) {
        case 0u: { // ReLU
            return max(x, vec4<f32>(0.0));
        }
        case 1u: { // Tanh
            return tanh(x);
        }
        case 2u: { // Sigmoid
            return 1.0 / (1.0 + exp(-x));
        }
        case 3u: { // LeakyReLU (alpha=0.01)
            return select(0.01 * x, x, x > vec4<f32>(0.0));
        }
        case 4u: { // GELU (approximation)
            return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
        }
        case 5u: { // Swish/SiLU
            return x * (1.0 / (1.0 + exp(-x)));
        }
        default: { // Linear (no activation)
            return x;
        }
    }
}

// === WEIGHT INITIALIZATION FUNCTIONS ===

/// Generate pseudo-random weights using node index as seed
fn generate_weight(node_idx: u32, param_idx: u32) -> f32 {
    let seed = node_idx * 1637u + param_idx * 1013u;
    let random = f32((seed * 1664525u + 1013904223u) % 0xFFFFFFFFu) / f32(0xFFFFFFFFu);
    return (random - 0.5) * 2.0 * WEIGHT_SCALE;
}

/// Generate bias values
fn generate_bias(node_idx: u32, param_idx: u32) -> f32 {
    return BIAS_INIT;
}

// === NODE UPDATE MECHANISMS ===

/**
 * GRU-style gated update mechanism.
 * 
 * Implements a simplified GRU update:
 * - Reset gate: r = σ(W_r * [h, m] + b_r)
 * - Update gate: z = σ(W_z * [h, m] + b_z)
 * - Candidate: h_tilde = tanh(W_h * [r ⊙ h, m] + b_h)
 * - Output: h_new = (1 - z) ⊙ h + z ⊙ h_tilde
 * 
 * Where h is current features, m is aggregated message
 */
fn gru_update(current: vec4<f32>, message: vec4<f32>, node_idx: u32) -> vec4<f32> {
    // Concatenate current features and message (simplified)
    let combined_input = current + message; // In practice, this would be proper concatenation
    
    // Reset gate computation
    // r = σ(W_r * combined_input + b_r)
    var reset_gate = combined_input;
    for (var i = 0; i < 4; i++) {
        let weight_r = generate_weight(node_idx, i + 0u);
        let bias_r = generate_bias(node_idx, i + 0u);
        reset_gate[i] = 1.0 / (1.0 + exp(-(weight_r * reset_gate[i] + bias_r)));
    }
    
    // Update gate computation  
    // z = σ(W_z * combined_input + b_z)
    var update_gate = combined_input;
    for (var i = 0; i < 4; i++) {
        let weight_z = generate_weight(node_idx, i + 4u);
        let bias_z = generate_bias(node_idx, i + 4u);
        update_gate[i] = 1.0 / (1.0 + exp(-(weight_z * update_gate[i] + bias_z)));
    }
    
    // Candidate hidden state
    // h_tilde = tanh(W_h * [r ⊙ h, m] + b_h)
    let reset_current = reset_gate * current;
    let candidate_input = reset_current + message;
    
    var candidate = candidate_input;
    for (var i = 0; i < 4; i++) {
        let weight_h = generate_weight(node_idx, i + 8u);
        let bias_h = generate_bias(node_idx, i + 8u);
        candidate[i] = tanh(weight_h * candidate[i] + bias_h);
    }
    
    // Final update: h_new = (1 - z) ⊙ h + z ⊙ h_tilde
    let one_minus_z = vec4<f32>(1.0) - update_gate;
    return one_minus_z * current + update_gate * candidate;
}

/**
 * Residual update mechanism with skip connections.
 * 
 * Computes: h_new = h + W * m + b
 * Where the original features are preserved via skip connection
 */
fn residual_update(current: vec4<f32>, message: vec4<f32>, node_idx: u32) -> vec4<f32> {
    // Apply learned transformation to message
    var transformed_message = message;
    for (var i = 0; i < 4; i++) {
        let weight = generate_weight(node_idx, i);
        let bias = generate_bias(node_idx, i);
        transformed_message[i] = weight * transformed_message[i] + bias;
    }
    
    // Add to current features (residual connection)
    return current + transformed_message;
}

/**
 * Simple additive update mechanism.
 * 
 * Computes: h_new = h + α * m
 * Where α is a learnable scaling factor
 */
fn simple_update(current: vec4<f32>, message: vec4<f32>, node_idx: u32) -> vec4<f32> {
    let alpha = generate_weight(node_idx, 0u);
    return current + alpha * message;
}

/**
 * Multiplicative update mechanism.
 * 
 * Computes: h_new = h ⊙ (1 + W * m + b)
 * Allows for feature-wise scaling based on messages
 */
fn multiplicative_update(current: vec4<f32>, message: vec4<f32>, node_idx: u32) -> vec4<f32> {
    var scaling_factor = vec4<f32>(1.0);
    
    for (var i = 0; i < 4; i++) {
        let weight = generate_weight(node_idx, i);
        let bias = generate_bias(node_idx, i);
        scaling_factor[i] += weight * message[i] + bias;
    }
    
    return current * scaling_factor;
}

/**
 * Attention-based update mechanism (placeholder).
 * 
 * Would implement: h_new = attention(h, m) * h + (1 - attention(h, m)) * f(m)
 * Where attention weights are learned based on current state and message
 */
fn attention_update(current: vec4<f32>, message: vec4<f32>, node_idx: u32) -> vec4<f32> {
    // Placeholder - would require attention weight computation
    // For now, fall back to GRU update
    return gru_update(current, message, node_idx);
}

// === NORMALIZATION FUNCTIONS ===

/// Layer normalization (simplified version)
fn layer_normalize(x: vec4<f32>) -> vec4<f32> {
    let mean = (x.x + x.y + x.z + x.w) / 4.0;
    let variance = ((x.x - mean) * (x.x - mean) + 
                   (x.y - mean) * (x.y - mean) + 
                   (x.z - mean) * (x.z - mean) + 
                   (x.w - mean) * (x.w - mean)) / 4.0;
    let std_dev = sqrt(variance + 1e-6);
    
    return (x - vec4<f32>(mean)) / vec4<f32>(std_dev);
}

/// Batch normalization (would require running statistics)
fn batch_normalize(x: vec4<f32>) -> vec4<f32> {
    // Placeholder - would require batch statistics
    return layer_normalize(x);
}

// === DROPOUT FUNCTIONS ===

/// Generate pseudo-random number for dropout
fn lcg_random(seed: ptr<function, u32>) -> f32 {
    *seed = (*seed * 1664525u) + 1013904223u;
    return f32(*seed) / f32(0xFFFFFFFFu);
}

/// Apply dropout to updated features
fn apply_dropout(features: vec4<f32>, dropout_rate: f32, seed: ptr<function, u32>) -> vec4<f32> {
    if (dropout_rate <= 0.0) {
        return features;
    }
    
    let keep_prob = 1.0 - dropout_rate;
    let scale = 1.0 / keep_prob;
    
    var mask = vec4<f32>(1.0);
    for (var i = 0; i < 4; i++) {
        if (lcg_random(seed) >= keep_prob) {
            mask[i] = 0.0;
        } else {
            mask[i] = scale;
        }
    }
    
    return features * mask;
}

// === MAIN COMPUTE SHADERS ===

/**
 * Main node update compute kernel.
 * 
 * Each thread processes one node, applying the configured update
 * mechanism to combine current features with aggregated messages.
 * 
 * ## Thread Mapping
 * - global_id.x: Node index to process
 * - Each thread handles one node independently
 * 
 * ## Algorithm Steps
 * 1. Load current node features
 * 2. Load aggregated messages from neighbors
 * 3. Apply update mechanism (GRU/Residual/Simple)
 * 4. Apply activation function
 * 5. Apply dropout if training
 * 6. Store updated features
 */
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;
    
    // Calculate number of nodes based on buffer size
    let num_nodes = arrayLength(&current_features) / params.hidden_dim;
    
    if (node_idx >= num_nodes) {
        return;
    }
    
    // Load current node features and aggregated messages
    let current_features_vec = load_node_features(node_idx, params.hidden_dim);
    let aggregated_message = load_aggregated_message(node_idx, params.hidden_dim);
    
    // Apply node update mechanism
    // For this implementation, we'll use GRU-style updates as default
    var updated_features_vec = gru_update(current_features_vec, aggregated_message, node_idx);
    
    // Apply layer normalization (optional)
    updated_features_vec = layer_normalize(updated_features_vec);
    
    // Apply activation function
    updated_features_vec = apply_activation(updated_features_vec, params.activation_function);
    
    // Apply dropout during training
    if (params.dropout_rate > 0.0) {
        var random_seed = params.random_seed + node_idx;
        updated_features_vec = apply_dropout(updated_features_vec, params.dropout_rate, &random_seed);
    }
    
    // Store the updated features
    store_updated_features(node_idx, updated_features_vec, params.hidden_dim);
}

/**
 * Residual update variant.
 * 
 * Uses residual connections for improved gradient flow,
 * especially beneficial in deep graph networks.
 */
@compute @workgroup_size(WORKGROUP_SIZE)
fn main_residual(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;
    let num_nodes = arrayLength(&current_features) / params.hidden_dim;
    
    if (node_idx >= num_nodes) {
        return;
    }
    
    let current_features_vec = load_node_features(node_idx, params.hidden_dim);
    let aggregated_message = load_aggregated_message(node_idx, params.hidden_dim);
    
    // Use residual update instead of GRU
    var updated_features_vec = residual_update(current_features_vec, aggregated_message, node_idx);
    
    // Apply normalization and activation
    updated_features_vec = layer_normalize(updated_features_vec);
    updated_features_vec = apply_activation(updated_features_vec, params.activation_function);
    
    // Apply dropout if training
    if (params.dropout_rate > 0.0) {
        var random_seed = params.random_seed + node_idx;
        updated_features_vec = apply_dropout(updated_features_vec, params.dropout_rate, &random_seed);
    }
    
    store_updated_features(node_idx, updated_features_vec, params.hidden_dim);
}

/**
 * Simple update variant for baseline comparisons.
 * 
 * Uses simple additive updates without gating mechanisms,
 * useful for ablation studies and baseline comparisons.
 */
@compute @workgroup_size(WORKGROUP_SIZE)
fn main_simple(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;
    let num_nodes = arrayLength(&current_features) / params.hidden_dim;
    
    if (node_idx >= num_nodes) {
        return;
    }
    
    let current_features_vec = load_node_features(node_idx, params.hidden_dim);
    let aggregated_message = load_aggregated_message(node_idx, params.hidden_dim);
    
    // Use simple additive update
    var updated_features_vec = simple_update(current_features_vec, aggregated_message, node_idx);
    
    // Apply activation and dropout
    updated_features_vec = apply_activation(updated_features_vec, params.activation_function);
    
    if (params.dropout_rate > 0.0) {
        var random_seed = params.random_seed + node_idx;
        updated_features_vec = apply_dropout(updated_features_vec, params.dropout_rate, &random_seed);
    }
    
    store_updated_features(node_idx, updated_features_vec, params.hidden_dim);
}

/**
 * Memory-optimized update for very large graphs.
 * 
 * Processes features in smaller chunks to reduce memory
 * bandwidth requirements for graphs with high-dimensional features.
 */
@compute @workgroup_size(WORKGROUP_SIZE)
fn main_memory_optimized(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;
    let num_nodes = arrayLength(&current_features) / params.hidden_dim;
    
    if (node_idx >= num_nodes) {
        return;
    }
    
    // Process features in chunks of 4 (vec4) for SIMD efficiency
    let num_chunks = (params.hidden_dim + 3u) / 4u;
    
    var random_seed = params.random_seed + node_idx;
    
    for (var chunk = 0u; chunk < num_chunks; chunk++) {
        let chunk_offset = chunk * 4u;
        let remaining_features = params.hidden_dim - chunk_offset;
        let chunk_size = min(4u, remaining_features);
        
        // Load chunk of current features
        let current_chunk = load_node_features_chunk(node_idx, chunk_offset, chunk_size);
        let message_chunk = load_aggregated_message_chunk(node_idx, chunk_offset, chunk_size);
        
        // Apply update to this chunk
        var updated_chunk = gru_update_chunk(current_chunk, message_chunk, node_idx, chunk);
        
        // Apply normalization, activation, and dropout to chunk
        updated_chunk = apply_activation(updated_chunk, params.activation_function);
        
        if (params.dropout_rate > 0.0) {
            updated_chunk = apply_dropout(updated_chunk, params.dropout_rate, &random_seed);
        }
        
        // Store updated chunk
        store_updated_features_chunk(node_idx, chunk_offset, updated_chunk, chunk_size);
    }
}

/// Helper function for chunked processing
fn load_node_features_chunk(node_idx: u32, offset: u32, size: u32) -> vec4<f32> {
    let base_idx = node_idx * params.hidden_dim + offset;
    
    var chunk = vec4<f32>(0.0);
    for (var i = 0u; i < size && i < 4u; i++) {
        if (base_idx + i < arrayLength(&current_features)) {
            chunk[i] = current_features[base_idx + i];
        }
    }
    
    return chunk;
}

/// Helper function for chunked message loading
fn load_aggregated_message_chunk(node_idx: u32, offset: u32, size: u32) -> vec4<f32> {
    let base_idx = node_idx * params.hidden_dim + offset;
    
    var chunk = vec4<f32>(0.0);
    for (var i = 0u; i < size && i < 4u; i++) {
        if (base_idx + i < arrayLength(&aggregated_messages)) {
            chunk[i] = aggregated_messages[base_idx + i];
        }
    }
    
    return chunk;
}

/// Helper function for chunked GRU update
fn gru_update_chunk(current: vec4<f32>, message: vec4<f32>, node_idx: u32, chunk_idx: u32) -> vec4<f32> {
    // Simplified GRU update for chunk
    let combined_input = current + message;
    
    // Reset gate
    var reset_gate = combined_input;
    for (var i = 0; i < 4; i++) {
        let weight_r = generate_weight(node_idx, chunk_idx * 4u + i);
        reset_gate[i] = 1.0 / (1.0 + exp(-(weight_r * reset_gate[i])));
    }
    
    // Update gate
    var update_gate = combined_input;
    for (var i = 0; i < 4; i++) {
        let weight_z = generate_weight(node_idx, chunk_idx * 4u + i + 100u);
        update_gate[i] = 1.0 / (1.0 + exp(-(weight_z * update_gate[i])));
    }
    
    // Candidate state
    let reset_current = reset_gate * current;
    var candidate = reset_current + message;
    for (var i = 0; i < 4; i++) {
        let weight_h = generate_weight(node_idx, chunk_idx * 4u + i + 200u);
        candidate[i] = tanh(weight_h * candidate[i]);
    }
    
    // Final update
    return (vec4<f32>(1.0) - update_gate) * current + update_gate * candidate;
}

/// Helper function for chunked feature storage
fn store_updated_features_chunk(node_idx: u32, offset: u32, features: vec4<f32>, size: u32) {
    let base_idx = node_idx * params.hidden_dim + offset;
    
    for (var i = 0u; i < size && i < 4u; i++) {
        if (base_idx + i < arrayLength(&updated_features)) {
            updated_features[base_idx + i] = features[i];
        }
    }
}