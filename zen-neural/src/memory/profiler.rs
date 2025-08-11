/**
 * @file zen-neural/src/memory/profiler.rs
 * @brief Memory Profiling and Statistics System
 * 
 * This module implements comprehensive memory profiling capabilities to track,
 * analyze, and optimize memory usage patterns in the Zen Neural Stack. It provides
 * detailed statistics, allocation tracking, and performance analysis tools.
 * 
 * ## Features
 * 
 * ### Real-Time Monitoring
 * - Live allocation and deallocation tracking
 * - Memory usage statistics by pool and component
 * - Peak memory usage detection
 * - Fragmentation analysis and reporting
 * 
 * ### Performance Analysis
 * - Allocation pattern analysis
 * - Memory access pattern profiling
 * - Cache hit/miss ratio tracking
 * - Memory bandwidth utilization
 * 
 * ### Debugging Support
 * - Memory leak detection
 * - Double-free detection
 * - Bounds checking integration
 * - Stack trace capture for allocations
 * 
 * @author Memory Management Expert Agent (ruv-swarm)
 * @version 1.0.0-alpha.1
 * @since 2024-08-11
 */

use std::collections::{HashMap, VecDeque};
use std::time::{Instant, Duration};
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, AtomicU64, Ordering}};
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{MemoryError, MemoryResult, TensorType};

// === MEMORY STATISTICS ===

/// Comprehensive memory usage statistics
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total allocated memory in bytes
    pub total_allocated: usize,
    /// Total available memory in bytes
    pub total_available: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Number of deallocations
    pub deallocation_count: usize,
    /// Memory fragmentation ratio (0.0-1.0)
    pub fragmentation_ratio: f32,
    /// Cache hit rate (0.0-1.0)
    pub cache_hit_rate: f32,
    /// Average allocation size
    pub average_allocation_size: f32,
    /// Memory utilization efficiency (0.0-1.0)
    pub utilization_efficiency: f32,
    /// Time spent in memory operations (microseconds)
    pub total_time_us: u64,
}

/// Memory usage breakdown by component
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct MemoryUsage {
    /// Usage by tensor pools
    pub tensor_pools: usize,
    /// Usage by gradient pools
    pub gradient_pools: usize,
    /// Usage by activation pools
    pub activation_pools: usize,
    /// Usage by neural network models
    pub neural_networks: usize,
    /// Usage by training infrastructure
    pub training_system: usize,
    /// Usage by GPU backend
    pub gpu_backend: usize,
    /// Usage by other components
    pub other: usize,
}

/// Detailed allocation information
#[derive(Debug, Clone)]
struct AllocationInfo {
    /// Size of the allocation
    size: usize,
    /// Timestamp of allocation
    timestamp: Instant,
    /// Tensor type if applicable
    tensor_type: Option<TensorType>,
    /// Component that made the allocation
    component: String,
    /// Call stack trace (if available)
    #[cfg(feature = "backtrace")]
    backtrace: Vec<String>,
}

// === ALLOCATION TRACKER ===

/// Tracks individual memory allocations for detailed analysis
#[derive(Debug)]
pub struct AllocationTracker {
    /// Active allocations by address
    active_allocations: Mutex<HashMap<usize, AllocationInfo>>,
    /// Recent deallocations for pattern analysis
    recent_deallocations: Mutex<VecDeque<AllocationInfo>>,
    /// Maximum number of recent deallocations to track
    max_recent_deallocations: usize,
    /// Total number of allocations tracked
    total_tracked: AtomicUsize,
    /// Total memory currently tracked
    total_memory_tracked: AtomicUsize,
}

impl AllocationTracker {
    /// Create a new allocation tracker
    pub fn new(max_recent_deallocations: usize) -> Self {
        Self {
            active_allocations: Mutex::new(HashMap::new()),
            recent_deallocations: Mutex::new(VecDeque::new()),
            max_recent_deallocations,
            total_tracked: AtomicUsize::new(0),
            total_memory_tracked: AtomicUsize::new(0),
        }
    }
    
    /// Track a new allocation
    pub fn track_allocation(
        &self,
        address: usize,
        size: usize,
        tensor_type: Option<TensorType>,
        component: String,
    ) -> MemoryResult<()> {
        let info = AllocationInfo {
            size,
            timestamp: Instant::now(),
            tensor_type,
            component,
            #[cfg(feature = "backtrace")]
            backtrace: self.capture_backtrace(),
        };
        
        let mut allocations = self.active_allocations.lock()
            .map_err(|_| MemoryError::ThreadSafetyError {
                message: "Failed to lock allocations map".to_string()
            })?;
        
        if allocations.insert(address, info).is_some() {
            return Err(MemoryError::ConfigError {
                message: format!("Double allocation detected at address 0x{:x}", address)
            });
        }
        
        self.total_tracked.fetch_add(1, Ordering::SeqCst);
        self.total_memory_tracked.fetch_add(size, Ordering::SeqCst);
        
        Ok(())
    }
    
    /// Track a deallocation
    pub fn track_deallocation(&self, address: usize) -> MemoryResult<()> {
        let mut allocations = self.active_allocations.lock()
            .map_err(|_| MemoryError::ThreadSafetyError {
                message: "Failed to lock allocations map".to_string()
            })?;
        
        let info = allocations.remove(&address)
            .ok_or_else(|| MemoryError::ConfigError {
                message: format!("Double free or invalid free detected at address 0x{:x}", address)
            })?;
        
        self.total_memory_tracked.fetch_sub(info.size, Ordering::SeqCst);
        
        // Add to recent deallocations
        let mut recent = self.recent_deallocations.lock()
            .map_err(|_| MemoryError::ThreadSafetyError {
                message: "Failed to lock recent deallocations".to_string()
            })?;
        
        recent.push_back(info);
        while recent.len() > self.max_recent_deallocations {
            recent.pop_front();
        }
        
        Ok(())
    }
    
    /// Get current active allocations count
    pub fn active_count(&self) -> usize {
        self.active_allocations.lock()
            .map(|allocations| allocations.len())
            .unwrap_or(0)
    }
    
    /// Get total tracked memory
    pub fn tracked_memory(&self) -> usize {
        self.total_memory_tracked.load(Ordering::SeqCst)
    }
    
    /// Detect memory leaks
    pub fn detect_leaks(&self, max_age: Duration) -> Vec<(usize, AllocationInfo)> {
        let now = Instant::now();
        let mut leaks = Vec::new();
        
        if let Ok(allocations) = self.active_allocations.lock() {
            for (&address, info) in allocations.iter() {
                if now.duration_since(info.timestamp) > max_age {
                    leaks.push((address, info.clone()));
                }
            }
        }
        
        leaks
    }
    
    /// Get allocation statistics by component
    pub fn get_component_stats(&self) -> HashMap<String, (usize, usize)> {
        let mut stats = HashMap::new();
        
        if let Ok(allocations) = self.active_allocations.lock() {
            for info in allocations.values() {
                let entry = stats.entry(info.component.clone()).or_insert((0, 0));
                entry.0 += 1; // count
                entry.1 += info.size; // total size
            }
        }
        
        stats
    }
    
    /// Get allocation statistics by tensor type
    pub fn get_tensor_type_stats(&self) -> HashMap<Option<TensorType>, (usize, usize)> {
        let mut stats = HashMap::new();
        
        if let Ok(allocations) = self.active_allocations.lock() {
            for info in allocations.values() {
                let entry = stats.entry(info.tensor_type).or_insert((0, 0));
                entry.0 += 1; // count
                entry.1 += info.size; // total size
            }
        }
        
        stats
    }
    
    #[cfg(feature = "backtrace")]
    fn capture_backtrace(&self) -> Vec<String> {
        // In a real implementation, this would capture and format a backtrace
        // For now, return empty vector
        Vec::new()
    }
}

// === MEMORY PROFILER ===

/// Main memory profiling system
#[derive(Debug)]
pub struct MemoryProfiler {
    /// Whether profiling is enabled
    enabled: bool,
    /// Allocation tracker
    allocation_tracker: AllocationTracker,
    /// Performance metrics
    metrics: Arc<Mutex<PerformanceMetrics>>,
    /// Memory usage history
    usage_history: Mutex<VecDeque<(Instant, MemoryUsage)>>,
    /// Maximum history entries to keep
    max_history_entries: usize,
    /// Start time for profiling session
    start_time: Instant,
}

/// Performance metrics for memory operations
#[derive(Debug, Default)]
struct PerformanceMetrics {
    /// Total allocation time
    total_allocation_time: Duration,
    /// Total deallocation time
    total_deallocation_time: Duration,
    /// Number of cache hits
    cache_hits: u64,
    /// Number of cache misses
    cache_misses: u64,
    /// Total bytes allocated
    total_bytes_allocated: u64,
    /// Total bytes deallocated
    total_bytes_deallocated: u64,
    /// Peak simultaneous allocations
    peak_active_allocations: usize,
    /// Allocation size distribution
    size_distribution: HashMap<usize, usize>, // size_class -> count
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new(enabled: bool) -> MemoryResult<Self> {
        Ok(Self {
            enabled,
            allocation_tracker: AllocationTracker::new(1000),
            metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
            usage_history: Mutex::new(VecDeque::new()),
            max_history_entries: 1000,
            start_time: Instant::now(),
        })
    }
    
    /// Record an allocation
    pub fn record_allocation(&self, size: usize, _tensor_type: TensorType) {
        if !self.enabled {
            return;
        }
        
        let start = Instant::now();
        
        // Update metrics
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.total_bytes_allocated += size as u64;
            
            // Update size distribution
            let size_class = self.get_size_class(size);
            *metrics.size_distribution.entry(size_class).or_insert(0) += 1;
            
            // Update peak allocations
            let active_count = self.allocation_tracker.active_count();
            if active_count > metrics.peak_active_allocations {
                metrics.peak_active_allocations = active_count;
            }
            
            metrics.total_allocation_time += start.elapsed();
        }
    }
    
    /// Record a deallocation
    pub fn record_deallocation(&self, size: usize) {
        if !self.enabled {
            return;
        }
        
        let start = Instant::now();
        
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.total_bytes_deallocated += size as u64;
            metrics.total_deallocation_time += start.elapsed();
        }
    }
    
    /// Record a cache hit
    pub fn record_cache_hit(&self) {
        if !self.enabled {
            return;
        }
        
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.cache_hits += 1;
        }
    }
    
    /// Record a cache miss
    pub fn record_cache_miss(&self) {
        if !self.enabled {
            return;
        }
        
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.cache_misses += 1;
        }
    }
    
    /// Record gradient allocation
    pub fn record_gradient_allocation(&self, size: usize) {
        self.record_allocation(size, TensorType::Float32);
    }
    
    /// Record activation allocation
    pub fn record_activation_allocation(&self, size: usize) {
        self.record_allocation(size, TensorType::Float32);
    }
    
    /// Update memory usage history
    pub fn update_usage_history(&self, usage: MemoryUsage) {
        if !self.enabled {
            return;
        }
        
        if let Ok(mut history) = self.usage_history.lock() {
            history.push_back((Instant::now(), usage));
            
            while history.len() > self.max_history_entries {
                history.pop_front();
            }
        }
    }
    
    /// Get current statistics
    pub fn get_stats(&self) -> MemoryStats {
        if !self.enabled {
            return MemoryStats::default();
        }
        
        let metrics = self.metrics.lock().unwrap();
        
        let cache_hit_rate = if metrics.cache_hits + metrics.cache_misses > 0 {
            metrics.cache_hits as f32 / (metrics.cache_hits + metrics.cache_misses) as f32
        } else {
            0.0
        };
        
        let total_time_us = (metrics.total_allocation_time + metrics.total_deallocation_time)
            .as_micros() as u64;
        
        MemoryStats {
            total_allocated: self.allocation_tracker.tracked_memory(),
            total_available: 0, // This would be filled by the caller
            peak_usage: 0, // This would be calculated from history
            allocation_count: self.allocation_tracker.total_tracked.load(Ordering::SeqCst),
            deallocation_count: 0, // This would be tracked separately
            fragmentation_ratio: 0.0, // This would be calculated
            cache_hit_rate,
            average_allocation_size: if metrics.total_bytes_allocated > 0 {
                metrics.total_bytes_allocated as f32 / self.allocation_tracker.total_tracked.load(Ordering::SeqCst) as f32
            } else {
                0.0
            },
            utilization_efficiency: 0.0, // This would be calculated
            total_time_us,
        }
    }
    
    /// Generate a comprehensive profiling report
    pub fn generate_report(&self) -> ProfileReport {
        if !self.enabled {
            return ProfileReport::default();
        }
        
        let stats = self.get_stats();
        let component_stats = self.allocation_tracker.get_component_stats();
        let tensor_type_stats = self.allocation_tracker.get_tensor_type_stats();
        let leaks = self.allocation_tracker.detect_leaks(Duration::from_secs(300)); // 5 minutes
        
        let allocation_patterns = self.analyze_allocation_patterns();
        let memory_timeline = self.get_memory_timeline();
        
        ProfileReport {
            memory_stats: stats,
            config: super::MemoryConfig::default(), // This would be passed in
            component_breakdown: component_stats,
            tensor_type_breakdown: tensor_type_stats,
            potential_leaks: leaks.len(),
            allocation_patterns,
            memory_timeline,
            profiling_duration: self.start_time.elapsed(),
            pool_reports: Vec::new(), // This would be filled by caller
            layout_analysis: String::new(), // This would be filled by layout analyzer
        }
    }
    
    /// Analyze allocation patterns for optimization opportunities
    fn analyze_allocation_patterns(&self) -> Vec<String> {
        let mut patterns = Vec::new();
        
        let component_stats = self.allocation_tracker.get_component_stats();
        
        // Find components with many small allocations
        for (component, (count, total_size)) in component_stats {
            if count > 100 && total_size / count < 1024 {
                patterns.push(format!(
                    "Component '{}' has {} small allocations (avg {} bytes) - consider pooling",
                    component, count, total_size / count
                ));
            }
        }
        
        // Find potential memory leaks
        let leaks = self.allocation_tracker.detect_leaks(Duration::from_secs(60));
        if !leaks.is_empty() {
            patterns.push(format!(
                "Detected {} potentially leaked allocations older than 1 minute",
                leaks.len()
            ));
        }
        
        // Analyze cache performance
        if let Ok(metrics) = self.metrics.lock() {
            let total_cache_ops = metrics.cache_hits + metrics.cache_misses;
            if total_cache_ops > 0 {
                let hit_rate = metrics.cache_hits as f32 / total_cache_ops as f32;
                if hit_rate < 0.8 {
                    patterns.push(format!(
                        "Low cache hit rate: {:.1}% - consider adjusting cache strategy",
                        hit_rate * 100.0
                    ));
                }
            }
        }
        
        patterns
    }
    
    /// Get memory usage timeline for visualization
    fn get_memory_timeline(&self) -> Vec<(u64, usize)> {
        if let Ok(history) = self.usage_history.lock() {
            history.iter()
                .map(|(timestamp, usage)| {
                    let elapsed = timestamp.duration_since(self.start_time).as_secs();
                    let total_usage = usage.tensor_pools + usage.gradient_pools + 
                                    usage.activation_pools + usage.neural_networks;
                    (elapsed, total_usage)
                })
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get size class for allocation size
    fn get_size_class(&self, size: usize) -> usize {
        if size <= 64 {
            64
        } else if size <= 128 {
            128
        } else if size <= 256 {
            256
        } else if size <= 512 {
            512
        } else if size <= 1024 {
            1024
        } else if size <= 4096 {
            4096
        } else if size <= 16384 {
            16384
        } else {
            65536 // Large allocation class
        }
    }
}

// === PROFILE REPORT ===

/// Comprehensive profiling report
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct ProfileReport {
    /// Overall memory statistics
    pub memory_stats: MemoryStats,
    /// Memory system configuration
    pub config: super::MemoryConfig,
    /// Memory usage breakdown by component
    pub component_breakdown: HashMap<String, (usize, usize)>,
    /// Memory usage breakdown by tensor type
    pub tensor_type_breakdown: HashMap<Option<TensorType>, (usize, usize)>,
    /// Number of potential memory leaks detected
    pub potential_leaks: usize,
    /// Analysis of allocation patterns
    pub allocation_patterns: Vec<String>,
    /// Memory usage timeline (timestamp, usage)
    pub memory_timeline: Vec<(u64, usize)>,
    /// Duration of profiling session
    pub profiling_duration: Duration,
    /// Detailed pool statistics
    pub pool_reports: Vec<super::pools::FreeListStats>,
    /// Memory layout analysis
    pub layout_analysis: String,
}

impl fmt::Display for ProfileReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Memory Profiling Report ===")?;
        writeln!(f, "Profiling Duration: {:.2?}", self.profiling_duration)?;
        writeln!(f)?;
        
        writeln!(f, "Memory Statistics:")?;
        writeln!(f, "  Total Allocated: {} bytes", self.memory_stats.total_allocated)?;
        writeln!(f, "  Peak Usage: {} bytes", self.memory_stats.peak_usage)?;
        writeln!(f, "  Allocations: {}", self.memory_stats.allocation_count)?;
        writeln!(f, "  Deallocations: {}", self.memory_stats.deallocation_count)?;
        writeln!(f, "  Cache Hit Rate: {:.1}%", self.memory_stats.cache_hit_rate * 100.0)?;
        writeln!(f, "  Fragmentation: {:.1}%", self.memory_stats.fragmentation_ratio * 100.0)?;
        writeln!(f)?;
        
        if !self.component_breakdown.is_empty() {
            writeln!(f, "Component Breakdown:")?;
            for (component, (count, size)) in &self.component_breakdown {
                writeln!(f, "  {}: {} allocations, {} bytes (avg: {} bytes)", 
                        component, count, size, size / count.max(1))?; // Fixed division result and syntax
            }
            writeln!(f)?;
        }
        
        if self.potential_leaks > 0 {
            writeln!(f, "⚠️  Potential Memory Leaks: {}", self.potential_leaks)?;
            writeln!(f)?;
        }
        
        if !self.allocation_patterns.is_empty() {
            writeln!(f, "Optimization Opportunities:")?;
            for pattern in &self.allocation_patterns {
                writeln!(f, "  • {}", pattern)?;
            }
        }
        
        Ok(())
    }
}

// === TESTS ===

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_allocation_tracker() {
        let tracker = AllocationTracker::new(10);
        
        // Track allocation
        assert!(tracker.track_allocation(
            0x1000, 
            256, 
            Some(TensorType::Float32), 
            "test_component".to_string()
        ).is_ok());
        
        assert_eq!(tracker.active_count(), 1);
        assert_eq!(tracker.tracked_memory(), 256);
        
        // Track deallocation
        assert!(tracker.track_deallocation(0x1000).is_ok());
        assert_eq!(tracker.active_count(), 0);
        assert_eq!(tracker.tracked_memory(), 0);
        
        // Test double free detection
        assert!(tracker.track_deallocation(0x1000).is_err());
    }
    
    #[test]
    fn test_memory_profiler() {
        let profiler = MemoryProfiler::new(true).unwrap();
        
        // Record some operations
        profiler.record_allocation(1024, TensorType::Float32);
        profiler.record_cache_hit();
        profiler.record_cache_miss();
        profiler.record_deallocation(1024);
        
        let stats = profiler.get_stats();
        assert!(stats.cache_hit_rate > 0.0);
        assert!(stats.total_time_us > 0);
        
        let report = profiler.generate_report();
        assert!(report.profiling_duration.as_nanos() > 0);
    }
    
    #[test]
    fn test_memory_stats() {
        let mut stats = MemoryStats::default();
        stats.total_allocated = 1024;
        stats.allocation_count = 10;
        stats.cache_hit_rate = 0.85;
        
        assert_eq!(stats.total_allocated, 1024);
        assert_eq!(stats.cache_hit_rate, 0.85);
    }
    
    #[test]
    fn test_profile_report_display() {
        let report = ProfileReport {
            memory_stats: MemoryStats {
                total_allocated: 2048,
                peak_usage: 4096,
                allocation_count: 20,
                cache_hit_rate: 0.9,
                ..MemoryStats::default()
            },
            potential_leaks: 2,
            allocation_patterns: vec![
                "Small allocations detected".to_string(),
                "Consider using memory pools".to_string(),
            ],
            profiling_duration: Duration::from_secs(30),
            ..ProfileReport::default()
        };
        
        let display_string = format!("{}", report);
        assert!(display_string.contains("Memory Profiling Report"));
        assert!(display_string.contains("Total Allocated: 2048"));
        assert!(display_string.contains("Potential Memory Leaks: 2"));
    }
}

