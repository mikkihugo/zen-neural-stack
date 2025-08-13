#!/usr/bin/env cargo +stable -Zscript
//! Test script for agent implementation

use std::time::Duration;
use tokio;

// Import the zen-swarm-core types
use zen_swarm_core::{
    agent::{DynamicAgent, AgentStatus},
    task::{Task, TaskPayload, TaskPriority, TaskId},
    async_swarm::{AsyncSwarm, AsyncSwarmConfig},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing Agent Implementation");
    
    // Test 1: Create a dynamic agent
    println!("\n1. Creating Dynamic Agent...");
    let mut agent = DynamicAgent::new("test-agent-1", vec!["compute".to_string(), "text".to_string()]);
    println!("   ✅ Agent created with ID: {}", agent.id());
    println!("   ✅ Agent capabilities: {:?}", agent.capabilities());
    println!("   ✅ Agent status: {:?}", agent.status());
    
    // Test 2: Create a task
    println!("\n2. Creating Task...");
    let task = Task::new("task-1", "compute")
        .with_priority(TaskPriority::Normal)
        .with_payload(TaskPayload::Text("Hello, world!".to_string()))
        .require_capability("compute");
    
    println!("   ✅ Task created with ID: {}", task.id);
    println!("   ✅ Task type: {}", task.task_type);
    println!("   ✅ Task capabilities: {:?}", task.required_capabilities);
    
    // Test 3: Check if agent can handle task
    println!("\n3. Checking Task Compatibility...");
    let can_handle = agent.can_handle(&task);
    println!("   ✅ Agent can handle task: {}", can_handle);
    
    // Test 4: Process task
    println!("\n4. Processing Task...");
    let start_time = std::time::Instant::now();
    
    match agent.process_task(task).await {
        Ok(result) => {
            let processing_time = start_time.elapsed();
            println!("   ✅ Task processed successfully!");
            println!("   ✅ Result status: {:?}", result.status);
            println!("   ✅ Execution time: {}ms", result.execution_time_ms);
            println!("   ✅ Wall clock time: {}ms", processing_time.as_millis());
            
            if let Some(output) = result.output {
                println!("   ✅ Output: {:?}", output);
            }
        },
        Err(e) => {
            println!("   ❌ Task processing failed: {}", e);
            return Err(e.into());
        }
    }
    
    // Test 5: Check agent metadata after processing
    println!("\n5. Checking Agent Metrics...");
    let metadata = agent.metadata();
    println!("   ✅ Tasks processed: {}", metadata.tasks_processed);
    println!("   ✅ Tasks succeeded: {}", metadata.tasks_succeeded);
    println!("   ✅ Tasks failed: {}", metadata.tasks_failed);
    println!("   ✅ Average processing time: {:.2}ms", metadata.avg_processing_time_ms);
    println!("   ✅ Final agent status: {:?}", agent.status());
    
    // Test 6: Create async swarm and test integration
    println!("\n6. Testing Async Swarm Integration...");
    let config = AsyncSwarmConfig::default();
    let mut swarm = AsyncSwarm::new(config);
    
    // Add agent to swarm
    let agent_id = swarm.add_agent(agent).await?;
    println!("   ✅ Agent added to swarm with ID: {}", agent_id);
    
    // Add task to swarm
    let task2 = Task::new("task-2", "compute")
        .with_payload(TaskPayload::Text("Swarm test".to_string()))
        .require_capability("compute");
    
    swarm.add_task(task2).await?;
    println!("   ✅ Task added to swarm");
    
    // Check swarm metrics
    let metrics = swarm.metrics().await;
    println!("   ✅ Swarm agents: {}", metrics.active_agents);
    println!("   ✅ Swarm queued tasks: {}", metrics.queued_tasks);
    
    // Process tasks
    println!("\n7. Processing Tasks in Swarm...");
    let results = swarm.process_tasks_concurrently(2).await?;
    println!("   ✅ Processed {} tasks", results.len());
    
    for (task_id, result) in results {
        match result {
            Ok(_) => println!("   ✅ Task {} completed successfully", task_id),
            Err(e) => println!("   ❌ Task {} failed: {}", task_id, e),
        }
    }
    
    println!("\n🎉 All tests completed successfully!");
    Ok(())
}