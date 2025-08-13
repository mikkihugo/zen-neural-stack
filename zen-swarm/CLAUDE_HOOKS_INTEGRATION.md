# 🪝 Claude Code Hooks Integration with Zen Swarm Daemon

## 🎯 **HOOKS ARCHITECTURE: Repository-Scoped Intelligence**

### **Principle**: ALL Claude Code hooks run in the Swarm Daemon, not THE COLLECTIVE

```
Claude Code ←→ Swarm Daemon (Repository Hooks)
                      ↓
                Intelligence Reports  
                      ↓
              THE COLLECTIVE (Pattern Sharing Only)
```

## 🔧 **HOOK RESPONSIBILITIES**

### **Swarm Daemon Handles:**
- ✅ All Claude Code hooks (pre-task, post-edit, etc.)
- ✅ Repository file operations
- ✅ Code analysis and intelligence generation
- ✅ Local optimization and caching
- ✅ Build and test coordination
- ✅ Project-specific learning

### **THE COLLECTIVE Handles:**
- 🧠 Cross-repository pattern sharing
- 📊 Global intelligence aggregation  
- 🌐 Multi-project optimization insights
- ❌ NO direct hook execution
- ❌ NO repository file access

## 🪝 **CLAUDE CODE HOOK INTEGRATION**

### **Hook Configuration in Swarm Daemon**
```rust
// zen-swarm/src/claude_hooks.rs
pub struct ClaudeHooksManager {
    /// Repository path for file operations
    repo_path: PathBuf,
    /// Intelligence cache for this repository
    intelligence_cache: Arc<RwLock<RepositoryIntelligence>>,
    /// A2A coordinator for reporting to THE COLLECTIVE
    a2a_coordinator: Arc<A2ACoordinator>,
    /// Active hooks configuration
    hooks_config: HooksConfig,
}

pub struct HooksConfig {
    pub pre_task_enabled: bool,
    pub post_edit_enabled: bool,
    pub pre_search_enabled: bool,
    pub session_hooks_enabled: bool,
    pub intelligence_reporting_enabled: bool,
}
```

### **Hook Implementations in Swarm Daemon**

#### **Pre-Task Hook (Repository Intelligence)**
```rust
pub async fn pre_task_hook(
    &self,
    task_description: &str,
    context: &TaskContext,
) -> SwarmResult<PreTaskResponse> {
    info!("🚀 Pre-task hook: {}", task_description);
    
    // 1. Analyze repository for relevant context
    let repo_context = self.analyze_repository_context(task_description).await?;
    
    // 2. Load relevant intelligence from cache
    let intelligence = self.load_relevant_intelligence(task_description).await?;
    
    // 3. Check for similar tasks from peer repositories (via THE COLLECTIVE)
    let cross_repo_insights = self.query_collective_for_similar_tasks(task_description).await?;
    
    // 4. Generate optimized task approach
    let task_optimization = self.optimize_task_approach(
        task_description,
        &repo_context,
        &intelligence,
        &cross_repo_insights,
    ).await?;
    
    Ok(PreTaskResponse {
        repo_context,
        suggested_approach: task_optimization,
        relevant_files: repo_context.relevant_files,
        intelligence_insights: intelligence.insights,
        cross_repo_learnings: cross_repo_insights,
    })
}
```

#### **Post-Edit Hook (Intelligence Generation)**
```rust
pub async fn post_edit_hook(
    &self,
    file_path: &Path,
    edit_type: EditType,
    context: &EditContext,
) -> SwarmResult<PostEditResponse> {
    info!("✏️ Post-edit hook: {}", file_path.display());
    
    // 1. Analyze the edit for patterns and intelligence
    let edit_analysis = self.analyze_edit(file_path, edit_type, context).await?;
    
    // 2. Update repository intelligence cache
    self.update_intelligence_cache(&edit_analysis).await?;
    
    // 3. Generate code quality insights
    let quality_insights = self.analyze_code_quality(file_path).await?;
    
    // 4. Check for optimization opportunities
    let optimizations = self.identify_optimizations(file_path, &edit_analysis).await?;
    
    // 5. Report valuable insights to THE COLLECTIVE
    if edit_analysis.has_valuable_insights() {
        self.report_insights_to_collective(&edit_analysis).await?;
    }
    
    // 6. Auto-format and optimize if configured
    if self.hooks_config.auto_format_enabled {
        self.auto_format_file(file_path).await?;
    }
    
    Ok(PostEditResponse {
        intelligence_updated: true,
        quality_score: quality_insights.score,
        suggestions: optimizations,
        auto_actions_taken: vec!["formatting", "intelligence_cache_update"],
    })
}
```

#### **Pre-Search Hook (Repository-Aware Search)**
```rust  
pub async fn pre_search_hook(
    &self,
    query: &str,
    search_context: &SearchContext,
) -> SwarmResult<PreSearchResponse> {
    info!("🔍 Pre-search hook: {}", query);
    
    // 1. Enhance query with repository context
    let enhanced_query = self.enhance_query_with_repo_context(query).await?;
    
    // 2. Check local repository intelligence first
    let local_results = self.search_local_intelligence(query).await?;
    
    // 3. Query peer repositories via A2A if needed
    let peer_results = if search_context.include_peer_repos {
        self.search_peer_repositories(query).await?
    } else {
        Vec::new()
    };
    
    // 4. Cache search results for future optimization
    self.cache_search_results(query, &local_results, &peer_results).await?;
    
    Ok(PreSearchResponse {
        enhanced_query,
        local_results,
        peer_results,
        search_optimizations: vec!["repository_context_enhancement", "peer_coordination"],
    })
}
```

## 🔄 **INTELLIGENCE REPORTING TO THE COLLECTIVE**

### **What Gets Reported**
```rust
pub async fn report_insights_to_collective(&self, insights: &RepositoryInsights) -> SwarmResult<()> {
    let intelligence_report = A2AMessage::IntelligenceResponse {
        request_id: Uuid::new_v4().to_string(),
        result: IntelligenceResult {
            reasoning: serde_json::json!({
                "repository": self.repo_path.display().to_string(),
                "insights_type": insights.insight_type,
                "confidence": insights.confidence,
            }),
            recommendations: insights.recommendations.clone(),
            additional_data: serde_json::json!({
                "code_patterns": insights.code_patterns,
                "performance_improvements": insights.performance_gains,
                "best_practices": insights.best_practices,
                "anti_patterns": insights.anti_patterns,
            }),
            processing_time_ms: insights.processing_time_ms,
        },
        confidence: insights.confidence,
        sources: vec![format!("swarm-daemon-{}", self.repo_path.file_name()
            .and_then(|n| n.to_str()).unwrap_or("unknown"))],
    };
    
    // Send to THE COLLECTIVE for cross-repository learning
    self.a2a_coordinator.send_message("the-collective", intelligence_report).await?;
    
    info!("🧠 Repository insights reported to THE COLLECTIVE");
    Ok(())
}
```

### **What THE COLLECTIVE Does**
- 📊 Aggregates insights from all repository swarms
- 🌐 Identifies cross-repository patterns
- 🎯 Generates global optimization recommendations
- 📤 Distributes successful patterns back to swarms
- ❌ Never directly executes hooks or touches files

## 📝 **HOOK CONFIGURATION**

### **Swarm Daemon Configuration**
```toml
# .swarm-daemon.toml
[claude_hooks]
enabled = true
pre_task_hook = true
post_edit_hook = true
pre_search_hook = true
session_hooks = true

[intelligence]
cache_enabled = true
collective_reporting = true
auto_optimization = true
cross_repo_learning = true

[auto_actions]
format_on_save = true
optimize_imports = true  
cache_search_results = true
intelligent_suggestions = true
```

### **Claude Code MCP Integration**
```bash
# Claude Code connects to swarm daemon, not THE COLLECTIVE
claude mcp add zen-swarm npx zen-swarm mcp connect --daemon-port 9001

# Hooks automatically available through MCP stdio interface
# No additional configuration needed
```

## 🚀 **BENEFITS OF SWARM-BASED HOOKS**

1. **🎯 Repository Context**: Hooks understand the specific repository
2. **⚡ Performance**: Local execution, no network latency
3. **🧠 Intelligence**: Local learning + global pattern sharing
4. **🔐 Privacy**: Repository data stays local, only insights shared
5. **🌐 Scaling**: Each repository gets dedicated hook intelligence
6. **🔄 Coordination**: Cross-repository learning without direct access

---

## ✅ **ARCHITECTURE SUMMARY**

```
Claude Code → Swarm Daemon Hooks → Repository Operations
                     ↓
                Intelligence Reports
                     ↓
                THE COLLECTIVE → Pattern Distribution → Other Swarm Daemons
```

**Perfect separation**: 
- **Swarm Daemon**: ALL repository operations + Claude hooks
- **THE COLLECTIVE**: ONLY intelligence coordination and pattern sharing

This gives you the best of both worlds: **local intelligence** with **global learning**!