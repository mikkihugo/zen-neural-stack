# 🔧 Swarm Daemon Task Execution Architecture

## 🎯 **SWARM EXECUTES ALL CODING TASKS**

### **Architecture**: CLI → Swarm Daemon (Executes) → THE COLLECTIVE (Intelligence)

```
Gemini CLI: "Fix this bug" 
    ↓
zen-swarm daemon:
├── 🧠 Uses Gemini via THE COLLECTIVE for reasoning  
├── 🔧 Analyzes repository code locally
├── ✏️ Makes actual file changes
├── 🧪 Runs tests to verify fix
└── 📊 Reports results back to CLI

GPT CLI: "Add feature X"
    ↓  
zen-swarm daemon:
├── 🧠 Uses GPT-4 via THE COLLECTIVE for planning
├── 🔧 Generates code locally  
├── ✏️ Creates/modifies files
├── 🧪 Runs tests and builds
└── 📊 Reports completion to CLI

Claude CLI: "Refactor this function"
    ↓
zen-swarm daemon:  
├── 🧠 Uses Claude via built-in integration
├── 🔧 Analyzes code structure locally
├── ✏️ Performs refactoring
├── 🧪 Validates changes
└── 📊 Reports refactoring results
```

## 🛠️ **SWARM DAEMON TASK EXECUTION ENGINE**

### **Task Execution Flow**
```rust
// zen-swarm/src/task_executor.rs
pub struct TaskExecutor {
    /// Repository path for file operations
    repo_path: PathBuf,
    /// A2A coordinator for LLM requests
    a2a_coordinator: Arc<A2ACoordinator>,
    /// Repository intelligence cache
    intelligence_cache: Arc<RwLock<RepositoryIntelligence>>,
    /// File system operations
    file_ops: Arc<FileOperations>,
    /// Build and test runner
    build_runner: Arc<BuildRunner>,
}

impl TaskExecutor {
    /// Execute coding task with specified LLM
    pub async fn execute_coding_task(
        &self,
        task_description: &str,
        preferred_llm: &str, // "gemini", "gpt-4", "claude"
        context: TaskContext,
    ) -> Result<CodingTaskResponse, SwarmError> {
        info!("🚀 Executing coding task with {}: {}", preferred_llm, task_description);
        
        let mut actions_taken = Vec::new();
        let mut files_modified = Vec::new();
        let start_time = std::time::Instant::now();
        
        // 1. ANALYZE: Understand the task and repository context
        let analysis = self.analyze_task(task_description, &context).await?;
        actions_taken.push("repository_analysis".to_string());
        
        // 2. PLAN: Get plan from specified LLM via THE COLLECTIVE
        let plan = self.get_llm_plan(task_description, preferred_llm, &analysis).await?;
        actions_taken.push(format!("planning_via_{}", preferred_llm));
        
        // 3. EXECUTE: Perform the actual coding work
        let execution_result = self.execute_plan(&plan, &mut files_modified, &mut actions_taken).await?;
        
        // 4. VALIDATE: Run tests and verify changes
        let validation_result = self.validate_changes(&files_modified).await?;
        actions_taken.push("validation_and_testing".to_string());
        
        // 5. REPORT: Generate comprehensive result
        let processing_time_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(CodingTaskResponse {
            task_result: execution_result.summary,
            files_modified,
            actions_taken,
            model_used: preferred_llm.to_string(),
            provider: self.get_provider_for_model(preferred_llm),
            usage: execution_result.token_usage,
            processing_time_ms,
        })
    }
    
    /// Get implementation plan from LLM via THE COLLECTIVE  
    async fn get_llm_plan(
        &self,
        task: &str,
        llm_model: &str,
        analysis: &TaskAnalysis,
    ) -> Result<ImplementationPlan, SwarmError> {
        let llm_request = A2AMessage::IntelligenceRequest {
            request_id: Uuid::new_v4().to_string(),
            capability: AICapability::LLMReasoning {
                model: llm_model.to_string(),
                context_length: 8192,
            },
            context: SwarmContext {
                swarm_id: format!("task-executor-{}", self.repo_path.file_name()
                    .and_then(|n| n.to_str()).unwrap_or("unknown")),
                active_agents: 1,
                current_tasks: vec!["planning".to_string()],
                available_resources: std::iter::from_fn(|| None)
                    .chain(std::iter::once(("task_description".to_string(), 
                           serde_json::json!(task))))
                    .chain(std::iter::once(("repository_analysis".to_string(),
                           serde_json::to_value(analysis)?)))
                    .chain(std::iter::once(("file_context".to_string(),
                           serde_json::json!(analysis.relevant_files))))
                    .collect(),
                local_capabilities: vec!["code_execution".to_string(), "file_operations".to_string()],
            },
            priority: Priority::Normal,
        };
        
        // Send to THE COLLECTIVE for LLM processing
        match self.a2a_coordinator.send_message("the-collective", llm_request).await? {
            A2AMessage::IntelligenceResponse { result, .. } => {
                Ok(ImplementationPlan::from_llm_response(result))
            }
            _ => Err(SwarmError::Communication("Invalid LLM response".to_string())),
        }
    }
    
    /// Execute the implementation plan (actual file operations)
    async fn execute_plan(
        &self,
        plan: &ImplementationPlan,
        files_modified: &mut Vec<String>,
        actions_taken: &mut Vec<String>,
    ) -> Result<ExecutionResult, SwarmError> {
        info!("🔧 Executing implementation plan: {} steps", plan.steps.len());
        
        let mut execution_result = ExecutionResult::new();
        
        for step in &plan.steps {
            match step {
                PlanStep::CreateFile { path, content } => {
                    self.file_ops.create_file(path, content).await?;
                    files_modified.push(path.clone());
                    actions_taken.push(format!("created_file_{}", path));
                }
                PlanStep::ModifyFile { path, changes } => {
                    self.file_ops.apply_changes(path, changes).await?;
                    files_modified.push(path.clone());
                    actions_taken.push(format!("modified_file_{}", path));
                }
                PlanStep::DeleteFile { path } => {
                    self.file_ops.delete_file(path).await?;
                    actions_taken.push(format!("deleted_file_{}", path));
                }
                PlanStep::RunCommand { command, args } => {
                    let output = self.build_runner.run_command(command, args).await?;
                    execution_result.command_outputs.push(output);
                    actions_taken.push(format!("executed_command_{}", command));
                }
                PlanStep::RunTests { test_pattern } => {
                    let test_result = self.build_runner.run_tests(test_pattern).await?;
                    execution_result.test_results.push(test_result);
                    actions_taken.push("executed_tests".to_string());
                }
            }
        }
        
        execution_result.summary = format!("Completed {} implementation steps", plan.steps.len());
        Ok(execution_result)
    }
    
    /// Validate changes by running tests and builds
    async fn validate_changes(&self, files_modified: &[String]) -> Result<ValidationResult, SwarmError> {
        info!("🧪 Validating changes to {} files", files_modified.len());
        
        // Run build to check for compilation errors
        let build_result = self.build_runner.run_build().await?;
        
        // Run test suite  
        let test_result = self.build_runner.run_all_tests().await?;
        
        // Check code quality/linting
        let lint_result = self.build_runner.run_linter().await?;
        
        Ok(ValidationResult {
            build_success: build_result.success,
            tests_passed: test_result.success,
            lint_clean: lint_result.success,
            build_output: build_result.output,
            test_output: test_result.output,
            lint_output: lint_result.output,
        })
    }
}
```

## 🎯 **EXAMPLE CLI TASK EXECUTIONS**

### **Gemini CLI: Bug Fix**
```bash
$ gemini "Fix the memory leak in src/cache.rs"

🚀 Executing coding task with gemini: Fix the memory leak in src/cache.rs
🔍 Analyzing repository context...
🧠 Planning via gemini-1.5-pro...
🔧 Executing implementation plan: 3 steps
   ✏️  Modified file: src/cache.rs
   🧪 Executed tests
   🔧 Applied memory leak fix
🧪 Validating changes to 1 files...
   ✅ Build: SUCCESS
   ✅ Tests: 15 passed, 0 failed
   ✅ Lint: CLEAN

✅ Task completed successfully!
   Files modified: src/cache.rs
   Actions taken: repository_analysis, planning_via_gemini, modified_file_src/cache.rs, executed_tests, validation_and_testing
   Model used: gemini-1.5-pro
   Processing time: 2.3s
```

### **GPT CLI: Feature Addition**  
```bash
$ gpt "Add a REST API endpoint for user profile management" --model gpt-4

🚀 Executing coding task with gpt-4: Add a REST API endpoint for user profile management  
🔍 Analyzing repository context...
🧠 Planning via gpt-4...
🔧 Executing implementation plan: 7 steps
   ✏️  Created file: src/routes/profile.rs
   ✏️  Modified file: src/main.rs  
   ✏️  Created file: tests/profile_tests.rs
   🧪 Executed tests
   🔧 Applied route registration
🧪 Validating changes to 3 files...
   ✅ Build: SUCCESS
   ✅ Tests: 23 passed, 0 failed  
   ✅ Lint: CLEAN

✅ Task completed successfully!
   Files modified: src/routes/profile.rs, src/main.rs, tests/profile_tests.rs
   Actions taken: repository_analysis, planning_via_gpt-4, created_file_src/routes/profile.rs, modified_file_src/main.rs, created_file_tests/profile_tests.rs, executed_tests, validation_and_testing  
   Model used: gpt-4
   Processing time: 4.1s
```

### **Claude CLI: Refactoring**
```bash
$ claude "Refactor the authentication module to use async/await"

🚀 Executing coding task with claude: Refactor the authentication module to use async/await
🔍 Analyzing repository context...
🧠 Planning via claude-3-sonnet...  
🔧 Executing implementation plan: 5 steps
   ✏️  Modified file: src/auth/mod.rs
   ✏️  Modified file: src/auth/jwt.rs
   ✏️  Modified file: src/auth/middleware.rs
   🧪 Executed tests
🧪 Validating changes to 3 files...
   ✅ Build: SUCCESS
   ✅ Tests: 31 passed, 0 failed
   ✅ Lint: CLEAN

✅ Task completed successfully!
   Files modified: src/auth/mod.rs, src/auth/jwt.rs, src/auth/middleware.rs
   Actions taken: repository_analysis, planning_via_claude, modified_file_src/auth/mod.rs, modified_file_src/auth/jwt.rs, modified_file_src/auth/middleware.rs, executed_tests, validation_and_testing
   Model used: claude-3-sonnet  
   Processing time: 3.7s
```

## 🏗️ **SWARM DAEMON HTTP ENDPOINTS**

### **Task Execution Endpoint**
```rust
// zen-swarm/src/daemon.rs - HTTP routes for task execution
async fn setup_task_routes(&self) -> Router {
    Router::new()
        .route("/tasks/execute", post(Self::handle_task_execution))
        .route("/tasks/status/:task_id", get(Self::handle_task_status))
        .route("/repository/analyze", post(Self::handle_repo_analysis))
        .with_state(Arc::new(self.clone()))
}

async fn handle_task_execution(
    State(daemon): State<Arc<SwarmDaemon>>,
    Json(request): Json<CodingTaskRequest>,
) -> Result<Json<CodingTaskResponse>, StatusCode> {
    match daemon.task_executor.execute_coding_task(
        &request.task_description,
        &request.preferred_llm,
        request.context,
    ).await {
        Ok(response) => Ok(Json(response)),
        Err(e) => {
            error!("Task execution failed: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
```

Perfect! Now the architecture is clear:
- **CLI tools** send tasks to swarm daemon
- **Swarm daemon executes** all the actual coding work
- **THE COLLECTIVE** provides LLM intelligence as needed
- **Results** flow back to CLI with full execution details

🎯 **Swarm = The Worker, THE COLLECTIVE = The Brain**