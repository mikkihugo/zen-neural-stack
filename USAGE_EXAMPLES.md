# 🚀 zen-swarm Usage Examples

## 🎯 **Quick Start Guide**

### **Step 1: Start zen-swarm daemon**
```bash
cd /path/to/your/repository
zen-swarm-daemon start --port 9001 --collective ws://collective.company.com:8080
```

### **Step 2: Generate MCP config for Claude Code**
```bash
# Auto-generates .zen-swarm-mcp.json
curl -X POST http://localhost:9001/mcp/generate-config
```

### **Step 3A: Use Claude Code with zen-swarm**
```bash
# Interactive mode with repository intelligence
claude --mcp-config .zen-swarm-mcp.json

# Non-interactive mode with JSON output
claude --mcp-config .zen-swarm-mcp.json -p --output-format json "Fix the memory leak in src/cache.rs"
```

### **Step 3B: Use code-mesh with zen-swarm**
```bash
# Build and register zen-swarm provider
cd zen-neural-stack/zen-swarm-provider
cargo build --release

# Use different LLMs through zen-swarm
code-mesh run --model zen-swarm/gemini-1.5-pro "Analyze this code for performance issues"
code-mesh run --model zen-swarm/gpt-4 "Generate comprehensive unit tests"
```

## 🛠️ **Real-World Workflows**

### **Workflow 1: Multi-LLM Code Review**
```bash
# 1. Fast initial analysis with Gemini
code-mesh run --model zen-swarm/gemini-1.5-flash "Quick code review of src/auth/" > review.md

# 2. Deep architectural analysis with Claude
claude --mcp-config .zen-swarm-mcp.json -p "Based on the initial review, provide detailed architecture recommendations" >> review.md

# 3. Security analysis with GPT-4
code-mesh run --model zen-swarm/gpt-4 "Security audit of authentication module" >> review.md
```

### **Workflow 2: Bug Fix with Cross-LLM Intelligence**
```bash
# 1. Identify bug patterns with Gemini (fast)
code-mesh run --model zen-swarm/gemini-1.5-flash "Find potential memory leaks in src/"

# 2. Fix with Claude (expert refactoring)
claude --mcp-config .zen-swarm-mcp.json "Fix the memory leak identified in src/cache.rs"

# 3. Generate tests with GPT-4 (comprehensive coverage)
code-mesh run --model zen-swarm/gpt-4 "Generate tests to prevent this memory leak from recurring"
```

### **Workflow 3: Feature Development**
```bash
# 1. Plan with Claude (architectural thinking)
claude --mcp-config .zen-swarm-mcp.json "Design a user authentication system with JWT tokens"

# 2. Implement with multiple models
code-mesh run --model zen-swarm/gemini-1.5-pro "Implement JWT token generation service"
code-mesh run --model zen-swarm/gpt-4 "Create middleware for JWT validation"

# 3. Documentation with Claude (clarity and depth)
claude --mcp-config .zen-swarm-mcp.json "Document the authentication system architecture"
```

## 📊 **Monitoring and Analytics**

### **CLI Interaction Logs**
```bash
# View all CLI interactions
curl http://localhost:9001/cli_interactions | jq

# View Claude Code interactions only
curl -X POST http://localhost:9001/cli_interactions -d '{"cli_tool": "claude"}' | jq

# View code-mesh interactions only  
curl -X POST http://localhost:9001/cli_interactions -d '{"cli_tool": "code-mesh"}' | jq
```

### **Repository Intelligence**
```bash
# Get repository intelligence gathered by zen-swarm
curl http://localhost:9001/repository_intelligence | jq

# Get daemon status and health
curl http://localhost:9001/daemon_status | jq
```

## 🎯 **Model Selection Guide**

### **When to use Claude (via Claude Code):**
- ✅ Complex refactoring and architectural changes
- ✅ Deep code analysis and debugging  
- ✅ Documentation and explanation writing
- ✅ Code review and best practices guidance

### **When to use Gemini (via code-mesh):**
- ✅ Fast code analysis and pattern recognition
- ✅ Quick bug identification and triage
- ✅ Performance optimization suggestions
- ✅ Code formatting and style improvements

### **When to use GPT-4 (via code-mesh):**
- ✅ Comprehensive test generation
- ✅ Edge case identification
- ✅ API integration and external service calls
- ✅ Complex business logic implementation

### **When to use GitHub Copilot (via code-mesh):**
- ✅ Context-aware code completion
- ✅ Boilerplate code generation
- ✅ Similar code pattern suggestions
- ✅ IDE-integrated development

## 🚀 **Advanced Features**

### **Cross-LLM Context Sharing**
```bash
# All interactions are automatically logged and shared
# Later LLM calls benefit from previous context

# Example: Gemini finds issue, Claude fixes it, GPT tests it
code-mesh run --model zen-swarm/gemini-1.5-flash "Find authentication vulnerabilities"
# zen-swarm stores findings in repository intelligence

claude --mcp-config .zen-swarm-mcp.json "Fix the authentication vulnerabilities"  
# Claude automatically gets context from Gemini's findings

code-mesh run --model zen-swarm/gpt-4 "Generate security tests for fixed auth module"
# GPT-4 gets context from both Gemini analysis and Claude fixes
```

### **Repository Learning**
```bash
# zen-swarm automatically learns from your repository:
# - Code patterns and conventions
# - Build system configuration
# - Testing frameworks used
# - Domain-specific knowledge
# - Performance optimizations applied

# Query learned patterns
curl http://localhost:9001/repository_intelligence | jq '.code_patterns'

# Query performance optimizations
curl http://localhost:9001/repository_intelligence | jq '.build_optimizations'
```

### **Cross-Repository Pattern Sharing**
```bash
# Successful patterns are shared via THE COLLECTIVE
# Other repositories benefit from your optimizations

# View patterns applied from other repositories
curl http://localhost:9001/repository_intelligence | jq '.applied_cross_repo_patterns'
```

## 🔧 **Configuration Examples**

### **zen-swarm daemon configuration**
```toml
# .zen-swarm-daemon.toml
[daemon]
port = 9001
repo_path = "/path/to/repository"
collective_endpoint = "ws://collective.company.com:8080"

[intelligence]
cache_enabled = true
collective_reporting = true
cross_repo_learning = true

[logging]
cli_interactions = true
json_format = true
log_level = "info"
```

### **MCP configuration (auto-generated)**
```json
{
  "servers": {
    "zen-swarm": {
      "command": "npx",
      "args": ["zen-swarm", "mcp", "start", "--daemon-port", "9001"],
      "env": {
        "ZEN_SWARM_REPO_PATH": "/path/to/repository"
      }
    }
  }
}
```

## 🎯 **Success Metrics**

Track your AI development effectiveness:

```bash
# Token usage across all LLMs
curl http://localhost:9001/cli_interactions | jq '[.[] | .token_usage] | add'

# Most effective model for different task types
curl http://localhost:9001/cli_interactions | jq 'group_by(.interaction_type) | map({type: .[0].interaction_type, models: [.[] | .cli_tool] | group_by(.) | map({model: .[0], count: length})})'

# Average processing time per model
curl http://localhost:9001/cli_interactions | jq 'group_by(.cli_tool) | map({model: .[0].cli_tool, avg_time: ([.[] | .processing_time_ms] | add / length)})'
```

---

**Result: Unified AI development with the best tool for each job!** 🚀