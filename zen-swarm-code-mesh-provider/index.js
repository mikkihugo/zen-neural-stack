/**
 * Zen-Swarm Provider for code-mesh
 * 
 * Integrates code-mesh with zen-swarm daemons to provide:
 * - Repository-scoped AI coordination
 * - Multi-LLM routing through THE COLLECTIVE
 * - Intelligence caching and learning
 * - Cross-project pattern sharing
 */

import { WebSocket } from 'ws';
import { v4 as uuidv4 } from 'uuid';
import fetch from 'node-fetch';

export class ZenSwarmProvider {
    constructor(config = {}) {
        this.name = 'zen-swarm';
        this.description = 'Zen-swarm distributed AI coordination provider';
        this.version = '1.0.0';
        
        this.swarmDaemonEndpoint = config.swarmDaemonEndpoint || 'http://localhost:9001';
        this.collectiveEndpoint = config.collectiveEndpoint || 'ws://localhost:8080';
        this.repoPath = config.repoPath || process.cwd();
        
        this.supportedModels = [
            // Multi-LLM support through THE COLLECTIVE
            'claude-3-sonnet', 'claude-3-opus', 'claude-3-haiku',
            'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-1.0-pro',
            'gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo',
            'llama-3.1-8b', 'llama-3.1-70b', 'codellama',
            'mistral-7b', 'mistral-8x7b'
        ];
        
        this.wsConnection = null;
        this.pendingRequests = new Map();
    }

    /**
     * Initialize provider connection to zen-swarm daemon
     */
    async initialize() {
        console.log('🐝 Initializing zen-swarm provider...');
        
        // Check if zen-swarm daemon is running
        try {
            const response = await fetch(`${this.swarmDaemonEndpoint}/health`);
            if (!response.ok) {
                throw new Error(`Swarm daemon not responding: ${response.status}`);
            }
            console.log('✅ Connected to zen-swarm daemon');
        } catch (error) {
            console.error('❌ Failed to connect to zen-swarm daemon:', error.message);
            throw new Error(`zen-swarm daemon not available at ${this.swarmDaemonEndpoint}`);
        }
    }

    /**
     * Execute AI request through zen-swarm coordination
     */
    async executeRequest(request) {
        const { 
            model, 
            prompt, 
            temperature = 0.7, 
            maxTokens = 4096, 
            context = {} 
        } = request;

        console.log(`🧠 Executing ${model} request via zen-swarm...`);
        
        const startTime = Date.now();
        const requestId = uuidv4();
        
        // Create zen-swarm task request
        const swarmRequest = {
            task_description: prompt,
            preferred_llm: model,
            context: {
                working_directory: this.repoPath,
                relevant_files: context.files || [],
                build_system: context.buildSystem,
                test_framework: context.testFramework,
                temperature,
                max_tokens: maxTokens
            }
        };

        try {
            // Send request to zen-swarm daemon
            const response = await fetch(`${this.swarmDaemonEndpoint}/tasks/execute`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Request-ID': requestId
                },
                body: JSON.stringify(swarmRequest)
            });

            if (!response.ok) {
                throw new Error(`Swarm execution failed: ${response.status}`);
            }

            const result = await response.json();
            const processingTime = Date.now() - startTime;

            // Log interaction for swarm monitoring
            await this.logInteraction({
                request_id: requestId,
                cli_tool: 'code-mesh',
                model: model,
                prompt: prompt,
                response: result.task_result,
                processing_time_ms: processingTime,
                token_usage: result.usage,
                status: 'completed'
            });

            return {
                content: result.task_result,
                usage: result.usage || {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0
                },
                model: result.model_used,
                provider: result.provider,
                processing_time_ms: processingTime,
                actions_taken: result.actions_taken,
                files_modified: result.files_modified
            };

        } catch (error) {
            console.error('❌ zen-swarm execution failed:', error);
            
            // Log failed interaction
            await this.logInteraction({
                request_id: requestId,
                cli_tool: 'code-mesh',
                model: model,
                prompt: prompt,
                processing_time_ms: Date.now() - startTime,
                status: 'failed',
                error: error.message
            });

            throw error;
        }
    }

    /**
     * Log interaction with swarm daemon for monitoring
     */
    async logInteraction(logData) {
        try {
            await fetch(`${this.swarmDaemonEndpoint}/cli/log-interaction`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    timestamp: new Date().toISOString(),
                    session_id: process.env.CODE_MESH_SESSION_ID || uuidv4(),
                    ...logData
                })
            });
        } catch (error) {
            console.warn('⚠️ Failed to log interaction:', error.message);
        }
    }

    /**
     * Get available models from zen-swarm
     */
    async getAvailableModels() {
        try {
            const response = await fetch(`${this.swarmDaemonEndpoint}/models/available`);
            if (response.ok) {
                const data = await response.json();
                return data.models || this.supportedModels;
            }
        } catch (error) {
            console.warn('⚠️ Failed to fetch available models, using defaults');
        }
        return this.supportedModels;
    }

    /**
     * Get repository intelligence from swarm
     */
    async getRepositoryIntelligence() {
        try {
            const response = await fetch(`${this.swarmDaemonEndpoint}/repository/intelligence`);
            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            console.warn('⚠️ Failed to fetch repository intelligence:', error.message);
            return null;
        }
    }

    /**
     * Generate MCP configuration for Claude Code integration
     */
    async generateMcpConfig() {
        try {
            const response = await fetch(`${this.swarmDaemonEndpoint}/mcp/generate-config`, {
                method: 'POST'
            });
            
            if (response.ok) {
                const data = await response.json();
                return data.config_path;
            }
        } catch (error) {
            console.warn('⚠️ Failed to generate MCP config:', error.message);
            return null;
        }
    }

    /**
     * Health check
     */
    async healthCheck() {
        try {
            const response = await fetch(`${this.swarmDaemonEndpoint}/health`);
            return {
                healthy: response.ok,
                status: response.status,
                daemon: 'zen-swarm',
                endpoint: this.swarmDaemonEndpoint
            };
        } catch (error) {
            return {
                healthy: false,
                error: error.message,
                daemon: 'zen-swarm',
                endpoint: this.swarmDaemonEndpoint
            };
        }
    }

    /**
     * Cleanup resources
     */
    async cleanup() {
        if (this.wsConnection) {
            this.wsConnection.close();
            this.wsConnection = null;
        }
        this.pendingRequests.clear();
        console.log('🧹 zen-swarm provider cleaned up');
    }
}

// Export for code-mesh integration
export default ZenSwarmProvider;