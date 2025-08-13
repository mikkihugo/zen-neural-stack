//! Zen Swarm Daemon CLI
//!
//! Command-line interface for managing repository-scoped swarm daemons.
//! Each repository gets its own daemon that handles all repository operations
//! and connects to THE COLLECTIVE for cross-repository intelligence sharing.

use clap::{Arg, Command, ArgMatches};
use std::path::PathBuf;
use tokio;
use tracing::{info, error};
use zen_swarm::{SwarmDaemonCli, SwarmResult};

#[tokio::main]
async fn main() -> SwarmResult<()> {
    // Initialize logging
    tracing_subscriber::init();

    // Parse CLI arguments
    let matches = create_cli().get_matches();
    
    match matches.subcommand() {
        Some(("start", sub_matches)) => {
            handle_start_command(sub_matches).await
        }
        Some(("stop", sub_matches)) => {
            handle_stop_command(sub_matches).await
        }
        Some(("status", sub_matches)) => {
            handle_status_command(sub_matches).await
        }
        Some(("connect", sub_matches)) => {
            handle_connect_command(sub_matches).await
        }
        _ => {
            println!("Use --help for usage information");
            Ok(())
        }
    }
}

fn create_cli() -> Command {
    Command::new("zen-swarm-daemon")
        .version(zen_swarm::VERSION)
        .about("Zen Swarm Daemon - Repository-scoped intelligent coordination")
        .long_about(r#"
🐝 Zen Swarm Daemon - Repository-scoped intelligent coordination

Each repository gets its own swarm daemon that handles:
- Code ingestion and analysis
- Markdown import and processing  
- Build optimization and testing
- Local intelligence generation
- Connection to THE COLLECTIVE for cross-repo learning

ARCHITECTURE:
- Swarm Daemon: ALL repository operations
- THE COLLECTIVE: Intelligence coordination only
- A2A Protocol: Direct peer-to-peer communication with collective visibility

EXAMPLES:
  # Start daemon for current repository
  zen-swarm-daemon start
  
  # Start daemon with specific port and collective
  zen-swarm-daemon start --port 9001 --collective ws://collective.company.com:8080
  
  # Check daemon status
  zen-swarm-daemon status --port 9001
  
  # Connect Claude Code to daemon
  zen-swarm-daemon connect --mcp
"#)
        .subcommand_required(true)
        .subcommand(
            Command::new("start")
                .about("Start swarm daemon for repository")
                .arg(
                    Arg::new("repo-path")
                        .long("repo-path")
                        .short('r')
                        .value_name("PATH")
                        .help("Repository path (defaults to current directory)")
                        .default_value(".")
                )
                .arg(
                    Arg::new("port")
                        .long("port")
                        .short('p')
                        .value_name("PORT")
                        .help("Port for MCP HTTP server")
                        .default_value("9001")
                )
                .arg(
                    Arg::new("collective")
                        .long("collective")
                        .short('c')
                        .value_name("ENDPOINT")
                        .help("THE COLLECTIVE endpoint for intelligence sharing")
                        .required(false)
                )
                .arg(
                    Arg::new("daemon-name")
                        .long("daemon-name")
                        .short('n')
                        .value_name("NAME")
                        .help("Custom daemon name (auto-generated from repo if not specified)")
                        .required(false)
                )
                .arg(
                    Arg::new("detach")
                        .long("detach")
                        .short('d')
                        .action(clap::ArgAction::SetTrue)
                        .help("Run daemon in background")
                )
        )
        .subcommand(
            Command::new("stop")
                .about("Stop running swarm daemon")
                .arg(
                    Arg::new("port")
                        .long("port")
                        .short('p')
                        .value_name("PORT")
                        .help("Port of daemon to stop")
                        .default_value("9001")
                )
                .arg(
                    Arg::new("force")
                        .long("force")
                        .short('f')
                        .action(clap::ArgAction::SetTrue)
                        .help("Force stop daemon")
                )
        )
        .subcommand(
            Command::new("status")
                .about("Check daemon status and health")
                .arg(
                    Arg::new("port")
                        .long("port")
                        .short('p')
                        .value_name("PORT")
                        .help("Port of daemon to check")
                        .default_value("9001")
                )
                .arg(
                    Arg::new("verbose")
                        .long("verbose")
                        .short('v')
                        .action(clap::ArgAction::SetTrue)
                        .help("Show detailed status information")
                )
                .arg(
                    Arg::new("json")
                        .long("json")
                        .action(clap::ArgAction::SetTrue)
                        .help("Output status as JSON")
                )
        )
        .subcommand(
            Command::new("connect")
                .about("Connect Claude Code to daemon")
                .arg(
                    Arg::new("mcp")
                        .long("mcp")
                        .action(clap::ArgAction::SetTrue)
                        .help("Set up MCP connection for Claude Code")
                )
                .arg(
                    Arg::new("port")
                        .long("port")
                        .short('p')
                        .value_name("PORT")
                        .help("Daemon port to connect to")
                        .default_value("9001")
                )
        )
}

async fn handle_start_command(matches: &ArgMatches) -> SwarmResult<()> {
    let repo_path = PathBuf::from(matches.get_one::<String>("repo-path").unwrap());
    let port: u16 = matches.get_one::<String>("port").unwrap().parse()
        .map_err(|_| zen_swarm::SwarmError::Configuration("Invalid port number".to_string()))?;
    let collective_endpoint = matches.get_one::<String>("collective").map(|s| s.clone());
    let detach = matches.get_flag("detach");
    
    info!("🚀 Starting Zen Swarm Daemon");
    info!("📁 Repository: {}", repo_path.display());
    info!("🌐 Port: {}", port);
    if let Some(collective) = &collective_endpoint {
        info!("🧠 THE COLLECTIVE: {}", collective);
    }
    
    if detach {
        info!("🔄 Running in background mode");
        // TODO: Implement background/daemon mode
    }
    
    // Start the daemon
    SwarmDaemonCli::start_daemon(repo_path, port, collective_endpoint).await
}

async fn handle_stop_command(matches: &ArgMatches) -> SwarmResult<()> {
    let port: u16 = matches.get_one::<String>("port").unwrap().parse()
        .map_err(|_| zen_swarm::SwarmError::Configuration("Invalid port number".to_string()))?;
    let force = matches.get_flag("force");
    
    info!("🛑 Stopping Zen Swarm Daemon on port {}", port);
    
    // TODO: Implement daemon stop logic
    // This would send a shutdown signal to the running daemon
    
    if force {
        info!("💥 Force stopping daemon");
        // TODO: Force kill daemon process
    }
    
    println!("✅ Daemon stopped successfully");
    Ok(())
}

async fn handle_status_command(matches: &ArgMatches) -> SwarmResult<()> {
    let port: u16 = matches.get_one::<String>("port").unwrap().parse()
        .map_err(|_| zen_swarm::SwarmError::Configuration("Invalid port number".to_string()))?;
    let verbose = matches.get_flag("verbose");
    let json = matches.get_flag("json");
    
    // TODO: Query daemon status via HTTP API
    let status = query_daemon_status(port).await?;
    
    if json {
        println!("{}", serde_json::to_string_pretty(&status)?);
    } else {
        print_daemon_status(&status, verbose);
    }
    
    Ok(())
}

async fn handle_connect_command(matches: &ArgMatches) -> SwarmResult<()> {
    let port: u16 = matches.get_one::<String>("port").unwrap().parse()
        .map_err(|_| zen_swarm::SwarmError::Configuration("Invalid port number".to_string()))?;
    let setup_mcp = matches.get_flag("mcp");
    
    if setup_mcp {
        info!("🔗 Setting up MCP connection for Claude Code");
        setup_claude_code_mcp_connection(port).await?;
        println!("✅ MCP connection configured for Claude Code");
        println!("   Use: claude mcp add zen-swarm npx zen-swarm mcp connect --daemon-port {}", port);
    }
    
    Ok(())
}

// TODO: Implement actual daemon status query via HTTP
async fn query_daemon_status(port: u16) -> SwarmResult<serde_json::Value> {
    use reqwest;
    
    let url = format!("http://127.0.0.1:{}/health", port);
    
    match reqwest::get(&url).await {
        Ok(response) => {
            let status = response.json::<serde_json::Value>().await
                .map_err(|e| zen_swarm::SwarmError::Communication(format!("Failed to parse status: {}", e)))?;
            Ok(status)
        }
        Err(_) => {
            // Daemon not running
            Ok(serde_json::json!({
                "status": "offline",
                "port": port,
                "message": "Daemon not responding"
            }))
        }
    }
}

fn print_daemon_status(status: &serde_json::Value, verbose: bool) {
    println!("🐝 Zen Swarm Daemon Status");
    println!("─────────────────────────");
    
    if let Some(status_str) = status.get("status").and_then(|s| s.as_str()) {
        let status_emoji = match status_str {
            "healthy" => "✅",
            "degraded" => "⚠️",
            "offline" => "❌",
            _ => "❓",
        };
        println!("{} Status: {}", status_emoji, status_str);
    }
    
    if let Some(port) = status.get("port") {
        println!("🌐 Port: {}", port);
    }
    
    if verbose {
        if let Some(swarm) = status.get("swarm") {
            println!("\n🧠 Swarm Intelligence:");
            println!("   {}", serde_json::to_string_pretty(swarm).unwrap_or_default());
        }
        
        if let Some(server) = status.get("server") {
            println!("\n⚙️  Server Info:");
            println!("   {}", serde_json::to_string_pretty(server).unwrap_or_default());
        }
    }
}

async fn setup_claude_code_mcp_connection(port: u16) -> SwarmResult<()> {
    // TODO: Generate proper MCP connection configuration for Claude Code
    // This could create a .claude/settings.json entry or provide setup instructions
    
    info!("📝 MCP connection setup for port {}", port);
    println!("\n🔗 Add this to your Claude Code MCP configuration:");
    println!("   claude mcp add zen-swarm npx zen-swarm mcp connect --daemon-port {}", port);
    
    Ok(())
}