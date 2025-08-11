#!/usr/bin/env python3
"""
Performance Monitoring Dashboard for zen-neural Phase 1
Provides real-time performance tracking and regression detection
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import sqlite3
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import webbrowser

class PerformanceDatabase:
    """SQLite database for storing benchmark results and tracking performance over time"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Benchmark results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                benchmark_name TEXT NOT NULL,
                test_name TEXT NOT NULL,
                duration_ns INTEGER,
                throughput REAL,
                memory_usage_mb REAL,
                cpu_usage_percent REAL,
                metadata TEXT,
                git_commit TEXT,
                environment_hash TEXT
            )
        ''')
        
        # Performance targets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_targets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_name TEXT UNIQUE NOT NULL,
                baseline_value REAL NOT NULL,
                target_improvement_factor REAL NOT NULL,
                current_achievement REAL,
                status TEXT DEFAULT 'pending',
                last_updated TEXT
            )
        ''')
        
        # Performance alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                benchmark_name TEXT,
                regression_percent REAL,
                acknowledged INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Initialize default targets
        self.init_performance_targets()
    
    def init_performance_targets(self):
        """Initialize performance targets for Phase 1"""
        targets = [
            ('dnn_operations', 1.0, 75.0),  # 75x average target (50-100x range)
            ('training_speed', 1.0, 35.0),  # 35x average target (20-50x range)
            ('memory_reduction', 1.0, 0.3),  # 70% reduction = 0.3x memory usage
            ('matrix_operations', 1.0, 30.0), # 30x average target (10-50x range)
            ('overall_system', 1.0, 10.0),   # 10x improvement target
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for name, baseline, target in targets:
            cursor.execute('''
                INSERT OR IGNORE INTO performance_targets 
                (target_name, baseline_value, target_improvement_factor, last_updated)
                VALUES (?, ?, ?, ?)
            ''', (name, baseline, target, datetime.utcnow().isoformat()))
        
        conn.commit()
        conn.close()
    
    def store_benchmark_result(self, benchmark_data: Dict[str, Any]):
        """Store benchmark result in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO benchmark_results 
            (timestamp, benchmark_name, test_name, duration_ns, throughput, 
             memory_usage_mb, cpu_usage_percent, metadata, git_commit, environment_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            benchmark_data.get('timestamp', datetime.utcnow().isoformat()),
            benchmark_data.get('benchmark_name', ''),
            benchmark_data.get('test_name', ''),
            benchmark_data.get('duration_ns', 0),
            benchmark_data.get('throughput', 0.0),
            benchmark_data.get('memory_usage_mb', 0.0),
            benchmark_data.get('cpu_usage_percent', 0.0),
            json.dumps(benchmark_data.get('metadata', {})),
            benchmark_data.get('git_commit', ''),
            benchmark_data.get('environment_hash', '')
        ))
        
        conn.commit()
        conn.close()
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, List]:
        """Get performance trends for the last N hours"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        
        cursor.execute('''
            SELECT benchmark_name, test_name, timestamp, duration_ns, throughput, memory_usage_mb
            FROM benchmark_results 
            WHERE timestamp > ?
            ORDER BY timestamp ASC
        ''', (since_time,))
        
        results = cursor.fetchall()
        conn.close()
        
        # Group by benchmark and test name
        trends = {}
        for row in results:
            benchmark_name, test_name, timestamp, duration_ns, throughput, memory_mb = row
            key = f"{benchmark_name}.{test_name}"
            
            if key not in trends:
                trends[key] = []
            
            trends[key].append({
                'timestamp': timestamp,
                'duration_ns': duration_ns,
                'throughput': throughput,
                'memory_usage_mb': memory_mb
            })
        
        return trends
    
    def detect_performance_regression(self, threshold_percent: float = 10.0) -> List[Dict]:
        """Detect performance regressions by comparing recent results with historical averages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent results (last hour)
        recent_time = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        cursor.execute('''
            SELECT benchmark_name, test_name, AVG(duration_ns) as avg_duration
            FROM benchmark_results 
            WHERE timestamp > ?
            GROUP BY benchmark_name, test_name
        ''', (recent_time,))
        
        recent_results = cursor.fetchall()
        
        regressions = []
        
        for benchmark_name, test_name, recent_avg in recent_results:
            # Get historical average (excluding last hour)
            cursor.execute('''
                SELECT AVG(duration_ns) as historical_avg
                FROM benchmark_results 
                WHERE benchmark_name = ? AND test_name = ? AND timestamp <= ?
            ''', (benchmark_name, test_name, recent_time))
            
            historical_result = cursor.fetchone()
            if historical_result and historical_result[0]:
                historical_avg = historical_result[0]
                
                # Calculate regression percentage
                regression_percent = ((recent_avg - historical_avg) / historical_avg) * 100
                
                if regression_percent > threshold_percent:
                    regressions.append({
                        'benchmark_name': benchmark_name,
                        'test_name': test_name,
                        'regression_percent': regression_percent,
                        'recent_avg_ns': recent_avg,
                        'historical_avg_ns': historical_avg
                    })
        
        conn.close()
        return regressions
    
    def store_performance_alert(self, alert_data: Dict[str, Any]):
        """Store performance alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_alerts 
            (timestamp, alert_type, severity, message, benchmark_name, regression_percent)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow().isoformat(),
            alert_data.get('alert_type', 'regression'),
            alert_data.get('severity', 'warning'),
            alert_data.get('message', ''),
            alert_data.get('benchmark_name', ''),
            alert_data.get('regression_percent', 0.0)
        ))
        
        conn.commit()
        conn.close()

class PerformanceMonitor:
    """Main performance monitoring class"""
    
    def __init__(self, benchmark_dir: str, db_path: str):
        self.benchmark_dir = Path(benchmark_dir)
        self.db = PerformanceDatabase(db_path)
        self.running = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval_seconds: int = 300):
        """Start continuous performance monitoring"""
        print(f"🔍 Starting performance monitoring (interval: {interval_seconds}s)")
        
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval_seconds,)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        print("⏹️  Stopping performance monitoring")
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        while self.running:
            try:
                # Check for new benchmark results
                self._process_new_results()
                
                # Detect performance regressions
                regressions = self.db.detect_performance_regression()
                for regression in regressions:
                    self._handle_performance_regression(regression)
                
                # Check performance targets
                self._check_performance_targets()
                
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
            
            time.sleep(interval_seconds)
    
    def _process_new_results(self):
        """Process new benchmark result files"""
        results_dir = self.benchmark_dir / "benchmark_results" / "reports"
        
        if not results_dir.exists():
            return
        
        # Look for new JSON result files
        for json_file in results_dir.glob("*.json"):
            # Skip if already processed (would need to track this)
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                # Process criterion benchmark results
                if "benchmarks" in data:
                    for benchmark in data["benchmarks"]:
                        self.db.store_benchmark_result({
                            'benchmark_name': benchmark.get('group_id', ''),
                            'test_name': benchmark.get('function_id', ''),
                            'duration_ns': benchmark.get('typical', 0),
                            'throughput': benchmark.get('throughput', 0.0),
                            'metadata': benchmark
                        })
                        
            except Exception as e:
                print(f"⚠️  Error processing {json_file}: {e}")
    
    def _handle_performance_regression(self, regression: Dict[str, Any]):
        """Handle detected performance regression"""
        message = (f"Performance regression detected in {regression['benchmark_name']}"
                  f".{regression['test_name']}: {regression['regression_percent']:.1f}% slower")
        
        severity = 'critical' if regression['regression_percent'] > 25 else 'warning'
        
        self.db.store_performance_alert({
            'alert_type': 'regression',
            'severity': severity,
            'message': message,
            'benchmark_name': regression['benchmark_name'],
            'regression_percent': regression['regression_percent']
        })
        
        print(f"🚨 {severity.upper()}: {message}")
    
    def _check_performance_targets(self):
        """Check if performance targets are being met"""
        # This would analyze recent benchmark results against targets
        # and update target achievement status
        pass
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for the web dashboard"""
        trends = self.db.get_performance_trends(hours=24)
        
        # Calculate current performance metrics
        dashboard_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'performance_trends': trends,
            'targets': {
                'dnn_operations': {'target': 75, 'current': 0, 'status': 'pending'},
                'training_speed': {'target': 35, 'current': 0, 'status': 'pending'},
                'memory_reduction': {'target': 70, 'current': 0, 'status': 'pending'},
                'matrix_operations': {'target': 30, 'current': 0, 'status': 'pending'},
                'overall_system': {'target': 10, 'current': 0, 'status': 'pending'},
            },
            'recent_regressions': self.db.detect_performance_regression(),
            'system_info': self._get_system_info()
        }
        
        return dashboard_data
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information"""
        import platform
        import psutil
        
        try:
            return {
                'platform': platform.system(),
                'cpu': platform.processor(),
                'cpu_count': str(psutil.cpu_count()),
                'memory_gb': f"{psutil.virtual_memory().total / (1024**3):.1f}",
                'python_version': platform.python_version()
            }
        except Exception:
            return {'error': 'Unable to gather system info'}

class DashboardHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard web interface"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        super().__init__()
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self._serve_dashboard()
        elif self.path == '/api/data':
            self._serve_api_data()
        elif self.path == '/api/targets':
            self._serve_targets_data()
        else:
            self.send_error(404)
    
    def _serve_dashboard(self):
        """Serve the main dashboard HTML"""
        html_content = self._generate_dashboard_html()
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def _serve_api_data(self):
        """Serve dashboard data as JSON"""
        data = self.monitor.generate_dashboard_data()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _serve_targets_data(self):
        """Serve performance targets data"""
        # This would query the database for current target status
        targets_data = {
            'targets': [
                {'name': 'DNN Operations', 'target': '50-100x', 'current': 'TBD', 'status': 'pending'},
                {'name': 'Training Speed', 'target': '20-50x', 'current': 'TBD', 'status': 'pending'},
                {'name': 'Memory Usage', 'target': '70% reduction', 'current': 'TBD', 'status': 'pending'},
                {'name': 'Matrix Operations', 'target': '10-50x', 'current': 'TBD', 'status': 'pending'},
                {'name': 'Overall System', 'target': '10x', 'current': 'TBD', 'status': 'pending'},
            ]
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(targets_data).encode())
    
    def _generate_dashboard_html(self) -> str:
        """Generate the dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>zen-neural Performance Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }
        .header { text-align: center; margin-bottom: 30px; }
        .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { text-align: center; margin: 10px; }
        .metric-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .metric-label { color: #7f8c8d; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-pending { background: #f39c12; }
        .status-met { background: #27ae60; }
        .status-missed { background: #e74c3c; }
        .alert { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .alert-warning { background: #fff3cd; border: 1px solid #ffeaa7; }
        .alert-critical { background: #f8d7da; border: 1px solid #f5c6cb; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; }
        .auto-refresh { position: fixed; top: 20px; right: 20px; background: #007bff; color: white; padding: 8px 16px; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="auto-refresh">Auto-refresh: <span id="countdown">30</span>s</div>
    
    <div class="header">
        <h1>🚀 zen-neural Performance Dashboard</h1>
        <h2>Phase 1: Rust vs JavaScript Performance Validation</h2>
        <p>Real-time monitoring of 50-100x performance improvement targets</p>
    </div>
    
    <div class="dashboard-grid">
        <div class="card">
            <h3>🎯 Performance Targets Status</h3>
            <div id="targets-status">Loading...</div>
        </div>
        
        <div class="card">
            <h3>📊 Recent Performance Metrics</h3>
            <div class="metric">
                <div class="metric-value" id="avg-speedup">-</div>
                <div class="metric-label">Average Speedup vs JavaScript</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="memory-reduction">-</div>
                <div class="metric-label">Memory Usage Reduction</div>
            </div>
        </div>
        
        <div class="card">
            <h3>🚨 Performance Alerts</h3>
            <div id="alerts-container">No alerts</div>
        </div>
        
        <div class="card">
            <h3>📈 Performance Trends (24h)</h3>
            <div id="trends-chart">Loading trends...</div>
        </div>
        
        <div class="card">
            <h3>💻 System Information</h3>
            <table id="system-info">
                <tr><td>Loading...</td><td>-</td></tr>
            </table>
        </div>
        
        <div class="card">
            <h3>🔍 Latest Benchmark Results</h3>
            <div id="latest-results">Loading...</div>
        </div>
    </div>

    <script>
        let countdown = 30;
        
        async function fetchDashboardData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Error fetching dashboard data:', error);
            }
        }
        
        function updateDashboard(data) {
            // Update targets status
            const targetsHtml = Object.entries(data.targets).map(([key, target]) => 
                `<div><span class="status-indicator status-${target.status}"></span>${key}: ${target.current}x (target: ${target.target}x)</div>`
            ).join('');
            document.getElementById('targets-status').innerHTML = targetsHtml;
            
            // Update system info
            const systemHtml = Object.entries(data.system_info).map(([key, value]) => 
                `<tr><td>${key}</td><td>${value}</td></tr>`
            ).join('');
            document.getElementById('system-info').innerHTML = systemHtml;
            
            // Update alerts
            const alertsHtml = data.recent_regressions.length > 0 
                ? data.recent_regressions.map(reg => 
                    `<div class="alert alert-warning">Performance regression: ${reg.benchmark_name}.${reg.test_name} (${reg.regression_percent.toFixed(1)}% slower)</div>`
                ).join('')
                : 'No performance regressions detected';
            document.getElementById('alerts-container').innerHTML = alertsHtml;
        }
        
        function updateCountdown() {
            countdown--;
            document.getElementById('countdown').textContent = countdown;
            
            if (countdown <= 0) {
                fetchDashboardData();
                countdown = 30;
            }
        }
        
        // Initial load
        fetchDashboardData();
        
        // Auto-refresh
        setInterval(updateCountdown, 1000);
        setInterval(fetchDashboardData, 30000);
    </script>
</body>
</html>
        """

def create_custom_handler(monitor):
    """Create a custom HTTP handler with monitor instance"""
    class CustomHandler(DashboardHTTPHandler):
        def __init__(self, *args, **kwargs):
            self.monitor = monitor
            super(BaseHTTPRequestHandler, self).__init__(*args, **kwargs)
    
    return CustomHandler

def main():
    parser = argparse.ArgumentParser(description='zen-neural Performance Monitor')
    parser.add_argument('--benchmark-dir', default='.',
                      help='Directory containing benchmark results')
    parser.add_argument('--db-path', default='performance_monitor.db',
                      help='Path to SQLite database')
    parser.add_argument('--port', type=int, default=8080,
                      help='Web dashboard port')
    parser.add_argument('--monitor-interval', type=int, default=300,
                      help='Monitoring interval in seconds')
    parser.add_argument('--no-web', action='store_true',
                      help='Disable web dashboard')
    
    args = parser.parse_args()
    
    # Initialize performance monitor
    monitor = PerformanceMonitor(args.benchmark_dir, args.db_path)
    
    # Start monitoring
    monitor.start_monitoring(args.monitor_interval)
    
    if not args.no_web:
        # Start web dashboard
        handler_class = create_custom_handler(monitor)
        httpd = HTTPServer(('localhost', args.port), handler_class)
        
        print(f"🌐 Performance dashboard available at: http://localhost:{args.port}")
        print("📊 Monitoring zen-neural Phase 1 performance targets:")
        print("   • DNN Operations: 50-100x improvement target")
        print("   • Training Speed: 20-50x improvement target")
        print("   • Memory Usage: 70% reduction target")
        print("   • Matrix Operations: 10-50x improvement target")
        print("   • Overall System: 10x improvement target")
        print()
        print("Press Ctrl+C to stop monitoring")
        
        try:
            # Try to open browser automatically
            webbrowser.open(f'http://localhost:{args.port}')
        except:
            pass
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n⏹️  Shutting down performance monitor...")
            httpd.shutdown()
    else:
        print("🔍 Performance monitoring started (no web dashboard)")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Shutting down performance monitor...")
    
    monitor.stop_monitoring()
    print("✅ Performance monitoring stopped")

if __name__ == '__main__':
    main()