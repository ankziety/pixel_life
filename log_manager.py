#!/usr/bin/env python3
"""
Centralized Logging Management System for Pixel Life

This module provides a comprehensive logging solution that addresses:
- Too many log files scattered across directories
- No tracking of log file counts and sizes
- No identification of best performing models
- No performance analysis of logs
- No CLI tools for log management

Features:
- Structured JSON logging with metadata
- Automatic log rotation and cleanup
- Performance tracking and best model identification
- CLI tools for log analysis and management
- Centralized log storage with organized structure
"""

import os
import json
import shutil
import argparse
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import hashlib
import glob
import time
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Structured log entry with metadata."""
    timestamp: str
    level: str
    message: str
    module: str
    function: str
    line: int
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_path: str
    timestamp: str
    generation: int
    episode_reward: float
    survival_rate: float
    training_steps: int
    model_size_mb: float
    training_time_seconds: float
    environment_size: int
    hyperparameters: Dict[str, Any]
    device: str
    framework: str


@dataclass
class LogStats:
    """Log statistics and metrics."""
    total_log_files: int
    total_size_mb: float
    oldest_log: Optional[str]
    newest_log: Optional[str]
    log_types: Dict[str, int]
    performance_models: int
    best_model_path: Optional[str]
    best_model_reward: Optional[float]


class LogManager:
    """Centralized logging management system."""
    
    def __init__(self, base_dir: str = "./logs"):
        self.base_dir = Path(base_dir)
        self.db_path = self.base_dir / "log_metadata.db"
        self.structured_logs_dir = self.base_dir / "structured"
        self.model_logs_dir = self.base_dir / "models"
        self.performance_logs_dir = self.base_dir / "performance"
        self.archive_dir = self.base_dir / "archive"
        
        # Create directories
        for dir_path in [self.structured_logs_dir, self.model_logs_dir, 
                        self.performance_logs_dir, self.archive_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Current request ID for tracking
        self.current_request_id = None
    
    def _init_database(self):
        """Initialize SQLite database for log metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS log_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                module TEXT NOT NULL,
                function TEXT NOT NULL,
                line INTEGER NOT NULL,
                request_id TEXT,
                user_id TEXT,
                session_id TEXT,
                duration_ms REAL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_path TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                generation INTEGER NOT NULL,
                episode_reward REAL NOT NULL,
                survival_rate REAL NOT NULL,
                training_steps INTEGER NOT NULL,
                model_size_mb REAL NOT NULL,
                training_time_seconds REAL NOT NULL,
                environment_size INTEGER NOT NULL,
                hyperparameters TEXT NOT NULL,
                device TEXT NOT NULL,
                framework TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS log_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                last_modified TIMESTAMP NOT NULL,
                metadata TEXT,
                created_at_db TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_request(self, request_id: str = None):
        """Start tracking a new request."""
        if request_id is None:
            request_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
        self.current_request_id = request_id
        return request_id
    
    def log(self, level: str, message: str, module: str = None, 
            function: str = None, line: int = None, **kwargs):
        """Log a structured message."""
        if module is None:
            frame = sys._getframe(1)
            module = frame.f_globals.get('__name__', 'unknown') or 'unknown'
            function = frame.f_code.co_name or 'unknown'
            line = frame.f_lineno or 0
        
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.upper(),
            message=message,
            module=module,
            function=function,
            line=line or 0,
            request_id=self.current_request_id,
            metadata=kwargs
        )
        
        # Save to database
        self._save_log_entry(entry)
        
        # Save to structured log file
        self._save_structured_log(entry)
        
        # Also log to standard logging
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"[{entry.request_id}] {message}")
    
    def _save_log_entry(self, entry: LogEntry):
        """Save log entry to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO log_entries 
            (timestamp, level, message, module, function, line, request_id, 
             user_id, session_id, duration_ms, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.timestamp, entry.level, entry.message, entry.module,
            entry.function, entry.line, entry.request_id, entry.user_id,
            entry.session_id, entry.duration_ms, 
            json.dumps(entry.metadata) if entry.metadata else None
        ))
        
        conn.commit()
        conn.close()
    
    def _save_structured_log(self, entry: LogEntry):
        """Save structured log to file."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = self.structured_logs_dir / f"{date_str}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(asdict(entry)) + '\n')
    
    def log_model_performance(self, performance: ModelPerformance):
        """Log model performance metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO model_performance 
            (model_path, timestamp, generation, episode_reward, survival_rate,
             training_steps, model_size_mb, training_time_seconds, 
             environment_size, hyperparameters, device, framework)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            performance.model_path, performance.timestamp, performance.generation,
            performance.episode_reward, performance.survival_rate,
            performance.training_steps, performance.model_size_mb,
            performance.training_time_seconds, performance.environment_size,
            json.dumps(performance.hyperparameters), performance.device,
            performance.framework
        ))
        
        conn.commit()
        conn.close()
        
        # Save to performance log file
        perf_file = self.performance_logs_dir / f"performance_{performance.timestamp}.json"
        with open(perf_file, 'w') as f:
            json.dump(asdict(performance), f, indent=2)
    
    def get_best_model(self) -> Optional[ModelPerformance]:
        """Get the best performing model based on episode reward."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM model_performance 
            ORDER BY episode_reward DESC 
            LIMIT 1
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_model_performance(row)
        return None
    
    def _row_to_model_performance(self, row) -> ModelPerformance:
        """Convert database row to ModelPerformance object."""
        return ModelPerformance(
            model_path=row[1],
            timestamp=row[2],
            generation=row[3],
            episode_reward=row[4],
            survival_rate=row[5],
            training_steps=row[6],
            model_size_mb=row[7],
            training_time_seconds=row[8],
            environment_size=row[9],
            hyperparameters=json.loads(row[10]),
            device=row[11],
            framework=row[12]
        )
    
    def get_log_stats(self) -> LogStats:
        """Get comprehensive log statistics."""
        # Count log files
        total_log_files = 0
        total_size_mb = 0
        log_types = {}
        oldest_log = None
        newest_log = None
        
        # Scan all log directories
        for log_dir in [self.structured_logs_dir, self.model_logs_dir, 
                       self.performance_logs_dir]:
            if log_dir.exists():
                for file_path in log_dir.rglob("*"):
                    if file_path.is_file():
                        total_log_files += 1
                        file_size = file_path.stat().st_size
                        total_size_mb += file_size / (1024 * 1024)
                        
                        file_type = file_path.suffix
                        log_types[file_type] = log_types.get(file_type, 0) + 1
                        
                        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if oldest_log is None or mtime < oldest_log:
                            oldest_log = mtime
                        if newest_log is None or mtime > newest_log:
                            newest_log = mtime
        
        # Get performance model count
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM model_performance')
        performance_models = cursor.fetchone()[0]
        
        # Get best model
        cursor.execute('''
            SELECT model_path, episode_reward FROM model_performance 
            ORDER BY episode_reward DESC LIMIT 1
        ''')
        best_model_row = cursor.fetchone()
        conn.close()
        
        best_model_path = best_model_row[0] if best_model_row else None
        best_model_reward = best_model_row[1] if best_model_row else None
        
        return LogStats(
            total_log_files=total_log_files,
            total_size_mb=total_size_mb,
            oldest_log=oldest_log.isoformat() if oldest_log else None,
            newest_log=newest_log.isoformat() if newest_log else None,
            log_types=log_types,
            performance_models=performance_models,
            best_model_path=best_model_path,
            best_model_reward=best_model_reward
        )
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up logs older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleaned_files = 0
        
        for log_dir in [self.structured_logs_dir, self.model_logs_dir, 
                       self.performance_logs_dir]:
            if log_dir.exists():
                for file_path in log_dir.rglob("*"):
                    if file_path.is_file():
                        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if mtime < cutoff_date:
                            # Move to archive instead of deleting
                            archive_path = self.archive_dir / file_path.name
                            shutil.move(str(file_path), str(archive_path))
                            cleaned_files += 1
        
        # Clean up database entries
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM log_entries 
            WHERE datetime(timestamp) < datetime(?)
        ''', (cutoff_date.isoformat(),))
        conn.commit()
        conn.close()
        
        return cleaned_files
    
    def search_logs(self, query: str, level: str = None, 
                   start_date: str = None, end_date: str = None,
                   limit: int = 100) -> List[LogEntry]:
        """Search logs with filters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        sql = '''
            SELECT * FROM log_entries 
            WHERE 1=1
        '''
        params = []
        
        if query:
            sql += ' AND (message LIKE ? OR module LIKE ? OR function LIKE ?)'
            params.extend([f'%{query}%'] * 3)
        
        if level:
            sql += ' AND level = ?'
            params.append(level.upper())
        
        if start_date:
            sql += ' AND timestamp >= ?'
            params.append(start_date)
        
        if end_date:
            sql += ' AND timestamp <= ?'
            params.append(end_date)
        
        sql += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_log_entry(row) for row in rows]
    
    def _row_to_log_entry(self, row) -> LogEntry:
        """Convert database row to LogEntry object."""
        return LogEntry(
            timestamp=row[1],
            level=row[2],
            message=row[3],
            module=row[4],
            function=row[5],
            line=row[6],
            request_id=row[7],
            user_id=row[8],
            session_id=row[9],
            duration_ms=row[10],
            metadata=json.loads(row[11]) if row[11] else None
        )
    
    def get_performance_history(self, limit: int = 50) -> List[ModelPerformance]:
        """Get performance history of models."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM model_performance 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_model_performance(row) for row in rows]


def main():
    """CLI interface for log management."""
    parser = argparse.ArgumentParser(description='Pixel Life Log Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show log statistics')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search logs')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--level', help='Log level filter')
    search_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    search_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    search_parser.add_argument('--limit', type=int, default=50, help='Max results')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old logs')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Days to keep')
    
    # Performance command
    perf_parser = subparsers.add_parser('performance', help='Show model performance')
    perf_parser.add_argument('--limit', type=int, default=20, help='Number of models to show')
    
    # Best model command
    best_parser = subparsers.add_parser('best', help='Show best performing model')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    log_manager = LogManager()
    
    if args.command == 'stats':
        stats = log_manager.get_log_stats()
        print("📊 Log Statistics")
        print("=" * 50)
        print(f"Total log files: {stats.total_log_files}")
        print(f"Total size: {stats.total_size_mb:.2f} MB")
        print(f"Oldest log: {stats.oldest_log}")
        print(f"Newest log: {stats.newest_log}")
        print(f"Performance models: {stats.performance_models}")
        print(f"Best model reward: {stats.best_model_reward:.2f}" if stats.best_model_reward else "No models")
        print(f"Log types: {stats.log_types}")
    
    elif args.command == 'search':
        results = log_manager.search_logs(
            query=args.query,
            level=args.level,
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit
        )
        print(f"🔍 Search Results ({len(results)} found)")
        print("=" * 50)
        for entry in results:
            print(f"[{entry.timestamp}] {entry.level}: {entry.message}")
            print(f"  Module: {entry.module}.{entry.function}:{entry.line}")
            if entry.request_id:
                print(f"  Request ID: {entry.request_id}")
            print()
    
    elif args.command == 'cleanup':
        cleaned = log_manager.cleanup_old_logs(days=args.days)
        print(f"🧹 Cleaned up {cleaned} old log files")
    
    elif args.command == 'performance':
        history = log_manager.get_performance_history(limit=args.limit)
        print("📈 Model Performance History")
        print("=" * 80)
        print(f"{'Generation':<10} {'Reward':<10} {'Survival':<10} {'Steps':<10} {'Size(MB)':<10} {'Device':<8} {'Path'}")
        print("-" * 80)
        for perf in history:
            print(f"{perf.generation:<10} {perf.episode_reward:<10.2f} {perf.survival_rate:<10.2f} "
                  f"{perf.training_steps:<10} {perf.model_size_mb:<10.1f} {perf.device:<8} {perf.model_path}")
    
    elif args.command == 'best':
        best_model = log_manager.get_best_model()
        if best_model:
            print("🏆 Best Performing Model")
            print("=" * 50)
            print(f"Model path: {best_model.model_path}")
            print(f"Episode reward: {best_model.episode_reward:.2f}")
            print(f"Survival rate: {best_model.survival_rate:.2f}")
            print(f"Training steps: {best_model.training_steps:,}")
            print(f"Model size: {best_model.model_size_mb:.1f} MB")
            print(f"Training time: {best_model.training_time_seconds:.1f} seconds")
            print(f"Environment size: {best_model.environment_size}")
            print(f"Device: {best_model.device}")
            print(f"Framework: {best_model.framework}")
            print(f"Generation: {best_model.generation}")
            print(f"Timestamp: {best_model.timestamp}")
        else:
            print("No performance data available")


if __name__ == "__main__":
    main() 