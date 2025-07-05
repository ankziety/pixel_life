#!/usr/bin/env python3
"""
Pixel Life Log Management Tools (modularized)

This module provides the PixelLogManager class and all log management/inspection logic for use in the main CLI or other modules.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import sqlite3
import glob
import shutil
from collections import defaultdict, Counter

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class PixelLogManager:
    """Enhanced log manager with CLI-friendly features."""
    
    def __init__(self, logs_dir: str = "./logs"):
        self.logs_dir = Path(logs_dir)
        self.db_path = self.logs_dir / "log_metadata.db"
        
    def get_log_overview(self) -> Dict[str, Any]:
        # ... (rest of PixelLogManager as in pixel_logs.py) ...
        overview = {
            'total_files': 0,
            'total_size_mb': 0,
            'log_types': Counter(),
            'recent_activity': [],
            'performance_models': [],
            'errors': [],
            'warnings': []
        }
        if not self.logs_dir.exists():
            return overview
        for file_path in self.logs_dir.rglob("*"):
            if file_path.is_file():
                overview['total_files'] += 1
                overview['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
                if file_path.suffix:
                    overview['log_types'][str(file_path.suffix)] += 1
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime > datetime.now() - timedelta(hours=24):
                    overview['recent_activity'].append({
                        'file': str(file_path.relative_to(self.logs_dir)),
                        'modified': mtime.isoformat(),
                        'size_mb': file_path.stat().st_size / (1024 * 1024)
                    })
        if self.db_path.exists():
            overview.update(self._get_database_stats())
        return overview
    def _get_database_stats(self) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT model_path, episode_reward, survival_rate, timestamp, generation
                FROM model_performance 
                ORDER BY episode_reward DESC 
                LIMIT 10
            ''')
            performance_models = [
                {
                    'path': row[0],
                    'reward': row[1],
                    'survival': row[2],
                    'timestamp': row[3],
                    'generation': row[4]
                }
                for row in cursor.fetchall()
            ]
            cursor.execute('''
                SELECT level, message, timestamp, module, function
                FROM log_entries 
                WHERE level IN ('ERROR', 'WARNING')
                ORDER BY timestamp DESC 
                LIMIT 20
            ''')
            errors = []
            warnings = []
            for row in cursor.fetchall():
                entry = {
                    'level': row[0],
                    'message': row[1],
                    'timestamp': row[2],
                    'module': row[3],
                    'function': row[4]
                }
                if row[0] == 'ERROR':
                    errors.append(entry)
                else:
                    warnings.append(entry)
            conn.close()
            return {
                'performance_models': performance_models,
                'errors': errors,
                'warnings': warnings
            }
        except Exception as e:
            return {
                'performance_models': [],
                'errors': [],
                'warnings': [],
                'db_error': str(e)
            }
    def show_overview(self):
        overview = self.get_log_overview()
        print(f"{Colors.HEADER}📊 Pixel Life Log Overview{Colors.ENDC}")
        print("=" * 60)
        print(f"{Colors.BOLD}📁 Total Files:{Colors.ENDC} {overview['total_files']:,}")
        print(f"{Colors.BOLD}💾 Total Size:{Colors.ENDC} {overview['total_size_mb']:.2f} MB")
        if overview['log_types']:
            print(f"\n{Colors.BOLD}📋 File Types:{Colors.ENDC}")
            for file_type, count in overview['log_types'].most_common():
                print(f"  {file_type}: {count:,} files")
        if overview['recent_activity']:
            print(f"\n{Colors.BOLD}🕒 Recent Activity (Last 24h):{Colors.ENDC}")
            for activity in overview['recent_activity'][:5]:
                print(f"  {activity['file']} ({activity['size_mb']:.2f} MB)")
        if overview['performance_models']:
            print(f"\n{Colors.BOLD}🏆 Top Performing Models:{Colors.ENDC}")
            for i, model in enumerate(overview['performance_models'][:5], 1):
                print(f"  {i}. Gen {model['generation']}: Reward {model['reward']:.2f}, "
                      f"Survival {model['survival']:.2f}")
        if overview['errors']:
            print(f"\n{Colors.FAIL}❌ Recent Errors:{Colors.ENDC}")
            for error in overview['errors'][:3]:
                print(f"  {error['timestamp']}: {error['message']}")
        if overview['warnings']:
            print(f"\n{Colors.WARNING}⚠️  Recent Warnings:{Colors.ENDC}")
            for warning in overview['warnings'][:3]:
                print(f"  {warning['timestamp']}: {warning['message']}")
    def search_logs(self, query: str, level: Optional[str] = None, limit: int = 20):
        if not self.db_path.exists():
            print(f"{Colors.FAIL}❌ No log database found. Run training first.{Colors.ENDC}")
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            sql = '''
                SELECT timestamp, level, message, module, function, line, request_id
                FROM log_entries 
                WHERE message LIKE ?
            '''
            params = [f'%{query}%']
            if level:
                sql += ' AND level = ?'
                params.append(level.upper())
            sql += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(str(limit))
            cursor.execute(sql, params)
            results = cursor.fetchall()
            conn.close()
            if not results:
                print(f"{Colors.WARNING}🔍 No logs found matching '{query}'{Colors.ENDC}")
                return
            print(f"{Colors.HEADER}🔍 Search Results for '{query}'{Colors.ENDC}")
            print("=" * 80)
            for timestamp, level, message, module, function, line, request_id in results:
                level_color = Colors.FAIL if level == 'ERROR' else \
                             Colors.WARNING if level == 'WARNING' else \
                             Colors.OKGREEN if level == 'INFO' else Colors.OKBLUE
                print(f"{level_color}[{level}]{Colors.ENDC} {timestamp}")
                print(f"  📝 {message}")
                print(f"  📍 {module}.{function}:{line}")
                if request_id:
                    print(f"  🆔 Request: {request_id}")
                print()
        except Exception as e:
            print(f"{Colors.FAIL}❌ Error searching logs: {e}{Colors.ENDC}")
    def show_performance(self, limit: int = 10):
        if not self.db_path.exists():
            print(f"{Colors.FAIL}❌ No performance data found. Run training first.{Colors.ENDC}")
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT generation, episode_reward, survival_rate, training_steps,
                       model_size_mb, training_time_seconds, device, timestamp
                FROM model_performance 
                ORDER BY episode_reward DESC 
                LIMIT ?
            ''', (limit,))
            results = cursor.fetchall()
            conn.close()
            if not results:
                print(f"{Colors.WARNING}📊 No performance data available{Colors.ENDC}")
                return
            print(f"{Colors.HEADER}📈 Model Performance History{Colors.ENDC}")
            print("=" * 100)
            print(f"{'Gen':<4} {'Reward':<8} {'Survival':<8} {'Steps':<8} {'Size(MB)':<8} {'Time(s)':<8} {'Device':<6} {'Date'}")
            print("-" * 100)
            for gen, reward, survival, steps, size, time, device, timestamp in results:
                if reward > 100:
                    color = Colors.OKGREEN
                elif reward > 50:
                    color = Colors.OKBLUE
                elif reward > 0:
                    color = Colors.WARNING
                else:
                    color = Colors.FAIL
                date_str = timestamp.split('T')[0] if 'T' in timestamp else timestamp[:10]
                print(f"{color}{gen:<4} {reward:<8.2f} {survival:<8.2f} {steps:<8} "
                      f"{size:<8.1f} {time:<8.1f} {device:<6} {date_str}{Colors.ENDC}")
            best_reward = max(r[1] for r in results)
            best_gen = next(r[0] for r in results if r[1] == best_reward)
            print(f"\n{Colors.BOLD}🏆 Best Model: Generation {best_gen} (Reward: {best_reward:.2f}){Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}❌ Error loading performance data: {e}{Colors.ENDC}")
    def cleanup_logs(self, days: int = 30, dry_run: bool = True):
        cutoff_date = datetime.now() - timedelta(days=days)
        files_to_clean = []
        total_size = 0
        if not self.logs_dir.exists():
            print(f"{Colors.WARNING}📁 No logs directory found{Colors.ENDC}")
            return
        for file_path in self.logs_dir.rglob("*"):
            if file_path.is_file():
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime < cutoff_date:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    files_to_clean.append((file_path, mtime, size_mb))
                    total_size += size_mb
        if not files_to_clean:
            print(f"{Colors.OKGREEN}✅ No old files to clean up{Colors.ENDC}")
            return
        print(f"{Colors.HEADER}🧹 Log Cleanup{Colors.ENDC}")
        print("=" * 50)
        print(f"Files older than {days} days: {len(files_to_clean)}")
        print(f"Total size to free: {total_size:.2f} MB")
        if dry_run:
            print(f"\n{Colors.WARNING}🔍 Dry run - no files will be deleted{Colors.ENDC}")
            print("Oldest files to be cleaned:")
            for file_path, mtime, size_mb in files_to_clean[:10]:
                print(f"  {file_path.name} ({size_mb:.2f} MB) - {mtime.date()}")
        else:
            archive_dir = self.logs_dir / "archive"
            archive_dir.mkdir(exist_ok=True)
            cleaned = 0
            for file_path, mtime, size_mb in files_to_clean:
                try:
                    archive_path = archive_dir / f"{mtime.strftime('%Y%m%d')}_{file_path.name}"
                    shutil.move(str(file_path), str(archive_path))
                    cleaned += 1
                except Exception as e:
                    print(f"{Colors.FAIL}❌ Error moving {file_path.name}: {e}{Colors.ENDC}")
            print(f"{Colors.OKGREEN}✅ Cleaned up {cleaned} files{Colors.ENDC}")
    def show_errors(self, limit: int = 20):
        if not self.db_path.exists():
            print(f"{Colors.FAIL}❌ No log database found{Colors.ENDC}")
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, level, message, module, function, line
                FROM log_entries 
                WHERE level IN ('ERROR', 'WARNING')
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            results = cursor.fetchall()
            conn.close()
            if not results:
                print(f"{Colors.OKGREEN}✅ No errors or warnings found{Colors.ENDC}")
                return
            print(f"{Colors.HEADER}⚠️  Recent Errors and Warnings{Colors.ENDC}")
            print("=" * 80)
            for timestamp, level, message, module, function, line in results:
                color = Colors.FAIL if level == 'ERROR' else Colors.WARNING
                print(f"{color}[{level}]{Colors.ENDC} {timestamp}")
                print(f"  📝 {message}")
                print(f"  📍 {module}.{function}:{line}")
                print()
        except Exception as e:
            print(f"{Colors.FAIL}❌ Error loading logs: {e}{Colors.ENDC}") 