#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控服务：线程状态管理和 WebSocket 推送
独立运行，不影响原脚本

添加文件持久化：支持跨进程数据共享
"""
import threading
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


# 数据文件路径
DATA_FILE = Path(__file__).parent / "monitor_data.json"


class MonitorService:
    """线程状态监控服务（单例模式）"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.thread_stats: Dict[int, Dict[str, Any]] = {}
            self.global_stats = {
                'active_threads': 0,
                'total_threads': 0,
                'queue_size': 0,
                'total_tasks': 0,
                'completed': 0,
                'skipped': 0,
                'failed': 0,
                'start_time': datetime.now().isoformat()
            }
            self.stats_lock = threading.Lock()
            self.initialized = True
    
    def register_thread(self, key_id: int, model: str, total_threads: int, total_tasks: int):
        """注册新线程"""
        with self.stats_lock:
            self.thread_stats[key_id] = {
                'status': 'working',
                'current_model': model,
                'completed': 0,
                'skipped': 0,
                'failed': 0,
                'consecutive_failures': 0,
                'current_task': None,
                'last_activity': datetime.now().isoformat(),
                'start_time': datetime.now().isoformat(),
                'stop_reason': None  # 停止原因
            }
            self.global_stats['active_threads'] += 1
            self.global_stats['total_threads'] = total_threads
            self.global_stats['total_tasks'] = total_tasks
            self._save_to_file()  # 保存到文件
    
    def update_task_start(self, key_id: int, img_name: str, strategy: str, repeat_index: int):
        """更新任务开始状态"""
        with self.stats_lock:
            if key_id in self.thread_stats:
                self.thread_stats[key_id]['status'] = 'working'
                self.thread_stats[key_id]['current_task'] = {
                    'img_name': img_name,
                    'strategy': strategy,
                    'repeat_index': repeat_index
                }
                self.thread_stats[key_id]['last_activity'] = datetime.now().isoformat()
            self._save_to_file()  # 保存到文件
    
    def update_task_complete(self, key_id: int, success: bool = True, skipped: bool = False):
        """更新任务完成状态"""
        with self.stats_lock:
            if key_id in self.thread_stats:
                if success:
                    self.thread_stats[key_id]['completed'] += 1
                    self.thread_stats[key_id]['consecutive_failures'] = 0
                    self.global_stats['completed'] += 1
                elif skipped:
                    self.thread_stats[key_id]['skipped'] += 1
                    self.global_stats['skipped'] += 1
                else:
                    self.thread_stats[key_id]['failed'] += 1
                    self.thread_stats[key_id]['consecutive_failures'] += 1
                    self.global_stats['failed'] += 1
                
                self.thread_stats[key_id]['current_task'] = None
                self.thread_stats[key_id]['status'] = 'waiting'
                self.thread_stats[key_id]['last_activity'] = datetime.now().isoformat()
            self._save_to_file()  # 保存到文件
    
    def update_model(self, key_id: int, new_model: str):
        """更新线程使用的模型"""
        with self.stats_lock:
            if key_id in self.thread_stats:
                self.thread_stats[key_id]['current_model'] = new_model
            self._save_to_file()  # 保存到文件
    
    def update_queue_size(self, size: int):
        """更新队列剩余任务数"""
        with self.stats_lock:
            self.global_stats['queue_size'] = size
            self._save_to_file()  # 保存到文件
    
    def mark_thread_stopped(self, key_id: int, reason: str = '所有模型配额已用完'):
        """标记线程已停止并设置原因"""
        with self.stats_lock:
            if key_id in self.thread_stats:
                self.thread_stats[key_id]['status'] = 'stopped'
                self.thread_stats[key_id]['stop_reason'] = reason
                self.global_stats['active_threads'] -= 1
                self.thread_stats[key_id]['last_activity'] = datetime.now().isoformat()
            self._save_to_file()  # 保存到文件
    
    def _save_to_file(self):
        """保存数据到文件（内部方法，调用时已持有锁）"""
        try:
            data = {
                'threads': dict(self.thread_stats),
                'global': dict(self.global_stats),
                'timestamp': datetime.now().isoformat()
            }
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            pass  # 静默处理文件写入错误
    
    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有统计数据"""
        with self.stats_lock:
            return {
                'threads': dict(self.thread_stats),
                'global': dict(self.global_stats),
                'timestamp': datetime.now().isoformat()
            }
    
    def reset(self):
        """重置所有数据（用于新任务）"""
        with self.stats_lock:
            self.thread_stats.clear()
            self.global_stats = {
                'active_threads': 0,
                'total_threads': 0,
                'queue_size': 0,
                'total_tasks': 0,
                'completed': 0,
                'skipped': 0,
                'failed': 0,
                'start_time': datetime.now().isoformat()
            }


# 全局单例实例
monitor = MonitorService()
