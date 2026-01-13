#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试监控服务是否工作
"""
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from monitor_service import monitor
import time

# 重置并初始化
monitor.reset()
monitor.register_thread(0, "gemini-2.5-flash", 3, 100)
monitor.register_thread(1, "gemini-2.5-flash", 3, 100)
monitor.register_thread(2, "gemini-2.5-flash-lite", 3, 100)

# 模拟一些任务活动
for i in range(5):
    monitor.update_task_start(0, f"test_image_{i}.jpg", "JSON", i+1)
    time.sleep(0.5)
    monitor.update_task_complete(0, success=True, skipped=False)
    monitor.update_queue_size(100 - i*10)

# 输出当前状态
stats = monitor.get_all_stats()
print(f"线程数量: {len(stats['threads'])}")
print(f"全局统计: {stats['global']}")
print(f"线程0统计: {stats['threads'].get(0, {})}")
