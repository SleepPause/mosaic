#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket 服务器：实时推送监控数据到前端
端口：9000

从文件读取数据以支持跨进程数据共享
"""
import asyncio
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# 获取当前脚本所在目录
CURRENT_DIR = Path(__file__).parent
DATA_FILE = CURRENT_DIR / "monitor_data.json"

app = FastAPI(title="Gemini 多线程监控")

# 挂载静态文件（前端资源）
app.mount("/static", StaticFiles(directory=CURRENT_DIR), name="static")


def load_stats_from_file() -> dict:
    """从文件加载统计数据"""
    try:
        if DATA_FILE.exists():
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        pass
    
    # 返回默认数据
    return {
        'threads': {},
        'global': {
            'active_threads': 0,
            'total_threads': 0,
            'queue_size': 0,
            'total_tasks': 0,
            'completed': 0,
            'skipped': 0,
            'failed': 0,
            'start_time': ''
        },
        'timestamp': ''
    }


@app.get("/")
async def root():
    """返回主页面"""
    return FileResponse(CURRENT_DIR / "index.html")


@app.get("/api/stats")
async def get_stats():
    """REST API: 获取当前统计数据（从文件读取）"""
    return load_stats_from_file()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点：实时推送数据"""
    await websocket.accept()
    print(f"✅ WebSocket 客户端已连接")
    
    try:
        while True:
            # 每秒推送最新数据（从文件读取）
            data = load_stats_from_file()
            await websocket.send_json(data)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print(f"❌ WebSocket 客户端已断开")
    except Exception as e:
        print(f"⚠️  WebSocket 错误: {e}")


def start_server(host: str = "0.0.0.0", port: int = 9000):
    """启动 WebSocket 服务器"""
    print(f"""
╔════════════════════════════════════════════════════════════╗
║  🖥️  Gemini 多线程监控服务器                                ║
╚════════════════════════════════════════════════════════════╝
    
🌐 访问地址: http://localhost:{port}
📡 WebSocket: ws://localhost:{port}/ws
🔧 API 端点: http://localhost:{port}/api/stats

按 Ctrl+C 停止服务器
""")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="warning"  # 减少日志输出
    )


if __name__ == "__main__":
    start_server()
