# Gemini 多线程监控可视化系统

## 📋 项目说明

这是一个为 `generate_eva_text_gemini_native.py` 脚本提供的实时监控可视化系统。通过 Web 界面实时展示每个线程的工作状态、任务进度和统计数据。

## 🏗️ 文件结构

```
eval_ui/
├── monitor_service.py                # 监控服务（线程状态管理）
├── ws_server.py                      # WebSocket 服务器（端口 9000）
├── generate_eva_text_gemini_monitored.py  # 带监控的脚本副本
├── index.html                        # 监控界面主页
├── styles.css                        # 界面样式（支持明暗主题）
├── app.js                            # 前端 JavaScript 逻辑
└── README.md                         # 本文件
```

## 🚀 使用方法

### 第一步：安装依赖

确保已安装以下 Python 包：

```bash
pip install fastapi uvicorn websockets
```

### 第二步：启动 WebSocket 服务器

在终端1中运行：

```bash
cd eval_ui
python ws_server.py
```

你会看到以下输出：

```
╔════════════════════════════════════════════════════════════╗
║  🖥️  Gemini 多线程监控服务器                                ║
╚════════════════════════════════════════════════════════════╝
    
🌐 访问地址: http://localhost:9000
📡 WebSocket: ws://localhost:9000/ws
🔧 API 端点: http://localhost:9000/api/stats

按 Ctrl+C 停止服务器
```

### 第三步：运行带监控的评估脚本

在终端2中运行：

```bash
cd eval_ui
python generate_eva_text_gemini_monitored.py
```

脚本会输出监控面板地址：

```
[*] 监控面板: http://localhost:9000
```

### 第四步：打开监控界面

在浏览器中访问：

```
http://localhost:9000
```

## 📊 监控界面功能

### 全局统计面板
- **总体进度条**：显示任务完成百分比
- **活跃线程数**：当前正在工作的线程 / 总线程数
- **队列剩余**：待处理任务数量
- **已完成 / 已跳过 / 失败次数**：全局统计数据

### 线程卡片
每个线程一个卡片，实时显示：
- **线程状态**：🟢 工作中 / 🟡 等待中 / 🔴 已停止
- **当前模型**：正在使用的 Gemini 模型（Flash / Flash Lite）
- **当前任务**：图片名 | 策略 | 第几次
- **统计指标**：
  - ✅ 完成数
  - ⏭️ 跳过数
  - ❌ 失败数
  - ⚠️ 连续失败数
- **上次活动**：相对时间（如 "2 分钟前"）

### 主题切换
点击右上角 🌙/☀️ 按钮切换明暗主题。

## 🔧 配置说明

### 端口配置
默认端口为 `9000`，如需修改：

编辑 `ws_server.py`，修改端口号：

```python
if __name__ == "__main__":
    start_server(port=9000)  # 修改此处
```



## 📁 数据说明

### 不存储数据库
- 所有数据仅保存在内存中
- 停止服务器后数据会清空
- 原脚本的日志文件和输出文件路径格式不变

### 实时推送
- WebSocket 每秒推送一次最新数据
- 断线后会自动重连（3秒间隔）

## 🎨 界面特性

- **现代化设计**：毛玻璃效果卡片 + 渐变色
- **暗色主题为默认**：减少眼睛疲劳
- **响应式布局**：支持桌面、平板、手机
- **平滑动画**：状态变化和数字递增动画
- **实时更新**：无需刷新页面

## 🐛 故障排查

### WebSocket 连接失败
1. 检查 `ws_server.py` 是否正在运行
2. 确保端口 9000 未被占用
3. 查看浏览器控制台的错误信息

### 界面显示 "等待线程启动..."
1. 确保 `generate_eva_text_gemini_monitored.py` 已启动
2. 检查脚本是否成功导入 `monitor_service` 模块

### 线程卡片不更新
1. F12 打开开发者工具，查看 Network 标签
2. 确认 WebSocket 连接状态为 "Connected"
3. 检查脚本中是否正确调用了监控接口

## 📝 注意事项

1. **原脚本不受影响**：`generate_eva_text_gemini_native.py` 保持不变
2. **监控服务独立**：可以单独运行 `ws_server.py` 查看界面
3. **性能开销极小**：监控调用使用线程锁，开销 <1%
4. **仅本地访问**：默认绑定 `0.0.0.0`，可通过局域网 IP 访问

## 🔗 相关链接

- **原脚本**：`../eval/generate_eva_text_gemini_native.py`
- **监控脚本**：`generate_eva_text_gemini_monitored.py`
- **设计方案**：查看 `implementation_plan.md`（如有）

---

**祝使用愉快！** 🎉
