// ===== WebSocket 连接管理 =====
let ws = null;
let reconnectTimer = null;
const WS_URL = `ws://${window.location.host}/ws`;

// ===== DOM 元素引用 =====
const elements = {
    // 连接状态
    connectionStatus: document.getElementById('connection-status'),
    statusDot: null,
    statusText: null,

    // 全局统计
    progressBar: document.getElementById('progress-bar'),
    progressPercentage: document.getElementById('progress-percentage'),
    progressDetails: document.getElementById('progress-details'),
    activeThreads: document.getElementById('active-threads'),
    totalThreads: document.getElementById('total-threads'),
    queueSize: document.getElementById('queue-size'),
    completed: document.getElementById('completed'),
    skipped: document.getElementById('skipped'),
    failed: document.getElementById('failed'),

    // 线程网格
    threadsGrid: document.getElementById('threads-grid'),

    // 主题切换
    themeToggle: document.getElementById('theme-toggle')
};

// 初始化状态元素
elements.statusDot = elements.connectionStatus.querySelector('.status-dot');
elements.statusText = elements.connectionStatus.querySelector('.status-text');

// ===== 过滤器管理 =====
const filterManager = {
    filters: {
        working: true,
        waiting: true,
        stopped: true
    },

    init() {
        document.getElementById('filter-working').addEventListener('change', (e) => {
            this.filters.working = e.target.checked;
            this.applyFilters();
        });

        document.getElementById('filter-waiting').addEventListener('change', (e) => {
            this.filters.waiting = e.target.checked;
            this.applyFilters();
        });

        document.getElementById('filter-stopped').addEventListener('change', (e) => {
            this.filters.stopped = e.target.checked;
            this.applyFilters();
        });
    },

    applyFilters() {
        const cards = document.querySelectorAll('.thread-card');
        const anyFilterActive = this.filters.working || this.filters.waiting || this.filters.stopped;

        cards.forEach(card => {
            // 获取卡片状态
            let status = '';
            if (card.classList.contains('working')) status = 'working';
            else if (card.classList.contains('waiting')) status = 'waiting';
            else if (card.classList.contains('stopped')) status = 'stopped';

            // 如果没有任何过滤器选中，隐藏所有
            if (!anyFilterActive) {
                card.classList.add('filtered-out');
            } else if (this.filters[status]) {
                card.classList.remove('filtered-out');
            } else {
                card.classList.add('filtered-out');
            }
        });
    }
};

// ===== 主题切换功能 =====
const themeManager = {
    isDark: true,

    init() {
        // 从 localStorage 读取主题
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {
            this.setTheme('light');
        }

        // 绑定切换按钮
        elements.themeToggle.addEventListener('click', () => this.toggle());
    },

    toggle() {
        this.setTheme(this.isDark ? 'light' : 'dark');
    },

    setTheme(theme) {
        this.isDark = theme === 'dark';

        if (this.isDark) {
            document.body.classList.add('dark-theme');
            elements.themeToggle.textContent = '🌙';
        } else {
            document.body.classList.remove('dark-theme');
            elements.themeToggle.textContent = '☀️';
        }

        localStorage.setItem('theme', theme);
    }
};

// ===== WebSocket 连接 =====
function connectWebSocket() {
    try {
        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log('✅ WebSocket 已连接');
            updateConnectionStatus('connected');
            if (reconnectTimer) {
                clearTimeout(reconnectTimer);
                reconnectTimer = null;
            }
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                updateUI(data);
            } catch (error) {
                console.error('❌ 解析数据失败:', error);
            }
        };

        ws.onerror = (error) => {
            console.error('❌ WebSocket 错误:', error);
            updateConnectionStatus('disconnected');
        };

        ws.onclose = () => {
            console.warn('⚠️  WebSocket 已断开，3秒后尝试重连...');
            updateConnectionStatus('disconnected');
            reconnectTimer = setTimeout(connectWebSocket, 3000);
        };
    } catch (error) {
        console.error('❌ WebSocket 连接失败:', error);
        updateConnectionStatus('disconnected');
        reconnectTimer = setTimeout(connectWebSocket, 3000);
    }
}

// ===== 更新连接状态 =====
function updateConnectionStatus(status) {
    elements.statusDot.className = 'status-dot ' + status;

    const statusTexts = {
        'connected': '已连接',
        'disconnected': '未连接',
        'connecting': '连接中...'
    };

    elements.statusText.textContent = statusTexts[status] || '未知';
}

// ===== 更新 UI =====
function updateUI(data) {
    updateGlobalStats(data.global);
    updateThreadCards(data.threads);
}

// ===== 更新全局统计 =====
function updateGlobalStats(global) {
    // 计算进度
    const total = global.total_tasks || 1;
    const done = global.completed + global.skipped;
    const percentage = Math.min(Math.round((done / total) * 100), 100);

    // 更新进度条
    elements.progressBar.style.width = percentage + '%';
    elements.progressPercentage.textContent = percentage + '%';
    elements.progressDetails.textContent = `${done.toLocaleString()} / ${total.toLocaleString()}`;

    // 更新统计数据（带动画）
    animateValue(elements.activeThreads, global.active_threads);
    animateValue(elements.totalThreads, global.total_threads);
    // 队列剩余 = 总任务数 - 已完成 - 已跳过
    const queueRemaining = Math.max(0, global.total_tasks - global.completed - global.skipped);
    animateValue(elements.queueSize, queueRemaining);
    animateValue(elements.completed, global.completed);
    animateValue(elements.skipped, global.skipped);
    animateValue(elements.failed, global.failed);
}

// ===== 数字递增动画 =====
function animateValue(element, newValue) {
    const current = parseInt(element.textContent) || 0;
    if (current === newValue) return;

    element.textContent = newValue.toLocaleString();
}

// ===== 更新线程卡片 =====
function updateThreadCards(threads) {
    const threadIds = Object.keys(threads).sort((a, b) => parseInt(a) - parseInt(b));

    if (threadIds.length === 0) {
        if (!elements.threadsGrid.querySelector('.empty-state')) {
            elements.threadsGrid.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">🔄</div>
                    <p>等待线程启动...</p>
                </div>
            `;
        }
        return;
    }

    // 清空空状态
    if (elements.threadsGrid.querySelector('.empty-state')) {
        elements.threadsGrid.innerHTML = '';
    }

    // 更新或创建线程卡片
    threadIds.forEach(keyId => {
        const thread = threads[keyId];
        const cardId = `thread-${keyId}`;
        let card = document.getElementById(cardId);

        if (!card) {
            card = createThreadCard(keyId, thread);
            elements.threadsGrid.appendChild(card);
        } else {
            updateThreadCard(card, thread);
        }
    });

    // 更新状态计数 (Bug 2 修复)
    updateFilterCounts(threads);

    // 应用过滤器 (Bug 1 修复)
    filterManager.applyFilters();
}

// ===== 创建线程卡片 =====
function createThreadCard(keyId, thread) {
    const card = document.createElement('div');
    card.id = `thread-${keyId}`;
    card.className = `thread-card ${thread.status}`;
    card.innerHTML = getThreadCardHTML(keyId, thread);
    return card;
}

// ===== 更新线程卡片 =====
function updateThreadCard(card, thread) {
    // 更新状态类
    card.className = `thread-card ${thread.status}`;

    // 更新状态标签
    const statusBadge = card.querySelector('.thread-status');
    statusBadge.className = `thread-status ${thread.status}`;
    statusBadge.innerHTML = getStatusHTML(thread.status);

    // 更新模型
    const modelBadge = card.querySelector('.thread-model');
    modelBadge.textContent = formatModelName(thread.current_model);

    // 更新当前任务
    const taskContent = card.querySelector('.task-content');
    if (thread.current_task) {
        const task = thread.current_task;
        taskContent.innerHTML = `
            <strong>${task.img_name}</strong><br>
            ${task.strategy} | 第 ${task.repeat_index} 次
        `;
        taskContent.className = 'task-content';
    } else {
        taskContent.textContent = thread.status === 'stopped' ? '已停止' : '等待任务...';
        taskContent.className = 'task-content task-empty';
    }

    // 更新统计数据（不显示跳过数）
    card.querySelector('[data-stat="completed"]').textContent = thread.completed;
    card.querySelector('[data-stat="failed"]').textContent = thread.failed;
    card.querySelector('[data-stat="consecutive"]').textContent = thread.consecutive_failures;

    // Bug 3 修复: 更新停止原因（如果是已停止状态）
    if (thread.status === 'stopped' && thread.stop_reason) {
        let stopReasonEl = card.querySelector('.stop-reason');
        if (!stopReasonEl) {
            // 如果不存在，创建元素
            stopReasonEl = document.createElement('div');
            stopReasonEl.className = 'stop-reason';
            // 插入到最后活动时间之前
            const activityEl = card.querySelector('.thread-activity');
            activityEl.parentNode.insertBefore(stopReasonEl, activityEl);
        }
        stopReasonEl.textContent = `停止原因: ${thread.stop_reason}`;
    } else {
        // 移除停止原因元素（如果存在）
        const stopReasonEl = card.querySelector('.stop-reason');
        if (stopReasonEl) {
            stopReasonEl.remove();
        }
    }

    // 更新最后活动时间
    const activityTime = card.querySelector('.thread-activity');
    activityTime.textContent = `最后活动: ${formatTime(thread.last_activity)}`;
}

// ===== 生成线程卡片 HTML =====
function getThreadCardHTML(keyId, thread) {
    const task = thread.current_task;
    const taskHTML = task
        ? `<strong>${task.img_name}</strong><br>${task.strategy} | 第 ${task.repeat_index} 次`
        : (thread.status === 'stopped' ? '已停止' : '等待任务...');

    const taskClass = task ? '' : 'task-empty';

    // Bug 3 修复: 添加停止原因显示
    const stopReasonHTML = (thread.status === 'stopped' && thread.stop_reason)
        ? `<div class="stop-reason">停止原因: ${thread.stop_reason}</div>`
        : '';

    return `
        <div class="thread-header">
            <div class="thread-id">Key-${keyId}</div>
            <div class="thread-status ${thread.status}">
                ${getStatusHTML(thread.status)}
            </div>
        </div>
        
        <div class="thread-model">${formatModelName(thread.current_model)}</div>
        
        <div class="thread-task">
            <div class="task-label">当前任务</div>
            <div class="task-content ${taskClass}">${taskHTML}</div>
        </div>
        
        <div class="thread-stats">
            <div class="thread-stat">
                <span class="thread-stat-icon">✅</span>
                <span>完成: <span class="thread-stat-value" data-stat="completed">${thread.completed}</span></span>
            </div>
            <div class="thread-stat">
                <span class="thread-stat-icon">❌</span>
                <span>失败: <span class="thread-stat-value" data-stat="failed">${thread.failed}</span></span>
            </div>
            <div class="thread-stat consecutive-failures">
                <span class="thread-stat-icon">⚠️</span>
                <span>连续失败: <span class="thread-stat-value" data-stat="consecutive">${thread.consecutive_failures}</span></span>
            </div>
        </div>
        
        ${stopReasonHTML}
        
        <div class="thread-activity">最后活动: ${formatTime(thread.last_activity)}</div>
    `;
}

// ===== 获取状态 HTML =====
function getStatusHTML(status) {
    const statusMap = {
        'working': '<span class="status-icon"></span>工作中',
        'waiting': '<span class="status-icon"></span>等待中',
        'stopped': '<span class="status-icon"></span>已停止'
    };
    return statusMap[status] || status;
}

// ===== 格式化模型名称 =====
function formatModelName(model) {
    const nameMap = {
        'gemini-2.5-flash': 'Flash',
        'gemini-2.5-flash-lite': 'Flash Lite',
        'gemini-2.0-flash-exp': 'Flash Exp'
    };
    return nameMap[model] || model;
}

// ===== 格式化时间（相对时间）=====
function formatTime(isoString) {
    try {
        const date = new Date(isoString);
        const now = new Date();
        const diff = Math.floor((now - date) / 1000); // 秒

        if (diff < 60) return `${diff} 秒前`;
        if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`;
        if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`;
        return `${Math.floor(diff / 86400)} 天前`;
    } catch {
        return '未知';
    }
}

// Bug 2 修复: 更新过滤器计数
function updateFilterCounts(threads) {
    const counts = { working: 0, waiting: 0, stopped: 0 };

    Object.values(threads).forEach(thread => {
        if (thread.status in counts) {
            counts[thread.status]++;
        }
    });

    // 更新计数显示
    const workingCount = document.getElementById('count-working');
    const waitingCount = document.getElementById('count-waiting');
    const stoppedCount = document.getElementById('count-stopped');

    if (workingCount) workingCount.textContent = `(${counts.working})`;
    if (waitingCount) waitingCount.textContent = `(${counts.waiting})`;
    if (stoppedCount) stoppedCount.textContent = `(${counts.stopped})`;
}

// ===== 初始化应用 =====
function init() {
    console.log('🚀 初始化监控面板...');
    themeManager.init();
    filterManager.init();
    connectWebSocket();
}

// 页面加载完成后初始化
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
