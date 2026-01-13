// 添加到 app.js 末尾的补丁代码

// ===== 更新过滤器计数 =====
function updateFilterCounts(threads) {
    const counts = { working: 0, waiting: 0, stopped: 0 };

    Object.values(threads).forEach(thread => {
        if (thread.status in counts) {
            counts[thread.status]++;
        }
    });

    // 更新计数显示（如果存在）
    const workingCount = document.getElementById('count-working');
    const waitingCount = document.getElementById('count-waiting');
    const stoppedCount = document.getElementById('count-stopped');

    if (workingCount) workingCount.textContent = `(${counts.working})`;
    if (waitingCount) waitingCount.textContent = `(${counts.waiting})`;
    if (stoppedCount) stoppedCount.textContent = `(${counts.stopped})`;
}

// 重新应用过滤器（修复版本）
filterManager.applyFilters = function () {
    const cards = document.querySelectorAll('.thread-card');
    let anyFilterActive = this.filters.working || this.filters.waiting || this.filters.stopped;

    cards.forEach(card => {
        const status = card.classList.contains('working') ? 'working' :
            card.classList.contains('waiting') ? 'waiting' : 'stopped';

        // 如果没有任何过滤器选中，隐藏所有卡片
        if (!anyFilterActive) {
            card.classList.add('filtered-out');
        } else if (this.filters[status]) {
            card.classList.remove('filtered-out');
        } else {
            card.classList.add('filtered-out');
        }
    });
};
