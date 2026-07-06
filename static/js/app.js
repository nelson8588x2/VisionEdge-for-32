/**
 * VisionEdge Web — 前端應用程式
 * 處理攝影機擷取、UI 互動和 API 通訊。
 */

// ============================================================
// 全域狀態
// ============================================================
const state = {
    currentMode: 'calibration',
    cameraRunning: false,
    isCalibrated: false,
    processing: false,
    // 選取框（Chat 模式）
    selection: null,
    isSelecting: false,
    selStart: null,
    // FPS 計算
    fps: 0,
    // 處理間隔
    processInterval: null,
};

// ============================================================
// DOM 元素
// ============================================================
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
    video: $('#video-feed'),
    canvas: $('#canvas-capture'),
    imgOriginal: $('#img-original'),
    imgCorrected: $('#img-corrected'),
    videoPlaceholder: $('#video-placeholder'),
    correctedPlaceholder: $('#corrected-placeholder'),
    canvasSelection: $('#canvas-selection'),
    // 面板
    panelOriginal: $('#panel-original'),
    panelCorrected: $('#panel-corrected'),
    // 狀態
    fpsLabel: $('#fps-label'),
    camInfo: $('#cam-info'),
    statusDot: $('#status-dot'),
    statusText: $('#status-text'),
    // 按鈕
    btnReset: $('#btn-reset'),
    btnCrop: $('#btn-crop'),
    btnChatSend: $('#btn-chat-send'),
    chatInput: $('#chat-input'),
    chatHistory: $('#chat-history'),
};

// ============================================================
// 攝影機管理（自動啟動）
// ============================================================
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'environment',
            }
        });
        els.video.srcObject = stream;
        els.video.classList.remove('hidden');
        els.videoPlaceholder.classList.add('hidden');

        // 取得攝影機資訊
        const track = stream.getVideoTracks()[0];
        const settings = track.getSettings();
        els.camInfo.textContent = `Camera: ${settings.width}x${settings.height}`;

        state.cameraRunning = true;

        // 開始處理迴圈
        startProcessingLoop();
    } catch (err) {
        console.error('攝影機啟動失敗:', err);
        els.videoPlaceholder.textContent = `攝影機錯誤: ${err.message}`;
        alert(`無法開啟攝影機: ${err.message}\n\n請確認已授予攝影機權限。`);
    }
}

// ============================================================
// 影像擷取
// ============================================================
function captureFrame() {
    if (!state.cameraRunning || !els.video.videoWidth) return null;

    const canvas = els.canvas;
    const ctx = canvas.getContext('2d');
    canvas.width = els.video.videoWidth;
    canvas.height = els.video.videoHeight;
    ctx.drawImage(els.video, 0, 0);

    return canvas.toDataURL('image/jpeg', 0.65);
}

// ============================================================
// 處理迴圈
// ============================================================
function startProcessingLoop() {
    stopProcessingLoop();
    // 每 150ms 發送一幀到伺服器進行校正（約 6-7 FPS）
    state.processInterval = setInterval(() => {
        if (!state.processing) {
            processFrame();
        }
    }, 150);
}

function stopProcessingLoop() {
    if (state.processInterval) {
        clearInterval(state.processInterval);
        state.processInterval = null;
    }
}

async function processFrame() {
    const imageData = captureFrame();
    if (!imageData) return;

    state.processing = true;
    const startTime = performance.now();

    try {
        const resp = await fetch('/api/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: imageData,
                wb_enabled: true,
                contrast_enabled: false,
            }),
        });

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        // 更新原始影像顯示
        if (data.original_display) {
            els.imgOriginal.src = `data:image/jpeg;base64,${data.original_display}`;
            els.imgOriginal.classList.remove('hidden');
            els.video.classList.add('hidden');
        }

        // 更新校正後影像顯示
        if (data.corrected_display) {
            els.imgCorrected.src = `data:image/jpeg;base64,${data.corrected_display}`;
            els.imgCorrected.classList.remove('hidden');
            els.correctedPlaceholder.classList.add('hidden');
        } else {
            els.imgCorrected.classList.add('hidden');
            els.correctedPlaceholder.classList.remove('hidden');
            els.correctedPlaceholder.textContent = '等待校正...';
        }

        // 更新狀態
        updateStatus(data.status_type, data.status);
        state.isCalibrated = data.is_calibrated;

        // 更新 FPS
        const elapsed = performance.now() - startTime;
        state.fps = 0.8 * state.fps + 0.2 * (1000 / elapsed);
        els.fpsLabel.textContent = `FPS: ${state.fps.toFixed(0)}`;

    } catch (err) {
        console.error('處理失敗:', err);
        updateStatus('error', `伺服器錯誤: ${err.message}`);
    } finally {
        state.processing = false;
    }
}

// ============================================================
// 視覺問答 (Chat)
// ============================================================
async function sendChatQuestion() {
    const question = els.chatInput.value.trim();
    if (!question) return;

    // 取得選取區域或整幀
    let imageData;
    if (state.selection) {
        imageData = getCroppedSelection();
    } else {
        imageData = captureFrame();
    }
    if (!imageData) return;

    appendChatMessage('user', question);
    els.chatInput.value = '';
    els.btnChatSend.disabled = true;
    appendChatMessage('loading', 'Thinking...');

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData, question }),
        });

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        removeLastChatMessage(); // 移除 loading
        appendChatMessage('ai', data.answer || data.status);
    } catch (err) {
        removeLastChatMessage();
        appendChatMessage('error', `錯誤: ${err.message}`);
    } finally {
        els.btnChatSend.disabled = false;
    }
}

function appendChatMessage(role, text) {
    const div = document.createElement('div');
    if (role === 'user') {
        div.className = 'chat-msg-user';
        div.innerHTML = `<span class="chat-role">You:</span><p class="text-xs text-light-700 mt-1">${escapeHtml(text)}</p>`;
    } else if (role === 'ai') {
        div.className = 'chat-msg-ai';
        div.innerHTML = `<span class="chat-role">AI:</span><p class="text-xs text-light-800 mt-1">${escapeHtml(text).replace(/\n/g, '<br>')}</p>`;
    } else if (role === 'loading') {
        div.className = 'chat-msg-ai loading-dot';
        div.innerHTML = `<span class="chat-role">AI:</span><p class="text-xs text-light-600 mt-1 italic">${text}</p>`;
    } else {
        div.className = 'chat-msg-error';
        div.innerHTML = `<p class="text-xs">${escapeHtml(text)}</p>`;
    }
    els.chatHistory.appendChild(div);
    els.chatHistory.scrollTop = els.chatHistory.scrollHeight;
}

function removeLastChatMessage() {
    const last = els.chatHistory.lastElementChild;
    if (last) last.remove();
}

function escapeHtml(text) {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ============================================================
// 選取框（Chat 模式的區域選取）
// ============================================================
function setupSelectionCanvas() {
    const canvas = els.canvasSelection;
    const container = canvas.parentElement;

    canvas.addEventListener('mousedown', (e) => {
        if (state.currentMode !== 'chat') return;
        const rect = canvas.getBoundingClientRect();
        state.isSelecting = true;
        state.selStart = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        state.selection = null;
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!state.isSelecting) return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        drawSelectionRect(state.selStart.x, state.selStart.y, x, y);
    });

    canvas.addEventListener('mouseup', (e) => {
        if (!state.isSelecting) return;
        state.isSelecting = false;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const w = Math.abs(x - state.selStart.x);
        const h = Math.abs(y - state.selStart.y);
        if (w > 10 && h > 10) {
            state.selection = {
                x: Math.min(state.selStart.x, x),
                y: Math.min(state.selStart.y, y),
                w, h,
                canvasW: canvas.width,
                canvasH: canvas.height,
            };
            els.chatInput.focus();
        }
    });

    // 觸控支援
    canvas.addEventListener('touchstart', (e) => {
        if (state.currentMode !== 'chat') return;
        e.preventDefault();
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        state.isSelecting = true;
        state.selStart = { x: touch.clientX - rect.left, y: touch.clientY - rect.top };
        state.selection = null;
    });

    canvas.addEventListener('touchmove', (e) => {
        if (!state.isSelecting) return;
        e.preventDefault();
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        const x = touch.clientX - rect.left;
        const y = touch.clientY - rect.top;
        drawSelectionRect(state.selStart.x, state.selStart.y, x, y);
    });

    canvas.addEventListener('touchend', (e) => {
        if (!state.isSelecting) return;
        state.isSelecting = false;
        const touch = e.changedTouches[0];
        const rect = canvas.getBoundingClientRect();
        const x = touch.clientX - rect.left;
        const y = touch.clientY - rect.top;
        const w = Math.abs(x - state.selStart.x);
        const h = Math.abs(y - state.selStart.y);
        if (w > 10 && h > 10) {
            state.selection = {
                x: Math.min(state.selStart.x, x),
                y: Math.min(state.selStart.y, y),
                w, h,
                canvasW: canvas.width,
                canvasH: canvas.height,
            };
            els.chatInput.focus();
        }
    });
}

function drawSelectionRect(x1, y1, x2, y2) {
    const canvas = els.canvasSelection;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#4a5aff';
    ctx.lineWidth = 2;
    ctx.fillStyle = 'rgba(74, 90, 255, 0.15)';
    const rx = Math.min(x1, x2);
    const ry = Math.min(y1, y2);
    const rw = Math.abs(x2 - x1);
    const rh = Math.abs(y2 - y1);
    ctx.fillRect(rx, ry, rw, rh);
    ctx.strokeRect(rx, ry, rw, rh);
}

function getCroppedSelection() {
    if (!state.selection || !els.imgCorrected.src) return captureFrame();

    // 使用校正後影像作為選取來源
    const img = els.imgCorrected;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // 計算比例：選取框座標 → 影像像素
    const dispW = els.canvasSelection.clientWidth;
    const dispH = els.canvasSelection.clientHeight;
    const scaleX = img.naturalWidth / dispW;
    const scaleY = img.naturalHeight / dispH;

    const sx = state.selection.x * scaleX;
    const sy = state.selection.y * scaleY;
    const sw = state.selection.w * scaleX;
    const sh = state.selection.h * scaleY;

    canvas.width = sw;
    canvas.height = sh;
    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);

    return canvas.toDataURL('image/jpeg', 0.85);
}

// ============================================================
// 狀態更新
// ============================================================
function updateStatus(type, text) {
    els.statusDot.className = `w-3 h-3 rounded-full dot-${type}`;
    els.statusText.textContent = text;
}

// ============================================================
// 模式切換
// ============================================================
function switchMode(mode) {
    state.currentMode = mode;

    // 更新標籤按鈕
    $$('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    // 隱藏所有模式面板
    $$('.mode-content').forEach(el => el.classList.add('hidden'));
    $(`#mode-${mode}`).classList.remove('hidden');

    // Calibration: 只顯示 Original、隱藏 Corrected
    // Chat: 只顯示 Corrected、隱藏 Original
    if (mode === 'calibration') {
        els.panelOriginal.classList.remove('hidden');
        els.panelCorrected.classList.add('hidden');
        els.canvasSelection.classList.add('hidden');
    } else if (mode === 'chat') {
        els.panelOriginal.classList.add('hidden');
        els.panelCorrected.classList.remove('hidden');
        els.canvasSelection.classList.remove('hidden');
        resizeSelectionCanvas();
    }

    // 處理迴圈：所有模式都持續校正
    stopProcessingLoop();
    if (state.cameraRunning) {
        startProcessingLoop();
    }
}

function resizeSelectionCanvas() {
    const canvas = els.canvasSelection;
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
}

// ============================================================
// 事件綁定
// ============================================================
function bindEvents() {
    // 模式切換
    $$('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => switchMode(btn.dataset.mode));
    });

    // Crop 按鈕：單獨裁切紙張區域
    els.btnCrop.addEventListener('click', async () => {
        const imageData = captureFrame();
        if (!imageData) return;
        try {
            const resp = await fetch('/api/crop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData }),
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            if (data.cropped) {
                els.imgCorrected.src = `data:image/jpeg;base64,${data.cropped}`;
                els.imgCorrected.classList.remove('hidden');
                els.correctedPlaceholder.classList.add('hidden');
            }
        } catch (err) {
            console.error('Crop 失敗:', err);
        }
    });

    // 重置
    els.btnReset.addEventListener('click', async () => {
        await fetch('/api/reset', { method: 'POST' });
        state.isCalibrated = false;
        updateStatus('idle', '已重置，搜尋紙張中...');
        els.imgCorrected.classList.add('hidden');
        els.correctedPlaceholder.classList.remove('hidden');
    });

    // 聊天
    els.btnChatSend.addEventListener('click', sendChatQuestion);
    els.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') sendChatQuestion();
    });

    // 選取框
    setupSelectionCanvas();

    // 視窗大小變更
    window.addEventListener('resize', () => {
        if (state.currentMode === 'chat') resizeSelectionCanvas();
    });
}

// ============================================================
// 初始化
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    bindEvents();
    // 自動啟動攝影機
    startCamera();
});
