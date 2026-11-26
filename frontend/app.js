// 模式说明弹窗
function showModeInfo() {
    const modal = document.createElement('div');
    modal.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); display: flex; align-items: center; justify-content: center; z-index: 10000;';
    modal.innerHTML = `
        <div style="background: var(--card-bg); border-radius: 16px; padding: 30px; max-width: 600px; max-height: 80vh; overflow-y: auto; position: relative;">
            <button onclick="this.closest('div').parentElement.remove()" style="position: absolute; top: 15px; right: 15px; background: transparent; border: none; color: var(--text-dim); font-size: 1.5rem; cursor: pointer;">&times;</button>
            <h2 style="color: var(--primary); margin-bottom: 20px;"><i class="fas fa-info-circle"></i> 关于演示模式</h2>
            
            <div style="margin-bottom: 20px;">
                <h3 style="color: var(--accent); font-size: 1.1rem; margin-bottom: 10px;">📊 当前实现功能</h3>
                <ul style="color: var(--text-dim); line-height: 1.8; padding-left: 20px;">
                    <li><strong>深度视觉感知</strong>: YOLO目标检测 + MediaPipe姿态估计</li>
                    <li><strong>多模态特征提取</strong>: 球衣颜色识别 + 运动轨迹分析</li>
                    <li><strong>决策画像生成</strong>: 五维能力雷达图 + AI评估建议</li>
                    <li><strong>演示数据策略</strong>: 基于真实检测框架 + 模板化能力分布</li>
                </ul>
            </div>
            
            <div style="margin-bottom: 20px;">
                <h3 style="color: #ffce56; font-size: 1.1rem; margin-bottom: 10px;">🎯 数据生成说明</h3>
                <ul style="color: var(--text-dim); line-height: 1.8; padding-left: 20px;">
                    <li><strong>球员识别</strong>: 真实的YOLO检测 + 颜色分类（排除裁判）</li>
                    <li><strong>编号标注</strong>: 随机分配2-30号，模拟背号识别效果</li>
                    <li><strong>能力评估</strong>: 预设4种差异化模板（防守型/进攻型/组织型/全能型）</li>
                    <li><strong>建议生成</strong>: 基于能力值动态匹配15+专业模板</li>
                </ul>
            </div>
            
            <div style="background: rgba(255,99,132,0.1); border-left: 3px solid #ff6384; padding: 15px; border-radius: 8px;">
                <p style="color: var(--text-dim); line-height: 1.6; margin: 0;">
                    <strong style="color: #ff6384;">⚠️ 学术说明</strong><br>
                    本演示旨在展示技术框架与功能流程。真实应用中，能力评估需结合历史数据、多场比赛跟踪及专业标注，当前版本侧重可视化与交互体验。
                </p>
            </div>
            
            <button onclick="this.closest('div').parentElement.remove()" style="width: 100%; margin-top: 20px; padding: 12px; background: var(--primary); color: var(--bg); border: none; border-radius: 8px; cursor: pointer; font-size: 1rem; font-weight: bold;">
                我知道了
            </button>
        </div>
    `;
    document.body.appendChild(modal);
}

console.log('[APP] ===== app.js 模块加载 (v2.2) =====');

document.addEventListener('DOMContentLoaded', () => {
    console.log('[APP] ===== DOMContentLoaded 触发 =====');
    console.log('[APP] 当前时间:', new Date().toISOString());
    
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const videoPreview = document.getElementById('video-preview');
    const uploadBtn = document.getElementById('upload-btn');
    const fileInfo = document.getElementById('file-info');
    const filenameDisplay = document.getElementById('filename-display');
    const statusBadge = document.getElementById('status');
    const progressBar = document.getElementById('progress-bar');
    const progressContainer = document.getElementById('progress-container');
    const analysisPlaceholder = document.getElementById('analysis-placeholder');
    const analysisContent = document.getElementById('analysis-content');
    const videoContainer = document.getElementById('video-container');

    const API_BASE = window.location.origin;

    let uploadedVideoId = null;
    
    // 暴露到全局，供 highlight.js 使用
    window.uploadedVideoId = null;
    let isAnalyzing = false;
    let heartbeatTimer = null;   // 全局心跳定时器

    // Phase A 全局追踪状态
    let phaseAProgress = -1;   // -1=未开始, 0~99=进行中, 100=完成
    let phaseATimer = null;    // 进度轮询定时器
    let phaseADone = false;    // 是否已完成（cached 或 progress=100）
    let phaseAWaitingOpen = false; // 用户已点击"标注球员"，等待 Phase A 完成后自动开弹窗

    // 启动 Phase A 全局追踪
    async function startPhaseA(videoId) {
        phaseAProgress = 0;
        phaseADone = false;
        try {
            const res = await fetch(`${API_BASE}/pre_track/${videoId}`, { method: 'POST' });
            const d = await res.json();
            if (d.status === 'cached') {
                // 已有缓存，直接标记完成并自动打开弹窗
                phaseAProgress = 100;
                phaseADone = true;
                updatePhaseAButton();
                if (!isAnalyzing && window.showJerseyModal) {
                    window.showJerseyModal(videoId);
                }
                return;
            }
        } catch (e) {
            console.warn('[PhaseA] /pre_track 请求失败:', e);
        }
        // 开始轮询进度
        if (phaseATimer) clearInterval(phaseATimer);
        phaseATimer = setInterval(() => pollPhaseA(videoId), 2000);
    }

    async function pollPhaseA(videoId) {
        try {
            const res = await fetch(`${API_BASE}/pre_track_progress/${videoId}`);
            const d = await res.json();
            phaseAProgress = typeof d.progress === 'number' ? d.progress : phaseAProgress;
            const done = d.cached || phaseAProgress >= 100;
            if (done) {
                phaseAProgress = 100;
                phaseADone = true;
                if (phaseATimer) { clearInterval(phaseATimer); phaseATimer = null; }
                updatePhaseAButton();
                // Phase A 完成后自动打开标注弹窗（未在分析中时）
                if (!isAnalyzing && window.showJerseyModal) {
                    phaseAWaitingOpen = false;
                    window.showJerseyModal(videoId);
                } else if (phaseAWaitingOpen) {
                    phaseAWaitingOpen = false;
                    window.showJerseyModal(videoId);
                }
            } else {
                updatePhaseAButton();
            }
        } catch (e) {
            // 忽略轮询错误，继续重试
        }
    }

    function updatePhaseAButton() {
        if (isAnalyzing) return;  // 主分析中不改按钮
        if (phaseADone) {
            uploadBtn.textContent = '标注球员';
            uploadBtn.disabled = false;
            uploadBtn.style.opacity = '1';
            statusBadge.textContent = '全局追踪完成，可开始标注';
        } else if (phaseAProgress >= 0) {
            const pct = phaseAProgress > 0 ? ` ${phaseAProgress}%` : '';
            uploadBtn.textContent = `追踪中${pct}，点击标注球员`;
            uploadBtn.disabled = false;   // 允许点击，进入等待界面
            uploadBtn.style.opacity = '0.8';
            statusBadge.textContent = `全局追踪进行中${pct}，可提前点击"标注球员"等待`;
        }
    }

    // 点击上传区域或文件名显示区域都可以触发重新选择
    dropZone.addEventListener('click', () => fileInput.click());
    
    // 允许在上传后通过点击 filenameDisplay 重新上传
    filenameDisplay.style.cursor = 'pointer';
    filenameDisplay.addEventListener('click', () => {
        if (!isAnalyzing) fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) handleFile(e.target.files[0]);
    });

    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = 'var(--primary)'; });
    dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = 'rgba(255, 255, 255, 0.2)'; });
    dropZone.addEventListener('drop', (e) => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); });

    function handleFile(file) {
        if (file && file.type.startsWith('video/')) {
            // 重置状态
            uploadedVideoId = null;
            window.uploadedVideoId = null;  // 同步到全局
            isAnalyzing = false;
            analysisContent.style.display = 'none';
            analysisPlaceholder.style.display = 'block';
            progressContainer.style.display = 'none';
            // 重置 Phase A 状态
            phaseAProgress = -1;
            phaseADone = false;
            phaseAWaitingOpen = false;
            if (phaseATimer) { clearInterval(phaseATimer); phaseATimer = null; }
            const oldOverlay = document.getElementById('phase-a-wait-overlay');
            if (oldOverlay) oldOverlay.remove();
            
            // 清理旧的视频 URL
            if (videoPreview.src) {
                URL.revokeObjectURL(videoPreview.src);
            }
            
            filenameDisplay.textContent = `准备上传: ${file.name}`;
            fileInfo.style.display = 'block';
            
            uploadBtn.disabled = true;
            uploadBtn.textContent = '正在极速上传...';
            uploadBtn.style.opacity = '0.5';
            
            autoUpload(file);
        }
    }

    async function autoUpload(file) {
        const formData = new FormData();
        formData.append('file', file);

        // 获取当前登录用户名，作为视频所有者
        const authRaw = localStorage.getItem('auth_user');
        const authUser = authRaw ? JSON.parse(authRaw) : null;
        const ownerParam = authUser ? encodeURIComponent(authUser.username) : 'guest';

        // 精彩视频在主分析完成后由 highlight.js 触发，上传时不再并行启动
        let uploadUrl = `${API_BASE}/upload?owner=${ownerParam}`;

        const xhr = new XMLHttpRequest();
        xhr.open('POST', uploadUrl, true);
        
        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const percent = (e.loaded / e.total * 100).toFixed(0);
                statusBadge.textContent = `上传中: ${percent}%`;
                uploadBtn.textContent = `上传中 (${percent}%)`;
            }
        };

        xhr.onload = function() {
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                uploadedVideoId = data.video_id;
                window.uploadedVideoId = data.video_id;  // 同步到全局
                
                filenameDisplay.innerHTML = `已就绪: ${file.name} <span style="color:var(--primary); font-size:0.8rem;">(点击可更换)</span>`;
                
                // 若勾选了精彩视频，派发上传完成事件让 highlight.js 立即开始轮询
                document.dispatchEvent(new CustomEvent('videoUploadCompleted', {
                    detail: { videoId: uploadedVideoId }
                }));
                
                // 创建一个本地 URL 并显示预览
                const videoURL = URL.createObjectURL(file);
                videoPreview.src = videoURL;
                videoPreview.load(); // 强制加载视频
                videoContainer.style.display = 'block';
                dropZone.style.display = 'none';
                
                // 确保视频可以播放
                videoPreview.addEventListener('loadeddata', () => {
                    console.log('视频预览加载成功');
                }, { once: true });
                
                // 上传完成后自动启动全局追踪 Phase A
                uploadBtn.disabled = false;
                uploadBtn.style.opacity = '0.8';
                uploadBtn.textContent = '追踪中，点击标注球员';
                statusBadge.textContent = '上传完成，正在自动启动全局追踪...';
                startPhaseA(uploadedVideoId);
            } else {
                statusBadge.textContent = '上传失败，请重试';
                uploadBtn.textContent = '重试上传';
                uploadBtn.disabled = false;
            }
        };
        xhr.send(formData);
    }

    uploadBtn.addEventListener('click', async () => {
        if (!uploadedVideoId || isAnalyzing) return;

        // Phase A 已完成：直接开标注弹窗
        if (phaseADone) {
            if (window.showJerseyModal) {
                window.showJerseyModal(uploadedVideoId);
            } else {
                window._triggerAnalysis(uploadedVideoId, [], false, [], {}, []);
            }
            return;
        }

        // Phase A 进行中：标记等待，显示实时进度浮层
        phaseAWaitingOpen = true;
        showPhaseAWaitOverlay();
    });

    // 显示 Phase A 等待进度浮层
    function showPhaseAWaitOverlay() {
        let overlay = document.getElementById('phase-a-wait-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'phase-a-wait-overlay';
            overlay.style.cssText = [
                'position:fixed;inset:0;z-index:9999;',
                'background:rgba(4,11,24,0.85);backdrop-filter:blur(6px);',
                'display:flex;align-items:center;justify-content:center;'
            ].join('');
            overlay.innerHTML = `
                <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(0,255,136,0.25);
                            border-radius:16px;padding:36px 40px;text-align:center;min-width:320px;max-width:440px;">
                    <div style="font-size:2.2rem;margin-bottom:12px;">
                        <i class="fas fa-satellite-dish" style="color:#00ff88;animation:fa-pulse 1.5s infinite;"></i>
                    </div>
                    <h3 style="color:#fff;margin:0 0 8px;font-size:1.1rem;">全局追踪进行中</h3>
                    <p style="color:rgba(255,255,255,0.5);font-size:0.85rem;margin:0 0 20px;">
                        系统正在对视频进行全局 YOLO 检测与球员追踪，完成后将自动进入标注界面
                    </p>
                    <div style="height:8px;background:rgba(255,255,255,0.1);border-radius:4px;overflow:hidden;margin-bottom:10px;">
                        <div id="phase-a-overlay-bar"
                             style="height:100%;width:0%;background:linear-gradient(90deg,#00ff88,#00d4ff);
                                    border-radius:4px;transition:width 0.5s ease;"></div>
                    </div>
                    <div id="phase-a-overlay-pct" style="color:#00ff88;font-size:1.3rem;font-weight:700;margin-bottom:4px;">0%</div>
                    <div id="phase-a-overlay-hint" style="color:rgba(255,255,255,0.4);font-size:0.8rem;">初始化追踪引擎...</div>
                    <button id="phase-a-bg-wait-btn"
                            style="margin-top:20px;padding:8px 28px;border-radius:20px;border:1px solid rgba(255,255,255,0.2);
                                   background:transparent;color:rgba(255,255,255,0.5);cursor:pointer;font-size:0.85rem;">
                        后台等待
                    </button>
                </div>`;
            document.body.appendChild(overlay);
        }
        overlay.style.display = 'flex';
        // 绑定"后台等待"按钮：关闭浮层并取消自动弹窗
        const bgWaitBtn = overlay.querySelector('#phase-a-bg-wait-btn');
        if (bgWaitBtn) {
            bgWaitBtn.onclick = () => {
                overlay.remove();
                phaseAWaitingOpen = false;  // 取消等待标志，Phase A 完成后不再自动弹窗
            };
        }
        // 实时刷新浮层进度
        updatePhaseAOverlay();
    }

    function updatePhaseAOverlay() {
        const overlay = document.getElementById('phase-a-wait-overlay');
        if (!overlay || overlay.style.display === 'none') return;
        const bar = document.getElementById('phase-a-overlay-bar');
        const pct = document.getElementById('phase-a-overlay-pct');
        const hint = document.getElementById('phase-a-overlay-hint');
        const prog = Math.max(0, phaseAProgress);
        if (bar) bar.style.width = prog + '%';
        if (pct) pct.textContent = prog + '%';
        if (hint) hint.textContent = prog < 30 ? '正在初始化 YOLO 检测引擎...'
            : prog < 60 ? '逐帧检测球员目标...'
            : prog < 90 ? '构建跨帧追踪链路...'
            : '生成追踪缓存，即将完成...';
        if (phaseADone) {
            if (bar) bar.style.width = '100%';
            if (pct) pct.textContent = '100%';
            if (hint) hint.textContent = '追踪完成！正在打开标注界面...';
            setTimeout(() => { overlay.remove(); }, 800);
        } else {
            setTimeout(updatePhaseAOverlay, 500);
        }
    }

    // ─── 全局分析触发函数（供弹窗确认/跳过调用） ───
    window._triggerAnalysis = async function (videoId, jerseyAnnotations, teamSwap, nonPlayerList, trackJerseyMap, excludedTrackIds, nonPlayerTrackIds) {
        if (!videoId || isAnalyzing) return;
        uploadedVideoId = videoId;
        window.uploadedVideoId = videoId;
        isAnalyzing = true;
        uploadBtn.disabled = true;
        uploadBtn.style.opacity = '0.6';
        uploadBtn.textContent = '正在深度分析中...';
        statusBadge.textContent = 'AI 引擎启动中，请稍候...';
        statusBadge.style.color = 'var(--accent)';

        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';

        // 显示右侧面板内联进度条
        const inlineWrap = document.getElementById('inline-progress-wrap');
        const inlinePct  = document.getElementById('inline-progress-pct');
        const inlineBar  = document.getElementById('inline-progress-bar');
        const inlineHint = document.getElementById('inline-progress-hint');
        const placeholderText = document.getElementById('placeholder-main-text');
        if (inlineWrap) inlineWrap.style.display = 'block';
        if (placeholderText) placeholderText.textContent = '正在深度分析视频...';

        // 若未勾选"生成精彩视频"，隐藏选项区域
        const highlightCheckbox = document.getElementById('generate-highlight-checkbox');
        if (highlightCheckbox && !highlightCheckbox.checked) {
            const wrapEl = highlightCheckbox.closest('div[style*="border-radius: 8px"]');
            if (wrapEl) wrapEl.style.display = 'none';
        }

        // ── 统一走 v2 接口（Phase B 流程），trackJerseyMap 为空时传 {} ──
        const safeTrackMap = (trackJerseyMap && typeof trackJerseyMap === 'object') ? trackJerseyMap : {};
        const safeExcluded = Array.isArray(excludedTrackIds) ? excludedTrackIds : [];
        const safeNonPlayer = Array.isArray(nonPlayerTrackIds) ? nonPlayerTrackIds : [];
        try {
            const v2Res = await fetch(`${API_BASE}/start_analysis_v2/${videoId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    track_jersey_map: safeTrackMap,
                    excluded_track_ids: safeExcluded,
                    non_player_track_ids: safeNonPlayer,
                    team_swap: !!teamSwap,
                })
            });
            // Phase A 缓存不存在时后端返回 phase_a_not_ready，给用户明确提示
            if (v2Res.ok) {
                const v2Data = await v2Res.json();
                if (v2Data.status === 'phase_a_not_ready') {
                    statusBadge.textContent = '全局追踪缓存未就绪，请等待追踪完成后重试';
                    statusBadge.style.color = '#ff6384';
                    isAnalyzing = false;
                    uploadBtn.disabled = false;
                    uploadBtn.style.opacity = '1';
                    // 重新启动 Phase A 并更新按钮状态
                    phaseADone = false;
                    startPhaseA(videoId);
                    progressContainer.style.display = 'none';
                    return;
                }
            }
        } catch (_) {}

        retryCount = 0;
        lastProgress = -1;
        staleProgressCount = 0;
        pollToken++;
        showResults(videoId, pollToken);
    };

    let retryCount = 0;
    let lastProgress = -1;       // 上一次轮询的进度值
    let staleProgressCount = 0;  // 进度无变化的轮询次数
    let pollToken = 0;           // 每次新分析递增，用于取消旧轮询
    const maxRetries = 3600; // 最多120分钟 (3600次 × 2秒，适应长视频分析)

    function getProgressHint(prog) {
        if (prog <= 1)  return '正在启动分析引擎...';
        if (prog <= 3)  return '正在加载 Phase A 追踪缓存...';
        if (prog <= 4)  return '正在初始化球队颜色模型...';
        if (prog <= 5)  return '正在构建球员标注映射...';
        if (prog <= 6)  return '正在初始化视频编码器...';
        if (prog <= 7)  return '正在加载辅助模型...';
        if (prog < 10)  return '准备开始逐帧渲染...';
        if (prog < 40) return '阶段一：背号识别与球员锁定...';
        if (prog < 45) return '阶段一完成，准备深度分析...';
        if (prog < 86) return '阶段二：逐帧目标追踪与战术分析...';
        if (prog < 88) return '阶段三：生成热力图与战术板...';
        if (prog < 95) return '阶段三：视频转码处理中...';
        if (prog < 98) return '阶段三：计算球员能力评分...';
        if (prog < 100) return '阶段三：写入分析报告...';
        return '分析完成！';
    }
    
    async function showResults(videoId, token) {
        // token 不匹配说明已启动新任务或用户切换，放弃此轮询
        if (token !== undefined && token !== pollToken) return;
        try {
            const response = await fetch(`${API_BASE}/results/${videoId}`);
            const data = await response.json();
            
            console.log(`[POLL] 轮询结果 (第${retryCount + 1}次):`, {
                progress: data.progress,
                hasDetailedAnalysis: !!data.detailed_analysis,
                hasAthletes: !!(data.detailed_analysis && data.detailed_analysis.athletes),
                athletesCount: data.detailed_analysis?.athletes?.length || 0
            });

            // 首先检查进度
            if (data.progress !== undefined) {
                const prog = parseInt(data.progress);
                if (prog >= 0) {
                    progressBar.style.width = `${prog}%`;
                    // 同步更新右侧内联进度条
                    const _inlineBar = document.getElementById('inline-progress-bar');
                    const _inlinePct = document.getElementById('inline-progress-pct');
                    const _inlineHint = document.getElementById('inline-progress-hint');
                    if (_inlineBar) _inlineBar.style.width = `${prog}%`;
                    if (_inlinePct) _inlinePct.textContent = `${prog}%`;
                    if (prog < 100) {
                        // 检测进度是否停滞（CPU模式下YOLO推理会较慢）
                        if (prog === lastProgress) {
                            staleProgressCount++;
                            // 停滞超过30次（约1分钟），提示正在处理中
                            if (staleProgressCount > 30 && staleProgressCount % 15 === 0) {
                                const dots = '.'.repeat((staleProgressCount / 15) % 4);
                                const msg = `分析进度: ${prog}%，处理中，请耐心等待${dots}`;
                                statusBadge.textContent = msg;
                                if (_inlineHint) _inlineHint.textContent = `AI 推理中，请耐心等待${dots}`;
                            } else if (staleProgressCount <= 30) {
                                statusBadge.textContent = `分析进度: ${prog}%`;
                                if (_inlineHint) _inlineHint.textContent = getProgressHint(prog);
                            }
                        } else {
                            staleProgressCount = 0;
                            lastProgress = prog;
                            statusBadge.textContent = `分析进度: ${prog}%`;
                            if (_inlineHint) _inlineHint.textContent = getProgressHint(prog);
                        }
                    } else if (prog === 100) {
                        // 进度达到100%，直接跳过后续轮询，渲染结果
                        console.log('[INFO] 分析完成，进度100%，开始渲染结果...');
                        if (_inlineHint) _inlineHint.textContent = '分析完成，正在生成报告...';
                    }
                }
            }

            // 检查是否有错误信息
            if (data.detailed_analysis && data.detailed_analysis.error) {
                statusBadge.textContent = '分析失败：' + (data.detailed_analysis.message || '未知错误');
                statusBadge.style.color = '#ff6384';
                progressContainer.style.display = 'none';
                isAnalyzing = false;
                if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
                uploadBtn.disabled = false;
                uploadBtn.textContent = '重试分析';
                
                // 显示详细错误信息
                analysisContent.innerHTML = `
                    <div style="padding: 30px; text-align: center; color: var(--text-dim);">
                        <i class="fas fa-exclamation-triangle" style="font-size: 3rem; color: #ff6384; margin-bottom: 20px;"></i>
                        <h3 style="color: #ff6384; margin-bottom: 15px;">视频分析失败</h3>
                        <p style="line-height: 1.6; margin-bottom: 20px;">${data.decision_summary || '未知错误'}</p>
                        <div style="background: rgba(255,99,132,0.1); border-left: 3px solid #ff6384; padding: 15px; border-radius: 8px; text-align: left; max-width: 600px; margin: 0 auto;">
                            <strong style="color: #ff6384;">可能的原因：</strong>
                            <ul style="margin-top: 10px; padding-left: 20px;">
                                <li>视频格式不支持（建议使用 MP4 格式）</li>
                                <li>视频文件损坏或无法读取</li>
                                <li>OCR/YOLO 依赖未正确安装</li>
                                <li>视频中未检测到足够的球员</li>
                            </ul>
                        </div>
                    </div>
                `;
                analysisPlaceholder.style.display = 'none';
                analysisContent.style.display = 'block';
                return;
            }

            // 🔑 修复3: 更严格的完成判断
            const isComplete = (
                data.progress === 100 && 
                data.detailed_analysis && 
                data.detailed_analysis.athletes && 
                Array.isArray(data.detailed_analysis.athletes) &&
                data.detailed_analysis.athletes.length > 0
            );

            if (!isComplete) {
                // 特殊处理：progress=100但数据未准备好（竞态条件）
                if (data.progress === 100 && !data.detailed_analysis) {
                    console.warn('[WARN] 进度100%但数据未就绪，延迟1秒重试...');
                    retryCount++;  // 增加重试计数
                    setTimeout(() => showResults(videoId, token), 1000);
                    return;
                }
                
                // 分析未完成，继续轮询
                if (data.progress === -1) {
                    statusBadge.textContent = '分析失败：后端处理异常';
                    statusBadge.style.color = '#ff6384';
                    isAnalyzing = false;
                    if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
                    uploadBtn.disabled = false;
                    uploadBtn.textContent = '重试分析';
                    return;
                }
                
                // 检查重试次数
                retryCount++;
                if (retryCount >= maxRetries) {
                    statusBadge.textContent = '分析超时（超过120分钟未响应）';
                    statusBadge.style.color = '#ff6384';
                    progressContainer.style.display = 'none';
                    isAnalyzing = false;
                    if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
                    uploadBtn.disabled = false;
                    uploadBtn.textContent = '重试分析';
                    return;
                }
                
                // 🔑 优化：根据进度调整轮询频率
                // 进度 < 50%: 每2秒
                // 进度 50-90%: 每1.5秒
                // 进度 = 100%: 每500ms（快速响应竞态条件）
                // 进度 > 90%: 每1秒
                const pollInterval = data.progress < 50 ? 2000 : 
                                   data.progress < 90 ? 1500 : 
                                   data.progress === 100 ? 500 :  // 100%时更快
                                   1000;
                
                setTimeout(() => showResults(videoId, token), pollInterval);
                return;
            }
            
            // ✅ 分析完成，渲染结果
            console.log('[SUCCESS] 分析完成，开始渲染结果页面');
            retryCount = 0;

            // 完成：停止心跳轮询
            if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
            isAnalyzing = false;
            statusBadge.textContent = '分析任务圆满完成';
            statusBadge.style.color = 'var(--primary)';
            progressContainer.style.display = 'none';
            uploadBtn.textContent = '分析完成';
            uploadBtn.disabled = true;
            
            // 🎬 触发精彩视频生成事件
            console.log('[APP] ===== 准备触发 videoAnalysisCompleted 事件 =====');
            console.log('[APP] uploadedVideoId:', uploadedVideoId);
            console.log('[APP] window.uploadedVideoId:', window.uploadedVideoId);
            
            const event = new CustomEvent('videoAnalysisCompleted', {
                detail: {
                    videoId: uploadedVideoId,
                    analysisData: data
                }
            });
            
            console.log('[APP] 触发事件:', event);
            document.dispatchEvent(event);
            console.log('[APP] ===== 事件已触发 =====');
            
            analysisPlaceholder.style.display = 'none';
            analysisContent.style.display = 'block';
            
            // 显示分析概览
            const summaryCard = document.getElementById('analysis-summary');
            if (summaryCard) {
                summaryCard.style.display = 'block';
                const avgConfidence = (data.detailed_analysis.athletes.reduce((sum, p) => sum + (p.path_accuracy || 0.95), 0) / data.detailed_analysis.athletes.length * 100).toFixed(0);
                document.getElementById('summary-confidence').textContent = avgConfidence + '%';
            }

            // ===== 控球率 / 热力图 / 战术板 =====
            const advStats = document.getElementById('advanced-stats');
            if (advStats) advStats.style.display = 'block';

            // 控球率
            const poss = data.detailed_analysis.possession;
            if (poss) {
                const pa = poss.team_a || 50;
                const pb = poss.team_b || 50;
                const barA = document.getElementById('poss-bar-a');
                const lblA = document.getElementById('poss-a-label');
                const lblB = document.getElementById('poss-b-label');
                if (barA) barA.style.width = pa + '%';
                if (lblA) lblA.textContent = `主队 ${pa}%`;
                if (lblB) lblB.textContent = `客队 ${pb}%`;
            }

            // 热力图
            const hmUrls = data.detailed_analysis.heatmap_urls || {};
            ['team_a', 'team_b'].forEach(team => {
                const suffix = team === 'team_a' ? 'a' : 'b';
                const img = document.getElementById(`heatmap-team-${suffix}`);
                const placeholder = document.getElementById(`heatmap-${suffix}-placeholder`);
                const url = hmUrls[team];
                if (img && url) {
                    img.style.display = 'none';
                    img.onload = () => {
                        img.style.display = 'block';
                        if (placeholder) placeholder.style.display = 'none';
                    };
                    img.onerror = () => {
                        img.style.display = 'none';
                        console.warn(`[HEATMAP] 图片加载失败: ${API_BASE}${url}`);
                        if (placeholder) {
                            placeholder.style.display = 'block';
                            placeholder.textContent = '热力图文件未找到（服务器路径异常），请重新分析';
                        }
                    };
                    img.src = `${API_BASE}${url}?_t=${Date.now()}`;
                } else if (placeholder && !url) {
                    // heatmap_urls 字段本身为空（分析时未生成热力图）
                    placeholder.style.display = 'block';
                    placeholder.textContent = '暂无数据（分析时未检测到足够球员轨迹）';
                }
            });

            // 战术板：优先加载动态轨迹数据，后备为静态图
            const tactUrl = data.detailed_analysis.tactical_url;
            // 用侧边栏容器 scope 查找，避免与全屏弹窗重复 ID 冲突
            const _sidebarTact = document.getElementById('sidebar-tactical-section');
            const _q = (id) => _sidebarTact ? _sidebarTact.querySelector('#' + id) : document.getElementById(id);
            const tactImg = _q('tactical-view');
            const tactPh  = _q('tactical-placeholder');

            // 生成战术分析简报（公共逻辑）
            function showTacticalDesc() {
                const tactDesc     = _q('tactical-desc');
                const tactDescText = _q('tactical-desc-text');
                if (!tactDesc || !tactDescText) return;
                const poss = data.detailed_analysis.possession || { team_a: 50, team_b: 50 };
                const athletes = data.detailed_analysis.athletes || [];
                const ballCount = data.detailed_analysis.ball_trajectory_count || 0;
                const midIdx = Math.floor(athletes.length / 2);
                const teamA = athletes.slice(0, midIdx || 1);
                const teamB = athletes.slice(midIdx || 1);
                const avgAbil = (team) => team.length
                    ? Math.round(team.reduce((s, p) => s + Object.values(p.abilities || {}).reduce((a,b)=>a+b,0) / (Object.keys(p.abilities||{}).length||1), 0) / team.length)
                    : 0;
                const aScore = avgAbil(teamA);
                const bScore = avgAbil(teamB);
                const domTeam = poss.team_a >= poss.team_b ? '主队' : '客队';
                const domPoss = Math.max(poss.team_a, poss.team_b);
                let styleA = poss.team_a >= 60 ? '以控球为核心，倾向阵地进攻' : (poss.team_a <= 40 ? '以防守反击为主，快速转换' : '控球与反击兼备，战术均衡');
                let styleB = poss.team_b >= 60 ? '以控球为核心，倾向阵地进攻' : (poss.team_b <= 40 ? '以防守反击为主，快速转换' : '控球与反击兼备，战术均衡');
                const allSkills = {};
                athletes.forEach(p => Object.entries(p.abilities||{}).forEach(([k,v])=>{ allSkills[k]=(allSkills[k]||0)+v; }));
                const topSkill = Object.entries(allSkills).sort((a,b)=>b[1]-a[1])[0]?.[0] || '';
                const formationHint = { '防守': '偏向防守阵型（如4-5-1或5-4-1）', '射门': '进攻型阵型（如4-3-3或3-4-3）', '传球': '控制型阵型（如4-3-3 Tiki-Taka）', '速度': '快节奏边路打法', '体能': '高强度压迫型战术' }[topSkill] || '均衡型阵型';
                tactDescText.textContent =
                    `本场比赛双方跑动轨迹覆盖全场，共追踪 ${ballCount} 个球轨迹采样点。` +
                    `控球率方面，${domTeam}以 ${domPoss}% 占据主动，` +
                    `主队${styleA}，客队${styleB}。` +
                    `综合能力评估：主队均分 ${aScore}/100，客队均分 ${bScore}/100，` +
                    `${aScore >= bScore ? '主队' : '客队'}在整体实力上略占优势。` +
                    `从轨迹分布看，全队整体趋向${formationHint}，` +
                    `两侧边路和中路均有明显跑动覆盖，攻守转换较为积极。`;
                tactDesc.style.display = 'block';
            }

            // 尝试加载动态战术板
            fetch(`${API_BASE}/tactical_data/${uploadedVideoId}`)
                .then(r => r.ok ? r.json() : null)
                .then(td => {
                    // 复用外层已查找的侧边栏 scope，避免二次 getElementById
                    const canvasWrap = _sidebarTact
                        ? _sidebarTact.querySelector('#tactical-canvas-wrap')
                        : document.getElementById('tactical-canvas-wrap');
                    const hasTracks = td && td.tactical_tracks && Object.keys(td.tactical_tracks).length > 0;
                    // 更新战术板标题
                    const tactTitle = _sidebarTact
                        ? _sidebarTact.querySelector('h3')
                        : document.querySelector('#sidebar-tactical-section h3');
                    if (hasTracks) {
                        // 显示动态 Canvas，隐藏静态图和 placeholder
                        if (canvasWrap) canvasWrap.style.display = 'block';
                        if (tactImg)   tactImg.style.display   = 'none';
                        if (tactPh)    tactPh.style.display    = 'none';
                        if (tactTitle) tactTitle.innerHTML = '<i class="fas fa-chess-board"></i> 动态战术板';
                        // 传入 scope，防止 initDynamicTacticalBoard 内部取到全屏弹窗的同名控件
                        initDynamicTacticalBoard(td, _sidebarTact);
                    } else if (tactUrl) {
                        // 降级为静态图
                        if (tactImg) {
                            tactImg.src = `${API_BASE}${tactUrl}?_t=${Date.now()}`;
                            tactImg.style.display = 'block';
                        }
                        if (tactPh) tactPh.style.display = 'none';
                        if (canvasWrap) canvasWrap.style.display = 'none';
                        if (tactTitle) tactTitle.innerHTML = '<i class="fas fa-chess-board"></i> 战术板 2D 鸟瞰图';
                    }
                    showTacticalDesc();
                })
                .catch(() => {
                    // 网络错误降级为静态图
                    if (tactUrl && tactImg) {
                        tactImg.src = `${API_BASE}${tactUrl}?_t=${Date.now()}`;
                        tactImg.style.display = 'block';
                        if (tactPh) tactPh.style.display = 'none';
                    }
                    showTacticalDesc();
                });

            
            // ===== 球员选择器（下拉框，支持全场所有球员） =====
            const selectorSection = document.getElementById('player-selector-section');
            const selectorA = document.getElementById('selector-team-a');
            const selectorB = document.getElementById('selector-team-b');
            if (selectorSection && selectorA && selectorB) {
                selectorSection.style.display = 'block';

                // 重置下拉框
                selectorA.innerHTML = '<option value="" style="color:#000;background:#fff;">— 请选择主队球员 —</option>';
                selectorB.innerHTML = '<option value="" style="color:#000;background:#fff;">— 请选择客队球员 —</option>';

                // 取全量球员（支持所有球员，不限数量）
                const allAthletes = data.detailed_analysis.athletes || [];
                const athletesA = data.detailed_analysis.athletes_team_a
                    || allAthletes.filter(a => a.team === 'team_a' || a.player_id?.startsWith('A'));
                const athletesB = data.detailed_analysis.athletes_team_b
                    || allAthletes.filter(a => a.team === 'team_b' || a.player_id?.startsWith('B'));

                // 存储所有球员数据，供选择时查询
                window._allAthletes = allAthletes;

                // 按球衣号排序
                const sortByNum = arr => [...arr].sort((a, b) => {
                    const na = a.jersey_number || parseInt((a.player_id||'').replace(/\D/g,'')) || 99;
                    const nb = b.jersey_number || parseInt((b.player_id||'').replace(/\D/g,'')) || 99;
                    return na - nb;
                });

                function buildOption(player) {
                    const opt = document.createElement('option');
                    const num = player.jersey_number || parseInt((player.player_id || '').replace(/\D/g,'')) || '?';
                    const avgScore = Object.keys(player.abilities || {}).length
                        ? Math.round(Object.values(player.abilities).reduce((a,b)=>a+b,0) / Object.keys(player.abilities).length)
                        : 0;
                    opt.value = player.player_id || num;
                    opt.textContent = `#${num} ${player.name}  （综合 ${avgScore}/100）`;
                    opt.style.cssText = 'color:#000;background:#fff;';
                    opt.dataset.playerId = player.player_id || num;
                    return opt;
                }

                sortByNum(athletesA).forEach(p => selectorA.appendChild(buildOption(p)));
                sortByNum(athletesB).forEach(p => selectorB.appendChild(buildOption(p)));

                // 下拉框 change 事件：选中球员后展示详情
                selectorA.addEventListener('change', () => {
                    if (!selectorA.value) return;
                    selectorB.value = '';  // 清除另一个下拉框
                    const player = allAthletes.find(p =>
                        (p.player_id || p.jersey_number?.toString()) === selectorA.value ||
                        p.player_id === selectorA.value
                    ) || athletesA.find(p => String(p.player_id || p.jersey_number) === selectorA.value);
                    if (player) showPlayerDetail(player, '255,130,40');
                });
                selectorB.addEventListener('change', () => {
                    if (!selectorB.value) return;
                    selectorA.value = '';  // 清除另一个下拉框
                    const player = allAthletes.find(p =>
                        (p.player_id || p.jersey_number?.toString()) === selectorB.value ||
                        p.player_id === selectorB.value
                    ) || athletesB.find(p => String(p.player_id || p.jersey_number) === selectorB.value);
                    if (player) showPlayerDetail(player, '80,120,255');
                });
            }

            // 关闭详情卡片
            const detailClose = document.getElementById('player-detail-close');
            if (detailClose) {
                detailClose.onclick = () => {
                    document.getElementById('player-detail-card').style.display = 'none';
                    const sa = document.getElementById('selector-team-a');
                    const sb = document.getElementById('selector-team-b');
                    if (sa) sa.value = '';
                    if (sb) sb.value = '';
                };
            }

            const container = document.getElementById('player-analysis-container');
            container.innerHTML = '';

            drawTracking(data.detailed_analysis);
        } catch (error) {
            console.warn('[POLL] 轮询请求失败:', error.message);
            retryCount++;
            if (retryCount < maxRetries) {
                statusBadge.textContent = `连接后端中... (${retryCount})`;
                setTimeout(() => showResults(videoId, token), 5000);
            } else {
                statusBadge.textContent = '连接超时，请检查后端服务';
                statusBadge.style.color = '#ff6384';
                isAnalyzing = false;
                uploadBtn.disabled = false;
                uploadBtn.textContent = '重试分析';
            }
        }
    }

    function showPlayerDetail(player, teamColor, scope) {
        const $ = (id) => scope ? scope.querySelector('#' + id) : document.getElementById(id);
        const card = $('player-detail-card');
        if (!card) return;

        const num = player.jersey_number || parseInt((player.player_id || '').replace(/\D/g,'')) || '?';
        const abilities = player.abilities || {};
        const avgScore = Object.keys(abilities).length
            ? Math.round(Object.values(abilities).reduce((a,b) => a+b, 0) / Object.keys(abilities).length)
            : 0;
        const confidence = ((player.path_accuracy || 0.95) * 100).toFixed(0);

        $('detail-player-name').innerHTML = `<i class="fas fa-user-check"></i> ${player.name}`;
        $('detail-confidence-badge').innerHTML = `<i class="fas fa-crosshairs"></i> 置信度 ${confidence}%`;
        $('detail-score-badge').textContent = `综合评分 ${avgScore}/100`;
        $('detail-suggestion').textContent = player.suggestion || '暂无评估数据';

        // 能力详情列表
        const abList = $('detail-abilities-list');
        abList.innerHTML = Object.entries(abilities).map(([key, val]) => `
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;font-size:0.85rem;">
                <span style="color:rgba(255,255,255,0.7);">${key}</span>
                <span style="color:var(--primary);font-weight:bold;">${val}</span>
            </div>
        `).join('');

        // 颜色主题（由队伍颜色决定）
        const isTeamA = player.team === 'team_a' || (player.player_id || '').startsWith('A');
        const colorTheme = isTeamA
            ? { bg: 'rgba(255,130,40,0.2)', border: '#ff8228' }
            : { bg: 'rgba(80,120,255,0.2)', border: '#5078ff' };

        card.style.display = 'block';

        // 判断是否在报告弹窗（position:fixed 浮窗模式）或主页（普通流布局）
        // 报告弹窗中 card 为 position:fixed，自动定位到屏幕右侧靠近战术板
        if (getComputedStyle(card).position === 'fixed') {
            // 尝试找到战术板容器，贴靠其右侧；若右侧空间不足则贴靠左侧
            const tacticalWrap = document.querySelector('#rpt-analysis-content #tactical-canvas-wrap, #rpt-analysis-content [id="tactical-canvas-wrap"]')
                              || document.getElementById('tactical-canvas-wrap');
            if (tacticalWrap) {
                const rect = tacticalWrap.getBoundingClientRect();
                const cardW = 420;
                const margin = 16;
                const rightSpace = window.innerWidth - rect.right - margin;
                const leftSpace  = rect.left - margin;
                if (rightSpace >= cardW) {
                    // 放战术板右侧
                    card.style.left  = (rect.right + margin) + 'px';
                    card.style.right = 'auto';
                } else if (leftSpace >= cardW) {
                    // 放战术板左侧
                    card.style.right = 'auto';
                    card.style.left  = (rect.left - cardW - margin) + 'px';
                } else {
                    // 空间不足，居右兜底
                    card.style.left  = 'auto';
                    card.style.right = '24px';
                }
                // 垂直对齐战术板中心
                const tactMid = rect.top + rect.height / 2;
                const cardH   = Math.min(card.offsetHeight || 500, window.innerHeight * 0.88);
                let topPx = tactMid - cardH / 2;
                topPx = Math.max(16, Math.min(topPx, window.innerHeight - cardH - 16));
                card.style.top       = topPx + 'px';
                card.style.transform = 'none';
            }
        } else {
            // 主页普通流布局：滚动到详情卡片
            card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        // 渲染雷达图 —— 用 setTimeout 确保浏览器完成重排（display:none→block）后再创建 Chart
        const canvas = $('detail-radar-canvas');
        if (!canvas) return;

        const _doRenderRadar = () => {
            // 销毁已有实例（避免重复创建报错）
            const existingChart = Chart.getChart(canvas);
            if (existingChart) existingChart.destroy();

            // 若 canvas 自身仍无宽度（极端情况：父容器还未布局），延迟重试一次
            if (canvas.offsetWidth === 0) {
                setTimeout(_doRenderRadar, 60);
                return;
            }

            const ctx = canvas.getContext('2d');
            const orderedLabels = ["防守", "射门", "传球", "速度", "体能"];
            const orderedValues = orderedLabels.map(label => abilities[label] || 0);

            new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: orderedLabels,
                    datasets: [{
                        data: orderedValues,
                        backgroundColor: colorTheme.bg,
                        borderColor: colorTheme.border,
                        borderWidth: 2,
                        pointBackgroundColor: colorTheme.border,
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: colorTheme.border
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            min: 0, max: 100,
                            ticks: { display: true, stepSize: 20, color: 'rgba(255,255,255,0.3)', backdropColor: 'transparent' },
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            angleLines: { color: 'rgba(255,255,255,0.1)' },
                            pointLabels: { color: 'rgba(255,255,255,0.8)', font: { size: 12 } }
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: { label: ctx => ctx.label + ': ' + ctx.parsed.r + '/100' }
                        }
                    }
                }
            });
        };
        // 30ms 后渲染：给浏览器足够时间完成 display:none→block 的重排
        setTimeout(_doRenderRadar, 30);
    }

    function renderRadarChart(canvasId, abilities, playerIndex) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.error(`Canvas not found: ${canvasId}`);
            return;
        }
        const ctx = canvas.getContext('2d');
        if (Chart.getChart(canvasId)) Chart.getChart(canvasId).destroy();
        
        // 为每个球员使用不同的颜色主题，增强视觉差异
        const colorThemes = [
            { bg: 'rgba(255, 99, 132, 0.2)', border: '#ff6384' },   // 红色
            { bg: 'rgba(54, 162, 235, 0.2)', border: '#36a2eb' },   // 蓝色
            { bg: 'rgba(255, 206, 86, 0.2)', border: '#ffce56' },   // 黄色
            { bg: 'rgba(75, 192, 192, 0.2)', border: '#4bc0c0' }    // 青色
        ];
        
        // 使用传入的索引来选择颜色
        const theme = colorThemes[playerIndex % 4];
        
        // 确保能力值按固定顺序显示（防守、射门、传球、速度、体能）
        const orderedLabels = ["防守", "射门", "传球", "速度", "体能"];
        const orderedValues = orderedLabels.map(label => abilities[label] || 0);
        
        console.log(`渲染雷达图 ${canvasId} (球员${playerIndex}):`, orderedValues);  // 调试日志
        
        new Chart(ctx, {
            type: 'radar',
            data: {
                labels: orderedLabels,
                datasets: [{
                    data: orderedValues,
                    backgroundColor: theme.bg,
                    borderColor: theme.border,
                    borderWidth: 2,
                    pointBackgroundColor: theme.border,
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: theme.border
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { 
                    r: { 
                        min: 0, 
                        max: 100,
                        ticks: { 
                            display: true,
                            stepSize: 20,
                            color: 'rgba(255, 255, 255, 0.3)',
                            backdropColor: 'transparent'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        angleLines: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        pointLabels: {
                            color: 'rgba(255, 255, 255, 0.8)',
                            font: { size: 12 }
                        }
                    } 
                },
                plugins: { 
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.parsed.r + '/100';
                            }
                        }
                    }
                }
            }
        });
    }

    function drawTracking(analysis) {
        const canvas = document.getElementById('tracking-overlay');
        const video = document.getElementById('video-preview');
        if (!canvas || !video) return;
        const ctx = canvas.getContext('2d');
        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
        analysis.athletes.forEach((player, idx) => {
            const color = idx < 2 ? '#00ff88' : '#2288ff';
            if (player.actual_path && player.actual_path.length > 0) {
                ctx.beginPath();
                ctx.strokeStyle = color;
                ctx.lineWidth = 3;
                ctx.moveTo((player.actual_path[0].x / video.videoWidth) * canvas.width, (player.actual_path[0].y / video.videoHeight) * canvas.height);
                player.actual_path.forEach(p => ctx.lineTo((p.x / video.videoWidth) * canvas.width, (p.y / video.videoHeight) * canvas.height));
                ctx.stroke();
            }
        });
    }

    // ── 全局心跳：每2秒检查后端分析状态，驱动进度条更新 ──────────────
    async function heartbeat() {
        try {
            const r = await fetch(`${API_BASE}/api/analyzing`);
            if (!r.ok) return;
            const data = await r.json();

            if (!data.tasks || data.tasks.length === 0) {
                // 没有进行中的任务
                if (isAnalyzing && uploadedVideoId) {
                    // 可能刚刚完成，触发一次 showResults 渲染最终结果
                    // showResults 内部会判断完成状态
                }
                return;
            }

            // 取进度最高的任务
            const task = data.tasks.sort((a, b) => b.progress - a.progress)[0];

            // ── 情况1: 前端未在分析状态，自动恢复 ──
            // 若标注弹窗正在显示（用户还在标注），跳过自动恢复，不干扰标注流程
            const jerseyModalVisible = document.getElementById('jersey-modal') &&
                document.getElementById('jersey-modal').style.display !== 'none';
            if (!isAnalyzing && !jerseyModalVisible) {
                console.log(`[HEARTBEAT] 检测到进行中的分析任务: ${task.video_id} (${task.progress}%)`);
                uploadedVideoId = task.video_id;
                window.uploadedVideoId = task.video_id;
                isAnalyzing = true;
                uploadBtn.disabled = true;
                uploadBtn.textContent = '正在深度分析中...';
                statusBadge.textContent = `恢复分析进度: ${task.progress}%`;
                statusBadge.style.color = 'var(--accent)';
                progressContainer.style.display = 'block';
                progressBar.style.width = `${task.progress}%`;
                const inlineWrap = document.getElementById('inline-progress-wrap');
                const inlinePct  = document.getElementById('inline-progress-pct');
                const inlineBar  = document.getElementById('inline-progress-bar');
                const inlineHint = document.getElementById('inline-progress-hint');
                const placeholderText = document.getElementById('placeholder-main-text');
                if (inlineWrap) inlineWrap.style.display = 'block';
                if (inlinePct)  inlinePct.textContent = `${task.progress}%`;
                if (inlineBar)  inlineBar.style.width = `${task.progress}%`;
                if (inlineHint) inlineHint.textContent = getProgressHint(task.progress);
                if (placeholderText) placeholderText.textContent = '正在深度分析视频...';
                retryCount = 0; lastProgress = task.progress; staleProgressCount = 0;
                pollToken++;
                showResults(task.video_id, pollToken);
                return;
            }

            // ── 情况2: 前端已在分析状态，心跳直接更新进度 UI（保底） ──
            if (isAnalyzing && uploadedVideoId) {
                const prog = task.progress;
                progressBar.style.width = `${prog}%`;
                const _inlineBar = document.getElementById('inline-progress-bar');
                const _inlinePct = document.getElementById('inline-progress-pct');
                const _inlineHint = document.getElementById('inline-progress-hint');
                if (_inlineBar) _inlineBar.style.width = `${prog}%`;
                if (_inlinePct) _inlinePct.textContent = `${prog}%`;
                if (_inlineHint) _inlineHint.textContent = getProgressHint(prog);
                if (prog !== lastProgress) {
                    lastProgress = prog;
                    staleProgressCount = 0;
                    statusBadge.textContent = `分析进度: ${prog}%`;
                }
            }
        } catch (e) {
            // 后端未启动或网络问题，静默忽略
        }
    }

    // 立即执行一次，再每2秒循环
    heartbeat();
    heartbeatTimer = setInterval(heartbeat, 2000);
});

// =====================================================================
//  动态战术板 —— 独立模块
// =====================================================================
function initDynamicTacticalBoard(td, scope) {
    // scope 可以是 Element（弹窗容器）或 undefined（主页，使用 document）
    const $ = (id) => scope ? scope.querySelector('#' + id) : document.getElementById(id);
    const canvas  = $('tactical-canvas');
    if (!canvas) return;

    // ── 防止同一 canvas 被多次 init 导致 RAF 双跑 / 事件监听叠加 ──
    // 1. 先停止旧实例的 RAF 动画循环（若有）
    if (canvas._tacticalRafStop) canvas._tacticalRafStop();
    // 2. 再撤销旧实例注册的所有 DOM 事件监听
    if (canvas._tacticalAbort) canvas._tacticalAbort.abort();
    const _ac  = new AbortController();
    canvas._tacticalAbort = _ac;
    const _sig = { signal: _ac.signal };

    const tracks    = td.tactical_tracks  || {};   // {tid: {team, pts:[{f,wx,wy}]}}
    // 确保所有 track 的 pts 存在且按帧号升序（二分查找前提）
    Object.values(tracks).forEach(tr => {
        if (!Array.isArray(tr.pts)) tr.pts = [];
        if (tr.pts.length > 1) tr.pts.sort((a, b) => a.f - b.f);
    });

    // ══════════════════════════════════════════════════════════════
    // 前端离线轨迹后处理 — 让战术板看起来像真实球员轨迹
    // 处理顺序：离群剔除 → 短轨合并 → 间隙插值 → 平滑
    // ══════════════════════════════════════════════════════════════

    // ── Step 0：离群点剔除（H矩阵突然漂移产生的瞬间跳变点）──────
    // 策略：对每条轨迹，若某点与前后两点的距离都远超正常移速，视为噪点，
    //       用前后点的中点替换（而非删除，保持帧号连续性）
    (function removeOutliers() {
        const MAX_SPEED_M_PER_FRAME = 1.2; // 球员极限冲刺 ~10m/s, 采样3帧@25fps=0.12s → 1.2m/采样帧
        Object.values(tracks).forEach(tr => {
            const pts = tr.pts;
            const n = pts.length;
            if (n < 3) return;
            for (let pass = 0; pass < 3; pass++) { // 多轮迭代消除连续离群
                for (let i = 1; i < n - 1; i++) {
                    const prev = pts[i - 1], cur = pts[i], next = pts[i + 1];
                    if (cur.wx === undefined || prev.wx === undefined || next.wx === undefined) continue;
                    const dPrev = Math.hypot(cur.wx - prev.wx, cur.wy - prev.wy);
                    const dNext = Math.hypot(cur.wx - next.wx, cur.wy - next.wy);
                    const gapPrev = Math.max(1, cur.f - prev.f);
                    const gapNext = Math.max(1, next.f - cur.f);
                    // 若当前点到前后两点的帧速均超过阈值，判定为离群
                    if (dPrev / gapPrev > MAX_SPEED_M_PER_FRAME &&
                        dNext / gapNext > MAX_SPEED_M_PER_FRAME) {
                        pts[i] = {
                            ...cur,
                            wx: parseFloat(((prev.wx + next.wx) / 2).toFixed(3)),
                            wy: parseFloat(((prev.wy + next.wy) / 2).toFixed(3)),
                            interp: true,
                        };
                    }
                }
            }
        });
    })();

    // ── Step 1：短碎片轨迹合并到同队最近的长轨迹 ──────────────────
    (function mergeShortTracks() {
        const MAX_GAP   = 120;   // 最大允许合并的帧间隔（约4.8s@25fps）
        const MERGE_D   = 6.0;   // 最大允许合并的空间距离（米）
        const MIN_SHORT = 40;    // 点数少于此值视为短轨迹（可合并候选）
        let changed = true, iter = 0;
        while (changed && iter++ < 8) {
            changed = false;
            // 按点数升序处理，优先合并最短的碎片
            const tids = Object.keys(tracks).sort((a, b) =>
                tracks[a].pts.length - tracks[b].pts.length);
            for (const stid of tids) {
                if (!tracks[stid]) continue;
                const sPts = tracks[stid].pts;
                if (!sPts || sPts.length === 0) { delete tracks[stid]; continue; }
                if (sPts.length >= MIN_SHORT) continue;
                const sTeam = tracks[stid].team;
                const sFmin = sPts[0].f, sFmax = sPts[sPts.length - 1].f;
                const sWx0 = sPts[0].wx, sWy0 = sPts[0].wy;
                const sWxN = sPts[sPts.length - 1].wx, sWyN = sPts[sPts.length - 1].wy;
                // 若短轨首末坐标缺失则无法计算距离，跳过
                if (sWx0 === undefined || sWxN === undefined) continue;
                let bestTid = null, bestScore = -1;
                for (const ltid of Object.keys(tracks)) {
                    if (ltid === stid || !tracks[ltid]) continue;
                    const ltr = tracks[ltid];
                    if (ltr.team !== sTeam || ltr.pts.length < 3) continue;
                    const lPts = ltr.pts;
                    const lFmax = lPts[lPts.length - 1].f, lFmin = lPts[0].f;
                    const lWxN = lPts[lPts.length - 1].wx, lWyN = lPts[lPts.length - 1].wy;
                    const lWx0 = lPts[0].wx, lWy0 = lPts[0].wy;
                    // 长轨首末坐标缺失则跳过
                    if (lWxN === undefined || lWx0 === undefined) continue;
                    let gap, dist;
                    if (sFmin > lFmax) {
                        gap = sFmin - lFmax; dist = Math.hypot(sWx0 - lWxN, sWy0 - lWyN);
                    } else if (sFmax < lFmin) {
                        gap = lFmin - sFmax; dist = Math.hypot(sWxN - lWx0, sWyN - lWy0);
                    } else continue;
                    if (gap <= MAX_GAP && dist <= MERGE_D) {
                        const score = (1 - dist / MERGE_D) * (1 - gap / MAX_GAP);
                        if (score > bestScore) { bestScore = score; bestTid = ltid; }
                    }
                }
                if (bestTid) {
                    const lPts = tracks[bestTid].pts;
                    const combined = sFmin > lPts[lPts.length - 1].f
                        ? lPts.concat(sPts) : sPts.concat(lPts);
                    combined.sort((a, b) => a.f - b.f);
                    tracks[bestTid].pts = combined;
                    delete tracks[stid];
                    changed = true;
                }
            }
        }
        // 清除合并后帧号重复的点（取后者）
        Object.values(tracks).forEach(tr => {
            const seen = new Map();
            tr.pts.forEach(p => seen.set(p.f, p));
            tr.pts = Array.from(seen.values()).sort((a, b) => a.f - b.f);
        });
    })();

    // ── Step 2：轨迹内部间隙线性插值（填补遮挡空白帧）──────────────
    // 两点间距离还需满足速度合理性，才做插值；否则说明是真实大位移，不插
    (function interpolateGaps() {
        const MAX_GAP = 120;            // 最大插值间隔帧数（约4.8s）
        const MAX_INTERP_SPEED = 1.5;   // 插值段最大允许平均帧速（米/帧），超出则不插值
        Object.values(tracks).forEach(tr => {
            const pts = tr.pts;
            if (pts.length < 2) return;
            const filled = [pts[0]];
            for (let i = 1; i < pts.length; i++) {
                const prev = pts[i - 1], cur = pts[i];
                const gap = cur.f - prev.f;
                if (gap > 1 && gap <= MAX_GAP && cur.wx !== undefined && prev.wx !== undefined) {
                    const dist = Math.hypot(cur.wx - prev.wx, cur.wy - prev.wy);
                    // 速度合理性检查：平均帧速不超过阈值才插值
                    if (dist / gap <= MAX_INTERP_SPEED) {
                        for (let s = 1; s < gap; s++) {
                            const t = s / gap;
                            filled.push({
                                f:      prev.f + s,
                                wx:     parseFloat((prev.wx + t * (cur.wx - prev.wx)).toFixed(3)),
                                wy:     parseFloat((prev.wy + t * (cur.wy - prev.wy)).toFixed(3)),
                                interp: true,
                            });
                        }
                    }
                }
                filled.push(cur);
            }
            tr.pts = filled;
        });
    })();

    // ── Step 3：Savitzky-Golay 风格加权平滑（消除H矩阵抖动）────────
    // 使用三角形权重（中间权重最高），比均值平滑更好保留极值和转向
    (function smoothTracks() {
        const HW = 3; // half-window，实际窗口7点
        // 三角形权重：[1,2,3,4,3,2,1]（归一化）
        const rawW = [1, 2, 3, 4, 3, 2, 1];
        const sumW = rawW.reduce((a, b) => a + b, 0);
        const weights = rawW.map(w => w / sumW);
        Object.values(tracks).forEach(tr => {
            const pts = tr.pts;
            const n = pts.length;
            if (n < HW * 2 + 1) return;
            const smoothed = pts.map((pt, i) => {
                if (pt.wx === undefined) return pt;
                let wx = 0, wy = 0, wSum = 0;
                for (let j = -HW; j <= HW; j++) {
                    const idx = i + j;
                    if (idx < 0 || idx >= n || pts[idx].wx === undefined) continue;
                    const w = weights[j + HW];
                    wx += pts[idx].wx * w;
                    wy += pts[idx].wy * w;
                    wSum += w;
                }
                const newPt = {
                    f:  pt.f,
                    wx: parseFloat((wx / wSum).toFixed(3)),
                    wy: parseFloat((wy / wSum).toFixed(3)),
                };
                if (pt.interp)    newPt.interp = true;
                if (pt.predicted) newPt.predicted = true;
                return newPt;
            });
            tr.pts = smoothed;
        });
    })();
    // ══════════════════════════════════════════════════════════════
    // 前端后处理结束
    // ══════════════════════════════════════════════════════════════
    const ballPts   = (td.ball_traj_frames || []).slice().sort((a, b) => a.f - b.f);  // 球轨迹也排序
    const meta      = td.tactical_meta   || {};
    const totalFrames = meta.total_frames > 0 ? meta.total_frames : 1;
    const videoFps    = meta.fps > 0 ? meta.fps : 25;
    const totalSecs   = totalFrames / videoFps;   // 视频总时长（秒）
    const coordSpace  = meta.coord_space  || 'image';  // 'pitch' | 'tactical' | 'image'
    const videoW      = meta.video_w  || 1920;
    const videoH      = meta.video_h  || 1080;
    // pitch 真实世界尺寸（米），仅 coord_space='pitch' 时使用
    const PITCH_L = meta.pitch_l || 105.0;
    const PITCH_W = meta.pitch_w || 68.0;

    // 格式化秒数为 mm:ss
    function fmtTime(sec) {
        const s = Math.max(0, Math.floor(sec));
        const m = Math.floor(s / 60);
        const ss = String(s % 60).padStart(2, '0');
        return `${m}:${ss}`;
    }
    // 战术板画布逻辑尺寸（与后端 build_tactical_view 保持一致）
    const CW = 800, CH = 520, M = 30;

    // 设置 Canvas 分辨率
    canvas.width  = CW;
    canvas.height = CH;
    const ctx = canvas.getContext('2d');

    // ── 坐标映射 ──
    // coord_space='pitch' ：wx/wy 为 pitch 米制坐标（场地中心原点），映射到 wc() 画布坐标
    // coord_space='tactical'：后端已把坐标投影到 800×520，直接用（旧方案兼容）
    // coord_space='image'  ：原始视频坐标，线性缩放到战术板
    function toCanvas(x, y) {
        if (coordSpace === 'pitch') {
            // 参照 sn-gamestate pitch.py：pitch坐标→画布坐标，与 wc() 函数公式一致
            return wc(x, y);
        }
        if (coordSpace === 'tactical') return [x, y];
        const tx = M + (x / videoW) * (CW - 2 * M);
        const ty = M + (y / videoH) * (CH - 2 * M);
        return [tx, ty];
    }

    // ── 取点坐标（兼容新旧格式）──
    // 新格式：{f, wx, wy}（pitch坐标）；旧格式：{f, x, y}（canvas或image坐标）
    function ptXY(pt) {
        if (pt.wx !== undefined && pt.wy !== undefined) return [pt.wx, pt.wy];
        return [pt.x ?? 0, pt.y ?? 0];
    }

    // ── 跳点过滤：连续两点间距超过阈值时认为是H矩阵漂移造成的虚假跳变，不连线 ──
    // 球员阈值：前端后处理（Step0~3）已消除大部分跳变，这里做最终兜底
    //   6.0m = 后处理后允许的最大帧间距（与 trail/drawFullTrail 渲染层保持一致）
    // 球轨迹阈值：大脚长传球速~30m/s，0.12s最多3.6m；但H矩阵误差可达10~20m
    //   设15m（比球员宽裕3倍），过滤H误差的同时允许真实长传
    const JUMP_THRESH      = coordSpace === 'pitch' ? 6.0 : 120;
    const JUMP_THRESH_BALL = coordSpace === 'pitch' ? 30.0 : 200;

    // ── 边界过滤：超出场地 +5m 的坐标视为H矩阵失效产生的噪点，不渲染 ──
    const BOUND_X = PITCH_L / 2 + 5;
    const BOUND_Y = PITCH_W / 2 + 5;
    function inBounds(wx, wy) {
        return coordSpace !== 'pitch' || (Math.abs(wx) <= BOUND_X && Math.abs(wy) <= BOUND_Y);
    }

    // ── 球队颜色 ──
    const TEAM_COLOR = { team_a: '#ff7828', team_b: '#3c64ff', ball: '#00ffff' };

    // ── 预计算所有 track 的帧范围 ──
    let globalMinF = Infinity, globalMaxF = -Infinity;
    Object.values(tracks).forEach(tr => {
        tr.pts.forEach(p => {
            if (p.f < globalMinF) globalMinF = p.f;
            if (p.f > globalMaxF) globalMaxF = p.f;
        });
    });
    ballPts.forEach(p => {
        if (p.f < globalMinF) globalMinF = p.f;
        if (p.f > globalMaxF) globalMaxF = p.f;
    });
    if (!isFinite(globalMinF)) globalMinF = 0;
    // 确保 globalMaxF 不小于 totalFrames，让进度条能覆盖视频全长
    // 即使分析提前结束，拖动进度条到末尾时不会"挤压"在前半段
    if (globalMaxF < totalFrames) globalMaxF = totalFrames;
    const frameRange = globalMaxF - globalMinF || 1;

    // ── 状态 ──
    let currentProgress = 0;   // 0.0 ~ 1.0
    let playing  = false;
    let speed    = 1;
    let filter   = 'all';      // 'all' | 'team_a' | 'team_b' | 'ball'
    let rafId    = null;
    let lastTs   = null;

    // 注册 RAF 停止回调，供下次 init 时清理
    canvas._tacticalRafStop = () => {
        playing = false;
        if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
    };

    // ── 单球员追踪模式状态 ──
    let isSingleMode = false;
    let selectedTid  = null;
    let hitZones     = [];   // [{tid, cx, cy}]，每帧 render 时更新

    // ── 场地线绘制 ──
    // PL/PW 用于场地线标注（FIFA标准值，固定）
    // PITCH_L/PITCH_W 来自后端 meta，用于 pitch坐标→画布坐标 转换（wc函数）
    const PL = 105, PW = 68;
    const cw = CW - 2 * M, ch = CH - 2 * M;
    function wc(wx, wy) {
        // pitch坐标（米制，场地中心为原点）→ 画布像素坐标
        // 先 clamp 到场地合法范围（±5% 余量），防止场外坐标渲染到画布边界外
        const halfL = PITCH_L / 2 * 1.05;
        const halfW = PITCH_W / 2 * 1.05;
        const cwx = Math.max(-halfL, Math.min(halfL, wx));
        const cwy = Math.max(-halfW, Math.min(halfW, wy));
        return [
            M + (cwx + PITCH_L / 2) / PITCH_L * cw,
            M + (cwy + PITCH_W / 2) / PITCH_W * ch
        ];
    }
    function drawPitch() {
        ctx.globalAlpha = 1;  // 每帧重置，防止上一帧轨迹透明度残留
        // 草地背景
        ctx.fillStyle = '#1e6e1e';
        ctx.fillRect(0, 0, CW, CH);
        // 深浅条纹
        const stripes = 8;
        for (let i = 0; i < stripes; i++) {
            ctx.fillStyle = i % 2 === 0 ? 'rgba(0,0,0,0.07)' : 'rgba(255,255,255,0.04)';
            ctx.fillRect(M + i * cw / stripes, M, cw / stripes, ch);
        }
        ctx.strokeStyle = 'rgba(255,255,255,0.85)';
        ctx.lineWidth   = 1.5;
        // 外框
        const [lx, ly] = wc(-PL/2, -PW/2);
        const [rx, ry] = wc( PL/2,  PW/2);
        ctx.strokeRect(lx, ly, rx-lx, ry-ly);
        // 中线
        line(wc(0,-PW/2), wc(0, PW/2));
        // 中圈
        const [cx_, cy_] = wc(0, 0);
        const R = 9.15;
        ctx.beginPath();
        ctx.ellipse(cx_, cy_, R/PL*cw, R/PW*ch, 0, 0, Math.PI*2);
        ctx.stroke();
        dot(wc(0,0), 3);
        // 大禁区
        rect(wc(-PL/2, -20.16), wc(-PL/2+16.5, 20.16));
        rect(wc( PL/2-16.5, -20.16), wc( PL/2, 20.16));
        // 小禁区
        rect(wc(-PL/2, -9.16), wc(-PL/2+5.5, 9.16));
        rect(wc( PL/2-5.5, -9.16), wc( PL/2, 9.16));
        // 点球点
        dot(wc(-PL/2+11, 0), 3);
        dot(wc( PL/2-11, 0), 3);
        // 罚球弧
        arc(wc(-PL/2+11, 0), R/PL*cw, R/PW*ch, -53, 53);
        arc(wc( PL/2-11, 0), R/PL*cw, R/PW*ch, 127, 233);
    }
    function line([x1,y1],[x2,y2]) { ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke(); }
    function rect([x1,y1],[x2,y2]) { ctx.strokeRect(x1,y1,x2-x1,y2-y1); }
    function dot([x,y], r) { ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fillStyle='rgba(255,255,255,0.85)'; ctx.fill(); }
    function arc([cx,cy], rx, ry, a1, a2) {
        ctx.beginPath();
        ctx.ellipse(cx, cy, rx, ry, 0, a1*Math.PI/180, a2*Math.PI/180);
        ctx.stroke();
    }

    // ── 获取某进度下各 track 的位置和尾迹 ──
    const TRAIL_LEN    = 25;   // 尾迹保留最近25个采样点（3帧采样密度更高，适当延长）
    // ACTIVE_WINDOW：根据实际数据点间距动态计算，确保降采样后球员圆点不被过滤消失
    // 下限从100→200：后端 max_lost=40 时卡尔曼可预测约40帧，ACTIVE_WINDOW须大于此值
    let ACTIVE_WINDOW = 300;
    // 动态计算：取所有 track 相邻点间距的 p90 × 6
    // p90 × 6 能覆盖 max_lost=40 帧内的连续遮挡段，防止球员圆点在遮挡期间消失
    // 同时排除断轨大间距（>50帧），避免 ACTIVE_WINDOW 被拉得过大
    {
        const gaps = [];
        Object.values(tracks).forEach(tr => {
            const pts = tr.pts;
            for (let i = 1; i < pts.length; i++) {
                const g = pts[i].f - pts[i-1].f;
                // 只收集正常范围内的间距（2~50帧），排除断轨重连的巨大间距污染
                if (g > 0 && g <= 50) gaps.push(g);
            }
        });
        if (gaps.length > 0) {
            gaps.sort((a, b) => a - b);
            const p90gap = gaps[Math.floor(gaps.length * 0.9)] || gaps[gaps.length - 1];
            ACTIVE_WINDOW = Math.min(2000, Math.max(150, Math.ceil(p90gap * 3)));
        }
        console.log(`[TACTICAL] ACTIVE_WINDOW=${ACTIVE_WINDOW} 帧（${gaps.length} 个间距样本）`);
    }

    // ── Catmull-Rom 样条插值（替代线性 lerp）──
    // 使用4个控制点（p0=前前，p1=当前，p2=下一，p3=下下）计算 t 处的平滑曲线位置
    // 相比线性插值，能自然还原球员的曲线跑动（转向、弧线跑）
    // alpha=0.5 为 Centripetal Catmull-Rom，避免锐角处产生"环形"（cusps）
    function catmullRomInterp(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, t) {
        // Catmull-Rom 矩阵公式（alpha = 0.5，centripetal）
        const t2 = t * t, t3 = t2 * t;
        const x = 0.5 * ((2 * p1x) +
            (-p0x + p2x) * t +
            (2 * p0x - 5 * p1x + 4 * p2x - p3x) * t2 +
            (-p0x + 3 * p1x - 3 * p2x + p3x) * t3);
        const y = 0.5 * ((2 * p1y) +
            (-p0y + p2y) * t +
            (2 * p0y - 5 * p1y + 4 * p2y - p3y) * t2 +
            (-p0y + 3 * p1y - 3 * p2y + p3y) * t3);
        return [x, y];
    }

    function getTrackState(pts, progress, forceShow) {
        const cutF = globalMinF + progress * frameRange;
        // 二分查找：找到最后一个 f <= cutF 的索引
        let lo = 0, hi = pts.length - 1, lastIdx = -1;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (pts[mid].f <= cutF) { lastIdx = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        if (lastIdx < 0) return null;
        const cur = pts[lastIdx];
        // 活跃窗口过滤：当前帧距 track 最新点超过 ACTIVE_WINDOW 帧，视为该 track 已离场，不渲染
        // forceShow=true（选中球员）时跳过此过滤，确保选中球员始终可见
        if (!forceShow && cutF - cur.f > ACTIVE_WINDOW) return null;
        const startIdx = Math.max(0, lastIdx - TRAIL_LEN + 1);

        // ── 帧间 Catmull-Rom 样条插值（替代线性 lerp）──
        // 后端每3帧采样一次，前端 RAF 60fps 驱动；
        // Catmull-Rom 利用前后共4个控制点，生成过每个采样点的平滑曲线
        // 相比线性插值，能自然还原球员曲线跑动（转向、弧线跑、盘带）
        // 插值前提：下一采样点存在且帧距 <= 20（排除长时段断轨的错误插值）
        let smoothedCur = cur;
        if (coordSpace === 'pitch' && cur.wx !== undefined) {
            const next = lastIdx + 1 < pts.length ? pts[lastIdx + 1] : null;
            const gapToNext = next ? next.f - cur.f : Infinity;
            // 插值间距上限收紧到15帧（后端3帧采样，15帧≈0.6s，超过此间距说明轨迹断开，不应插值）
            if (next && gapToNext > 0 && gapToNext <= 30 && next.wx !== undefined) {
                // 跳变检测：若前后两点距离 > JUMP_THRESH，不插值（避免在跳变段插出中间虚假轨迹）
                const dist = Math.hypot(next.wx - cur.wx, next.wy - cur.wy);
                if (dist <= JUMP_THRESH) {
                    const t = (cutF - cur.f) / gapToNext;  // 0 ~ 1 进度
                    // 获取4个控制点（Catmull-Rom 需要 p-1, p0, p1, p2）
                    const prev = lastIdx > 0 ? pts[lastIdx - 1] : null;
                    const next2 = lastIdx + 2 < pts.length ? pts[lastIdx + 2] : null;
                    // 检查前后点是否有效（在跳变范围内）
                    const prevOk = prev && prev.wx !== undefined &&
                        Math.hypot(cur.wx - prev.wx, cur.wy - prev.wy) <= JUMP_THRESH;
                    const next2Ok = next2 && next2.wx !== undefined &&
                        Math.hypot(next2.wx - next.wx, next2.wy - next.wy) <= JUMP_THRESH;
                    // p0: 前一点（若无效则用 cur 镜像）
                    const p0x = prevOk ? prev.wx : (2 * cur.wx - next.wx);
                    const p0y = prevOk ? prev.wy : (2 * cur.wy - next.wy);
                    // p3: 后后点（若无效则用 next 镜像）
                    const p3x = next2Ok ? next2.wx : (2 * next.wx - cur.wx);
                    const p3y = next2Ok ? next2.wy : (2 * next.wy - cur.wy);
                    const [iwx, iwy] = catmullRomInterp(p0x, p0y, cur.wx, cur.wy, next.wx, next.wy, p3x, p3y, t);
                    smoothedCur = { ...cur, wx: iwx, wy: iwy };
                }
            }

            // ── 三点中值滤波（消除残余单帧噪声，仅对原始采样点时刻执行）──
            if (smoothedCur === cur) {
                const prev1 = lastIdx > 0 ? pts[lastIdx - 1] : null;
                const prev2 = lastIdx > 1 ? pts[lastIdx - 2] : null;
                if (prev1 && prev2 && prev1.wx !== undefined && prev2.wx !== undefined) {
                    const wxVals = [prev2.wx, prev1.wx, cur.wx].sort((a, b) => a - b);
                    const wyVals = [prev2.wy, prev1.wy, cur.wy].sort((a, b) => a - b);
                    const mwx = wxVals[1], mwy = wyVals[1];
                    const dev = Math.hypot(mwx - cur.wx, mwy - cur.wy);
                    // 阈值1.5m：仅过滤明显单帧噪声，不影响正常球员移动
                    if (dev > 1.5) {
                        smoothedCur = { ...cur, wx: mwx, wy: mwy };
                    }
                }
            }
        }

        return { trail: pts.slice(startIdx, lastIdx + 1), cur: smoothedCur };
    }

    // ── 缓存 DOM 元素，避免重复 querySelector ──
    const seekEl        = $('tact-seek');
    const timeEl        = $('tact-time');
    const playBtn       = $('tact-play-btn');
    const selectedLbl   = $('tact-selected-label');
    const resetBtn      = $('tact-reset-btn');
    const speedEl       = $('tact-speed');
    const singleModeBtn = $('tact-single-mode');

    // ── 渲染一帧 ──
    // 空间去重阈值：pitch模式用米制（1.0m），只去重真正重叠的同一球员多轨迹
    // 不能过大：2.5m会把距离近的不同球员（密集对抗、角球）也去重，导致球员"消失"
    const DEDUP_DIST = coordSpace === 'pitch'
        ? 1.0
        : 1.0 / PITCH_L * cw;   // image/tactical 模式下对应的像素距离

    // ── 球坐标辅助函数（纯函数，提到 render 外避免每帧重建闭包）──
    // ballInBounds：球坐标为像素格式（非 wx/wy pitch格式）时的边界过滤，允许轻微超出视频边界
    const ballInBounds = (bx, by) => bx >= -50 && bx <= videoW + 50 && by >= -50 && by <= videoH + 50;
    // 球始终用 image→canvas 线性映射，不受 coordSpace 影响
    const ballToCanvas = (bx, by) => [
        M + (bx / videoW) * (CW - 2 * M),
        M + (by / videoH) * (CH - 2 * M)
    ];
    // toBallCanvas：统一球点坐标转换（提到 render 外避免每帧重建）
    // 若数据点有 wx/wy 视为 pitch 坐标，否则视为 image 像素坐标
    const toBallCanvas = (p) => {
        const hasPitch = p.wx !== undefined && p.wy !== undefined;
        const [rx, ry] = hasPitch ? [p.wx, p.wy] : [p.x ?? 0, p.y ?? 0];
        const valid = hasPitch ? inBounds(rx, ry) : ballInBounds(rx, ry);
        const [cx, cy] = hasPitch ? toCanvas(rx, ry) : ballToCanvas(rx, ry);
        return { cx, cy, valid, raw: [rx, ry], hasPitch };
    };

    // ── 一次性 debug ──
    let _dbgDone = false;
    function _debugOnce() {
        if (_dbgDone) return; _dbgDone = true;
        console.group('[TACTICAL DEBUG]');
        console.log('coordSpace:', coordSpace);
        console.log('totalFrames:', totalFrames, 'videoFps:', videoFps, 'animDuration:', animDuration);
        console.log('globalMinF:', globalMinF, 'globalMaxF:', globalMaxF, 'frameRange:', frameRange);
        console.log('ballPts.length:', ballPts.length, 'first ball pt:', ballPts[0]);
        const trackIds = Object.keys(tracks);
        console.log('tracks count:', trackIds.length);
        if (trackIds.length > 0) {
            const sampleTr = tracks[trackIds[0]];
            console.log('sample track[0] team:', sampleTr.team, 'pts.length:', sampleTr.pts.length, 'first pt:', sampleTr.pts[0]);
        }
        console.log('JUMP_THRESH:', JUMP_THRESH, 'BOUND_X:', BOUND_X, 'BOUND_Y:', BOUND_Y);
        console.groupEnd();
    }

    function render() {
        _debugOnce();
        drawPitch();
        hitZones = [];   // 每帧重置，收集可点击的球员圆点

        const showBall = filter === 'all' || filter === 'ball';

        // 球员轨迹
        // ── 第一步：收集所有 track 的当前状态，空间去重（同队距离 < DEDUP_DIST 只保留帧号更新的）──
        // 原因：Re-ID 断轨后重新关联会产生新 track_id，同一球员可能同时有多个活跃 track
        // 对每支队伍维护一个已确认的"代表点"列表，新点与所有已确认点距离均 >= DEDUP_DIST 才加入
        const activeByTeam = {};  // { team: [{wx, wy, trail, curF, tid, team}] }

        Object.entries(tracks).forEach(([tid, tr]) => {
            const isThisSelected = isSingleMode && tid === selectedTid;
            // 筛选过滤：选中球员不受 filter 影响，始终显示
            if (!isThisSelected) {
                // filter='all' 时全显；否则只显示与 filter 匹配的队伍
                if (filter === 'team_a' && tr.team !== 'team_a') return;
                if (filter === 'team_b' && tr.team !== 'team_b') return;
                if (filter === 'ball') return;  // 只显示球，所有球员隐藏
            }
            const state = getTrackState(tr.pts, currentProgress, isThisSelected);
            if (!state) return;
            const [curWx, curWy] = ptXY(state.cur);
            if (!inBounds(curWx, curWy)) return;

            const teamList = activeByTeam[tr.team] || [];
            activeByTeam[tr.team] = teamList;

            // 检查是否与已有点过近
            let tooClose = false;
            for (let k = 0; k < teamList.length; k++) {
                const d = Math.hypot(curWx - teamList[k].wx, curWy - teamList[k].wy);
                if (d < DEDUP_DIST) {
                    // 选中球员强制替换当前槽位（保留选中 tid）
                    if (isThisSelected) {
                        teamList[k] = { wx: curWx, wy: curWy, trail: state.trail, curF: state.cur.f, team: tr.team, tid };
                    } else if (state.cur.f > teamList[k].curF && teamList[k].tid !== selectedTid) {
                        // 普通去重：保留帧号更新的（但不能覆盖选中球员）
                        teamList[k] = { wx: curWx, wy: curWy, trail: state.trail, curF: state.cur.f, team: tr.team, tid };
                    }
                    tooClose = true;
                    break;
                }
            }
            if (!tooClose) {
                teamList.push({ wx: curWx, wy: curWy, trail: state.trail, curF: state.cur.f, team: tr.team, tid });
            }
        });

        // ── 第二步：渲染去重后的球员列表 ──
        // 遍历实际收集到的所有队伍（而非硬编码），兼容后端产生非 team_a/team_b 标签的情况
        Object.keys(activeByTeam).forEach(team => {
            const list = activeByTeam[team] || [];
            const color = TEAM_COLOR[team] || '#aaa';
            list.forEach(item => {
                const trail = item.trail;
                const isSelected = isSingleMode && selectedTid === item.tid;
                // 单球员模式下：非选中球员半透明
                const baseAlpha = isSingleMode && !isSelected ? 0.25 : 1;

                // 尾迹：选中球员用累积全程轨迹，普通球员用滑动窗口
                if (isSelected) {
                    // 累积轨迹：从第0帧到当前帧的全部路径
                    if (item.tid && tracks[item.tid]) {
                        drawFullTrail(tracks[item.tid], currentProgress, color);
                    }
                } else if (trail.length >= 1) {
                    ctx.strokeStyle = color;
                    ctx.lineWidth   = 2;
                    // 去除 trail 中坐标完全相同的连续重复点（H矩阵冻结时产生），只保留有位移的点
                    const dedupTrail = [trail[0]];
                    for (let i = 1; i < trail.length; i++) {
                        const [px, py] = ptXY(trail[i]);
                        const [qx, qy] = ptXY(dedupTrail[dedupTrail.length - 1]);
                        if (Math.hypot(px - qx, py - qy) > 0.05) dedupTrail.push(trail[i]);
                    }
                    // 画采样点之间的轨迹线（去重后）—— 用 Catmull-Rom 贝塞尔曲线使尾迹平滑
                    // 将相邻点间的切线方向转换为 bezierCurveTo 的控制点
                    const n = dedupTrail.length;
                    if (n >= 2) {
                        // 预先转换所有点到 canvas 坐标
                        const cPts = dedupTrail.map(p => {
                            const [wx_, wy_] = ptXY(p);
                            return inBounds(wx_, wy_) ? toCanvas(wx_, wy_) : null;
                        });
                        for (let i = 1; i < n; i++) {
                            if (!cPts[i-1] || !cPts[i]) continue;
                            const [wx1, wy1] = ptXY(dedupTrail[i-1]);
                            const [wx2, wy2] = ptXY(dedupTrail[i]);
                            // 后处理已消除跳变，用宽松阈值（6m）仅过滤真正的离场瞬移
                            if (Math.hypot(wx2 - wx1, wy2 - wy1) > 6.0) continue;
                            const alpha = baseAlpha * (0.4 + 0.6 * (i / Math.max(1, n - 1)));
                            ctx.globalAlpha = alpha;
                            // Catmull-Rom → Bezier 控制点（tension = 0.5）
                            // 切线向量：使用前后相邻点计算
                            const p0c = cPts[Math.max(0, i - 2)] || cPts[i - 1];
                            const p1c = cPts[i - 1];
                            const p2c = cPts[i];
                            const p3c = cPts[Math.min(n - 1, i + 1)] || cPts[i];
                            const cp1x = p1c[0] + (p2c[0] - p0c[0]) / 6;
                            const cp1y = p1c[1] + (p2c[1] - p0c[1]) / 6;
                            const cp2x = p2c[0] - (p3c[0] - p1c[0]) / 6;
                            const cp2y = p2c[1] - (p3c[1] - p1c[1]) / 6;
                            ctx.beginPath();
                            ctx.moveTo(p1c[0], p1c[1]);
                            ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2c[0], p2c[1]);
                            ctx.stroke();
                        }
                    }
                    // 补画最后采样点 → 当前插值位置（平滑轨迹末端）
                    const lastTrailPt = trail[trail.length - 1];
                    const [lwx, lwy] = ptXY(lastTrailPt);
                    if (inBounds(lwx, lwy) && inBounds(item.wx, item.wy)) {
                        const interpDist = Math.hypot(item.wx - lwx, item.wy - lwy);
                        if (interpDist <= 6.0) {
                            ctx.globalAlpha = baseAlpha;
                            ctx.beginPath();
                            const [lx, ly] = toCanvas(lwx, lwy);
                            const [ix, iy] = toCanvas(item.wx, item.wy);
                            ctx.moveTo(lx, ly);
                            ctx.lineTo(ix, iy);
                            ctx.stroke();
                        }
                    }
                }
                // 当前位置圆点
                ctx.globalAlpha = baseAlpha;
                ctx.lineWidth   = isSelected ? 2 : 1;   // 防止 drawFullTrail(2.5) 或上帧遗留值影响描边
                const [cx, cy] = toCanvas(item.wx, item.wy);
                // 收集 hitZone（用于点击检测）
                hitZones.push({ tid: item.tid, cx, cy });
                if (isSelected) {
                    drawGlowDot(cx, cy, color);
                } else {
                    ctx.beginPath();
                    ctx.arc(cx, cy, 6, 0, Math.PI * 2);
                    ctx.fillStyle = color;
                    ctx.fill();
                    ctx.strokeStyle = '#fff';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
                ctx.globalAlpha = 1;
            });
        });

        // 球轨迹
        // 注意：ball_traj_frames 存的是 pitch 坐标 {f, wx, wy}（与球员轨迹坐标系一致）
        // toBallCanvas 内部通过 hasPitch 标志自动选择 pitch→canvas 映射（wc 函数）
        if (showBall && ballPts.length > 0) {
            const cutF = globalMinF + currentProgress * frameRange;
            // 二分查找：找到最后一个 f <= cutF 的截止索引
            let bLo = 0, bHi = ballPts.length - 1, bEnd = -1;
            while (bLo <= bHi) {
                const mid = (bLo + bHi) >> 1;
                if (ballPts[mid].f <= cutF) { bEnd = mid; bLo = mid + 1; }
                else bHi = mid - 1;
            }
            // 从截止索引往前取候选点（窗口取 TRAIL_LEN*4 防边界点过多导致有效点不足）
            const ballSlice = bEnd >= 0 ? ballPts.slice(Math.max(0, bEnd - TRAIL_LEN * 4 + 1), bEnd + 1) : [];

            // ── 球轨迹中值滤波预处理 ──────────────────────────────────────────
            // 对每个点检查是否为孤立噪声（偏离前后点的中位值超过阈值）
            // 球在0.12s内极限移动：大脚球速~30m/s → 3.6m；设5m过滤误检
            const BALL_MEDIAN_THRESH = 5.0;  // 米
            const filteredBallSlice = ballSlice.filter((p, i, arr) => {
                // 边界点直接保留
                if (i === 0 || i === arr.length - 1) return true;
                const prev = arr[i - 1], next = arr[i + 1];
                if (!prev || !next) return true;
                const hasPitch = p.wx !== undefined;
                if (!hasPitch) return true;
                // 前后点与当前点的距离
                const d_prev = Math.hypot(p.wx - prev.wx, p.wy - prev.wy);
                const d_next = Math.hypot(p.wx - next.wx, p.wy - next.wy);
                const d_span = Math.hypot(next.wx - prev.wx, next.wy - prev.wy);
                // 如果当前点到前后点距离之和 >> 前后点直接距离，认为是噪声绕路点
                // 绕路比例超过2倍时过滤（前后跨越 d_span，但经过当前点需要 d_prev+d_next >> d_span）
                if (d_span > 0 && (d_prev + d_next) > d_span * 2.5 && d_prev > BALL_MEDIAN_THRESH && d_next > BALL_MEDIAN_THRESH) {
                    return false;
                }
                return true;
            });

            // ── 球当前位置插值平滑（Catmull-Rom）────────────────────────────
            // 在最后一个已知点与下一个点之间做插值，使球的运动更流畅
            let ballCurPoint = null;
            if (bEnd >= 0 && bEnd < ballPts.length) {
                const bCur = ballPts[bEnd];
                const bNext = bEnd + 1 < ballPts.length ? ballPts[bEnd + 1] : null;
                const bGap = bNext ? bNext.f - bCur.f : Infinity;
                // 插值条件：下一点存在，间距<=45帧，且不跳变
                if (bNext && bGap > 0 && bGap <= 45 && bCur.wx !== undefined && bNext.wx !== undefined) {
                    const bDist = Math.hypot(bNext.wx - bCur.wx, bNext.wy - bCur.wy);
                    if (bDist <= JUMP_THRESH_BALL) {
                        const bt = (cutF - bCur.f) / bGap;
                        const bPrev = bEnd > 0 ? ballPts[bEnd - 1] : null;
                        const bNext2 = bEnd + 2 < ballPts.length ? ballPts[bEnd + 2] : null;
                        const bPrevOk = bPrev && bPrev.wx !== undefined &&
                            Math.hypot(bCur.wx - bPrev.wx, bCur.wy - bPrev.wy) <= JUMP_THRESH_BALL;
                        const bNext2Ok = bNext2 && bNext2.wx !== undefined &&
                            Math.hypot(bNext2.wx - bNext.wx, bNext2.wy - bNext.wy) <= JUMP_THRESH_BALL;
                        const bp0x = bPrevOk ? bPrev.wx : (2 * bCur.wx - bNext.wx);
                        const bp0y = bPrevOk ? bPrev.wy : (2 * bCur.wy - bNext.wy);
                        const bp3x = bNext2Ok ? bNext2.wx : (2 * bNext.wx - bCur.wx);
                        const bp3y = bNext2Ok ? bNext2.wy : (2 * bNext.wy - bCur.wy);
                        const [ibwx, ibwy] = catmullRomInterp(bp0x, bp0y, bCur.wx, bCur.wy, bNext.wx, bNext.wy, bp3x, bp3y, bt);
                        ballCurPoint = { ...bCur, wx: ibwx, wy: ibwy };
                    }
                }
            }

            // 过滤边界外点，取最近 TRAIL_LEN 个，缓存画布坐标避免重复计算
            const recentMapped = filteredBallSlice
                .map(toBallCanvas)
                .filter(p => p.valid)
                .slice(-TRAIL_LEN);

            if (recentMapped.length > 1) {
                ctx.strokeStyle = '#00ffff';
                ctx.lineWidth   = 2;
                // 球轨迹也用 Catmull-Rom 贝塞尔曲线（球的抛物线、弧线等非线性运动）
                for (let i = 1; i < recentMapped.length; i++) {
                    const p1 = recentMapped[i - 1];
                    const p2 = recentMapped[i];
                    if (p1.hasPitch !== p2.hasPitch) continue;
                    if (Math.hypot(p2.raw[0] - p1.raw[0], p2.raw[1] - p1.raw[1]) > JUMP_THRESH_BALL) continue;
                    ctx.globalAlpha = 0.2 + 0.8 * (i / Math.max(1, recentMapped.length - 1));
                    // Catmull-Rom → Bezier 控制点
                    const p0 = recentMapped[Math.max(0, i - 2)];
                    const p3 = recentMapped[Math.min(recentMapped.length - 1, i + 1)];
                    const cp1x = p1.cx + (p2.cx - p0.cx) / 6;
                    const cp1y = p1.cy + (p2.cy - p0.cy) / 6;
                    const cp2x = p2.cx - (p3.cx - p1.cx) / 6;
                    const cp2y = p2.cy - (p3.cy - p1.cy) / 6;
                    ctx.beginPath();
                    ctx.moveTo(p1.cx, p1.cy);
                    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.cx, p2.cy);
                    ctx.stroke();
                }
            }
            if (recentMapped.length > 0) {
                ctx.globalAlpha = 1;
                // 优先使用插值平滑位置（Catmull-Rom），回退到最后采样点
                let bx, by;
                if (ballCurPoint) {
                    const mapped = toBallCanvas(ballCurPoint);
                    if (mapped.valid) {
                        bx = mapped.cx; by = mapped.cy;
                    }
                }
                if (bx === undefined) {
                    bx = recentMapped[recentMapped.length - 1].cx;
                    by = recentMapped[recentMapped.length - 1].cy;
                }
                ctx.beginPath();
                ctx.arc(bx, by, 5, 0, Math.PI * 2);
                ctx.fillStyle = '#00ffff';
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        }

        ctx.globalAlpha = 1;

        // 更新进度显示（seekEl/timeEl 已在 init 时缓存，无需每帧 querySelector）
        if (seekEl) seekEl.value = (currentProgress * 100).toFixed(2);
        if (timeEl) timeEl.textContent = `${fmtTime(currentProgress * totalSecs)} / ${fmtTime(totalSecs)}`;

        // 更新左侧体能数据面板
        updateStatsPanel();
    }

    // 动画推进每帧的进度增量基准（视频实际时长，秒）；至少 1s 防除零
    const animDuration = Math.max(1, totalFrames / videoFps);

    // ── 动画循环 ──
    function animate(ts) {
        if (!playing) { rafId = null; return; }
        if (lastTs !== null) {
            // clamp dt 最大 1/15s，防止 Tab 切换/页面隐藏后 RAF 节流恢复时一步跳帧
            const dt = Math.min((ts - lastTs) / 1000, 1 / 15);
            // 以视频实际时长为1倍速基准；tactical_tracks 每5帧采样一次，但动画总时长与采样间隔无关
            currentProgress += speed * dt / animDuration;
            if (currentProgress >= 1) {
                currentProgress = 1;
                playing = false;
                rafId = null;   // 播放自然结束，显式清零，避免 _tacticalRafStop 误判动画仍在运行
                updatePlayBtn();
                render();
                return;
            }
        }
        lastTs = ts;
        render();
        if (playing) rafId = requestAnimationFrame(animate);
    }

    function updatePlayBtn() {
        if (!playBtn) return;
        playBtn.innerHTML = playing ? '<i class="fas fa-pause"></i>' : '<i class="fas fa-play"></i>';
    }

    // ── 控件绑定（统一使用缓存的 DOM 引用）──
    playBtn?.addEventListener('click', () => {
        if (currentProgress >= 1) { currentProgress = 0; }
        playing = !playing;
        updatePlayBtn();
        if (playing) {
            // 先取消可能残留的旧 raf（防止快速连点时产生两个并行动画循环）
            if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
            lastTs = null;
            rafId = requestAnimationFrame(animate);
        } else {
            if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
        }
    }, _sig);

    resetBtn?.addEventListener('click', () => {
        playing = false;
        updatePlayBtn();
        if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
        currentProgress = 0;
        lastTs = null;
        render();
    }, _sig);

    seekEl?.addEventListener('input', (e) => {
        // 拖动进度条时暂停动画循环，避免 animate 与 seek 竞争 currentProgress
        if (playing) {
            playing = false;
            updatePlayBtn();
            if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
        }
        currentProgress = Math.max(0, Math.min(1, parseFloat(e.target.value) / 100));
        lastTs = null;   // 清除时间戳，防止恢复播放时 dt 异常
        render();
    }, _sig);

    // speedEl 为 <select>，监听 change 即可
    speedEl?.addEventListener('change', (e) => {
        const v = parseFloat(e.target.value);
        speed = (isFinite(v) && v > 0) ? v : 1;  // 防御非法值
    }, _sig);

    // 精确查找当前战术板控制栏容器，避免多实例时 document.querySelectorAll 跨实例干扰
    // 优先：#tactical-controls → scope 传入容器 → canvas 父级链中最近包含 .tact-filter 的祖先 → document（保底）
    const controlsEl = $('tactical-controls') || scope || (() => {
        let el = canvas.parentElement;
        while (el && el !== document.body) {
            if (el.querySelector('.tact-filter')) return el;
            el = el.parentElement;
        }
        return document;
    })();
    // 收集本实例的 tact-filter 按钮，避免 controlsEl===document 时跨实例污染样式
    const tactFilterBtns = Array.from(controlsEl.querySelectorAll('.tact-filter'));
    tactFilterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filter = btn.dataset.filter;
            tactFilterBtns.forEach(b => {
                const isActive = b === btn;
                b.style.background = isActive ? getFilterBg(filter) : 'transparent';
                b.style.opacity    = isActive ? '1' : '0.7';
            });
            render();
        }, _sig);
    });

    function getFilterBg(f) {
        return { all: 'rgba(0,255,136,0.18)', team_a: 'rgba(255,120,40,0.2)', team_b: 'rgba(60,100,255,0.2)', ball: 'rgba(0,255,255,0.18)' }[f] || 'transparent';
    }

    // ── 绘制从头到当前进度的完整渐变轨迹（二分查找，避免大数据量时每帧 filter 全扫）──
    function drawFullTrail(tr, progress, color) {
        const cutF = globalMinF + progress * frameRange;
        const allPts = tr.pts;
        // 二分找到最后一个 f <= cutF 的索引
        let lo = 0, hi = allPts.length - 1, endIdx = -1;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (allPts[mid].f <= cutF) { endIdx = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        if (endIdx < 1) return;  // endIdx<1 即少于2个点，无法连线
        ctx.lineWidth   = 2.5;
        ctx.strokeStyle = color;
        // 最近 TRAIL_FADE 段做渐变，更早的固定为 0.05（保持路径可见但不抢镜）
        const TRAIL_FADE = 80;  // 3帧采样密度更高，适当增大渐变段数
        // 性能保护：历史远端（tailPos > TRAIL_FADE）alpha 固定 0.05，
        // 视觉与全量渲染基本一致，但每隔 SPARSE_STEP 画一段即可降低开销
        const SPARSE_STEP = Math.max(1, Math.floor((endIdx - TRAIL_FADE) / 300));
        for (let i = 1; i <= endIdx; i++) {
            const tailPos = endIdx - i;  // 距末尾的段数（最新段=0，最亮）
            // 历史远端稀疏渲染：跳过非关键段（SPARSE_STEP>1 时生效）
            if (tailPos > TRAIL_FADE && (i % SPARSE_STEP !== 0)) continue;
            const [wx1, wy1] = ptXY(allPts[i-1]);
            const [wx2, wy2] = ptXY(allPts[i]);
            if (!inBounds(wx1, wy1) || !inBounds(wx2, wy2)) continue;
            // 后处理已消除跳变，用宽松阈值（6m）仅过滤真正的离场瞬移
            if (Math.hypot(wx2 - wx1, wy2 - wy1) > 6.0) continue;
            const alpha = tailPos <= TRAIL_FADE ? (0.05 + 0.95 * (1 - tailPos / TRAIL_FADE)) : 0.05;
            ctx.globalAlpha = alpha;
            // Catmull-Rom → Bezier 控制点（平滑历史轨迹）
            const c1 = toCanvas(wx1, wy1);
            const c2 = toCanvas(wx2, wy2);
            const [wx0, wy0] = ptXY(allPts[Math.max(0, i - 2)]);
            const [wx3, wy3] = ptXY(allPts[Math.min(endIdx, i + 1)]);
            const c0 = inBounds(wx0, wy0) ? toCanvas(wx0, wy0) : c1;
            const c3 = inBounds(wx3, wy3) ? toCanvas(wx3, wy3) : c2;
            const cp1x = c1[0] + (c2[0] - c0[0]) / 6;
            const cp1y = c1[1] + (c2[1] - c0[1]) / 6;
            const cp2x = c2[0] - (c3[0] - c1[0]) / 6;
            const cp2y = c2[1] - (c3[1] - c1[1]) / 6;
            ctx.beginPath();
            ctx.moveTo(c1[0], c1[1]);
            ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, c2[0], c2[1]);
            ctx.stroke();
        }
        ctx.globalAlpha = 1;
    }

    // ── 绘制带光晕的选中圆点 ──
    function drawGlowDot(cx, cy, color) {
        ctx.save();
        try {
            ctx.shadowColor = color;
            ctx.shadowBlur  = 18;
            ctx.beginPath();
            ctx.arc(cx, cy, 8, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
        } finally {
            ctx.restore();
        }
    }

    // ── 体能数据面板 ──
    // 常量
    const SPRINT_THRESHOLD  = 5.5;   // m/s，FIFA冲刺阈值
    // 帧间跳变距离阈值：采样间隔 5帧@25fps=0.2s，球员极限冲刺 ~10m/s → 2m/0.2s
    // 设 3.0m 已极宽裕；旧值 12.0m 对应 60m/s 根本无法过滤 H矩阵噪声导致的虚假高速
    const STATS_JUMP_THRESH = coordSpace === 'pitch' ? 3.0 : 120;
    // 人类极限速度钳制：博尔特百米极速 12.42m/s (44.7km/h)
    // H矩阵微小抖动仍可能产生 >12.5m/s 的虚假速度，用硬上限兜底
    const MAX_HUMAN_SPEED   = 12.5;  // m/s
    // 冲刺最短持续时间：0.5秒（视频轨迹噪声比GPS大，1.0s门限过严导致全0）
    const SPRINT_MIN_DUR_SEC = 0.5;

    // 预计算缓存（选中球员切换时重建）
    let _statsTid     = null;
    let _statsCache   = null;  // { cumDist, speedArr, maxSpArr, sprintEnds }

    function buildStatsCache(tid) {
        if (_statsTid === tid && _statsCache) return _statsCache;
        const tr = tracks[tid];
        if (!tr || !Array.isArray(tr.pts) || tr.pts.length < 2) {
            _statsTid = tid; _statsCache = null; return null;
        }
        const pts = tr.pts;
        const n   = pts.length;
        const rawSpeed = new Float64Array(n);   // 原始帧间速度（未平滑）
        const cumDist  = new Float64Array(n);
        const speedArr = new Float64Array(n);   // 最终平滑 + 钳制后的速度
        const maxSpArr = new Float64Array(n);
        const dtArr    = new Float64Array(n);

        // 第一遍：计算原始帧间速度 + 跳变过滤 + 速度钳制
        for (let i = 1; i < n; i++) {
            const p0 = pts[i-1], p1 = pts[i];
            const dx = (p1.wx ?? p1.x ?? 0) - (p0.wx ?? p0.x ?? 0);
            const dy = (p1.wy ?? p1.y ?? 0) - (p0.wy ?? p0.y ?? 0);
            const dt = (p1.f - p0.f) / videoFps;
            const d  = Math.hypot(dx, dy);
            let v = (d < STATS_JUMP_THRESH && dt > 0) ? d / dt : 0;
            // 钳制到人类极限速度，消除 H矩阵微小抖动的残余噪声
            if (v > MAX_HUMAN_SPEED) v = 0;
            rawSpeed[i] = v;
            dtArr[i]    = dt;
            // 跳变点不累加距离；超速点也不累加（视为噪声位移）
            cumDist[i]  = cumDist[i-1] + (v > 0 ? Math.min(d, MAX_HUMAN_SPEED * dt) : 0);
        }

        // 第二遍：三点中值平滑，消除单帧 H矩阵尖刺噪声
        // 中值滤波保边性好：真实加速/减速不会被模糊，孤立尖刺被邻居替换
        speedArr[0] = rawSpeed[0];
        if (n > 1) speedArr[n-1] = rawSpeed[n-1];
        for (let i = 1; i < n - 1; i++) {
            const a = rawSpeed[i-1], b = rawSpeed[i], c = rawSpeed[i+1];
            // 三值取中值
            speedArr[i] = (a <= b)
                ? (b <= c ? b : (a <= c ? c : a))
                : (a <= c ? a : (b <= c ? c : b));
        }

        // 第三遍：5点滑动均值二次平滑，消除H矩阵抖动产生的速度锯齿（让冲刺段连续）
        const smWin = 2;
        for (let i = smWin; i < n - smWin; i++) {
            let sum = 0;
            for (let k = -smWin; k <= smWin; k++) sum += speedArr[i + k];
            speedArr[i] = sum / (smWin * 2 + 1);
        }

        // 第四遍：累计最高速度
        let runMax = 0;
        for (let i = 0; i < n; i++) {
            runMax = Math.max(runMax, speedArr[i]);
            maxSpArr[i] = runMax;
        }

        // 冲刺片段：用实际时间累计（而非点数）判断是否满足最短持续时间
        // 记录每段结束时的 pts 索引（供二分查找计冲刺次数）
        const sprintEnds = [];
        let inSprint = false, sprintDur = 0;
        for (let i = 1; i < n; i++) {
            if (speedArr[i] >= SPRINT_THRESHOLD) {
                if (!inSprint) { inSprint = true; sprintDur = 0; }
                sprintDur += dtArr[i];
            } else {
                if (inSprint) {
                    if (sprintDur >= SPRINT_MIN_DUR_SEC) sprintEnds.push(i - 1);
                    inSprint = false; sprintDur = 0;
                }
            }
        }
        if (inSprint && sprintDur >= SPRINT_MIN_DUR_SEC) sprintEnds.push(n - 1);

        _statsTid   = tid;
        _statsCache = { cumDist, speedArr, maxSpArr, sprintEnds, pts };
        return _statsCache;
    }

    // 速度分级：颜色 + 标签
    function speedGrade(v) {
        if (v < 1.5)  return { color: '#888888', label: '静止',   pct: Math.min(100, v / 1.5  * 15) };
        if (v < 3.0)  return { color: '#00cc66', label: '慢跑',   pct: 15 + (v - 1.5) / 1.5 * 20 };
        if (v < 5.5)  return { color: '#ffcc00', label: '快跑',   pct: 35 + (v - 3.0) / 2.5 * 30 };
        if (v < 7.5)  return { color: '#ff7700', label: '冲刺',   pct: 65 + (v - 5.5) / 2.0 * 25 };
        return               { color: '#ff3333', label: '极速冲刺', pct: Math.min(100, 90 + (v - 7.5) / 3 * 10) };
    }

    // ── 体能面板 DOM 引用：一次性缓存，避免每帧重复查询 ──
    const _sp = {
        panel  : document.getElementById('player-stats-panel'),
        noPlayer: document.getElementById('stats-no-player'),
        title  : document.getElementById('stats-player-title'),
        elapsed: document.getElementById('stats-elapsed'),
        bar    : document.getElementById('stats-speed-bar'),
        slbl   : document.getElementById('stats-speed-label'),
        is     : document.getElementById('stats-instant-speed'),
        ikh    : document.getElementById('stats-instant-kmh'),
        td     : document.getElementById('stats-total-dist'),
        tkm    : document.getElementById('stats-total-km'),
        ms     : document.getElementById('stats-max-speed'),
        mkh    : document.getElementById('stats-max-kmh'),
        sc     : document.getElementById('stats-sprint-count'),
        avg    : document.getElementById('stats-avg-speed'),
    };

    // 上一帧的显示值，用于跳过没有变化的 DOM 写入
    // panelState: null=初始, 'hidden'=隐藏, 'no-player'=已选单球员模式但未选人, 'player'=已选中球员
    let _spPrev = { iS: -1, tD: -1, mS: -1, sC: -1, eS: -1, barPct: -1, barColor: '', titleTid: null, noPlayerShown: null, panelState: null };

    function updateStatsPanel() {
        if (!_sp.panel) return;

        // 辅助：同步调整并排 grid 列数（速度板隐藏时战术板撑满全宽）
        const _tactRow = document.getElementById('tactical-speed-row');
        function _setRowLayout(showSpeed) {
            if (!_tactRow) return;
            _tactRow.style.gridTemplateColumns = showSpeed ? '1fr 1fr' : '1fr';
        }

        // 不在单球员模式：隐藏面板（脏检查）
        if (!isSingleMode) {
            if (_spPrev.panelState !== 'hidden') {
                _sp.panel.style.display = 'none';
                _setRowLayout(false);
                _spPrev.panelState = 'hidden';
            }
            return;
        }

        // 单球员模式：面板可见（脏检查）
        if (_spPrev.panelState === 'hidden' || _spPrev.panelState === null) {
            _sp.panel.style.display = 'block';
            _setRowLayout(true);
        }

        if (!selectedTid) {
            // 未选中球员（脏检查：仅状态切换时写 DOM）
            if (_spPrev.panelState !== 'no-player') {
                if (_sp.noPlayer) _sp.noPlayer.style.display = 'block';
                if (_sp.title) _sp.title.innerHTML = '<i class="fas fa-running"></i> 体能实时数据';
                [_sp.is,_sp.ikh,_sp.td,_sp.tkm,_sp.ms,_sp.mkh,_sp.sc,_sp.avg].forEach(el => { if (el) el.textContent = '—'; });
                if (_sp.bar)  { _sp.bar.style.width = '0%'; _sp.bar.style.background = '#888'; }
                if (_sp.slbl) _sp.slbl.textContent = '—';
                if (_sp.elapsed) _sp.elapsed.textContent = '00:00';
            }
            // 重置 prev cache
            _spPrev = { iS: -1, tD: -1, mS: -1, sC: -1, eS: -1, barPct: -1, barColor: '', titleTid: null, noPlayerShown: true, panelState: 'no-player' };
            return;
        }

        // 已选中球员（脏检查：noPlayer 仅从显示切为隐藏时写一次）
        if (_spPrev.noPlayerShown !== false) {
            if (_sp.noPlayer) _sp.noPlayer.style.display = 'none';
            _spPrev.noPlayerShown = false;
        }
        _spPrev.panelState = 'player';

        // 更新标题（仅在选中球员切换时重写 innerHTML，其余帧跳过）
        if (selectedTid !== _spPrev.titleTid) {
            const tr = tracks[selectedTid];
            const teamName  = tr?.team === 'team_a' ? '主队' : tr?.team === 'team_b' ? '客队' : '未知';
            const dotColor  = TEAM_COLOR[tr?.team] || '#ffdc00';
            const matched   = window._allAthletes?.find(a => String(a.track_id) === String(selectedTid));
            const playerLbl = matched?.jersey_number ? `#${matched.jersey_number} ${teamName}` : `${teamName}球员`;
            if (_sp.title) _sp.title.innerHTML = `<i class="fas fa-running" style="color:${dotColor}"></i> ${playerLbl}`;
            _spPrev.titleTid = selectedTid;
        }

        // 获取预计算缓存
        const cache = buildStatsCache(selectedTid);
        if (!cache) return;

        // 找当前帧对应的 pts 索引（二分）
        const cutF = globalMinF + currentProgress * frameRange;
        const pts  = cache.pts;
        let lo = 0, hi = pts.length - 1, curIdx = -1;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (pts[mid].f <= cutF) { curIdx = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        if (curIdx < 0) return;

        // O(1) 读取预计算数据
        const instantSpeed = cache.speedArr[curIdx];
        const totalDist    = cache.cumDist[curIdx];
        const maxSpeed     = cache.maxSpArr[curIdx];
        const elapsedSec   = (pts[curIdx].f - pts[0].f) / videoFps;
        const avgSpeed     = elapsedSec > 0.5 ? totalDist / elapsedSec : 0;
        // 冲刺次数：sprintEnds 中 <= curIdx 的数量（二分）
        const se = cache.sprintEnds;
        let sprintCount = 0;
        { let lo2=0, hi2=se.length-1;
          while (lo2<=hi2){ const m=(lo2+hi2)>>1; if(se[m]<=curIdx){sprintCount=m+1;lo2=m+1;}else hi2=m-1;} }

        // 速度条（每帧必更新——变化最频繁）
        const grade = speedGrade(instantSpeed);
        if (grade.pct !== _spPrev.barPct || grade.color !== _spPrev.barColor) {
            if (_sp.bar)  { _sp.bar.style.width = `${grade.pct.toFixed(1)}%`; _sp.bar.style.background = grade.color; }
            if (_sp.slbl) _sp.slbl.textContent = grade.label;
            _spPrev.barPct   = grade.pct;
            _spPrev.barColor = grade.color;
        }

        const fmt1 = v => v.toFixed(1);

        // 瞬时速度（每帧可能变化，直接更新）
        if (instantSpeed !== _spPrev.iS) {
            if (_sp.is)  _sp.is.textContent  = `${fmt1(instantSpeed)} m/s`;
            if (_sp.ikh) _sp.ikh.textContent = `${fmt1(instantSpeed * 3.6)} km/h`;
            _spPrev.iS = instantSpeed;
        }
        // 累计跑动（随时间单调递增，但相邻帧几乎不变，用整数化后比较）
        const tdRnd = Math.round(totalDist * 10);
        if (tdRnd !== _spPrev.tD) {
            if (_sp.td)  _sp.td.textContent  = totalDist >= 1000
                ? `${(totalDist/1000).toFixed(2)} km`
                : `${Math.round(totalDist)} m`;
            if (_sp.tkm) _sp.tkm.textContent = totalDist >= 1000
                ? `${Math.round(totalDist)} m 已跑`
                : `${(totalDist/1000).toFixed(3)} km`;
            _spPrev.tD = tdRnd;
        }
        // 最高速度（只增不减）
        if (maxSpeed !== _spPrev.mS) {
            if (_sp.ms)  _sp.ms.textContent  = `${fmt1(maxSpeed)} m/s`;
            if (_sp.mkh) _sp.mkh.textContent = `${fmt1(maxSpeed * 3.6)} km/h`;
            _spPrev.mS = maxSpeed;
        }
        // 冲刺次数（整数，变化很少）
        if (sprintCount !== _spPrev.sC) {
            if (_sp.sc) _sp.sc.textContent = `${sprintCount} 次`;
            _spPrev.sC = sprintCount;
        }
        // 平均速度 + 已用时间（每整秒更新一次，用 Math.floor 截断）
        const eFloor = Math.floor(elapsedSec);
        if (eFloor !== _spPrev.eS) {
            if (_sp.avg) _sp.avg.textContent = `${fmt1(avgSpeed)} m/s  (${fmt1(avgSpeed * 3.6)} km/h)`;
            if (_sp.elapsed) _sp.elapsed.textContent = fmtTime(elapsedSec);
            _spPrev.eS = eFloor;
        }
    }

    // ── 更新选中球员标签 ──
    function updateSelectedLabel() {
        if (!selectedLbl) return;
        if (!isSingleMode) { selectedLbl.style.display = 'none'; return; }
        selectedLbl.style.display = 'inline';
        if (!selectedTid) {
            selectedLbl.textContent = '▶ 未选中';
            selectedLbl.style.color = '#ffdc00';
            return;
        }
        const tr       = tracks[selectedTid];
        const teamName = tr?.team === 'team_a' ? '主队' : tr?.team === 'team_b' ? '客队' : (tr?.team || '未知');
        const dotColor = TEAM_COLOR[tr?.team] || '#aaa';
        const matched  = window._allAthletes?.find(
            a => String(a.track_id) === String(selectedTid)
        );
        const label = matched?.jersey_number
            ? `#${matched.jersey_number} ${teamName}`
            : `${teamName}球员`;
        selectedLbl.textContent = `▶ ${label}`;
        selectedLbl.style.color = dotColor;
    }

    // ── 单球员模式：Canvas 点击 hitTest ──
    canvas.addEventListener('click', (e) => {
        if (!isSingleMode) return;
        const canvasRect = canvas.getBoundingClientRect();
        const scaleX = CW / canvasRect.width;
        const scaleY = CH / canvasRect.height;
        const mx = (e.clientX - canvasRect.left) * scaleX;
        const my = (e.clientY - canvasRect.top)  * scaleY;

        let hit     = null;
        let minDist = Infinity;
        const HIT_R = 20;
        hitZones.forEach(z => {
            const d = Math.hypot(mx - z.cx, my - z.cy);
            if (d < HIT_R && d < minDist) { minDist = d; hit = z.tid; }
        });

        selectedTid = (hit === selectedTid) ? null : hit;
        // 切换球员时清除预计算缓存，触发重建
        _statsTid = null; _statsCache = null;
        updateSelectedLabel();
        render();
    }, _sig);

    // ── 单球员模式按钮 ──
    singleModeBtn?.addEventListener('click', () => {
        isSingleMode = !isSingleMode;
        selectedTid  = null;
        _statsTid = null; _statsCache = null;
        if (singleModeBtn) {
            singleModeBtn.style.background = isSingleMode ? 'rgba(255,220,0,0.18)' : 'transparent';
            singleModeBtn.style.color      = isSingleMode ? '#ffdc00' : 'rgba(255,220,0,0.7)';
        }
        canvas.style.cursor = isSingleMode ? 'pointer' : 'default';
        updateSelectedLabel();
        render();
    }, _sig);

    // 初始渲染第一帧
    render();
}

// =====================================================================
//  球员号码标注弹窗 —— 独立模块
// =====================================================================
(function () {
    let _videoId    = null;
    let _frameUrl   = null;
    let _detections = [];
    let _frameW     = 1280;
    let _frameH     = 720;
    let _activeDet  = null;
    let _img        = null;
    let _teamColors = { team_a: [128,128,128], team_b: [128,128,128] };
    let _homeTeam   = null;
    // ── 多帧状态 ──
    let _allFrames         = [];   // 后端返回的所有帧数据
    let _curFrameIdx       = 0;    // 当前帧下标
    let _globalAnnotations = {};   // 跨帧持久化标注 {posKey: {jersey,team,x,y}}
    let _frameDetections   = {};   // {frameIdx: detections[]}
    const API       = window.location.origin;

    // ── roundRect 兼容 polyfill ──
    if (!CanvasRenderingContext2D.prototype.roundRect) {
        CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
            r = Math.min(r, w / 2, h / 2);
            this.moveTo(x + r, y);
            this.arcTo(x + w, y,     x + w, y + h, r);
            this.arcTo(x + w, y + h, x,     y + h, r);
            this.arcTo(x,     y + h, x,     y,     r);
            this.arcTo(x,     y,     x + w, y,     r);
            this.closePath();
        };
    }

    window.showJerseyModal = function (videoId) {
        _videoId    = videoId;
        _detections = [];
        _activeDet  = null;
        _frameUrl   = null;
        _img        = null;
        _teamColors = { team_a: [128,128,128], team_b: [128,128,128] };
        _homeTeam   = null;

        // ── 多帧状态 ──
        _allFrames      = [];   // 后端返回的全部帧数据
        _curFrameIdx    = 0;    // 当前显示的帧下标
        _globalAnnotations = {}; // jerseyNumber → {team} 跨帧持久化标注
        _frameDetections = {};  // frameIdx → detections[]（每帧独立检测框，含 jersey 状态）

        const modal = document.getElementById('jersey-modal');
        modal.style.display = 'block';
        document.body.style.overflow = 'hidden';

        // 确保第一步可见，第二步隐藏
        const step1Footer = document.getElementById('jersey-step1-footer');
        const step2Panel  = document.getElementById('jersey-team-select-panel');
        if (step1Footer) step1Footer.style.display = 'flex';
        if (step2Panel)  step2Panel.style.display  = 'none';

        document.getElementById('jersey-frame-loading').style.display = 'flex';
        document.getElementById('jersey-loading-text').textContent = '正在提取关键帧...';
        document.getElementById('jersey-modal-subtitle').textContent = 'AI 正在定位关键帧，请稍候...';

        _fetchKeyFrame(videoId);
    };

    window.closeJerseyModal = function () {
        document.getElementById('jersey-modal').style.display = 'none';
        document.body.style.overflow = '';
        _hideBubble();
    };

    // skipJerseyModal 已移除：标注是必须流程，不允许跳过

    // 第一步完成 → 进入第二步（选择主队颜色）
    window.showTeamSelectPanel = function () {
        const step1Footer = document.getElementById('jersey-step1-footer');
        const step2Panel  = document.getElementById('jersey-team-select-panel');
        if (step1Footer) step1Footer.style.display = 'none';
        if (step2Panel)  step2Panel.style.display  = 'block';
        _hideBubble();

        // 渲染两队颜色色块
        const swatchA = document.getElementById('team-color-swatch-a');
        const swatchB = document.getElementById('team-color-swatch-b');
        if (swatchA) {
            const [r, g, b] = _teamColors.team_a;
            swatchA.style.background = `rgb(${r},${g},${b})`;
        }
        if (swatchB) {
            const [r, g, b] = _teamColors.team_b;
            swatchB.style.background = `rgb(${r},${g},${b})`;
        }

        // 重置选中状态
        _homeTeam = null;
        _resetTeamSelectUI();
    };

    // 返回第一步
    window.backToStep1 = function () {
        const step1Footer = document.getElementById('jersey-step1-footer');
        const step2Panel  = document.getElementById('jersey-team-select-panel');
        if (step1Footer) step1Footer.style.display = 'flex';
        if (step2Panel)  step2Panel.style.display  = 'none';
    };

    // 用户点击选择主队
    window.selectHomeTeam = function (team) {
        _homeTeam = team;
        _resetTeamSelectUI();

        // 高亮选中的色块
        const selId = team === 'team_a' ? 'team-select-a' : 'team-select-b';
        const selEl = document.getElementById(selId);
        if (selEl) {
            selEl.style.borderColor = 'rgba(255,130,40,0.8)';
            selEl.style.background  = 'rgba(255,130,40,0.1)';
            selEl.classList.add('home-selected');
            const badge = selEl.querySelector('.home-badge');
            if (badge) badge.style.display = 'block';
        }

        // 解锁提交按钮
        const btn = document.getElementById('jersey-submit-btn');
        if (btn) {
            btn.disabled         = false;
            btn.style.background = 'linear-gradient(135deg,rgba(0,255,136,0.22),rgba(0,212,255,0.15))';
            btn.style.border     = '1px solid rgba(0,255,136,0.4)';
            btn.style.color      = '#00ff88';
            btn.style.cursor     = 'pointer';
        }
    };

    function _resetTeamSelectUI() {
        ['team-select-a', 'team-select-b'].forEach(id => {
            const el = document.getElementById(id);
            if (!el) return;
            el.classList.remove('home-selected');
            el.style.borderColor = 'rgba(255,255,255,0.12)';
            el.style.background  = 'rgba(255,255,255,0.04)';
            const badge = el.querySelector('.home-badge');
            if (badge) badge.style.display = 'none';
        });
    }

    // 第二步确认提交
    window.confirmJerseyModal = function () {
        if (!_homeTeam) return;   // 未选主队时不提交
        const teamSwap = (_homeTeam === 'team_b');

        // ── 汇总所有帧的标注 ──
        _saveCurrentFrameAnnotations();

        const jerseyMap = {};           // jerseyNumber → {jersey_number, team}  (旧方案兼容)
        const nonPlayerList = [];       // 旧方案兼容
        const trackJerseyMap = {};      // track_id → {jersey, team}  (v2 精确方案)
        const excludedTrackIds = [];    // 排除的 track_id 列表（视频渲染用：non_player + 未标注）
        const nonPlayerTrackIds = [];   // 仅明确标为非球员的 track_id（热力图/战术板排除用）
        const seenExcluded = new Set();
        const seenNonPlayer = new Set();

        // 第一遍：先收集所有已标注号码的 track_id（以此为准，不能被 excluded 覆盖）
        // 注意：允许同一号码对应多个 track_id（用于手动连接追踪断裂的同一球员）
        const annotatedTrackIds = new Set();
        Object.values(_frameDetections).forEach(dets => {
            (dets || []).forEach(d => {
                if (d.team !== 'non_player' && d.jersey != null && d.track_id) {
                    annotatedTrackIds.add(d.track_id);
                    trackJerseyMap[d.track_id] = { jersey: d.jersey, team: d.team };
                    // jerseyMap 以 track_id 为 key，允许同号码多个 track（手动 ID 连接）
                    jerseyMap[d.track_id] = { det_id: d.id, jersey_number: d.jersey, team: d.team };
                }
            });
        });

        // 第二遍：收集非球员 & 未标注（只要不在已标注集合里才能进 excluded）
        // 关键修复：同一 track_id 在某帧已标注、在另一帧未标注是正常现象（用户只在关键帧标注），
        //           以"标注过"为准，不能因为在其他帧未标注就把它加入 excludedTrackIds。
        Object.values(_frameDetections).forEach(dets => {
            (dets || []).forEach(d => {
                if (d.team === 'non_player') {
                    nonPlayerList.push({ det_id: d.id, x: d.x, y: d.y, w: d.w, h: d.h });
                    if (d.track_id && !annotatedTrackIds.has(d.track_id) && !seenExcluded.has(d.track_id)) {
                        excludedTrackIds.push(d.track_id);
                        seenExcluded.add(d.track_id);
                    }
                    if (d.track_id && !seenNonPlayer.has(d.track_id)) {
                        nonPlayerTrackIds.push(d.track_id);
                        seenNonPlayer.add(d.track_id);
                    }
                } else if (d.jersey == null) {
                    // 未标注号码的球员：已标注过的 track 不再加入 excluded
                    nonPlayerList.push({ det_id: d.id, x: d.x, y: d.y, w: d.w, h: d.h });
                    if (d.track_id && !annotatedTrackIds.has(d.track_id) && !seenExcluded.has(d.track_id)) {
                        excludedTrackIds.push(d.track_id);
                        seenExcluded.add(d.track_id);
                    }
                }
            });
        });

        const annotations = Object.values(jerseyMap);

        // ── 人数合规校验：按唯一号码+队伍组合计数（同号码多 track 视为同一球员）──
        const uniqueA = new Set(Object.values(trackJerseyMap).filter(v => v.team === 'team_a').map(v => v.jersey));
        const uniqueB = new Set(Object.values(trackJerseyMap).filter(v => v.team === 'team_b').map(v => v.jersey));
        const countA = uniqueA.size;
        const countB = uniqueB.size;
        const MAX_TEAM = 11;
        if (countA > MAX_TEAM || countB > MAX_TEAM) {
            const overA = countA > MAX_TEAM ? `主队(橙色)已标 ${countA} 人` : '';
            const overB = countB > MAX_TEAM ? `客队(蓝色)已标 ${countB} 人` : '';
            const overMsg = [overA, overB].filter(Boolean).join('、');
            const confirmed = window.confirm(
                `⚠️ 标准足球每队上场 11 人，当前 ${overMsg}，超出部分将被系统自动截断，只保留出现最多的前11名球员。\n\n建议返回修正后再提交，是否继续？`
            );
            if (!confirmed) return;
        }

        window.closeJerseyModal();
        window._triggerAnalysis(_videoId, annotations, teamSwap, nonPlayerList, trackJerseyMap, excludedTrackIds, nonPlayerTrackIds);
    };

    window.resetJerseyLabels = function () {
        // 清除全部帧的标注状态
        Object.values(_frameDetections).forEach(dets => {
            (dets || []).forEach((d, i) => {
                d.jersey = null;
                if (d.team === 'non_player') {
                    d.team = d._prevTeam || (i % 2 === 0 ? 'team_a' : 'team_b');
                }
            });
        });
        // 清除跨帧全局标注缓存，防止回填复现
        _globalAnnotations = {};
        // 重新加载当前帧
        _detections = _frameDetections[_curFrameIdx] || [];
        _hideBubble();
        _activeDet = null;
        _render();
        _updateProgress();
    };

    window.cancelJerseyInput = function () { _hideBubble(); _activeDet = null; _render(); };

    // ── 辅助：将 _detections 中的修改同步回 _frameDetections[_curFrameIdx] ──
    function _syncDetToFrameStore(det) {
        const stored = (_frameDetections[_curFrameIdx] || []).find(d => d.id === det.id);
        if (stored) {
            stored.team      = det.team;
            stored.jersey    = det.jersey;
            stored._prevTeam = det._prevTeam;
        }
    }

    // ── 标记/撤销非球员（通过 track_id 跨帧同步） ──
    window.markAsNonPlayer = function () {
        if (_activeDet === null) return;
        const det = _detections.find(d => d.id === _activeDet);
        if (!det) return;

        const isUndoing = (det.team === 'non_player');
        const tid = det.track_id || null;

        if (isUndoing) {
            det.team      = det._prevTeam || 'team_a';
            det.jersey    = null;
            det._autoFilled = false; // 撤销时清除自动回填标志，确保 _saveCurrentFrameAnnotations 能保存
        } else {
            det._prevTeam   = det.team;
            det.team        = 'non_player';
            det.jersey      = null;
            det._autoFilled = false; // 用户主动标注，不应是自动回填
        }

        // 同步回当前帧的 _frameDetections 原始存储
        _syncDetToFrameStore(det);

        // 通过 track_id 同步到其他帧的同一追踪目标
        if (tid) {
            Object.keys(_frameDetections).forEach(fi => {
                if (parseInt(fi) === _curFrameIdx) return;
                (_frameDetections[fi] || []).forEach(fd => {
                    if (fd.track_id !== tid) return;
                    if (isUndoing) {
                        // 撤销：恢复所有帧中同 track_id 的非球员标记
                        if (fd.team === 'non_player') {
                            fd.team       = fd._prevTeam || 'team_a';
                            fd.jersey     = null;
                            fd._autoFilled = false;
                        }
                    } else {
                        // 标记：将所有帧中同 track_id 设为非球员
                        if (fd.team !== 'non_player') {
                            fd._prevTeam  = fd.team;
                            fd.team       = 'non_player';
                            fd.jersey     = null;
                            fd._autoFilled = false;
                        }
                    }
                });
            });

            // 同步清理/添加全局标注中同 track_id 的非球员记录
            if (isUndoing) {
                // 撤销：删除 _globalAnnotations 中所有属于该 track_id 的非球员记录
                Object.keys(_globalAnnotations).forEach(key => {
                    const ann = _globalAnnotations[key];
                    if (ann.track_id === tid && ann.isNonPlayer) {
                        delete _globalAnnotations[key];
                    }
                });
            } else {
                // 标记：在 _globalAnnotations 中写入/更新该 track_id 的非球员标注
                // 使用当前帧的 det 信息作为来源记录
                const npKey = `np_tid_${tid}`;
                _globalAnnotations[npKey] = {
                    jersey: null, team: 'non_player', isNonPlayer: true,
                    track_id: tid,
                    x: det.x, y: det.y, w: det.w, h: det.h,
                    fromFrame: _curFrameIdx,
                };
            }
        }

        _hideBubble();
        _activeDet = null;
        _render();
        _updateProgress();
    };

    window.confirmJerseyInput = function () {
        const val = parseInt(document.getElementById('jersey-number-input').value);
        if (_activeDet !== null && val >= 1 && val <= 99) {
            const det = _detections.find(d => d.id === _activeDet);
            if (det) {
                det.jersey = val;
                _syncDetToFrameStore(det);  // 同步回原始存储
            }
        }
        _hideBubble();
        _activeDet = null;
        _render();
        _updateProgress();
    };

    async function _fetchKeyFrame(videoId) {
        try {
            // ── Phase A：先触发全局追踪，然后轮询直到完成 ──
            document.getElementById('jersey-loading-text').textContent = '正在进行全局追踪分析，请稍候...';

            // 1. 启动 Phase A
            const preTrackRes = await fetch(`${API}/pre_track/${videoId}`, { method: 'POST' });
            const preTrackData = await preTrackRes.json();

            // 2. 轮询 Phase A 进度（除非已缓存）
            if (preTrackData.status !== 'cached') {
                let phaseADone = false;
                // 最长等待 95 分钟（1小时视频 Phase A 约需 50-80 分钟）
                const POLL_TIMEOUT_MS = 95 * 60 * 1000;
                const pollStart = Date.now();
                let consecutiveErrors = 0;

                while (!phaseADone) {
                    await new Promise(r => setTimeout(r, 3000));

                    // 超时保护
                    if (Date.now() - pollStart > POLL_TIMEOUT_MS) {
                        throw new Error('Phase A 超时（超过95分钟），请检查服务器日志');
                    }

                    let progData;
                    try {
                        const progRes = await fetch(`${API}/pre_track_progress/${videoId}`);
                        if (!progRes.ok) {
                            consecutiveErrors++;
                            if (consecutiveErrors >= 5) throw new Error('服务器连续5次无响应，可能已重启');
                            continue;
                        }
                        progData = await progRes.json();
                        consecutiveErrors = 0;
                    } catch (fetchErr) {
                        consecutiveErrors++;
                        if (consecutiveErrors >= 5) throw fetchErr;
                        document.getElementById('jersey-loading-text').textContent =
                            `连接中断，重试中 (${consecutiveErrors}/5)...`;
                        await new Promise(r => setTimeout(r, 5000));
                        continue;
                    }

                    const p = Number(progData.progress) || 0;
                    const elapsed = Math.round((Date.now() - pollStart) / 1000 / 60);
                    document.getElementById('jersey-loading-text').textContent =
                        `全局追踪中... ${Math.max(0, Math.round(p))}%（已用时 ${elapsed} 分钟）`;

                    // cached=true 是最可靠的完成信号（pkl 写入后才设置）
                    if (progData.cached) {
                        phaseADone = true;
                    } else if (p >= 95) {
                        // 再额外等 2 秒确认 pkl 已写完
                        await new Promise(r => setTimeout(r, 2000));
                        phaseADone = true;
                    } else if (p < 0) {
                        // 进度为 -1 可能是服务器重启导致内存丢失，再查一次 cached 状态
                        // 后端已修复：若 pkl 存在会将 progress 修正为 95，此处仍做保底检查
                        throw new Error('Phase A 追踪分析失败（progress=-1），请重新上传或检查服务器日志');
                    }
                }
            }

            document.getElementById('jersey-loading-text').textContent = '正在选取标注帧...';

            // 3. 调用 v2 选帧接口
            const res  = await fetch(`${API}/quick_detect_v2/${videoId}`);
            if (!res.ok) {
                let detail = `服务器响应异常 ${res.status}`;
                try {
                    const errJson = await res.json();
                    if (errJson.detail) detail = errJson.detail;
                } catch(_) {}
                throw new Error(detail);
            }
            const data = await res.json();

            // ── 多帧支持 ──
            _allFrames = data.frames && data.frames.length > 0
                ? data.frames
                : [{ frame_url: data.frame_url, frame_time: data.frame_time,
                     player_count: (data.detections||[]).length,
                     detections: data.detections || [], team_colors: data.team_colors }];

            _frameW     = data.frame_w  || 1280;
            _frameH     = data.frame_h  || 720;
            _teamColors = data.team_colors || { team_a: [128,128,128], team_b: [128,128,128] };

            // 初始化每帧检测框（加入 jersey/id/track_id 字段）
            // 当 Phase A 颜色聚类失败时 d.team 可能为 'unknown'，
            // 此时用 A/B 交替分配作为初始 team，确保提交时 trackJerseyMap 里有有效 team
            _allFrames.forEach((fd, fi) => {
                _frameDetections[fi] = (fd.detections || []).map((d, i) => ({
                    id    : fi * 1000 + i,
                    x: d.x, y: d.y, w: d.w, h: d.h,
                    team  : (d.team && d.team !== 'unknown') ? d.team : (i % 2 === 0 ? 'team_a' : 'team_b'),
                    jersey: null,
                    track_id: d.track_id || null,
                    _fromFrame: fi,
                }));
            });

            _curFrameIdx = 0;
            _switchToFrame(0);

        } catch (e) {
            console.warn('[JerseyModal] 关键帧提取失败:', e.message);
            _showLoadError('关键帧提取失败，请稍后重试或刷新页面');
        }
    }

    // ── 切换到指定帧 ─────────────────────────────────────────────────────
    function _switchToFrame(fi) {
        if (!_allFrames || fi < 0 || fi >= _allFrames.length) return;

        // 保存当前帧的标注到全局
        _saveCurrentFrameAnnotations();

        _curFrameIdx = fi;
        const fd = _allFrames[fi];
        _frameUrl   = fd.frame_url;
        _teamColors = fd.team_colors || _teamColors;
        // 每帧用自己的实际尺寸，确保检测框坐标系与帧图一致
        if (fd.frame_w) _frameW = fd.frame_w;
        if (fd.frame_h) _frameH = fd.frame_h;

            // 恢复该帧检测框（深拷贝，防止回填结果污染 _frameDetections 的原始存储）
        _detections = (_frameDetections[fi] || []).map(d => Object.assign({}, d));
        _applyGlobalAnnotations(_detections, fi);

        // 更新导航 UI
        _updateFrameNav();

        // 加载帧图片
        _activeDet = null;
        _hideBubble();
        document.getElementById('jersey-frame-loading').style.display = 'flex';
        document.getElementById('jersey-loading-text').textContent = '正在加载关键帧...';

        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            _img = img;
            document.getElementById('jersey-frame-loading').style.display = 'none';
            const totalAnnotated = _countGlobalAnnotations();
            const totalPlayers   = _allFrames.reduce((s, f) => s + (f.player_count || 0), 0);
            document.getElementById('jersey-modal-subtitle').textContent =
                `帧 ${fi + 1}/${_allFrames.length}｜第 ${fd.frame_time || '--'} 秒｜本帧 ${_detections.length} 人｜共已标注 ${totalAnnotated} 人`;
            // 更新图例颜色
            const dotA = document.getElementById('legend-dot-a');
            const dotB = document.getElementById('legend-dot-b');
            if (dotA) { const [r,g,b] = _teamColors.team_a; dotA.style.background = `rgba(${r},${g},${b},0.6)`; dotA.style.borderColor = `rgb(${r},${g},${b})`; }
            if (dotB) { const [r,g,b] = _teamColors.team_b; dotB.style.background = `rgba(${r},${g},${b},0.6)`; dotB.style.borderColor = `rgb(${r},${g},${b})`; }
            _render();
            _updateProgress();
        };
        img.onerror = () => {
            // 当前帧图加载失败，自动跳到下一帧；全部帧都失败才报错
            if (_curFrameIdx + 1 < _allFrames.length) {
                console.warn(`[JerseyModal] 帧 ${_curFrameIdx} 图片加载失败，自动跳到下一帧`);
                _switchToFrame(_curFrameIdx + 1);
            } else {
                _showLoadError('帧图加载失败，请重新触发分析');
            }
        };
        img.src = `${API}${_frameUrl}?_t=${Date.now()}`;
    }

    // 保存当前帧已标注号码到全局（含 track_id、isNonPlayer）
    function _saveCurrentFrameAnnotations() {
        if (!_detections || _detections.length === 0) return;
        _detections.forEach(d => {
            if (d._autoFilled) {
                // 自动回填的框不写入全局标注；
                // 但若该框是回填的非球员且 track_id 已被用户撤销，
                // 需清除 _globalAnnotations 中残留的同 track_id 非球员记录
                // （由之前帧保存的 fX_id key）
                if (d.track_id) {
                    const npKey = `np_tid_${d.track_id}`;
                    if (!_globalAnnotations[npKey]) {
                        // np_tid_ key 已不存在，说明已撤销，清理同 tid 的所有 fX_id 非球员记录
                        Object.keys(_globalAnnotations).forEach(k => {
                            const a = _globalAnnotations[k];
                            if (a && a.track_id === d.track_id && a.isNonPlayer) {
                                delete _globalAnnotations[k];
                            }
                        });
                    }
                }
                return;
            }

            const key = `f${_curFrameIdx}_${d.id}`;

            if (d.team === 'non_player') {
                // 非球员标记也存入全局（用于跨帧同步）
                _globalAnnotations[key] = {
                    jersey: null, team: 'non_player', isNonPlayer: true,
                    track_id: d.track_id || null,
                    x: d.x, y: d.y, w: d.w, h: d.h,
                    fromFrame: _curFrameIdx
                };
            } else {
                // 普通球员框（含撤销后恢复的）：删除该框曾经留下的非球员记录
                delete _globalAnnotations[key];
                if (d.jersey != null) {
                    _globalAnnotations[key] = {
                        jersey: d.jersey, team: d.team, isNonPlayer: false,
                        track_id: d.track_id || null,
                        x: d.x, y: d.y, w: d.w, h: d.h,
                        fromFrame: _curFrameIdx
                    };
                }
            }
        });
        // 将用户主动标注的框同步回 _frameDetections（仅保存主动标注，不保存自动回填结果）
        if (_curFrameIdx !== undefined && (_frameDetections[_curFrameIdx] || []).length > 0) {
            _frameDetections[_curFrameIdx] = _detections.map(d => {
                if (d._autoFilled) {
                    const orig = Object.assign({}, d);
                    orig.jersey = null;
                    if (orig.team === 'non_player') orig.team = orig._prevTeam || 'team_a';
                    orig._autoFilled = false;
                    return orig;
                }
                return Object.assign({}, d);
            });
        }
    }

    // 跨帧一对一回填：优先使用 track_id 精确匹配，降级到坐标距离匹配
    function _applyGlobalAnnotations(detections, excludeFrame) {
        // 只取来自其他帧的标注
        const anns = Object.values(_globalAnnotations).filter(a => a.fromFrame !== excludeFrame);
        if (anns.length === 0) return;

        const usedDetIds  = new Set(); // 已匹配的检测框 id
        const usedTidAnns = new Set(); // 已匹配的 ann.track_id（防止同一 ann 重复匹配多框）

        // ── 第一轮：track_id 精确匹配（球员号码回填） ──
        const unset = detections.filter(d => d.jersey == null && d.team !== 'non_player');
        for (const d of unset) {
            if (!d.track_id) continue;
            for (const ann of anns) {
                if (!ann.track_id || ann.isNonPlayer) continue;
                if (usedDetIds.has(d.id) || usedTidAnns.has(ann.track_id)) continue;
                if (d.track_id === ann.track_id) {
                    d.jersey      = ann.jersey;
                    d.team        = ann.team;
                    d._autoFilled = true;
                    usedDetIds.add(d.id);
                    usedTidAnns.add(ann.track_id);
                    break;
                }
            }
        }

        // ── 第二轮：track_id 同步非球员标记 ──
        for (const d of detections) {
            if (!d.track_id || usedDetIds.has(d.id)) continue;
            for (const ann of anns) {
                if (!ann.track_id || !ann.isNonPlayer) continue;
                if (ann.track_id === d.track_id) {
                    d._prevTeam   = d.team;
                    d.team        = 'non_player';
                    d.jersey      = null;
                    d._autoFilled = true;
                    usedDetIds.add(d.id);
                    break;
                }
            }
        }

        // ── 第三轮：坐标距离降级匹配（无 track_id 的球员框） ──
        const stillUnset = detections.filter(d =>
            d.jersey == null && d.team !== 'non_player' && !usedDetIds.has(d.id));
        // 只取未通过 track_id 匹配过的、且非非球员的 ann
        const coordAnns = anns.filter(a =>
            !a.isNonPlayer && (!a.track_id || !usedTidAnns.has(a.track_id)));
        if (stillUnset.length === 0 || coordAnns.length === 0) return;

        const CROSS_FRAME_TOL = 80;
        const pairs = [];
        for (const d of stillUnset) {
            const cx = d.x + d.w / 2;
            const cy = d.y + d.h / 2;
            for (const ann of coordAnns) {
                const annCx = ann.x + ann.w / 2;
                const annCy = ann.y + ann.h / 2;
                const dx = Math.abs(annCx - cx);
                const dy = Math.abs(annCy - cy);
                if (dx < CROSS_FRAME_TOL && dy < CROSS_FRAME_TOL) {
                    pairs.push({ d, ann, dist: dx * dx + dy * dy });
                }
            }
        }

        // 按距离排序，贪心一对一匹配
        const usedCoordAnns = new Set();
        pairs.sort((a, b) => a.dist - b.dist);
        for (const { d, ann } of pairs) {
            if (usedDetIds.has(d.id)) continue;
            const annId = `${ann.fromFrame}_${ann.jersey}_${ann.team}`;
            if (usedCoordAnns.has(annId)) continue;
            d.jersey      = ann.jersey;
            d.team        = ann.team;
            d._autoFilled = true;
            usedDetIds.add(d.id);
            usedCoordAnns.add(annId);
        }
    }

    // 计算全局已标注总人数（去重：按 jersey+team 唯一，仅统计球员，不含非球员）
    function _countGlobalAnnotations() {
        const seen = new Set();
        Object.values(_globalAnnotations).forEach(a => {
            if (!a.isNonPlayer && a.jersey != null) seen.add(`${a.jersey}_${a.team}`);
        });
        return seen.size;
    }

    // 更新帧导航栏 UI
    function _updateFrameNav() {
        const nav = document.getElementById('jersey-frame-nav');
        if (!nav || !_allFrames) return;
        if (_allFrames.length <= 1) { nav.style.display = 'none'; return; }
        nav.style.display = 'flex';
        document.getElementById('jersey-frame-prev').disabled = (_curFrameIdx === 0);
        document.getElementById('jersey-frame-next').disabled = (_curFrameIdx >= _allFrames.length - 1);
        document.getElementById('jersey-frame-indicator').textContent =
            `帧 ${_curFrameIdx + 1} / ${_allFrames.length}`;
        // 帧点指示器
        const dots = document.getElementById('jersey-frame-dots');
        if (dots) {
            dots.innerHTML = _allFrames.map((f, i) => {
                const annotated = (_frameDetections[i] || []).filter(d => d.jersey != null).length;
                const active = i === _curFrameIdx;
                return `<span onclick="jerseyGoToFrame(${i})" title="帧${i+1}｜${f.player_count||0}人${annotated?'｜已标'+annotated:''}"
                    style="display:inline-block;width:${active?'22px':'10px'};height:10px;border-radius:5px;cursor:pointer;
                    background:${active?'#00ff88':annotated?'rgba(0,255,136,0.4)':'rgba(255,255,255,0.2)'};
                    border:1.5px solid ${active?'#00ff88':annotated?'rgba(0,255,136,0.6)':'rgba(255,255,255,0.2)'};
                    transition:all 0.2s;margin:0 2px;"></span>`;
            }).join('');
        }
    }

    // 帧导航：上一帧 / 下一帧 / 跳转（_switchToFrame 内部已调用 _saveCurrentFrameAnnotations，不重复调用）
    window.jerseyPrevFrame = function () { _switchToFrame(_curFrameIdx - 1); };
    window.jerseyNextFrame = function () { _switchToFrame(_curFrameIdx + 1); };
    window.jerseyGoToFrame = function (fi) { _switchToFrame(fi); };

    function _showLoadError(msg) {
        const loadingEl = document.getElementById('jersey-frame-loading');
        const textEl    = document.getElementById('jersey-loading-text');
        if (textEl) textEl.textContent = msg;
        if (loadingEl) {
            const spin = loadingEl.querySelector('div');
            if (spin) spin.style.display = 'none';
        }
    }

    function _render() {
        const canvas = document.getElementById('jersey-canvas');
        if (!canvas || !_img) return;
        canvas.width  = _frameW;
        canvas.height = _frameH;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(_img, 0, 0, _frameW, _frameH);

        // 预计算本帧中每个号码被几个 track 使用（>1 表示用户手动连接了断裂轨迹）
        const jerseyUsageCount = {};
        _detections.forEach(d => {
            if (d.jersey != null && d.team !== 'non_player') {
                jerseyUsageCount[d.jersey] = (jerseyUsageCount[d.jersey] || 0) + 1;
            }
        });

        _detections.forEach(d => {
            const isActive    = d.id === _activeDet;
            const isNonPlayer = d.team === 'non_player';
            const isLabeled   = d.jersey != null && !isNonPlayer;
            // 同号码多 track：用户手动连接了断裂轨迹
            const isLinked    = isLabeled && (jerseyUsageCount[d.jersey] || 0) > 1;
            const tc          = _teamColors[d.team] || [128, 128, 128];
            const [tr, tg, tb] = tc;

            let fillColor, strokeColor, labelColor;
            if (isNonPlayer) {
                fillColor   = 'rgba(255,180,0,0.18)';
                strokeColor = isActive ? '#fff' : '#ffb400';
                labelColor  = '#ffb400';
            } else if (isLinked) {
                // 橙黄色：表示该 track 与同号码其他 track 手动连接
                fillColor   = 'rgba(255,200,50,0.28)'; strokeColor = isActive ? '#fff' : '#ffc832';
                labelColor  = '#ffc832';
            } else if (isLabeled) {
                fillColor   = 'rgba(0,255,136,0.22)'; strokeColor = isActive ? '#fff' : '#00ff88';
                labelColor  = '#00ff88';
            } else {
                fillColor   = `rgba(${tr},${tg},${tb},0.22)`;
                strokeColor = isActive ? '#fff' : `rgb(${tr},${tg},${tb})`;
                labelColor  = `rgb(${tr},${tg},${tb})`;
            }

            ctx.save();
            if (isActive) { ctx.shadowColor = strokeColor; ctx.shadowBlur = 14; }
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth   = isActive ? 3 : 2;
            ctx.fillStyle   = fillColor;
            ctx.beginPath();
            ctx.roundRect(d.x, d.y, d.w, d.h, 4);
            ctx.fill(); ctx.stroke();
            ctx.shadowBlur = 0;

            // 非球员画叉
            if (isNonPlayer) {
                ctx.strokeStyle = labelColor;
                ctx.lineWidth   = 2;
                ctx.globalAlpha = 0.7;
                ctx.beginPath(); ctx.moveTo(d.x + 4, d.y + 4); ctx.lineTo(d.x + d.w - 4, d.y + d.h - 4); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(d.x + d.w - 4, d.y + 4); ctx.lineTo(d.x + 4, d.y + d.h - 4); ctx.stroke();
                ctx.globalAlpha = 1.0;
            }

            const label    = isNonPlayer ? '×' : (isLabeled ? (isLinked ? `#${d.jersey}⚡` : `#${d.jersey}`) : '?');
            const fontSize = Math.max(13, d.h * 0.28);
            ctx.font         = `bold ${fontSize}px sans-serif`;
            ctx.fillStyle    = isNonPlayer ? 'rgba(80,40,0,0.6)' : (isLabeled ? 'rgba(0,0,0,0.55)' : 'rgba(0,0,0,0.45)');
            ctx.textAlign    = 'center';
            ctx.textBaseline = 'middle';
            const cx = d.x + d.w / 2, cy = d.y + d.h / 2;
            const tw = ctx.measureText(label).width + 10;
            const th = fontSize + 6;
            ctx.beginPath();
            ctx.roundRect(cx - tw / 2, cy - th / 2, tw, th, th / 2);
            ctx.fill();
            ctx.fillStyle = labelColor;
            ctx.fillText(label, cx, cy);
            ctx.restore();
        });
    }

    // Canvas 点击事件（需要在弹窗模块中绑定，因为 app.js DOMContentLoaded 时 canvas 已在 DOM 中）
    document.addEventListener('click', function (e) {
        const canvas = document.getElementById('jersey-canvas');
        if (!canvas || e.target !== canvas) return;
        const rect   = canvas.getBoundingClientRect();
        const scaleX = _frameW / rect.width;
        const scaleY = _frameH / rect.height;
        const mx     = (e.clientX - rect.left) * scaleX;
        const my     = (e.clientY - rect.top)  * scaleY;

        let hit = null;
        for (let i = _detections.length - 1; i >= 0; i--) {
            const d = _detections[i];
            if (mx >= d.x && mx <= d.x + d.w && my >= d.y && my <= d.y + d.h) { hit = d; break; }
        }

        if (!hit) { _hideBubble(); _activeDet = null; _render(); return; }
        _activeDet = hit.id;
        _render();

        const bubble = document.getElementById('jersey-input-bubble');
        const lbl    = document.getElementById('jersey-bubble-label');
        const input  = document.getElementById('jersey-number-input');
        const sb     = rect.width / _frameW;
        const bx     = (hit.x + hit.w / 2) * sb;
        const byBelow = (hit.y + hit.h) * sb + 6;   // 球员框下方
        const byAbove = hit.y * sb;                  // 球员框上方基准

        lbl.innerHTML = (function() {
            if (hit.team === 'non_player') {
                return `<i class="fas fa-user-slash"></i> <span style="color:#ffb400">非球员</span> 球衣号码`;
            }
            const tc  = _teamColors[hit.team] || [128,128,128];
            const css = `color:rgb(${tc[0]},${tc[1]},${tc[2]})`;
            const grp = hit.team === 'team_a' ? 'A 组' : 'B 组';
            // 统计当前帧同号码已有多少个不同 track 被标注（用于提示重复标注连接轨迹）
            const curJersey = hit.jersey;
            let dupTip = '';
            if (curJersey != null) {
                const dupCount = _detections.filter(d => d.jersey === curJersey && d.id !== hit.id && d.team !== 'non_player').length;
                if (dupCount > 0) {
                    dupTip = ` <span style="color:#ffce56;font-size:0.78em;" title="同号码多个 track 将在后端自动合并为同一球员">⚡ 已连接 ${dupCount} 个</span>`;
                }
            }
            return `<i class="fas fa-user"></i> <span style="${css}">${grp}</span> 球衣号码${dupTip}`;
        })();
        input.value   = (hit.jersey && hit.team !== 'non_player') ? hit.jersey : '';
        // 非球员按钮状态
        const npBtn = document.getElementById('jersey-nonplayer-btn');
        if (npBtn) {
            if (hit.team === 'non_player') {
                npBtn.style.background  = 'rgba(255,180,0,0.28)';
                npBtn.style.borderColor = 'rgba(255,180,0,0.65)';
                npBtn.innerHTML = '<i class="fas fa-user-check"></i> 已标为非球员（点击撤销）';
            } else {
                npBtn.style.background  = 'rgba(255,180,0,0.1)';
                npBtn.style.borderColor = 'rgba(255,180,0,0.35)';
                npBtn.innerHTML = '<i class="fas fa-user-slash"></i> 标为非球员（裁判/工作人员）';
            }
        }

        // 先显示（不可见位置）以获取真实高度，再计算方向
        bubble.style.visibility = 'hidden';
        bubble.style.display    = 'block';
        bubble.style.top        = '0px';
        const bubbleH = bubble.offsetHeight || 140;
        bubble.style.visibility = '';

        // 下方空间不够则向上弹出，并确保不超出顶部
        let finalTop;
        if (byBelow + bubbleH > rect.height - 4) {
            // 向上：气泡底边贴球员框顶部
            finalTop = Math.max(4, byAbove - bubbleH - 6);
        } else {
            finalTop = byBelow;
        }
        // 水平：确保不超出右边界，也不超出左边界
        const bubbleW = bubble.offsetWidth || 185;
        let finalLeft = bx - bubbleW / 2;
        finalLeft = Math.max(4, Math.min(finalLeft, rect.width - bubbleW - 4));

        bubble.style.left = `${finalLeft}px`;
        bubble.style.top  = `${finalTop}px`;
        setTimeout(() => input.focus(), 50);
    });

    function _hideBubble() {
        const b = document.getElementById('jersey-input-bubble');
        if (b) b.style.display = 'none';
    }

    function _updateProgress() {
        const dotsEl = document.getElementById('jersey-progress-dots');
        const textEl = document.getElementById('jersey-progress-text');
        if (!dotsEl) return;
        const total      = _detections.length;
        const labeled    = _detections.filter(d => d.jersey != null && d.team !== 'non_player').length;
        const nonPlayers = _detections.filter(d => d.team === 'non_player').length;
        dotsEl.innerHTML = _detections.map((d, idx) => {
            let bg, title;
            if (d.team === 'non_player') {
                bg    = '#ffb400';
                title = `非球员 #${idx + 1}（点击定位）`;
            } else if (d.jersey != null) {
                bg    = '#00ff88';
                title = `球员 #${idx + 1}  球衣 ${d.jersey}号（点击定位）`;
            } else {
                bg    = 'rgba(255,255,255,0.18)';
                title = `球员 #${idx + 1}  未标注（点击定位）`;
            }
            const isHighlighted = d.id === _activeDet;
            const ring = isHighlighted
                ? 'box-shadow:0 0 0 2px #fff,0 0 8px rgba(255,255,255,0.7);'
                : '';
            return `<span data-det-id="${d.id}" title="${title}" style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${bg};transition:background 0.2s,box-shadow 0.2s;cursor:pointer;${ring}"></span>`;
        }).join('');

        // 绑定点击：高亮对应框，并打开输入气泡
        dotsEl.querySelectorAll('span[data-det-id]').forEach(span => {
            span.addEventListener('click', function () {
                const id  = parseInt(this.getAttribute('data-det-id'));
                const det = _detections.find(d => d.id === id);
                if (!det) return;

                // 设为激活框并重绘
                _activeDet = id;
                _render();
                _updateProgress();   // 同步高亮选中点

                // 弹出输入气泡（复用画布点击逻辑）
                const canvas = document.getElementById('jersey-canvas');
                if (!canvas) return;
                const rect = canvas.getBoundingClientRect();
                const sb   = rect.width / _frameW;

                const bubble = document.getElementById('jersey-input-bubble');
                const lbl    = document.getElementById('jersey-bubble-label');
                const input  = document.getElementById('jersey-number-input');

                lbl.innerHTML = (function() {
                    if (det.team === 'non_player') {
                        return `<i class="fas fa-user-slash"></i> <span style="color:#ffb400">非球员</span> 球衣号码`;
                    }
                    const tc  = _teamColors[det.team] || [128,128,128];
                    const css = `color:rgb(${tc[0]},${tc[1]},${tc[2]})`;
                    const grp = det.team === 'team_a' ? 'A 组' : 'B 组';
                    return `<i class="fas fa-user"></i> <span style="${css}">${grp}</span> 球衣号码`;
                })();
                input.value = (det.jersey && det.team !== 'non_player') ? det.jersey : '';

                const npBtn = document.getElementById('jersey-nonplayer-btn');
                if (npBtn) {
                    if (det.team === 'non_player') {
                        npBtn.style.background  = 'rgba(255,180,0,0.28)';
                        npBtn.style.borderColor = 'rgba(255,180,0,0.65)';
                        npBtn.innerHTML = '<i class="fas fa-user-check"></i> 已标为非球员（点击撤销）';
                    } else {
                        npBtn.style.background  = 'rgba(255,180,0,0.1)';
                        npBtn.style.borderColor = 'rgba(255,180,0,0.35)';
                        npBtn.innerHTML = '<i class="fas fa-user-slash"></i> 标为非球员（裁判/工作人员）';
                    }
                }

                // 计算气泡位置（与画布点击逻辑一致）
                const bx     = (det.x + det.w / 2) * sb;
                const byBelow = (det.y + det.h) * sb + 6;
                const byAbove = det.y * sb;

                bubble.style.visibility = 'hidden';
                bubble.style.display    = 'block';
                bubble.style.top        = '0px';
                const bubbleH = bubble.offsetHeight || 140;
                bubble.style.visibility = '';

                let finalTop;
                if (byBelow + bubbleH > rect.height - 4) {
                    finalTop = Math.max(4, byAbove - bubbleH - 6);
                } else {
                    finalTop = byBelow;
                }
                const bubbleW = bubble.offsetWidth || 185;
                let finalLeft = bx - bubbleW / 2;
                finalLeft = Math.max(4, Math.min(finalLeft, rect.width - bubbleW - 4));

                bubble.style.left = `${finalLeft}px`;
                bubble.style.top  = `${finalTop}px`;
                setTimeout(() => input.focus(), 50);
            });
        });

        // 统计每队当前跨帧去重后的人数（用于实时警告）
        const teamCounts = { team_a: new Set(), team_b: new Set() };
        Object.values(_frameDetections).forEach(dets => {
            (dets || []).forEach(d => {
                if (d.jersey != null && d.team && d.team !== 'non_player') {
                    teamCounts[d.team] && teamCounts[d.team].add(d.jersey);
                }
            });
        });
        const cntA = teamCounts.team_a.size, cntB = teamCounts.team_b.size;
        const overLimit = cntA > 11 || cntB > 11;
        const parts = [`已标注 ${labeled}/${total}`];
        if (cntA > 0 || cntB > 0) {
            const aStr = `橙队${cntA}${cntA > 11 ? '⚠️' : ''}`;
            const bStr = `蓝队${cntB}${cntB > 11 ? '⚠️' : ''}`;
            parts.push(`${aStr} / ${bStr}`);
        }
        if (nonPlayers > 0) parts.push(`非球员 ${nonPlayers}`);
        if (overLimit) parts.push('超过11人将被截断');
        if (textEl) textEl.textContent = total > 0 ? parts.join('  ') : '';
        if (textEl) textEl.style.color = overLimit ? '#ffb400' : '';
        const btn = document.getElementById('jersey-confirm-btn');
        if (btn && (labeled > 0 || nonPlayers > 0)) {
            btn.style.borderColor = 'rgba(0,255,136,0.65)';
            btn.style.boxShadow   = '0 0 22px rgba(0,255,136,0.18)';
        }
    }

    document.addEventListener('keydown', e => {
        // 标注弹窗打开时禁用 ESC 关闭（标注是必须流程）
        // 其他弹窗仍保持正常行为
    });
})();
