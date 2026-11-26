from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import os
import sys
import uuid
import shutil
import json
import asyncio
import datetime

import database
from database import VideoModel, AthleteProfile, UserModel, get_db, SessionLocal
import analysis
from auto_cleaner import get_cleaner, auto_clean_on_startup, periodic_clean_task
from path_config import ensure_data_dirs, get_export_dir, get_project_root, get_upload_dir

# Windows 编码修复：强制使用 UTF-8
# 兼容 NSSM 服务模式（stdout 已是 BufferedWriter，无 .buffer 属性）
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    import codecs
    try:
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except Exception:
        pass

@asynccontextmanager
async def lifespan(app):
    """应用生命周期管理：启动与关闭"""
    # ===== STARTUP =====
    print("\n" + "="*60)
    print("🚀 Application Starting...")
    print("="*60)
    
    # 1. 启动时自动清理数据库
    # 🔥 临时禁用：避免删除正在使用的视频记录
    print("\n[STARTUP] Step 1: Database cleanup")
    print("[INFO] Startup cleanup DISABLED (to preserve video records)")
    # await auto_clean_on_startup()  # 注释掉

    # 1b. 初始化内置账号（admin/demo）到 users 表
    try:
        _ensure_default_users()
        print("[INFO] [OK] Default users initialized")
    except Exception as e:
        print(f"[WARNING] Default users init failed: {e}")
    
    # 3. 启动后台定期清理任务
    print("\n[STARTUP] Step 3: Start periodic cleanup task")
    _background_tasks['cleaner'] = asyncio.create_task(periodic_clean_task())
    print("[INFO] Periodic cleanup enabled (interval: 60 minutes)")
    
    print("\n" + "="*60)
    print("✅ Application Started Successfully")
    print("="*60 + "\n")

    yield  # 应用运行中

    # ===== SHUTDOWN =====
    print("\n[SHUTDOWN] Stopping background tasks...")
    
    # 取消后台任务
    if 'cleaner' in _background_tasks:
        _background_tasks['cleaner'].cancel()
        try:
            await _background_tasks['cleaner']
        except asyncio.CancelledError:
            print("[INFO] Cleanup task stopped")
    
    print("[SHUTDOWN] Application closed")

app = FastAPI(lifespan=lifespan)

# 配置 CORS（必须在路由之前）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:9999", "http://localhost:9999"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 前端静态文件禁用缓存中间件（防止 JS/CSS 旧版本缓存导致显示异常）
from starlette.middleware.base import BaseHTTPMiddleware
class NoCacheFrontendMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/frontend/") and (path.endswith(".js") or path.endswith(".css") or path.endswith(".html")):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheFrontendMiddleware)

# 初始化数据库
database.init_db()

# 初始化 AI 管理器

# 后台任务存储
_background_tasks = {}

# 任务锁机制，防止重复启动
highlight_task_locks = set()  # 存储正在执行的 video_id


# ===== Pydantic 数据模型 =====


# 目录配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # 项目根目录
# uploads 放在用户目录下，避免 backend 目录权限问题
PROJECT_ROOT = get_project_root()
_USER_DATA = ensure_data_dirs()
UPLOAD_DIR = get_upload_dir()
EXPORT_DIR = get_export_dir()
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# 挂载静态文件目录（前端页面）
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
# 注意：/exports 不再使用 StaticFiles mount，改由下方自定义路由处理（支持 CORS + Range）
print(f"[INFO] Data directory: {_USER_DATA}")

print(f"[INFO] 导出目录: {EXPORT_DIR}")
print(f"[INFO] 导出目录存在: {os.path.exists(EXPORT_DIR)}")
print(f"[INFO] 导出目录内容: {os.listdir(EXPORT_DIR) if os.path.exists(EXPORT_DIR) else '无'}")

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    owner: Optional[str] = None,          # 通过查询参数传递当前登录用户名
    generate_highlight: Optional[str] = None,   # "1" 表示同时启动精彩视频生成
    target_duration: Optional[str] = None,
    enable_slowmo: Optional[str] = None,
    enable_zoom: Optional[str] = None,
    enable_pip: Optional[str] = None,
):
    file_id = str(uuid.uuid4())
    file_extension = file.filename.split(".")[-1]
    saved_filename = f"{file_id}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, saved_filename)
    
    # 使用异步迭代写入，防止 1GB 大文件保存时阻塞 FastAPI 主线程
    async def save_file():
        with open(file_path, "wb") as buffer:
            while content := await file.read(1024 * 1024): # 1MB chunk
                buffer.write(content)

    await save_file()
    
    # 存入数据库（绑定 owner，游客统一记为 "guest"）
    new_video = VideoModel(
        video_uuid=file_id,
        filename=saved_filename,
        owner=owner or "guest",
    )
    db.add(new_video)
    db.commit()
    
    # ─── 不在上传时立即启动分析，等前端弹窗标注完成后调用 /start_analysis ───
    # 初始化进度为 -1（等待启动），前端可据此判断尚未开始
    analysis.progress_store[file_id] = -1

    # 精彩视频标记暂存，/start_analysis 时再用
    if generate_highlight == "1":
        _jersey_annotation_store.setdefault(f"_highlight_{file_id}", True)
        print(f"[UPLOAD] 精彩视频将在 /start_analysis 后启动: {file_id}")

    return {"status": "success", "video_id": file_id, "filename": file.filename}


@app.post("/start_analysis/{video_id}")
async def start_analysis(video_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    前端号码标注弹窗确认/跳过后调用此接口正式启动主分析任务。
    此时 _jersey_annotation_store 中已存有（或没有）手动标注数据。
    """
    video = db.query(VideoModel).filter(VideoModel.video_uuid == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="视频记录不存在")

    file_path = os.path.join(UPLOAD_DIR, video.filename)
    abs_file_path = os.path.abspath(file_path)
    if not os.path.exists(abs_file_path):
        raise HTTPException(status_code=404, detail="视频文件不存在")

    # 读取手动标注数据（可能为空）
    jersey_annotations = _jersey_annotation_store.get(video_id, [])
    non_player_bboxes  = _non_player_store.get(video_id, [])
    print(f"[StartAnalysis] video_id={video_id}, 手动标注={len(jersey_annotations)}条, 非球员={len(non_player_bboxes)}个")

    # 若 Phase A 缓存已存在，直接进入 Phase B（渲染阶段），不重跑全量分析
    cache_path = analysis._get_phase_a_cache_path(video_id)
    if os.path.exists(cache_path):
        # 防止 Phase B 重复启动（Phase B 运行中进度为 1~99，非 95）
        current_progress = analysis.progress_store.get(video_id, -1)
        if 0 < current_progress < 95:
            return {"status": "already_started", "video_id": video_id}
        print(f"[StartAnalysis] Phase A 缓存命中，直接启动 Phase B: {video_id}")
        analysis.progress_store[video_id] = 1
        background_tasks.add_task(
            analysis.process_video_phase_b,
            abs_file_path, video_id,
            jersey_annotations=jersey_annotations,
            non_player_bboxes=non_player_bboxes,
        )
        return {"status": "started_phase_b", "video_id": video_id, "annotations_count": len(jersey_annotations)}

    # 无 Phase A 缓存：防止重复启动旧全量分析
    current_progress = analysis.progress_store.get(video_id, -1)
    if 0 < current_progress < 95:
        return {"status": "already_started", "video_id": video_id}

    analysis.progress_store[video_id] = 0
    background_tasks.add_task(analysis.process_video, abs_file_path, video_id, jersey_annotations, non_player_bboxes)

    return {"status": "started", "video_id": video_id, "annotations_count": len(jersey_annotations)}

@app.get("/api/analyzing")
async def get_analyzing_tasks():
    """返回当前正在分析中的任务列表（进度 1~99），供前端页面刷新后自动恢复轮询"""
    tasks = [
        {"video_id": vid, "progress": prog}
        for vid, prog in analysis.progress_store.items()
        if 0 < prog < 100
    ]
    return {"status": "success", "tasks": tasks}


@app.get("/results/{video_id}")
async def get_results(video_id: str, db: Session = Depends(get_db)):
    # 尝试从数据库获取结果
    video = db.query(VideoModel).filter(VideoModel.video_uuid == video_id).first()
    
    # 🔑 修复4: 增加健壮性检查
    if not video:
        # 检查内存中是否有正在进行的分析
        current_progress = analysis.progress_store.get(video_id, None)
        
        if current_progress is None:
            return {
                "error": "task_not_found",
                "message": "分析任务不存在或已被清理",
                "progress": -1,
                "detailed_analysis": {
                    "error": True,
                    "message": "请重新上传视频开始新的分析任务"
                }
            }
        else:
            return {
                "video_id": video_id,
                "progress": current_progress,
                "decision_summary": f"正在分析... (当前进度: {current_progress}%)",
                "detailed_analysis": None  # 明确返回None
            }
    
    profile = db.query(AthleteProfile).filter(AthleteProfile.video_id == video.id).first()
    
    if profile:
        # 验证数据完整性
        if not profile.detailed_analysis:
            return {
                "video_id": video_id,
                "progress": -1,
                "error": "data_incomplete",
                "message": "分析结果数据不完整",
                "detailed_analysis": {
                    "error": True,
                    "message": "数据库记录存在但内容不完整，请重新分析"
                }
            }
        
        # 验证athletes字段
        athletes = profile.detailed_analysis.get('athletes', [])
        if not isinstance(athletes, list) or len(athletes) == 0:
            return {
                "video_id": video_id,
                "progress": -1,
                "error": "invalid_data",
                "message": "分析结果中没有球员数据",
                "detailed_analysis": profile.detailed_analysis
            }
        
        return {
            "video_id": video_id,
            "posture_score": profile.overall_score,
            "decision_summary": profile.decision_summary,
            "detailed_analysis": profile.detailed_analysis,
            "progress": 100,
            "key_frames": []
        }
    
    else:
        # video存在但profile不存在 → 正在分析中
        current_progress = analysis.progress_store.get(video_id, 0)
        
        # 特殊处理：progress=100但数据库无记录（竞态条件）
        if current_progress == 100:
            await asyncio.sleep(0.5)  # 等待500ms让数据库提交完成（使用异步 sleep 不阻塞事件循环）
            # 重新查询
            profile = db.query(AthleteProfile).filter(AthleteProfile.video_id == video.id).first()
            if profile:
                return {
                    "video_id": video_id,
                    "posture_score": profile.overall_score,
                    "decision_summary": profile.decision_summary,
                    "detailed_analysis": profile.detailed_analysis,
                    "progress": 100,
                    "key_frames": []
                }
        
        status_msg = "正在扫描全场主力球员并生成 YOLO 标注视频，请稍候..."
        if current_progress == -1:
            status_msg = "分析过程中发生错误，请检查视频格式或后端日志"
        
        return {
            "video_id": video_id,
            "posture_score": 0,
            "progress": current_progress,
            "decision_summary": f"{status_msg} (当前进度: {current_progress}%)",
            "detailed_analysis": None,  # 明确返回None
            "key_frames": []
        }

# 在main.py中添加这个函数
@app.get("/docs", include_in_schema=False)
async def custom_docs():
    return HTMLResponse('''<!DOCTYPE html>
<html>
<head>
    <title>足球分析AI系统 - API文档</title>
    <!-- 使用国内可访问的CDN -->
    <link rel="stylesheet" href="https://cdn.staticfile.org/swagger-ui/4.15.5/swagger-ui.css">
    <style>
        html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
        *, *:before, *:after { box-sizing: inherit; }
        body { margin: 0; background: #fafafa; }
        .swagger-ui .topbar { display: none; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.staticfile.org/swagger-ui/4.15.5/swagger-ui-bundle.js"></script>
    <script src="https://cdn.staticfile.org/swagger-ui/4.15.5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            window.ui = SwaggerUIBundle({
                url: '/openapi.json',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                displayRequestDuration: true
            });
        }
    </script>
</body>
</html>''')

@app.get("/tactical_data/{video_id}")
async def get_tactical_data(video_id: str, db: Session = Depends(get_db)):
    """
    返回动态战术板所需的逐帧轨迹数据（JSON）。
    包含 tactical_tracks（按 track_id 分组）、ball_traj_frames 和 tactical_meta。
    数据量保护：tactical_tracks 总点数超限时自动降采样，避免大 JSON 响应超时。
    """
    video = db.query(VideoModel).filter(VideoModel.video_uuid == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    profile = db.query(AthleteProfile).filter(AthleteProfile.video_id == video.id).first()
    if not profile or not profile.detailed_analysis:
        raise HTTPException(status_code=404, detail="Analysis not ready")
    da = profile.detailed_analysis

    tactical_tracks = da.get("tactical_tracks", {})
    ball_traj_frames = da.get("ball_traj_frames", [])
    tactical_meta = da.get("tactical_meta", {})

    # ── 数据量保护：总点数超过 50000 时降采样，保证接口响应在 2 秒内 ──
    MAX_PTS = 50000
    total_pts = sum(len(v.get("pts", [])) for v in tactical_tracks.values())
    if total_pts > MAX_PTS:
        step = max(2, round(total_pts / MAX_PTS))
        tactical_tracks = {
            tid: {**info, "pts": info.get("pts", [])[::step]}
            for tid, info in tactical_tracks.items()
        }
        print(f"[TACTICAL_DATA] 降采样 step={step}，原 {total_pts} 点 → "
              f"{sum(len(v.get('pts',[])) for v in tactical_tracks.values())} 点")

    # ball_traj_frames 同样限制
    MAX_BALL = 10000
    if len(ball_traj_frames) > MAX_BALL:
        bstep = max(2, len(ball_traj_frames) // MAX_BALL)
        ball_traj_frames = ball_traj_frames[::bstep]

    return {
        "tactical_tracks": tactical_tracks,
        "ball_traj_frames": ball_traj_frames,
        "tactical_meta": tactical_meta,
    }


@app.api_route("/download/{video_id}", methods=["GET", "HEAD", "OPTIONS"])
async def download_video(video_id: str, request: Request, db: Session = Depends(get_db)):
    """
    专用下载端点，设置正确的响应头确保视频可下载和播放
    """
    # 处理 OPTIONS 预检请求
    if request.method == "OPTIONS":
        from fastapi.responses import Response
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
    
    video = db.query(VideoModel).filter(VideoModel.video_uuid == video_id).first()
    if not video:
        return {"error": "Video not found"}
    
    profile = db.query(AthleteProfile).filter(AthleteProfile.video_id == video.id).first()
    if not profile or not profile.detailed_analysis.get("export_url"):
        return {"error": "Annotated video not ready"}
    
    # 构建文件路径
    filename = profile.detailed_analysis["export_url"].replace("/exports/", "")
    file_path = os.path.join(EXPORT_DIR, filename)
    
    if not os.path.exists(file_path):
        return {"error": "File not found", "path": file_path}
    
    # 返回文件响应，设置正确的 MIME 类型和下载头
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=f"足球分析_{video_id[:8]}.mp4",
        headers={
            "Content-Disposition": f'attachment; filename="football_analysis_{video_id[:8]}.mp4"',
            "Cache-Control": "no-cache",
            "Accept-Ranges": "bytes",
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.api_route("/preview/{video_id}", methods=["GET", "HEAD", "OPTIONS"])
async def preview_video(video_id: str, request: Request, db: Session = Depends(get_db)):
    """
    在线预览端点，支持 Range 请求的视频流播放
    """
    # 处理 OPTIONS 预检请求
    if request.method == "OPTIONS":
        from fastapi.responses import Response
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
    
    video = db.query(VideoModel).filter(VideoModel.video_uuid == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    profile = db.query(AthleteProfile).filter(AthleteProfile.video_id == video.id).first()
    if not profile or not profile.detailed_analysis.get("export_url"):
        raise HTTPException(status_code=404, detail="Annotated video not ready")
    
    # 构建文件路径
    filename = profile.detailed_analysis["export_url"].replace("/exports/", "")
    file_path = os.path.join(EXPORT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    # 获取文件大小
    file_size = os.path.getsize(file_path)
    
    # 处理 HEAD 请求（只返回头信息，不返回文件内容）
    if request.method == "HEAD":
        from fastapi.responses import Response
        return Response(
            status_code=200,
            headers={
                "Content-Type": "video/mp4",
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-cache"
            }
        )
    
    # 处理 Range 请求（视频播放必需）
    range_header = request.headers.get("range")
    
    if range_header:
        # 解析 Range 头: "bytes=start-end"
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0]) if range_match[0] else 0
        end = int(range_match[1]) if len(range_match) > 1 and range_match[1] else file_size - 1
        
        # 限制范围
        start = max(0, start)
        end = min(end, file_size - 1)
        content_length = end - start + 1
        
        # 读取指定范围的数据
        def iter_file():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(1024 * 1024, remaining)  # 1MB chunks
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data
        
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": "video/mp4",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
        
        return StreamingResponse(
            iter_file(),
            status_code=206,  # Partial Content
            headers=headers
        )
    
    # 无 Range 请求，返回完整文件
    def iter_full_file():
        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):  # 1MB chunks
                yield chunk
    
    headers = {
        "Content-Length": str(file_size),
        "Content-Type": "video/mp4",
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache",
        "Access-Control-Allow-Origin": "*"
    }
    
    return StreamingResponse(
        iter_full_file(),
        headers=headers
    )


# 根路由：重定向到前端主页面
@app.get("/")
def read_root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/frontend/index.html")

# 添加健康检查
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "football-ai", "timestamp": datetime.datetime.now().isoformat()}


# ===== 自定义静态文件端点（支持 CORS + Range 请求）=====
@app.api_route("/exports/{filename:path}", methods=["GET", "HEAD", "OPTIONS"])
async def serve_export_file(filename: str, request: Request):
    """
    自定义静态文件服务，支持 CORS 和 Range 请求
    """
    # 处理 OPTIONS 预检请求
    if request.method == "OPTIONS":
        from fastapi.responses import Response
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
    
    file_path = os.path.join(EXPORT_DIR, filename)
    
    if not os.path.exists(file_path):
        print(f"[EXPORTS] 文件不存在: {file_path}  (EXPORT_DIR={EXPORT_DIR})")
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    # 根据扩展名判断 Content-Type
    import mimetypes
    content_type, _ = mimetypes.guess_type(file_path)
    if not content_type:
        content_type = "application/octet-stream"

    # 获取文件大小
    file_size = os.path.getsize(file_path)
    
    # 处理 HEAD 请求
    if request.method == "HEAD":
        from fastapi.responses import Response
        return Response(
            status_code=200,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-cache"
            }
        )
    
    # 处理 Range 请求
    range_header = request.headers.get("range")
    
    if range_header:
        # 解析 Range 头
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0]) if range_match[0] else 0
        end = int(range_match[1]) if len(range_match) > 1 and range_match[1] else file_size - 1
        
        start = max(0, start)
        end = min(end, file_size - 1)
        content_length = end - start + 1
        
        def iter_file():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(1024 * 1024, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data
        
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": content_type,
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
        
        return StreamingResponse(
            iter_file(),
            status_code=206,
            headers=headers
        )
    
    # 完整文件响应
    def iter_full_file():
        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk
    
    headers = {
        "Content-Length": str(file_size),
        "Content-Type": content_type,
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache",
        "Access-Control-Allow-Origin": "*"
    }
    
    return StreamingResponse(
        iter_full_file(),
        headers=headers
    )

print(f"[INFO] 静态文件路由已配置: /exports -> {EXPORT_DIR}")


# ===== 缓存同步辅助函数 =====


# ===== AI 缓存管理端点 =====






# ===== 数据库自动清理端点 =====
@app.post("/api/database/clean")
async def clean_database(
    db: Session = Depends(get_db),
    clean_type: str = "full"  # full, duplicates, orphaned, failed
):
    """
    手动触发数据库清理
    
    参数：
        clean_type: 清理类型
            - "full": 完整清理（默认）
            - "duplicates": 只清理重复记录
            - "orphaned": 只清理孤立视频
            - "failed": 只清理失败记录
    
    响应：
    {
        "status": "success",
        "total_removed": 5,
        "details": {
            "duplicates": {"removed_count": 2, ...},
            "orphaned": {"removed_count": 1, ...},
            "failed": {"removed_count": 2, ...}
        }
    }
    """
    try:
        cleaner = get_cleaner()
        
        if clean_type == "duplicates":
            result = cleaner.clean_duplicates(db)
        elif clean_type == "orphaned":
            result = cleaner.clean_orphaned_videos(db)
        elif clean_type == "failed":
            result = cleaner.clean_failed_videos(db)
        else:  # full
            result = cleaner.full_clean(db)
        
        return result
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Database cleanup failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/database/status")
async def get_database_status(db: Session = Depends(get_db)):
    """
    获取数据库状态信息
    
    响应：
    {
        "total_videos": 15,
        "total_profiles": 15,
        "duplicates": 0,
        "orphaned": 0,
        "failed": 0,
        "health": "good"
    }
    """
    try:
        from collections import defaultdict
        
        # 统计总数
        total_videos = db.query(VideoModel).count()
        total_profiles = db.query(AthleteProfile).count()
        
        # 检查重复
        all_videos = db.query(VideoModel).all()
        uuid_count = defaultdict(int)
        for video in all_videos:
            uuid_count[video.video_uuid] += 1
        
        duplicate_count = sum(1 for count in uuid_count.values() if count > 1)
        
        # 检查孤立
        orphaned_count = db.query(VideoModel).outerjoin(
            AthleteProfile, VideoModel.id == AthleteProfile.video_id
        ).filter(AthleteProfile.id == None).count()
        
        # 检查失败
        failed_count = 0
        videos_with_profiles = db.query(VideoModel).join(
            AthleteProfile, VideoModel.id == AthleteProfile.video_id
        ).all()
        
        for video in videos_with_profiles:
            profile = db.query(AthleteProfile).filter(
                AthleteProfile.video_id == video.id
            ).first()
            
            if profile and profile.detailed_analysis:
                detailed = profile.detailed_analysis
                if isinstance(detailed, dict) and detailed.get('error'):
                    failed_count += 1
        
        # 评估健康状态
        issues = duplicate_count + orphaned_count + failed_count
        if issues == 0:
            health = "excellent"
        elif issues <= 3:
            health = "good"
        elif issues <= 10:
            health = "fair"
        else:
            health = "poor"
        
        # 检测幽灵档案（AthleteProfile 无对应 Video 记录）
        ghost_profiles = db.query(AthleteProfile).outerjoin(
            VideoModel, AthleteProfile.video_id == VideoModel.id
        ).filter(VideoModel.id == None).count()

        return {
            "status": "success",
            "total_videos": total_videos,
            "total_profiles": total_profiles,
            "duplicates": duplicate_count,
            "orphaned": orphaned_count,
            "failed": failed_count,
            "ghost_profiles": ghost_profiles,
            "issues_total": issues + ghost_profiles,
            "health": health,
            "auto_clean_enabled": True,
            "auto_clean_interval": "10 minutes"
        }

    except Exception as e:
        import traceback
        print(f"[ERROR] Get database status failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/api/database/fix-heatmap-urls")
async def fix_heatmap_urls(db: Session = Depends(get_db)):
    """
    扫描所有 heatmap_urls 为空的记录，
    检查磁盘上是否存在对应热力图文件，若存在则补全数据库字段。
    """
    heatmap_dir = os.path.join(EXPORT_DIR, "heatmaps")
    fixed = []
    skipped = []

    profiles = db.query(AthleteProfile).join(
        VideoModel, AthleteProfile.video_id == VideoModel.id
    ).filter(AthleteProfile.detailed_analysis != None).all()

    import copy
    for profile in profiles:
        da = profile.detailed_analysis or {}
        video = db.query(VideoModel).filter(VideoModel.id == profile.video_id).first()
        if not video:
            continue
        uuid = video.video_uuid
        changed = False
        new_da = copy.deepcopy(da)

        # ── 补全 heatmap_urls（team_a / team_b 热力图）──
        if not da.get("heatmap_urls"):
            urls = {}
            for team in ("team_a", "team_b"):
                fname = f"heatmap_{uuid}_{team}.png"
                fpath = os.path.join(heatmap_dir, fname)
                if os.path.exists(fpath):
                    urls[team] = f"/exports/heatmaps/{fname}"
            if urls:
                new_da["heatmap_urls"] = urls
                changed = True

        # ── 补全 tactical_url（战术鸟瞰图）──
        if not da.get("tactical_url"):
            tactical_fname = f"tactical_{uuid}.png"
            tactical_fpath = os.path.join(heatmap_dir, tactical_fname)
            if os.path.exists(tactical_fpath):
                new_da["tactical_url"] = f"/exports/heatmaps/{tactical_fname}"
                changed = True

        if changed:
            profile.detailed_analysis = new_da
            fixed.append({
                "video_uuid": uuid,
                "heatmap_urls": new_da.get("heatmap_urls"),
                "tactical_url": new_da.get("tactical_url"),
            })
        else:
            skipped.append(uuid)

    if fixed:
        db.commit()

    return {
        "fixed_count": len(fixed),
        "skipped_count": len(skipped),
        "fixed": fixed,
        "skipped_no_file": skipped,
    }


@app.post("/api/database/clean-ghosts")
async def clean_ghost_records(db: Session = Depends(get_db)):
    """
    清理所有幽灵数据记录

    幽灵数据定义：
      1. 幽灵档案（Ghost Profiles）  — AthleteProfile 记录对应的 video_id
                                      在 videos 表中已不存在
      2. 孤立视频（Orphaned Videos） — VideoModel 无任何 AthleteProfile，
                                      且上传文件也不存在于磁盘
      3. 文件孤儿（File Orphans）    — uploads/ 目录中存在 mp4 文件但
                                      数据库中无对应 video_uuid 记录

    返回每类幽灵数据的清理数量，不影响正常分析记录。
    """
    report = {
        "ghost_profiles_removed": 0,
        "orphaned_videos_removed": 0,
        "file_orphans_removed": 0,
        "details": [],
    }

    try:
        # ── 1. 清理幽灵档案 ───────────────────────────────────────────────────
        ghost_profiles = db.query(AthleteProfile).outerjoin(
            VideoModel, AthleteProfile.video_id == VideoModel.id
        ).filter(VideoModel.id == None).all()

        for gp in ghost_profiles:
            report["details"].append({
                "type": "ghost_profile",
                "athlete_profile_id": gp.id,
                "video_id_ref": gp.video_id,
            })
            db.delete(gp)

        report["ghost_profiles_removed"] = len(ghost_profiles)

        # ── 2. 清理孤立视频（无档案 + 文件不存在） ────────────────────────────
        orphaned_videos = db.query(VideoModel).outerjoin(
            AthleteProfile, VideoModel.id == AthleteProfile.video_id
        ).filter(AthleteProfile.id == None).all()

        orphan_removed = 0
        for v in orphaned_videos:
            # 排除正在分析中的视频
            if str(v.id) in highlight_task_locks or v.video_uuid in highlight_task_locks:
                continue
            # 只清理磁盘文件也不存在的记录（文件存在说明可能正在分析）
            video_file = os.path.join(UPLOAD_DIR, f"{v.video_uuid}.mp4")
            if not os.path.exists(video_file):
                report["details"].append({
                    "type": "orphaned_video",
                    "video_id": v.id,
                    "video_uuid": v.video_uuid,
                    "filename": v.filename,
                })
                db.delete(v)
                orphan_removed += 1

        report["orphaned_videos_removed"] = orphan_removed

        # ── 3. 清理文件孤儿（磁盘文件无数据库记录） ──────────────────────────
        if os.path.exists(UPLOAD_DIR):
            # 获取数据库中所有 video_uuid
            db_uuids = {row.video_uuid for row in db.query(VideoModel.video_uuid).all()}

            file_orphans_removed = 0
            for fname in os.listdir(UPLOAD_DIR):
                if not fname.endswith(".mp4"):
                    continue
                # _highlights.mp4 是生成的精彩视频，不是源文件
                if fname.endswith("_highlights.mp4"):
                    continue
                uuid_part = fname.replace(".mp4", "")
                if uuid_part not in db_uuids:
                    file_path = os.path.join(UPLOAD_DIR, fname)
                    try:
                        os.remove(file_path)
                        report["details"].append({
                            "type": "file_orphan",
                            "filename": fname,
                            "path": file_path,
                        })
                        file_orphans_removed += 1
                    except OSError as e:
                        report["details"].append({
                            "type": "file_orphan_error",
                            "filename": fname,
                            "error": str(e),
                        })

            report["file_orphans_removed"] = file_orphans_removed

        db.commit()

        total = (
            report["ghost_profiles_removed"]
            + report["orphaned_videos_removed"]
            + report["file_orphans_removed"]
        )
        report["total_removed"] = total
        report["status"] = "success"
        report["message"] = (
            f"清理完成：幽灵档案 {report['ghost_profiles_removed']} 条，"
            f"孤立视频 {report['orphaned_videos_removed']} 条，"
            f"文件孤儿 {report['file_orphans_removed']} 个"
        )

        print(f"[CLEAN_GHOSTS] {report['message']}")
        return report

    except Exception as e:
        db.rollback()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ===== AI 智能对话端点 =====













# ===== AI 幽灵数据管理端点 =====





@app.get("/api/videos")
async def get_analyzed_videos(db: Session = Depends(get_db), owner: Optional[str] = None):
    """
    获取所有已完成分析的视频列表
    返回格式：[{id: video_uuid, name: filename, analyzed_at: timestamp}]
    
    参数：
      owner: 登录用户名（不传则返回全部，传入则只返回该用户的视频；
             admin 用户返回全部）
    
    🔄 自动功能：
    1. 查询数据库中的有效视频
    3. 去重（同一个video_uuid只保留最新记录）
    """
    try:
        print("[INFO] 正在查询已分析的视频...")
        
        # 步骤2: 查询所有有关联 AthleteProfile 的视频（即已完成分析的）
        query = db.query(VideoModel).join(
            AthleteProfile,
            VideoModel.id == AthleteProfile.video_id
        )

        # 按 owner 过滤：admin 看全部，其他用户只看自己的
        if owner and owner != "admin":
            query = query.filter(VideoModel.owner == owner)

        videos_with_analysis = query.all()
        
        print(f"[INFO] 数据库中找到 {len(videos_with_analysis)} 个已分析的视频")
        
        # 步骤3: 去重处理（使用字典确保每个video_uuid只保留一条）
        video_dict = {}
        
        for video in videos_with_analysis:
            # 获取关联的分析数据
            profile = db.query(AthleteProfile).filter(
                AthleteProfile.video_id == video.id
            ).first()
            
            # 检查是否分析失败
            if profile and profile.detailed_analysis:
                detailed = profile.detailed_analysis
                # 跳过分析失败的视频
                if isinstance(detailed, dict) and detailed.get('error'):
                    print(f"[INFO] Skip failed video: {video.filename} (error: {detailed.get('message', 'Unknown')})")
                    continue
                
                video_info = {
                    "id": video.video_uuid,
                    "name": video.filename or f"Video_{video.video_uuid[:8]}",
                    "analyzed_at": database.to_beijing_time(video.upload_time).isoformat() if hasattr(video, 'upload_time') and video.upload_time else None,
                    "db_id": video.id,
                    # 历史记录额外字段
                    "has_highlight": os.path.exists(
                        os.path.join(UPLOAD_DIR, f"{video.video_uuid}_highlights.mp4")
                    ),
                    "highlight_url": f"/api/download-highlight/{video.video_uuid}" if os.path.exists(
                        os.path.join(UPLOAD_DIR, f"{video.video_uuid}_highlights.mp4")
                    ) else None,
                    "highlight_size_mb": round(
                        os.path.getsize(os.path.join(UPLOAD_DIR, f"{video.video_uuid}_highlights.mp4")) / 1048576, 1
                    ) if os.path.exists(os.path.join(UPLOAD_DIR, f"{video.video_uuid}_highlights.mp4")) else None,
                    "athlete_count": db.query(AthleteProfile).filter(AthleteProfile.video_id == video.id).count(),
                }
                
                # 去重逻辑：如果已存在相同UUID，保留最新的（upload_time最大或db_id最大）
                if video.video_uuid in video_dict:
                    existing = video_dict[video.video_uuid]
                    existing_time = existing.get('analyzed_at') or ''
                    current_time = video_info.get('analyzed_at') or ''
                    
                    # 保留时间更新的，或时间相同时保留ID更大的
                    if current_time > existing_time or (current_time == existing_time and video.id > existing['db_id']):
                        print(f"[DEDUP] Replace duplicate: {video.filename} (old_id={existing['db_id']}, new_id={video.id})")
                        video_dict[video.video_uuid] = video_info
                    else:
                        print(f"[DEDUP] Skip duplicate: {video.filename} (keep_id={existing['db_id']}, skip_id={video.id})")
                else:
                    video_dict[video.video_uuid] = video_info
                    print(f"[INFO] Add video: {video_info['name']} (ID: {video_info['id'][:8]}...)")
        
        # 移除临时的db_id字段
        result = []
        for video_info in video_dict.values():
            del video_info['db_id']
            result.append(video_info)
        
        # 按上传时间倒序排序（最新的在前）
        result.sort(key=lambda x: x.get('analyzed_at') or '', reverse=True)
        
        print(f"[INFO] [OK] Return {len(result)} unique valid videos (raw: {len(videos_with_analysis)})")
        
        return {
            "status": "success",
            "count": len(result),
            "videos": result
        }
    
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to fetch videos: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "videos": []
        }


# ============================================================================
#  删除历史记录 API
# ============================================================================

@app.delete("/api/videos/{video_id}")
async def delete_video_record(video_id: str, db: Session = Depends(get_db)):
    """
    删除指定视频的历史记录：
      1. 删除 videos 表记录
      2. 删除 athlete_profiles 表关联记录
      4. 删除磁盘上的原始视频文件、导出视频文件、热力图文件、phase_a 缓存
    """
    import traceback as _tb
    try:
        video = db.query(VideoModel).filter(VideoModel.video_uuid == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="记录不存在")

        deleted_files = []
        errors = []

        # ── 1. 收集需要删除的文件路径 ────────────────────────────────────────
        profile = db.query(AthleteProfile).filter(AthleteProfile.video_id == video.id).first()
        files_to_delete = []

        # 原始上传视频
        upload_path = os.path.join(UPLOAD_DIR, video.filename)
        files_to_delete.append(upload_path)

        # phase_a 缓存
        from analysis import _get_phase_a_cache_path
        phase_a_path = _get_phase_a_cache_path(video_id)
        files_to_delete.append(phase_a_path)

        if profile and profile.detailed_analysis:
            da = profile.detailed_analysis
            # 导出标注视频
            export_url = da.get("export_url", "")
            if export_url:
                export_fname = export_url.replace("/exports/", "").lstrip("/")
                files_to_delete.append(os.path.join(EXPORT_DIR, export_fname))
            # 精彩视频
            highlight_url = da.get("highlight_url", "")
            if highlight_url:
                hl_fname = highlight_url.replace("/exports/", "").lstrip("/")
                files_to_delete.append(os.path.join(EXPORT_DIR, hl_fname))
            # 热力图
            heatmap_urls = da.get("heatmap_urls", {})
            if isinstance(heatmap_urls, dict):
                for hurl in heatmap_urls.values():
                    if hurl:
                        hm_fname = hurl.replace("/exports/heatmaps/", "").replace("/exports/", "").lstrip("/")
                        files_to_delete.append(os.path.join(EXPORT_DIR, "heatmaps", hm_fname))
            elif isinstance(heatmap_urls, list):
                for hurl in heatmap_urls:
                    if hurl:
                        hm_fname = hurl.replace("/exports/heatmaps/", "").replace("/exports/", "").lstrip("/")
                        files_to_delete.append(os.path.join(EXPORT_DIR, "heatmaps", hm_fname))

        # ── 2. 删除磁盘文件 ──────────────────────────────────────────────────
        for fpath in files_to_delete:
            if fpath and os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    deleted_files.append(fpath)
                    print(f"[DELETE] 已删除文件: {fpath}")
                except Exception as fe:
                    errors.append(f"文件删除失败: {fpath} ({fe})")
                    print(f"[DELETE][WARN] {fe}")

        # ── 3. 删除数据库记录（按外键顺序）──────────────────────────────────
        # AthleteProfile
        db.query(AthleteProfile).filter(AthleteProfile.video_id == video.id).delete()
        # 内存中的进度/知识库缓存清理
        analysis.progress_store.pop(video_id, None)
        # 主记录
        db.delete(video)
        db.commit()
        print(f"[DELETE] 历史记录已删除: video_uuid={video_id}")

        return {
            "status": "success",
            "video_id": video_id,
            "deleted_files": deleted_files,
            "warnings": errors,
        }

    except HTTPException:
        raise
    except Exception as e:
        _tb.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
#  精彩视频生成 API
# ============================================================================

from pydantic import BaseModel

class HighlightVideoRequest(BaseModel):
    """精彩视频生成请求"""
    video_id: str
    target_duration: float = 180.0  # 默认3分钟
    enable_slowmo: bool = True      # 启用慢动作
    enable_zoom: bool = True        # 启用特写
    enable_pip: bool = True         # 启用画中画
    enable_bgm: bool = True         # 启用激昂背景音乐（去除原音频）
    bgm_volume: float = 0.88        # BGM 音量（0~1）

# 精彩视频生成进度存储  { video_id: int(0-100/-1) }
highlight_progress_store = {}
# 心跳时间戳：记录每次进度更新的时间，前端用于判断是否真正卡住
highlight_heartbeat_store = {}   # { video_id: float(timestamp) }
# 错误信息存储
highlight_error_store = {}       # { video_id: str }

@app.post("/api/generate-highlight")
async def generate_highlight_video(
    request: HighlightVideoRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    生成精彩视频
    
    功能：
    1. 自动识别精彩片段（特别是进球时刻）
    2. 应用特效：慢动作、特写、画中画
    3. 生成约3分钟的精华视频
    """
    
    try:
        # 验证视频是否存在
        video = db.query(VideoModel).filter(VideoModel.video_uuid == request.video_id).first()
        
        print(f"[DEBUG] 查询视频ID: {request.video_id}")
        
        if not video:
            raise HTTPException(status_code=404, detail="视频不存在")
        
        print(f"[DEBUG] 找到视频记录，filename={video.filename}")
        
        # 构建视频文件路径
        input_path = os.path.join(UPLOAD_DIR, video.filename)
        
        print(f"[DEBUG] 准备查找文件路径: {input_path}")
        
        if not os.path.exists(input_path):
            raise HTTPException(status_code=404, detail=f"视频文件不存在: {video.filename}")
        
        # 🔥 任务锁检查：防止同一视频的重复启动
        if request.video_id in highlight_task_locks:
            current_progress = highlight_progress_store.get(request.video_id, 0)
            print(f"[HIGHLIGHT] ⚠️ 任务已在进行中！video_id={request.video_id}, 进度={current_progress}%")
            print(f"[HIGHLIGHT] ⚠️ 拒绝重复启动请求（使用任务锁机制）")
            return {
                "status": "already_running",
                "message": f"精彩视频生成任务已在进行中（进度: {current_progress}%）",
                "video_id": request.video_id,
                "current_progress": current_progress
            }
        
        # 🔥 获取任务锁
        highlight_task_locks.add(request.video_id)
        print(f"[HIGHLIGHT] 🔒 已获取任务锁: {request.video_id}")
        
        # 初始化进度
        highlight_progress_store[request.video_id] = 0
        print(f"[HIGHLIGHT] ✅ 初始化进度为 0%，准备启动任务")
        
        # 生成输出文件名
        output_filename = f"{request.video_id}_highlights.mp4"
        output_path = os.path.join(UPLOAD_DIR, output_filename)
        
        # 后台任务生成精彩视频
        # 注意：background_tasks.add_task 必须传同步函数，
        # FastAPI 会自动用线程池执行，不能传 async 函数（会导致 coroutine 永不执行）
        background_tasks.add_task(
            _generate_highlight_task,
            request.video_id,
            input_path,
            output_path,
            request.target_duration,
            request.enable_slowmo,
            request.enable_zoom,
            request.enable_pip,
            request.enable_bgm,
            request.bgm_volume
        )
        
        return {
            "status": "success",
            "message": "精彩视频生成任务已启动",
            "video_id": request.video_id,
            "output_filename": output_filename
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ERROR] 启动精彩视频生成失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _generate_highlight_task(
    video_id: str,
    input_path: str,
    output_path: str,
    target_duration: float,
    enable_slowmo: bool,
    enable_zoom: bool,
    enable_pip: bool,
    enable_bgm: bool = True,
    bgm_volume: float = 0.88,
    force_frame_scan: bool = False,  # True时跳过快速路径，始终逐帧YOLO扫描（射门检测更准）
):
    """后台任务：生成精彩视频（优先复用分析数据，跳过重新扫描视频）"""
    import time as _time

    def _set_progress(pct: int, label: str = ""):
        highlight_progress_store[video_id] = pct
        highlight_heartbeat_store[video_id] = _time.time()
        print(f"[HIGHLIGHT] 📊 进度: {pct}%  {label}")

    try:
        print(f"\n[HIGHLIGHT] ==================== 任务开始 ====================")
        print(f"[HIGHLIGHT] 视频ID: {video_id}")
        print(f"[HIGHLIGHT] 输入文件: {input_path}")
        print(f"[HIGHLIGHT] 输出文件: {output_path}")
        print(f"[HIGHLIGHT] 目标时长: {target_duration}秒")
        print(f"[HIGHLIGHT] 特效设置: 慢动作={enable_slowmo}, 特写={enable_zoom}, 画中画={enable_pip}, BGM={enable_bgm}(音量{bgm_volume})")
        print(f"[HIGHLIGHT] =====================================================\n")

        _set_progress(1, "初始化任务")

        # ── 验证输入文件 ──────────────────────────────────────────────────────
        if not os.path.exists(input_path):
            print(f"[HIGHLIGHT ERROR] ❌ 输入文件不存在: {input_path}")
            highlight_progress_store[video_id] = -1
            return

        # ── 获取视频基本信息 ──────────────────────────────────────────────────
        import cv2 as _cv2
        cap_info = _cv2.VideoCapture(input_path)
        vid_total_frames = 0
        vid_fps = 25.0
        vid_duration = 0.0
        if cap_info.isOpened():
            vid_total_frames = int(cap_info.get(_cv2.CAP_PROP_FRAME_COUNT))
            vid_fps = cap_info.get(_cv2.CAP_PROP_FPS) or 25.0
            vid_duration = vid_total_frames / vid_fps
            print(f"[HIGHLIGHT] 📹 视频信息: {vid_duration:.1f}秒, {vid_fps:.1f} FPS, {vid_total_frames} 帧")
            cap_info.release()

        # 若视频时长不足目标时长，自动降低目标时长，避免重复拼接
        if vid_duration > 0 and vid_duration < target_duration:
            print(f"[HIGHLIGHT] ⚠️ 视频实际时长 {vid_duration:.1f}s < 目标时长 {target_duration:.0f}s，"
                  f"自动调整目标时长为 {vid_duration:.1f}s")
            target_duration = vid_duration

        _set_progress(3, "检测模式选择")

        # ── 尝试从数据库读取已有分析数据（快速路径，仅在 force_frame_scan=False 时使用） ────────────
        analysis_data = None
        if not force_frame_scan:
            try:
                from database import SessionLocal, VideoModel, AthleteProfile
                db_sess = SessionLocal()
                try:
                    vid_rec = db_sess.query(VideoModel).filter(
                        VideoModel.video_uuid == video_id
                    ).first()
                    if vid_rec:
                        profile = db_sess.query(AthleteProfile).filter(
                            AthleteProfile.video_id == vid_rec.id
                        ).first()
                        if profile and profile.detailed_analysis:
                            da = profile.detailed_analysis
                            ball_traj = da.get("ball_traj_frames", [])
                            tact_tracks = da.get("tactical_tracks", {})
                            tmeta = da.get("tactical_meta", {})
                            shot_events = da.get("shot_event_frames", [])
                            # 兼容旧版分析结果：只要有 tactical_tracks 或 ball_traj 任一不为空即可走快速路径
                            # shot_event_frames 在旧版中可能不存在，不作为必要条件
                            if ball_traj or tact_tracks:
                                analysis_data = {
                                    "ball_traj_frames": ball_traj,
                                    "tactical_tracks":  tact_tracks,
                                    "fps":          tmeta.get("fps", vid_fps),
                                    "total_frames": tmeta.get("total_frames", vid_total_frames) or vid_total_frames,
                                    "pitch_l":      tmeta.get("pitch_l", 105.0),
                                    "pitch_w":      tmeta.get("pitch_w", 68.0),
                                    "shot_event_frames": shot_events,
                                }
                                print(f"[HIGHLIGHT] ✅ 快速路径：读取到 {len(ball_traj)} 个球轨迹点，"
                                      f"{len(tact_tracks)} 条球员轨迹，{len(shot_events)} 个射门事件帧")
                            else:
                                print(f"[HIGHLIGHT] ⚠️ tactical_tracks={len(tact_tracks)} ball_traj={len(ball_traj)}，"
                                      f"分析数据为空，退回完整扫描模式")
                finally:
                    db_sess.close()
            except Exception as db_err:
                print(f"[HIGHLIGHT] ⚠️ 读取分析数据失败（退回完整扫描）: {db_err}")
                analysis_data = None
        else:
            print("[HIGHLIGHT] 🔍 逐帧扫描模式（force_frame_scan=True），跳过快速路径")

        _set_progress(5, "加载模块")

        # ── 导入生成器 ────────────────────────────────────────────────────────
        from highlight_generator import HighlightVideoGenerator

        # ── YOLO 模型（完整扫描模式才真正用到，快速路径可不加载） ─────────────
        yolo_model = None
        if not analysis_data:
            _set_progress(6, "加载 YOLO 模型（完整扫描模式）")
            try:
                from analysis import get_yolo_model
                yolo_model = get_yolo_model()
                print(f"[HIGHLIGHT] ✅ YOLO 模型加载成功")
            except Exception as e:
                print(f"[HIGHLIGHT ERROR] ❌ YOLO 模型加载失败: {e}")
                import traceback
                traceback.print_exc()
                highlight_progress_store[video_id] = -1
                return
        else:
            print(f"[HIGHLIGHT] 快速路径：跳过 YOLO 加载")

        _set_progress(8, "创建生成器")
        generator = HighlightVideoGenerator(yolo_model=yolo_model)

        # ── 进度回调 ──────────────────────────────────────────────────────────
        def progress_callback(progress: int):
            _set_progress(progress)

        # ── 生成精彩视频 ──────────────────────────────────────────────────────
        result = generator.generate_highlight_video(
            input_video=input_path,
            output_video=output_path,
            target_duration=target_duration,
            enable_slowmo=enable_slowmo,
            enable_zoom=enable_zoom,
            enable_pip=enable_pip,
            enable_bgm=enable_bgm,
            bgm_volume=bgm_volume,
            progress_callback=progress_callback,
            analysis_data=analysis_data,   # 传入快速路径数据
        )

        if result['success']:
            print(f"[HIGHLIGHT] ✅ 精彩视频生成成功: {output_path}")
            _set_progress(100, "完成")
        else:
            err = result.get('error', '未知错误')
            print(f"[HIGHLIGHT] ❌ 精彩视频生成失败: {err}")
            highlight_progress_store[video_id] = -1
            highlight_error_store[video_id] = err

    except Exception as e:
        print(f"[HIGHLIGHT] ❌ 生成过程出错: {e}")
        import traceback
        traceback.print_exc()
        highlight_progress_store[video_id] = -1
        highlight_error_store[video_id] = str(e)

    finally:
        # 无论成功还是失败，都要释放任务锁
        if video_id in highlight_task_locks:
            highlight_task_locks.remove(video_id)
            print(f"[HIGHLIGHT] 🔓 已释放任务锁: {video_id}")


@app.get("/api/highlight-progress/{video_id}")
async def get_highlight_progress(video_id: str):
    """
    查询精彩视频生成进度
    额外返回 last_heartbeat（上次进度更新的 Unix 时间戳），前端可用于
    判断任务是否真正卡住（超过 N 秒无心跳则提示用户）。
    """
    import time as _time
    progress  = highlight_progress_store.get(video_id)
    heartbeat = highlight_heartbeat_store.get(video_id)
    error_msg = highlight_error_store.get(video_id, "")

    if progress is None:
        return {
            "status": "not_found",
            "message": "精彩视频生成任务不存在",
            "progress": None
        }
    elif progress == -1:
        # 分析错误类型，给出友好提示
        if "系统找不到指定的文件" in error_msg or "WinError 2" in error_msg:
            friendly = "ffmpeg 路径异常，请重启后端服务后重试"
        elif "TimeoutExpired" in error_msg or "超时" in error_msg:
            friendly = "视频处理超时，建议缩短视频时长或降低分辨率后重试"
        elif "codec" in error_msg.lower() or "decoder" in error_msg.lower():
            friendly = "视频编解码格式不支持，请将视频转换为 H.264/MP4 格式后重试"
        elif "No such file" in error_msg or "不存在" in error_msg:
            friendly = "视频文件已被删除，请重新上传"
        elif error_msg:
            friendly = f"生成失败：{error_msg[:120]}"
        else:
            friendly = "精彩视频生成失败，请重新上传视频后重试"
        return {
            "status": "failed",
            "message": friendly,
            "error_detail": error_msg,
            "progress": -1
        }
    elif progress == 100:
        output_filename = f"{video_id}_highlights.mp4"
        output_path = os.path.join(UPLOAD_DIR, output_filename)
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            # 获取精彩视频实际时长
            try:
                import cv2 as _cv2_h
                cap_h = _cv2_h.VideoCapture(output_path)
                h_frames = int(cap_h.get(_cv2_h.CAP_PROP_FRAME_COUNT))
                h_fps    = cap_h.get(_cv2_h.CAP_PROP_FPS) or 25.0
                cap_h.release()
                actual_duration = round(h_frames / h_fps, 1)
            except Exception:
                actual_duration = None
            return {
                "status": "completed",
                "message": "精彩视频生成完成",
                "progress": 100,
                "output_file": output_filename,
                "download_url": f"/api/download-highlight/{video_id}",
                "file_size_mb": round(file_size, 2),
                "duration": actual_duration,
            }
        else:
            return {
                "status": "error",
                "message": "精彩视频文件不存在",
                "progress": -1
            }
    else:
        return {
            "status": "processing",
            "message": f"正在生成精彩视频... ({progress}%)",
            "progress": progress,
            "last_heartbeat": heartbeat   # Unix timestamp，前端可做心跳检测
        }


@app.get("/api/download-highlight/{video_id}")
async def download_highlight_video(video_id: str):
    """下载精彩视频（强制下载）"""
    
    output_filename = f"{video_id}_highlights.mp4"
    output_path = os.path.join(UPLOAD_DIR, output_filename)
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="精彩视频文件不存在")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"highlights_{video_id}.mp4",
        headers={"Content-Disposition": f"attachment; filename=highlights_{video_id}.mp4"}
    )


@app.get("/api/stream-highlight/{video_id}")
async def stream_highlight_video(video_id: str, request: Request):
    """精彩视频流式预览（支持 HTTP Range，供浏览器 <video> 标签使用）"""
    output_filename = f"{video_id}_highlights.mp4"
    output_path = os.path.join(UPLOAD_DIR, output_filename)

    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="精彩视频文件不存在")

    file_size = os.path.getsize(output_path)
    range_header = request.headers.get("range")

    if range_header:
        # 解析 Range: bytes=start-end
        try:
            range_val = range_header.strip().replace("bytes=", "")
            parts = range_val.split("-")
            start = int(parts[0])
            end = int(parts[1]) if parts[1] else file_size - 1
        except Exception:
            raise HTTPException(status_code=416, detail="Invalid Range header")

        end = min(end, file_size - 1)
        chunk_size = end - start + 1

        def iter_file():
            with open(output_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
            "Content-Type": "video/mp4",
        }
        return StreamingResponse(iter_file(), status_code=206, headers=headers)
    else:
        # 无 Range 头：返回全文件
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": "video/mp4",
        }
        def iter_full():
            with open(output_path, "rb") as f:
                while True:
                    data = f.read(65536)
                    if not data:
                        break
                    yield data
        return StreamingResponse(iter_full(), status_code=200, headers=headers)


# ============================================================================
#  用户认证模块
# ============================================================================
import hashlib
import secrets
import re as _re

# 管理员会话Token存储：{token: {username, expire_time}}
_admin_tokens: dict = {}

# 访问日志（内存，最多保留500条）
_access_logs: list = []


def _get_user_db() -> Session:
    """获取数据库Session（用于非依赖注入场景）"""
    return SessionLocal()


def _ensure_default_users():
    password = os.environ.get("COACHARGUS_ADMIN_PASSWORD")
    if not password:
        return
    if len(password) < 12:
        raise ValueError("COACHARGUS_ADMIN_PASSWORD must contain at least 12 characters")
    with _get_user_db() as db:
        if _lookup_user(db, "admin") is None:
            db.add(UserModel(
                username="admin",
                email="admin@coachargus.invalid",
                password_hash=hashlib.sha256(password.encode()).hexdigest(),
                role="admin",
            ))
            db.commit()


def _lookup_user(db: Session, username_or_email: str) -> Optional[UserModel]:
    """按用户名或邮箱查找用户"""
    user = db.query(UserModel).filter(UserModel.username == username_or_email).first()
    if not user:
        user = db.query(UserModel).filter(UserModel.email == username_or_email).first()
    return user


def _beijing_now_str() -> str:
    """返回东八区当前时间字符串（yyyy-MM-dd HH:mm:ss）"""
    tz8 = datetime.timezone(datetime.timedelta(hours=8))
    return datetime.datetime.now(tz8).strftime("%Y-%m-%d %H:%M:%S")


def _add_access_log(action: str, username: str, detail: str = "", log_type: str = "blue", ip: str = ""):
    """记录访问日志（时间以东八区存储）"""
    _access_logs.insert(0, {
        "action": action,
        "user": username,
        "detail": detail,
        "type": log_type,
        "ip": ip,
        "time": _beijing_now_str()
    })
    # 限制日志条数
    while len(_access_logs) > 500:
        _access_logs.pop()

def _verify_admin_token(request: Request) -> str:
    """验证管理员Token，返回username；验证失败抛出403"""
    token = request.headers.get("X-Admin-Token", "")
    if not token:
        raise HTTPException(status_code=403, detail="缺少管理员Token")
    session = _admin_tokens.get(token)
    if not session:
        raise HTTPException(status_code=403, detail="Token无效或已过期")
    if datetime.datetime.now().timestamp() > session["expire_time"]:
        _admin_tokens.pop(token, None)
        raise HTTPException(status_code=403, detail="Token已过期，请重新登录")
    return session["username"]




class AuthLoginRequest(BaseModel):
    username: str
    password: str

class AuthRegisterRequest(BaseModel):
    username: str
    email: str
    password: str

@app.post("/api/auth/login")
async def auth_login(req: AuthLoginRequest, request: Request, db: Session = Depends(get_db)):
    """用户登录（支持用户名或邮箱）"""
    user = _lookup_user(db, req.username.strip())
    if not user or not user.is_active:
        _add_access_log("user_login_fail", req.username, "账号不存在或已禁用", "red",
                        ip=request.client.host if request.client else "")
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    pwd_hash = hashlib.sha256(req.password.encode()).hexdigest()
    if user.password_hash != pwd_hash:
        _add_access_log("user_login_fail", req.username, "密码错误", "red",
                        ip=request.client.host if request.client else "")
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    _add_access_log("user_login", user.username, "用户登录成功", "green",
                    ip=request.client.host if request.client else "")
    return {"username": user.username, "role": user.role, "email": user.email or ""}


@app.post("/api/auth/admin-login")
async def admin_login(req: AuthLoginRequest, request: Request, db: Session = Depends(get_db)):
    """
    管理员专用登录接口
    - 仅允许 role=admin 的账号
    - 成功后返回 session token（有效期2小时）
    """
    user = _lookup_user(db, req.username.strip())
    pwd_hash = hashlib.sha256(req.password.encode()).hexdigest()

    if not user or user.password_hash != pwd_hash:
        _add_access_log("admin_login_fail", req.username, "密码错误", "red",
                        ip=request.client.host if request.client else "")
        raise HTTPException(status_code=401, detail="账号或密码错误")

    if user.role != "admin":
        _add_access_log("admin_login_fail", req.username, "非管理员账号尝试登录后台", "red",
                        ip=request.client.host if request.client else "")
        raise HTTPException(status_code=403, detail="无管理员权限")

    # 生成安全Token
    token = secrets.token_urlsafe(32)
    expire = (datetime.datetime.now() + datetime.timedelta(hours=2)).timestamp()
    _admin_tokens[token] = {"username": user.username, "expire_time": expire}

    _add_access_log("admin_login", user.username, "管理员登录后台成功", "gold",
                    ip=request.client.host if request.client else "")

    return {
        "username": user.username,
        "role": user.role,
        "token": token,
        "expire_in": 7200,
    }

@app.post("/api/auth/register")
async def auth_register(req: AuthRegisterRequest, request: Request, db: Session = Depends(get_db)):
    """
    用户注册
    - 用户名唯一性检查
    - 邮箱唯一性检查
    - 邮箱格式验证
    - 用户名长度限制（4-20字符，字母数字下划线）
    - 密码强度要求（至少8位）
    """
    username = req.username.strip()
    email = req.email.strip().lower()
    password = req.password

    # 1. 用户名格式校验
    if len(username) < 4 or len(username) > 20:
        raise HTTPException(status_code=400, detail="用户名长度须在4到20个字符之间")
    if not _re.match(r'^[a-zA-Z0-9_\u4e00-\u9fa5]+$', username):
        raise HTTPException(status_code=400, detail="用户名只能包含字母、数字、下划线或中文")

    # 2. 邮箱格式校验
    if not _re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
        raise HTTPException(status_code=400, detail="邮箱格式不正确")

    # 3. 密码强度校验
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="密码至少8位")

    # 4. 用户名唯一性检查
    if db.query(UserModel).filter(UserModel.username == username).first():
        raise HTTPException(status_code=409, detail="用户名已被注册，请换一个")

    # 5. 邮箱唯一性检查
    if db.query(UserModel).filter(UserModel.email == email).first():
        raise HTTPException(status_code=409, detail="该邮箱已被注册，请使用其他邮箱或直接登录")

    # 6. 写入数据库
    new_user = UserModel(
        username=username,
        email=email,
        password_hash=hashlib.sha256(password.encode()).hexdigest(),
        role="user",
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    _add_access_log("register", username, f"新用户注册: {email}", "blue",
                    ip=request.client.host if request.client else "")
    return {"message": "注册成功", "username": username}




# ============================================================================
#  访问日志 & 用户管理 API
# ============================================================================

@app.post("/api/admin/access-log")
async def post_access_log(request: Request):
    """前端写入访问日志（管理员操作日志，需Token）"""
    _verify_admin_token(request)
    try:
        body = await request.json()
        _add_access_log(
            action=body.get("action", "unknown"),
            username=body.get("username", "unknown"),
            detail=body.get("detail", ""),
            log_type=body.get("type", "blue"),
            ip=request.client.host if request.client else ""
        )
        return {"ok": True}
    except Exception:
        return {"ok": False}

@app.post("/api/access-log")
async def post_public_access_log(request: Request):
    """公开访问日志上报（普通用户/主页访问，无需Token）"""
    try:
        body = await request.json()
        action = body.get("action", "page_view")
        username = body.get("username", "游客")
        detail = body.get("detail", "")
        log_type = body.get("type", "blue")
        # 安全过滤：只允许上报特定类型，防止滥用
        allowed_actions = {"page_view", "user_login", "user_logout", "upload_start", "analysis_start"}
        if action not in allowed_actions:
            return {"ok": False, "reason": "action not allowed"}
        _add_access_log(
            action=action,
            username=username,
            detail=detail,
            log_type=log_type,
            ip=request.client.host if request.client else ""
        )
        return {"ok": True}
    except Exception:
        return {"ok": False}

@app.get("/api/admin/access-logs")
async def get_access_logs(request: Request, limit: int = 100):
    """获取访问日志列表（需管理员Token）"""
    _verify_admin_token(request)
    return {"logs": _access_logs[:limit], "total": len(_access_logs)}

@app.get("/api/admin/users")
async def get_users_list(request: Request, db: Session = Depends(get_db)):
    """获取注册用户列表（需管理员Token）"""
    _verify_admin_token(request)
    users = db.query(UserModel).order_by(UserModel.created_at.asc()).all()
    # 在线状态：最近 30 分钟内有任意操作日志则视为在线
    tz8 = datetime.timezone(datetime.timedelta(hours=8))
    cutoff = (datetime.datetime.now(tz8) - datetime.timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
    # 排除失败日志（login_fail），其余任何操作都算活跃
    online_users = {
        l["user"] for l in _access_logs
        if not l.get("action", "").endswith("_fail")
        and l.get("time", "") >= cutoff
    }
    result = [
        {
            "username": u.username,
            "role": u.role,
            "email": u.email,
            "created_at": (database.to_beijing_time(u.created_at).strftime("%Y-%m-%d %H:%M:%S")
                          if u.created_at and database.to_beijing_time(u.created_at) else ""),
            "is_active": u.is_active,
            "status": "online" if u.username in online_users else "offline",
        }
        for u in users
    ]
    return {"users": result, "total": len(result)}


# ============================================================================
#  数据统计 & 管理 API
# ============================================================================

@app.get("/api/stats/overview")
async def get_stats_overview(db: Session = Depends(get_db)):
    """
    系统概览统计
    返回视频数量、球员数量、精彩视频数量等
    """
    try:
        # 统计视频总数
        total_videos = db.query(VideoModel).count()

        # 统计球员档案总数
        total_athletes = db.query(AthleteProfile).count()

        # 统计已生成精彩视频数（检查文件存在）
        highlight_files = [
            f for f in os.listdir(UPLOAD_DIR)
            if f.endswith("_highlights.mp4")
        ] if os.path.exists(UPLOAD_DIR) else []
        total_highlights = len(highlight_files)

        # 统计上传文件总大小（MB）
        total_size_mb = 0.0
        if os.path.exists(UPLOAD_DIR):
            for f in os.listdir(UPLOAD_DIR):
                fp = os.path.join(UPLOAD_DIR, f)
                if os.path.isfile(fp):
                    total_size_mb += os.path.getsize(fp)
        total_size_mb = round(total_size_mb / (1024 * 1024), 1)

        # 正在进行的生成任务
        active_tasks = len(highlight_task_locks)

        return {
            "total_videos": total_videos,
            "total_athletes": total_athletes,
            "total_highlights": total_highlights,
            "total_size_mb": total_size_mb,
            "active_tasks": active_tasks,
            "registered_users": db.query(UserModel).count(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats/videos")
async def get_video_stats(db: Session = Depends(get_db)):
    """
    视频分析统计列表
    返回所有视频的分析结果概要
    """
    try:
        # ORM字段名为 upload_time，对外 API 字段名统一用 created_at
        videos = db.query(VideoModel).order_by(VideoModel.upload_time.desc()).limit(50).all()
        result = []
        for v in videos:
            vid_int = v.id                  # 整数主键，用于数据库关联查询
            video_id = str(vid_int)         # 字符串形式，用于进度缓存查找
            video_uuid = str(v.video_uuid or "")

            # has_highlight 以 video_uuid 命名的精彩集锦文件为准
            has_highlight = bool(video_uuid) and os.path.exists(
                os.path.join(UPLOAD_DIR, f"{video_uuid}_highlights.mp4")
            )

            # athlete_count：用整数主键比较，避免字符串与整数类型不匹配
            athletes = db.query(AthleteProfile).filter(
                AthleteProfile.video_id == vid_int
            ).count()

            # 球员详情：从 detailed_analysis 中读取真实球员列表长度
            profile = db.query(AthleteProfile).filter(
                AthleteProfile.video_id == vid_int
            ).first()
            if profile and profile.detailed_analysis:
                da = profile.detailed_analysis
                # 若有真实球员列表，以列表长度为准
                real_athletes = da.get("athletes", [])
                if isinstance(real_athletes, list) and len(real_athletes) > 0:
                    athletes = len(real_athletes)

            # 将 UTC 时间转为东八区
            bj_time = database.to_beijing_time(v.upload_time)
            result.append({
                "video_id": video_id,
                "video_uuid": video_uuid,
                "filename": v.filename,
                "created_at": bj_time.strftime("%Y-%m-%d %H:%M:%S") if bj_time else None,
                "athlete_count": athletes,
                "has_highlight": has_highlight,
                "highlight_progress": highlight_progress_store.get(video_uuid, highlight_progress_store.get(video_id, 0)),
            })
        return {"videos": result, "total": len(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats/athletes")
async def get_athlete_stats(db: Session = Depends(get_db)):
    """
    球员统计数据汇总
    返回评分分布、平均能力值等
    """
    try:
        athletes = db.query(AthleteProfile).all()
        if not athletes:
            return {"total": 0, "avg_score": 0, "score_distribution": {}, "top_players": []}

        scores = []
        top_players = []
        for a in athletes:
            score = getattr(a, 'overall_score', None) or 0
            # 跳过分析失败（score=0且有error字段）的记录
            da = a.detailed_analysis or {}
            if da.get("error"):
                continue
            scores.append(score)

            # 读取真实球员人数
            real_athletes = da.get("athletes", [])
            athlete_count = len(real_athletes) if isinstance(real_athletes, list) else 0

            # 查找关联视频的 uuid 和 filename
            video = db.query(VideoModel).filter(VideoModel.id == a.video_id).first()
            video_uuid = str(video.video_uuid) if video and video.video_uuid else str(a.video_id)
            filename = video.filename if video else ""

            top_players.append({
                "id": str(a.id),
                "video_id": video_uuid,       # 改为返回 video_uuid，前端显示更友好
                "filename": filename,
                "athlete_count": athlete_count,
                "overall_score": round(score, 1),
            })

        top_players.sort(key=lambda x: x["overall_score"], reverse=True)

        # 评分段分布
        distribution = {"<60": 0, "60-75": 0, "75-85": 0, "85+": 0}
        for s in scores:
            if s < 60:   distribution["<60"] += 1
            elif s < 75: distribution["60-75"] += 1
            elif s < 85: distribution["75-85"] += 1
            else:        distribution["85+"] += 1

        return {
            "total": len(athletes),
            "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "score_distribution": distribution,
            "top_players": top_players[:10],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/video/{video_id}")
async def admin_delete_video(video_id: str, request: Request, db: Session = Depends(get_db)):
    """管理员删除视频及关联数据（需管理员Token）"""
    admin_user = _verify_admin_token(request)
    try:
        # video_id 为整数主键字符串，转换为 int 确保匹配
        vid_int = int(video_id)
        video = db.query(VideoModel).filter(VideoModel.id == vid_int).first()
        if not video:
            raise HTTPException(status_code=404, detail=f"视频不存在: {video_id}")

        filename = str(video.filename)
        video_uuid = str(video.video_uuid)

        deleted_files = []

        # ── 1. 删除 UPLOAD_DIR 中的原始视频和精彩集锦 ──────────────────────
        for suffix in ["", "_highlights"]:
            for name_key in [video_uuid, str(vid_int)]:
                fp = os.path.join(UPLOAD_DIR, f"{name_key}{suffix}.mp4")
                if os.path.exists(fp):
                    os.remove(fp)
                    deleted_files.append(fp)
        # 也尝试删除以原始文件名保存的视频（部分早期上传方式）
        original_fp = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(original_fp):
            os.remove(original_fp)
            deleted_files.append(original_fp)

        # ── 2. 删除 EXPORT_DIR 中所有与该视频相关的导出文件 ──────────────
        # 2a. 标注视频（annotated_{uuid}.mp4）
        annotated_fp = os.path.join(EXPORT_DIR, f"annotated_{video_uuid}.mp4")
        if os.path.exists(annotated_fp):
            os.remove(annotated_fp)
            deleted_files.append(annotated_fp)

        # 2b. 热力图和战术板（存放在 EXPORT_DIR/heatmaps/ 子目录）
        heatmap_dir = os.path.join(EXPORT_DIR, "heatmaps")
        heatmap_patterns = [
            f"heatmap_{video_uuid}_team_a.png",
            f"heatmap_{video_uuid}_team_b.png",
            f"heatmap_{video_uuid}_team_A.png",
            f"heatmap_{video_uuid}_team_B.png",
            f"tactical_{video_uuid}.png",
        ]
        for fname in heatmap_patterns:
            fp = os.path.join(heatmap_dir, fname)
            if os.path.exists(fp):
                os.remove(fp)
                deleted_files.append(fp)

        # 2c. 扫描 EXPORT_DIR 及子目录，兜底删除所有包含 video_uuid 的文件
        if os.path.exists(EXPORT_DIR):
            for root, _dirs, files in os.walk(EXPORT_DIR):
                for fname in files:
                    if video_uuid in fname:
                        fp = os.path.join(root, fname)
                        if fp not in deleted_files:
                            try:
                                os.remove(fp)
                                deleted_files.append(fp)
                            except Exception:
                                pass

        # ── 3. 删除所有关联数据库记录 ──────────────────────────────────────
        db.query(AthleteProfile).filter(AthleteProfile.video_id == vid_int).delete()
        db.query(VideoModel).filter(VideoModel.id == vid_int).delete()
        db.commit()

        # ── 4. 清理内存进度缓存 ────────────────────────────────────────────
        highlight_progress_store.pop(video_uuid, None)
        highlight_progress_store.pop(str(vid_int), None)
        highlight_task_locks.discard(video_uuid)
        highlight_task_locks.discard(str(vid_int))

        _add_access_log("delete", admin_user, f"删除视频: {filename}（清理文件 {len(deleted_files)} 个）", "red",
                        ip=request.client.host if request.client else "")
        return {"message": "删除成功", "video_id": video_id, "deleted_files": len(deleted_files)}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/reset-all-data")
async def admin_reset_all_data(request: Request, db: Session = Depends(get_db)):
    """管理员：清空所有视频、分析数据、AI缓存（需管理员Token）"""
    admin_user = _verify_admin_token(request)
    try:
        # 1. 删除所有上传文件和导出文件
        deleted_files = 0
        for search_dir in [UPLOAD_DIR]:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for fname in files:
                        fp = os.path.join(root, fname)
                        try:
                            os.remove(fp)
                            deleted_files += 1
                        except Exception:
                            pass

        # 2. 清空数据库所有相关表（按外键依赖顺序）
        profile_del  = db.query(AthleteProfile).delete()
        video_del    = db.query(VideoModel).delete()
        db.commit()

        # 3. 清空内存缓存
        highlight_progress_store.clear()
        highlight_task_locks.clear()

        _add_access_log("admin_reset", admin_user, "重置所有数据", "red",
                        ip=request.client.host if request.client else "")
        return {
            "message": "重置成功",
            "deleted_files": deleted_files,
            "deleted_rows": {
                "videos": video_del,
                "athlete_profiles": profile_del,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/system-info")
async def get_system_info(request: Request):
    """系统信息（需管理员Token）"""
    _verify_admin_token(request)
    import platform
    try:
        import psutil, sys as _sys
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        # Windows 使用 C:\ ，其他系统使用 /
        _disk_path = 'C:\\' if _sys.platform.startswith('win') else '/'
        disk = psutil.disk_usage(_disk_path).percent
    except Exception:
        cpu = mem = disk = None

    info = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "active_highlight_tasks": list(highlight_task_locks),
        "progress_store_size": len(highlight_progress_store),
        "upload_dir": UPLOAD_DIR,
        "cpu_percent": cpu,
        "memory_percent": mem,
        "disk_usage_percent": disk,
    }

    return info




# ─────────────────────────────────────────────────────────────────────────────
#  球员号码标注弹窗 API
# ─────────────────────────────────────────────────────────────────────────────

# 临时存储：{video_id: [{"det_id":int,"jersey_number":int,"team":str}, ...]}
_jersey_annotation_store: dict = {}
# 非球员 bbox 临时存储：{video_id: [{"det_id":int,"x":int,"y":int,"w":int,"h":int}, ...]}
_non_player_store: dict = {}

@app.get("/quick_detect/{video_id}")
async def quick_detect(video_id: str, db: Session = Depends(get_db)):
    """
    提取视频中人数最多的 TOP-N 关键帧，并返回每帧的球员检测框（供前端多帧累积标注使用）。
    - 从视频均匀抽取若干帧，对每帧运行 YOLO 检测
    - 选出人数最多的 TOP 5 帧（相邻帧间距 > 5 秒，保证画面多样性）
    - 每帧保存为 JPEG，返回帧图 URL 列表和检测框坐标
    """
    import cv2, base64, tempfile

    # 查找视频文件（支持 .mp4 和上传时记录的原始扩展名）
    video = db.query(VideoModel).filter(VideoModel.video_uuid == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")

    # 尝试常见扩展名
    video_path = None
    for ext in ["mp4", "avi", "mov", "mkv"]:
        candidate = os.path.join(UPLOAD_DIR, f"{video_id}.{ext}")
        if os.path.exists(candidate):
            video_path = candidate
            break
    if video_path is None:
        candidate2 = os.path.join(UPLOAD_DIR, video.filename)
        if os.path.exists(candidate2):
            video_path = candidate2
    if video_path is None:
        raise HTTPException(status_code=404, detail="视频文件不存在")

    # ── 抽帧 & 对每帧 YOLO 检测 ────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="无法读取视频")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 均匀采样 20 帧，跳过前后各 5%
    sample_count = 20
    skip_start   = int(total_frames * 0.05)
    skip_end     = int(total_frames * 0.05)
    usable       = max(1, total_frames - skip_start - skip_end)
    step         = max(1, usable // sample_count)
    sample_idxs  = [skip_start + i * step for i in range(sample_count) if skip_start + i * step < total_frames - skip_end]

    try:
        import torch as _torch_qd
        from ultralytics import YOLO as _YOLO
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _base_dir = os.path.dirname(_script_dir)
        _best_pt_candidates = [
            os.path.join(_base_dir, 'best.pt'),
            os.path.join(_script_dir, 'best.pt'),
        ]
        yolo_path = next((p for p in _best_pt_candidates if os.path.exists(p)), None)
        _yolo = _YOLO(yolo_path) if yolo_path else None
        if _yolo and _torch_qd.cuda.is_available():
            _yolo.to('cuda')
    except Exception:
        _yolo = None

    # 收集所有候选帧信息
    all_candidates = []   # [{"fidx": int, "frame": np.ndarray, "dets": [...]}]

    for fidx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        dets = []
        if _yolo is not None:
            try:
                # conf=0.45 减少低质量重复框；iou=0.4 让 YOLO 内置 NMS 更积极合并重叠框
                _qd_half = _torch_qd.cuda.is_available()
                results = _yolo(frame, classes=[0], conf=0.45, iou=0.4, imgsz=1280, verbose=False, half=_qd_half)
                boxes = results[0].boxes if results else None
                if boxes is not None:
                    raw_dets = []
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if (y2 - y1) < 20:
                            continue
                        raw_dets.append({
                            "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                            "conf": float(box.conf[0])
                        })
                    # 二次 NMS：消除置信度不同但框高度重叠的冗余检测（同一人被检测多次）
                    raw_dets.sort(key=lambda d: d["conf"], reverse=True)
                    def _iou(a, b):
                        ax1, ay1 = a["x"], a["y"]
                        ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
                        bx1, by1 = b["x"], b["y"]
                        bx2, by2 = bx1 + b["w"], by1 + b["h"]
                        ix = max(0, min(ax2, bx2) - max(ax1, bx1))
                        iy = max(0, min(ay2, by2) - max(ay1, by1))
                        inter = ix * iy
                        union = a["w"] * a["h"] + b["w"] * b["h"] - inter
                        return inter / union if union > 0 else 0.0
                    kept = []
                    for d in raw_dets:
                        if all(_iou(d, k) < 0.35 for k in kept):
                            kept.append(d)
                    for d in kept:
                        del d["conf"]
                    dets = kept
            except Exception:
                pass
        all_candidates.append({"fidx": fidx, "frame": frame, "dets": dets})

    cap.release()

    if not all_candidates:
        raise HTTPException(status_code=500, detail="无法从视频中提取关键帧")

    # ── 选 TOP-5 帧：按人数降序，相邻帧间距 > 5 秒（避免重复画面）──────────
    all_candidates.sort(key=lambda c: len(c["dets"]), reverse=True)
    MIN_FRAME_GAP = int(fps * 5)   # 5 秒间距
    TOP_N = 5
    selected = []
    for c in all_candidates:
        # 距离已选帧都超过 MIN_FRAME_GAP 才纳入
        if all(abs(c["fidx"] - s["fidx"]) >= MIN_FRAME_GAP for s in selected):
            selected.append(c)
        if len(selected) >= TOP_N:
            break

    # 至少保留1帧（兜底）
    if not selected:
        selected = [all_candidates[0]]

    # ── 颜色聚类辅助函数（对单帧 dets + frame 做 K-means 分队）────────────
    import numpy as np

    def _cluster_frame(frame_img, dets):
        """对单帧球员做颜色聚类分队，返回 (dets_with_team, team_colors)"""
        tc = {"team_a": [255, 130, 40], "team_b": [60, 100, 255]}
        if len(dets) < 2:
            for i, d in enumerate(dets):
                d["team"] = "team_a" if i % 2 == 0 else "team_b"
            return dets, tc
        try:
            feats, bgr_means = [], []
            fh, fw = frame_img.shape[:2]
            for d in dets:
                x, y, w, h = d["x"], d["y"], d["w"], d["h"]
                t  = max(0, y + int(h * 0.2));  b  = min(fh, y + int(h * 0.5))
                xl = max(0, x + int(w * 0.2));  xr = min(fw, x + int(w * 0.8))
                roi = frame_img[t:b, xl:xr]
                if roi.size == 0:
                    feats.append([0.0, 0.0]); bgr_means.append([128, 128, 128]); continue
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                hsv_flat = hsv.reshape(-1, 3).astype(float)
                feats.append([float(hsv_flat[:, 0].mean()), float(hsv_flat[:, 1].mean())])
                bgr_flat = roi.reshape(-1, 3).astype(float)
                bgr_means.append([float(bgr_flat[:, c].mean()) for c in range(3)])
            feat_arr = np.array(feats, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, _ = cv2.kmeans(feat_arr, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
            for i, d in enumerate(dets):
                d["team"] = "team_a" if int(labels[i][0]) == 0 else "team_b"
            def _rgb(cid):
                idxs = [i for i in range(len(dets)) if int(labels[i][0]) == cid]
                if not idxs: return [128, 128, 128]
                return [int(np.mean([bgr_means[i][2] for i in idxs])),
                        int(np.mean([bgr_means[i][1] for i in idxs])),
                        int(np.mean([bgr_means[i][0] for i in idxs]))]
            tc = {"team_a": _rgb(0), "team_b": _rgb(1)}
        except Exception as _ce:
            print(f"[quick_detect] 颜色聚类失败: {_ce}")
            for i, d in enumerate(dets):
                d["team"] = "team_a" if i % 2 == 0 else "team_b"
        return dets, tc

    # ── 保存所有选中帧为 JPEG，并组装返回数据 ────────────────────────────
    tmp_dir = os.path.join(UPLOAD_DIR, "tmp_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    frames_data = []
    global_team_colors = None   # 用人数最多帧的颜色作为全局代表色

    for fi, c in enumerate(selected):
        dets_with_team, tc = _cluster_frame(c["frame"], c["dets"])
        fname = f"{video_id}_keyframe_{fi}.jpg"
        fpath = os.path.join(tmp_dir, fname)
        cv2.imwrite(fpath, c["frame"], [cv2.IMWRITE_JPEG_QUALITY, 88])
        if fi == 0:
            global_team_colors = tc
            # 兼容旧版：仍保存 _keyframe.jpg（analysis.py 读取 non_player_bboxes 用）
            cv2.imwrite(os.path.join(tmp_dir, f"{video_id}_keyframe.jpg"),
                        c["frame"], [cv2.IMWRITE_JPEG_QUALITY, 88])
        frames_data.append({
            "frame_url":  f"/tmp_frames/{fname}",
            "frame_time": round(c["fidx"] / fps, 1),
            "player_count": len(dets_with_team),
            "detections": dets_with_team,
            "team_colors": tc,
        })

    return {
        "video_id":    video_id,
        "frame_w":     frame_w,
        "frame_h":     frame_h,
        "frames":      frames_data,              # 多帧数据列表
        "team_colors": global_team_colors,       # 全局代表色（第一帧）
        # ── 向下兼容旧版单帧字段 ──
        "frame_url":   frames_data[0]["frame_url"],
        "frame_time":  frames_data[0]["frame_time"],
        "detections":  frames_data[0]["detections"],
    }


@app.post("/jersey_annotations/{video_id}")
async def save_jersey_annotations(video_id: str, request: Request):
    """
    接收前端号码标注弹窗提交的球衣号码数据，暂存供主分析流程读取。
    请求体：{"annotations": [...], "team_swap": bool, "non_player_bboxes": [...]}
    """
    try:
        body = await request.json()
        annotations      = body.get("annotations", [])
        team_swap        = bool(body.get("team_swap", False))   # 用户是否互换了主/客队
        non_player_bboxes = body.get("non_player_bboxes", [])   # 非球员 bbox 列表

        # 如果用户选择了互换，把所有标注里的 team_a <-> team_b 对调
        if team_swap:
            swap_map = {"team_a": "team_b", "team_b": "team_a"}
            for ann in annotations:
                ann["team"] = swap_map.get(ann.get("team", "team_a"), "team_a")
            print(f"[JerseyAnnotation] team_swap=True，已互换主/客队")

        _jersey_annotation_store[video_id] = annotations

        # 存储非球员 bbox，供 analysis.process_video 使用
        if non_player_bboxes:
            _non_player_store[video_id] = non_player_bboxes
            print(f"[JerseyAnnotation] video_id={video_id}, 收到 {len(annotations)} 条标注, {len(non_player_bboxes)} 个非球员")
        else:
            print(f"[JerseyAnnotation] video_id={video_id}, 收到 {len(annotations)} 条标注")
        return {"status": "ok", "count": len(annotations)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"解析请求失败: {e}")


@app.get("/tmp_frames/{filename}")
async def serve_tmp_frame(filename: str):
    """提供关键帧 JPEG 文件访问"""
    frame_path = os.path.join(UPLOAD_DIR, "tmp_frames", filename)
    if not os.path.exists(frame_path):
        raise HTTPException(status_code=404, detail="帧文件不存在")
    return FileResponse(frame_path, media_type="image/jpeg")


# ═══════════════════════════════════════════════════════════════════════════════
# 追踪前置方案：Phase A / Phase B 接口
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/pre_track/{video_id}")
async def pre_track(video_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    上传视频后调用：启动 Phase A（全局YOLO检测+追踪），结果缓存到 pkl。
    完成后前端调用 /quick_detect_v2 读取缓存选帧。
    """
    video = db.query(VideoModel).filter(VideoModel.video_uuid == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="视频记录不存在")

    file_path = os.path.join(UPLOAD_DIR, video.filename)
    abs_file_path = os.path.abspath(file_path)
    if not os.path.exists(abs_file_path):
        raise HTTPException(status_code=404, detail="视频文件不存在")

    # 优先检查磁盘缓存（pkl 存在 = Phase A 已完成，无论进程是否重启）
    cache_path = analysis._get_phase_a_cache_path(video_id)
    if os.path.exists(cache_path):
        # 同步内存状态，防止前端轮询时拿到 -1 误判为失败
        analysis.phase_a_progress_store[video_id] = 95
        return {"status": "cached", "video_id": video_id}

    # 防止重复启动（进程内存中有正在运行的任务）
    current_progress = analysis.phase_a_progress_store.get(video_id, -1)
    if 0 < current_progress < 95:
        return {"status": "already_running", "progress": current_progress}

    analysis.phase_a_progress_store[video_id] = 1
    background_tasks.add_task(analysis.process_video_phase_a, abs_file_path, video_id)
    return {"status": "started", "video_id": video_id}


@app.get("/pre_track_progress/{video_id}")
async def pre_track_progress(video_id: str):
    """查询 Phase A 进度"""
    cache_path = analysis._get_phase_a_cache_path(video_id)
    cached = os.path.exists(cache_path)
    prog = analysis.phase_a_progress_store.get(video_id, -1)
    # 重启后内存进度丢失（prog=-1），但若 pkl 已存在则视为成功完成
    if cached and prog <= 0:
        prog = 95
        analysis.phase_a_progress_store[video_id] = 95
    return {"video_id": video_id, "progress": prog, "cached": cached}


@app.get("/quick_detect_v2/{video_id}")
async def quick_detect_v2(video_id: str, db: Session = Depends(get_db)):
    """
    读取 Phase A 缓存，用贪心集合覆盖选帧，返回每帧截图+带 track_id 的检测框。
    前端用 track_id 做跨帧精确回填和非球员排除。
    """
    video = db.query(VideoModel).filter(VideoModel.video_uuid == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")

    try:
        result = analysis.phase_a_select_frames(video_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Phase A 缓存不存在，请先调用 /pre_track")

    if not result["frames"]:
        track_count = len(result.get("all_tracks", {}))
        raise HTTPException(
            status_code=500,
            detail=(
                f"关键帧提取失败：未追踪到任何球员（track数={track_count}）。"
                "可能原因：①视频草地颜色过暗导致场景过滤器跳过所有帧；"
                "②视频分辨率过低导致球员检测框被尺寸过滤器过滤；"
                "③模型类别不匹配（请查看后端日志中的 [Phase A][diag] 诊断信息）。"
                "请重新上传视频并调用 /pre_track 重新处理。"
            )
        )

    # 从视频中提取选中帧的图片
    import cv2 as _cv2

    video_path = os.path.join(UPLOAD_DIR, video.filename)
    cap = _cv2.VideoCapture(video_path)

    tmp_dir = os.path.join(UPLOAD_DIR, "tmp_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    fps = result["fps"]
    frames_data = []

    # 标注去重规则：两个人的框重合度（IoU）> 0.70 时仅保留一个
    # 采用“先按面积从大到小排序，再贪心保留”的方式，避免重复图标叠在一起。
    def _iou_xywh(a, b):
        ax1, ay1 = float(a.get("x", 0)), float(a.get("y", 0))
        aw, ah = float(a.get("w", 0)), float(a.get("h", 0))
        bx1, by1 = float(b.get("x", 0)), float(b.get("y", 0))
        bw, bh = float(b.get("w", 0)), float(b.get("h", 0))
        ax2, ay2 = ax1 + max(0.0, aw), ay1 + max(0.0, ah)
        bx2, by2 = bx1 + max(0.0, bw), by1 + max(0.0, bh)

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0

        area_a = max(0.0, aw) * max(0.0, ah)
        area_b = max(0.0, bw) * max(0.0, bh)
        union = area_a + area_b - inter
        return (inter / union) if union > 0 else 0.0

    def _dedup_overlapping_dets(dets, iou_threshold=0.35):
        if not dets:
            return []
        # 同一 track_id 只保留面积最大的框（去除重复追踪）
        track_best = {}
        no_track = []
        for det in dets:
            tid = det.get("track_id")
            if tid is None:
                no_track.append(det)
                continue
            area = float(det.get("w", 0)) * float(det.get("h", 0))
            if tid not in track_best or area > float(track_best[tid].get("w", 0)) * float(track_best[tid].get("h", 0)):
                track_best[tid] = det
        # 不同 track_id 的框全部保留（不按 IoU 互相吞并）
        return list(track_best.values()) + no_track

    for fi, frame_info in enumerate(result["frames"]):
        fno = frame_info["fno"]

        # 修复：用 imencode 验证帧数据完整性，最多重试3次，每次往前多读帧以预热解码器
        frame_buf = None
        _actual_frame_h, _actual_frame_w = result["height"], result["width"]
        for attempt in range(3):
            seek_to = max(0, fno - attempt * 8)
            cap.set(_cv2.CAP_PROP_POS_FRAMES, seek_to)
            _tmp_frame = None
            for _ in range(fno - seek_to + 1):
                ret, f = cap.read()
                if ret:
                    _tmp_frame = f
            if _tmp_frame is not None:
                # 记录实际帧尺寸（以真实解码帧为准，防止与缓存中存储的视频元信息不一致）
                _actual_frame_h, _actual_frame_w = _tmp_frame.shape[:2]
                ok_enc, buf = _cv2.imencode('.jpg', _tmp_frame, [_cv2.IMWRITE_JPEG_QUALITY, 88])
                if ok_enc and len(buf) > 3000:
                    frame_buf = buf
                    break  # 编码成功且大小合理

        if frame_buf is None:
            continue  # 跳过此帧，对应 frames_data 不追加，保持索引一致

        fname = f"{video_id}_phase_a_frame_{fi}.jpg"
        fpath = os.path.join(tmp_dir, fname)
        with open(fpath, 'wb') as _wf:
            _wf.write(frame_buf.tobytes())

        # 从 Phase A 缓存的 team_hsv_centers 生成帧代表色（不重新聚类，避免 team_a/b 标签翻转）
        import numpy as _np

        dets = _dedup_overlapping_dets(frame_info["detections"], iou_threshold=0.35)
        # 直接使用 Phase A 已确定的 team 字段，只需把 HSV 中心转为 RGB 用于前端显示
        cached_centers = result.get("team_hsv_centers", {})  # {"team_a":[H,S], "team_b":[H,S]}

        def _hsv_to_rgb(h, s):
            px = _np.uint8([[[int(h), int(s), 180]]])
            bgr = _cv2.cvtColor(px, _cv2.COLOR_HSV2BGR)[0][0]
            return [int(bgr[2]), int(bgr[1]), int(bgr[0])]

        tc = {"team_a": [128, 128, 128], "team_b": [70, 130, 255]}
        if "team_a" in cached_centers and "team_b" in cached_centers:
            try:
                tc = {
                    "team_a": _hsv_to_rgb(*cached_centers["team_a"]),
                    "team_b": _hsv_to_rgb(*cached_centers["team_b"]),
                }
            except Exception:
                pass

        frames_data.append({
            "frame_url": f"/tmp_frames/{fname}",
            "frame_time": round(fno / fps, 1),
            "player_count": len(dets),
            "detections": dets,
            "team_colors": tc,
            "frame_w": _actual_frame_w,
            "frame_h": _actual_frame_h,
        })

    cap.release()

    # 兼容旧版字段 + 全局代表色用第一帧
    global_tc = frames_data[0]["team_colors"] if frames_data else {"team_a": [128, 128, 128], "team_b": [128, 128, 128]}

    return {
        "video_id": video_id,
        "frame_w": result["width"],
        "frame_h": result["height"],
        "frames": frames_data,
        "team_colors": global_tc,
        "all_tracks": result["all_tracks"],
        # 向下兼容
        "frame_url": frames_data[0]["frame_url"] if frames_data else "",
        "frame_time": frames_data[0]["frame_time"] if frames_data else 0,
        "detections": frames_data[0]["detections"] if frames_data else [],
    }


@app.post("/start_analysis_v2/{video_id}")
async def start_analysis_v2(video_id: str, request: Request,
                             background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Phase B：前端标注完成后调用。
    请求体：{
        "track_jersey_map": {tid: {"jersey": int, "team": str}, ...},
        "excluded_track_ids": [tid, ...],
        "team_swap": bool
    }
    """
    video = db.query(VideoModel).filter(VideoModel.video_uuid == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="视频记录不存在")

    body = await request.json()
    track_jersey_map = body.get("track_jersey_map", {})
    excluded_track_ids = set(body.get("excluded_track_ids", []))
    non_player_track_ids = set(body.get("non_player_track_ids", []))
    team_swap = bool(body.get("team_swap", False))

    # team_swap 处理
    if team_swap and track_jersey_map:
        swap_map = {"team_a": "team_b", "team_b": "team_a"}
        for tid, info in track_jersey_map.items():
            info["team"] = swap_map.get(info.get("team", "team_a"), info.get("team", "team_a"))

    file_path = os.path.join(UPLOAD_DIR, video.filename)
    abs_file_path = os.path.abspath(file_path)

    # 校验 Phase A 缓存是否存在，未完成则返回明确错误而不是让 Phase B 静默失败
    cache_path = analysis._get_phase_a_cache_path(video_id)
    if not os.path.exists(cache_path):
        phase_a_prog = analysis.phase_a_progress_store.get(video_id, -1)
        return {
            "status": "phase_a_not_ready",
            "video_id": video_id,
            "phase_a_progress": phase_a_prog,
            "message": "Phase A 追踪缓存不存在，请先完成标注弹窗流程（弹窗会自动等待 Phase A 完成）"
        }

    # 同时保存到旧 store 以兼容旧代码
    jersey_annotations = []
    for tid, info in track_jersey_map.items():
        jersey_val = info.get("jersey")
        if jersey_val is None:
            continue  # 防御：跳过 jersey 为 None 的条目，避免 int(None) 抛 TypeError
        jersey_annotations.append({
            "det_id": int(tid),
            "jersey_number": int(jersey_val),
            "team": info.get("team", "team_a"),
        })
    non_player_bboxes = [{"det_id": int(t)} for t in excluded_track_ids]

    # 防止 Phase B 重复启动（Phase B 运行中进度为 1~94）
    current_progress = analysis.progress_store.get(video_id, -1)
    if 0 < current_progress < 95:
        return {"status": "already_started", "video_id": video_id}

    _jersey_annotation_store[video_id] = jersey_annotations
    _non_player_store[video_id] = non_player_bboxes

    analysis.progress_store[video_id] = 1
    background_tasks.add_task(
        analysis.process_video_phase_b,
        abs_file_path, video_id,
        jersey_annotations=jersey_annotations,
        non_player_bboxes=non_player_bboxes,
        excluded_track_ids=excluded_track_ids,
        non_player_track_ids=non_player_track_ids,
        track_jersey_map=track_jersey_map,
    )

    return {"status": "started", "video_id": video_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)
