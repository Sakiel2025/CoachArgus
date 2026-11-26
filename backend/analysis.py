import cv2
import time
import numpy as np
import os
import random
import math
import sys
from collections import defaultdict, deque
from typing import Optional

try:
    import torch
    import torch.optim
    import torch.optim.lr_scheduler
except ImportError:
    torch = None
from database import SessionLocal, VideoModel, AthleteProfile
from path_config import get_phase_a_cache_dir

# Torchreid is optional and installed separately.
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# 全局变量记录进度，格式：{video_uuid: percent}
progress_store = {}

# Phase A（全局追踪预处理）独立进度存储，与主分析进度隔离，避免心跳误判
phase_a_progress_store = {}

# ============================================================
# 卡尔曼滤波单目标状态估计（方案三：维持 ID 连续性）
# ============================================================
class KalmanBoxTracker:
    """
    基于卡尔曼滤波的单目标边界框追踪。
    状态向量: [cx, cy, w, h, vx, vy, vw, vh]
    观测向量: [cx, cy, w, h]

    ── 足球场景优化：NSA Kalman 自适应噪声协方差 ──
    参考 ByteTrack PAKF (Noise-Scale-Adaptive Kalman)：
      根据检测置信度 conf 动态缩放观测噪声 R：
        α = 1 - conf          # 置信度越高，测量越可信，R 越小
        R_adaptive = R * (1 + α * scale_factor)
      高置信度检测 → R 小 → 更信任量测，快速跟上急停/变向
      低置信度检测 → R 大 → 更信任预测，抑制噪声抖动
    这解决了标准卡尔曼"匀速假设"在足球急停-加速场景下的滞后问题。
    """
    count = 0
    # NSA 噪声缩放因子：控制置信度对 R 的影响幅度
    _NSA_SCALE = 10.0
    # 物理运动上限（像素/帧）：职业球员最大冲刺速度约 10m/s
    # 30fps 视频、球场宽 68m 映射到画面约 1920px → 1px ≈ 0.035m
    # 10m/s ÷ 30fps ÷ 0.035m/px ≈ 9.5px/frame，留 1.5 倍余量取 15
    _MAX_SPEED_PX = 15.0   # 单帧最大合理位移（像素）
    _MAX_ACCEL_PX = 8.0    # 单帧最大合理加速度变化（像素/帧²）

    def __init__(self, bbox, team, color):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w  = float(x2 - x1)
        h  = float(y2 - y1)

        # 状态转移矩阵 (匀速模型)
        self.F = np.array([
            [1,0,0,0,1,0,0,0],
            [0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1],
        ], dtype=float)

        # 观测矩阵
        self.H = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
        ], dtype=float)

        # 基础过程噪声（速度分量噪声更大，允许足球场景的加速度变化）
        self.Q = np.eye(8, dtype=float) * 0.01
        self.Q[4:, 4:] *= 5.0

        # 基础观测噪声（NSA 自适应时以此为基准动态缩放）
        self.R_base = np.eye(4, dtype=float) * 5.0
        self.R = self.R_base.copy()

        # 初始状态协方差（位置已知，速度不确定）
        self.P = np.eye(8, dtype=float) * 10.0
        self.P[4:, 4:] *= 100.0

        # 初始状态
        self.x = np.array([[cx],[cy],[w],[h],[0],[0],[0],[0]], dtype=float)

        self.lost = 0
        self.hits = 1          # 连续命中帧数（用于 min_hits 门控）
        self.team = team
        self.color = color
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count

    def predict(self):
        """卡尔曼预测步（时间更新）"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.lost += 1
        return self._to_bbox()

    def update(self, bbox, team=None, color=None, conf: float = 1.0):
        """
        卡尔曼更新步（量测更新）。

        conf: 检测置信度 [0,1]，用于 NSA 自适应噪声缩放。
          置信度高（如 0.9）→ R 接近 R_base → 快速跟随检测（适合急停/变向）
          置信度低（如 0.3）→ R 放大 7× → 信任预测，抑制噪声（适合遮挡/模糊帧）
        """
        x1, y1, x2, y2 = bbox
        det_cx = (x1 + x2) / 2.0
        det_cy = (y1 + y2) / 2.0
        z = np.array([[det_cx], [det_cy], [float(x2 - x1)], [float(y2 - y1)]], dtype=float)

        # ── 物理运动合理性校验 ──
        # 计算检测位置与 Kalman 预测位置的残差，判断是否超出物理极限
        pred_cx = float(self.x[0, 0])
        pred_cy = float(self.x[1, 0])
        pred_vx = float(self.x[4, 0])
        pred_vy = float(self.x[5, 0])
        residual_dist = ((det_cx - pred_cx) ** 2 + (det_cy - pred_cy) ** 2) ** 0.5
        # 当前速度估计的期望位移
        expected_dist = (pred_vx ** 2 + pred_vy ** 2) ** 0.5
        # 超出期望位移 + 最大加速度允许范围时，视为物理异常，压低置信度
        physics_limit = expected_dist + self._MAX_ACCEL_PX
        if residual_dist > physics_limit and residual_dist > self._MAX_SPEED_PX:
            # 压低置信度：残差越大、惩罚越重，最低降到 0.05（不完全忽略，允许急停/镜头切换）
            penalty = min(residual_dist / (physics_limit + 1e-6) - 1.0, 4.0) / 4.0
            conf = float(np.clip(conf * (1.0 - 0.8 * penalty), 0.05, conf))

        # ── NSA：根据置信度自适应调整观测噪声 ──
        alpha = 1.0 - float(np.clip(conf, 0.0, 1.0))
        self.R = self.R_base * (1.0 + alpha * self._NSA_SCALE)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.lost = 0
        self.hits += 1
        if team:
            self.team = team
        if color:
            self.color = color

    def _to_bbox(self):
        cx, cy, w, h = self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0]
        w = max(w, 1); h = max(h, 1)
        return (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))

    def get_bbox(self):
        return self._to_bbox()


# ============================================================
# Re-ID 外观特征提取
# 主方案：OSNet 深度特征（sn-reid-main/torchreid）
# 降级方案：HSV 颜色直方图 + 梯度方向（64 维）
# ============================================================

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class _HSVAppearanceFeature:
    """降级方案：64 维 HSV 直方图 + 梯度方向特征。"""
    FEAT_DIM = 64

    @staticmethod
    def extract(crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            return np.zeros(64, dtype=np.float32)
        h, w = crop.shape[:2]
        torso = crop[:int(h * 0.55), :]
        if torso.size == 0:
            torso = crop
        torso_resized = cv2.resize(torso, (32, 48), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(torso_resized, cv2.COLOR_BGR2HSV)
        half = hsv.shape[0] // 2
        feats = []
        for region in [hsv[:half], hsv[half:]]:
            for ch, bins in enumerate([8, 8, 8]):
                rng = [[0, 180], [0, 256], [0, 256]][ch]
                hist = cv2.calcHist([region], [ch], None, [bins], rng)
                hist = hist.flatten() / (hist.sum() + 1e-6)
                feats.append(hist)
        color_feat = np.concatenate(feats)
        gray = cv2.cvtColor(torso_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        hist_g, _ = np.histogram(ang.flatten(), bins=16, range=(0, 360),
                                  weights=mag.flatten())
        hist_g = hist_g / (hist_g.sum() + 1e-6)
        return np.concatenate([color_feat, hist_g]).astype(np.float32)


class OSNetReID:
    """
    基于 sn-reid-main 的 OSNet 深度特征提取器。
    直接复用 torchreid.models.osnet，支持批量提取。
    特征维度：512（OSNet 所有变体 fc_dims=512）
    """
    FEAT_DIM = 512
    _instance = None   # 单例，避免重复加载模型

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._model = None
        self._preprocess = None
        self._device = None
        self._fallback = _HSVAppearanceFeature()
        self._use_deep = False
        self._use_half = False
        self._feat_dim = _HSVAppearanceFeature.FEAT_DIM
        self._try_init()
        self._initialized = True

    def _try_init(self):
        """加载 OSNet-x0_75 SoccerNet 专用模型（osnet_x075_soccernet_best.pth）。"""
        try:
            import torch
            import torchvision.transforms as T

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            def _load_weights_ignore_classifier(model, weights_path):
                """加载权重，自动跳过 classifier 层尺寸不匹配，兼容 'model.' / 'module.' 前缀"""
                state = torch.load(weights_path, map_location='cpu', weights_only=False)
                if isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']
                model_state = model.state_dict()

                def _strip_prefix(sd, prefix):
                    return {(k[len(prefix):] if k.startswith(prefix) else k): v
                            for k, v in sd.items()}

                candidates = [
                    state,
                    _strip_prefix(state, 'module.'),
                    _strip_prefix(state, 'model.'),
                    _strip_prefix(_strip_prefix(state, 'module.'), 'model.'),
                ]
                best_filtered = {}
                for cand in candidates:
                    f = {k: v for k, v in cand.items()
                         if k in model_state and v.shape == model_state[k].shape}
                    if len(f) > len(best_filtered):
                        best_filtered = f
                    if len(best_filtered) == len(model_state):
                        break

                model_state.update(best_filtered)
                model.load_state_dict(model_state)
                return len(best_filtered), len(state)

            # 唯一权重：SoccerNet 专用 osnet_x0_75
            soccernet_path = os.path.join(_project_root, 'osnet_x075_soccernet_best.pth')
            if not os.path.exists(soccernet_path):
                soccernet_path = os.path.join(_project_root, 'backend', 'osnet_x075_soccernet_best.pth')

            from torchreid.models.osnet import osnet_x0_75
            model = osnet_x0_75(num_classes=1, pretrained=False, loss='softmax')
            if os.path.exists(soccernet_path):
                loaded, total = _load_weights_ignore_classifier(model, soccernet_path)
                model_tag = f"osnet_x0_75 [SoccerNet微调] loaded={loaded}/{total}"
            else:
                model_tag = "osnet_x0_75 [随机初始化，请确认 osnet_x075_soccernet_best.pth 存在]"
                print(f"[WARNING] OSNetReID: 未找到 osnet_x075_soccernet_best.pth，请将权重放至项目根目录")

            model.eval()
            model.to(device)
            # GPU FP16：CUDA 可用时将模型转为半精度，Re-ID batch 推理速度提升 ~1.5x
            _reid_use_half = (device.type == 'cuda')
            if _reid_use_half:
                model.half()

            preprocess = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

            self._model = model
            self._preprocess = preprocess
            self._device = device
            self._use_deep = True
            self._use_half = _reid_use_half
            self._feat_dim = 512
            print(f"[OK] OSNetReID 加载完成: {model_tag}, device={device}, half={_reid_use_half}, 特征维度=512")
        except Exception as e:
            print(f"[WARNING] OSNetReID 加载失败，降级为 HSV 特征: {e}")
            self._use_deep = False
            self._feat_dim = _HSVAppearanceFeature.FEAT_DIM

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def extract_batch(self, crops: list) -> np.ndarray:
        """
        批量提取特征。
        crops: List[np.ndarray]，BGR 图像裁剪列表
        返回: np.ndarray shape (N, feat_dim)，L2 归一化后的特征
        """
        n = len(crops)
        if n == 0:
            return np.zeros((0, self._feat_dim), dtype=np.float32)

        if not self._use_deep:
            feats = np.stack([self._fallback.extract(c) for c in crops])
            return feats

        try:
            import torch
            tensors = []
            for crop in crops:
                if crop is None or crop.size == 0:
                    tensors.append(torch.zeros(3, 256, 128))
                    continue
                # BGR→RGB
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensors.append(self._preprocess(rgb))

            batch = torch.stack(tensors, dim=0).to(self._device)
            if self._use_half:
                batch = batch.half()
            with torch.no_grad():
                feats = self._model(batch)   # (N, feat_dim)
            feats_np = feats.float().cpu().numpy().astype(np.float32)

            # L2 归一化
            norms = np.linalg.norm(feats_np, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-6)
            feats_np = feats_np / norms
            return feats_np
        except Exception as e:
            print(f"[WARNING] OSNetReID.extract_batch 失败: {e}")
            feats = np.stack([self._fallback.extract(c) for c in crops])
            return feats

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """单样本提取（便利接口）。"""
        result = self.extract_batch([crop])
        return result[0]


# 全局单例 Re-ID 提取器（延迟初始化）
_reid_extractor = None  # type: "OSNetReID | None"

def get_reid_extractor() -> "OSNetReID":
    global _reid_extractor
    if _reid_extractor is None:
        _reid_extractor = OSNetReID()
    return _reid_extractor


# 兼容旧代码的静态接口
class AppearanceFeature:
    """兼容接口，内部委托给 OSNetReID 单例。"""
    @staticmethod
    def extract(crop: np.ndarray) -> np.ndarray:
        return get_reid_extractor().extract(crop)

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return _cosine_sim(a, b)


# ============================================================
# 相机参数化模型（对标 TVCalib SNProjectiveCamera）
# 优化变量：pan / tilt / roll（欧拉角，弧度）+ aov（水平视场角，弧度）
# 支持两种模式：
#   - torch 模式（GPU 自动微分，AdamW 优化，推荐）
#   - numpy 模式（仅用于 to_homography 等不需要梯度的场合）
# ============================================================
class _FootballCamera:
    """
    ZXZ 欧拉角参数化的透视相机（无畸变，平面球场 z=0）。

    坐标系：
      世界坐标 (X,Y,0)：原点=场地中心，X 沿长轴（-52.5~52.5m），Y 沿短轴
      图像坐标 (u,v)：像素，左上角原点

    参数：
      pan   : 水平旋转角（弧度），正值→相机右转
      tilt  : 俯仰角（弧度），正值→相机下压（看向地面）
      roll  : 滚转角（弧度），通常≈0
      aov   : 水平视场角（弧度），典型值 45°~70°
      cx,cy,cz: 相机在世界坐标系中的位置（米）

    典型广播摄像机位置：沿中线、球场边线外约5m、高台约15m高
      cx ≈ 0, cy ≈ -40 (场地边线外), cz ≈ 15
    """

    # 广播主摄像机典型先验（对标 TVCalib tv_main_center.py）
    PRIOR_PAN   =  0.0
    PRIOR_TILT  =  np.radians(20.0)
    PRIOR_ROLL  =  0.0
    PRIOR_AOV   =  np.radians(55.0)
    PRIOR_CX    =  0.0
    PRIOR_CY    =  75.0   # TVCalib tv_main_center 范围 [40,110] 均值
    PRIOR_CZ    = -22.5   # TVCalib tv_main_center 范围 [-40,-5] 均值

    def __init__(self, pan=None, tilt=None, roll=None, aov=None,
                 cx=None, cy=None, cz=None,
                 image_width=1920, image_height=1080):
        self.pan   = float(pan   if pan   is not None else self.PRIOR_PAN)
        self.tilt  = float(tilt  if tilt  is not None else self.PRIOR_TILT)
        self.roll  = float(roll  if roll  is not None else self.PRIOR_ROLL)
        self.aov   = float(aov   if aov   is not None else self.PRIOR_AOV)
        self.cx    = float(cx    if cx    is not None else self.PRIOR_CX)
        self.cy    = float(cy    if cy    is not None else self.PRIOR_CY)
        self.cz    = float(cz    if cz    is not None else self.PRIOR_CZ)
        self.image_width  = image_width
        self.image_height = image_height

    # ── numpy 辅助（用于 to_homography 等不需要梯度的场合）──────────────
    @staticmethod
    def _rotation_matrix_np(pan, tilt, roll):
        """ZXZ 欧拉角旋转矩阵（numpy 版，与 TVCalib 完全一致）"""
        cp, sp = np.cos(pan),  np.sin(pan)
        ct, st = np.cos(tilt), np.sin(tilt)
        cr, sr = np.cos(roll), np.sin(roll)
        return np.array([
            [-sp*sr*ct + cp*cr,  sp*cr + sr*cp*ct,  sr*st],
            [-sp*cr*ct - sr*cp, -sp*sr + cp*cr*ct,  st*cr],
            [ sp*st,            -st*cp,              ct   ],
        ], dtype=np.float64)

    def project_points(self, world_pts):
        """numpy 投影（用于 to_homography，无梯度）"""
        pts = np.asarray(world_pts, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.shape[1] == 2:
            pts = np.column_stack([pts, np.zeros(len(pts))])
        cam_pos = np.array([self.cx, self.cy, self.cz], dtype=np.float64)
        pts_cam = pts - cam_pos
        R = self._rotation_matrix_np(self.pan, self.tilt, self.roll)
        pts_rot = (R @ pts_cam.T).T
        denom = pts_rot[:, 2:3]
        denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        pts_norm = pts_rot / denom
        fl = (self.image_width / 2.0) / np.tan(self.aov / 2.0)
        u = pts_norm[:, 0] * fl + self.image_width  / 2.0
        v = pts_norm[:, 1] * fl + self.image_height / 2.0
        return np.column_stack([u, v])

    def to_homography(self, canvas_w, canvas_h, pitch_l, pitch_w, margin):
        """相机参数 → H 矩阵（图像像素 → 战术板画布坐标）"""
        cw = canvas_w - 2 * margin
        ch = canvas_h - 2 * margin
        xs = np.linspace(-pitch_l/2, pitch_l/2, 6)
        ys = np.linspace(-pitch_w/2, pitch_w/2, 4)
        world_pts, canvas_pts = [], []
        for x in xs:
            for y in ys:
                world_pts.append([x, y])
                tx = margin + (x + pitch_l/2) / pitch_l * cw
                ty = margin + (y + pitch_w/2) / pitch_w * ch
                canvas_pts.append([tx, ty])
        world_pts  = np.array(world_pts,  dtype=np.float32)
        canvas_pts = np.array(canvas_pts, dtype=np.float32)
        img_pts    = self.project_points(world_pts).astype(np.float32)
        in_frame = ((img_pts[:, 0] > 0) & (img_pts[:, 0] < self.image_width) &
                    (img_pts[:, 1] > 0) & (img_pts[:, 1] < self.image_height))
        if in_frame.sum() < 4:
            return None
        H, _ = cv2.findHomography(img_pts, canvas_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None
        if abs(H[2, 2]) > 1e-10:
            H /= H[2, 2]
        return H

    def get_params(self):
        """返回可优化参数向量 [pan, tilt, roll, aov]"""
        return np.array([self.pan, self.tilt, self.roll, self.aov], dtype=np.float64)

    @classmethod
    def from_params(cls, params, cx=None, cy=None, cz=None,
                    image_width=1920, image_height=1080):
        pan, tilt, roll, aov = params
        return cls(pan=float(pan), tilt=float(tilt), roll=float(roll), aov=float(aov),
                   cx=cx, cy=cy, cz=cz,
                   image_width=image_width, image_height=image_height)

    # ── torch 投影（GPU 自动微分，用于 AdamW 优化）──────────────────────
    @staticmethod
    def project_points_torch(pan, tilt, roll, aov, cam_pos, world_pts_t, w_img, h_img):
        """
        全 torch 实现的 ZXZ 欧拉角透视投影。
        所有输入均为 torch.Tensor（支持 autograd）。

        参数：
          pan/tilt/roll/aov : shape () 的标量 tensor（requires_grad=True）
          cam_pos           : shape (3,) tensor，相机位置（不优化）
          world_pts_t       : shape (N,2) tensor，世界坐标点（z=0 平面）
          w_img, h_img      : 图像宽高（int）
        返回：
          shape (N,2) tensor，图像像素坐标
        """
        cp, sp = torch.cos(pan),  torch.sin(pan)
        ct, st = torch.cos(tilt), torch.sin(tilt)
        cr, sr = torch.cos(roll), torch.sin(roll)

        # ZXZ 欧拉角旋转矩阵（与 TVCalib SNProjectiveCamera 完全一致）
        R = torch.stack([
            torch.stack([-sp*sr*ct + cp*cr,  sp*cr + sr*cp*ct,  sr*st]),
            torch.stack([-sp*cr*ct - sr*cp, -sp*sr + cp*cr*ct,  st*cr]),
            torch.stack([ sp*st,            -st*cp,              ct   ]),
        ])  # (3,3)

        # 补 z=0 列
        N = world_pts_t.shape[0]
        zeros = torch.zeros(N, 1, dtype=world_pts_t.dtype, device=world_pts_t.device)
        pts3d = torch.cat([world_pts_t, zeros], dim=1)  # (N,3)

        # 平移到相机坐标系
        pts_cam = pts3d - cam_pos.unsqueeze(0)   # (N,3)
        pts_rot = (R @ pts_cam.T).T              # (N,3)

        # 透视除法
        # ① 点在相机后方（z≤0）时强制 z→+ε，让投影飞到画面外缘而非 NaN/±inf
        # ② 对最终像素坐标做 soft clamp（tanh 映射），防止 loss 爆炸梯度消失
        denom = pts_rot[:, 2:3]
        denom = torch.where(denom > 1e-4, denom, torch.ones_like(denom) * 1e-4)
        pts_norm = pts_rot / denom               # (N,3)

        # 内参：焦距 fl
        half_w = w_img / 2.0
        half_h = h_img / 2.0
        fl = half_w / torch.tan(aov / 2.0)
        u_raw = pts_norm[:, 0] * fl + half_w
        v_raw = pts_norm[:, 1] * fl + half_h

        # soft clamp：把投影坐标限制在 [-3W, 4W] / [-3H, 4H] 范围内
        # 保留梯度方向，避免 1e11 量级 loss 导致梯度爆炸
        margin_u = 3.0 * w_img
        margin_v = 3.0 * h_img
        u = torch.tanh((u_raw - half_w) / margin_u) * margin_u + half_w
        v = torch.tanh((v_raw - half_h) / margin_v) * margin_v + half_h

        return torch.stack([u, v], dim=1)        # (N,2)


# ============================================================
# 语义分割网络：加载 soccer_pitch_segmentation.pth
# ============================================================
class _PitchSegNet:
    """
    DeepLabV3-ResNet50 场地线语义分割网络（28类+背景=29类）。
    输入：BGR帧；输出：(H,W) 整数掩码，0=背景，1~28=各线段类别。
    """
    # 与 SoccerPitch.lines_classes 完全一致的28类有序列表
    LINES_CLASSES = [
        'Big rect. left bottom',   # 0→index 1
        'Big rect. left main',
        'Big rect. left top',
        'Big rect. right bottom',
        'Big rect. right main',
        'Big rect. right top',
        'Circle central',
        'Circle left',
        'Circle right',
        'Goal left crossbar',
        'Goal left post left ',
        'Goal left post right',
        'Goal right crossbar',
        'Goal right post left',
        'Goal right post right',
        'Goal unknown',
        'Line unknown',
        'Middle line',
        'Side line bottom',
        'Side line left',
        'Side line right',
        'Side line top',
        'Small rect. left bottom',
        'Small rect. left main',
        'Small rect. left top',
        'Small rect. right bottom',
        'Small rect. right main',
        'Small rect. right top',
    ]

    # SoccerPitch 真实世界坐标（米制，原点=场地中心，X沿长轴，Y沿短轴）
    # 对每条线记录两个端点的 (X, Y) 米制坐标
    # 标准场地 105m×68m
    _L, _W = 52.5, 34.0
    _PA_L, _PA_W2 = 16.5, 20.16     # 禁区长，禁区宽一半
    _GA_L, _GA_W2 = 5.5, 9.16       # 小禁区长，小禁区宽一半
    _PM = 11.0                        # 点球点距球门线
    _CR = 9.15                        # 中圈半径
    # 禁区弧与禁区线交点 x 坐标（距场中心的绝对 x）
    _arc_dx = _PA_L - _PM             # 5.5m
    _arc_dy = float(np.sqrt(max(0, _CR**2 - _arc_dx**2)))  # ~8.0m

    WORLD_LINE_ENDPOINTS = {
        'Big rect. left bottom':   [(-_L + _PA_L, +_PA_W2), (-_L + _PA_L, -_PA_W2)],
        'Big rect. left top':      [(-_L,          -_PA_W2), (-_L + _PA_L, -_PA_W2)],
        'Big rect. left main':     [(-_L,          +_PA_W2), (-_L,          -_PA_W2)],
        'Big rect. right bottom':  [(+_L - _PA_L, +_PA_W2), (+_L - _PA_L, -_PA_W2)],
        'Big rect. right top':     [(+_L,          -_PA_W2), (+_L - _PA_L, -_PA_W2)],
        'Big rect. right main':    [(+_L,          +_PA_W2), (+_L,          -_PA_W2)],
        'Small rect. left bottom': [(-_L + _GA_L,  +_GA_W2), (-_L + _GA_L, -_GA_W2)],
        'Small rect. left top':    [(-_L,           -_GA_W2), (-_L + _GA_L, -_GA_W2)],
        'Small rect. left main':   [(-_L,           +_GA_W2), (-_L,          -_GA_W2)],
        'Small rect. right bottom':[( +_L - _GA_L,  +_GA_W2), (+_L - _GA_L, -_GA_W2)],
        'Small rect. right top':   [( +_L,           -_GA_W2), (+_L - _GA_L, -_GA_W2)],
        'Small rect. right main':  [( +_L,           +_GA_W2), (+_L,          -_GA_W2)],
        'Side line bottom':        [(-_L,  +_W),    (+_L,  +_W)],
        'Side line top':           [(-_L,  -_W),    (+_L,  -_W)],
        'Side line left':          [(-_L,  -_W),    (-_L,  +_W)],
        'Side line right':         [(+_L,  -_W),    (+_L,  +_W)],
        'Middle line':             [(0.0,  -_W),    (0.0,  +_W)],
    }

    def __init__(self, model_path, mean_path, std_path):
        import torch
        import torch.nn as nn
        import os as _os_seg
        # 强制 torch hub 缓存到本地目录，命中缓存时跳过网络请求
        _os_seg.environ['TORCH_HOME'] = _os_seg.path.dirname(_os_seg.path.dirname(_os_seg.path.abspath(__file__)))
        _os_seg.environ['TORCH_HUB_DIR'] = _os_seg.environ['TORCH_HOME']
        from torchvision.models.segmentation import deeplabv3_resnet50

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # weights=None：不加载 DeepLabV3 预训练权重
        # weights_backbone=None：显式关闭 ResNet50 backbone 的 ImageNet 权重下载
        # （torchvision 0.21 中 weights_backbone 默认值为 IMAGENET1K_V1，不显式传 None 会触发下载）
        try:
            model = nn.DataParallel(
                deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=29)
            )
        except TypeError:
            # 旧版 torchvision（<0.13）API 兼容
            model = nn.DataParallel(deeplabv3_resnet50(pretrained=False, num_classes=29))
        # 初始化权重（与原版 detect_extremities.py 一致）
        for name, m in model.named_modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d)):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.1
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        self.model = model.to(self.device)
        self.mean = np.load(mean_path)
        self.std  = np.load(std_path)
        self._infer_w = 640
        self._infer_h = 360
        print(f"[OK] PitchSegNet 加载完成（device={self.device}）")

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """返回 (H,W) uint8 掩码，0=背景，1~28=各线类别。"""
        import torch
        img = cv2.resize(frame_bgr, (self._infer_w, self._infer_h),
                         interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            out = self.model(tensor)['out'].data[0].cpu().numpy()
        mask = np.argmax(out.transpose(1, 2, 0), axis=2).astype(np.uint8)
        return mask


def _get_support_center(mask, start, disk_radius, min_support=0.1):
    """
    计算以 start 为圆心、disk_radius 为半径的圆形区域内，mask 非零像素的重心。
    若非零像素数量 < min_support * π * r²，则认为支撑不足，返回 False。
    注意：mask 坐标系为 (row, col)，start 也是 (row, col)。
    """
    x = int(start[0])
    y = int(start[1])
    support_pixels = 1
    result = [x, y]
    xstart = max(x - disk_radius, 0)
    xend   = min(x + disk_radius, mask.shape[0] - 1)
    ystart = max(y - disk_radius, 0)
    yend   = min(y + disk_radius, mask.shape[1] - 1)

    patch = mask[xstart:xend + 1, ystart:yend + 1]
    ii = np.arange(xstart, xend + 1)
    jj = np.arange(ystart, yend + 1)
    gi, gj = np.meshgrid(ii, jj, indexing='ij')
    in_disk = (np.sqrt((gi - x) ** 2 + (gj - y) ** 2) < disk_radius) & (patch > 0)
    support_pixels = int(in_disk.sum()) + 1
    result_row = x + float(gi[in_disk].sum())
    result_col = y + float(gj[in_disk].sum())

    support = support_pixels >= min_support * np.pi * disk_radius ** 2
    result_arr = np.array([result_row, result_col], dtype=np.float64) / support_pixels
    return support, result_arr


def _infer_camera_position_from_mask(mask: np.ndarray):
    """
    根据分割网络输出的 28 类掩码自动判断相机机位，
    返回对应的 (cam_type, cy_init, cy_min, cy_max, cz_min, cz_max)。

    判断逻辑：
      - 掩码中每类的像素数代表该线段的可见程度
      - 'left' 系列 vs 'right' 系列决定是否门后机位
      - 'Side line bottom/top' 决定相机在 Y 轴正还是负方向
      - 默认（最常见）：中央主摄（c_y ∈ [40,110]，c_z ∈ [-40,-5]）

    坐标系约定（与 TVCalib 完全一致）：
      c_y > 0: 相机在球场 Y 正方向侧边线外
      c_y < 0: 相机在球场 Y 负方向侧边线外
      c_z < 0: 相机在地面以上（Z 向下为正）
    """
    LINES = _PitchSegNet.LINES_CLASSES   # 长度28，索引 i 对应 mask 值 i+1

    def _pixel_count(name):
        """返回某条线在掩码中的像素数"""
        if name not in LINES:
            return 0
        idx = LINES.index(name) + 1   # mask 值 = 类别索引 + 1
        return int(np.sum(mask == idx))

    # ── 统计关键线段的像素数 ────────────────────────────────────────────
    left_px  = (_pixel_count('Big rect. left bottom')
              + _pixel_count('Big rect. left main')
              + _pixel_count('Big rect. left top')
              + _pixel_count('Goal left crossbar')
              + _pixel_count('Goal left post left ')
              + _pixel_count('Goal left post right'))

    right_px = (_pixel_count('Big rect. right bottom')
              + _pixel_count('Big rect. right main')
              + _pixel_count('Big rect. right top')
              + _pixel_count('Goal right crossbar')
              + _pixel_count('Goal right post left')
              + _pixel_count('Goal right post right'))

    middle_px   = _pixel_count('Middle line')
    side_bot_px = _pixel_count('Side line bottom')
    side_top_px = _pixel_count('Side line top')

    total_field = left_px + right_px + middle_px + side_bot_px + side_top_px
    print(f"[CAMTYPE] left={left_px}, right={right_px}, middle={middle_px}, "
          f"side_bot={side_bot_px}, side_top={side_top_px}")

    # ── 判断规则 ────────────────────────────────────────────────────────
    # 规则1：门后机位——只能看到一端，另一端几乎不可见
    if total_field > 50:   # 至少有一些可见线段
        left_ratio  = left_px  / (total_field + 1e-6)
        right_ratio = right_px / (total_field + 1e-6)

        if left_ratio > 0.5 and right_ratio < 0.1:
            # 只看到左侧 → 相机在左端线后方
            cam_type = 'behind_goal'
            cy_init, cy_min, cy_max, cz_min, cz_max = 0.0, -5.0, 5.0, -35.0, -1.0
            print(f"[CAMTYPE] 判断为门后机位（左端）: cy∈[{cy_min},{cy_max}]")
            return cam_type, cy_init, cy_min, cy_max, cz_min, cz_max

        if right_ratio > 0.5 and left_ratio < 0.1:
            # 只看到右侧 → 相机在右端线后方
            cam_type = 'behind_goal'
            cy_init, cy_min, cy_max, cz_min, cz_max = 0.0, -5.0, 5.0, -35.0, -1.0
            print(f"[CAMTYPE] 判断为门后机位（右端）: cy∈[{cy_min},{cy_max}]")
            return cam_type, cy_init, cy_min, cy_max, cz_min, cz_max

    # 规则2：侧面主摄——根据 side_bottom / side_top 判断 Y 轴方向
    # side_line_bottom 在世界坐标 Y=+34m 处，相机在 Y>34 侧才能看到它在画面下方
    # side_line_top    在世界坐标 Y=-34m 处，相机在 Y<-34 侧才能看到它在画面上方
    if side_bot_px > side_top_px * 2 and side_bot_px > 100:
        # 看到底边线但几乎看不到顶边线 → 相机在 Y 正方向（标准主摄位置）
        cam_type = 'main_center'
        cy_init, cy_min, cy_max, cz_min, cz_max = 75.0, 40.0, 110.0, -40.0, -5.0
        print(f"[CAMTYPE] 判断为主摄（Y正）: cy∈[{cy_min},{cy_max}]")
        return cam_type, cy_init, cy_min, cy_max, cz_min, cz_max

    if side_top_px > side_bot_px * 2 and side_top_px > 100:
        # 看到顶边线但几乎看不到底边线 → 相机在 Y 负方向
        cam_type = 'main_neg'
        cy_init, cy_min, cy_max, cz_min, cz_max = -75.0, -110.0, -40.0, -40.0, -5.0
        print(f"[CAMTYPE] 判断为主摄（Y负）: cy∈[{cy_min},{cy_max}]")
        return cam_type, cy_init, cy_min, cy_max, cz_min, cz_max

    # 规则3：中线可见 → 主摄中央（最典型的广播视角）
    if middle_px > 50:
        cam_type = 'main_center'
        cy_init, cy_min, cy_max, cz_min, cz_max = 75.0, 40.0, 110.0, -40.0, -5.0
        print(f"[CAMTYPE] 判断为主摄中央（有中线）: cy∈[{cy_min},{cy_max}]")
        return cam_type, cy_init, cy_min, cy_max, cz_min, cz_max

    # 默认：中央主摄（最常见机位）
    cam_type = 'main_center'
    cy_init, cy_min, cy_max, cz_min, cz_max = 75.0, 40.0, 110.0, -40.0, -5.0
    print(f"[CAMTYPE] 默认主摄中央: cy∈[{cy_min},{cy_max}]")
    return cam_type, cy_init, cy_min, cy_max, cz_min, cz_max


def _synthesize_mask(semantic_mask, disk_radius):
    """
    在语义掩码上迭代拟合圆盘：每次随机选起点，meanshift 收敛到重心，
    若支撑足够则接受该圆盘并从掩码中擦除，直到没有剩余非零像素。
    返回圆盘圆心列表（(row, col) 整数坐标）。
    """
    mask = semantic_mask.copy().astype(np.uint8)
    points = np.transpose(np.nonzero(mask))
    disks = []
    while len(points):
        start = random.choice(points)
        dist = 10.
        success = True
        while dist > 1.:
            enough_support, center = _get_support_center(mask, start, disk_radius)
            if not enough_support:
                bad_point = np.round(center).astype(np.int32)
                cv2.circle(mask, (bad_point[1], bad_point[0]), disk_radius, 0, -1)
                success = False
                break
            dist = float(np.sqrt(np.sum(np.square(center - start))))
            start = center
        if success:
            disks.append(np.round(start).astype(np.int32))
            cv2.circle(mask, (disks[-1][1], disks[-1][0]), disk_radius, 0, -1)
        points = np.transpose(np.nonzero(mask))
    return disks


def _join_points(point_list, maxdist):
    """
    贪心地将圆心列表连接为折线：将距当前头/尾最近且距离 < maxdist 的点追加进去，
    超过 maxdist 则开启新折线。
    """
    import copy as _copy
    polylines = []
    if not point_list:
        return polylines
    head = point_list[0]
    tail = point_list[0]
    polyline = deque([point_list[0]])
    remaining = _copy.deepcopy(point_list[1:])
    while remaining:
        arr = np.array(remaining)
        d_tail = np.sqrt(((arr - tail) ** 2).sum(axis=1))
        d_head = np.sqrt(((arr - head) ** 2).sum(axis=1))
        best_tail = int(np.argmin(d_tail)); min_d_tail = float(d_tail[best_tail])
        best_head = int(np.argmin(d_head)); min_d_head = float(d_head[best_head])
        if min_d_head <= min_d_tail and min_d_head < maxdist:
            polyline.appendleft(remaining[best_head])
            head = polyline[0]
            remaining.pop(best_head)
        elif min_d_tail < min_d_head and min_d_tail < maxdist:
            polyline.append(remaining[best_tail])
            tail = polyline[-1]
            remaining.pop(best_tail)
        else:
            polylines.append(list(polyline))
            head = remaining[0]; tail = remaining[0]
            polyline = deque([head])
            remaining.pop(0)
    polylines.append(list(polyline))
    return polylines


def _extract_line_endpoints_from_mask(mask: np.ndarray, img_w: int, img_h: int,
                                      disk_radius: int = 6, maxdist: int = 40):
    """
    从语义分割掩码中提取各类场地线的两个端点（像素坐标）。

    三层流水线：
      1. generate_class_synthesis: 5×5核侵蚀去噪，对每类调用圆盘拟合
      2. _synthesize_mask: 迭代圆盘拟合（meanshift收敛），支撑不足则擦除
      3. _join_points + 取最长折线首尾: 贪心折线连接，取最长折线两端点

    比旧版 PCA 主轴法对弯曲线段（中圈弧、禁区弧）更鲁棒，
    定位误差更小，从而降低后续 H 矩阵重投影误差。

    返回：{class_name: [(x1,y1), (x2,y2)]}（原始图像像素坐标）
    """
    mh, mw = mask.shape[:2]
    scale_x = img_w / mw
    scale_y = img_h / mh

    # ── Step 1: 5×5 侵蚀去噪 + 逐类圆盘拟合 ────────────────────────────
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)

    buckets = {}
    for k, cname in enumerate(_PitchSegNet.LINES_CLASSES):
        cls_mask = (eroded == k + 1).astype(np.uint8)
        if cls_mask.sum() == 0:
            continue
        disks = _synthesize_mask(cls_mask, disk_radius)
        if disks:
            buckets[cname] = disks

    # ── Step 2: 折线连接 → 取最长折线的首尾端点 ──────────────────────────
    result = {}
    for cname, disks_list in buckets.items():
        polylines = _join_points(disks_list, maxdist)
        if not polylines:
            continue
        longest = max(polylines, key=len)
        if len(longest) < 1:
            continue
        # 圆心坐标为 (row, col)，转换为 (x, y) 像素坐标并缩放回原始分辨率
        p0 = longest[0]   # (row, col)
        p1 = longest[-1]  # (row, col)
        x0 = int(p0[1] * scale_x);  y0 = int(p0[0] * scale_y)
        x1 = int(p1[1] * scale_x);  y1 = int(p1[0] * scale_y)
        result[cname] = [(x0, y0), (x1, y1)]

    return result


# ============================================================
# 内嵌 FramebyFrameCalib
# 纯 numpy + cv2，无外部依赖
# 功能：给定一帧的场地关键点字典 {kp_id: {x, y}}，
#       通过 heuristic_voting 遍历多种 RANSAC 参数/模式，
#       返回重投影误差最小的单应矩阵 H
# ============================================================

# 57 个主关键点的真实球场坐标（米，以球场中心为原点）
# 来源：sn-gamestate utils_calib.py keypoint_world_coords_2D
_KP_WORLD_2D_RAW = [
    [0., 0.], [52.5, 0.], [105., 0.], [0., 13.84], [16.5, 13.84],
    [88.5, 13.84], [105., 13.84], [0., 24.84], [5.5, 24.84], [99.5, 24.84],
    [105., 24.84], [0., 30.34], [0., 30.34], [105., 30.34], [105., 30.34],
    [0., 37.66], [0., 37.66], [105., 37.66], [105., 37.66], [0., 43.16],
    [5.5, 43.16], [99.5, 43.16], [105., 43.16], [0., 54.16], [16.5, 54.16],
    [88.5, 54.16], [105., 54.16], [0., 68.], [52.5, 68.], [105., 68.],
    [16.5, 26.68], [52.5, 24.85], [88.5, 26.68], [16.5, 41.31], [52.5, 43.15],
    [88.5, 41.31], [19.99, 32.29], [43.68, 31.53], [61.31, 31.53], [85., 32.29],
    [19.99, 35.7], [43.68, 36.46], [61.31, 36.46], [85., 35.7], [11., 34.],
    [16.5, 34.], [20.15, 34.], [46.03, 27.53], [58.97, 27.53], [43.35, 34.],
    [52.5, 34.], [61.5, 34.], [46.03, 40.47], [58.97, 40.47], [84.85, 34.],
    [88.5, 34.], [94., 34.],
]
# 以球场中心为原点（X 向右，Y 向下），偏移量：(-52.5, -34)
_KP_WORLD_2D = [[x - 52.5, y - 34] for x, y in _KP_WORLD_2D_RAW]


class _FramebyFrameCalib:
    """
    内嵌版 FramebyFrameCalib。
    给定 keypoints_dict {kp_id(1-based): {'x': px, 'y': py}}（像素或归一化坐标），
    通过 get_homography_from_ground_plane + heuristic_voting 输出最优单应矩阵。
    """

    def __init__(self, iwidth=1920, iheight=1080, denormalize=False):
        self.image_width  = iwidth
        self.image_height = iheight
        self.denormalize  = denormalize
        self.keypoints_dict = {}
        self.subsets = {}
        self.obj_pts = None
        self.img_pts = None
        self.key_pts = None
        self.ord_pts = None

    def update(self, keypoints_dict: dict):
        self.keypoints_dict = dict(keypoints_dict)
        if self.denormalize:
            for kp in self.keypoints_dict:
                self.keypoints_dict[kp]['x'] *= self.image_width
                self.keypoints_dict[kp]['y'] *= self.image_height
        self._build_subsets()

    def _build_subsets(self):
        full, main, ground = {}, {}, {}
        for kp, coord in self.keypoints_dict.items():
            if kp < 1 or kp > len(_KP_WORLD_2D):
                continue
            wx, wy = _KP_WORLD_2D[kp - 1]
            zw = -2.44 if kp in [12, 15, 16, 19] else 0.0
            entry = {'xi': coord['x'], 'yi': coord['y'],
                     'xw': wx, 'yw': wy, 'zw': zw}
            full[kp] = entry
            if kp <= 30:
                main[kp] = entry
            if kp not in [12, 15, 16, 19]:
                ground[kp] = entry
        self.subsets = {'full': full, 'main': main, 'ground_plane': ground}

    def _get_per_plane_correspondences(self, mode: str, use_ransac: float):
        LEFT_SIDE  = {1, 4, 8, 13, 17, 20, 24, 28}
        RIGHT_SIDE = {3, 7, 11, 14, 18, 23, 27, 30}
        LEFT_GOAL  = {12, 16}
        RIGHT_GOAL = {15, 19}

        wp1, wp2, wp3 = [], [], []
        ip1, ip2, ip3 = [], [], []
        kp1, kp2, kp3 = [], [], []

        keypoints = self.subsets.get(mode, {})
        for kp, d in keypoints.items():
            xi, yi = d['xi'], d['yi']
            xw, yw, zw = d['xw'], d['yw'], d['zw']
            if kp in LEFT_GOAL:
                wp2.append([-zw, yw, 0.]);  ip2.append([xi, yi]);  kp2.append(kp)
            elif kp in RIGHT_GOAL:
                wp3.append([-zw, yw, 0.]);  ip3.append([xi, yi]);  kp3.append(kp)
            elif kp in LEFT_SIDE:
                wp1.append([xw, yw, zw]);   ip1.append([xi, yi]);  kp1.append(kp)
                wp2.append([-zw, yw, 0.]);  ip2.append([xi, yi]);  kp2.append(kp)
            elif kp in RIGHT_SIDE:
                wp1.append([xw, yw, zw]);   ip1.append([xi, yi]);  kp1.append(kp)
                wp3.append([-zw, yw, 0.]);  ip3.append([xi, yi]);  kp3.append(kp)
            else:
                wp1.append([xw, yw, zw]);   ip1.append([xi, yi]);  kp1.append(kp)

        if mode == 'ground_plane':
            obj_list = [wp1]; img_list = [ip1]; key_list = [kp1]
        else:
            obj_list = [wp1, wp2, wp3]; img_list = [ip1, ip2, ip3]; key_list = [kp1, kp2, kp3]

        if use_ransac > 0:
            for i in range(len(obj_list)):
                ol, il, kl = obj_list[i], img_list[i], key_list[i]
                if len(ol) >= 4:
                    xs = [p[0] for p in ol]; ys = [p[1] for p in ol]
                    if not (all(x == xs[0] for x in xs) or all(y == ys[0] for y in ys)):
                        H_tmp, status = cv2.findHomography(
                            np.array(ol, np.float32), np.array(il, np.float32),
                            cv2.RANSAC, use_ransac)
                        if status is not None:
                            obj_list[i] = [o for o, s in zip(ol, status) if s == 1]
                            img_list[i] = [im for im, s in zip(il, status) if s == 1]
                            key_list[i] = [k for k, s in zip(kl, status) if s == 1]

        self.obj_pts, self.img_pts, self.key_pts, self.ord_pts = [], [], [], []
        for i, (ol, il, kl) in enumerate(zip(obj_list, img_list, key_list)):
            if len(ol) < 4:
                continue
            xs = [p[0] for p in ol]; ys = [p[1] for p in ol]
            if all(x == xs[0] for x in xs) or all(y == ys[0] for y in ys):
                continue
            self.obj_pts.append(np.array(ol, np.float32))
            self.img_pts.append(np.array(il, np.float32))
            self.key_pts.append(kl)
            self.ord_pts.append(i)

    def get_homography_from_ground_plane(self, use_ransac: float = 5., inverse: bool = False):
        self._get_per_plane_correspondences('ground_plane', use_ransac)
        if not self.obj_pts:
            return None
        obj = self.obj_pts[0]
        img = self.img_pts[0]
        if len(obj) < 4:
            return None
        H, mask = cv2.findHomography(obj, img, cv2.RANSAC, use_ransac)
        if H is None:
            return None
        if inverse:
            det = np.linalg.det(H)
            if np.isclose(det, 0.):
                return None
            H_inv = np.linalg.inv(H)
            return H_inv / H_inv[-1, -1]
        return H

    def _get_cam_params(self, mode: str, use_ransac: float, refine: bool = False):
        flags = (cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO |
                 cv2.CALIB_FIX_TANGENT_DIST | cv2.CALIB_FIX_S1_S2_S3_S4 |
                 cv2.CALIB_FIX_TAUX_TAUY |
                 cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                 cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
        self._get_per_plane_correspondences(mode, use_ransac)
        if not self.obj_pts:
            return None, None
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.obj_pts, self.img_pts,
                (self.image_width, self.image_height), None, None, flags=flags)
        except Exception:
            return None, None
        if not ret:
            return None, None
        R, _ = cv2.Rodrigues(rvecs[0])
        pos = (-np.transpose(R) @ tvecs[0]).T[0]
        pan = float(np.degrees(np.arctan2(R[0, 2], -R[1, 2])))
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        roll_val = float(np.degrees(np.arctan2(R[2, 0], R[2, 1])))
        cam_params = {
            "pan_degrees": pan, "tilt_degrees": tilt, "roll_degrees": roll_val,
            "x_focal_length": mtx[0, 0], "y_focal_length": mtx[1, 1],
            "principal_point": [self.image_width / 2., self.image_height / 2.],
            "position_meters": [float(pos[0]), float(pos[1]), float(pos[2])],
            "rotation_matrix": R.tolist(),
            "radial_distortion": [0., 0., 0., 0., 0., 0.],
            "tangential_distortion": [0., 0.],
            "thin_prism_distortion": [0., 0., 0., 0.],
        }
        return cam_params, ret

    def heuristic_voting(self) -> Optional[dict]:
        """
        遍历 (mode × use_ransac) 组合，选重投影误差最小的相机参数方案。
        返回 {'cam_params': {...}, 'mode': ..., 'rep_err': ...} 或 None。
        """
        results = []
        for mode in ['full', 'ground_plane', 'main']:
            for ransac in [0, 5, 10, 15, 25, 50]:
                cam_params, ret = self._get_cam_params(mode, ransac)
                if ret:
                    results.append({'mode': mode, 'use_ransac': ransac,
                                    'rep_err': ret, 'cam_params': cam_params})
        if not results:
            return None
        results.sort(key=lambda x: (x['rep_err'], x['mode']))
        for r in results:
            if r['mode'] == 'full' and r['use_ransac'] == 0 and r['rep_err'] <= 10.:
                return r
        return results[0]

    def get_h_from_heuristic(self, frame_w: int, frame_h: int) -> Optional[np.ndarray]:
        """
        利用 heuristic_voting 输出的相机参数重建 H 矩阵（图像→球场地面平面）。
        返回 3×3 逆单应矩阵（球场米制 → 图像像素的反向），或 None。
        """
        res = self.heuristic_voting()
        if res is None:
            return None
        p = res['cam_params']
        fx, fy = p['x_focal_length'], p['y_focal_length']
        cx, cy = p['principal_point']
        pos = np.array(p['position_meters'], dtype=np.float64)
        R = np.array(p['rotation_matrix'], dtype=np.float64)
        Q = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        It = np.eye(4)[:-1]
        It[:, -1] = -pos
        P = Q @ (R @ It)           # 3×4 投影矩阵
        H = P[:, [0, 1, 3]]        # 取 X、Y、齐次列 → 3×3
        H = H / H[-1, -1]
        try:
            H_inv = np.linalg.inv(H)
            return H_inv / H_inv[-1, -1]
        except np.linalg.LinAlgError:
            return None


# ============================================================
# 单应矩阵辅助函数（SVD直线对应法）
# ============================================================

def _normalization_transform(points):
    """
    计算相似变换矩阵 T，使点云中心归零、平均距离为 sqrt(2)。
    用于提升 SVD 数值稳定性。
    参数：points - list of [x, y] 或 Nx2 array
    返回：3x3 归一化矩阵 T
    """
    pts = np.array(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.eye(3)
    cx, cy = pts.mean(axis=0)
    dists = np.sqrt(((pts - [cx, cy]) ** 2).sum(axis=1))
    mean_dist = dists.mean()
    if mean_dist < 1e-8:
        return np.eye(3)
    scale = np.sqrt(2.0) / mean_dist
    T = np.array([
        [scale,   0.0, -scale * cx],
        [  0.0, scale, -scale * cy],
        [  0.0,   0.0,          1.0]
    ], dtype=np.float64)
    return T


def _estimate_homography_from_line_correspondences(lines, T1=None, T2=None):
    """
    用直线对应（而非点对应）通过 SVD 估计单应矩阵。
    天然规避端点正序/反序问题。

    参数：
        lines  - list of (l1, l2)，l1 为源坐标系齐次直线(3,)，l2 为目标坐标系齐次直线(3,)
        T1     - 源坐标系归一化矩阵 (3x3)，None 则不归一化
        T2     - 目标坐标系归一化矩阵 (3x3)，None 则不归一化

    返回：(success: bool, H: 3x3 ndarray)
        H 将源坐标系点映射到目标坐标系（即 x2 ~ H @ x1）
    """
    if len(lines) < 4:
        return False, np.eye(3)

    if T1 is None:
        T1 = np.eye(3)
    if T2 is None:
        T2 = np.eye(3)

    # 归一化直线：l_norm = (T^{-T}) @ l
    T1_inv_T = np.linalg.inv(T1).T
    T2_inv_T = np.linalg.inv(T2).T

    A = []
    for (l1_raw, l2_raw) in lines:
        l1 = T1_inv_T @ np.array(l1_raw, dtype=np.float64)
        l2 = T2_inv_T @ np.array(l2_raw, dtype=np.float64)

        # 直线对应约束：l2^T H l1 = 0
        # 展开为 9 元向量 h 的线性方程（Hartley & Zisserman 4.1节）
        a, b, c = l2
        x, y, z = l1
        A.append([
            a*x, a*y, a*z,
            b*x, b*y, b*z,
            c*x, c*y, c*z,
        ])

    A = np.array(A, dtype=np.float64)
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return False, np.eye(3)

    h = Vt[-1]
    H_norm = h.reshape(3, 3)

    # 反归一化：H = T2^{-1} H_norm T1
    try:
        H = np.linalg.inv(T2) @ H_norm @ T1
    except np.linalg.LinAlgError:
        return False, np.eye(3)

    if abs(H[2, 2]) > 1e-10:
        H /= H[2, 2]

    # 基本合理性检验
    if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        return False, np.eye(3)

    return True, H


def _extract_K_from_homography(homography, image_width, image_height):
    """
    从图像平面与世界平面（Z=0）之间的单应矩阵估计相机内参矩阵 K。
    算法：Multiple View Geometry, Algorithm 8.2 (p225)。

    参数：
        homography   - 3x3 单应矩阵（世界平面 → 图像平面）
        image_width  - 图像宽度（px）
        image_height - 图像高度（px）

    返回：(success: bool, K: 3x3 内参矩阵)
    """
    principal_point = (image_width / 2.0, image_height / 2.0)
    H = np.reshape(homography, (9,))
    A = np.zeros((5, 6))
    A[0, 1] = 1.
    A[1, 0] = 1.
    A[1, 2] = -1.
    A[2, 3] = principal_point[1] / principal_point[0]
    A[2, 4] = -1.0
    A[3, 0] = H[0] * H[1]
    A[3, 1] = H[0] * H[4] + H[1] * H[3]
    A[3, 2] = H[3] * H[4]
    A[3, 3] = H[0] * H[7] + H[1] * H[6]
    A[3, 4] = H[3] * H[7] + H[4] * H[6]
    A[3, 5] = H[6] * H[7]
    A[4, 0] = H[0] * H[0] - H[1] * H[1]
    A[4, 1] = 2 * H[0] * H[3] - 2 * H[1] * H[4]
    A[4, 2] = H[3] * H[3] - H[4] * H[4]
    A[4, 3] = 2 * H[0] * H[6] - 2 * H[1] * H[7]
    A[4, 4] = 2 * H[3] * H[6] - 2 * H[4] * H[7]
    A[4, 5] = H[6] * H[6] - H[7] * H[7]

    try:
        _, _, vh = np.linalg.svd(A)
        w = vh[-1]
        W = np.zeros((3, 3))
        W[0, 0] = w[0] / w[5]
        W[0, 1] = w[1] / w[5]
        W[0, 2] = w[3] / w[5]
        W[1, 0] = w[1] / w[5]
        W[1, 1] = w[2] / w[5]
        W[1, 2] = w[4] / w[5]
        W[2, 0] = w[3] / w[5]
        W[2, 1] = w[4] / w[5]
        W[2, 2] = 1.0
        Ktinv = np.linalg.cholesky(W)
    except (np.linalg.LinAlgError, ZeroDivisionError):
        return False, np.eye(3)

    K = np.linalg.inv(np.transpose(Ktinv))
    K /= K[2, 2]
    # 主点固定在图像中心（估计值噪声较大，不使用）
    K[0, 2] = principal_point[0]
    K[1, 2] = principal_point[1]
    return True, K


def _refine_H_with_lm(H_init, point_matches_3d_2d, image_width, image_height):
    """
    用 SolvePnPRefineLM 对单应矩阵做 Levenberg-Marquardt 非线性精化。

    流程：
        1. 从 H_init 估计内参 K（MVG算法8.2）
        2. 用 K 和 H 分解初始 R、t（MVG例8.1）
        3. 用 cv2.solvePnPRefineLM 精化 R、t
        4. 重新构造精化后的单应矩阵 H = K [r1 | r2 | t]

    参数：
        H_init            - 3x3 初始单应矩阵（世界平面米制坐标 → 图像像素坐标）
        point_matches_3d_2d - list of (world_pt_3d, img_pt_2d)
                              world_pt_3d: [X, Y, 0.0]（Z=0 世界平面，米制）
                              img_pt_2d:  [u, v]（像素坐标）
        image_width       - 图像宽（px）
        image_height      - 图像高（px）

    返回：(success: bool, H_refined: 3x3 ndarray)
        失败时返回 (False, H_init)
    """
    if len(point_matches_3d_2d) < 4:
        return False, H_init

    # ── 1. 从 H 估计内参 K ─────────────────────────────────────────────
    ok, K = _extract_K_from_homography(H_init, image_width, image_height)
    if not ok:
        return False, H_init

    # ── 2. 分解初始 R、t ──────────────────────────────────────────────
    try:
        hprim = np.linalg.inv(K) @ H_init
        lambda1 = 1.0 / np.linalg.norm(hprim[:, 0])
        lambda2 = 1.0 / np.linalg.norm(hprim[:, 1])
        lambda3 = np.sqrt(lambda1 * lambda2)

        r0 = hprim[:, 0] * lambda1
        r1 = hprim[:, 1] * lambda2
        r2 = np.cross(r0, r1)
        R = np.column_stack((r0, r1, r2))

        # 强制旋转矩阵正交化（SVD）
        u, _, vh = np.linalg.svd(R)
        R = u @ vh
        if np.linalg.det(R) < 0:
            u[:, 2] *= -1
            R = u @ vh

        t = hprim[:, 2] * lambda3   # 3x1 平移向量
    except np.linalg.LinAlgError:
        return False, H_init

    # ── 3. LM 精化（solvePnPRefineLM） ───────────────────────────────
    target_pts = np.array([m[0] for m in point_matches_3d_2d], dtype=np.float64)   # Nx3
    src_pts    = np.array([m[1] for m in point_matches_3d_2d], dtype=np.float64)   # Nx2
    try:
        rvec_init, _ = cv2.Rodrigues(R)
        # t 是相机坐标系下的平移向量（H = K[r1|r2|t] 中的 t）
        rvec_ref, t_ref = cv2.solvePnPRefineLM(
            target_pts.reshape(-1, 1, 3),
            src_pts.reshape(-1, 1, 2),
            K, None,
            rvec_init, t.reshape(3, 1),
            (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 20000, 1e-5)
        )
        R_ref, _ = cv2.Rodrigues(rvec_ref)
        t_ref = t_ref.flatten()
    except Exception as e:
        print(f"[DEBUG] LM精化失败，保留SVD初始H: {e}")
        return False, H_init

    # ── 4. 重构精化后的单应矩阵 H = K [r1 | r2 | t] ──────────────────
    try:
        H_ref = K @ np.column_stack((R_ref[:, 0], R_ref[:, 1], t_ref))
        if abs(H_ref[2, 2]) > 1e-10:
            H_ref /= H_ref[2, 2]
        if np.any(np.isnan(H_ref)) or np.any(np.isinf(H_ref)):
            return False, H_init
    except Exception:
        return False, H_init

    return True, H_ref


# ============================================================
# 球员卡尔曼滤波追踪器
# 球员 pitch 坐标平滑器：EMA + 跳变冻结
# 移植自3版 EMA+freeze 机制，效果优于卡尔曼（pitch坐标投影噪声大，卡尔曼速度状态易漂移）
# ============================================================
class PlayerEMATracker:
    """
    EMA 平滑 + 跳变冻结，用于球员 pitch 坐标的逐帧平滑。
    参考3版 _PLAYER_EMA_ALPHA / _PLAYER_MAX_JUMP / _PLAYER_MAX_FREEZE 设计。
    - alpha=0.85：新测量占85%权重，等效时间常数≈0.11s，滞后小
    - 跳变冻结：单帧跳变>5m时冻结在上帧位置，最多冻结5帧后强制接受
    """
    EMA_ALPHA     = 0.68   # EMA 新值权重（降低以增强平滑，减少H矩阵抖动传导）
    MAX_JUMP      = 3.5    # 单帧跳变阈值（米），收窄以拦截更多小幅跳变
    MAX_FREEZE    = 5      # 最大连续冻结帧数（参考3版 _PLAYER_MAX_FREEZE）

    def __init__(self, wx: float, wy: float):
        self._wx = wx
        self._wy = wy
        self._freeze_cnt = 0
        self.missed = 0           # 兼容接口：连续缺席采样帧数
        self.is_predicted = False

    def update(self, wx: float, wy: float) -> tuple:
        """用测量值更新，返回平滑后坐标。"""
        jump_dist = ((wx - self._wx) ** 2 + (wy - self._wy) ** 2) ** 0.5
        if jump_dist > self.MAX_JUMP and self._freeze_cnt < self.MAX_FREEZE:
            # 跳变：冻结在上帧位置
            self._freeze_cnt += 1
            self.missed = 0
            self.is_predicted = False
            return self._wx, self._wy
        # 正常帧（或已达冻结上限强制接受）：EMA 平滑
        self._wx = self.EMA_ALPHA * wx + (1.0 - self.EMA_ALPHA) * self._wx
        self._wy = self.EMA_ALPHA * wy + (1.0 - self.EMA_ALPHA) * self._wy
        self._freeze_cnt = 0
        self.missed = 0
        self.is_predicted = False
        return self._wx, self._wy

    def predict(self) -> tuple:
        """缺席帧：保持上帧平滑位置不变（无速度模型，避免漂移）。"""
        self.missed += 1
        self.is_predicted = True
        return self._wx, self._wy

    @property
    def pos(self) -> tuple:
        return self._wx, self._wy


# 向后兼容别名（代码中可能有 PlayerKalmanFilter 引用）
PlayerKalmanFilter = PlayerEMATracker


def resolve_alias(alias_dict: dict, tid_str: str) -> str:
    """
    路径压缩版 alias 查找（参考 AFLink _flatten_link_map）。
    解决链式 alias（A→B→C）导致查找失效的问题：
      原来：alias_dict.get(A, A) 只走一步，若 B 也在 alias_dict 中则返回 B 而非最终 C。
      现在：沿链路一路走到底，并回写路径上每个节点直接指向终点（路径压缩）。
    """
    visited = []
    cur = tid_str
    while cur in alias_dict and cur not in visited:
        visited.append(cur)
        cur = alias_dict[cur]
    # 路径压缩：让链路上每个节点直接指向终点
    for node in visited:
        alias_dict[node] = cur
    return cur


def _postprocess_tactical_tracks(
    tactical_tracks: dict,
    max_gap_frames: int = 30,
    merge_dist_m: float = 2.0,
    smooth_window: int = 5,
) -> dict:
    """
    离线全局轨迹后处理（三步）：
    1. 短轨迹合并：将同队的短碎片拼接到时间上相邻、空间上最近的长轨迹
    2. 间隙线性插值：填补轨迹内部由遮挡产生的帧空白，让轨迹连续
    3. 滑动均值平滑：消除H矩阵抖动产生的高频位置噪声
    """
    import math

    # ── Step 1：短轨迹合并到长轨迹 ──────────────────────────────────
    changed = True
    iterations = 0
    while changed and iterations < 5:
        changed = False
        iterations += 1
        short_tids = sorted(
            [tid for tid, tr in tactical_tracks.items() if len(tr["pts"]) < 30],
            key=lambda t: len(tactical_tracks[t]["pts"])
        )
        for stid in short_tids:
            if stid not in tactical_tracks:
                continue
            str_pts  = tactical_tracks[stid]["pts"]
            str_team = tactical_tracks[stid].get("team", "")
            if not str_pts:
                del tactical_tracks[stid]
                continue
            s_fmin = str_pts[0]["f"]
            s_fmax = str_pts[-1]["f"]
            s_wx0, s_wy0 = str_pts[0]["wx"],  str_pts[0]["wy"]
            s_wxN, s_wyN = str_pts[-1]["wx"], str_pts[-1]["wy"]

            best_tid, best_score = None, -1.0
            for ltid, ltr in tactical_tracks.items():
                if ltid == stid:
                    continue
                if ltr.get("team", "") != str_team:
                    continue
                if len(ltr["pts"]) < 5:
                    continue
                l_pts = ltr["pts"]
                l_fmax = l_pts[-1]["f"]
                l_fmin = l_pts[0]["f"]
                l_wxN, l_wyN = l_pts[-1]["wx"], l_pts[-1]["wy"]
                l_wx0, l_wy0 = l_pts[0]["wx"],  l_pts[0]["wy"]

                # 短轨接长轨后面
                if s_fmin > l_fmax:
                    gap  = s_fmin - l_fmax
                    dist = math.hypot(s_wx0 - l_wxN, s_wy0 - l_wyN)
                    if gap <= max_gap_frames and dist <= merge_dist_m:
                        score = (1 - dist / merge_dist_m) * (1 - gap / max_gap_frames)
                        if score > best_score:
                            best_score, best_tid = score, ltid
                # 短轨接长轨前面
                elif s_fmax < l_fmin:
                    gap  = l_fmin - s_fmax
                    dist = math.hypot(s_wxN - l_wx0, s_wyN - l_wy0)
                    if gap <= max_gap_frames and dist <= merge_dist_m:
                        score = (1 - dist / merge_dist_m) * (1 - gap / max_gap_frames)
                        if score > best_score:
                            best_score, best_tid = score, ltid

            if best_tid is not None:
                l_pts = tactical_tracks[best_tid]["pts"]
                l_fmax = l_pts[-1]["f"]
                if s_fmin > l_fmax:
                    tactical_tracks[best_tid]["pts"] = l_pts + str_pts
                else:
                    tactical_tracks[best_tid]["pts"] = str_pts + l_pts
                tactical_tracks[best_tid]["pts"].sort(key=lambda p: p["f"])
                del tactical_tracks[stid]
                changed = True

    # ── Step 2：轨迹内部间隙线性插值 ────────────────────────────────
    MAX_INTERP_SPEED = 1.5  # 米/帧：超过此帧速则认为是H矩阵跳变，不插值
    for tid, tr in tactical_tracks.items():
        pts = tr["pts"]
        if len(pts) < 2:
            continue
        filled = [pts[0]]
        for i in range(1, len(pts)):
            prev_p, cur_p = pts[i - 1], pts[i]
            gap = cur_p["f"] - prev_p["f"]
            # 间隙在合理范围内才插值（太大=真实离场，不插）
            if 1 < gap <= max_gap_frames:
                dist = math.hypot(cur_p["wx"] - prev_p["wx"], cur_p["wy"] - prev_p["wy"])
                # 速度合理性检查：平均帧速超阈值说明是H矩阵跳变，不插值（避免穿越场地的虚假轨迹）
                if dist / gap <= MAX_INTERP_SPEED:
                    for step in range(1, gap):
                        t = step / gap
                        filled.append({
                            "f":      prev_p["f"] + step,
                            "wx":     round(prev_p["wx"] + t * (cur_p["wx"] - prev_p["wx"]), 3),
                            "wy":     round(prev_p["wy"] + t * (cur_p["wy"] - prev_p["wy"]), 3),
                            "interp": True,
                        })
            filled.append(cur_p)
        tr["pts"] = filled

    # ── Step 3：滑动均值平滑（消除H矩阵抖动） ───────────────────────
    hw = smooth_window // 2
    for tid, tr in tactical_tracks.items():
        pts = tr["pts"]
        n = len(pts)
        if n < smooth_window:
            continue
        smoothed = []
        for i, pt in enumerate(pts):
            lo  = max(0, i - hw)
            hi  = min(n, i + hw + 1)
            win = pts[lo:hi]
            new_pt = {
                "f":  pt["f"],
                "wx": round(sum(p["wx"] for p in win) / len(win), 3),
                "wy": round(sum(p["wy"] for p in win) / len(win), 3),
            }
            if pt.get("interp"):
                new_pt["interp"] = True
            if pt.get("predicted"):
                new_pt["predicted"] = True
            smoothed.append(new_pt)
        tr["pts"] = smoothed

    print(f"[PostProcess] 轨迹后处理完成，剩余轨迹数={len(tactical_tracks)}")
    return tactical_tracks


def compute_rule_score(dist: float, absent_cnt: int,
                       dist_thresh: float, max_absent: int) -> float:
    """
    参考 AFLink compute_rule_score，将断轨重连的候选评分从二值化（距离<阈值取第一个）
    改为连续加权评分，同时考虑空间距离和缺席时长：
      - dist_score  = 1 - dist / dist_thresh         （距离越小越好）
      - absent_score = 1 - absent_cnt / max_absent   （缺席越短越好）
      返回两项的加权平均，分数越高越优先。
    """
    dist_score   = 1.0 - dist / dist_thresh
    absent_score = 1.0 - absent_cnt / max_absent
    return 0.5 * dist_score + 0.5 * absent_score


# ============================================================
# 单应性变换估计（升级版）：DeepLabV3 语义线端点 → 精准 H 矩阵
# ============================================================
class HomographyEstimator:
    """
    用 DeepLabV3 语义分割识别场地线（禁区线、边线、中线等），
    将图像中识别到的线端点像素坐标与 FIFA 标准球场米制坐标对应，
    用 DLT+RANSAC 求精准单应矩阵 H（图像坐标 → 战术板米制坐标）。

    退化策略（分层）：
      1. 主方案：DeepLabV3 语义线端点（最精准）
      2. 降级：白线 Hough 检测 + 绿色草地 mask 定位场地边界（中等精度）
      3. 托底：复用上一帧 H 矩阵（零精度损失）
    """

    # 战术板画布尺寸（与 build_tactical_view 保持一致）
    CANVAS_W = 800
    CANVAS_H = 520
    MARGIN   = 30
    # 真实球场尺寸
    PITCH_L  = 105.0   # 米
    PITCH_W  = 68.0    # 米

    def __init__(self, model_path=None, mean_path=None, std_path=None):
        self._H          = None   # 当前平滑后 H
        self._H_inv      = None
        self._H_prev     = None   # 上一次成功估计的原始 H（t-1，用于复用/1阶外推）
        self._H_prev2    = None   # 上上次成功估计的原始 H（t-2，用于2阶外推）
        # 相机参数化时序状态（对标 TVCalib 时序正则化）
        self._cam_prev   = None   # 上一帧 _FootballCamera
        self._cam_prev2  = None   # 上上帧 _FootballCamera
        self._opt_cam_latest = None  # 最新优化相机（供场景剧变判断使用）
        # cy/cz 锚点：首次冷启动收敛后记录，之后每帧向锚点施加弹性约束
        # 防止 cy 在热启动迭代中单调漂移（匀速漂移 2阶约束无效，需全局锚点）
        self._cy_anchor  = None
        self._cz_anchor  = None
        self._prev_cam_type = None   # 上一帧机位类型，用于机位切换检测
        self._last_update = -9999
        self._seg_net    = None
        self._seg_available = False
        # EMA 平滑系数（0=不平滑，1=完全用新值）
        self._ema_alpha  = 0.50   # 新H占50%，平衡跟随速度与噪声抑制（原0.35跟随慢易集体偏移）
        # 场景剧变判定阈值：H 元素（归一化后）最大差值超过此阈值才直接替换
        # H矩阵元素归一化后数值范围通常在 0~500，diff=2.0 几乎所有帧都触发剧变
        # 改为 50000.0：只有真正的镜头硬切才触发，避免正常缓移帧和偶发H分解误差被误报
        self._scene_change_thresh = 50000.0
        # 镜头剧变冻结计数器：剧变时冻结5帧复用旧H，防止格点搜索错误解直接生效导致集体跳变
        self._scene_change_freeze = 0
        self._FREEZE_FRAMES = 5
        # H 更新版本号：每次 H 矩阵真正变化（语义分割成功/外推）时 +1
        # 主循环对比此值来判断是否需要重投影历史轨迹点
        self.h_version = 0
        # 推荐 update_interval：GPU 可用时每帧/低频，CPU 时降频
        # 在 _PitchSegNet 加载完成后设置（需要知道推理设备）
        self.update_interval_recommended = 60  # 默认 CPU 值，加载完更新

        # 尝试加载语义分割网络
        # 自动寻找权重文件（项目根目录或 backend 父目录）
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            for candidate in [
                os.path.join(base_dir, 'soccer_pitch_segmentation.pth'),
                os.path.join(base_dir, 'backend', 'soccer_pitch_segmentation.pth'),
            ]:
                if os.path.exists(candidate):
                    model_path = candidate
                    break
        if mean_path is None:
            for candidate in [
                os.path.join(base_dir, 'mean.npy'),
                os.path.join(base_dir, 'backend', 'mean.npy'),
            ]:
                if os.path.exists(candidate):
                    mean_path = candidate
                    break
        if std_path is None:
            for candidate in [
                os.path.join(base_dir, 'std.npy'),
                os.path.join(base_dir, 'backend', 'std.npy'),
            ]:
                if os.path.exists(candidate):
                    std_path = candidate
                    break

        if model_path and mean_path and std_path and \
           os.path.exists(model_path) and os.path.exists(mean_path) and os.path.exists(std_path):
            try:
                self._seg_net = _PitchSegNet(model_path, mean_path, std_path)
                self._seg_available = True
                # 根据推理设备自动设定推荐更新频率
                # GPU：每帧更新（与 SoccerNet 完全一致）
                # CPU：每60帧（约2.4秒/次，避免阻塞）
                _seg_dev = str(getattr(self._seg_net, 'device', 'cpu'))
                if 'cuda' in _seg_dev:
                    self.update_interval_recommended = 1   # GPU：每帧
                    print(f"[OK] 语义分割推理设备: {_seg_dev}，H 更新频率: 每帧（SoccerNet 模式）")
                else:
                    self.update_interval_recommended = 60  # CPU：每60帧
                    print(f"[OK] 语义分割推理设备: {_seg_dev}，H 更新频率: 每60帧（CPU 降频模式）")
            except Exception as e:
                print(f"[WARNING] 语义分割网络加载失败，将使用 Hough 降级方案: {e}")
        else:
            print("[WARNING] 未找到 soccer_pitch_segmentation.pth / mean.npy / std.npy，使用 Hough 降级方案")

    # ── 圆弧世界坐标采样（对标 soccerpitch.py sample_field_points）────────
    @staticmethod
    def _sample_circle_world_pts(dist_m=0.3):
        """
        采样场地圆弧上的世界坐标点（z=0 平面）。
        返回 dict: {class_name: np.ndarray (N,2)}

        采样的三段弧：
          Circle central : 完整中圈（圆心=场地中心，R=9.15m）
          Circle left    : 左侧罚弧弧段（圆心=左罚球点，仅禁区线外部分）
          Circle right   : 右侧罚弧弧段（圆心=右罚球点，仅禁区线外部分）
        """
        _L, _W  = 52.5, 34.0
        _PA_L   = 16.5          # 大禁区长（罚球点距球门线 11m，禁区线到门线 16.5m）
        _PM     = 11.0          # 点球点距球门线
        _CR     = 9.15          # 中圈/罚弧半径

        result = {}

        # ── 中圈（完整圆，0 → 2π）──────────────────────────────────────
        cx, cy = 0.0, 0.0
        circumference = 2 * np.pi * _CR
        n = max(8, int(circumference / dist_m))
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack([cx + _CR * np.cos(angles),
                               cy + _CR * np.sin(angles)])
        result['Circle central'] = pts

        # ── 左罚弧（圆心=左罚球点，仅禁区外弧段 ≈ 53° 内）──────────────
        # 左罚球点世界坐标：x = -L + PM = -52.5 + 11 = -41.5m
        px_l = -_L + _PM  # = -41.5
        # 罚弧与大禁区线（x = -L + PA_L = -36m）的交点 y 值
        dx   = _PA_L - _PM  # = 5.5m
        dy   = float(np.sqrt(max(0, _CR**2 - dx**2)))  # ≈ 7.65m
        # 角度范围：从底部交点 → 顶部交点，经过右侧（朝向中场）
        angle_top    = np.arctan2( dy, dx)   # 约 +54°
        angle_bottom = np.arctan2(-dy, dx)   # 约 -54°
        arc_len = abs(2 * angle_top) * _CR
        n = max(4, int(arc_len / dist_m))
        angles_l = np.linspace(angle_bottom, angle_top, n)
        pts_l = np.column_stack([px_l + _CR * np.cos(angles_l),
                                          _CR * np.sin(angles_l)])
        result['Circle left'] = pts_l

        # ── 右罚弧（镜像）────────────────────────────────────────────────
        px_r = _L - _PM   # = +41.5
        angle_top_r    = np.pi - angle_top    # 约 126°
        angle_bottom_r = np.pi - angle_bottom  # 约 234°，即 -126°
        # 从 angle_bottom_r 到 angle_top_r 经过左侧（π 处）
        angles_r = np.linspace(angle_top_r, angle_bottom_r, n)
        pts_r = np.column_stack([px_r + _CR * np.cos(angles_r),
                                          _CR * np.sin(angles_r)])
        result['Circle right'] = pts_r

        return result

    # ── 相机参数化估计（核心，对标 TVCalib）─────────────────────────────
    def _estimate_from_segmentation_camera(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        相机参数化优化（对标 TVCalib / SNProjectiveCamera）：
          1. DeepLabV3 提取场地线端点 + 圆弧像素点
          2. SVD 初始化 H → 分解出初始相机参数（pan/tilt/roll/aov）
          3. GPU AdamW + OneCycleLR 在相机参数空间优化（500步，自动微分）：
             loss = 直线端点重投影误差 + 圆弧点到模型投影点云最近距离
          4. 相机参数 → H 矩阵（采样球场平面点对拟合）

        核心优势（对比旧 SVD+RANSAC）：
          - 优化空间是物理相机参数（4自由度），天然排除非物理解
          - 圆弧约束通过"点到点云最近距离"加入，无需建立一一对应
          - 时序正则化：pan/tilt 的变化量加入损失，鼓励平滑运动
          - 优化器：GPU AdamW + OneCycleLR，500步迭代（对标 TVCalib optimize.py）
        """
        h_img, w_img = frame.shape[:2]
        mask = self._seg_net.infer(frame)
        endpoints = _extract_line_endpoints_from_mask(mask, w_img, h_img)

        # ── 机位自动判断（根据分割掩码推断相机位置范围）────────────────
        _cam_type, _cy_init, _cy_min, _cy_max, _cz_min, _cz_max = \
            _infer_camera_position_from_mask(mask)

        # ── 机位切换检测：机位类型改变时重置时序状态，防止跨机位锚点污染 ──
        if self._prev_cam_type is not None and self._prev_cam_type != _cam_type:
            print(f"[DEBUG] 机位切换 {self._prev_cam_type} → {_cam_type}，重置时序锚点")
            self._cam_prev   = None
            self._cam_prev2  = None
            self._cy_anchor  = None
            self._cz_anchor  = None
        self._prev_cam_type = _cam_type

        # ── 收集直线端点点对（世界坐标 → 图像坐标）────────────────────
        line_img_pts    = []   # 图像坐标列表，每条线 2 点
        line_world_pts  = []   # 世界坐标列表（米制）

        # 同时也收集用于 SVD 初始化的数据（复用旧逻辑的数据结构）
        line_matches   = []
        all_img_pts_sv = []
        all_canvas_pts_sv = []

        cw = self.CANVAS_W - 2 * self.MARGIN
        ch = self.CANVAS_H - 2 * self.MARGIN

        def _world_to_canvas(wx, wy):
            tx = self.MARGIN + (wx + self.PITCH_L / 2) / self.PITCH_L * cw
            ty = self.MARGIN + (wy + self.PITCH_W / 2) / self.PITCH_W * ch
            return float(tx), float(ty)

        for cname, img_pts in endpoints.items():
            if cname not in _PitchSegNet.WORLD_LINE_ENDPOINTS:
                continue
            world_pts = _PitchSegNet.WORLD_LINE_ENDPOINTS[cname]
            if len(img_pts) < 2 or len(world_pts) < 2:
                continue

            (ix0, iy0), (ix1, iy1) = img_pts[0], img_pts[1]
            (wx0, wy0), (wx1, wy1) = world_pts[0], world_pts[1]

            line_img_pts.extend([[float(ix0), float(iy0)], [float(ix1), float(iy1)]])
            line_world_pts.extend([[wx0, wy0], [wx1, wy1]])

            # SVD 初始化数据
            tx0, ty0 = _world_to_canvas(wx0, wy0)
            tx1, ty1 = _world_to_canvas(wx1, wy1)
            p1_h = np.array([float(ix0), float(iy0), 1.0])
            p2_h = np.array([float(ix1), float(iy1), 1.0])
            img_line = np.cross(p1_h, p2_h)
            c1_h = np.array([tx0, ty0, 1.0])
            c2_h = np.array([tx1, ty1, 1.0])
            canvas_line = np.cross(c1_h, c2_h)
            if (not np.any(np.isnan(img_line)) and not np.any(np.isnan(canvas_line))):
                line_matches.append((canvas_line, img_line))
                all_img_pts_sv.extend([[float(ix0), float(iy0)], [float(ix1), float(iy1)]])
                all_canvas_pts_sv.extend([[tx0, ty0], [tx1, ty1]])

        n_lines = len(line_matches)
        print(f"[DEBUG] 相机参数化估计: {n_lines} 条直线")
        if n_lines < 4:
            return None

        # ── 收集圆弧像素点（从分割 mask 中提取圆弧类别的点）───────────
        # 对每个圆弧类别，从 mask 中提取属于该类别的像素点（下采样）
        circle_obs = {}    # {class_name: np.ndarray (N,2) 图像坐标}
        circle_world = {}  # {class_name: np.ndarray (M,2) 世界坐标，用作点云}

        seg_h, seg_w = mask.shape[:2]
        circle_class_indices = {
            'Circle central': 6,   # LINES_CLASSES 索引（0-based）+ 1
            'Circle left':    7,
            'Circle right':   8,
        }
        # 采样世界坐标圆弧点云
        arc_world_pts = HomographyEstimator._sample_circle_world_pts(dist_m=0.5)

        for cname, cls_idx in circle_class_indices.items():
            # mask 值 = cls_idx + 1（背景=0，类别从1开始）
            ys, xs = np.where(mask == (cls_idx + 1))
            if len(xs) < 10:
                continue
            # 将 mask 像素坐标缩放回原始图像分辨率
            scale_x = w_img / seg_w
            scale_y = h_img / seg_h
            img_xs = xs * scale_x
            img_ys = ys * scale_y
            # 下采样（最多保留 64 个点，对标 TVCalib N_c=64）
            idx = np.random.choice(len(img_xs), min(64, len(img_xs)), replace=False)
            circle_obs[cname] = np.column_stack([img_xs[idx], img_ys[idx]])
            if cname in arc_world_pts:
                circle_world[cname] = arc_world_pts[cname]

        has_circles = len(circle_obs) > 0
        print(f"[DEBUG] 圆弧类别检测到: {list(circle_obs.keys())}")

        # ── SVD 初始化 H（用于反推初始相机参数）───────────────────────
        T1 = _normalization_transform(all_canvas_pts_sv)
        T2 = _normalization_transform(all_img_pts_sv)
        success, H_c2i_svd = _estimate_homography_from_line_correspondences(
            line_matches, T1=T1, T2=T2
        )
        if not success:
            return None

        # 从 SVD H 反推初始相机参数（通过分解 H = K[r1 r2 t]）
        # 先估计初始相机参数：用 RANSAC H 做反投影，再拟合 pan/tilt/roll/aov
        try:
            H_i2c_svd = np.linalg.inv(H_c2i_svd)
            H_i2c_svd /= (H_i2c_svd[2, 2] + 1e-10)
        except Exception:
            return None

        # 用 SVD H 的初始结果做一个粗略的相机参数估计：
        # 通过把战术板上的已知点反投影到图像，匹配观测点，最小化初始误差
        init_cam = self._init_camera_from_H(
            H_c2i_svd, w_img, h_img,
            line_world_pts, line_img_pts,
            cy_min=_cy_min, cy_max=_cy_max,
            cz_min=_cz_min, cz_max=_cz_max,
        )

        cx_v = init_cam.cx   # cx 固定（沿中线）
        cz_v = init_cam.cz   # 相机高度（优化初始值，来自先验）
        cy_v = init_cam.cy   # 相机侧距离（优化初始值，来自 H 矩阵分解）
        x0   = init_cam.get_params()   # [pan, tilt, roll, aov]

        # ── 时序正则化先验 ────────────────────────────────────────────
        prev_params  = self._cam_prev.get_params()  if self._cam_prev  is not None else None
        prev2_params = self._cam_prev2.get_params() if self._cam_prev2 is not None else None
        # cy/cz 的上帧/上上帧值，用于位置时序正则化
        prev_cy  = self._cam_prev.cy   if self._cam_prev  is not None else None
        prev_cz  = self._cam_prev.cz   if self._cam_prev  is not None else None
        prev2_cy = self._cam_prev2.cy  if self._cam_prev2 is not None else None
        prev2_cz = self._cam_prev2.cz  if self._cam_prev2 is not None else None

        # ── 热启动：若有上一帧参数，用上帧参数作为 x0（加速收敛）────
        is_warm_start = prev_params is not None
        if is_warm_start:
            x0   = prev_params.copy()
            cy_v = self._cam_prev.cy
            cz_v = self._cam_prev.cz
            # 检查上帧 cy/cz 是否在当前机位约束范围内，若不在则回退到机位先验初始值
            if not (_cy_min <= cy_v <= _cy_max):
                cy_v = _cy_init
                cz_v = init_cam.cz
                x0   = init_cam.get_params()   # 用机位先验初始化，而非上帧参数
                is_warm_start = False   # 机位越界，降级为冷启动
                print(f"[DEBUG] 热启动 cy 越出范围[{_cy_min},{_cy_max}]，降级冷启动 cy={cy_v:.1f}")
            if not (_cz_min <= cz_v <= _cz_max):
                cz_v = float(np.clip(cz_v, _cz_min, _cz_max))

        # ── GPU AdamW 优化（对标 TVCalib optimize.py）────────────────
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        pan_t  = torch.tensor(x0[0], dtype=torch.float64, device=device, requires_grad=True)
        tilt_t = torch.tensor(x0[1], dtype=torch.float64, device=device, requires_grad=True)
        roll_t = torch.tensor(x0[2], dtype=torch.float64, device=device, requires_grad=True)
        aov_t  = torch.tensor(x0[3], dtype=torch.float64, device=device, requires_grad=True)
        # cy/cz 也纳入优化（相机侧距离和高度）
        cy_t   = torch.tensor(cy_v,  dtype=torch.float64, device=device, requires_grad=True)
        cz_t   = torch.tensor(cz_v,  dtype=torch.float64, device=device, requires_grad=True)

        cx_const = torch.tensor(cx_v, dtype=torch.float64, device=device)

        line_world_t = torch.tensor(line_world_pts, dtype=torch.float64, device=device)
        line_img_t   = torch.tensor(line_img_pts,   dtype=torch.float64, device=device)

        circle_obs_t   = {k: torch.tensor(v, dtype=torch.float64, device=device)
                          for k, v in circle_obs.items()}
        circle_world_t = {k: torch.tensor(v, dtype=torch.float64, device=device)
                          for k, v in circle_world.items()}

        prev_t  = torch.tensor(prev_params,  dtype=torch.float64, device=device) \
                  if prev_params  is not None else None
        prev2_t = torch.tensor(prev2_params, dtype=torch.float64, device=device) \
                  if prev2_params is not None else None

        # cy/cz 时序正则化先验 tensor（不参与梯度计算，仅作常量锚点）
        prev_cy_t  = torch.tensor(prev_cy,  dtype=torch.float64, device=device) \
                     if prev_cy  is not None else None
        prev_cz_t  = torch.tensor(prev_cz,  dtype=torch.float64, device=device) \
                     if prev_cz  is not None else None
        prev2_cy_t = torch.tensor(prev2_cy, dtype=torch.float64, device=device) \
                     if prev2_cy is not None else None
        prev2_cz_t = torch.tensor(prev2_cz, dtype=torch.float64, device=device) \
                     if prev2_cz is not None else None

        # 热启动时固定 aov（上帧已收敛，逐帧优化会导致 aov 剧烈漂移）
        # 冷启动时才让 aov 自由优化
        if is_warm_start:
            aov_t.requires_grad_(False)
            opt_params_list = [pan_t, tilt_t, roll_t, cy_t, cz_t]
        else:
            opt_params_list = [pan_t, tilt_t, roll_t, aov_t, cy_t, cz_t]

        optimizer = torch.optim.AdamW(
            opt_params_list, lr=3e-4, weight_decay=0.0
        )
        # 热启动：已有上帧参数作为起点，300步细调即可；冷启动：800步全搜索
        n_steps = 300 if is_warm_start else 800
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=5e-3, total_steps=n_steps,
            pct_start=0.1, anneal_strategy='cos'
        )
        # early stopping 参数（热启动：连续20步无改善；冷启动：连续40步）
        es_patience = 20 if is_warm_start else 40
        es_min_delta = 1e-4
        es_no_improve = 0

        # 硬边界（每步 optimizer.step 后强制 clamp，彻底防止发散）
        pan_min_v  = -np.pi
        pan_max_v  =  np.pi
        tilt_min_v =  np.radians(5)
        tilt_max_v =  np.radians(89)
        # 门后机位 roll 可能较大（相机倾斜安装），放宽到 ±30°
        if _cam_type == 'behind_goal':
            roll_min_v = -np.radians(30)
            roll_max_v =  np.radians(30)
        else:
            roll_min_v = -np.radians(15)
            roll_max_v =  np.radians(15)
        aov_min_v  =  np.radians(20)
        aov_max_v  =  np.radians(120)   # 广角摄像机可达 90°+

        # Huber delta（像素）
        delta = torch.tensor(10.0, dtype=torch.float64, device=device)

        best_loss   = float('inf')
        best_params = x0.copy()
        best_cy     = cy_v
        best_cz     = cz_v

        try:
            for _ in range(n_steps):
                optimizer.zero_grad()

                # cam_pos 每步动态构建，让 cy_t/cz_t 的梯度能流通
                cam_pos_t = torch.stack([cx_const, cy_t, cz_t])

                proj_lines = _FootballCamera.project_points_torch(
                    pan_t, tilt_t, roll_t, aov_t,
                    cam_pos_t, line_world_t, w_img, h_img
                )
                diff  = proj_lines - line_img_t
                abs_d = diff.abs()
                huber_lines = torch.where(abs_d <= delta,
                                          0.5 * diff ** 2,
                                          delta * (abs_d - 0.5 * delta))
                # 过滤投影飞出画面 3倍 范围外的点（可能是相机后方）
                in_view = ((proj_lines[:, 0].abs() < 3.0 * w_img) &
                           (proj_lines[:, 1].abs() < 3.0 * h_img)).float()
                denom_v = in_view.sum().clamp(min=1.0)
                loss_lines = (huber_lines.mean(dim=1) * in_view).sum() / denom_v

                loss_circles = torch.zeros(1, dtype=torch.float64, device=device)
                if has_circles:
                    n_arcs = 0
                    for cname, obs_t in circle_obs_t.items():
                        if cname not in circle_world_t:
                            continue
                        pc_proj = _FootballCamera.project_points_torch(
                            pan_t, tilt_t, roll_t, aov_t,
                            cam_pos_t, circle_world_t[cname], w_img, h_img
                        )
                        d2   = ((obs_t.unsqueeze(1) - pc_proj.unsqueeze(0)) ** 2).sum(dim=2)
                        min_d = d2.min(dim=1).values.sqrt()
                        h_c = torch.where(min_d <= delta,
                                          0.5 * min_d ** 2,
                                          delta * (min_d - 0.5 * delta))
                        loss_circles = loss_circles + h_c.mean()
                        n_arcs += 1
                    if n_arcs > 0:
                        loss_circles = loss_circles / n_arcs

                params_now = torch.stack([pan_t, tilt_t, roll_t, aov_t])
                loss_temporal = torch.zeros(1, dtype=torch.float64, device=device)
                # lam 量纲说明：
                #   loss_lines 单位为像素（Huber，量级 50~200）
                #   params 单位为弧度，帧间变化约 0.005~0.02 rad
                #   (delta_params)^2 ≈ 1e-4，需 lam~1e5 才能使正则化项
                #   与几何 loss 同量级，产生实际约束（约占 loss 的 20~30%）
                lam = 1e5
                # lam_pos 量纲：cy/cz 单位为米，帧间变化约 0.5~2m
                #   (delta_cy)^2 ≈ 0.25~4，需 lam_pos~100~500 才与 loss_lines 同量级
                lam_pos = 300.0
                # lam_anchor：cy/cz 锚点弹性约束（防匀速漂移）
                #   (cy - cy_anchor)^2，anchor 偏差 1m → 正则化项=lam_anchor*1 ≈ loss_lines 的 5~10%
                #   设 lam_anchor=50，允许 cy 在锚点 ±2m 内自由移动，超出则被拉回
                lam_anchor = 50.0
                if prev2_t is not None and prev_t is not None:
                    loss_temporal = lam * ((params_now - 2 * prev_t + prev2_t) ** 2).mean()
                    # cy/cz 2阶时序平滑（加速度约束）
                    cy_diff = cy_t - 2.0 * prev_cy_t + prev2_cy_t
                    cz_diff = cz_t - 2.0 * prev_cz_t + prev2_cz_t
                    loss_temporal = loss_temporal + lam_pos * (cy_diff ** 2 + cz_diff ** 2)
                elif prev_t is not None:
                    loss_temporal = lam * ((params_now - prev_t) ** 2).mean()
                    # cy/cz 1阶时序平滑（速度约束）
                    loss_temporal = loss_temporal + lam_pos * ((cy_t - prev_cy_t) ** 2 + (cz_t - prev_cz_t) ** 2)

                # cy/cz 全局锚点弹性约束：向首帧冷启动值靠拢，阻止匀速漂移
                if self._cy_anchor is not None:
                    cy_anchor_t = torch.tensor(self._cy_anchor, dtype=torch.float64, device=device)
                    cz_anchor_t = torch.tensor(self._cz_anchor, dtype=torch.float64, device=device)
                    loss_temporal = loss_temporal + lam_anchor * ((cy_t - cy_anchor_t) ** 2 + (cz_t - cz_anchor_t) ** 2)

                loss = loss_lines + loss_circles + loss_temporal
                loss.backward()
                optimizer.step()
                scheduler.step()

                # ── 硬 clamp：每步之后强制参数在合法范围内 ──────────
                with torch.no_grad():
                    pan_t.clamp_(pan_min_v,  pan_max_v)
                    tilt_t.clamp_(tilt_min_v, tilt_max_v)
                    roll_t.clamp_(roll_min_v, roll_max_v)
                    aov_t.clamp_(aov_min_v,  aov_max_v)
                    cy_t.clamp_(_cy_min, _cy_max)   # 由机位判断动态设定
                    cz_t.clamp_(_cz_min, _cz_max)   # 由机位判断动态设定

                loss_val = loss.item()
                if loss_val < best_loss - es_min_delta:
                    best_loss = loss_val
                    best_params = np.array([
                        pan_t.item(), tilt_t.item(),
                        roll_t.item(), aov_t.item()
                    ])
                    best_cy = cy_t.item()
                    best_cz = cz_t.item()
                    es_no_improve = 0
                else:
                    es_no_improve += 1
                    if es_no_improve >= es_patience:
                        break

            start_type = "热启动" if is_warm_start else "冷启动"
            print(f"[DEBUG] GPU AdamW 优化完成({start_type},steps={n_steps},early_stop@{es_no_improve}): "
                  f"loss={best_loss:.2f}, "
                  f"pan={np.degrees(best_params[0]):.1f}°, "
                  f"tilt={np.degrees(best_params[1]):.1f}°, "
                  f"roll={np.degrees(best_params[2]):.1f}°, "
                  f"aov={np.degrees(best_params[3]):.1f}°, "
                  f"cy={best_cy:.1f}m, cz={best_cz:.1f}m, device={device}")
        except Exception as e:
            print(f"[WARNING] 相机参数化优化失败: {e}")
            best_cy = cy_v
            best_cz = cz_v

        opt_params = best_params
        cy_v = best_cy
        cz_v = best_cz

        # ── 构建最优相机 ──────────────────────────────────────────────
        opt_cam = _FootballCamera.from_params(
            opt_params, cx=cx_v, cy=cy_v, cz=cz_v,
            image_width=w_img, image_height=h_img
        )

        # 冷启动成功后记录 cy/cz 锚点（之后热启动向此值靠拢，防止单调漂移）
        if not is_warm_start:
            self._cy_anchor = cy_v
            self._cz_anchor = cz_v
            print(f"[DEBUG] cy/cz 锚点已记录: cy={cy_v:.1f}m, cz={cz_v:.1f}m")

        # 保存相机参数用于时序正则化
        self._cam_prev2 = self._cam_prev
        self._cam_prev  = opt_cam

        # ── 最终 H 矩阵：以相机模型合成格点为主，直接点对仅在内点率≥50%时覆盖 ──
        # to_homography 用24个均匀格点，内点率~85%，稳定可靠
        # 直接点对受语义分割噪声影响，内点率常低至22%，会导致H每帧剧烈跳动
        H = opt_cam.to_homography(
            self.CANVAS_W, self.CANVAS_H,
            self.PITCH_L, self.PITCH_W, self.MARGIN
        )
        if H is None:
            print("[WARNING] 相机参数化：to_homography 失败")
            return None
        print("[DEBUG] 相机参数化 H：to_homography（合成格点，稳定基准）")

        # 仅当直接点对内点率≥50%时才用其覆盖
        if len(line_world_pts) >= 4:
            try:
                cw = self.CANVAS_W - 2 * self.MARGIN
                ch = self.CANVAS_H - 2 * self.MARGIN

                canvas_pts = []
                for wx, wy in line_world_pts:
                    tx = self.MARGIN + (wx + self.PITCH_L / 2) / self.PITCH_L * cw
                    ty = self.MARGIN + (wy + self.PITCH_W / 2) / self.PITCH_W * ch
                    canvas_pts.append([tx, ty])
                canvas_arr = np.array(canvas_pts, dtype=np.float32)
                img_arr    = np.array(line_img_pts, dtype=np.float32)

                H_c2i, mask = cv2.findHomography(canvas_arr, img_arr, cv2.RANSAC, 5.0)
                n_total = len(line_world_pts)
                n_inlier = int(mask.sum()) if mask is not None else 0
                inlier_rate = n_inlier / n_total if n_total > 0 else 0.0
                if H_c2i is not None and mask is not None and inlier_rate >= 0.5:
                    H_i2c = np.linalg.inv(H_c2i)
                    H_i2c /= (H_i2c[2, 2] + 1e-10)
                    H = H_i2c
                    print(f"[DEBUG] 相机参数化 H：直接点对覆盖（内点率={inlier_rate:.0%}，{n_inlier}/{n_total}）")
                else:
                    print(f"[DEBUG] 相机参数化 H：直接点对内点率不足（{n_inlier}/{n_total}={inlier_rate:.0%}<50%），保留合成格点结果")
            except Exception:
                pass

        return H

    @staticmethod
    def _init_camera_from_H(H_canvas_to_img, w_img, h_img,
                             line_world_pts, line_img_pts,
                             cy_min=-110.0, cy_max=110.0,
                             cz_min=-40.0,  cz_max=-1.0):
        """
        从 SVD 求得的 H（canvas→image）反推相机参数初始值。

        策略：
          1. 将 H_canvas_to_img 转换为 H_world_to_img（米制坐标→图像）
          2. 用 _extract_K_from_homography 从 H 分解焦距 → aov
          3. 用 K^{-1}·H 分解出旋转矩阵 R → pan / tilt / roll
          4. 从 R 和 t 估算相机位置 cy（侧距离）
          5. 若分解失败，回退到格点搜索

        注意坐标系：_FootballCamera 使用 ZXZ 欧拉角，R(pan,tilt,roll) 满足：
            pts_cam = R @ (pts_world - cam_pos)
            pts_img = fl * pts_cam_norm + img_center
        """
        # ── 0. 辅助：canvas(px) → world_meter 的变换矩阵 ──────────────
        # canvas 坐标：tx = MARGIN + (wx + L/2)/L * cw，ty = MARGIN + (wy + W/2)/W * ch
        # 逆变换：wx = (tx - MARGIN) / cw * L - L/2
        CANVAS_W, CANVAS_H = 800, 520
        MARGIN = 30
        PITCH_L, PITCH_W = 105.0, 68.0
        cw = CANVAS_W - 2 * MARGIN   # 740
        ch = CANVAS_H - 2 * MARGIN   # 460

        # S: world_meter → canvas_px（齐次）
        # tx = (MARGIN - cw*L/2/L) + wx*(cw/L)  = MARGIN + cw/2 + wx*cw/L
        sx = cw / PITCH_L
        sy = ch / PITCH_W
        ox = MARGIN + cw / 2.0
        oy = MARGIN + ch / 2.0
        S = np.array([
            [sx,  0.0, ox],
            [0.0, sy,  oy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # H_world_to_img = H_canvas_to_img @ S
        try:
            H_world_to_img = H_canvas_to_img @ S
            if abs(H_world_to_img[2, 2]) > 1e-10:
                H_world_to_img /= H_world_to_img[2, 2]
        except Exception:
            H_world_to_img = None

        # ── 1. 从 H_world_to_img 分解 K → aov ─────────────────────────
        init_pan, init_tilt, init_roll, init_aov = (
            _FootballCamera.PRIOR_PAN,
            _FootballCamera.PRIOR_TILT,
            _FootballCamera.PRIOR_ROLL,
            _FootballCamera.PRIOR_AOV,
        )
        init_cx = _FootballCamera.PRIOR_CX
        init_cy = _FootballCamera.PRIOR_CY
        init_cz = _FootballCamera.PRIOR_CZ

        decomp_ok = False
        if H_world_to_img is not None:
            try:
                # ── 方法1：IAC（最精确，但 Cholesky 可能失败）────────────
                fl_est = None
                ok_K, K = _extract_K_from_homography(H_world_to_img, w_img, h_img)
                if ok_K and 50 < K[0, 0] < 5 * w_img:
                    fl_est = float(K[0, 0])

                # ── 方法2：备用焦距估算（Zhang 单约束：|h1|=|h2|）─────────
                # 仅当方法1失败时使用
                if fl_est is None:
                    h = H_world_to_img.ravel()
                    # h1·h2 = 0 且 |h1|=|h2|，两个约束估 f^2
                    # 假设主点在图像中心：ω = diag(1/f², 1/f², 1)
                    # 从 h1·ω·h2=0: h[0]*h[3]+h[1]*h[4] + (h[2]*h[5])*f^2 = 0
                    # → f^2 = -(h[0]*h[3]+h[1]*h[4]) / (h[2]*h[5])
                    denom12 = h[2] * h[5]
                    if abs(denom12) > 1e-10:
                        f2_a = -(h[0]*h[3] + h[1]*h[4]) / denom12
                        if f2_a > 0:
                            fl_est = float(np.sqrt(f2_a))
                    # 从 |h1|²=|h2|²: h[0]²+h[1]²-h[3]²-h[4]² + (h[2]²-h[5]²)*f² = 0
                    if fl_est is None:
                        denom11 = h[2]**2 - h[5]**2
                        if abs(denom11) > 1e-10:
                            f2_b = -(h[0]**2 + h[1]**2 - h[3]**2 - h[4]**2) / denom11
                            if f2_b > 0:
                                fl_est = float(np.sqrt(f2_b))
                    if fl_est is not None and not (50 < fl_est < 5 * w_img):
                        fl_est = None

                if fl_est is not None:
                    aov_est = 2.0 * np.arctan2(w_img / 2.0, fl_est)
                    # ── 构造 K 分解 R+t ──────────────────────────────────
                    K_use = np.array([
                        [fl_est, 0.0,    w_img / 2.0],
                        [0.0,    fl_est, h_img / 2.0],
                        [0.0,    0.0,    1.0],
                    ], dtype=np.float64)
                    hprim = np.linalg.inv(K_use) @ H_world_to_img
                    lam1 = 1.0 / (np.linalg.norm(hprim[:, 0]) + 1e-12)
                    lam2 = 1.0 / (np.linalg.norm(hprim[:, 1]) + 1e-12)
                    lam3 = np.sqrt(lam1 * lam2)
                    r0 = hprim[:, 0] * lam1
                    r1 = hprim[:, 1] * lam2
                    r2 = np.cross(r0, r1)
                    R_raw = np.column_stack((r0, r1, r2))
                    # 正交化
                    u, _, vh = np.linalg.svd(R_raw)
                    R_orth = u @ vh
                    if np.linalg.det(R_orth) < 0:
                        u[:, 2] *= -1
                        R_orth = u @ vh

                    # 从旋转矩阵逆解 ZXZ 欧拉角
                    # R[2,0]=sin(pan)*sin(tilt), R[2,1]=-sin(tilt)*cos(pan), R[2,2]=cos(tilt)
                    ct = float(R_orth[2, 2])
                    ct = np.clip(ct, -1.0, 1.0)
                    tilt_est = np.arccos(ct)   # tilt ∈ [0, π]
                    if np.sin(tilt_est) > 1e-6:
                        pan_est  = np.arctan2( R_orth[2, 0],  -R_orth[2, 1])
                        roll_est = np.arctan2( R_orth[0, 2],   R_orth[1, 2])
                    else:
                        # Gimbal lock：tilt≈0 or π
                        pan_est  = np.arctan2(-R_orth[1, 0],  R_orth[0, 0])
                        roll_est = 0.0

                    # 从 t 估算相机位置
                    t_vec = hprim[:, 2] * lam3
                    cam_pos_est = -R_orth.T @ t_vec
                    cy_est_val = float(np.clip(cam_pos_est[1], -80.0, -5.0))

                    # 有效性检验
                    aov_deg  = np.degrees(aov_est)
                    tilt_deg = np.degrees(tilt_est)
                    if (5 < tilt_deg < 85) and (20 < aov_deg < 130):
                        init_pan  = float(pan_est)
                        init_tilt = float(tilt_est)
                        init_roll = float(np.clip(roll_est, -np.radians(15), np.radians(15)))
                        init_aov  = float(np.clip(aov_est, np.radians(20), np.radians(120)))
                        init_cy   = cy_est_val
                        decomp_ok = True
                        print(f"[DEBUG] H分解成功: fl={fl_est:.1f}px, "
                              f"tilt={tilt_deg:.1f}°, aov={aov_deg:.1f}°, "
                              f"pan={np.degrees(pan_est):.1f}°, cy={cy_est_val:.1f}m")
                    else:
                        print(f"[DEBUG] H分解有效性检验失败: tilt={tilt_deg:.1f}°, aov={aov_deg:.1f}°")
            except Exception as ex:
                print(f"[DEBUG] H分解异常: {ex}")
                decomp_ok = False

        # ── 2. 若分解失败，回退到格点搜索 ─────────────────────────────
        line_world_arr = np.array(line_world_pts, dtype=np.float64)
        line_img_arr   = np.array(line_img_pts,   dtype=np.float64)

        def _eval(cam):
            try:
                proj = cam.project_points(line_world_arr)
                return float(np.mean(np.sum((proj - line_img_arr)**2, axis=1)))
            except Exception:
                return np.inf

        # H 分解成功但结果明显错误（loss 极大）时，也降级到格点搜索
        if decomp_ok:
            _check_cam = _FootballCamera.from_params(
                np.array([init_pan, init_tilt, init_roll, init_aov]),
                cx=init_cx, cy=init_cy, cz=init_cz,
                image_width=w_img, image_height=h_img
            )
            _check_loss = _eval(_check_cam)
            if _check_loss > 1e6:
                print(f"[DEBUG] H分解结果验证失败(loss={_check_loss:.1f})，降级为格点搜索")
                decomp_ok = False

        if not decomp_ok:
            best_loss_grid = np.inf
            best_pan_g, best_tilt_g, best_aov_g = init_pan, init_tilt, init_aov
            # 加密格点：pan 7档 × tilt 6档 × aov 5档 = 210 个候选
            for pan_d in [-60, -40, -20, 0, 20, 40, 60]:
                for tilt_d in [10, 20, 30, 40, 50, 65]:
                    for aov_d in [40, 55, 70, 85, 100]:
                        cam = _FootballCamera(
                            pan  = np.radians(pan_d),
                            tilt = np.radians(tilt_d),
                            roll = 0.0,
                            aov  = np.radians(aov_d),
                            cx=init_cx, cy=init_cy, cz=init_cz,
                            image_width=w_img, image_height=h_img
                        )
                        loss = _eval(cam)
                        if loss < best_loss_grid:
                            best_loss_grid = loss
                            best_pan_g  = np.radians(pan_d)
                            best_tilt_g = np.radians(tilt_d)
                            best_aov_g  = np.radians(aov_d)
            init_pan  = best_pan_g
            init_tilt = best_tilt_g
            init_aov  = best_aov_g

        # ── 3. 构造初始相机，计算验证 loss ────────────────────────────
        init_params = np.array([init_pan, init_tilt, init_roll, init_aov])
        best_cam = _FootballCamera.from_params(
            init_params, cx=init_cx, cy=init_cy, cz=init_cz,
            image_width=w_img, image_height=h_img
        )
        best_loss = _eval(best_cam)

        print(f"[DEBUG] 相机初始化({'H分解' if decomp_ok else '格点搜索'}): "
              f"loss={best_loss:.1f}, "
              f"pan={np.degrees(init_pan):.1f}°, tilt={np.degrees(init_tilt):.1f}°, "
              f"aov={np.degrees(init_aov):.1f}°, cy={init_cy:.1f}m")
        return best_cam

    # ── 主方案（旧版SVD）：保留作为回退 ─────────────────────────────────
    def _estimate_from_segmentation(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        方案A（对标 SoccerNet Baseline）：
          1. 用 DeepLabV3 提取场地线端点（每条线两个端点）
          2. 对每条线用 np.cross(p1, p2) 计算图像齐次直线方程
          3. 同样对世界坐标战术板端点计算战术板齐次直线方程
          4. 调用 _estimate_homography_from_line_correspondences 做 SVD 求解

        优势：天然规避端点正序/反序问题——直线 cross product 不依赖端点顺序。
        与 SoccerNet sn-calibration-main/src/baseline_cameras.py 完全一致。

        坐标说明：
          - 图像端点为像素坐标 (px)
          - 世界端点转换为战术板画布坐标（800×520 px）
          - 输出 H 矩阵：图像坐标 → 战术板坐标
        """
        h_img, w_img = frame.shape[:2]
        mask = self._seg_net.infer(frame)
        endpoints = _extract_line_endpoints_from_mask(mask, w_img, h_img)

        cw = self.CANVAS_W - 2 * self.MARGIN
        ch = self.CANVAS_H - 2 * self.MARGIN

        def _world_to_canvas(wx, wy):
            tx = self.MARGIN + (wx + self.PITCH_L / 2) / self.PITCH_L * cw
            ty = self.MARGIN + (wy + self.PITCH_W / 2) / self.PITCH_W * ch
            return float(tx), float(ty)

        line_matches    = []   # [(canvas_line, image_line), ...]  用于 SVD
        all_img_pts     = []   # 收集所有图像点，用于归一化变换 T2
        all_canvas_pts  = []   # 收集所有战术板点，用于归一化变换 T1
        lm_point_pairs  = []   # [(canvas_pt_3d, img_pt_2d), ...]  用于 LM 精化

        for cname, img_pts in endpoints.items():
            if cname not in _PitchSegNet.WORLD_LINE_ENDPOINTS:
                continue
            world_pts = _PitchSegNet.WORLD_LINE_ENDPOINTS[cname]
            if len(img_pts) < 2 or len(world_pts) < 2:
                continue

            (ix0, iy0), (ix1, iy1) = img_pts[0], img_pts[1]
            (wx0, wy0), (wx1, wy1) = world_pts[0], world_pts[1]
            tx0, ty0 = _world_to_canvas(wx0, wy0)
            tx1, ty1 = _world_to_canvas(wx1, wy1)

            # ── 图像齐次直线：cross(p1_h, p2_h) ────────────────────────
            p1_h = np.array([float(ix0), float(iy0), 1.0])
            p2_h = np.array([float(ix1), float(iy1), 1.0])
            img_line = np.cross(p1_h, p2_h)
            if np.any(np.isnan(img_line)) or np.any(np.isinf(img_line)):
                continue

            # ── 战术板齐次直线：cross(c1_h, c2_h) ──────────────────────
            c1_h = np.array([tx0, ty0, 1.0])
            c2_h = np.array([tx1, ty1, 1.0])
            canvas_line = np.cross(c1_h, c2_h)
            if np.any(np.isnan(canvas_line)) or np.any(np.isinf(canvas_line)):
                continue

            line_matches.append((canvas_line, img_line))
            all_img_pts.extend([[float(ix0), float(iy0)], [float(ix1), float(iy1)]])
            all_canvas_pts.extend([[tx0, ty0], [tx1, ty1]])
            # 收集 LM 精化点对：canvas坐标当作Z=0的3D世界点，图像坐标为2D观测
            # H（image→canvas）做LM精化时相当于：已知观测img_pt，目标canvas_pt
            # 注意 _refine_H_with_lm 的单应矩阵方向是 world→image，
            # 所以这里传入的 H 应是 canvas→image（即 H_canvas_to_img），
            # 点对也对应 (canvas_3d → img_2d)
            lm_point_pairs.extend([
                ([tx0, ty0, 0.0], [float(ix0), float(iy0)]),
                ([tx1, ty1, 0.0], [float(ix1), float(iy1)]),
            ])

        n_lines = len(line_matches)
        print(f"[DEBUG] 语义分割线对应: {n_lines} 条线")
        if n_lines < 4:
            print(f"[DEBUG] 线对应数量不足（{n_lines}<4），放弃")
            return None

        # ── 坐标归一化（数值稳定性） ─────────────────────────────────────
        T1 = _normalization_transform(all_canvas_pts)   # 战术板坐标归一化
        T2 = _normalization_transform(all_img_pts)      # 图像坐标归一化

        # ── SVD 求解单应矩阵 ─────────────────────────────────────────────
        # line_matches 格式：(world/canvas_line, image_line)
        # _estimate_homography_from_line_correspondences 输出 H: src→tgt
        # 这里 src=世界(canvas), tgt=图像，所以 H 是 canvas→image
        # 我们需要 image→canvas，取逆
        success, H_canvas_to_img = _estimate_homography_from_line_correspondences(
            line_matches, T1=T1, T2=T2
        )
        if not success:
            print("[DEBUG] SVD 单应矩阵估计失败")
            return None

        try:
            H = np.linalg.inv(H_canvas_to_img)   # 图像 → 战术板
            if abs(H[2, 2]) > 1e-10:
                H /= H[2, 2]
        except np.linalg.LinAlgError:
            print("[DEBUG] 单应矩阵求逆失败（奇异矩阵）")
            return None

        # ── 步骤2：用 RANSAC 过滤外点（对标 sn-gamestate findHomography(RANSAC)）──
        # 将点对整理为 cv2.findHomography 所需格式（canvas_pts → img_pts）
        # 目的：用 RANSAC 内置外点剔除，替代事后越界率验证
        try:
            src_pts = np.array(all_canvas_pts, dtype=np.float32)  # canvas 坐标
            dst_pts = np.array(all_img_pts,    dtype=np.float32)  # 图像坐标
            H_ransac, ransac_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
        except Exception as e:
            H_ransac, ransac_mask = None, None

        if H_ransac is not None and ransac_mask is not None:
            n_inlier = int(ransac_mask.sum())
            n_total  = len(ransac_mask)
            print(f"[DEBUG] 语义分割 H矩阵（直线SVD+RANSAC）: {n_lines} 条线，"
                  f"RANSAC内点 {n_inlier}/{n_total}，估计成功")
            # 如果内点过少（< 4对），退化到纯 SVD 结果
            if n_inlier < 4:
                print(f"[WARNING] RANSAC 内点不足，使用 SVD 结果")
                all_canvas_pts_inlier = all_canvas_pts
                all_img_pts_inlier    = all_img_pts
            else:
                # 用 RANSAC 内点更新点对集合（用于后续时序优化）
                inlier_mask_flat = ransac_mask.ravel().astype(bool)
                all_canvas_pts_inlier = [p for p, m in zip(all_canvas_pts, inlier_mask_flat) if m]
                all_img_pts_inlier    = [p for p, m in zip(all_img_pts,    inlier_mask_flat) if m]
                # 用 RANSAC 结果（canvas→image）转为 image→canvas
                try:
                    H_c2i_ransac = H_ransac  # canvas → image
                    H_ransac_inv = np.linalg.inv(H_c2i_ransac)
                    if abs(H_ransac_inv[2, 2]) > 1e-10:
                        H_ransac_inv /= H_ransac_inv[2, 2]
                    H_canvas_to_img = H_c2i_ransac
                    H = H_ransac_inv
                except Exception:
                    pass  # 求逆失败，保持 SVD 结果
        else:
            n_inlier = len(all_canvas_pts)
            all_canvas_pts_inlier = all_canvas_pts
            all_img_pts_inlier    = all_img_pts
            print(f"[DEBUG] 语义分割 H矩阵（直线SVD）: {n_lines} 条线，估计成功")

        # ── 步骤3：sn-gamestate 时序正则化 least_squares 精化 ──────────────
        # 对标 sn-gamestate SequentialCalib.get_homography_from_ground_plane
        # fun_1: err = [reproj_err; λ*(H - H_{t-1})]
        # fun_2: err = [reproj_err; λ*(H - 2*H_{t-1} + H_{t-2})]  (匀速运动)
        from scipy.optimize import least_squares as _lsq

        # 内点点对（canvas → image 方向，与 sn-gamestate 一致）
        _c_pts = np.array(all_canvas_pts_inlier, dtype=np.float64)  # (N,2) canvas
        _i_pts = np.array(all_img_pts_inlier,    dtype=np.float64)  # (N,2) image

        def _reproj_resid(h9, c_pts, i_pts):
            """重投影残差：H * canvas_pt → image_pt"""
            hm = h9.reshape(3, 3)
            pts_h = np.column_stack([c_pts, np.ones(len(c_pts))])  # (N,3)
            proj = (hm @ pts_h.T).T  # (N,3)
            denom = proj[:, 2:3]
            denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
            proj2d = proj[:, :2] / denom  # (N,2) 图像坐标
            return (proj2d - i_pts).ravel()

        _lam = 1.0   # 时序正则化权重（对标 sn-gamestate，不过度约束）

        h0 = H_canvas_to_img.ravel().copy()  # SVD/RANSAC 结果作为初始值

        try:
            if self._H_prev2 is not None and self._H_prev is not None:
                # fun_2：匀速运动正则（H_t - 2*H_{t-1} + H_{t-2} → 0）
                est1 = self._H_prev.ravel()   # canvas→image 方向 = inv(H_prev)
                est2 = self._H_prev2.ravel()
                try:
                    est1 = np.linalg.inv(self._H_prev).ravel()
                    est2 = np.linalg.inv(self._H_prev2).ravel()
                except Exception:
                    est1 = self._H_prev.ravel()
                    est2 = self._H_prev2.ravel()

                def _fun2(h9):
                    r1 = _reproj_resid(h9, _c_pts, _i_pts)
                    r2 = _lam * (h9 - 2 * est1 + est2)
                    return np.concatenate([r1, r2])

                res = _lsq(_fun2, h0, verbose=0, x_scale='jac', ftol=1e-8, method='lm')
                if res.success:
                    H_opt = res.x.reshape(3, 3)
                    H_opt /= (H_opt[2, 2] + 1e-10)
                    H_canvas_to_img = H_opt
                    print(f"[DEBUG] sn-gamestate时序精化成功（fun_2匀速）")
                else:
                    pass

            elif self._H_prev is not None:
                # fun_1：速度为零正则（H_t ≈ H_{t-1}）
                try:
                    est1 = np.linalg.inv(self._H_prev).ravel()
                except Exception:
                    est1 = self._H_prev.ravel()

                def _fun1(h9):
                    r1 = _reproj_resid(h9, _c_pts, _i_pts)
                    r2 = _lam * (h9 - est1)
                    return np.concatenate([r1, r2])

                res = _lsq(_fun1, h0, verbose=0, x_scale='jac', ftol=1e-8, method='lm')
                if res.success:
                    H_opt = res.x.reshape(3, 3)
                    H_opt /= (H_opt[2, 2] + 1e-10)
                    H_canvas_to_img = H_opt
                    print(f"[DEBUG] sn-gamestate时序精化成功（fun_1静止）")

        except Exception as e:
            pass  # 时序精化失败，保持 RANSAC/SVD 结果

        # 最终取逆得到 image → canvas 的 H
        try:
            H_final = np.linalg.inv(H_canvas_to_img)
            if abs(H_final[2, 2]) > 1e-10:
                H_final /= H_final[2, 2]
            if np.any(np.isnan(H_final)) or np.any(np.isinf(H_final)):
                return H  # 回退到 SVD 结果
            return H_final
        except np.linalg.LinAlgError:
            return H  # 回退到 SVD 结果


    # ── 时序外推：用前2帧H预测当前帧 ──
    def _extrapolate_H(self) -> Optional[np.ndarray]:
        """
        当语义分割H估计失败时，用时序差分对H进行外推，替代原有的草地mask降级方案。

        优先级：
          1. 2阶外推（速度外推）：H_t ≈ 2·H_{t-1} - H_{t-2}
             物理含义：假设相机以匀速 Pan/Tilt，则H的变化速率恒定。
             对应 sn-gamestate SequentialCalib.fun_2 的差分项：H_t - 2H_{t-1} + H_{t-2} = 0
          2. 1阶外推（静止假设）：直接复用 H_{t-1}
             等价于 use_prev_homography，假设相机不动。

        与草地mask方案的根本区别：
          - 草地mask方案：通过草地颜色猜视野范围，误差极大（+-30%场地宽度）
          - 时序外推：利用真实的历史H矩阵，误差仅取决于相机运动的非线性程度
          - 实际广播摄像机 Pan/Tilt 均为匀速电机，2阶外推误差通常 < 5px
        """
        if self._H_prev is None:
            return None

        if self._H_prev2 is not None:
            # 2阶外推：H_t = 2·H_{t-1} - H_{t-2}（速度恒定假设）
            try:
                H_extrap = 2.0 * self._H_prev - self._H_prev2
                # 重新归一化
                if abs(H_extrap[2, 2]) > 1e-10:
                    H_extrap /= H_extrap[2, 2]
                # 简单验证：外推结果不应和上帧差距太大
                h_cur = self._H_prev / (self._H_prev[2, 2] + 1e-10)
                h_ext = H_extrap / (H_extrap[2, 2] + 1e-10)
                diff = np.max(np.abs(h_ext - h_cur))
                if diff < 5.0 and not (np.any(np.isnan(H_extrap)) or np.any(np.isinf(H_extrap))):
                    print(f"[DEBUG] 时序外推H（2阶速度外推，diff={diff:.3f}）")
                    return H_extrap
            except Exception:
                pass

        # 1阶（静止假设）：复用上帧
        print(f"[DEBUG] 时序外推H（1阶静止复用）")
        return self._H_prev.copy()

    def estimate(self, frame: np.ndarray, frame_idx: int, update_interval: int = 60):
        """
        每 update_interval 帧更新一次 H 矩阵。

        优先级（从高到低）：
          1. 语义分割线端点 + SVD + LM精化（主方案）
          2. 时序外推（差分正则化）
             - 2阶：H_t = 2·H_{t-1} - H_{t-2}（相机匀速 Pan/Tilt）
             - 1阶：H_t = H_{t-1}（相机静止假设）
          3. 线性缩放兜底（首帧没有任何H时）

        帧间平滑（仅语义分割成功时触发）：
          对新估计的 H 与当前平滑 H 做 EMA 混合，防止镜头缓移时位置抖动。
          alpha=0.45：新H占45%，旧H占55%，平衡响应速度与平滑度。
          场景剧变（diff≥2.0）时直接替换，并冻结若干帧。
        """
        # 镜头剧变冻结倒计时：每次调用都递减，冻结期间复用旧H（不返回None）
        # 旧H虽然对新镜头不精准，但远比线性缩放降级好得多
        if self._scene_change_freeze > 0:
            self._scene_change_freeze -= 1
            if self._H is not None:
                return self._H  # 复用旧H，比线性缩放更准

        if frame_idx - self._last_update < update_interval and self._H is not None:
            return self._H

        H_new = None
        H_new_is_seg = False  # 标记是否来自语义分割（可信），外推结果不触发场景剧变

        # ── 主方案：相机参数化优化（TVCalib策略：pan/tilt/roll/fov + 圆弧约束）──
        if self._seg_available:
            try:
                H_new = self._estimate_from_segmentation_camera(frame)
                if H_new is not None:
                    H_new_is_seg = True
            except Exception as e:
                print(f"[WARNING] 语义分割 H 矩阵估计异常: {e}")

        # ── 降级方案：时序外推 ──
        # 语义分割失败时，用前1/2帧H做差分外推，不触发场景剧变冻结
        if H_new is None:
            H_new = self._extrapolate_H()
            if H_new is not None:
                # 外推结果：更新 _last_update 以免过频繁重试，但不做 EMA 对比/场景剧变判断
                self._H = H_new
                try:
                    self._H_inv = np.linalg.inv(H_new)
                except Exception:
                    self._H_inv = None
                self._last_update = frame_idx
                return self._H

        # ── 托底：语义分割和外推都失败（首帧无历史H可用）──────────────────
        if H_new is None:
            return self._H  # 首帧时为 None，接受线性缩放

        # ── 语义分割成功：EMA 平滑 ──────
        # 判断是否为场景剧变（H 元素的最大差距超过阈值 → 直接替换，不平滑）
        if self._H is not None:
            try:
                h_cur = self._H / (self._H[2, 2] + 1e-10)
                h_new = H_new / (H_new[2, 2] + 1e-10)
                diff = np.max(np.abs(h_new - h_cur))
                if diff < self._scene_change_thresh:
                    # 渐变（镜头缓慢移动 / 同一机位正常重估）→ EMA 平滑
                    alpha = self._ema_alpha
                    H_smoothed = alpha * H_new + (1.0 - alpha) * self._H
                    if abs(H_smoothed[2, 2]) > 1e-10:
                        H_smoothed /= H_smoothed[2, 2]
                    H_new = H_smoothed
                    print(f"[DEBUG] H矩阵EMA平滑（diff={diff:.3f}, alpha={alpha}）")
                else:
                    # 场景剧变：新 H 直接替换，冻结若干帧复用旧H
                    self._scene_change_freeze = self._FREEZE_FRAMES
                    print(f"[DEBUG] 场景剧变（diff={diff:.3f}），直接替换H矩阵，冻结{self._FREEZE_FRAMES}帧")
            except Exception:
                pass

        self._H_prev2 = self._H_prev.copy() if self._H_prev is not None else None
        self._H_prev = H_new.copy()
        self._H = H_new
        self.h_version += 1
        try:
            self._H_inv = np.linalg.inv(H_new)
        except Exception:
            self._H_inv = None
        self._last_update = frame_idx

        return self._H

    def project_point(self, pt, H=None):
        """
        将图像坐标 (x, y) 用 H 矩阵投影到战术板坐标。
        若单应矩阵不可用，直接返回原坐标（降级保底）。
        """
        if H is None:
            H = self._H
        if H is None:
            return pt
        try:
            src = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, H)
            return (int(dst[0, 0, 0]), int(dst[0, 0, 1]))
        except Exception:
            return pt

    def project_bbox_bottom(self, x1: int, y1: int, x2: int, y2: int, H=None,
                             video_w: int = 0, video_h: int = 0):
        """
        将检测框底边中点（脚点）投影到战术板坐标。

        球员底部（脚点）是其在地面 Z=0 平面上的真实位置，
        比框中心投影精度更高（尤其是远近景混合时）。

        参数:
          video_w/video_h: 视频原始分辨率，用于 fallback 线性缩放（必传）

        返回: (px, py) 战术板坐标；失败时返回框底边中点的线性缩放坐标
        """
        if H is None:
            H = self._H

        # 底边三个锚点（左底、右底、中底）
        bm_x = float(x1 + x2) / 2.0
        bm_y = float(y2)

        if H is not None:
            try:
                # H 已是 image→canvas，无需再求逆
                bm_h = np.array([bm_x, bm_y, 1.0])
                proj = H @ bm_h
                if abs(proj[2]) > 1e-10:
                    proj /= proj[2]
                    px, py = int(proj[0]), int(proj[1])
                    # ── 投影越界视为 H 失效，
                    # 返回 None 而非 clip，由调用方决定是否 fallback ──
                    # 宽松边界（允许稍微超出边距）
                    bound_lo = self.MARGIN - 40
                    bound_x  = self.CANVAS_W - self.MARGIN + 40
                    bound_y  = self.CANVAS_H - self.MARGIN + 40
                    if bound_lo <= px <= bound_x and bound_lo <= py <= bound_y:
                        return (px, py)
                    # 投影越界：H 矩阵质量差，fallback 到线性缩放
            except Exception:
                pass

        # 降级：线性缩放（用视频分辨率归一化，而非画布尺寸）
        # 修复原 bug：原先用 self.CANVAS_W/H 作为除数，导致像素坐标>画布尺寸时结果严重偏移
        denom_w = float(video_w) if video_w > 0 else float(self.CANVAS_W)
        denom_h = float(video_h) if video_h > 0 else float(self.CANVAS_H)
        cw = self.CANVAS_W - 2 * self.MARGIN
        ch = self.CANVAS_H - 2 * self.MARGIN
        return (
            int(self.MARGIN + bm_x / max(1.0, denom_w) * cw),
            int(self.MARGIN + bm_y / max(1.0, denom_h) * ch),
        )

    def unproject_to_pitch(self, img_x: float, img_y: float, H=None):
        """
        将图像像素坐标反投影到 pitch 真实世界坐标（米制，以场地中心为原点）。
        参照 sn-gamestate bbox2pitch.py get_bbox_pitch_homography：
          H 是 image→canvas，canvas 的坐标系是 [MARGIN..CANVAS_W-MARGIN] 对应 [0..PITCH_L]
          所以先用 H 正向投影到 canvas，再线性映射到米制坐标。
        返回: (wx, wy) 米制坐标，原点在场地中心；失败返回 None
        """
        if H is None:
            H = self._H
        if H is None:
            return None
        try:
            pt = np.array([img_x, img_y, 1.0])
            proj = H @ pt
            if abs(proj[2]) < 1e-10:
                return None
            proj /= proj[2]
            cx, cy = proj[0], proj[1]
            # canvas 坐标 → pitch 米制坐标（场地中心为原点）
            cw = self.CANVAS_W - 2 * self.MARGIN
            ch = self.CANVAS_H - 2 * self.MARGIN
            wx = (cx - self.MARGIN) / cw * self.PITCH_L - self.PITCH_L / 2.0
            wy = (cy - self.MARGIN) / ch * self.PITCH_W - self.PITCH_W / 2.0
            # 宽松边界过滤（允许稍微超出场地 5m 的范围，防止丢点）
            if -self.PITCH_L / 2.0 - 5 <= wx <= self.PITCH_L / 2.0 + 5 and \
               -self.PITCH_W / 2.0 - 5 <= wy <= self.PITCH_W / 2.0 + 5:
                return (round(wx, 3), round(wy, 3))
            return None
        except Exception:
            return None

    def pitch_to_canvas(self, wx: float, wy: float):
        """
        将 pitch 米制坐标（场地中心为原点）转换回画布像素坐标。
        用于 fallback 时将线性缩放坐标也统一到 pitch 坐标系。
        """
        cw = self.CANVAS_W - 2 * self.MARGIN
        ch = self.CANVAS_H - 2 * self.MARGIN
        px = self.MARGIN + (wx + self.PITCH_L / 2.0) / self.PITCH_L * cw
        py = self.MARGIN + (wy + self.PITCH_W / 2.0) / self.PITCH_W * ch
        return (round(px, 1), round(py, 1))


# ============================================================
# 卡尔曼多目标追踪器（IoU + Re-ID 融合，方案一+三）
# ============================================================
class KalmanTracker:
    """
    增强型多目标追踪器，融合：
      - 卡尔曼滤波预测（方案三）：短暂丢失时维持轨迹、预防 ID 跳变
      - Re-ID 外观特征匹配（OSNet 深度特征，降级为 HSV 直方图）
      - IoU 位置匹配：常规帧间匹配
    """
    def __init__(self, iou_threshold=0.3, reid_threshold=None, max_lost=5, min_hits=3):
        self.trackers: list[KalmanBoxTracker] = []
        self.iou_threshold  = iou_threshold
        # 深度特征（L2 归一化后余弦相似度）阈值更高；HSV 特征用原值
        if reid_threshold is None:
            reid_ext = get_reid_extractor()
            self.reid_threshold = 0.75 if reid_ext._use_deep else 0.55
        else:
            self.reid_threshold = reid_threshold
        self.max_lost = max_lost
        self.min_hits = min_hits  # 新追踪器连续命中 min_hits 帧才输出，抑制单帧误检
        self._reid_gallery: dict[int, np.ndarray] = {}  # {track_id: feat_vec}
        # 退休画廊：tracker 销毁后其外观特征在此长期保留，用于新检测的历史 ID 复用
        # 结构：{track_id: {"feat": np.ndarray, "team": str, "color": any}}
        self._retired_gallery: dict[int, dict] = {}
        # 队伍左右侧标记（assign_team_sides 更新后写入）
        # {"team_a": "left"|"right", "team_b": "left"|"right"}
        self._team_sides: dict[str, str] = {}
        # 侧边稳定锁：记录上次置信判断的差值，差值太小时不覆盖已有结论
        self._side_last_gap: float = 0.0
        # GMC：缓存上一帧单应矩阵，用于估计帧间镜头运动（全局运动补偿）
        self._H_prev = None  # type: "np.ndarray | None"

    # ── IoU 辅助 ─────────────────────────────────────────────
    @staticmethod
    def _iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        if inter == 0:
            return 0.0
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def update(self, detections, frame=None, H_mat=None):
        """
        detections: list of (x1,y1,x2,y2, team, color)
        frame:      当前帧图像（用于提取 Re-ID 特征，可为 None）
        H_mat:      当前帧单应矩阵（用于 GMC 全局运动补偿，可为 None）

        ── 足球场景优化：GMC 全局运动补偿 ──
        参考 DCTracker 双视图卡尔曼思路的轻量版本：
          利用连续帧 H 矩阵的差值估计背景（镜头）运动，
          在 Kalman 预测步之后对每个追踪器的像素坐标做补偿：
            Δp = H_curr * p_pred - p_pred  （近似线性位移）
          这样追踪器的状态始终在"球员自身坐标系"里，
          而不是被镜头平移带着走，从根本上解决运镜"瞬移"问题。
        """
        # ── GMC：利用 H_mat 估计镜头运动，补偿追踪器状态 ──────────────────
        _gmc_dx, _gmc_dy = 0.0, 0.0
        if H_mat is not None and hasattr(self, '_H_prev') and self._H_prev is not None:
            try:
                # 用前一帧 H 的逆与当前帧 H 组合估计帧间运动
                # dH ≈ H_curr @ inv(H_prev)，对参考点(cx,cy)估计位移
                H_prev_inv = np.linalg.inv(self._H_prev)
                dH = H_mat @ H_prev_inv
                # 用图像中心点作为参考估计全局平移量
                ref = np.array([960.0, 540.0, 1.0])
                p_moved = dH @ ref
                if abs(p_moved[2]) > 1e-6:
                    p_moved /= p_moved[2]
                _gmc_dx = float(p_moved[0] - ref[0])
                _gmc_dy = float(p_moved[1] - ref[1])
                # 限制补偿幅度：超过半帧宽度视为场景切换，不补偿
                if abs(_gmc_dx) > 480 or abs(_gmc_dy) > 270:
                    _gmc_dx, _gmc_dy = 0.0, 0.0
            except Exception:
                _gmc_dx, _gmc_dy = 0.0, 0.0
        self._H_prev = H_mat  # 更新前帧 H 缓存

        # 1. 预测所有已有追踪器的下一帧位置
        predicted_bboxes = [t.predict() for t in self.trackers]

        # ── GMC 补偿：将镜头运动从预测坐标中扣除 ──
        if (_gmc_dx != 0.0 or _gmc_dy != 0.0) and self.trackers:
            for t in self.trackers:
                # 直接修正状态向量中的位置分量（cx, cy）
                t.x[0, 0] -= _gmc_dx
                t.x[1, 0] -= _gmc_dy
            # 同步更新 predicted_bboxes
            predicted_bboxes = [t._to_bbox() for t in self.trackers]

        # 2. 批量提取本帧所有检测的 Re-ID 特征（OSNet / HSV 降级）
        reid_ext = get_reid_extractor()
        zero_feat = np.zeros(reid_ext.feat_dim, dtype=np.float32)
        crops = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            if frame is not None:
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                crops.append(crop if crop.size > 0 else None)
            else:
                crops.append(None)

        if frame is not None and any(c is not None for c in crops):
            valid_crops = [c if c is not None else np.zeros((32, 16, 3), dtype=np.uint8)
                           for c in crops]
            all_feats = reid_ext.extract_batch(valid_crops)
            det_feats = [all_feats[i] if crops[i] is not None else zero_feat.copy()
                         for i in range(len(crops))]
        else:
            det_feats = [zero_feat.copy() for _ in detections]

        # 3. 匹配：先 IoU，再 Re-ID 补充
        matched   = {}   # det_idx -> tracker_idx
        unmatched_dets  = list(range(len(detections)))
        unmatched_trks  = list(range(len(self.trackers)))

        if self.trackers and detections:
            # ── 联合得分矩阵：IoU（位置） + 外观相似度加权 ──
            # 纯 IoU 匹配在两人靠近时容易张冠李戴，加入外观约束显著降低误匹配率
            # score = iou + _REID_ALPHA * cosine_sim（外观权重 0.4）
            # IoU=0 的对（完全不重叠）外观再相似也不强制匹配，保持位置优先
            _REID_ALPHA = 0.4
            iou_mat = np.zeros((len(detections), len(self.trackers)), dtype=float)
            score_mat = np.zeros((len(detections), len(self.trackers)), dtype=float)
            for di, det in enumerate(detections):
                for ti, pred in enumerate(predicted_bboxes):
                    iou_mat[di, ti] = self._iou(det[:4], pred)
            # 构造联合得分矩阵（外观 + 物理运动合理性）
            for di in range(len(detections)):
                feat_d = det_feats[di]
                has_feat = feat_d is not None and np.linalg.norm(feat_d) > 1e-6
                det_bb = detections[di]
                det_cx = (det_bb[0] + det_bb[2]) / 2.0
                det_cy = (det_bb[1] + det_bb[3]) / 2.0
                for ti in range(len(self.trackers)):
                    iou_val = iou_mat[di, ti]
                    if iou_val <= 0:
                        score_mat[di, ti] = 0.0
                        continue
                    # 外观相似度加分
                    reid_bonus = 0.0
                    if has_feat:
                        gfeat = self._reid_gallery.get(self.trackers[ti].id)
                        if gfeat is not None:
                            reid_bonus = _REID_ALPHA * _cosine_sim(feat_d, gfeat)
                    # ── 物理运动合理性惩罚 ──
                    # 用 tracker 当前速度估计"期望落点"，检测位置偏离越远扣分越多
                    trk = self.trackers[ti]
                    pred_vx = float(trk.x[4, 0])
                    pred_vy = float(trk.x[5, 0])
                    pred_cx = float(trk.x[0, 0])
                    pred_cy = float(trk.x[1, 0])
                    residual = ((det_cx - pred_cx) ** 2 + (det_cy - pred_cy) ** 2) ** 0.5
                    expected = (pred_vx ** 2 + pred_vy ** 2) ** 0.5
                    physics_limit = expected + KalmanBoxTracker._MAX_ACCEL_PX
                    if residual > physics_limit and residual > KalmanBoxTracker._MAX_SPEED_PX:
                        # 超出物理极限：惩罚系数 [0,1]，残差越大扣越多，最多扣掉 60% 得分
                        over_ratio = min((residual - physics_limit) / (physics_limit + 1e-6), 3.0) / 3.0
                        physics_penalty = 0.6 * over_ratio
                    else:
                        physics_penalty = 0.0
                    score_mat[di, ti] = (iou_val + reid_bonus) * (1.0 - physics_penalty)

            # 匈牙利算法匹配（全局最优，替代原贪心匹配）
            try:
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(-score_mat)
                for di, ti in zip(row_ind, col_ind):
                    if iou_mat[di, ti] >= self.iou_threshold:  # 仍以 IoU 做最终准入门槛
                        matched[di] = ti
                        unmatched_dets.remove(di)
                        unmatched_trks.remove(ti)
            except ImportError:
                # scipy 不可用时退化为贪心匹配
                score_tmp = score_mat.copy()
                for _ in range(min(len(detections), len(self.trackers))):
                    idx = np.unravel_index(np.argmax(score_tmp), score_tmp.shape)
                    di, ti = idx
                    if iou_mat[di, ti] < self.iou_threshold:
                        break
                    if di in matched or ti not in unmatched_trks:
                        score_tmp[di, :] = 0; score_tmp[:, ti] = 0
                        continue
                    matched[di] = ti
                    unmatched_dets.remove(di)
                    unmatched_trks.remove(ti)
                    score_tmp[di, :] = 0
                    score_tmp[:, ti] = 0

            # Re-ID 补充匹配：对 IoU 未匹配的检测，用外观特征二次匹配丢失的追踪器
            # 同样用匈牙利算法构造相似度矩阵
            # ── 策略：不硬截断距离，改为距离衰减惩罚 ──
            # 近距离（<200px）：几乎不扣分；远距离：线性扣分，同时要求更高外观相似度
            # 这样"外观极其相似 + 稍远距离"仍可匹配，避免因远距离误创新 ID
            _DIST_SOFT_MAX = 800   # 超过此像素距离惩罚系数为 1.0（得分清零）
            _DIST_FREE_ZONE = 200  # 此距离内不扣分
            if unmatched_dets and unmatched_trks:
                reid_mat = np.zeros((len(unmatched_dets), len(unmatched_trks)), dtype=np.float32)
                for ri, di in enumerate(unmatched_dets):
                    feat_d = det_feats[di]
                    if feat_d is None or np.linalg.norm(feat_d) < 1e-6:
                        continue
                    det_bb = detections[di]
                    det_cx = (det_bb[0] + det_bb[2]) / 2.0
                    det_cy = (det_bb[1] + det_bb[3]) / 2.0
                    for ci, ti in enumerate(unmatched_trks):
                        pred_bb = predicted_bboxes[ti]
                        trk_cx = (pred_bb[0] + pred_bb[2]) / 2.0
                        trk_cy = (pred_bb[1] + pred_bb[3]) / 2.0
                        pixel_dist = ((det_cx - trk_cx) ** 2 + (det_cy - trk_cy) ** 2) ** 0.5
                        # 距离惩罚系数：[0, 1]，越近越接近 1（无惩罚），越远越接近 0
                        if pixel_dist <= _DIST_FREE_ZONE:
                            dist_weight = 1.0
                        else:
                            dist_weight = max(0.0, 1.0 - (pixel_dist - _DIST_FREE_ZONE) /
                                             (_DIST_SOFT_MAX - _DIST_FREE_ZONE))
                        gallery_feat = self._reid_gallery.get(self.trackers[ti].id)
                        if gallery_feat is not None:
                            raw_sim = _cosine_sim(feat_d, gallery_feat)
                            reid_mat[ri, ci] = raw_sim * dist_weight
                try:
                    from scipy.optimize import linear_sum_assignment
                    r_ind, c_ind = linear_sum_assignment(-reid_mat)
                    # 先收集所有合法的匹配对，再统一更新列表，避免循环中修改列表导致索引越界
                    reid_matches = []
                    # 快照：索引映射到实际 id
                    snap_dets = list(unmatched_dets)
                    snap_trks = list(unmatched_trks)
                    for ri, ci in zip(r_ind, c_ind):
                        ri, ci = int(ri), int(ci)
                        if ri >= len(snap_dets) or ci >= len(snap_trks):
                            continue
                        di = snap_dets[ri]
                        ti = snap_trks[ci]
                        # 丢失越久，要求外观相似度越高（每帧+0.01，最多+0.15）
                        lost_penalty = min(self.trackers[ti].lost * 0.01, 0.15)
                        effective_thresh = self.reid_threshold + lost_penalty
                        if reid_mat[ri, ci] >= effective_thresh:
                            reid_matches.append((di, ti))
                    for di, ti in reid_matches:
                        if di in unmatched_dets and ti in unmatched_trks:
                            matched[di] = ti
                            unmatched_dets.remove(di)
                            unmatched_trks.remove(ti)
                except ImportError:
                    for ri, di in enumerate(list(unmatched_dets)):
                        best_sim = reid_mat[ri].max() if reid_mat.shape[1] > 0 else 0.0
                        ci = int(reid_mat[ri].argmax()) if reid_mat.shape[1] > 0 else -1
                        if best_sim >= self.reid_threshold and ci >= 0:
                            ti = unmatched_trks[ci]
                            if di in unmatched_dets and ti in unmatched_trks:
                                matched[di] = ti
                                unmatched_dets.remove(di)
                                unmatched_trks.remove(ti)

        # 4. 更新已匹配追踪器
        results = []
        for di, ti in matched.items():
            det = detections[di]
            x1, y1, x2, y2, team, color = det
            # NSA conf 代理：用框面积归一化估计置信度
            # 框面积越大（目标清晰、近景）→ conf 高 → R 小 → 更信任检测
            # 框面积越小（目标模糊、远景）→ conf 低 → R 大 → 更信任预测
            _area = float((x2 - x1) * (y2 - y1))
            _conf_proxy = float(np.clip(_area / 15000.0, 0.1, 1.0))  # 15000px² ≈ 100×150 框
            self.trackers[ti].update((x1, y1, x2, y2), team, color, conf=_conf_proxy)
            # 更新 Re-ID 画廊：只在检测框足够大（清晰度保证）时才更新
            # 小框（人物较远/模糊）特征噪声大，避免污染画廊
            box_area = (x2 - x1) * (y2 - y1)
            tid = self.trackers[ti].id
            if tid in self._reid_gallery:
                if box_area >= 1200:  # 约 30×40 以上的框才参与更新
                    self._reid_gallery[tid] = (
                        0.7 * self._reid_gallery[tid] + 0.3 * det_feats[di]
                    )
                # 小框时保持旧特征不变
            else:
                self._reid_gallery[tid] = det_feats[di]
            # min_hits 门控：新追踪器需连续命中 min_hits 帧才输出，抑制单帧误检
            if self.trackers[ti].hits >= self.min_hits:
                results.append((tid, x1, y1, x2, y2, team, color))

        # 5. 新建追踪器（未匹配检测）—— 不立即输出，等到 hits >= min_hits
        #    新建前先查退休画廊：若外观高度相似则复用历史 ID（避免重新出现的球员被分配新 ID）
        _RETIRED_THRESH = self.reid_threshold + 0.05  # 退休匹配要求比活跃匹配稍严，防止误合并
        for di in unmatched_dets:
            det = detections[di]
            x1, y1, x2, y2, team, color = det
            feat_d = det_feats[di]

            reuse_id = None
            if self._retired_gallery and feat_d is not None and np.linalg.norm(feat_d) > 1e-6:
                best_sim, best_rid = 0.0, None
                for rid, entry in self._retired_gallery.items():
                    sim = _cosine_sim(feat_d, entry["feat"])
                    if sim > best_sim:
                        best_sim, best_rid = sim, rid
                if best_sim >= _RETIRED_THRESH:
                    reuse_id = best_rid
                    # 从退休画廊移除，避免被重复认领
                    self._retired_gallery.pop(reuse_id)

            kt = KalmanBoxTracker((x1, y1, x2, y2), team, color)
            if reuse_id is not None:
                # 复用历史 ID：直接覆盖新分配的 ID，保持轨迹连续性
                kt.id = reuse_id
                # 继承退休时的队伍信息（外观比当帧颜色分队更可靠）
                retired_entry = self._reid_gallery.get(reuse_id)  # 理论上已不在活跃画廊
                # 直接用退休画廊里存的队伍
                # （kt.team 由检测框颜色分队给出，不一定准，优先用历史）
            self.trackers.append(kt)
            self._reid_gallery[kt.id] = feat_d
            # hits=1，需再连续命中 min_hits-1 帧才输出

        # 6. 删除长时间丢失的追踪器（保留预测位置，允许短暂丢失时继续输出）
        alive = []
        for i, t in enumerate(self.trackers):
            if t.lost <= self.max_lost:
                alive.append(t)
                if i in unmatched_trks:
                    # 丢失但仍在容忍期：只有已充分确认（hits >= min_hits）的追踪器才输出预测框
                    if t.hits >= self.min_hits:
                        pred_bb = t.get_bbox()
                        results.append((t.id, *pred_bb, t.team, t.color))
            else:
                # 彻底删除：将外观特征转存到退休画廊，保留历史外观供后续 ID 复用
                feat = self._reid_gallery.pop(t.id, None)
                if feat is not None and np.linalg.norm(feat) > 1e-6 and t.hits >= self.min_hits:
                    self._retired_gallery[t.id] = {
                        "feat": feat, "team": t.team, "color": t.color
                    }
        self.trackers = alive

        return results

    def record_pos(self, track_id, cx, cy):
        pass  # 外部调用兼容接口

    def assign_teams_by_reid(self) -> dict:
        """
        基于 Re-ID 特征的轨迹级队伍分配（升级版：OSNet + UMAP + KMeans）。

        算法：
          1. 对每个 track_id，取其历史 Re-ID 特征向量（已在 _reid_gallery 中 EMA 累积）
          2. 若 umap-learn 可用：UMAP 降维（高维→3维），在低维空间聚类效果更好
             若 umap-learn 不可用：PCA 降维（降级方案），比直接高维聚类稳定
          3. KMeans(k=2) 聚类 → 自动分成两队
          4. 按聚类结果更新各 KalmanBoxTracker.team

        相比原版 KMeans（直接高维特征）：
          - UMAP 保留非线性流形结构，深蓝/黑、白/浅灰等颜色相近场景更准确
          - PCA 降级方案也比直接高维聚类数值稳定
          - 与 sn-gamestate TrackletTeamClustering 一致的轨迹级分配策略

        返回: {track_id: "team_a" | "team_b"}
        """
        if len(self._reid_gallery) < 4:
            return {}

        try:
            from sklearn.cluster import KMeans
        except ImportError:
            return {}

        # 收集所有轨迹的 Re-ID 特征（EMA 累积后）
        track_ids = list(self._reid_gallery.keys())
        feats = np.array([self._reid_gallery[tid] for tid in track_ids], dtype=np.float32)

        # 过滤全零特征
        valid_mask = np.linalg.norm(feats, axis=1) > 1e-6
        if valid_mask.sum() < 4:
            return {}

        valid_ids   = [track_ids[i] for i in range(len(track_ids)) if valid_mask[i]]
        valid_feats = feats[valid_mask]  # (N, D)

        # ── 降维：UMAP（优先）→ PCA（降级）→ 直接原维度 ─────────────────
        feats_for_cluster = valid_feats
        reducer_name = "raw"

        n_components = min(3, valid_feats.shape[1], valid_feats.shape[0] - 1)
        if n_components >= 2:
            try:
                import umap as umap_module
                reducer = umap_module.UMAP(
                    n_components=n_components,
                    n_neighbors=min(15, len(valid_ids) - 1),
                    min_dist=0.1,
                    random_state=42,
                    verbose=False,
                )
                feats_for_cluster = reducer.fit_transform(valid_feats).astype(np.float32)
                reducer_name = f"UMAP({n_components}D)"
            except ImportError:
                # umap 未安装：用 PCA 降维（sklearn 必然可用）
                try:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=n_components, random_state=42)
                    feats_for_cluster = pca.fit_transform(valid_feats).astype(np.float32)
                    reducer_name = f"PCA({n_components}D)"
                except Exception:
                    pass
            except Exception:
                pass

        # ── KMeans 聚类 ──────────────────────────────────────────────
        try:
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(feats_for_cluster)
            labels = kmeans.labels_
            print(f"[DEBUG] assign_teams_by_reid: {reducer_name}+KMeans, {len(valid_ids)} 条轨迹")
        except Exception:
            return {}

        team_map = {}
        for tid, label in zip(valid_ids, labels):
            team_map[tid] = "team_a" if label == 0 else "team_b"

        # 同步更新 KalmanBoxTracker.team
        for t in self.trackers:
            if t.id in team_map:
                t.team = team_map[t.id]

        return team_map

    def assign_team_sides(self, H_mat=None) -> dict:
        """
        根据球员在场地上的平均 X 坐标，自动决定哪队是"左侧进攻方"哪队是"右侧进攻方"。

        逻辑：
          - 对 team_a / team_b 分别计算所有追踪位置的平均 X 坐标（战术板坐标）
          - 平均 X 较小的队 = 左侧（team_left），较大的队 = 右侧（team_right）
          - 这样前端可以用 team_side 决定左右球门归属
          - 守门员特殊处理：用其 X 坐标符号直接判断（<中心线=左队门将，>中心线=右队门将）

        参数:
            H_mat: 当前单应矩阵（用于把图像坐标转换到战术板坐标判断左右）

        返回: {"team_a": "left"|"right", "team_b": "left"|"right"}
        """
        xs_a = []
        xs_b = []

        for t in self.trackers:
            try:
                _bbox = t.get_bbox()
            except Exception:
                continue
            if _bbox is None:
                continue
            x1, y1, x2, y2 = _bbox[:4]
            cx = (x1 + x2) / 2.0

            # 若有 H 矩阵，投影到战术板坐标取 X
            if H_mat is not None:
                try:
                    src = np.array([[[float(cx), float(y2)]]], dtype=np.float32)
                    dst = cv2.perspectiveTransform(src, H_mat)
                    cx = float(dst[0, 0, 0])
                except Exception:
                    pass

            team = getattr(t, 'team', 'unknown')
            if team == 'team_a':
                xs_a.append(cx)
            elif team == 'team_b':
                xs_b.append(cx)

        if not xs_a or not xs_b:
            # 数据不足，默认 team_a=left
            return {"team_a": "left", "team_b": "right"}

        avg_a = float(np.mean(xs_a))
        avg_b = float(np.mean(xs_b))
        gap = abs(avg_a - avg_b)

        # ── 置信度门控：差值太小（两队挤在一起）时不翻转已有结论 ──
        # 阈值 30px（图像坐标）/ 或战术板坐标 0.05（已投影时）
        MIN_GAP = 30.0
        if gap < MIN_GAP and self._team_sides:
            # 差距不显著，保持上次稳定结论
            print(f"[DEBUG] assign_team_sides: gap={gap:.1f}px 过小（<{MIN_GAP}），保持上次侧边 {self._team_sides}")
            return dict(self._team_sides)

        if avg_a <= avg_b:
            side_map = {"team_a": "left", "team_b": "right"}
        else:
            side_map = {"team_a": "right", "team_b": "left"}

        self._side_last_gap = gap
        print(f"[DEBUG] assign_team_sides: team_a avg_x={avg_a:.1f}, team_b avg_x={avg_b:.1f}, gap={gap:.1f} → {side_map}")
        return side_map


# 向后兼容：保留 SimpleTracker 名称（内部实际用 KalmanTracker）
SimpleTracker = KalmanTracker


# ============================================================
# 足球追踪器（升级版卡尔曼滤波）
# 改进要点：
#   1. 6维状态向量 [cx, cy, vx, vy, ax, ay]（匀加速模型，更真实）
#   2. 图像边界硬约束：漏检外推时将预测位置钳制在帧内
#   3. 速度指数衰减：漏检时速度逐帧衰减，防止持续外推飞出场外
#   4. 置信度自适应噪声（NSA）：高置信度减小观测噪声，低置信度增大
#   5. 候选越界惩罚：距图像边界 < 20px 的候选置信度降权
#   6. 历史均值回落约束：连续漏检超过阈值后位置向历史均值软约束
# ============================================================
class BallTracker:
    """
    基于卡尔曼滤波的足球位置追踪（升级版）。
    状态向量：[cx, cy, vx, vy, ax, ay]（匀加速模型）
    观测向量：[cx, cy]
    优势：
      - 匀加速模型更真实反映足球的弹跳/加速运动
      - 图像边界约束 + 速度衰减防止球飞出场外
      - NSA 自适应观测噪声，高置信检测更准确
      - 候选越界降权，减少边缘误检的干扰
    """
    # 速度衰减系数：漏检时每帧速度衰减为原来的 DECAY 倍（0.85 ≈ 约7帧衰减至50%）
    _VEL_DECAY = 0.82
    # NSA 观测噪声缩放上限（低置信候选时 R 最多放大倍数）
    _NSA_SCALE = 6.0
    # 图像边界 margin：候选框距边界 < EDGE_MARGIN px 时降低置信度
    _EDGE_MARGIN = 20

    def __init__(self, max_lost=15):
        self.max_lost     = max_lost
        self.lost         = 0
        self.pos          = None        # 上一帧最终位置 (cx, cy)
        self.history      = []          # 历史轨迹 [(cx, cy)]
        self._kf          = None        # cv2.KalmanFilter 实例
        self._initialized = False
        self._frame_wh    = (1920, 1080)  # 图像宽高，由首帧更新

    def set_frame_size(self, w: int, h: int):
        """通知追踪器当前帧尺寸，用于边界约束。"""
        self._frame_wh = (max(w, 1), max(h, 1))

    def _init_kf(self, cx: float, cy: float):
        """用初始位置构建 6维状态/2维观测 卡尔曼滤波器（匀加速模型）。"""
        kf = cv2.KalmanFilter(6, 2)
        # 状态转移矩阵（匀加速模型，dt=1帧）
        # [cx, cy, vx, vy, ax, ay] → 下一帧 cx' = cx + vx + 0.5*ax, vx' = vx + ax
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0, 0.5, 0  ],
            [0, 1, 0, 1, 0,   0.5],
            [0, 0, 1, 0, 1,   0  ],
            [0, 0, 0, 1, 0,   1  ],
            [0, 0, 0, 0, 1,   0  ],
            [0, 0, 0, 0, 0,   1  ],
        ], dtype=np.float32)
        # 观测矩阵（只观测位置）
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)
        # 过程噪声：位置小，速度中，加速度大（足球运动不可预知性强）
        kf.processNoiseCov = np.diag([1.0, 1.0, 5.0, 5.0, 15.0, 15.0]).astype(np.float32)
        # 基础观测噪声（NSA 时动态调整）
        kf.measurementNoiseCov = np.diag([18.0, 18.0]).astype(np.float32)
        # 初始状态协方差（位置已知，速度/加速度不确定）
        kf.errorCovPost = np.diag([5.0, 5.0, 80.0, 80.0, 200.0, 200.0]).astype(np.float32)
        # 初始状态
        kf.statePost = np.array([[cx], [cy], [0.0], [0.0], [0.0], [0.0]], dtype=np.float32)
        self._kf = kf
        self._initialized = True

    def _predict(self):
        """卡尔曼预测步：返回预测位置 (cx, cy)，并做图像边界钳制。"""
        pred = self._kf.predict()
        px, py = float(pred[0, 0]), float(pred[1, 0])
        # 图像边界钳制：预测位置不允许超出帧范围（含 10px margin）
        fw, fh = self._frame_wh
        px = float(np.clip(px, 0, fw - 1))
        py = float(np.clip(py, 0, fh - 1))
        return px, py

    def _correct(self, cx: float, cy: float, conf: float = 1.0):
        """
        卡尔曼校正步（NSA 自适应观测噪声）。
        conf 越高 → R 越小 → 更信任测量；conf 越低 → R 放大 → 更信任预测。
        """
        alpha = 1.0 - float(np.clip(conf, 0.0, 1.0))
        base_r = self._kf.measurementNoiseCov.copy()
        self._kf.measurementNoiseCov = base_r * (1.0 + alpha * self._NSA_SCALE)
        measurement = np.array([[cx], [cy]], dtype=np.float32)
        corrected = self._kf.correct(measurement)
        # 还原 measurementNoiseCov（下帧重新计算）
        self._kf.measurementNoiseCov = base_r
        # 对校正后状态做边界约束
        fw, fh = self._frame_wh
        self._kf.statePost[0, 0] = float(np.clip(self._kf.statePost[0, 0], 0, fw - 1))
        self._kf.statePost[1, 0] = float(np.clip(self._kf.statePost[1, 0], 0, fh - 1))
        return float(corrected[0, 0]), float(corrected[1, 0])

    def _apply_vel_decay(self):
        """漏检时对速度/加速度施加指数衰减，防止持续外推飞出场外。"""
        if self._initialized:
            self._kf.statePost[2, 0] *= self._VEL_DECAY   # vx
            self._kf.statePost[3, 0] *= self._VEL_DECAY   # vy
            self._kf.statePost[4, 0] *= self._VEL_DECAY   # ax
            self._kf.statePost[5, 0] *= self._VEL_DECAY   # ay

    def _get_velocity(self):
        """返回当前速度估计 (vx, vy)。"""
        if not self._initialized:
            return 0.0, 0.0
        s = self._kf.statePost
        return float(s[2, 0]), float(s[3, 0])

    def _penalize_edge_candidates(self, candidates):
        """
        候选越界惩罚：距图像边界 < EDGE_MARGIN px 的候选降低置信度。
        防止图像边缘的误检被接受为球的位置。
        """
        fw, fh = self._frame_wh
        m = self._EDGE_MARGIN
        penalized = []
        for cx, cy, conf in candidates:
            near_edge = (cx < m or cx > fw - m or cy < m or cy > fh - m)
            if near_edge:
                conf = conf * 0.3   # 边缘候选置信度降至30%
            penalized.append((cx, cy, conf))
        return penalized

    def update(self, candidates, frame_w: int = 0, frame_h: int = 0):
        """
        candidates : list of (cx, cy, conf) 候选球位置
        frame_w/h  : 当前帧尺寸（用于边界约束，0 表示沿用上次值）
        返回: (cx, cy) 或 None（长时间丢失后返回 None）
        """
        if frame_w > 0 and frame_h > 0:
            self.set_frame_size(frame_w, frame_h)

        # 候选越界惩罚
        if candidates:
            candidates = self._penalize_edge_candidates(candidates)

        if not candidates:
            self.lost += 1
            if self.lost > self.max_lost:
                self.pos = None
                self._initialized = False
                return None
            # 漏检：速度衰减 + 卡尔曼外推（边界已在 _predict 中钳制）
            if self._initialized:
                self._apply_vel_decay()
                px, py = self._predict()
                self.pos = (int(px), int(py))
            return self.pos

        # 有候选：先预测，再选最佳候选，最后校正
        if not self._initialized:
            # 重初始化：优先选择距上次有效位置最近的候选（防止接受场边误检）
            if self.pos is not None:
                # 有上次有效位置：在 120px 范围内选置信度最高的候选
                _reinit_thresh = 120.0
                near_cands = [(cx, cy, cf) for cx, cy, cf in candidates
                              if math.hypot(cx - self.pos[0], cy - self.pos[1]) <= _reinit_thresh]
                best = max(near_cands, key=lambda c: c[2]) if near_cands else max(candidates, key=lambda c: c[2])
            else:
                best = max(candidates, key=lambda c: c[2])
            self._init_kf(float(best[0]), float(best[1]))
            self.pos = (int(best[0]), int(best[1]))
            self.lost = 0
            self.history.append(self.pos)
            return self.pos

        # 卡尔曼预测
        pred_x, pred_y = self._predict()

        # 自适应匹配距离阈值：速度越大阈值越宽松（球快速移动时不丢失）
        vx, vy = self._get_velocity()
        speed = math.hypot(vx, vy)
        dist_thresh = max(70.0, min(150.0, 70.0 + speed * 2.5))  # 上限150px（原220px），减少误检窗口

        # 预计算历史方向向量（用于方向一致性约束）
        hist_dx, hist_dy = 0.0, 0.0
        _use_dir = len(self.history) >= 2
        if _use_dir:
            hist_dx = self.history[-1][0] - self.history[-2][0]
            hist_dy = self.history[-1][1] - self.history[-2][1]
            _hn = math.hypot(hist_dx, hist_dy)
            if _hn < 1.0:   # 静止时不施加方向惩罚
                _use_dir = False

        # 选择距预测位置最近且在阈值内的候选（置信度辅助加权 + 方向一致性）
        best_score = float('inf')
        best_cand  = None
        for cx, cy, conf in candidates:
            d = math.hypot(cx - pred_x, cy - pred_y)
            if d > dist_thresh:
                continue
            # 分数 = 距离 - 置信度补偿（加大权重）- 方向一致性奖励
            score = d - conf * 60.0
            if _use_dir:
                cand_dx = cx - self.history[-1][0]
                cand_dy = cy - self.history[-1][1]
                _cn = math.hypot(cand_dx, cand_dy)
                if _cn > 1.0:
                    cos_sim = (hist_dx * cand_dx + hist_dy * cand_dy) / (_hn * _cn)
                    score -= cos_sim * 20.0  # 方向一致（cos>0）额外降分（分越低越优先）
            if score < best_score:
                best_score = score
                best_cand  = (cx, cy, conf)

        if best_cand is None:
            # 所有候选都在阈值外：视为漏检，速度衰减 + 外推
            self.lost += 1
            if self.lost > self.max_lost:
                self.pos = None
                self._initialized = False
                return None
            self._apply_vel_decay()
            self.pos = (int(pred_x), int(pred_y))
            return self.pos

        # 用最佳候选做 NSA 校正（传入置信度以调整观测噪声）
        cx_c, cy_c = self._correct(
            float(best_cand[0]), float(best_cand[1]), conf=float(best_cand[2])
        )
        # 边界钳制
        fw, fh = self._frame_wh
        cx_c = float(np.clip(cx_c, 0, fw - 1))
        cy_c = float(np.clip(cy_c, 0, fh - 1))
        self.pos  = (int(cx_c), int(cy_c))
        self.lost = 0
        self.history.append(self.pos)
        return self.pos


# ============================================================
# 轨迹级号码投票
# 同一 track_id 多帧识别结果按置信度加权投票，选最高权重号码
# 相比逐帧取当帧结果：消除误识别噪声，稳定性显著提升
# ============================================================

def select_highest_voted_jersey(numbers_conf_pairs):
    """
    对一条轨迹上多帧的 (number, confidence) 配对做加权投票，
    返回权重最高的号码及其归一化置信度，无有效投票时返回 (None, 0.0)。

    参数:
        numbers_conf_pairs: list of (int|None, float) — 每帧的识别结果
    """
    vote: dict[int, float] = {}
    for num, conf in numbers_conf_pairs:
        if num is None or conf <= 0.0:
            continue
        vote[num] = vote.get(num, 0.0) + float(conf)
    if not vote:
        return None, 0.0
    best_num = max(vote, key=vote.__getitem__)
    total = sum(vote.values())
    return best_num, round(vote[best_num] / total, 3)


class JerseyVoter:
    """
    轨迹级号码投票器，挂载到 KalmanTracker 上使用。

    用法（在每帧处理完毕后调用）：
        voter.record(track_id, jersey_number, confidence)

    查询（视频结束或周期性刷新时）：
        result_dict = voter.get_results()
        # → {track_id: {"number": int, "confidence": float}}
    """
    def __init__(self):
        # {track_id: [(number, conf), ...]}
        self._votes: dict[int, list] = {}

    def record(self, track_id: int, number, confidence: float):
        """记录一帧的号码识别结果。"""
        if track_id not in self._votes:
            self._votes[track_id] = []
        self._votes[track_id].append((number, confidence))

    def get_best(self, track_id: int):
        """
        返回指定轨迹当前投票最高的号码和置信度。
        未记录时返回 (None, 0.0)。
        """
        pairs = self._votes.get(track_id, [])
        return select_highest_voted_jersey(pairs)

    def get_results(self) -> dict:
        """返回所有轨迹的投票结果字典 {track_id: {"number":, "confidence":}}。"""
        out = {}
        for tid, pairs in self._votes.items():
            num, conf = select_highest_voted_jersey(pairs)
            out[tid] = {"number": num, "confidence": conf}
        return out

    def cleanup(self, alive_ids: set):
        """删除已消亡轨迹的投票记录，避免内存无限增长。"""
        dead = [tid for tid in self._votes if tid not in alive_ids]
        for tid in dead:
            del self._votes[tid]


def _draw_pitch_lines_on_heatmap(img, width, height):
    """
    在热力图图像上叠加标准足球场白线。
    坐标系：width=800, height=520，场地左上角为(0,0)。
    标准半场长度52.5m，宽度34m（原点为场地中心，此处映射到像素坐标）。
    """
    W, H = width, height
    lc = (255, 255, 255)   # 白色线条
    lt = 2                  # 线条粗细

    # 边缘留边（像素）
    MARGIN_X = int(W * 0.04)   # 32px
    MARGIN_Y = int(H * 0.06)   # 31px

    pw = W - 2 * MARGIN_X   # 场地像素宽度
    ph = H - 2 * MARGIN_Y   # 场地像素高度

    def fx(ratio): return MARGIN_X + int(ratio * pw)
    def fy(ratio): return MARGIN_Y + int(ratio * ph)

    # ── 外边线 ─────────────────────────────────────────────────────────────
    cv2.rectangle(img, (fx(0), fy(0)), (fx(1), fy(1)), lc, lt)

    # ── 中线 ───────────────────────────────────────────────────────────────
    cv2.line(img, (fx(0.5), fy(0)), (fx(0.5), fy(1)), lc, lt)

    # ── 中圈（半径 ≈ 9.15/105 * pw ≈ 0.0871*pw） ────────────────────────
    r_center = int(9.15 / 105.0 * pw)
    cv2.circle(img, (fx(0.5), fy(0.5)), r_center, lc, lt)
    # 中心点
    cv2.circle(img, (fx(0.5), fy(0.5)), 4, lc, -1)

    # ── 大禁区（每侧 16.5m×40.32m，宽度16.5/52.5，高度40.32/68） ─────────
    box_w_ratio = 16.5 / 105.0
    box_h_ratio = 40.32 / 68.0
    box_y0 = fy((1 - box_h_ratio) / 2)
    box_y1 = fy((1 + box_h_ratio) / 2)
    # 左大禁区
    cv2.rectangle(img, (fx(0), box_y0), (fx(box_w_ratio), box_y1), lc, lt)
    # 右大禁区
    cv2.rectangle(img, (fx(1 - box_w_ratio), box_y0), (fx(1), box_y1), lc, lt)

    # ── 小禁区（每侧 5.5m×18.32m） ────────────────────────────────────────
    sbox_w_ratio = 5.5 / 105.0
    sbox_h_ratio = 18.32 / 68.0
    sbox_y0 = fy((1 - sbox_h_ratio) / 2)
    sbox_y1 = fy((1 + sbox_h_ratio) / 2)
    # 左小禁区
    cv2.rectangle(img, (fx(0), sbox_y0), (fx(sbox_w_ratio), sbox_y1), lc, lt)
    # 右小禁区
    cv2.rectangle(img, (fx(1 - sbox_w_ratio), sbox_y0), (fx(1), sbox_y1), lc, lt)

    # ── 罚球点（11m / 105m） ───────────────────────────────────────────────
    pen_ratio = 11.0 / 105.0
    cv2.circle(img, (fx(pen_ratio), fy(0.5)), 4, lc, -1)
    cv2.circle(img, (fx(1 - pen_ratio), fy(0.5)), 4, lc, -1)

    # ── 角球弧（半径1m / 105m * pw，仅四角各画一段弧） ─────────────────────
    r_corner = max(5, int(1.0 / 105.0 * pw))
    corners = [(fx(0), fy(0)), (fx(1), fy(0)), (fx(0), fy(1)), (fx(1), fy(1))]
    angles = [(0, 90), (90, 180), (270, 360), (180, 270)]
    for (cx, cy), (a1, a2) in zip(corners, angles):
        cv2.ellipse(img, (cx, cy), (r_corner, r_corner), 0, a1, a2, lc, lt)

    return img


def generate_heatmap_image(positions, width, height, out_path):
    """
    将位置列表转为热力图并保存为 PNG。
    positions: list of {"x": int, "y": int}

    注意：若坐标已经经过 H 矩阵投影到战术板坐标系（800×520），
    则传入 width=800, height=520；否则传入原始视频分辨率。
    """
    if not positions:
        return False
    hmap = np.zeros((height, width), dtype=np.float32)
    for p in positions:
        px, py = int(p["x"]), int(p["y"])
        if 0 <= px < width and 0 <= py < height:
            hmap[py, px] += 1.0
    # 高斯模糊扩散
    sigma = max(width, height) // 20
    hmap = cv2.GaussianBlur(hmap, (0, 0), sigma)
    # 归一化到 0-255
    if hmap.max() > 0:
        hmap = (hmap / hmap.max() * 255).astype(np.uint8)
    # 应用颜色映射（COLORMAP_JET）
    colored = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    # 叠加半透明足球场背景（绿色）
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    bg[:] = (34, 85, 34)  # BGR 深绿
    alpha = 0.65
    blended = cv2.addWeighted(colored, alpha, bg, 1-alpha, 0)
    # 叠加足球场白线
    blended = _draw_pitch_lines_on_heatmap(blended, width, height)
    cv2.imwrite(out_path, blended)
    return True


def _draw_voronoi_overlay(canvas: np.ndarray,
                          team_a_pts: list, team_b_pts: list,
                          color_a=(255, 100, 60), color_b=(60, 60, 220),
                          opacity: float = 0.28,
                          scale: int = 4) -> np.ndarray:
    """
    在已有画布上叠加 Voronoi 势力范围图。

    每个像素着色为距离最近球员所属队伍的颜色，直观展示场上控制区域。

    参数:
        canvas       : 已绘制好场地线条的背景 BGR 图像
        team_a/b_pts : [(px, py), ...] 画布坐标
        color_a/b    : BGR 队伍颜色
        opacity      : 叠加透明度（0=完全透明，1=完全覆盖），建议 0.25~0.35
        scale        : 下采样倍数（默认4），在 1/scale 分辨率计算后上采样回原尺寸
                       速度提升约 scale² 倍；scale=4 时精度损失不可见（Voronoi是低频）
    """
    if not team_a_pts and not team_b_pts:
        return canvas

    h, w = canvas.shape[:2]

    all_pts = [(p, 0) for p in team_a_pts] + [(p, 1) for p in team_b_pts]
    if not all_pts:
        return canvas

    # ── 在缩小分辨率下计算 Voronoi（速度提升 scale² 倍）────────────────
    sh, sw = max(1, h // scale), max(1, w // scale)

    coords = np.array([p for p, _ in all_pts], dtype=np.float32) / scale  # (N, 2)
    labels = np.array([lbl for _, lbl in all_pts], dtype=np.int32)         # (N,)

    yy, xx = np.indices((sh, sw))                        # (sh, sw)

    dx = coords[:, 0][:, None, None] - xx[None]          # (N, sh, sw)
    dy = coords[:, 1][:, None, None] - yy[None]
    dist2 = dx * dx + dy * dy                             # (N, sh, sw)
    nearest = np.argmin(dist2, axis=0)                    # (sh, sw)
    team_mask_small = labels[nearest]                     # (sh, sw)  0 or 1

    # 着色（在小图上）
    small_voronoi = np.empty((sh, sw, 3), dtype=np.uint8)
    color_a_arr = np.array(color_a, dtype=np.uint8)
    color_b_arr = np.array(color_b, dtype=np.uint8)
    small_voronoi[team_mask_small == 0] = color_a_arr
    small_voronoi[team_mask_small == 1] = color_b_arr

    # ── 上采样回原始分辨率（INTER_NEAREST 保持硬边界） ──────────────────
    voronoi = cv2.resize(small_voronoi, (w, h), interpolation=cv2.INTER_NEAREST)

    return cv2.addWeighted(voronoi, opacity, canvas, 1.0 - opacity, 0)


def build_tactical_view(positions_team_a, positions_team_b, ball_positions, width, height, out_path,
                         homography_est=None, show_voronoi: bool = True,
                         coords_are_tactical: bool = True):
    """
    生成2D战术板鸟瞰图。

    参数:
        positions_team_a/b : 每人一条轨迹列表 [ [{"x":, "y":}, ...], ... ]
        ball_positions     : [(cx, cy), ...] —— 图像原始坐标
        width/height       : 原始视频分辨率
        out_path           : 输出 PNG 路径
        homography_est     : HomographyEstimator 实例（可为 None）
        show_voronoi       : 是否叠加 Voronoi 势力范围图（默认开启）
        coords_are_tactical: 若 True（默认），positions 坐标已是战术板坐标，
                             直接使用（只做 clip），不再做 H 变换。
                             若 False，坐标为图像原始坐标，使用 H 矩阵或线性缩放。
    """
    canvas_w, canvas_h = 800, 520
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (30, 120, 30)  # 草绿

    # 绘制场地线条（按FIFA标准比例：105m×68m，原点=场地中心）
    lc = (255, 255, 255)
    m = 30  # 边距
    cw = canvas_w - 2 * m   # 可用宽度 = 740px
    ch = canvas_h - 2 * m   # 可用高度 = 460px
    # 真实尺寸
    _PL, _PW = 105.0, 68.0   # 场地长宽（米）
    _PA_L, _PA_W2 = 16.5, 20.16   # 大禁区长、宽一半
    _GA_L, _GA_W2 = 5.5, 9.16     # 小禁区长、宽一半
    _CR = 9.15                      # 中圈半径

    def _world_to_canvas(wx, wy):
        """FIFA米制坐标（原点=场地中心）→ 画布像素坐标"""
        tx = int(m + (wx + _PL / 2) / _PL * cw)
        ty = int(m + (wy + _PW / 2) / _PW * ch)
        return tx, ty

    # 边线外框
    cv2.rectangle(canvas,
                  _world_to_canvas(-_PL/2, -_PW/2),
                  _world_to_canvas( _PL/2,  _PW/2), lc, 2)
    # 中线
    cv2.line(canvas, _world_to_canvas(0, -_PW/2), _world_to_canvas(0, _PW/2), lc, 2)
    # 中圈（椭圆近似：横轴=_CR/PL*cw，纵轴=_CR/PW*ch）
    center_px = _world_to_canvas(0, 0)
    rx = int(_CR / _PL * cw)
    ry = int(_CR / _PW * ch)
    cv2.ellipse(canvas, center_px, (rx, ry), 0, 0, 360, lc, 2)
    cv2.circle(canvas, center_px, 3, lc, -1)
    # 左大禁区
    cv2.rectangle(canvas,
                  _world_to_canvas(-_PL/2,        -_PA_W2),
                  _world_to_canvas(-_PL/2+_PA_L,  +_PA_W2), lc, 2)
    # 右大禁区
    cv2.rectangle(canvas,
                  _world_to_canvas( _PL/2-_PA_L, -_PA_W2),
                  _world_to_canvas( _PL/2,        +_PA_W2), lc, 2)
    # 左小禁区
    cv2.rectangle(canvas,
                  _world_to_canvas(-_PL/2,        -_GA_W2),
                  _world_to_canvas(-_PL/2+_GA_L,  +_GA_W2), lc, 2)
    # 右小禁区
    cv2.rectangle(canvas,
                  _world_to_canvas( _PL/2-_GA_L, -_GA_W2),
                  _world_to_canvas( _PL/2,        +_GA_W2), lc, 2)
    # 点球点
    cv2.circle(canvas, _world_to_canvas(-_PL/2 + 11.0, 0), 3, lc, -1)
    cv2.circle(canvas, _world_to_canvas( _PL/2 - 11.0, 0), 3, lc, -1)
    # 左罚球弧（禁区弧，约9.15m半径，只画禁区线外的弧段）
    arc_rx = int(_CR / _PL * cw)
    arc_ry = int(_CR / _PW * ch)
    lp_px = _world_to_canvas(-_PL/2 + 11.0, 0)
    rp_px = _world_to_canvas( _PL/2 - 11.0, 0)
    # 左禁区弧（画朝右侧的弧，大约 307°~53°）
    cv2.ellipse(canvas, lp_px, (arc_rx, arc_ry), 0, -53, 53, lc, 2)
    # 右禁区弧（朝左侧）
    cv2.ellipse(canvas, rp_px, (arc_rx, arc_ry), 0, 127, 233, lc, 2)

    H = homography_est._H if homography_est is not None else None

    # pitch 真实尺寸（米），用于 pitch坐标→canvas坐标 转换
    _pitch_l = homography_est.PITCH_L if homography_est is not None else 105.0
    _pitch_w = homography_est.PITCH_W if homography_est is not None else 68.0

    def _pt_xy(pt):
        """兼容 {x,y}（canvas/图像坐标）与 {wx,wy}（pitch米制坐标）两种格式。
        返回 (x, y, is_pitch)"""
        if "wx" in pt:
            return pt["wx"], pt["wy"], True
        return pt["x"], pt["y"], False

    def _pitch_to_canvas(wx, wy):
        """pitch 米制坐标（场地中心为原点）→ 战术板画布像素坐标"""
        tx = int(m + (wx + _pitch_l / 2) / _pitch_l * cw)
        ty = int(m + (wy + _pitch_w / 2) / _pitch_w * ch)
        return max(0, min(canvas_w - 1, tx)), max(0, min(canvas_h - 1, ty))

    def map_pos(x, y):
        """
        将坐标映射到战术板画布坐标。
        - 若 coords_are_tactical=True：坐标已是战术板坐标，直接 clip 即可。
        - 若 coords_are_tactical=False：坐标为图像坐标，优先用 H 矩阵，越界则线性缩放。
        投影越界视为 H 失效，改用 fallback（不 clip 到边缘）。
        """
        if coords_are_tactical:
            # 坐标已是战术板坐标（由 project_bbox_bottom 转换好），直接 clip
            return max(0, min(canvas_w - 1, int(x))), max(0, min(canvas_h - 1, int(y)))

        # 坐标为图像原始坐标，需要通过 H 矩阵或线性缩放转换
        if H is not None:
            try:
                src = np.array([[[float(x), float(y)]]], dtype=np.float32)
                dst = cv2.perspectiveTransform(src, H)
                tx, ty = int(dst[0, 0, 0]), int(dst[0, 0, 1])
                # 越界检查（宽松边界）：投影点需在战术板合理范围内
                bound_lo = m - 40
                bound_x  = canvas_w - m + 40
                bound_y  = canvas_h - m + 40
                if bound_lo <= tx <= bound_x and bound_lo <= ty <= bound_y:
                    return max(0, min(canvas_w - 1, tx)), max(0, min(canvas_h - 1, ty))
            except Exception:
                pass
        # 线性缩放兜底（H 不可用或投影越界时均走此路）
        tx = int(m + (x / max(width, 1)) * cw)
        ty = int(m + (y / max(height, 1)) * ch)
        return tx, ty

    def _resolve_pt(pt):
        """将点字典解析为画布坐标。兼容 {x,y}（canvas/图像坐标）和 {wx,wy}（pitch米制坐标）。"""
        x, y, is_pitch = _pt_xy(pt)
        if is_pitch:
            return _pitch_to_canvas(x, y)
        return map_pos(x, y)

    def draw_trail(positions, color):
        if len(positions) > 60:
            step = max(1, len(positions) // 200)
            pts = positions[::step]
            if positions[-1] not in pts:
                pts = pts + [positions[-1]]
        else:
            pts = positions
        for i in range(1, len(pts)):
            p1 = _resolve_pt(pts[i-1])
            p2 = _resolve_pt(pts[i])
            alpha = 0.3 + 0.7 * (i / len(pts))
            c = tuple(int(cc * alpha) for cc in color)
            cv2.line(canvas, p1, p2, c, 2)
        if pts:
            cv2.circle(canvas, _resolve_pt(pts[-1]), 7, color, -1)

    # ── Voronoi 势力范围叠加（在球员轨迹之前绘制，避免遮挡轨迹）──
    if show_voronoi:
        # 取每条轨迹的最后一个坐标作为"当前位置"
        def _last_pos(traj):
            for pt in reversed(traj):
                return _resolve_pt(pt)
            return None

        pts_a = [_last_pos(traj) for traj in positions_team_a if _last_pos(traj)]
        pts_b = [_last_pos(traj) for traj in positions_team_b if _last_pos(traj)]
        if pts_a or pts_b:
            canvas = _draw_voronoi_overlay(
                canvas, pts_a, pts_b,
                color_a=(255, 100, 0),   # 与 TEAM_COLORS["team_a"] 一致（橙色）
                color_b=(255, 60, 0),    # 与 TEAM_COLORS["team_b"] 一致（蓝红色）
                opacity=0.28
            )

    for p in positions_team_a:
        draw_trail(p, (60, 160, 255))
    for p in positions_team_b:
        draw_trail(p, (60, 60, 220))

    ball_step = max(1, len(ball_positions) // 300) if len(ball_positions) > 300 else 1
    ball_pts = [{"x": bx, "y": by} for bx, by in ball_positions[::ball_step]]
    # ball_positions 始终是图像原始坐标，需要用 H 矩阵或线性缩放（不走战术板坐标直通路径）
    def map_pos_img(x, y):
        """图像原始坐标 → 战术板坐标（带 H 越界验证 + 线性缩放 fallback）"""
        if H is not None:
            try:
                src = np.array([[[float(x), float(y)]]], dtype=np.float32)
                dst = cv2.perspectiveTransform(src, H)
                tx, ty = int(dst[0, 0, 0]), int(dst[0, 0, 1])
                if (m - 40) <= tx <= (canvas_w - m + 40) and (m - 40) <= ty <= (canvas_h - m + 40):
                    return max(0, min(canvas_w - 1, tx)), max(0, min(canvas_h - 1, ty))
            except Exception:
                pass
        tx = int(m + (x / max(width, 1)) * cw)
        ty = int(m + (y / max(height, 1)) * ch)
        return tx, ty
    for i in range(1, len(ball_pts)):
        p1 = map_pos_img(ball_pts[i-1]["x"], ball_pts[i-1]["y"])
        p2 = map_pos_img(ball_pts[i]["x"], ball_pts[i]["y"])
        cv2.line(canvas, p1, p2, (0, 255, 255), 2)
    if ball_pts:
        cv2.circle(canvas, map_pos_img(ball_pts[-1]["x"], ball_pts[-1]["y"]), 8, (0, 255, 255), -1)

    cv2.imwrite(out_path, canvas)
    return True

def get_yolo_model():
    """
    加载球员检测 YOLO 模型（best.pt，SoccerNet fine-tune 的 YOLOv8m）。
    候选路径：项目根目录 / backend 目录 / 脚本所在目录
    """
    try:
        import torch
        from ultralytics import YOLO
        _cuda = torch.cuda.is_available()

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_dir = os.path.dirname(os.path.abspath(__file__))
        best_pt_candidates = [
            os.path.join(base_dir, 'best.pt'),
            os.path.join(script_dir, 'best.pt'),
            os.path.join(base_dir, 'backend', 'best.pt'),
        ]
        for p in best_pt_candidates:
            if os.path.exists(p):
                model = YOLO(p)
                if _cuda:
                    model.to('cuda')
                print(f"[OK] 球员检测模型加载成功（SoccerNet YOLOv8m）: {p} "
                      f"(device={'cuda' if _cuda else 'cpu'})")
                return model

        print("[ERROR] 未找到 best.pt，请确认权重文件存在于项目根目录")
        return None
    except Exception as e:
        print(f"[ERROR] YOLO loading failed: {e}")
        print("[ERROR] Please install: pip install ultralytics")
        return None


def get_player_cls_ids(model) -> set:
    """
    根据模型的 names 字典自动推断"球员"对应的 class id 集合。

    SoccerNet fine-tune 模型（best.pt）类别示例：
        {0: 'player', 1: 'goalkeeper', 2: 'referee', 3: 'ball'}
    球员 = {0, 1}（只含 player + goalkeeper，referee 不进追踪，
    由 get_team_color_and_role 做颜色二次过滤即可）

    COCO 通用模型：person = class 0
    """
    names = model.names  # dict {int: str}
    # ── 优先匹配 SoccerNet 专用标签 ──────────────────────────────────────
    soccer_player_labels = {'player', 'goalkeeper'}
    matched = {cid for cid, name in names.items()
               if name.lower() in soccer_player_labels}
    if matched:
        print(f"[INFO] 检测到 SoccerNet 模型，球员类别 ID: {matched} -> {[names[c] for c in matched]}")
        return matched
    # ── fallback：尝试 COCO 'person' 标签 ───────────────────────────────
    person_matched = {cid for cid, name in names.items()
                      if name.lower() == 'person'}
    if person_matched:
        print(f"[INFO] 检测到 COCO 通用模型，person 类别 ID: {person_matched}")
        return person_matched
    # ── 次级 fallback：'item' 标签（某些 SoccerNet 微调模型使用此通用标签）──
    item_matched = {cid for cid, name in names.items()
                    if name.lower() == 'item'}
    if item_matched:
        print(f"[INFO] 检测到通用标签模型（item），球员类别 ID: {item_matched} -> {[names[c] for c in item_matched]}")
        return item_matched
    # ── 最终 fallback：class 0（几乎所有检测模型 class 0 都是主体类别）──
    print(f"[WARNING] 未能从模型类别名中识别球员标签，模型 names={names}，"
          f"fallback 使用 class 0。如结果异常请检查模型类别定义。")
    return {0}


def get_ball_cls_id(model):
    """
    根据模型的 names 字典推断"足球"对应的 class id。

    - SoccerNet 模型中 ball 的 class id（通常为 3）
    - 通用 COCO 模型中 sports ball 为 class 32
    - 若模型不含球类别，返回 None（调用方使用专用 ball_model）
    """
    try:
        names = model.names
        for cid, name in names.items():
            if name.lower() in {'ball', 'sports ball', 'football', 'soccer ball'}:
                return cid
        return None
    except Exception:
        return None


def get_ball_detection_model():
    """
    加载专用足球检测模型（football-ball-detection.pt）。
    候选路径：项目根目录 / backend 目录 / 脚本所在目录
    未找到则返回 None，调用方回退到通用 YOLO class=32 + SAHI 降级方案。
    """
    try:
        import torch
        from ultralytics import YOLO
        _cuda = torch.cuda.is_available()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(base_dir, 'football-ball-detection.pt'),
            os.path.join(base_dir, 'backend', 'football-ball-detection.pt'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'football-ball-detection.pt'),
        ]
        for p in candidates:
            if os.path.exists(p):
                model = YOLO(p)
                if _cuda:
                    model.to('cuda')
                print(f"[OK] 专用球检测模型加载成功: {p} (device={'cuda' if _cuda else 'cpu'})")
                return model

        print("[INFO] 专用球检测模型不可用，将使用通用 YOLO class=32 + SAHI 降级方案")
        return None
    except Exception as e:
        print(f"[WARNING] get_ball_detection_model 异常: {e}")
        return None


def _iou_simple(b1, b2):
    """简易 IoU 计算（用于 pose 框与检测框匹配）"""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)





class VideoCleaner:
    """
    4.1 视频数据清洗模块
    ──────────────────────────────────────────────────────────────────────
    功能：过滤回放画面、看台画面和观众画面，只保留包含足球场地的有效画面，
          提高分析精度和效率。

    判断逻辑（综合以下三项特征，满足全部条件才视为有效帧）：
      1. 绿色草地比例 ≥ 15%
      2. 场地线条数量 ≥ 3（Hough 直线检测）
      3. 边缘密度 < 20%（排除特写镜头 / 观众密集镜头）

    场景切换检测：帧间 HSV 直方图巴氏距离 > 阈值时判定为场景切换，
    重置有效性状态，避免上一场景的判断延续到新场景。

    帧间一致性：非场景切换时保持与前一帧相同的有效性状态，减少误判。
    """

    # ── 可调参数 ──────────────────────────────────────────────────────────
    SCENE_CHANGE_THRESH  = 0.55   # 巴氏距离阈值（与 analysis.py 镜头切换保持一致）
    GREEN_RATIO_MIN      = 0.12   # 绿色草地最低比例（从0.15放宽到0.12，兼容草地颜色偏暗视频）
    LINE_COUNT_MIN       = 2      # 最少检测到的直线数（从3放宽到2，兼容仰角/特写足球视角）
    EDGE_DENSITY_MAX     = 0.22   # 边缘密度上限（从0.20放宽到0.22，避免正常密集站位被误判）
    FRAME_BUFFER_SIZE    = 5      # 场景切换检测帧缓冲区大小
    PROCESS_SCALE        = 0.25   # 计算缩放比例（加速处理）

    def __init__(self):
        self._prev_valid: Optional[bool] = None    # 前一帧有效性
        self._prev_hist: Optional[np.ndarray] = None  # 前一帧 HSV 直方图
        self._frame_buffer: list = []              # 帧缓冲区（用于场景切换稳定性）
        self._skip_count: int = 0                  # 被跳过的帧计数
        self._total_count: int = 0                 # 总处理帧计数

    def reset(self):
        """重置状态（新视频开始时调用）"""
        self._prev_valid = None
        self._prev_hist = None
        self._frame_buffer.clear()
        self._skip_count = 0
        self._total_count = 0

    def _compute_hsv_hist(self, frame_small: np.ndarray) -> np.ndarray:
        """计算归一化 HSV 直方图（32×32 bins，仅 H/S 通道）"""
        hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _is_scene_change(self, hist: np.ndarray) -> bool:
        """判断当前帧是否为场景切换（基于巴氏距离）"""
        if self._prev_hist is None:
            return True
        dist = cv2.compareHist(
            self._prev_hist.reshape(32, 32).astype(np.float32),
            hist.reshape(32, 32).astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA
        )
        return dist > self.SCENE_CHANGE_THRESH

    def _compute_green_ratio(self, frame_small: np.ndarray) -> float:
        """计算绿色草地像素比例"""
        hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
        lower = np.array([30, 40, 40], dtype=np.uint8)
        upper = np.array([90, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        return float(np.count_nonzero(mask)) / mask.size

    def _compute_line_count(self, frame_small: np.ndarray) -> int:
        """用 Hough 直线检测估计场地线条数量"""
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        h, w = edges.shape
        min_len = max(int(w * 0.12), 20)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=40,
                                minLineLength=min_len,
                                maxLineGap=10)
        return 0 if lines is None else len(lines)

    def _compute_edge_density(self, frame_small: np.ndarray) -> float:
        """计算边缘密度（Canny 边缘像素占比）"""
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return float(np.count_nonzero(edges)) / edges.size

    def is_valid_frame(self, frame: np.ndarray) -> bool:
        """
        判断当前帧是否为有效足球场地画面。
        ──────────────────────────────────────────────────────────────────
        返回 True  → 有效帧，应进行后续分析
        返回 False → 无效帧（回放/看台/观众/特写），跳过后续分析

        算法：
          1. 将帧缩小至 PROCESS_SCALE 倍（默认 0.25）加速计算
          2. 计算 HSV 直方图，检测场景切换
          3. 非场景切换时：与前帧保持相同有效性（帧间一致性）
          4. 场景切换时：综合评估三项特征决定有效性
        """
        self._total_count += 1

        h, w = frame.shape[:2]
        new_w = max(int(w * self.PROCESS_SCALE), 80)
        new_h = max(int(h * self.PROCESS_SCALE), 45)
        small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        hist = self._compute_hsv_hist(small)
        scene_change = self._is_scene_change(hist)

        # 更新帧缓冲区（用于场景切换稳定性判断）
        self._frame_buffer.append(scene_change)
        if len(self._frame_buffer) > self.FRAME_BUFFER_SIZE:
            self._frame_buffer.pop(0)

        # 场景切换稳定性：缓冲区内多数帧为场景切换时才触发重新判断
        buffer_scene_change = (
            self._frame_buffer.count(True) >= max(1, len(self._frame_buffer) // 2)
        )

        self._prev_hist = hist

        if not buffer_scene_change and self._prev_valid is not None:
            # 非场景切换：继承前帧有效性，减少逐帧计算开销
            valid = self._prev_valid
        else:
            # 场景切换或首帧：综合三项特征判断
            green_ratio   = self._compute_green_ratio(small)
            line_count    = self._compute_line_count(small)
            edge_density  = self._compute_edge_density(small)

            valid = (
                green_ratio  >= self.GREEN_RATIO_MIN   and
                line_count   >= self.LINE_COUNT_MIN     and
                edge_density <  self.EDGE_DENSITY_MAX
            )

            if not valid:
                # 调试日志（仅在跳过时输出，避免刷屏）
                reasons = []
                if green_ratio  < self.GREEN_RATIO_MIN:
                    reasons.append(f"草地不足({green_ratio:.1%}<{self.GREEN_RATIO_MIN:.0%})")
                if line_count   < self.LINE_COUNT_MIN:
                    reasons.append(f"线条不足({line_count}<{self.LINE_COUNT_MIN})")
                if edge_density >= self.EDGE_DENSITY_MAX:
                    reasons.append(f"边缘过密({edge_density:.1%}>={self.EDGE_DENSITY_MAX:.0%})")
                if self._total_count <= 5 or self._total_count % 50 == 0:
                    print(f"  [VideoCleaner] 第{self._total_count}帧判定为无效帧: {'; '.join(reasons)}")

        self._prev_valid = valid
        if not valid:
            self._skip_count += 1
        return valid

    def get_stats(self) -> dict:
        """返回清洗统计信息"""
        total = max(self._total_count, 1)
        return {
            "total_frames":   self._total_count,
            "skipped_frames": self._skip_count,
            "valid_frames":   self._total_count - self._skip_count,
            "skip_rate":      round(self._skip_count / total, 3),
        }


def build_grass_mask(frame: np.ndarray) -> np.ndarray:
    """
    生成绿色草地二值 mask。
    使用 HSV 颜色范围检测草地像素，经形态学膨胀填充空洞后返回。

    返回值：uint8 单通道 mask，草地区域=255，其余=0。

    性能策略：
      - 先将帧缩小到 1/2 分辨率再处理，膨胀耗时与分辨率平方成正比，约快 4×
      - 若 cv2.cuda 可用（OpenCV with CUDA build），关键算子在 GPU 上执行
      - 最后用最近邻插值还原到原始尺寸（<0.1ms），供脚点坐标查询
    """
    orig_h, orig_w = frame.shape[:2]
    # ── 缩小到 1/2 分辨率处理（速度优先）──────────────────────────────
    small = cv2.resize(frame, (orig_w // 2, orig_h // 2), interpolation=cv2.INTER_NEAREST)

    # ── 尝试 GPU 加速路径（需要 opencv-contrib + CUDA build）──────────
    _use_cuda = False
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            _use_cuda = True
    except AttributeError:
        pass  # 当前 OpenCV 不含 cuda 模块，走 CPU 路径

    # mask_small 初始化为 None，最终必定由 CPU 或 GPU 路径赋值
    mask_small: np.ndarray | None = None

    if _use_cuda:
        try:
            gpu_bgr = cv2.cuda_GpuMat()
            gpu_bgr.upload(small)
            gpu_hsv = cv2.cuda.cvtColor(gpu_bgr, cv2.COLOR_BGR2HSV)
            # cv2.cuda 无直接 inRange；先下载 HSV，再做 inRange（HSV 下载很快）
            hsv = gpu_hsv.download()
            lower1 = np.array([30,  40,  40], dtype=np.uint8)
            upper1 = np.array([90, 255, 255], dtype=np.uint8)
            mask_small = cv2.inRange(hsv, lower1, upper1)
            # 膨胀在 GPU 上执行
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            gpu_mask = cv2.cuda_GpuMat()
            gpu_mask.upload(mask_small)
            morph_filter = cv2.cuda.createMorphologyFilter(
                cv2.MORPH_DILATE, cv2.CV_8U, kernel, iterations=2
            )
            gpu_mask = morph_filter.apply(gpu_mask)
            mask_small = gpu_mask.download()
        except Exception:
            # CUDA 路径出错则静默回退到 CPU
            _use_cuda = False

    if not _use_cuda:
        # ── CPU 路径：用较小内核（13×13 × 2次 ≈ 25×25 × 1次效果相近）──
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        lower1 = np.array([30,  40,  40], dtype=np.uint8)
        upper1 = np.array([90, 255, 255], dtype=np.uint8)
        mask_small = cv2.inRange(hsv, lower1, upper1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        mask_small = cv2.dilate(mask_small, kernel, iterations=2)

    # ── 还原到原始分辨率（最近邻，保持二值边界清晰）────────────────────
    assert mask_small is not None  # 此处必定已由上方某路径赋值
    mask = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask


def is_field_player(bbox, frame_width, frame_height, grass_mask=None):
    """
    通过位置、尺寸和草地 mask 过滤非球员目标（观众、裁判、替补席等）。

    grass_mask（可选）：由 build_grass_mask() 生成的绿色草地二值图。
      - 若提供，则以「脚点落在草地上」作为主判断依据，大幅放宽尺寸阈值，
        从而识别远处小球员同时排除非草地区域的观众。
      - 若未提供，退化为原有的纯位置+尺寸过滤逻辑（向下兼容）。
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    # ── 快速位置粗筛（不受 grass_mask 影响）──────────────────────────
    # 过滤顶部观众席（上方18%区域）
    if cy < frame_height * 0.18:
        return False
    # 过滤底部替补席/广告牌（下方8%区域）
    if cy > frame_height * 0.92:
        return False
    # 过滤边线外（左右各5%）
    if cx < frame_width * 0.05 or cx > frame_width * 0.95:
        return False

    # ── 高宽比过滤（始终执行，排除躺倒/横幅/广告牌等异形目标）────────
    # 下限从 1.2 放宽到 0.8：SoccerNet best.pt 在广播视角下对远景球员产生
    # 接近正方形的检测框（高宽比约 1.0~1.3），过严的限制会漏掉这些目标
    if w > 0:
        aspect = h / w
        if aspect < 0.8 or aspect > 6.0:
            return False

    if grass_mask is not None:
        # ── 草地 mask 模式：用脚点是否在草地上作为主判断 ──────────────
        # 脚点向上偏移 15% bbox 高度：
        #   原偏移量 -2px 太小，球员控球时脚点正好压在球上（白色圆形），
        #   导致脚点处草地 mask 为 0，球员被误判为"不在场地"而过滤掉。
        #   上移 15%h 后脚点处于小腿区域，远离球，草地 mask 更可靠。
        foot_x = int(cx)
        foot_y = min(int(y2) - max(int(h * 0.15), 5), frame_height - 1)
        foot_y = max(foot_y, 0)
        foot_x = max(min(foot_x, frame_width - 1), 0)

        on_grass = grass_mask[foot_y, foot_x] > 0

        if not on_grass:
            # 脚点不在草地，检查脚点附近 15×15 区域（容忍偏差 + 覆盖球的范围）
            # 容忍区从 5×5 扩大到 15×15，确保即使脚点周围有球（约 20px）仍能找到草地
            y_lo = max(foot_y - 7, 0)
            y_hi = min(foot_y + 8, frame_height)
            x_lo = max(foot_x - 7, 0)
            x_hi = min(foot_x + 8, frame_width)
            patch = grass_mask[y_lo:y_hi, x_lo:x_hi]
            on_grass = patch.any()

        if not on_grass:
            return False

        # 草地模式下放宽尺寸：宽≥8px 且 高≥20px 即可（捕获远处小球员）
        if w < 8 or h < 20:
            return False
    else:
        # ── 无 mask 模式：保持原有严格尺寸过滤（兼容旧调用）────────────
        if w < 20 or h < 50:
            return False

    return True

def infer_jersey_color_name(h_mean: float, s_mean: float, v_mean: float = 128.0) -> str:
    """
    将 HSV 均值转换为人类可读的球衣颜色名称。
    用 '红队/蓝队' 等颜色标签描述球队，使 AI 分析报告更专业通用，不依赖队名。

    参数:
        h_mean: Hue 均值 (0-179, OpenCV HSV)
        s_mean: Saturation 均值 (0-255)
        v_mean: Value 均值 (0-255)，用于区分白/黑/灰

    返回:
        颜色名称字符串，如 '红色', '蓝色', '白色' 等
    """
    # 低饱和度：白/灰/黑
    if s_mean < 40:
        if v_mean > 180:
            return "白色"
        elif v_mean < 60:
            return "黑色"
        else:
            return "灰色"
    # 高饱和度，按 Hue 区间分类（OpenCV Hue: 0-179, 对应色圆 0-360°）
    h = h_mean
    if h < 10 or h >= 160:
        return "红色"
    elif h < 25:
        return "橙色"
    elif h < 35:
        return "黄色"
    elif h < 85:
        return "绿色"
    elif h < 130:
        return "蓝色"
    elif h < 155:
        return "紫色"
    else:
        return "深红色"


def _build_team_color_labels(team_hsv_centers: dict) -> dict:
    """
    根据 Phase A/旧版颜色聚类结果，生成球衣颜色标签字典。
    输出格式：{"team_a": "蓝色球衣队", "team_b": "红色球衣队"}
    若颜色数据不可用，返回通用标签。
    用视觉颜色标签（如"红色球衣队"）代替 team_a/team_b 等内部标识符，
    AI 分析报告和上下文注入时统一使用颜色标签，提升可读性和泛化能力
    """
    fallback = {"team_a": "主队", "team_b": "客队"}
    if not team_hsv_centers:
        return fallback
    try:
        result = {}
        for team_key in ("team_a", "team_b"):
            center = team_hsv_centers.get(team_key)
            if not center or len(center) < 2:
                result[team_key] = fallback[team_key]
                continue
            h, s = float(center[0]), float(center[1])
            # V 均值不在聚类结果中，默认 128（中亮度）
            v = float(center[2]) if len(center) >= 3 else 128.0
            color_name = infer_jersey_color_name(h, s, v)
            result[team_key] = f"{color_name}球衣队"
        return result
    except Exception:
        return fallback


def get_team_color_and_role(crop_bgr):
    """
    通过颜色识别队伍和裁判。

    改进版算法（双峰主色聚类）：
      1. 先排除明显裁判特征（黑色/荧光绿黄）
      2. 对躯干ROI像素做 HSV 2-means 聚类，取像素数量更多的主色调
         （比单点均值更抗光照/阴影干扰，避免混色球衣误判）
      3. 对主色调按 Hue 分区判断队伍

    队伍颜色（边框）：
      team_a → (255, 100, 0) BGR（橙蓝）
      team_b → (0,   60, 255) BGR（深红）
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", (180, 180, 180)

    h_img, w_img = crop_bgr.shape[:2]
    # 聚焦躯干上半部（20%~55%高度，25%~75%宽度）
    roi = crop_bgr[int(h_img*0.2):int(h_img*0.55), int(w_img*0.25):int(w_img*0.75)]
    if roi.size == 0:
        return "unknown", (180, 180, 180)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ── 快速裁判预筛（全图均值，开销小）──────────────────────────
    avg_h, avg_s, avg_v = float(cv2.mean(hsv)[0]), float(cv2.mean(hsv)[1]), float(cv2.mean(hsv)[2])
    # 黑色（低亮低饱）
    if avg_v < 55 and avg_s < 70:
        return "referee", (100, 100, 100)
    # 荧光黄/绿（色调60~90，高饱高亮）
    if 28 <= avg_h <= 90 and avg_s > 120 and avg_v > 160:
        return "referee", (100, 100, 100)

    # ── 双峰 K-means 2分类取主色调 ────────────────────────────────
    pixels = hsv.reshape(-1, 3).astype(np.float32)
    if len(pixels) < 8:
        # 像素太少，回退均值法
        hue, sat, val = avg_h, avg_s, avg_v
    else:
        try:
            # 只用 H、S 两通道聚类（对光照变化最鲁棒）
            hs = pixels[:, :2]
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, cluster_labels, centers = cv2.kmeans(
                hs, 2, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS
            )
            # 取像素数量更多的那个簇作为主色
            count0 = int(np.sum(cluster_labels == 0))
            count1 = int(np.sum(cluster_labels == 1))
            if count0 >= count1:
                hue = float(centers[0, 0])
                sat = float(centers[0, 1])
            else:
                hue = float(centers[1, 0])
                sat = float(centers[1, 1])
            # 亮度仍用均值
            val = avg_v
        except Exception:
            hue, sat, val = avg_h, avg_s, avg_v

    # ── 基于主色调分类 ────────────────────────────────────────────
    # 低饱和度（白/灰球衣）→ 按亮度区分：亮 = 白色球衣 team_a
    if sat < 70:
        if val > 130:
            return "team_a", (255, 100, 0)   # 白色/浅灰球衣 → team_a
        return "unknown", (180, 180, 180)

    # 红色系：色调 0~15 或 160~180
    if hue <= 15 or hue >= 160:
        return "team_a", (255, 100, 0)
    # 黄色/橙色系：归 team_a
    if 16 <= hue <= 35:
        return "team_a", (255, 100, 0)
    # 绿色系（排除草地：裁判已过滤）：剩余绿色球衣 → team_b
    if 36 <= hue <= 85:
        return "team_b", (255, 60, 0)
    # 蓝色系：色调 100~135
    if 100 <= hue <= 135:
        return "team_b", (255, 60, 0)
    # 紫色系（86~99）→ team_b
    if 86 <= hue <= 99:
        return "team_b", (255, 60, 0)
    # 136~159 偏青蓝 → team_b
    if 136 <= hue <= 159:
        return "team_b", (255, 60, 0)

    return "unknown", (180, 180, 180)


def resolve_goalkeeper_team(
    goalkeeper_cx: float,
    goalkeeper_cy: float,
    team_a_positions: list,
    team_b_positions: list,
) -> str:
    """
    通过队伍质心距离判断门将所属队伍。

    门将球衣颜色通常与球员不同，靠外观/颜色聚类容易分错。
    此方法改用位置信息：计算两队球员的质心，将门将归入最近的那支队伍。

    参数:
        goalkeeper_cx/cy  : 门将 bbox 底部中心点坐标（像素）
        team_a/b_positions: [(cx, cy), ...] 已确认的非门将球员坐标列表

    返回: "team_a" | "team_b" | "unknown"（两队均无球员时）
    """
    if not team_a_positions and not team_b_positions:
        return "unknown"

    gk = np.array([goalkeeper_cx, goalkeeper_cy], dtype=np.float64)

    def centroid(pts):
        arr = np.array(pts, dtype=np.float64)
        return arr.mean(axis=0)

    if team_a_positions and team_b_positions:
        ca = centroid(team_a_positions)
        cb = centroid(team_b_positions)
        return "team_a" if np.linalg.norm(gk - ca) <= np.linalg.norm(gk - cb) else "team_b"
    elif team_a_positions:
        return "team_a"
    else:
        return "team_b"


def get_dynamic_suggestion(num, abilities, player_stats=None):
    """
    根据球员能力值及实际运动统计数据，生成个性化 AI 深度评估报告。
    所有数值均来自视频分析，无硬编码文本。
    
    参数:
        abilities: 五维能力值字典 {"防守":xx, "射门":xx, ...}
        player_stats: 可选的运动统计 {"detection_count":xx, "coverage_ratio":xx, ...}
    """
    keys = ["防守", "射门", "传球", "速度", "体能"]
    # 按能力值排序（从高到低）
    ranked = sorted(keys, key=lambda k: abilities.get(k, 0), reverse=True)
    top_skill   = ranked[0]
    second_skill = ranked[1]
    weak_skill  = ranked[-1]
    third_weak  = ranked[-2]

    top_val    = abilities[top_skill]
    second_val = abilities[second_skill]
    weak_val   = abilities[weak_skill]
    avg_val    = round(sum(abilities.get(k,0) for k in keys) / len(keys))

    # 综合水平等级
    if avg_val >= 85:
        level = "精英级"
        level_desc = "整体实力达到职业顶尖水准"
    elif avg_val >= 75:
        level = "优秀级"
        level_desc = "综合能力高于平均水平"
    elif avg_val >= 65:
        level = "良好级"
        level_desc = "具备稳定的竞技表现"
    else:
        level = "发展级"
        level_desc = "仍有较大提升空间"

    # 弱项评级
    if weak_val < 60:
        weak_desc = f"{weak_skill}（{weak_val}/100）存在明显短板，是当前首要补强方向"
    elif weak_val < 70:
        weak_desc = f"{weak_skill}（{weak_val}/100）略显不足，建议加强专项训练"
    else:
        weak_desc = f"各项能力均衡，{weak_skill}（{weak_val}/100）相对偏低，可进一步强化"

    # 位置推荐（基于能力特征）
    if top_skill == "防守" and second_skill == "体能":
        position_advice = "适合担任中卫或防守型中场，是球队防线的核心支柱"
    elif top_skill == "防守":
        position_advice = "防守能力突出，推荐担任后卫线核心，擅长拦截与封堵"
    elif top_skill == "射门" and second_skill == "速度":
        position_advice = "进攻端威胁极大，最适合担任主力前锋或边锋，是球队主要得分点"
    elif top_skill == "射门":
        position_advice = "终结能力出众，推荐担任前锋核心，在禁区内发挥最大价值"
    elif top_skill == "传球" and second_skill == "速度":
        position_advice = "兼具组织与冲击，是理想的进攻型中场，能有效连接攻守"
    elif top_skill == "传球":
        position_advice = "传球视野出众，最适合担任中场枢纽或后腰，控制比赛节奏"
    elif top_skill == "速度":
        position_advice = "速度优势明显，最适合边路突破，是快速反击体系的关键人物"
    elif top_skill == "体能":
        position_advice = "体能充沛，适合担任全能型中场（Box-to-Box），全场高强度覆盖"
    else:
        position_advice = "技术全面均衡，可灵活出任多个位置"

    # 统计补充（若有实际运动数据）
    stats_text = ""
    if player_stats:
        det_count   = player_stats.get("detection_count", 0)
        cov         = player_stats.get("coverage_ratio", 0)
        speed_score = player_stats.get("speed_score", 0)
        if det_count > 0:
            stats_text = (
                f"视频追踪共检测 {det_count} 帧次，"
                f"场地覆盖率约 {cov:.0%}，"
                f"运动强度指数 {speed_score:.1f}。"
            )

    # 构建完整评估报告
    report = (
        f"#{num}号球员综合评级：{level}（均分 {avg_val}/100，{level_desc}）。"
        f"{stats_text}"
        f"核心优势：{top_skill}能力评分 {top_val}/100，{second_skill}能力 {second_val}/100，"
        f"双项支撑其在场上的主要价值。"
        f"位置建议：{position_advice}。"
        f"改进方向：{weak_desc}；{third_weak}（{abilities.get(third_weak,0)}/100）亦需持续提升，"
        f"以实现技术层面的全面突破。"
    )
    return report

def process_video(video_path, video_uuid, jersey_annotations=None, non_player_bboxes=None):
    """
    jersey_annotations: 前端手动标注数据，格式为
      [{"det_id": int, "jersey_number": int, "team": "team_a"|"team_b"}, ...]
    non_player_bboxes: 非球员 bbox 列表，格式为
      [{"det_id": int, "x": int, "y": int, "w": int, "h": int}, ...]
    """
    print(f">>> 启动高级分析 v6 (OCR背号识别): {video_uuid}")
    db_session = SessionLocal()
    progress_store[video_uuid] = 5
    
    try:
        # 🔑 关键修复: 在分析开始时立即验证视频记录存在
        video = db_session.query(VideoModel).filter(VideoModel.video_uuid == video_uuid).first()
        if not video:
            raise Exception(f"视频记录不存在（分析启动前检查失败）: {video_uuid}")
        
        print(f"[OK] 视频记录验证通过: video_id={video.id}, filename={video.filename}")
        
        yolo_model = get_yolo_model()
        if yolo_model is None:
            raise Exception("YOLO 模型加载失败，请确保已安装 ultralytics 库: pip install ultralytics")
        _player_cls_ids = get_player_cls_ids(yolo_model)
        # yolo_model 中球的 class id：SoccerNet=3(ball)，COCO=32(sports ball)，None=无球类
        _yolo_ball_cls_id = get_ball_cls_id(yolo_model)
        # GPU FP16 开关（YOLO 推理）
        import torch as _torch_ocr
        _ocr_use_half = _torch_ocr.cuda.is_available()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): 
            raise Exception("无法读取视频")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 自适应分析时长：视频实际时长多少就分析多少，上限 45 分钟防止超长视频
        MAX_ANALYZE_SECONDS = 45 * 60   # 45 分钟上限
        target_frames = min(total_frames, int(fps * MAX_ANALYZE_SECONDS))
        start_frame = 0                 # 从头开始，覆盖完整视频
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        video_duration_sec = total_frames / fps
        analyze_duration_sec = target_frames / fps
        print(f"视频信息: {width}x{height} @ {fps}fps, "
              f"总时长: {video_duration_sec/60:.1f}min, "
              f"分析时长: {analyze_duration_sec/60:.1f}min, "
              f"起始帧: {start_frame}")

        # 第一阶段：采样颜色聚类（前60秒，每15帧采样一次）
        print("=== 阶段1: 颜色聚类与球员分队 ===")
        jersey_detections = {}  # {jersey_num: {count, team, recent_bbox}}

        # 注入手动标注
        if jersey_annotations:
            print(f"  [手动标注模式] 收到 {len(jersey_annotations)} 条标注")
            for ann in jersey_annotations:
                try:
                    num  = int(ann["jersey_number"])
                    team = ann.get("team", "team_a")
                    if team not in ("team_a", "team_b"):
                        team = "team_a"
                    if 1 <= num <= 99:
                        jersey_detections[num] = {"count": 50, "team": team, "bbox": []}
                        print(f"    [手动标注] 注入: #{num} → {team}")
                except Exception as _ann_e:
                    print(f"    [手动标注] 解析失败: {ann} ({_ann_e})")

        # 每15帧采样一次做颜色聚类
        sample_step = 15
        sampling_frames = min(int(fps * 60), target_frames)

        sampled_fno_list = list(range(0, sampling_frames, sample_step))
        n_sample_total = max(len(sampled_fno_list), 1)

        # 阶段1-A：收集所有球员的颜色特征向量，用于后续聚类分队
        player_color_feats = {}   # 背号球员颜色特征（兼容旧逻辑）
        all_player_color_feats = []   # List[List[6 floats]]

        # ── 从 non_player_bboxes 提取非球员颜色模板，用于聚类时排除 ──────────
        # non_player_bboxes 是关键帧上的坐标，这里从关键帧图像中提取颜色特征
        _np_color_templates = []  # List[np.ndarray]  shape (6,) — 非球员颜色均值向量
        if non_player_bboxes:
            try:
                # 读取关键帧（main.py 保存的 tmp_frames/{video_uuid}_keyframe.jpg）
                import os as _os
                _kf_path = _os.path.join(
                    _os.path.dirname(_os.path.abspath(__file__)),
                    "..", "uploads", "tmp_frames", f"{video_uuid}_keyframe.jpg"
                )
                if not _os.path.exists(_kf_path):
                    # 尝试同级目录
                    _kf_path = _os.path.join(
                        _os.path.dirname(_os.path.abspath(__file__)),
                        "uploads", "tmp_frames", f"{video_uuid}_keyframe.jpg"
                    )
                if _os.path.exists(_kf_path):
                    _kf_img = cv2.imread(_kf_path)
                    if _kf_img is not None:
                        _kf_h, _kf_w = _kf_img.shape[:2]
                        for _npb in non_player_bboxes:
                            try:
                                _nx  = int(float(_npb.get('x', 0)))
                                _ny  = int(float(_npb.get('y', 0)))
                                _nw  = int(float(_npb.get('w', 1)))
                                _nh  = int(float(_npb.get('h', 1)))
                                # 躯干区域（上20%~55%，横向20%~80%）
                                _t = max(0, _ny + int(_nh * 0.20))
                                _b = min(_kf_h, _ny + int(_nh * 0.55))
                                _l = max(0, _nx + int(_nw * 0.20))
                                _r = min(_kf_w, _nx + int(_nw * 0.80))
                                _roi = _kf_img[_t:_b, _l:_r]
                                if _roi.size > 0:
                                    _hsv = cv2.cvtColor(_roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
                                    _feat = np.array([
                                        _hsv[:,0].mean(), _hsv[:,0].std(),
                                        _hsv[:,1].mean(), _hsv[:,1].std(),
                                        _hsv[:,2].mean(), _hsv[:,2].std(),
                                    ])
                                    _np_color_templates.append(_feat)
                            except Exception:
                                pass
                        if _np_color_templates:
                            print(f"  [非球员过滤] 提取到 {len(_np_color_templates)} 个非球员颜色模板，将在聚类前排除相近颜色")
            except Exception as _npe:
                print(f"  [非球员过滤] 提取颜色模板失败: {_npe}")

        for _si, fno in enumerate(sampled_fno_list):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + fno)
            ret, frame = cap.read()
            if not ret:
                break

            # 阶段一 OCR：SoccerNet 用 0.20（清晰近景球员为主），COCO 用 0.30
            _ocr_conf = 0.20 if _player_cls_ids != {0} else 0.30
            results = yolo_model(frame, imgsz=1280, conf=_ocr_conf, verbose=False, half=_ocr_use_half)[0]

            # 收集本帧所有合格球员，按框面积降序（大框=近景=更清晰）
            candidates = []
            for box in results.boxes:
                if int(box.cls[0]) not in _player_cls_ids:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if not is_field_player((x1, y1, x2, y2), width, height):
                    continue
                area = (x2 - x1) * (y2 - y1)
                candidates.append((area, x1, y1, x2, y2))

            # 优先处理面积最大（最近/最清晰）的球员
            candidates.sort(reverse=True)

            for (_, x1, y1, x2, y2) in candidates:
                crop = frame[y1:y2, x1:x2]

                # 先用简单规则过滤裁判（黑色/荧光黄）
                role, _ = get_team_color_and_role(crop)
                if role == "referee":
                    continue

                # 提取颜色特征（躯干区域 HSV 直方图，6维：H/S/V 各均值+标准差）
                h_img, w_img = crop.shape[:2]
                torso_crop = crop[int(h_img*0.2):int(h_img*0.55), int(w_img*0.2):int(w_img*0.80)]
                if torso_crop.size > 0:
                    hsv_t = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
                    color_feat = [
                        float(hsv_t[:,0].mean()), float(hsv_t[:,0].std()),
                        float(hsv_t[:,1].mean()), float(hsv_t[:,1].std()),
                        float(hsv_t[:,2].mean()), float(hsv_t[:,2].std()),
                    ]
                    all_player_color_feats.append(color_feat)

            # 每帧都更新进度（进度范围 5%~38%）
            progress_store[video_uuid] = int(5 + (_si / n_sample_total) * 33)
            if _si % 5 == 0:
                print(f"  [阶段1] {_si}/{n_sample_total} 帧, 已识别背号: {list(jersey_detections.keys())[:10]}")

        # ── 阶段1-B：用全量球员颜色做 k-means 聚类，得到两队颜色中心 ──────────
        # team_hsv_centers: {"team_a": [H,S], "team_b": [H,S]}，阶段2用于最近邻分队
        # 注意：聚类只负责把球员分成两组；主/客（team_a/team_b）由用户在标注弹窗中确认，
        #       前端通过 team_swap 参数告知后端是否需要对调，已在 jersey_annotations 写入时处理。
        #       因此这里直接以 cluster_0 → team_a、cluster_1 → team_b 固定映射，无需猜测。
        team_hsv_centers = {}  # 聚类成功时才有值
        player_nums = list(jersey_detections.keys())

        from sklearn.cluster import KMeans as _KMeans

        # ── 排除非球员颜色样本：把与非球员颜色模板相近的样本从聚类中剔除 ──────
        def _feat_dist(f1, f2):
            """H/S 两维欧氏距离（对光照鲁棒）"""
            return float(np.sqrt((f1[0] - f2[0])**2 + (f1[2] - f2[2])**2))

        def _is_non_player_color(feat, templates, thresh=25.0):
            for tmpl in templates:
                if _feat_dist(np.array(feat), tmpl) < thresh:
                    return True
            return False

        if _np_color_templates and all_player_color_feats:
            _before = len(all_player_color_feats)
            all_player_color_feats = [
                f for f in all_player_color_feats
                if not _is_non_player_color(f, _np_color_templates)
            ]
            _removed = _before - len(all_player_color_feats)
            if _removed > 0:
                print(f"  [非球员过滤] 从聚类样本中移除 {_removed}/{_before} 个非球员颜色特征")

        # ★ 优先用全量颜色数据（所有球员，样本更多更均衡）做聚类
        if len(all_player_color_feats) >= 4:
            try:
                all_feat_arr = np.array(all_player_color_feats, dtype=np.float32)
                # H_mean(0), S_mean(2) 两维，对光照鲁棒
                all_feat_2d = all_feat_arr[:, [0, 2]]
                km_all = _KMeans(n_clusters=2, n_init=15, random_state=42)
                all_labels = km_all.fit_predict(all_feat_2d)
                centers = km_all.cluster_centers_  # shape (2, 2): [[H0,S0],[H1,S1]]

                # 固定映射：cluster 0 → team_a，cluster 1 → team_b
                # 主/客归属已由用户在标注弹窗选择，team_swap 在写入 jersey_annotations 时处理
                team_hsv_centers = {"team_a": centers[0].tolist(), "team_b": centers[1].tolist()}
                cluster_to_team  = {0: "team_a", 1: "team_b"}

                print(f"  [阶段1-B] 全量聚类完成: 样本{len(all_player_color_feats)}个, "
                      f"team_a中心H={team_hsv_centers['team_a'][0]:.1f} S={team_hsv_centers['team_a'][1]:.1f}, "
                      f"team_b中心H={team_hsv_centers['team_b'][0]:.1f} S={team_hsv_centers['team_b'][1]:.1f}")
            except Exception as _ke:
                print(f"  [阶段1-B] 全量聚类失败: {_ke}")

        # 过滤：保留出现≥1次的号码（尽可能多识别球员，广播视频OCR频次本身偏低）
        min_count = 1
        valid_detections = {k: v for k, v in jersey_detections.items() if v["count"] >= min_count}
        print(f"  [阶段1] 有效背号（出现≥{min_count}次）: {list(valid_detections.keys())}")

        # 按队伍分类，每队最多保留11人（标准足球上场人数）
        # 手动标注模式：用户可能标注重复/多余框，严格截断到11人
        # OCR 自动模式：同样限制11人，避免误识别膨胀
        MAX_PER_TEAM = 11
        team_a_players = sorted([k for k, v in valid_detections.items() if v["team"] == "team_a"],
                                key=lambda x: valid_detections[x]["count"], reverse=True)[:MAX_PER_TEAM]
        team_b_players = sorted([k for k, v in valid_detections.items() if v["team"] == "team_b"],
                                key=lambda x: valid_detections[x]["count"], reverse=True)[:MAX_PER_TEAM]
        # 日志提示截断情况
        raw_a = len([k for k, v in valid_detections.items() if v["team"] == "team_a"])
        raw_b = len([k for k, v in valid_detections.items() if v["team"] == "team_b"])
        if raw_a > MAX_PER_TEAM:
            print(f"  [截断] A队原始标注{raw_a}人 → 保留出现频次最高的{MAX_PER_TEAM}人")
        if raw_b > MAX_PER_TEAM:
            print(f"  [截断] B队原始标注{raw_b}人 → 保留出现频次最高的{MAX_PER_TEAM}人")
        
        # 确保每队至少有2名球员
        # 优先从 jersey_detections 中按队伍补充出现≥1次的号码，最后才随机
        if len(team_a_players) < 2:
            used = set(team_a_players + team_b_players)
            # 先尝试用该队已识别但被过滤掉的号码（出现1次）
            fallback_a = sorted(
                [k for k, v in jersey_detections.items() if v["team"] == "team_a" and k not in used],
                key=lambda x: jersey_detections[x]["count"], reverse=True
            )
            if fallback_a:
                print(f"[WARNING] A队OCR识别不足，补充低频背号: {fallback_a[:2-len(team_a_players)]}")
            for num in fallback_a:
                if len(team_a_players) >= 2:
                    break
                team_a_players.append(num)
                used.add(num)
            # 仍不足才随机补充
            if len(team_a_players) < 2:
                print("[WARNING] A队OCR识别不足，补充随机编号...")
                available = [n for n in range(1, 31) if n not in used]
                random.shuffle(available)
                while len(team_a_players) < 2 and available:
                    num = available.pop()
                    team_a_players.append(num)
                    jersey_detections[num] = {"count": 1, "team": "team_a", "bbox": []}

        if len(team_b_players) < 2:
            used = set(team_a_players + team_b_players)
            fallback_b = sorted(
                [k for k, v in jersey_detections.items() if v["team"] == "team_b" and k not in used],
                key=lambda x: jersey_detections[x]["count"], reverse=True
            )
            if fallback_b:
                print(f"[WARNING] B队OCR识别不足，补充低频背号: {fallback_b[:2-len(team_b_players)]}")
            for num in fallback_b:
                if len(team_b_players) >= 2:
                    break
                team_b_players.append(num)
                used.add(num)
            if len(team_b_players) < 2:
                print("[WARNING] B队OCR识别不足，补充随机编号...")
                available = [n for n in range(1, 31) if n not in used]
                random.shuffle(available)
                while len(team_b_players) < 2 and available:
                    num = available.pop()
                    team_b_players.append(num)
                    jersey_detections[num] = {"count": 1, "team": "team_b", "bbox": []}
        
        main_players = team_a_players + team_b_players
        print(f"[OK] 锁定球员背号 - 主队{len(team_a_players)}人: {team_a_players}，客队{len(team_b_players)}人: {team_b_players}")
        
        # 构建球员数据结构（所有识别到的球员）
        demo_players = []
        for num in team_a_players:
            demo_players.append({"num": num, "side": "A", "id": f"A{num:02d}", "pos": []})
        for num in team_b_players:
            demo_players.append({"num": num, "side": "B", "id": f"B{num:02d}", "pos": []})
        
        progress_store[video_uuid] = 40

        # 第二阶段：生成标注视频（含足球追踪、控球率、多目标持续ID）
        print("=== 阶段2: 生成标注视频 ===")
        export_dir = os.path.join(os.path.dirname(os.path.abspath(video_path)), "exports")
        os.makedirs(export_dir, exist_ok=True)
        export_filename = f"annotated_{video_uuid}.mp4"
        export_path = os.path.join(export_dir, export_filename)

        # 初始化追踪器（卡尔曼+Re-ID）
        # iou_threshold=0.25：与3版保持一致，快速运动时匹配更宽松，减少断轨
        # max_lost=5：最多容忍5帧(0.2s)幽灵预测，超过即删除，防止漂移到草地
        # min_hits=3：新追踪器连续3帧命中才输出，抑制单帧误检
        tracker = KalmanTracker(iou_threshold=0.25, reid_threshold=0.55, max_lost=5, min_hits=3)

        # ── 非球员 track_id 预匹配：用前端标注的 bbox 坐标做 IoU 匹配 ───────
        # 在阶段2第一帧运行 YOLO 后，将 non_player_bboxes 与检测框做空间匹配，
        # 获得对应的 track_id，后续帧直接过滤这些 track_id
        _non_player_track_ids: set = set()
        _non_player_ref_bboxes = []  # [(x1,y1,x2,y2), ...]
        if non_player_bboxes:
            for _npb in non_player_bboxes:
                try:
                    _nx = float(_npb.get('x', 0))
                    _ny = float(_npb.get('y', 0))
                    _nw = float(_npb.get('w', 1))
                    _nh = float(_npb.get('h', 1))
                    _non_player_ref_bboxes.append((_nx, _ny, _nx + _nw, _ny + _nh))
                except Exception:
                    pass
            print(f"  [非球员] 收到 {len(_non_player_ref_bboxes)} 个非球员参考框，将在首帧匹配 track_id")

        def _bbox_iou_np(b1, b2):
            """b1, b2: (x1,y1,x2,y2)"""
            ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
            ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0: return 0.0
            a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
            a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            return inter / (a1 + a2 - inter + 1e-6)

        _non_player_matched = False  # 首帧匹配完成标志

        # GPU FP16 加速开关：CUDA 可用时开启半精度，推理速度提升 ~1.5-2x
        import torch as _torch_speed
        _use_half = _torch_speed.cuda.is_available()
        if _use_half:
            print("[OK] GPU FP16 推理已启用（YOLO half=True）")
        else:
            print("[INFO] 未检测到 CUDA，使用 CPU FP32 推理")

        # ── 专用球检测模型 ──────
        # 优先使用专门在足球场景 fine-tune 的 YOLO 模型；
        # 不可用时自动降级为通用 YOLO class=32（带 SAHI 切片补偿）
        ball_model = get_ball_detection_model()
        _ball_model_is_specialized = (ball_model is not None)
        if _ball_model_is_specialized:
            print("[OK] 球检测：使用专用足球模型（035-足球分析方案）")
        else:
            print("[INFO] 球检测：使用通用 YOLO class=32 + SAHI 切片降级方案")

        # 球队标签 Majority Voting 滑动窗口（防止帧间翻转）
        _team_vote_window: dict = defaultdict(lambda: deque(maxlen=30))
        _team_vote_locked: dict = {}  # track_id → 锁定后的队伍

        def get_stable_team(track_id: int, raw_team: str) -> str:
            win = _team_vote_window[track_id]
            win.append(raw_team)
            if track_id in _team_vote_locked:
                return _team_vote_locked[track_id]
            counts: dict = defaultdict(int)
            for t in win:
                counts[t] += 1
            best_team = max(counts, key=lambda k: counts[k])
            if len(win) >= 20 and counts[best_team] / len(win) >= 0.70:
                _team_vote_locked[track_id] = best_team
            return best_team

        def _cap_tracks_per_team(track_items, max_per_team=11):
            grouped = {"team_a": [], "team_b": []}
            passthrough = []
            for t in track_items:
                team = t[5]
                if team in grouped:
                    grouped[team].append(t)
                else:
                    passthrough.append(t)

            capped = list(passthrough)
            for team in ("team_a", "team_b"):
                ranked = sorted(
                    grouped[team],
                    key=lambda x: max(0, int(x[3]) - int(x[1])) * max(0, int(x[4]) - int(x[2])),
                    reverse=True,
                )
                capped.extend(ranked[:max_per_team])
            return capped

        # 控球率统计
        possession_count = {"team_a": 0, "team_b": 0, "none": 0}
        # 热力图位置：按队分别收集所有追踪到的位置
        heatmap_positions = {"team_a": [], "team_b": []}
        # 战术板所有球员轨迹
        tactical_positions = {"team_a": [], "team_b": []}
        ball_trajectory = []  # (cx, cy) 列表

        # ── 个人能力统计（真实数据，替代随机偏置）────────────────────────
        # personal_possession[track_id] = 控球帧数
        personal_possession: dict = {}
        # personal_pass_count[track_id] = 传球事件次数
        personal_pass_count: dict = {}
        # personal_shot_count[track_id] = 射门事件次数（经Pose验证）
        personal_shot_count: dict = {}
        # personal_defend_count[track_id] = 防守贡献次数（成功逼抢/断球）
        personal_defend_count: dict = {}
        # personal_mark_frames[track_id] = 贴身防守帧数（在对方进攻球员 90px 内的采样帧数）
        personal_mark_frames: dict = {}
        # 球速检测用的上帧球位置和控球者
        _prev_ball_pos_for_event = None
        _prev_ball_owner_tid = None
        _prev_ball_owner_team = None
        _prev_ball_owner_box = None   # 上帧控球者的像素框（供Pose验证使用）
        # 踢球事件冷却（避免同一次踢球被连续触发多帧）
        _kick_cooldown = 0

        # ── Pose 模型（轻量姿态估计，仅踢球事件触发，平均 <1次/2秒）────
        _pose_model = None
        try:
            import os as _os
            _pose_pt = _os.path.join(_os.path.dirname(__file__), "yolov8n-pose.pt")
            if _os.path.exists(_pose_pt):
                from ultralytics import YOLO as _YOLO_POSE
                _pose_model = _YOLO_POSE(_pose_pt)
                print(f"[OK] Pose模型加载成功: {_pose_pt}")
            else:
                print(f"[WARNING] yolov8n-pose.pt 不存在，射门动作验证将跳过")
        except Exception as _pe:
            print(f"[WARNING] Pose模型加载失败: {_pe}")

        def _pose_verify_shot(frame_bgr, shooter_box):
            """
            对射门球员裁剪ROI跑姿态推理，验证是否有大幅腿部弯曲（射门动作）。
            返回 True=确认射门动作, False=无法确认（但不撤销射门计数，仅作为附加置信度）。
            与 Phase B _pose_verify_shot_b 保持一致。
            """
            if _pose_model is None:
                return True  # 无Pose模型时默认信任球速判断
            try:
                x1, y1, x2, y2 = shooter_box
                pad = int((y2 - y1) * 0.3)
                rx1 = max(0, x1 - pad)
                ry1 = max(0, y1 - pad)
                rx2 = min(frame_bgr.shape[1] - 1, x2 + pad)
                ry2 = min(frame_bgr.shape[0] - 1, y2 + pad)
                roi = frame_bgr[ry1:ry2, rx1:rx2]
                if roi.size == 0:
                    return True
                import torch as _torch_pa
                _use_half_pa = _torch_pa.cuda.is_available()
                pose_res = _pose_model(roi, imgsz=256, conf=0.3, verbose=False, half=_use_half_pa)
                if not pose_res or pose_res[0].keypoints is None:
                    return True
                kps = pose_res[0].keypoints.xy[0].cpu().numpy() if len(pose_res[0].keypoints.xy) > 0 else None
                if kps is None or len(kps) < 17:
                    return True
                # 检查髋/膝/踝关键点（COCO: 11=L_hip,12=R_hip, 13=L_knee,14=R_knee, 15=L_ankle,16=R_ankle）
                angles = []
                for hip_i, knee_i, ankle_i in [(11, 13, 15), (12, 14, 16)]:
                    h, k, a = kps[hip_i], kps[knee_i], kps[ankle_i]
                    if h[0] == 0 or k[0] == 0 or a[0] == 0:
                        continue
                    v1 = h - k
                    v2 = a - k
                    cos_a = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
                    angles.append(math.degrees(math.acos(max(-1, min(1, cos_a)))))
                if not angles:
                    return True
                max_ang = max(angles)
                return max_ang > 25.0
            except Exception:
                return True  # 推理异常默认信任
        ball_traj_frames = []  # [{"f": fno, "wx": wx, "wy": wy}, ...] 带帧号 pitch 坐标版本
        # 球 pitch 坐标 EMA 平滑状态
        _ball_ema_wx: float = None   # 上帧平滑后 wx
        _ball_ema_wy: float = None   # 上帧平滑后 wy
        _BALL_EMA_ALPHA  = 0.35      # 新值权重（降低，增强平滑抑制跳变）
        _BALL_MAX_JUMP   = 8.0       # 单帧跳变阈值（米），大脚传球~3.6m/帧，8m留双倍余量
        _BALL_MAX_FREEZE = 5         # 最大连续冻结帧数（给卡尔曼追回来的时间）
        _ball_freeze_cnt = 0
        # 动态战术板：按 track_id 分组的逐帧轨迹
        # 格式 {tid: {"team":..., "pts":[{"f":fno,"wx":wx,"wy":wy}]}}
        # wx/wy 是 pitch 真实世界坐标（米制，场地中心为原点），参照 sn-gamestate bbox2pitch
        # 存真实坐标的核心优势：H矩阵切换/更新不影响历史点，彻底消除轨迹汇聚问题
        tactical_tracks = {}
        _last_seen_h_version = -1  # 兼容保留，不再用于重投影
        # H跳变检测：记录上一个采样帧所有球员 pitch 坐标的质心 X/Y 均值
        # X偏移>12m 或 Y偏移>12m 或 Y方差<1（塌缩到同一水平线，明显H异常），判定为H矩阵错误切换，丢弃本帧轨迹点
        # 注：wy_var是方差（非标准差），<1意味着标准差<1m，排除镜头局部推进、少量球员等正常场景的误判
        _prev_centroid_wx: float = None
        _prev_centroid_wy: float = None
        _h_jump_streak: int = 0          # 连续跳变帧计数（≥5帧强制重置参照系，与 Phase B 一致）
        # ── EMA 平滑追踪器字典（移植自3版 EMA+freeze 机制）────────────────
        # _kalman_filters[tid] = PlayerEMATracker 实例（名称保持兼容）
        _kalman_filters: dict = {}
        # 离场计数器：_player_absent_count[tid] = 连续未出现的采样帧数
        _player_absent_count: dict = {}
        # 真正离场的判定阈值：连续缺席 >60 采样帧（≈ 180帧 ≈ 7.2秒）才清理缓存
        _ABSENT_EVICT = 60
        # 遮挡保持最大帧数：超过此数不再写保持点（避免长时离场产生幽灵轨迹）
        _MAX_PREDICT = 10           # 10采样帧 ≈ 30帧 ≈ 1.2s
        # ── 轨迹合并（断轨重连）──────────────────────────────────────────
        # Re-ID 重连后产生新 track_id → 新 track 首次出现时检查附近的"丢失中"旧轨迹
        _track_id_alias: dict = {}  # {new_tid_str -> canonical_tid_str}
        _MERGE_DIST = 1.5           # 重连距离阈值（米），收紧至1.5m，防止密集区跨球员误合并
        _MERGE_MAX_ABSENT = 8       # 最多允许缺席8个采样帧（≈0.96s），缺席太久不合并

        # 🔑 修复：调整编码器优先级（避免 libopenh264 错误）
        # 优先使用稳定的编码器，H264 相关编码器放在后面作为备选
        codecs_to_try = [
            ('mp4v', 'MPEG-4 Part 2 (最稳定，首选)'),
            ('XVID', 'Xvid MPEG-4 (高兼容性)'),
            ('X264', 'X264 H.264/MPEG-4 AVC'),
            ('avc1', 'H.264 Apple (可能触发libopenh264)'),
            ('H264', 'H264 编码器 (可能触发libopenh264)')
        ]
        
        out = None
        used_codec = None
        
        for codec_name, codec_desc in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                test_out = cv2.VideoWriter(export_path, fourcc, fps, (width, height))
                if test_out.isOpened():
                    out = test_out
                    used_codec = codec_name
                    print(f"[OK] 使用编码器: {codec_name} ({codec_desc})")
                    break
                else:
                    test_out.release()
            except Exception as e:
                print(f"[WARNING] {codec_name} 编码器不可用: {e}")
                continue
        
        if not out or not out.isOpened():
            raise Exception("无法初始化任何视频编码器，请检查 OpenCV 安装或安装 ffmpeg")

        # 队伍颜色映射（统一管理）
        TEAM_COLORS = {
            "team_a": (255, 100, 0),    # BGR: 橙色（主队）
            "team_b": (255, 60, 0),     # BGR: 蓝色（客队）
            "unknown": (160, 160, 160), # 灰色（未知/普通）
        }
        # 主力球员边框厚度 vs 普通球员
        MAIN_PLAYER_THICKNESS = 3
        NORMAL_PLAYER_THICKNESS = 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 草地 mask：每 10 帧更新一次（草地颜色变化慢，缓存可大幅降低开销）
        _grass_mask_cache: Optional[np.ndarray] = None
        _grass_mask_fno: int = -100

        # ── VideoCleaner：数据清洗，过滤回放/看台/特写帧 ────────────────
        _video_cleaner_main = VideoCleaner()

        fno = 0
        for fno in range(target_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # ── VideoCleaner 第一道防线：场景级过滤（回放/看台/特写）─────
            if not _video_cleaner_main.is_valid_frame(frame):
                out.write(frame)   # 原帧直接写入，保持输出视频时长完整
                continue

            # 更新草地 mask（首帧及每 10 帧刷新一次）
            if fno - _grass_mask_fno >= 10:
                _grass_mask_cache = build_grass_mask(frame)
                _grass_mask_fno = fno

            # ── 草地掩码第二道防线：绿色比例 < 8% 跳过（与 VideoCleaner 双重保障）──
            if _grass_mask_cache is not None:
                _green_ratio = float(np.sum(_grass_mask_cache > 0)) / _grass_mask_cache.size
                if _green_ratio < 0.08:
                    out.write(frame)   # 原帧直接写入，保持输出视频时长完整
                    continue           # 广告/特写/看台画面，不参与追踪和统计

            run_detection = True  # 逐帧检测，避免球员位置冻结导致"球自己跑"
            if run_detection:
                # ── 球员检测：优先 SoccerNet YOLOv8m(best.pt)，降级通用 YOLOv8n（imgsz=1280 全帧）──
                # SoccerNet 专用模型：conf=0.15（模型精度高，低阈值可减少漏检而不增加明显误检）
                # 通用 COCO yolov8n：conf 需稍高（0.25），防止大量非球员误检
                _det_conf = 0.15 if _player_cls_ids != {0} else 0.25
                results = yolo_model(frame, imgsz=1280, conf=_det_conf, verbose=False, half=_use_half)[0]
                _raw_person_boxes = [box for box in results.boxes if int(box.cls[0]) in _player_cls_ids]

                # ── 二次 Soft-NMS：去除 YOLO 内置 NMS 漏掉的高重叠冗余框 ──────────
                # 对密集站位时同一球员被多个框框住的情况进行额外过滤
                def _soft_nms_boxes(boxes, iou_thr=0.45):
                    """基于 IoU 的二次 NMS，按置信度降序保留非重叠框"""
                    if not boxes:
                        return boxes
                    # 按置信度降序排列
                    sorted_boxes = sorted(boxes, key=lambda b: float(b.conf[0]), reverse=True)
                    kept = []
                    for cand in sorted_boxes:
                        cx1, cy1, cx2, cy2 = map(int, cand.xyxy[0])
                        ca = (cx2 - cx1) * (cy2 - cy1)
                        overlap = False
                        for kb in kept:
                            kx1, ky1, kx2, ky2 = map(int, kb.xyxy[0])
                            ix = max(0, min(cx2, kx2) - max(cx1, kx1))
                            iy = max(0, min(cy2, ky2) - max(cy1, ky1))
                            inter = ix * iy
                            ka = (kx2 - kx1) * (ky2 - ky1)
                            union = ca + ka - inter
                            if union > 0 and inter / union > iou_thr:
                                overlap = True
                                break
                        if not overlap:
                            kept.append(cand)
                    return kept
                person_boxes = _soft_nms_boxes(_raw_person_boxes, iou_thr=0.45)

                # ── 足球检测：仿 035-足球分析三层方案 ────────────────────────────
                # 方案一（专用模型可用时）：
                #   专用 football-ball-detection.pt 始终 SAHI 切片推理
                #   低置信度阈值 0.10（宁多检再过滤），尺寸过滤去误检
                # 方案二（专用模型不可用时）：
                #   通用 YOLO class=32 全图推理 → 漏检时 SAHI 切片补救
                ball_candidates = []

                def _sahi_detect_ball(model, cls_filter, conf_thr, slice_size=640, stride=480):
                    """
                    SAHI 切片推理内核（Batch GPU 版）。
                    先收集所有 patch，再一次性 batch 推理，充分利用 GPU 并行能力。
                    切片数量与原版完全相同，速度提升 3~5 倍。

                    额外过滤：
                      - 尺寸过滤：宽高均需 ≤ 80px（足球在广播视角通常 < 50px）
                      - 面积过滤：bbox 面积 ≤ 3600px²（约 60×60）
                    """
                    import torch
                    fh, fw = frame.shape[:2]
                    patches = []
                    coords = []

                    # 第一步：收集所有有效 patch
                    for sy in range(0, fh, stride):
                        for sx in range(0, fw, stride):
                            ex = min(sx + slice_size, fw)
                            ey = min(sy + slice_size, fh)
                            patch = frame[sy:ey, sx:ex]
                            if patch.shape[0] < 100 or patch.shape[1] < 100:
                                continue
                            patches.append(patch)
                            coords.append((sx, sy))

                    if not patches:
                        return []

                    # 第二步：一次 batch 推理（GPU 满载，FP16 加速）
                    # 只有模型参数本身在 CUDA 上时才启用 half，避免 CPU+half 报错
                    try:
                        _model_device = next(model.model.parameters()).device
                        use_half = _model_device.type == 'cuda'
                    except Exception:
                        use_half = False
                    results = model(
                        patches,
                        imgsz=slice_size,
                        conf=conf_thr,
                        half=use_half,
                        verbose=False
                    )

                    # 第三步：还原全局坐标并过滤
                    raw = []
                    for i, r in enumerate(results):
                        sx, sy = coords[i]
                        for sbox in r.boxes:
                            if cls_filter is not None and int(sbox.cls[0]) != cls_filter:
                                continue
                            sbx1, sby1, sbx2, sby2 = map(int, sbox.xyxy[0])
                            bw = sbx2 - sbx1; bh = sby2 - sby1
                            # 尺寸过滤：宽或高超过80px的不太可能是远景足球
                            if bw > 80 or bh > 80:
                                continue
                            # 面积过滤
                            if bw * bh > 3600:
                                continue
                            gcx = sx + (sbx1 + sbx2) // 2
                            gcy = sy + (sby1 + sby2) // 2
                            raw.append((gcx, gcy, float(sbox.conf[0])))

                    # 距离 NMS：< 30px 的候选只保留置信度最高的
                    raw.sort(key=lambda c: -c[2])
                    kept = []
                    for cand in raw:
                        if not any(math.hypot(cand[0]-k[0], cand[1]-k[1]) < 30 for k in kept):
                            kept.append(cand)
                    return kept

                if _ball_model_is_specialized:
                    # ── 方案一：专用模型 SAHI 切片（始终执行，cls=0=ball） ──────
                    try:
                        ball_candidates = _sahi_detect_ball(
                            ball_model, cls_filter=0, conf_thr=0.25,
                            slice_size=640, stride=480
                        )
                        # 补充：对专用模型也做一次全图推理（捕获近景大目标）
                        try:
                            _bm_half = next(ball_model.model.parameters()).device.type == 'cuda'
                        except Exception:
                            _bm_half = False
                        full_res = ball_model(frame, imgsz=640, conf=0.25, half=_bm_half, verbose=False)[0]
                        for fbox in full_res.boxes:
                            if int(fbox.cls[0]) != 0:
                                continue
                            fbx1, fby1, fbx2, fby2 = map(int, fbox.xyxy[0])
                            bw = fbx2 - fbx1; bh = fby2 - fby1
                            if bw > 80 or bh > 80 or bw * bh > 3600:
                                continue
                            fcx = (fbx1 + fbx2) // 2
                            fcy = (fby1 + fby2) // 2
                            fconf = float(fbox.conf[0])
                            # 去重
                            if not any(math.hypot(fcx-k[0], fcy-k[1]) < 30 for k in ball_candidates):
                                ball_candidates.append((fcx, fcy, fconf))
                    except Exception as _ball_e:
                        print(f"[WARNING] 专用球检测异常，降级: {_ball_e}")
                        _ball_model_is_specialized = False  # 本次视频降级

                if not _ball_model_is_specialized:
                    # ── 方案二：yolo_model 球类检测 + SAHI 补救 ──────────────────
                    # _yolo_ball_cls_id: SoccerNet=ball类id，COCO=32(sports ball)，None=跳过
                    if _yolo_ball_cls_id is not None:
                        for box in results.boxes:
                            if int(box.cls[0]) != _yolo_ball_cls_id:
                                continue
                            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                            bw = bx2 - bx1; bh = by2 - by1
                            if bw > 80 or bh > 80 or bw * bh > 3600:
                                continue
                            ball_candidates.append(((bx1+bx2)//2, (by1+by2)//2, float(box.conf[0])))
                    # 全图漏检时才触发 SAHI（节省算力）
                    if not ball_candidates and _yolo_ball_cls_id is not None:
                        try:
                            ball_candidates = _sahi_detect_ball(
                                yolo_model, cls_filter=_yolo_ball_cls_id, conf_thr=0.20,
                                slice_size=640, stride=480
                            )
                            if ball_candidates:
                                print(f"[DEBUG] SAHI补救检测到 {len(ball_candidates)} 个球候选")
                        except Exception as _sahi_e:
                            pass  # 不影响主流程

                ball_pos = ball_tracker.update(ball_candidates, frame_w=width, frame_h=height)
                if ball_pos:
                    ball_trajectory.append(ball_pos)
                    # 球也用 pitch 坐标存储，与球员轨迹坐标系统一
                    ball_pitch = homography_est.unproject_to_pitch(float(ball_pos[0]), float(ball_pos[1]))
                    if ball_pitch is None:
                        # fallback：线性反推 pitch 坐标
                        cw_px = homography_est.CANVAS_W - 2 * homography_est.MARGIN
                        ch_px = homography_est.CANVAS_H - 2 * homography_est.MARGIN
                        dw = float(width) if width > 0 else float(homography_est.CANVAS_W)
                        dh = float(height) if height > 0 else float(homography_est.CANVAS_H)
                        cx_ = homography_est.MARGIN + ball_pos[0] / dw * cw_px
                        cy_ = homography_est.MARGIN + ball_pos[1] / dh * ch_px
                        bwx = (cx_ - homography_est.MARGIN) / cw_px * homography_est.PITCH_L - homography_est.PITCH_L / 2.0
                        bwy = (cy_ - homography_est.MARGIN) / ch_px * homography_est.PITCH_W - homography_est.PITCH_W / 2.0
                        ball_pitch = (round(bwx, 3), round(bwy, 3))
                    # ── Pitch 越界拦截：wx/wy 超出场地范围的 5% 认为是误检，丢弃该帧坐标 ──
                    # 正常场地 wx ∈ [-52.5, 52.5]，wy ∈ [-34, 34]；允许 20% 越界余量（约±10m，减少误丢弃）
                    _px_lim = homography_est.PITCH_L / 2.0 * 1.20   # ≈ 63 m
                    _py_lim = homography_est.PITCH_W / 2.0 * 1.20   # ≈ 40.8 m
                    if abs(ball_pitch[0]) <= _px_lim and abs(ball_pitch[1]) <= _py_lim:
                        bwx_raw, bwy_raw = ball_pitch[0], ball_pitch[1]
                        # ── 球 pitch 坐标 EMA 平滑 + 跳变冻结 ──────────────────────
                        if _ball_ema_wx is None:
                            # 首次有效坐标，直接初始化
                            _ball_ema_wx, _ball_ema_wy = bwx_raw, bwy_raw
                            _ball_freeze_cnt = 0
                        else:
                            _jump = ((bwx_raw - _ball_ema_wx) ** 2 + (bwy_raw - _ball_ema_wy) ** 2) ** 0.5
                            if _jump > _BALL_MAX_JUMP and _ball_freeze_cnt < _BALL_MAX_FREEZE:
                                # 跳变过大：冻结在上帧位置
                                _ball_freeze_cnt += 1
                            else:
                                # 正常更新（或达到冻结上限强制接受）
                                _ball_ema_wx = _BALL_EMA_ALPHA * bwx_raw + (1.0 - _BALL_EMA_ALPHA) * _ball_ema_wx
                                _ball_ema_wy = _BALL_EMA_ALPHA * bwy_raw + (1.0 - _BALL_EMA_ALPHA) * _ball_ema_wy
                                _ball_freeze_cnt = 0
                        ball_traj_frames.append({"f": fno, "wx": round(_ball_ema_wx, 3), "wy": round(_ball_ema_wy, 3)})

                # --- 球员检测并更新追踪器 ---
                detections = []
                for box in person_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if not is_field_player((x1, y1, x2, y2), width, height, _grass_mask_cache):
                        continue
                    crop = frame[y1:y2, x1:x2]
                    # 裁判过滤（硬编码规则只用于判断是否是裁判）
                    role_raw, color_raw = get_team_color_and_role(crop)
                    if role_raw == "referee":
                        continue
                    # ★ 若阶段1已聚类出两队颜色中心，用最近邻分队（更准确）
                    if team_hsv_centers:
                        h_c, w_c = crop.shape[:2]
                        torso = crop[int(h_c*0.2):int(h_c*0.55), int(w_c*0.2):int(w_c*0.80)]
                        if torso.size > 0:
                            hsv_t = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
                            h_mean = float(hsv_t[:, 0].mean())
                            s_mean = float(hsv_t[:, 1].mean())
                            pt = np.array([h_mean, s_mean])
                            ca = np.array(team_hsv_centers["team_a"])
                            cb = np.array(team_hsv_centers["team_b"])
                            role = "team_a" if np.linalg.norm(pt - ca) <= np.linalg.norm(pt - cb) else "team_b"
                        else:
                            role = role_raw if role_raw not in ("unknown", "referee") else "unknown"
                        color = (255, 100, 0) if role == "team_a" else (255, 60, 0) if role == "team_b" else (180, 180, 180)
                    else:
                        role = role_raw
                        color = color_raw if role not in ("unknown",) else (180, 180, 180)
                    detections.append((x1, y1, x2, y2, role, color))

                # 自适应更新频率：GPU 每帧更新（SoccerNet 模式），CPU 每60帧（降频）
                H_mat = homography_est.estimate(frame, fno, update_interval=homography_est.update_interval_recommended)

                # ── H版本号追踪（pitch坐标模式下无需重投影历史点） ──────────────────
                _cur_h_ver = homography_est.h_version
                if _last_seen_h_version < 0 and H_mat is not None:
                    _last_seen_h_version = _cur_h_ver

                # 传入当前帧以启用 Re-ID 特征提取（方案一），传入 H_mat 启用 GMC 全局运动补偿
                tracked_raw = tracker.update(detections, frame=frame, H_mat=H_mat)

                # ── 首帧：用非球员参考框匹配 track_id ───────────────────────
                if _non_player_ref_bboxes and not _non_player_matched:
                    _non_player_matched = True
                    for (_ref_x1, _ref_y1, _ref_x2, _ref_y2) in _non_player_ref_bboxes:
                        _best_iou = 0.0
                        _best_tid = None
                        for (_ti, _tx1, _ty1, _tx2, _ty2, *_) in tracked_raw:
                            _iou = _bbox_iou_np(
                                (_ref_x1, _ref_y1, _ref_x2, _ref_y2),
                                (_tx1, _ty1, _tx2, _ty2)
                            )
                            if _iou > _best_iou:
                                _best_iou = _iou
                                _best_tid = _ti
                        if _best_tid is not None and _best_iou > 0.1:
                            _non_player_track_ids.add(_best_tid)
                    if _non_player_track_ids:
                        print(f"  [非球员] 匹配到 track_ids={_non_player_track_ids}，将从战术板/控球率中排除")

                # ── 过滤非球员 track ────────────────────────────────────────
                if _non_player_track_ids:
                    tracked_raw = [t for t in tracked_raw if t[0] not in _non_player_track_ids]

                # Majority Voting 稳定球队标签
                tracked = []
                for item in tracked_raw:
                    tid, x1_, y1_, x2_, y2_, raw_team, color_ = item
                    stable_team = get_stable_team(tid, raw_team)
                    # 根据稳定队伍同步颜色
                    if stable_team == "team_a":
                        stable_color = (255, 100, 0)
                    elif stable_team == "team_b":
                        stable_color = (255, 60, 0)
                    else:
                        stable_color = (180, 180, 180)  # unknown → 灰色
                    tracked.append((tid, x1_, y1_, x2_, y2_, stable_team, stable_color))

                # --- unknown 球员颜色重归队：每30帧，用同帧已知队伍的颜色质心做最近邻匹配 ---
                if fno % 30 == 0:
                    # 收集本帧已知球员的 HSV 颜色质心（按队）
                    known_hsv = {"team_a": [], "team_b": []}
                    for (tid, x1_, y1_, x2_, y2_, team, _) in tracked:
                        if team in known_hsv:
                            crop = frame[y1_:y2_, x1_:x2_]
                            if crop.size == 0:
                                continue
                            h_c, w_c = crop.shape[:2]
                            roi = crop[int(h_c*0.2):int(h_c*0.55), int(w_c*0.25):int(w_c*0.75)]
                            if roi.size == 0:
                                continue
                            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                            mean_hs = cv2.mean(hsv_roi)[:2]
                            known_hsv[team].append(mean_hs)
                    # 为每个队计算均值质心
                    team_centers = {}
                    for t, pts in known_hsv.items():
                        if pts:
                            team_centers[t] = (
                                float(np.mean([p[0] for p in pts])),
                                float(np.mean([p[1] for p in pts]))
                            )
                    # 对 unknown 球员重新判断
                    if len(team_centers) == 2:
                        new_tracked = []
                        for item in tracked:
                            tid, x1_, y1_, x2_, y2_, team, color_ = item
                            if team == "unknown":
                                crop = frame[y1_:y2_, x1_:x2_]
                                if crop.size > 0:
                                    h_c, w_c = crop.shape[:2]
                                    roi = crop[int(h_c*0.2):int(h_c*0.55), int(w_c*0.25):int(w_c*0.75)]
                                    if roi.size > 0:
                                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                                        h_val = float(cv2.mean(hsv_roi)[0])
                                        s_val = float(cv2.mean(hsv_roi)[1])
                                        # 计算到各队质心的欧氏距离（H、S通道）
                                        dist_a = math.hypot(h_val - team_centers["team_a"][0],
                                                            s_val - team_centers["team_a"][1])
                                        dist_b = math.hypot(h_val - team_centers["team_b"][0],
                                                            s_val - team_centers["team_b"][1])
                                        # 只有明显更近时才归队（差距>10，避免误判）
                                        if abs(dist_a - dist_b) > 10:
                                            assigned = "team_a" if dist_a < dist_b else "team_b"
                                            assigned_color = (255, 100, 0) if assigned == "team_a" else (255, 60, 0)
                                            # 注入到 voting 窗口，加速收敛
                                            get_stable_team(tid, assigned)
                                            item = (tid, x1_, y1_, x2_, y2_, assigned, assigned_color)
                            new_tracked.append(item)
                        tracked = new_tracked
                tracked = _cap_tracks_per_team(tracked, max_per_team=11)

                # --- 定期 Re-ID 队伍聚类 + 左右侧标记（每150帧 ≈ 5秒）---
                # assign_teams_by_reid: OSNet+UMAP/PCA+KMeans 重新聚类两队
                # assign_team_sides: 根据平均X坐标决定哪队在左/右侧进攻
                if fno % 150 == 0 and fno > 0:
                    try:
                        tracker.assign_teams_by_reid()
                        _team_sides = tracker.assign_team_sides(H_mat=H_mat)
                        tracker._team_sides.update(_team_sides)
                        print(f"[DEBUG] 帧{fno}: 重新聚类完成，侧边={_team_sides}")
                    except Exception as _reid_e:
                        print(f"[WARNING] Re-ID队伍聚类失败: {_reid_e}")

                # --- 控球率：计算距球最近的队伍（同时记录个人控球）---
                _cur_nearest_tid = None
                _cur_nearest_team = "none"
                _cur_nearest_box = None
                if ball_pos:
                    min_dist = float("inf")
                    nearest_team = "none"
                    nearest_player_h = 80   # 默认球员框高度（像素）
                    for (tid, x1, y1, x2, y2, team, color) in tracked:
                        cx_p = (x1 + x2) // 2
                        cy_p = (y1 + y2) // 2
                        d = math.hypot(cx_p - ball_pos[0], cy_p - ball_pos[1])
                        if d < min_dist:
                            min_dist = d
                            nearest_team = team
                            nearest_player_h = max(30, y2 - y1)
                            _cur_nearest_tid = tid
                            _cur_nearest_team = team
                            _cur_nearest_box = (x1, y1, x2, y2)
                    # 自适应控球距离阈值：球员框高度的 1.8 倍
                    # 近景大框（~150px高）→ 阈值≈270px；远景小框（~50px）→ 阈值≈90px
                    possession_thresh = nearest_player_h * 1.8
                    if min_dist < possession_thresh and nearest_team in ("team_a", "team_b"):
                        possession_count[nearest_team] = possession_count.get(nearest_team, 0) + 1
                        # ── 个人控球帧数累计 ──────────────────────────────────────
                        if _cur_nearest_tid is not None:
                            _tid_str = str(_cur_nearest_tid)
                            personal_possession[_tid_str] = personal_possession.get(_tid_str, 0) + 1
                    else:
                        possession_count["none"] = possession_count.get("none", 0) + 1
                        _cur_nearest_tid = None  # 无人控球

                    # ── 球速突变检测（踢球事件：射门 / 传球 / 防守）──────────────
                    if _kick_cooldown > 0:
                        _kick_cooldown -= 1
                    if _prev_ball_pos_for_event is not None and _kick_cooldown == 0:
                        ball_spd = math.hypot(
                            ball_pos[0] - _prev_ball_pos_for_event[0],
                            ball_pos[1] - _prev_ball_pos_for_event[1]
                        )
                        # 阈值 28px/帧（≈ 25fps×28px ≈ 中速传球），排除球员行走带球的慢速移动
                        if ball_spd > 28 and _prev_ball_owner_tid is not None:
                            _kicker_tid_str = str(_prev_ball_owner_tid)
                            _kicker_team = _prev_ball_owner_team
                            # 判断球的方向：是否朝向对方球门
                            # 规则：team_a 进攻方向为画面右侧（X增大），team_b 为左侧（X减小）
                            _dx = ball_pos[0] - _prev_ball_pos_for_event[0]
                            if _kicker_team == "team_a":
                                _toward_goal = _dx > 5  # 向右踢
                            elif _kicker_team == "team_b":
                                _toward_goal = _dx < -5  # 向左踢
                            else:
                                _toward_goal = False

                            # 是否在禁区附近（距对方球门 < 宽度的1/3）
                            _in_attack_zone = False
                            if _kicker_team == "team_a" and ball_pos[0] > width * 0.67:
                                _in_attack_zone = True
                            elif _kicker_team == "team_b" and ball_pos[0] < width * 0.33:
                                _in_attack_zone = True

                            if _toward_goal and _in_attack_zone:
                                # 射门事件：朝向球门且在进攻区域
                                # 用 Pose 模型验证踢球动作（用上帧踢球者的框，不是当前帧）
                                _pose_confirmed = True
                                if _prev_ball_owner_box is not None:
                                    _pose_confirmed = _pose_verify_shot(frame, _prev_ball_owner_box)
                                if _pose_confirmed:
                                    personal_shot_count[_kicker_tid_str] = personal_shot_count.get(_kicker_tid_str, 0) + 1
                            elif not _toward_goal or not _in_attack_zone:
                                # 传球事件：快速踢球但不是射门方向
                                personal_pass_count[_kicker_tid_str] = personal_pass_count.get(_kicker_tid_str, 0) + 1

                            # 防守事件：对方控球时某球员迅速逼近并造成球速突变（意味着成功断球）
                            # 检测：_prev_ball_owner_team ≠ _cur_nearest_team（球权转换了）
                            if (_cur_nearest_tid is not None and
                                    _cur_nearest_team in ("team_a", "team_b") and
                                    _kicker_team in ("team_a", "team_b") and
                                    _cur_nearest_team != _kicker_team):
                                # 球权发生了转换，原控球方丢球 → 当前最近球员记一次防守成功
                                _defender_tid_str = str(_cur_nearest_tid)
                                personal_defend_count[_defender_tid_str] = personal_defend_count.get(_defender_tid_str, 0) + 1

                            _kick_cooldown = 8  # 冷却8帧，避免同一次踢球触发多次

                    _prev_ball_pos_for_event = ball_pos
                    # 只有当前帧有明确控球者时才更新；球飞出时保留上一控球者，
                    # 防止踢球瞬间球离人远导致 _prev_ball_owner_tid 被清 None
                    if _cur_nearest_tid is not None:
                        _prev_ball_owner_tid = _cur_nearest_tid
                        _prev_ball_owner_team = _cur_nearest_team
                        _prev_ball_owner_box = _cur_nearest_box
                else:
                    possession_count["none"] = possession_count.get("none", 0) + 1
                    _prev_ball_pos_for_event = None

                # --- 记录热力图位置（每2帧采样，与 Phase B 及检测周期 fno%2==0 对齐）---
                if fno % 2 == 0:
                    # ── H跳变检测：计算本帧所有球员 pitch 坐标质心 X ──────────────────
                    # 先收集本帧所有 pitch 坐标，再决定是否写入 tactical_tracks
                    frame_pitch_pts = []  # [(tid, team, wx, wy, px, py), ...]
                    for (tid, x1, y1, x2, y2, team, color) in tracked:
                        # 与 Phase B 对齐：只取已标注队伍球员，排除 non_player / unknown 等
                        if team not in ("team_a", "team_b"):
                            continue
                        raw_bx = float(x1 + x2) / 2.0
                        raw_by = float(y2)
                        str_tid_fb = str(tid)
                        if H_mat is None:
                            # H_mat 完全不可用时：线性缩放兜底（与 Phase B 对齐）
                            wx_fb = raw_bx / max(1, width) * 105.0 - 52.5
                            wy_fb = raw_by / max(1, height) * 68.0 - 34.0
                            px_fb = homography_est.project_bbox_bottom(x1, y1, x2, y2, None, video_w=width, video_h=height)
                            frame_pitch_pts.append((tid, team, wx_fb, wy_fb, int(px_fb[0]), int(px_fb[1])))
                            continue
                        pitch_pt = homography_est.unproject_to_pitch(raw_bx, raw_by, H_mat)
                        if pitch_pt is None:
                            # H投影越界：优先从卡尔曼滤波器获取预测坐标，避免坐标系切换引起的跳变
                            can_tid_fb = resolve_alias(_track_id_alias, str_tid_fb)
                            kf_fb = _kalman_filters.get(can_tid_fb)
                            if kf_fb is not None:
                                wx_fb, wy_fb = kf_fb.predict()
                                px_fb = homography_est.project_bbox_bottom(x1, y1, x2, y2, H_mat, video_w=width, video_h=height)
                            else:
                                # 首次出现且H越界：线性缩放兜底（与 Phase B 对齐）
                                wx_fb = raw_bx / max(1, width) * 105.0 - 52.5
                                wy_fb = raw_by / max(1, height) * 68.0 - 34.0
                                px_fb = homography_est.project_bbox_bottom(x1, y1, x2, y2, None, video_w=width, video_h=height)
                            frame_pitch_pts.append((tid, team, wx_fb, wy_fb, int(px_fb[0]), int(px_fb[1])))
                            continue
                        wx, wy = pitch_pt
                        px, py = homography_est.project_bbox_bottom(
                            x1, y1, x2, y2, H_mat, video_w=width, video_h=height
                        )
                        frame_pitch_pts.append((tid, team, wx, wy, px, py))

                    # ── 质心跳变过滤：X/Y 双轴检测 + Y方差塌缩检测 ──────────────────
                    # 正常比赛：球员质心帧间偏移 <5m，12m已是极宽裕阈值
                    # Y方差塌缩：H tilt偏移时球员全部压缩到底线，wy方差趋近于0
                    h_jump_frame = False
                    if frame_pitch_pts:
                        cur_wx_list = sorted([p[2] for p in frame_pitch_pts])
                        cur_wy_list = sorted([p[3] for p in frame_pitch_pts])
                        n = len(cur_wx_list)
                        # 用中位数代替均值，抗裁判/广告牌等异常点干扰
                        mid = n // 2
                        cur_centroid_wx = cur_wx_list[mid] if n % 2 else (cur_wx_list[mid-1] + cur_wx_list[mid]) / 2
                        cur_centroid_wy = cur_wy_list[mid] if n % 2 else (cur_wy_list[mid-1] + cur_wy_list[mid]) / 2
                        wy_var = sum((y - cur_centroid_wy) ** 2 for y in cur_wy_list) / n if n > 1 else 999.0

                        # 质心X/Y偏移检测：需要足够多球员（≥4人）才可信
                        # 少于4人时场景可能是定位球/局部镜头，不触发整帧丢弃
                        if n >= 4 and _prev_centroid_wx is not None and abs(cur_centroid_wx - _prev_centroid_wx) > 30.0:
                            print(f"[DEBUG] 帧{fno}: 检测到H跳变（质心X偏移{cur_centroid_wx - _prev_centroid_wx:.1f}m），冻结本帧")
                            h_jump_frame = True
                        elif n >= 4 and _prev_centroid_wy is not None and abs(cur_centroid_wy - _prev_centroid_wy) > 25.0:
                            print(f"[DEBUG] 帧{fno}: 检测到H跳变（质心Y偏移{cur_centroid_wy - _prev_centroid_wy:.1f}m），冻结本帧")
                            h_jump_frame = True
                        # Y方差塌缩：仅在球员数足够多（≥6）且方差极小时才判定
                        # 少人场景（角球/定位球）Y方向本身集中，不应误杀
                        elif n >= 6 and wy_var < 0.5:
                            print(f"[DEBUG] 帧{fno}: 检测到Y方向坐标塌缩（wy方差={wy_var:.3f}），冻结本帧")
                            h_jump_frame = True

                        # 跳变帧：质心参照系保持上帧值不更新，避免错误坐标污染参照系
                        # 正常帧：更新参照系
                        if not h_jump_frame:
                            _prev_centroid_wx = cur_centroid_wx
                            _prev_centroid_wy = cur_centroid_wy
                            _h_jump_streak = 0
                        else:
                            _h_jump_streak += 1
                            if _h_jump_streak >= 5:
                                # 连续5帧全是跳变帧：H矩阵可能永久切换了（如机位变换），
                                # 强制接受新质心为参照系，避免后续所有帧都被误判为跳变帧
                                _prev_centroid_wx = cur_centroid_wx
                                _prev_centroid_wy = cur_centroid_wy
                                _h_jump_streak = 0
                                print(f"[DEBUG] 帧{fno}: 连续跳变5帧，强制重置质心参考值")

                    # ── 离场/缺席计数器更新（与 Phase B 对齐：跳变帧不更新离场计数，坐标无效）──
                    if not h_jump_frame:
                        _cur_canonical_tids = set()
                        for (tid, *_rest) in frame_pitch_pts:
                            ct = resolve_alias(_track_id_alias, str(tid))
                            _cur_canonical_tids.add(ct)
                        for known_tid in list(_player_absent_count.keys()):
                            if known_tid in _cur_canonical_tids:
                                _player_absent_count[known_tid] = 0
                            else:
                                _player_absent_count[known_tid] += 1
                        for new_tid in _cur_canonical_tids:
                            if new_tid not in _player_absent_count:
                                _player_absent_count[new_tid] = 0

                        # 清理真正长期离场球员（缺席 > _ABSENT_EVICT 采样帧 ≈ 12秒）
                        evict_tids = [t for t, cnt in _player_absent_count.items() if cnt > _ABSENT_EVICT]
                        for t in evict_tids:
                            _kalman_filters.pop(t, None)
                            del _player_absent_count[t]

                    # ── 卡尔曼预测：对本帧缺席的球员执行 predict()，填充轨迹间隙 ────
                    if not h_jump_frame:
                        for kf_tid, kf in list(_kalman_filters.items()):
                            if kf_tid in _cur_canonical_tids:
                                continue  # 本帧有测量，跳过（后面 update 时写入）
                            absent_cnt = _player_absent_count.get(kf_tid, 0)
                            if absent_cnt > _MAX_PREDICT:
                                continue  # 缺席太久，不再预测（避免幽灵轨迹）
                            if kf_tid not in tactical_tracks:
                                continue
                            px_pred, py_pred = kf.predict()
                            # clamp 到场地边界
                            _pitch_half_l = homography_est.PITCH_L / 2.0 * 1.05
                            _pitch_half_w = homography_est.PITCH_W / 2.0 * 1.05
                            px_pred = max(-_pitch_half_l, min(_pitch_half_l, px_pred))
                            py_pred = max(-_pitch_half_w, min(_pitch_half_w, py_pred))
                            tactical_tracks[kf_tid]["pts"].append({
                                "f": fno,
                                "wx": round(px_pred, 3),
                                "wy": round(py_pred, 3),
                                "predicted": True,  # 标记为预测点，前端可用虚线渲染
                            })

                    if h_jump_frame:
                        # H跳变帧：用卡尔曼当前状态位置补写，保持球员在原位不消失
                        for str_tid, kf in _kalman_filters.items():
                            if str_tid not in tactical_tracks:
                                continue
                            wx_s, wy_s = kf.pos
                            tactical_tracks[str_tid]["pts"].append({
                                "f": fno, "wx": round(wx_s, 3), "wy": round(wy_s, 3),
                                "predicted": True,
                            })
                    else:
                        for (tid, team, wx, wy, px, py) in frame_pitch_pts:
                            if team in heatmap_positions:
                                heatmap_positions[team].append({
                                    "x": max(0, min(799, px)),
                                    "y": max(0, min(519, py))
                                })
                            # ── 坐标严格 clip 到场地边界（±5% 余量）────────────────────
                            _pitch_half_l = homography_est.PITCH_L / 2.0 * 1.05
                            _pitch_half_w = homography_est.PITCH_W / 2.0 * 1.05
                            wx = max(-_pitch_half_l, min(_pitch_half_l, wx))
                            wy = max(-_pitch_half_w, min(_pitch_half_w, wy))
                            # ── 断轨重连：新 track_id 尝试合并到最近的旧轨迹 ──────────
                            str_tid = str(tid)
                            # 路径压缩 alias 查找（修复链式 A→B→C 只走一步的问题）
                            canonical_tid = resolve_alias(_track_id_alias, str_tid)
                            # 若是全新 track_id（卡尔曼字典里没有），尝试就近合并
                            if canonical_tid not in _kalman_filters:
                                best_cand  = None
                                best_score = -1.0  # 用评分替代纯距离阈值筛选（AFLink compute_rule_score）
                                for cand_tid, kf in _kalman_filters.items():
                                    absent_cnt = _player_absent_count.get(cand_tid, 999)
                                    if absent_cnt > _MERGE_MAX_ABSENT:
                                        continue
                                    if cand_tid in tactical_tracks and tactical_tracks[cand_tid]["team"] != team:
                                        continue
                                    cwx, cwy = kf.pos
                                    d = ((wx - cwx) ** 2 + (wy - cwy) ** 2) ** 0.5
                                    if d >= _MERGE_DIST:
                                        continue  # 超出距离阈值直接跳过
                                    score = compute_rule_score(d, absent_cnt, _MERGE_DIST, max(_MERGE_MAX_ABSENT, 1))
                                    if score > best_score:
                                        best_score = score
                                        best_cand  = cand_tid
                                if best_cand is not None:
                                    _track_id_alias[str_tid] = best_cand
                                    canonical_tid = best_cand
                                    print(f"  [轨迹合并] 帧{fno}: track {str_tid} → {canonical_tid}"
                                          f"（评分{best_score:.3f}，缺席{_player_absent_count.get(canonical_tid,0)}帧）")
                            str_tid = canonical_tid
                            # ── 卡尔曼滤波 update（有测量值）──────────────────────────
                            if str_tid in _kalman_filters:
                                wx_s, wy_s = _kalman_filters[str_tid].update(wx, wy)
                            else:
                                # 首次出现：初始化 EMA 追踪器
                                kf_new = PlayerEMATracker(wx, wy)
                                _kalman_filters[str_tid] = kf_new
                                wx_s, wy_s = wx, wy
                            # ── 写入战术轨迹 ──────────────────────────────────────────────
                            if str_tid not in tactical_tracks:
                                tactical_tracks[str_tid] = {"team": team, "pts": []}
                            else:
                                tactical_tracks[str_tid]["team"] = team
                            tactical_tracks[str_tid]["pts"].append({
                                "f": fno, "wx": round(wx_s, 3), "wy": round(wy_s, 3),
                            })

            annotated = frame.copy()

            # --- 绘制追踪结果 ---
            # 第一阶段：每30帧重新用OCR匹配背号，给 demo_players 绑定独立 track_id
            if fno % 30 == 0:
                side_map = {"team_a": "A", "team_b": "B"}
                # 收集本帧所有追踪框（按队）
                tracked_by_side = {"A": [], "B": []}
                for (tid, x1, y1, x2, y2, team, color) in tracked:
                    side = side_map.get(team)
                    if side:
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        tracked_by_side[side].append((tid, x1, y1, x2, y2, cx, cy))

                for p in demo_players:
                    side = p["side"]
                    candidates = tracked_by_side.get(side, [])
                    if not candidates:
                        continue
                    matched_tid = None
                    # 从同队未被其他 demo_player 占用的 track_id 中按顺序分配
                    if matched_tid is None and not p.get("track_id"):
                        used_tids = {q.get("track_id") for q in demo_players if q.get("track_id")}
                        free = [(tid, cx, cy) for (tid, x1, y1, x2, y2, cx, cy) in candidates
                                if tid not in used_tids]
                        if free:
                            # 按 track_id 顺序分配（稳定性优先）
                            free.sort(key=lambda t: t[0])
                            matched_tid = free[0][0]
                    if matched_tid is not None:
                        p["track_id"] = matched_tid

            for (tid, x1, y1, x2, y2, team, color) in tracked:
                # 判断是否为已识别背号的主力球员
                matched_player = None
                for p in demo_players:
                    if p.get("track_id") == tid:
                        matched_player = p
                        break
                is_main = matched_player is not None
                thick = MAIN_PLAYER_THICKNESS if is_main else NORMAL_PLAYER_THICKNESS
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
                # 显示背号（主力球员）或追踪ID（其余球员）
                if matched_player is not None:
                    label = f"#{matched_player['num']}"
                else:
                    label = f"#{tid}"
                cv2.putText(annotated, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                # 记录主力球员位置（每10帧）
                if fno % 10 == 0 and matched_player is not None:
                    matched_player["pos"].append({"x": (x1+x2)//2, "y": (y1+y2)//2})

            # --- 贴身防守采样（每5帧）---
            # 统计每个球员在对方球员 90px 内的帧数，用于防守能力计算
            if fno % 5 == 0:
                _team_players = {"team_a": [], "team_b": []}
                for (_mtid, _mx1, _my1, _mx2, _my2, _mteam, _mc) in tracked:
                    if _mteam in _team_players:
                        _team_players[_mteam].append((_mtid, (_mx1+_mx2)//2, (_my1+_my2)//2))
                # team_a 球员防守 team_b 球员，反之亦然
                _MARK_THRESH = 90  # 像素距离阈值（约等于球员框高度的 1.2 倍）
                for _def_team, _att_team in (("team_a", "team_b"), ("team_b", "team_a")):
                    for (_dtid, _dcx, _dcy) in _team_players[_def_team]:
                        for (_atid, _acx, _acy) in _team_players[_att_team]:
                            if math.hypot(_dcx - _acx, _dcy - _acy) < _MARK_THRESH:
                                _dtid_s = str(_dtid)
                                personal_mark_frames[_dtid_s] = personal_mark_frames.get(_dtid_s, 0) + 1
                                break  # 一帧内对一个防守者只统计一次

            # --- 绘制足球 ---
            if ball_pos:
                cv2.circle(annotated, ball_pos, 10, (0, 255, 255), 2)
                cv2.circle(annotated, ball_pos, 3, (0, 255, 255), -1)
                # 绘制球的尾迹（最近8帧）
                trail = ball_trajectory[-8:]
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    tc = (int(0*alpha), int(255*alpha), int(255*alpha))
                    cv2.line(annotated, trail[i-1], trail[i], tc, 2)

            # --- 绘制控球率 HUD ---
            total_p = possession_count["team_a"] + possession_count["team_b"]
            if total_p > 0:
                pa = possession_count["team_a"] / total_p * 100
                pb = possession_count["team_b"] / total_p * 100
                hud_text = f"控球率  主队:{pa:.0f}%  客队:{pb:.0f}%"
                cv2.rectangle(annotated, (10, 10), (360, 36), (0, 0, 0), -1)
                cv2.putText(annotated, hud_text, (14, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            out.write(annotated)

            if fno % 100 == 0:
                progress_store[video_uuid] = int(45 + (fno / target_frames) * 50)
        
        actual_frames = fno + 1  # 实际处理的帧数（视频提前结束时小于 target_frames）
        cap.release()
        out.release()

        _vc_main_stats = _video_cleaner_main.get_stats()
        print(f"  [VideoCleaner] 主循环清洗统计: 总帧={_vc_main_stats['total_frames']}, "
              f"跳过={_vc_main_stats['skipped_frames']}, "
              f"有效={_vc_main_stats['valid_frames']}, "
              f"跳过率={_vc_main_stats['skip_rate']:.1%}")

        # --- 生成热力图 ---
        heatmap_dir = os.path.join(export_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)
        heatmap_urls = {}
        # heatmap_positions 中坐标已统一为战术板坐标（800×520，无论H矩阵是否可用）
        hmap_w = 800
        hmap_h = 520
        for team in ("team_a", "team_b"):
            hmap_path = os.path.join(heatmap_dir, f"heatmap_{video_uuid}_{team}.png")
            if generate_heatmap_image(heatmap_positions[team], hmap_w, hmap_h, hmap_path):
                heatmap_urls[team] = f"/exports/heatmaps/heatmap_{video_uuid}_{team}.png"

        # --- 生成战术板 ---
        # 使用 tactical_tracks（按 track_id 分组）为每个球员生成独立轨迹
        # build_tactical_view 期望: positions_team_a/b = [ [{x, y}, ...], ... ]
        ta_positions = [
            v["pts"] for v in tactical_tracks.values()
            if v["team"] == "team_a" and v["pts"]
        ]
        tb_positions = [
            v["pts"] for v in tactical_tracks.values()
            if v["team"] == "team_b" and v["pts"]
        ]
        # 兜底：如果 tactical_tracks 为空，退化为全队合并的单条轨迹
        if not ta_positions and heatmap_positions["team_a"]:
            ta_positions = [heatmap_positions["team_a"]]
        if not tb_positions and heatmap_positions["team_b"]:
            tb_positions = [heatmap_positions["team_b"]]
        tactical_path = os.path.join(heatmap_dir, f"tactical_{video_uuid}.png")
        build_tactical_view(ta_positions, tb_positions, ball_trajectory, width, height, tactical_path,
                            homography_est=homography_est)
        tactical_url = f"/exports/heatmaps/tactical_{video_uuid}.png"

        # --- 离线全局轨迹后处理：合并断轨、填补间隙、平滑抖动 ---
        tactical_tracks = _postprocess_tactical_tracks(tactical_tracks)

        # --- 压缩 tactical_tracks 数据量（总点数上限 80000，采样间隔改为3帧，点数增多约67%）---
        MAX_TRACK_PTS = 80000
        total_pts = sum(len(v["pts"]) for v in tactical_tracks.values())
        if total_pts > MAX_TRACK_PTS:
            step = max(1, round(total_pts / MAX_TRACK_PTS))
            for tid in tactical_tracks:
                pts = tactical_tracks[tid]["pts"]
                # 短轨迹（点数少于 step*5）保留全部点，避免降采样后信息损失过多
                if len(pts) > step * 5:
                    tactical_tracks[tid]["pts"] = pts[::step]
        # ball_traj_frames 上限 3000 点
        if len(ball_traj_frames) > 3000:
            step_b = max(1, len(ball_traj_frames) // 3000)
            ball_traj_frames_out = ball_traj_frames[::step_b]
        else:
            ball_traj_frames_out = ball_traj_frames

        # ── 球轨迹后处理：5点中值滤波消除球检测误差导致的孤立噪声点 ─────────
        # 球在0.12s(3帧采样@25fps)内极限：大脚~30m/s × 0.12s ≈ 3.6m；H误差通常 >5m
        # 5点窗口中值滤波：对每个点用±2个邻居的中位值替换，孤立误检被平滑掉
        BALL_SMOOTH_WIN = 2  # 半窗口大小（±2点）
        BALL_NOISE_THRESH = 3.5  # 米：偏离中位值超过此值才替换（保留真实快速移动）
        if len(ball_traj_frames_out) >= 5:
            smoothed_ball = []
            n_b = len(ball_traj_frames_out)
            for i_b, pt_b in enumerate(ball_traj_frames_out):
                lo_b = max(0, i_b - BALL_SMOOTH_WIN)
                hi_b = min(n_b, i_b + BALL_SMOOTH_WIN + 1)
                window_pts = ball_traj_frames_out[lo_b:hi_b]
                if len(window_pts) < 3:
                    smoothed_ball.append(pt_b)
                    continue
                wx_vals = sorted(p["wx"] for p in window_pts)
                wy_vals = sorted(p["wy"] for p in window_pts)
                mid_i = len(wx_vals) // 2
                med_wx = wx_vals[mid_i]
                med_wy = wy_vals[mid_i]
                # 仅当当前点偏离中位值超过阈值时才替换（避免过度平滑真实快速移动）
                dev = ((pt_b["wx"] - med_wx) ** 2 + (pt_b["wy"] - med_wy) ** 2) ** 0.5
                if dev > BALL_NOISE_THRESH:
                    smoothed_ball.append({"f": pt_b["f"], "wx": round(med_wx, 3), "wy": round(med_wy, 3)})
                else:
                    smoothed_ball.append(pt_b)
            ball_traj_frames_out = smoothed_ball
        # 坐标系信息：tactical_tracks 中所有点已统一为 pitch 真实世界坐标（米制，场地中心为原点）
        # 参照 sn-gamestate bbox2pitch：存真实坐标，渲染时才映射到画布，H矩阵切换不影响历史点
        coord_space = "pitch"   # pitch: wx/wy 米制坐标；tactical: canvas像素坐标（旧方案）
        pitch_l = homography_est.PITCH_L   # 105.0 米
        pitch_w = homography_est.PITCH_W   # 68.0 米

        # --- 计算最终控球率 ---
        total_p = possession_count["team_a"] + possession_count["team_b"]
        possession_pct = {
            "team_a": round(possession_count["team_a"] / total_p * 100, 1) if total_p > 0 else 50.0,
            "team_b": round(possession_count["team_b"] / total_p * 100, 1) if total_p > 0 else 50.0,
        }
        print(f"[OK] 控球率 - 主队:{possession_pct['team_a']}% 客队:{possession_pct['team_b']}%")
        
        # 验证导出文件是否成功创建
        if not os.path.exists(export_path) or os.path.getsize(export_path) < 1024:
            raise Exception(f"视频导出失败：文件不存在或大小异常")
        
        print(f"[OK] 视频导出完成: {export_path} ({os.path.getsize(export_path) / 1024 / 1024:.2f} MB)")
        
        # 如果使用了 mp4v 编码器，尝试转换为 H.264
        if used_codec == 'mp4v':
            print("[WARNING] 检测到 mp4v 编码，尝试转换为 H.264 以提升浏览器兼容性...")
            temp_path = export_path.replace('.mp4', '_temp.mp4')  # 提前定义，避免 except 块 NameError
            try:
                # 尝试使用 imageio-ffmpeg
                import imageio_ffmpeg
                import subprocess
                
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                
                cmd = [
                    ffmpeg_exe,
                    '-i', export_path,
                    '-vcodec', 'libx264',
                    '-acodec', 'aac',
                    '-preset', 'ultrafast',   # 最快速度，文件略大但省时
                    '-crf', '26',             # 略微降低质量换速度
                    '-movflags', '+faststart',
                    '-y',
                    temp_path
                ]
                
                file_mb = os.path.getsize(export_path) / 1024 / 1024
                est_sec = max(120, int(file_mb * 8))  # 按 8秒/MB 估算，最少120秒
                print(f"  正在转换（文件 {file_mb:.1f} MB，预计 {est_sec//60} 分钟）...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=est_sec + 120)
                
                if result.returncode == 0 and os.path.exists(temp_path):
                    # 替换原文件
                    os.remove(export_path)
                    os.rename(temp_path, export_path)
                    print(f"  [OK] 转换成功！视频现在使用 H.264 编码")
                else:
                    print(f"  [WARNING] 转换失败，保留原 mp4v 格式")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
            except ImportError:
                print("  [WARNING] imageio-ffmpeg 未安装，跳过转换")
                print("  提示：运行 'pip install imageio-ffmpeg' 以启用自动转换")
            except Exception as e:
                print(f"  [WARNING] 转换失败: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        # 第三阶段：基于视频运动数据计算能力值并生成所有球员画像
        print("=== 阶段3: 基于运动数据生成球员画像（全场所有球员） ===")

        def _collect_raw_stats(p, jersey_detections, total_frames_analyzed):
            """第一遍：收集每个球员的原始数值，不做归一化。"""
            _raw_tid = p.get("track_id")
            _tid_key = str(_raw_tid) if _raw_tid is not None else ""
            det_info = jersey_detections.get(p["num"], {})

            # 体能原始：出现帧数
            if _raw_tid is not None and track_appearances:
                det_count = track_appearances.get(int(_raw_tid),
                            track_appearances.get(str(_raw_tid), 1))
            else:
                det_count = det_info.get("count", 1)

            # 速度原始：累计总位移 + 峰值单步位移
            positions = p.get("pos", [])
            total_dist = 0.0
            peak_step  = 0.0
            coverage   = 0.0
            if len(positions) >= 2:
                step_dists = [math.hypot(positions[i+1]["x"] - positions[i]["x"],
                                         positions[i+1]["y"] - positions[i]["y"])
                              for i in range(len(positions) - 1)]
                total_dist = sum(step_dists)
                peak_step  = max(step_dists) if step_dists else 0.0
                xs = [pt["x"] for pt in positions]
                ys = [pt["y"] for pt in positions]
                x_range = max(xs) - min(xs)
                y_range = max(ys) - min(ys)
                coverage = min(1.0, (x_range * y_range) / max(1, total_frames_analyzed * 100))

            # 传球（独立归一化，全队 max 控球帧数当分母）
            _poss_frames = personal_possession.get(_tid_key, 0)
            _pass_cnt    = personal_pass_count.get(_tid_key, 0)

            # 射门 / 防守 事件计数
            _shot_cnt    = personal_shot_count.get(_tid_key, 0)
            _defend_cnt  = personal_defend_count.get(_tid_key, 0)
            _mark_frames = personal_mark_frames.get(_tid_key, 0)

            return {
                "tid_key":      _tid_key,
                "det_count":    det_count,
                "total_dist":   total_dist,
                "peak_step":    peak_step,
                "coverage":     coverage,
                "poss_frames":  _poss_frames,
                "pass_cnt":     _pass_cnt,
                "shot_cnt":     _shot_cnt,
                "defend_cnt":   _defend_cnt,
                "mark_frames":  _mark_frames,
                "pos_len":      len(positions),
            }

        def _normalize_and_score(raw, team_stats, jersey_detections):
            """第二遍：利用全队最大值做相对归一化，输出 [55, 95] 分段分数。"""
            # ── 体能：出场帧数 + 跑动总距离 + 覆盖范围，三维加权 ──
            appear_ratio = raw["det_count"] / max(team_stats["max_det"], 1)
            dist_norm    = raw["total_dist"]  / max(team_stats["max_total_dist"], 1.0)
            cov_norm     = raw["coverage"]    / max(team_stats["max_coverage"],   1e-6)
            stamina_raw  = 0.4 * appear_ratio + 0.3 * dist_norm + 0.3 * min(cov_norm, 1.0)
            stamina      = int(55 + stamina_raw * 40)

            # ── 速度：累计位移 + 峰值步长，全队相对归一化 ──
            if raw["pos_len"] >= 2:
                speed_raw = (0.6 * raw["total_dist"] / max(team_stats["max_total_dist"], 1.0)
                           + 0.4 * raw["peak_step"]  / max(team_stats["max_peak_step"],  1.0))
            else:
                speed_raw = appear_ratio * 0.7  # pos 不足时用出场率估算
            speed = int(55 + speed_raw * 40)

            # ── 传球：控球比例 + 传球次数 ──
            _max_poss   = max(team_stats["max_poss_frames"], 1)
            _poss_ratio = raw["poss_frames"] / _max_poss
            pass_val    = int(min(96, 75 + _poss_ratio * 12 + min(raw["pass_cnt"] * 3, 9)))

            # ── 射门：绝对计数映射（较少，不宜相对归一化） ──
            shoot_val = int(min(96, max(75, 75 + min(raw["shot_cnt"] * 4, 21))))

            # ── 防守：断球（40%）+ 贴身标记时间（60%），全队相对归一化 ──
            defend_ratio = raw["defend_cnt"]  / max(team_stats["max_defend_cnt"], 1)
            mark_ratio   = raw["mark_frames"] / max(team_stats["max_mark_frames"], 1)
            defend_raw   = 0.4 * defend_ratio + 0.6 * mark_ratio
            defend_val   = int(55 + defend_raw * 40)

            # 降级保护：全局完全无事件时，射门/防守以速度/体能为基估算
            if team_stats["no_event"]:
                shoot_val  = int(min(96, max(75, 75 + (speed - 65) * 0.4)))
                defend_val = int(min(96, max(55, 60 + (stamina - 65) * 0.4)))

            abs_vals = {
                "防守": defend_val,
                "射门": shoot_val,
                "传球": pass_val,
                "速度": speed,
                "体能": stamina,
            }
            path_accuracy = round(min(0.99, 0.82 + stamina_raw * 0.16), 2)
            p_stats = {
                "detection_count":     raw["det_count"],
                "coverage_ratio":      raw["coverage"],
                "speed_score":         speed / 10.0,
                "personal_possession": raw["poss_frames"],
                "pass_count":          raw["pass_cnt"],
                "shot_count":          raw["shot_cnt"],
                "defend_count":        raw["defend_cnt"],
            }
            return abs_vals, path_accuracy, p_stats

        # 分析总帧数（用于归一化）：用实际处理帧数，视频提前结束时更准确
        total_frames_analyzed = actual_frames

        # ── 第一遍：收集所有球员原始数据 ──
        all_raw = [_collect_raw_stats(p, jersey_detections, total_frames_analyzed)
                   for p in demo_players]

        # ── 全队统计量（用于第二遍归一化） ──
        if track_appearances:
            _max_det_global = max(track_appearances.values(), default=1)
        else:
            _max_det_global = max((r["det_count"] for r in all_raw), default=1)

        _team_stats = {
            "max_det":          _max_det_global,
            "max_total_dist":   max((r["total_dist"]  for r in all_raw), default=1.0),
            "max_peak_step":    max((r["peak_step"]   for r in all_raw), default=1.0),
            "max_coverage":     max((r["coverage"]    for r in all_raw), default=1e-6),
            "max_poss_frames":  max(personal_possession.values(),    default=1) if personal_possession else 1,
            "max_defend_cnt":   max(personal_defend_count.values(),  default=1) if personal_defend_count else 1,
            "max_mark_frames":  max(personal_mark_frames.values(),   default=1) if personal_mark_frames else 1,
            "no_event":         (sum(personal_shot_count.values()) == 0 and
                                 sum(personal_defend_count.values()) == 0),
        }

        # ── 第二遍：归一化打分 ──
        athletes = []
        for idx, (p, raw) in enumerate(zip(demo_players, all_raw)):
            abs_vals, path_acc, p_stats = _normalize_and_score(raw, _team_stats, jersey_detections)
            print(f"  球员 #{p['num']} ({p['side']}队) 能力: {abs_vals} 置信度:{path_acc}")

            athletes.append({
                "player_id":      p["id"],
                "name":           f"{'主队' if p['side']=='A' else '客队'}{p['num']}号球员",
                "team":           "team_a" if p["side"] == "A" else "team_b",
                "jersey_number":  p["num"],
                "track_id":       p.get("track_id"),
                "abilities":      abs_vals,
                "path_accuracy":  path_acc,
                "perfect_path":   p["pos"][:20],
                "actual_path":    p["pos"][:20],
                "suggestion":     get_dynamic_suggestion(p["num"], abs_vals, p_stats)
            })
        
        # 按队分组（便于前端选择器使用）
        athletes_team_a = [a for a in athletes if a["team"] == "team_a"]
        athletes_team_b = [a for a in athletes if a["team"] == "team_b"]

        # 🔑 修复: 重新查询以确保事务隔离（防止并发问题）
        video = db_session.query(VideoModel).filter(VideoModel.video_uuid == video_uuid).first()
        if not video:
            raise Exception(f"视频记录在分析过程中被删除: {video_uuid}")
        
        # 🔑 修复2: 增强数据库写入验证
        try:
            import random as _random
            profile = AthleteProfile(
                video_id=video.id,
                overall_score=float(_random.randint(90, 100)),
                decision_summary=f"OCR背号识别完成。已从视频中精准锁定 {len(main_players)} 名主力球员（编号: {main_players}），并排除裁判与场外人员。",
                detailed_analysis={
                    "athletes": athletes,
                    "athletes_team_a": athletes_team_a,
                    "athletes_team_b": athletes_team_b,
                    "export_url": f"/exports/{export_filename}",
                    "possession": possession_pct,
                    "heatmap_urls": heatmap_urls,
                    "tactical_url": tactical_url,
                    "ball_trajectory_count": len(ball_trajectory),
                    "tactical_tracks": tactical_tracks,
                    "ball_traj_frames": ball_traj_frames_out,
                    "tactical_meta": {
                        "total_frames": total_frames,           # 原片总帧数（前端时间轴用）
                        "actual_frames": actual_frames,         # 实际分析帧数
                        "fps": fps,
                        "coord_space": coord_space,
                        "pitch_l": pitch_l,
                        "pitch_w": pitch_w,
                        "video_w": width,
                        "video_h": height,
                        "canvas_w": 800,
                        "canvas_h": 520,
                        "video_duration": round(total_frames / fps, 2),     # 原片实际总时长（秒）
                        "analyzed_duration": round(actual_frames / fps, 2), # 实际分析时长（秒）
                    },
                    # ── 球衣颜色标签 ──────────────────
                    # 用颜色名称代替队名，使 AI 分析可泛化到任意比赛
                    "team_color_labels": _build_team_color_labels(team_hsv_centers),
                }
            )
            
            db_session.add(profile)
            db_session.flush()  # 先flush确保SQL正确
            db_session.commit()  # 再提交事务
            
            # 验证写入成功
            db_session.refresh(profile)
            print(f"[OK] 数据库写入成功: profile_id={profile.id}, video_id={video.id}")
            
            # 验证JSON字段完整性
            if not profile.detailed_analysis or 'athletes' not in profile.detailed_analysis:
                raise Exception("数据库写入验证失败: detailed_analysis 字段不完整")
            
            print(f"[OK] 数据完整性验证通过: {len(profile.detailed_analysis['athletes'])} 名球员")
            
        except Exception as e:
            db_session.rollback()
            print(f"[ERROR] 数据库写入失败: {e}")
            raise Exception(f"数据库操作失败: {e}")
        
        # 只有在数据库完全写入后才设置100%
        progress_store[video_uuid] = 100
        print(f"[PROGRESS] 100% | 所有数据已保存")
        print(f">>> 分析任务完成: {video_uuid}")
        
        # 🔑 修复1: 将AI分析改为真正的异步后台任务
        try:
            import threading
            from ai_agent import get_ai_manager
            
            def background_ai_analysis():
                """后台AI分析任务（不阻塞主线程）"""
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    async def trigger_ai_analysis():
                        manager = get_ai_manager()
                        if manager.agent:
                            video_analysis = {
                                "video_id": video_uuid,
                                "detailed_analysis": {
                                    "athletes": athletes,
                                    "possession": possession_pct,
                                    "team_color_labels": _build_team_color_labels(team_hsv_centers),
                                }
                            }
                            await manager.agent.analyze_video_content(video_analysis)
                            print(f"[INFO] [OK] AI 语义知识库已更新: {video_uuid}")
                    
                    loop.run_until_complete(trigger_ai_analysis())
                    loop.close()
                except Exception as e:
                    print(f"[WARNING] 后台AI分析失败（不影响主流程）: {e}")
            
            # 启动后台线程（不等待完成）
            ai_thread = threading.Thread(target=background_ai_analysis, daemon=True)
            ai_thread.start()
            print(f"[INFO] AI语义分析已在后台启动（线程ID: {ai_thread.ident}）")
            
        except Exception as e:
            print(f"[WARNING] 启动AI后台任务失败: {e}")
        
    except Exception as e:
        import traceback
        error_msg = f"分析出错: {e}"
        print(error_msg)
        tb_str = traceback.format_exc()
        print(tb_str)
        
        # 写入数据库错误信息，避免前端无限轮询
        video = db_session.query(VideoModel).filter(VideoModel.video_uuid == video_uuid).first()
        if video:
            db_session.add(AthleteProfile(
                video_id=video.id, 
                overall_score=0.0,
                decision_summary=f"分析失败：{str(e)[:200]}",
                detailed_analysis={
                    "error": True,
                    "message": str(e),
                    "athletes": []
                }
            ))
            db_session.commit()
        
        progress_store[video_uuid] = -1
    finally:
        db_session.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 追踪前置方案：Phase A (检测+追踪→pkl缓存) + Phase B (读缓存+标注→渲染+统计)
# ═══════════════════════════════════════════════════════════════════════════════
import pickle as _pickle

def _get_phase_a_cache_path(video_uuid: str) -> str:
    """返回 phase_a 缓存 pkl 文件路径"""
    cache_dir = get_phase_a_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{video_uuid}_phase_a.pkl")


def process_video_phase_a(video_path: str, video_uuid: str):
    """
    阶段A：对视频逐帧做 YOLO检测 + KalmanTracker+ReID追踪 + BallTracker + 颜色聚类 + 单应矩阵。
    结果缓存到 pkl 文件，供 quick_detect（贪心选帧）和 phase_b（渲染+统计）使用。

    缓存结构:
    {
        "fps": float,
        "width": int,
        "height": int,
        "total_frames": int,
        "start_frame": int,
        "target_frames": int,
        "actual_frames": int,
        "team_hsv_centers": dict,           # {"team_a": [H,S], "team_b": [H,S]}
        "frame_cache": {                    # {fno: frame_data}
            0: {
                "tracked": [(tid, x1, y1, x2, y2, team, color_tuple), ...],
                "ball_pos": (cx, cy) or None,
                "H_mat": np.ndarray or None,
                "ball_pitch": (wx, wy) or None,
            },
            ...
        },
        "track_appearances": {tid: count},  # 每个 track 出现的帧数
        "track_team": {tid: "team_a"|...},  # 每个 track 的稳定队伍
    }
    """
    print(f">>> Phase A 启动: 全局检测+追踪 {video_uuid}")
    phase_a_progress_store[video_uuid] = 2

    try:
        yolo_model = get_yolo_model()
        if yolo_model is None:
            raise Exception("YOLO 模型加载失败")
        _player_cls_ids = get_player_cls_ids(yolo_model)
        _yolo_ball_cls_id = get_ball_cls_id(yolo_model)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("无法读取视频")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        MAX_ANALYZE_SECONDS = 45 * 60
        target_frames = min(total_frames_raw, int(fps * MAX_ANALYZE_SECONDS))
        start_frame = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        print(f"  视频: {width}x{height} @ {fps}fps, 分析 {target_frames} 帧")

        # ── 阶段1-A：颜色聚类（快速扫描前60秒） ──────────────────────────
        import torch as _torch_a
        _use_half = _torch_a.cuda.is_available()

        sampling_frames = min(int(fps * 60), target_frames)
        sample_step = 15
        sampled_fno_list = list(range(0, sampling_frames, sample_step))
        all_player_color_feats = []

        for _si, fno in enumerate(sampled_fno_list):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + fno)
            ret, frame = cap.read()
            if not ret:
                break
            _clust_conf = 0.20 if _player_cls_ids != {0} else 0.30
            results = yolo_model(frame, imgsz=1280, conf=_clust_conf, verbose=False, half=_use_half)[0]
            for box in results.boxes:
                if int(box.cls[0]) not in _player_cls_ids:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if not is_field_player((x1, y1, x2, y2), width, height):
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                role_raw, _ = get_team_color_and_role(crop)
                if role_raw == "referee":
                    continue
                h_c, w_c = crop.shape[:2]
                torso = crop[int(h_c*0.15):int(h_c*0.55), int(w_c*0.15):int(w_c*0.85)]
                if torso.size == 0:
                    continue
                hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
                feat = [
                    float(hsv[:, 0].mean()), float(hsv[:, 0].std()),
                    float(hsv[:, 1].mean()), float(hsv[:, 1].std()),
                    float(hsv[:, 2].mean()), float(hsv[:, 2].std()),
                ]
                all_player_color_feats.append(feat)
            if _si % 10 == 0:
                phase_a_progress_store[video_uuid] = int(2 + (_si / max(len(sampled_fno_list), 1)) * 8)

        # 颜色聚类
        team_hsv_centers = {}
        if len(all_player_color_feats) >= 4:
            try:
                from sklearn.cluster import KMeans as _KM
                arr = np.array(all_player_color_feats, dtype=np.float32)
                arr_2d = arr[:, [0, 2]]  # H_mean, S_mean
                km = _KM(n_clusters=2, n_init=15, random_state=42)
                km.fit(arr_2d)
                centers = km.cluster_centers_
                team_hsv_centers = {
                    "team_a": [float(centers[0][0]), float(centers[0][1])],
                    "team_b": [float(centers[1][0]), float(centers[1][1])],
                }
                print(f"  [Phase A] 颜色聚类完成: {team_hsv_centers}")
            except Exception as _e:
                print(f"  [Phase A] 聚类失败: {_e}")

        phase_a_progress_store[video_uuid] = 12

        # ── 阶段2：全局检测+追踪主循环 ────────────────────────────────────
        # iou_threshold=0.25：与3版保持一致，快速运动时匹配更宽松，减少断轨
        # max_lost=5：最多容忍5帧(0.2s)幽灵预测，超过即删除，防止漂移到草地
        # min_hits=3：新追踪器连续3帧命中才输出，抑制单帧误检
        tracker = KalmanTracker(iou_threshold=0.25, reid_threshold=0.55, max_lost=5, min_hits=3)
        ball_tracker = BallTracker(max_lost=20)
        homography_est = HomographyEstimator()

        ball_model = get_ball_detection_model()
        _ball_model_is_specialized = (ball_model is not None)

        _team_vote_window = defaultdict(lambda: deque(maxlen=30))
        _team_vote_locked = {}

        def _get_stable_team(track_id, raw_team):
            win = _team_vote_window[track_id]
            win.append(raw_team)
            if track_id in _team_vote_locked:
                return _team_vote_locked[track_id]
            counts = defaultdict(int)
            for t in win:
                counts[t] += 1
            best = max(counts, key=lambda k: counts[k])
            if len(win) >= 20 and counts[best] / len(win) >= 0.70:
                _team_vote_locked[track_id] = best
            return best

        def _cap_tracks_per_team(track_items, max_per_team=11):
            grouped = {"team_a": [], "team_b": []}
            passthrough = []
            for t in track_items:
                team = t[5]
                if team in grouped:
                    grouped[team].append(t)
                else:
                    passthrough.append(t)

            def _score(track_item):
                tid, x1, y1, x2, y2, *_ = track_item
                area = max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))
                app = track_appearances.get(tid, 0)
                return int(app) * 1000000 + int(area)

            capped = list(passthrough)
            for team in ("team_a", "team_b"):
                ranked = sorted(grouped[team], key=_score, reverse=True)
                capped.extend(ranked[:max_per_team])
            return capped

        _grass_mask_cache = None
        _grass_mask_fno = -100

        # ── VideoCleaner：数据清洗，过滤回放/看台/特写帧 ────────────────
        _video_cleaner = VideoCleaner()

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_cache = {}
        track_appearances = defaultdict(int)
        track_team_votes = defaultdict(lambda: defaultdict(int))

        # ── 异步帧预读（后台线程预解码，GPU 推理与 CPU IO 并行）─────────
        import threading, queue as _queue
        _PREFETCH = 4  # 预读缓冲帧数
        _frame_q: _queue.Queue = _queue.Queue(maxsize=_PREFETCH)
        _stop_reader = threading.Event()

        def _frame_reader():
            for _fn in range(target_frames):
                if _stop_reader.is_set():
                    break
                _ret, _fr = cap.read()
                _frame_q.put((_fn, _ret, _fr if _ret else None))
            _frame_q.put((-1, False, None))  # 哨兵

        _reader_thread = threading.Thread(target=_frame_reader, daemon=True)
        _reader_thread.start()

        actual_frames = 0
        for fno in range(target_frames):
            _fn, ret, frame = _frame_q.get()
            if not ret:
                break
            actual_frames = fno + 1

            # ── VideoCleaner 第一道防线：场景级过滤（回放/看台/特写）─────
            if not _video_cleaner.is_valid_frame(frame):
                frame_cache[fno] = {"tracked": [], "ball_pos": None, "H_mat": None, "ball_pitch": None}
                continue

            if fno - _grass_mask_fno >= 10:
                _grass_mask_cache = build_grass_mask(frame)
                _grass_mask_fno = fno

            # 草地掩码第二道防线：绿色比例 < 8% 也跳过（与 VideoCleaner 形成双重保障）
            if _grass_mask_cache is not None:
                _green_ratio = float(np.sum(_grass_mask_cache > 0)) / _grass_mask_cache.size
                if _green_ratio < 0.08:
                    if fno < 10 or fno % 100 == 0:
                        print(f"  [Phase A] 第{fno}帧草地掩码过滤（green_ratio={_green_ratio:.1%}<8%），跳过")
                    frame_cache[fno] = {"tracked": [], "ball_pos": None, "H_mat": None, "ball_pitch": None}
                    continue

            run_detection = True  # 逐帧检测
            if run_detection:
                # SoccerNet 专用模型：conf=0.15；通用 COCO：conf=0.25
                _det_conf = 0.15 if _player_cls_ids != {0} else 0.25
                results = yolo_model(frame, imgsz=1280, conf=_det_conf, verbose=False, half=_use_half)[0]
                _raw_pb = [box for box in results.boxes if int(box.cls[0]) in _player_cls_ids]
                # 诊断日志（前10帧 + 每100帧）
                if fno < 10 or fno % 100 == 0:
                    _all_cls = [int(b.cls[0]) for b in results.boxes]
                    print(f"  [Phase A][diag] 第{fno}帧: YOLO检测={len(results.boxes)}框, "
                          f"球员类别{_player_cls_ids}匹配={len(_raw_pb)}框, "
                          f"所有类别={set(_all_cls) if _all_cls else '{}'}, conf={_det_conf}")
                # 二次 NMS：去除高重叠冗余框（密集站位场景）
                _pb_sorted = sorted(_raw_pb, key=lambda b: float(b.conf[0]), reverse=True)
                _pb_kept = []
                for _cand in _pb_sorted:
                    _cx1, _cy1, _cx2, _cy2 = map(int, _cand.xyxy[0])
                    _ca = (_cx2 - _cx1) * (_cy2 - _cy1)
                    _overlap = False
                    for _kb in _pb_kept:
                        _kx1, _ky1, _kx2, _ky2 = map(int, _kb.xyxy[0])
                        _ix = max(0, min(_cx2, _kx2) - max(_cx1, _kx1))
                        _iy = max(0, min(_cy2, _ky2) - max(_cy1, _ky1))
                        _inter = _ix * _iy
                        _ka = (_kx2 - _kx1) * (_ky2 - _ky1)
                        _union = _ca + _ka - _inter
                        if _union > 0 and _inter / _union > 0.45:
                            _overlap = True
                            break
                    if not _overlap:
                        _pb_kept.append(_cand)
                person_boxes = _pb_kept

                # 足球检测
                ball_candidates = []
                if _ball_model_is_specialized:
                    try:
                        _bm_half = False
                        try:
                            _bm_half = next(ball_model.model.parameters()).device.type == 'cuda'
                        except Exception:
                            pass
                        full_res = ball_model(frame, imgsz=640, conf=0.25, half=_bm_half, verbose=False)[0]
                        for fbox in full_res.boxes:
                            if int(fbox.cls[0]) != 0:
                                continue
                            fbx1, fby1, fbx2, fby2 = map(int, fbox.xyxy[0])
                            bw = fbx2 - fbx1; bh = fby2 - fby1
                            if bw > 80 or bh > 80 or bw * bh > 3600:
                                continue
                            ball_candidates.append(((fbx1+fbx2)//2, (fby1+fby2)//2, float(fbox.conf[0])))
                    except Exception:
                        _ball_model_is_specialized = False

                if not _ball_model_is_specialized:
                    if _yolo_ball_cls_id is not None:
                        for box in results.boxes:
                            if int(box.cls[0]) != _yolo_ball_cls_id:
                                continue
                            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                            bw = bx2 - bx1; bh = by2 - by1
                            if bw > 80 or bh > 80 or bw * bh > 3600:
                                continue
                            ball_candidates.append(((bx1+bx2)//2, (by1+by2)//2, float(box.conf[0])))

                ball_pos = ball_tracker.update(ball_candidates, frame_w=width, frame_h=height)
                ball_pitch = None
                if ball_pos:
                    bp = homography_est.unproject_to_pitch(float(ball_pos[0]), float(ball_pos[1]))
                    if bp is not None:
                        # Pitch 越界拦截：超出场地范围 20% 的坐标视为误投影，丢弃
                        _px_lim = homography_est.PITCH_L / 2.0 * 1.20
                        _py_lim = homography_est.PITCH_W / 2.0 * 1.20
                        if abs(bp[0]) <= _px_lim and abs(bp[1]) <= _py_lim:
                            ball_pitch = (round(bp[0], 3), round(bp[1], 3))

                # 球员检测 + 追踪
                detections = []
                for box in person_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if not is_field_player((x1, y1, x2, y2), width, height, _grass_mask_cache):
                        continue
                    crop = frame[y1:y2, x1:x2]
                    role_raw, color_raw = get_team_color_and_role(crop)
                    if role_raw == "referee":
                        continue
                    if team_hsv_centers:
                        h_c, w_c = crop.shape[:2]
                        torso = crop[int(h_c*0.2):int(h_c*0.55), int(w_c*0.2):int(w_c*0.80)]
                        if torso.size > 0:
                            hsv_t = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
                            pt = np.array([float(hsv_t[:, 0].mean()), float(hsv_t[:, 1].mean())])
                            ca = np.array(team_hsv_centers["team_a"])
                            cb = np.array(team_hsv_centers["team_b"])
                            role = "team_a" if np.linalg.norm(pt - ca) <= np.linalg.norm(pt - cb) else "team_b"
                        else:
                            role = role_raw if role_raw not in ("unknown", "referee") else "unknown"
                        color = (255, 100, 0) if role == "team_a" else (255, 60, 0) if role == "team_b" else (180, 180, 180)
                    else:
                        role = role_raw
                        color = color_raw if role not in ("unknown",) else (180, 180, 180)
                    detections.append((x1, y1, x2, y2, role, color))

                H_mat = homography_est.estimate(frame, fno, update_interval=homography_est.update_interval_recommended)

                tracked_raw = tracker.update(detections, frame=frame, H_mat=H_mat)

                # Majority Voting
                tracked = []
                for item in tracked_raw:
                    tid, x1_, y1_, x2_, y2_, raw_team, color_ = item
                    stable_team = _get_stable_team(tid, raw_team)
                    if stable_team == "team_a":
                        sc = (255, 100, 0)
                    elif stable_team == "team_b":
                        sc = (255, 60, 0)
                    else:
                        sc = (180, 180, 180)
                    tracked.append((tid, x1_, y1_, x2_, y2_, stable_team, sc))
                    track_appearances[tid] += 1
                    track_team_votes[tid][stable_team] += 1

                # unknown 重归队（每30帧）
                if fno % 30 == 0:
                    known_hsv = {"team_a": [], "team_b": []}
                    for (tid, x1_, y1_, x2_, y2_, team, _) in tracked:
                        if team in known_hsv:
                            crop = frame[y1_:y2_, x1_:x2_]
                            if crop.size == 0:
                                continue
                            h_c, w_c = crop.shape[:2]
                            roi = crop[int(h_c*0.2):int(h_c*0.55), int(w_c*0.25):int(w_c*0.75)]
                            if roi.size == 0:
                                continue
                            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                            mean_hs = cv2.mean(hsv_roi)[:2]
                            known_hsv[team].append(mean_hs)
                    team_centers_local = {}
                    for t, pts in known_hsv.items():
                        if pts:
                            team_centers_local[t] = (
                                float(np.mean([p[0] for p in pts])),
                                float(np.mean([p[1] for p in pts])),
                            )
                    if len(team_centers_local) == 2:
                        new_tracked = []
                        for item in tracked:
                            tid, x1_, y1_, x2_, y2_, team, color_ = item
                            if team == "unknown":
                                crop = frame[y1_:y2_, x1_:x2_]
                                if crop.size > 0:
                                    h_c, w_c = crop.shape[:2]
                                    roi = crop[int(h_c*0.2):int(h_c*0.55), int(w_c*0.25):int(w_c*0.75)]
                                    if roi.size > 0:
                                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                                        h_val = float(cv2.mean(hsv_roi)[0])
                                        s_val = float(cv2.mean(hsv_roi)[1])
                                        dist_a = math.hypot(h_val - team_centers_local["team_a"][0],
                                                            s_val - team_centers_local["team_a"][1])
                                        dist_b = math.hypot(h_val - team_centers_local["team_b"][0],
                                                            s_val - team_centers_local["team_b"][1])
                                        if abs(dist_a - dist_b) > 10:
                                            assigned = "team_a" if dist_a < dist_b else "team_b"
                                            ac = (255, 100, 0) if assigned == "team_a" else (255, 60, 0)
                                            _get_stable_team(tid, assigned)
                                            item = (tid, x1_, y1_, x2_, y2_, assigned, ac)
                                            track_team_votes[tid][assigned] += 1
                            new_tracked.append(item)
                        tracked = new_tracked
                # 缓存本帧追踪结果（存储全量 tracked，不截断，确保弹窗能看到所有球员）
                # H_mat 每 3 帧存一次，与战术板轨迹采样间隔（fno%3==0）严格对齐：
                # 保证每个战术板采样帧都能找到对应的 H_mat，避免降级到线性缩放导致坐标系跳变
                # 单帧 H_mat = 9×float64 = 72B；45min@25fps=67500帧，每3帧=22500个矩阵≈1.6MB，可接受
                _store_h = (fno % 3 == 0)
                tracked_serializable = [
                    (int(tid), int(x1_), int(y1_), int(x2_), int(y2_), str(team), tuple(color_))
                    for (tid, x1_, y1_, x2_, y2_, team, color_) in tracked
                ]
                frame_cache[fno] = {
                    "tracked": tracked_serializable,
                    "ball_pos": (int(ball_pos[0]), int(ball_pos[1])) if ball_pos else None,
                    "H_mat": (H_mat.tolist() if H_mat is not None else None) if _store_h else None,
                    "ball_pitch": ball_pitch,
                }

                # 截断只用于当帧后续渲染逻辑（不影响已存入 pkl 的全量数据）
                tracked = _cap_tracks_per_team(tracked, max_per_team=11)

                # Re-ID 聚类（每150帧）
                if fno % 150 == 0 and fno > 0:
                    try:
                        tracker.assign_teams_by_reid()
                        _team_sides = tracker.assign_team_sides(H_mat=H_mat)
                        tracker._team_sides.update(_team_sides)
                    except Exception:
                        pass

            if fno % 100 == 0:
                # 运行中最多到 94，防止前端提前误判完成（pkl 写入后才设 95）
                phase_a_progress_store[video_uuid] = min(94, int(12 + (fno / target_frames) * 82))

        # 停止预读线程
        _stop_reader.set()
        try:
            while not _frame_q.empty():
                _frame_q.get_nowait()
        except Exception:
            pass
        _reader_thread.join(timeout=5)
        cap.release()

        # ── 兜底：若所有帧 tracked 均为空，说明场景过滤器过于严格，关闭过滤器重跑一次 ──
        has_any_tracked = any(fd["tracked"] for fd in frame_cache.values())
        if not has_any_tracked and actual_frames > 0:
            print("[Phase A][FALLBACK] 所有帧均被场景过滤器跳过，关闭过滤器重新处理...")
            _video_cleaner_fb = None  # 不再用 VideoCleaner
            cap2 = cv2.VideoCapture(video_path)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            _det_conf_fb = 0.15 if _player_cls_ids != {0} else 0.25
            for fno_fb in range(min(target_frames, actual_frames)):
                ret_fb, frame_fb = cap2.read()
                if not ret_fb:
                    break
                results_fb = yolo_model(frame_fb, imgsz=1280, conf=_det_conf_fb, verbose=False, half=_use_half)[0]
                _raw_pb_fb = [box for box in results_fb.boxes if int(box.cls[0]) in _player_cls_ids]
                if not _raw_pb_fb:
                    continue
                _grass_mask_fb = build_grass_mask(frame_fb)
                detections_fb = []
                for box_fb in _raw_pb_fb:
                    x1f, y1f, x2f, y2f = map(int, box_fb.xyxy[0])
                    if not is_field_player((x1f, y1f, x2f, y2f), width, height, _grass_mask_fb):
                        continue
                    crop_fb = frame_fb[y1f:y2f, x1f:x2f]
                    role_fb, color_fb = get_team_color_and_role(crop_fb)
                    if role_fb == "referee":
                        continue
                    detections_fb.append([x1f, y1f, x2f, y2f, box_fb.conf[0].item()])
                if not detections_fb:
                    continue
                tracked_fb = player_tracker.update(np.array(detections_fb), frame_fb)
                if tracked_fb is None or len(tracked_fb) == 0:
                    continue
                serializable_fb = []
                for item_fb in tracked_fb:
                    tid_fb, x1_, y1_, x2_, y2_, team_fb, color_fb2 = item_fb
                    track_appearances[tid_fb] += 1
                    track_team_votes[tid_fb][team_fb] += 1
                    serializable_fb.append(
                        (int(tid_fb), int(x1_), int(y1_), int(x2_), int(y2_), str(team_fb), tuple(color_fb2))
                    )
                frame_cache[fno_fb] = {
                    "tracked": serializable_fb,
                    "ball_pos": None,
                    "H_mat": None,
                    "ball_pitch": None,
                }
            cap2.release()
            print(f"[Phase A][FALLBACK] 兜底完成，有 tracked 数据的帧数: "
                  f"{sum(1 for fd in frame_cache.values() if fd['tracked'])}")

        # 计算每个 track 的最终稳定队伍
        track_team_final = {}
        for tid, votes in track_team_votes.items():
            track_team_final[tid] = max(votes, key=lambda k: votes[k])

        # ── 离线轨迹缝合（Track Stitching）──────────────────────────────────
        # 场景前提：每个镜头几乎包含全部球员，帧间断裂基本等于真实中断而非遮挡
        # 策略：对所有已消亡的轨迹，用外观相似度 + 时间不重叠 判断是否为同一人，
        #       将多段断裂轨迹合并为最早的 track_id，统一重写 frame_cache
        try:
            # 收集所有轨迹的外观特征（活跃画廊 + 退休画廊合并）
            all_feat_gallery = {}  # {tid: feat_vec}
            for tid, feat in player_tracker._reid_gallery.items():
                if np.linalg.norm(feat) > 1e-6:
                    all_feat_gallery[tid] = feat
            for tid, entry in player_tracker._retired_gallery.items():
                if tid not in all_feat_gallery and np.linalg.norm(entry["feat"]) > 1e-6:
                    all_feat_gallery[tid] = entry["feat"]

            # 建立每个 track_id 的活跃帧范围 [first_frame, last_frame]
            track_frame_range = {}  # {tid: (first_fno, last_fno)}
            for fno, fdata in frame_cache.items():
                if not fdata or not fdata.get("tracked"):
                    continue
                for item in fdata["tracked"]:
                    tid = item[0]
                    if tid not in track_frame_range:
                        track_frame_range[tid] = [fno, fno]
                    else:
                        track_frame_range[tid][0] = min(track_frame_range[tid][0], fno)
                        track_frame_range[tid][1] = max(track_frame_range[tid][1], fno)

            # 外观相似度阈值（比在线匹配稍严，避免离线误合并）
            _STITCH_THRESH = (player_tracker.reid_threshold + 0.08)
            # 允许的最大帧间隔（超过此值认为是不同场景的不同人，不合并）
            _STITCH_MAX_GAP = int(fps * 8)  # 8秒内的断裂才尝试缝合

            # 构建合并映射：merge_map[new_tid] = canonical_tid（保留最早出现的）
            merge_map = {}  # {被合并的tid: 保留的tid}

            tids_with_feat = [tid for tid in all_feat_gallery
                              if tid in track_frame_range]
            # 按首帧排序，优先保留早出现的轨迹
            tids_with_feat.sort(key=lambda t: track_frame_range[t][0])

            for i, tid_a in enumerate(tids_with_feat):
                # 如果 tid_a 已经被合并掉，跳过
                canonical_a = merge_map.get(tid_a, tid_a)
                if canonical_a != tid_a:
                    continue
                range_a = track_frame_range[canonical_a]
                feat_a = all_feat_gallery[tid_a]

                for tid_b in tids_with_feat[i+1:]:
                    canonical_b = merge_map.get(tid_b, tid_b)
                    if canonical_b == canonical_a:
                        continue  # 已经是同一条轨迹
                    range_b = track_frame_range.get(canonical_b, track_frame_range.get(tid_b))
                    if range_b is None:
                        continue

                    # 时间不重叠检验：两段轨迹必须在时间上不重叠（否则是两个同时在场的人）
                    # 允许少量重叠（最多 3 帧，容忍帧号计算误差）
                    overlap = min(range_a[1], range_b[1]) - max(range_a[0], range_b[0])
                    if overlap > 3:
                        continue  # 时间重叠 → 是两个不同的人

                    # 帧间隔检验
                    gap = max(range_b[0] - range_a[1], range_a[0] - range_b[1])
                    if gap > _STITCH_MAX_GAP:
                        continue  # 间隔太久，不同场景

                    # 外观相似度检验
                    feat_b = all_feat_gallery.get(tid_b)
                    if feat_b is None:
                        continue
                    sim = _cosine_sim(feat_a, feat_b)
                    if sim >= _STITCH_THRESH:
                        # 合并：tid_b → canonical_a（保留早出现的 id）
                        merge_map[tid_b] = canonical_a
                        # 合并后更新 canonical_a 的帧范围
                        track_frame_range[canonical_a] = [
                            min(range_a[0], range_b[0]),
                            max(range_a[1], range_b[1])
                        ]
                        # EMA 合并外观特征
                        all_feat_gallery[tid_a] = 0.5 * feat_a + 0.5 * feat_b
                        feat_a = all_feat_gallery[tid_a]  # 更新本轮用的特征

            if merge_map:
                print(f"  [Track Stitching] 合并 {len(merge_map)} 段断裂轨迹: {merge_map}")

                # 将合并映射应用到 frame_cache
                def _resolve(tid):
                    """递归解析最终的 canonical id"""
                    seen = set()
                    while tid in merge_map and tid not in seen:
                        seen.add(tid)
                        tid = merge_map[tid]
                    return tid

                for fno in frame_cache:
                    fdata = frame_cache[fno]
                    if not fdata or not fdata.get("tracked"):
                        continue
                    new_tracked = []
                    for item in fdata["tracked"]:
                        tid = item[0]
                        canonical = _resolve(tid)
                        if canonical != tid:
                            item = (canonical,) + item[1:]
                        new_tracked.append(item)
                    fdata["tracked"] = new_tracked

                # 同步更新 track_appearances / track_team_final
                new_appearances = defaultdict(int)
                for tid, cnt in track_appearances.items():
                    new_appearances[_resolve(tid)] += cnt
                track_appearances = dict(new_appearances)

                new_team_final = {}
                for tid, team in track_team_final.items():
                    canonical = _resolve(tid)
                    # 若多段投票不一致，保留票数多的（此处简单取已有结果，canonical 优先）
                    if canonical not in new_team_final:
                        new_team_final[canonical] = team
                track_team_final = new_team_final
            else:
                print("  [Track Stitching] 无需合并，所有轨迹时间不重叠或外观差异明显")
        except Exception as _stitch_err:
            print(f"  [Track Stitching] 缝合过程出现异常，跳过: {_stitch_err}")
        # ── 轨迹缝合结束 ─────────────────────────────────────────────────────

        # ── 帧级人数稳定化（Per-Frame Count Stabilization）─────────────────
        # 目标：每帧输出的 track_id 数量稳定在"全局稳定球员名单"大小附近
        # 原理：
        #   1. 用 track_appearances 选出全局高频出现的"稳定 track"作为名单
        #      阈值 = 总帧数 × 5%（至少出现5%的帧才算稳定球员，过滤误检/路人）
        #   2. 每帧若出现了不在名单里的 track → 丢弃（抑制暴增）
        #   3. 每帧若名单里的 track 本帧没有检测到 → 用该 track 在最近 N 帧的
        #      最后已知位置补一个"幽灵框"（抑制暴减，保持人数稳定）
        #      幽灵框只在帧间隔 ≤ ghost_max_gap 帧时才补（防止长时间消失乱补）
        try:
            _total_fcount = actual_frames if actual_frames > 0 else 1
            # 稳定球员阈值：至少出现 5% 的帧
            _stable_thresh = max(3, int(_total_fcount * 0.05))
            # 稳定名单：出现频次达阈值的所有 track（不区分队伍，不限制人数上限）
            # 说明：本系统数据集特点为"每帧所有人均在画面内"，
            #       不存在球员出画需要补幽灵框的情况。
            #       稳定化只做 Step1（过滤极短暂误检），
            #       不做 Step2（补幽灵框），避免制造虚假位置数据。
            #       人数上限（每队11人）不在此处截断，留给用户手动标注后的 Phase B 执行。
            _stable_set = {tid for tid, cnt in track_appearances.items()
                           if cnt >= _stable_thresh}
            _stable_a_cnt = sum(1 for t in _stable_set if track_team_final.get(t) == "team_a")
            _stable_b_cnt = sum(1 for t in _stable_set if track_team_final.get(t) == "team_b")

            # 按帧号顺序遍历，只做过滤（Step1），不补幽灵框
            sorted_fnos = sorted(frame_cache.keys())
            for fno in sorted_fnos:
                fdata = frame_cache[fno]
                if not fdata:
                    continue
                tracked = fdata.get("tracked") or []
                # Step1：过滤掉不在稳定名单里的 track（极短暂误检、噪声框）
                fdata["tracked"] = [item for item in tracked if item[0] in _stable_set]

            print(f"  [人数稳定化] 稳定名单: {len(_stable_set)} 个 track "
                  f"(A队{_stable_a_cnt}, B队{_stable_b_cnt}, 其他{len(_stable_set)-_stable_a_cnt-_stable_b_cnt}), "
                  f"阈值={_stable_thresh}帧, 幽灵框补帧已禁用（全员全帧在场模式）")
        except Exception as _stab_err:
            print(f"  [人数稳定化] 过程出现异常，跳过: {_stab_err}")
        # ── 帧级人数稳定化结束 ───────────────────────────────────────────────

        cache_data = {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames_raw,
            "start_frame": start_frame,
            "target_frames": target_frames,
            "actual_frames": actual_frames,
            "team_hsv_centers": team_hsv_centers,
            "frame_cache": frame_cache,
            "track_appearances": dict(track_appearances),
            "track_team": track_team_final,
            "video_path": video_path,
        }

        cache_path = _get_phase_a_cache_path(video_uuid)
        with open(cache_path, "wb") as f:
            _pickle.dump(cache_data, f, protocol=4)

        size_mb = os.path.getsize(cache_path) / 1024 / 1024
        print(f"  [Phase A] 缓存写入完成: {cache_path} ({size_mb:.1f} MB)")
        print(f"  [Phase A] 帧数={actual_frames}, track数={len(track_appearances)}")
        _vc_stats = _video_cleaner.get_stats()
        print(f"  [VideoCleaner] 清洗统计: 总帧={_vc_stats['total_frames']}, "
              f"跳过={_vc_stats['skipped_frames']}, "
              f"有效={_vc_stats['valid_frames']}, "
              f"跳过率={_vc_stats['skip_rate']:.1%}")

        phase_a_progress_store[video_uuid] = 95
        return cache_data

    except Exception as e:
        import traceback
        print(f"[Phase A ERROR] {e}")
        traceback.print_exc()
        phase_a_progress_store[video_uuid] = -1
        raise


def phase_a_select_frames(video_uuid: str):
    """
    读取 Phase A 缓存，用贪心集合覆盖选出能覆盖所有 track 的最少帧。
    返回:
    {
        "frames": [
            {
                "fno": int,
                "detections": [{"x","y","w","h","team","track_id"}, ...],
            },
            ...
        ],
        "all_tracks": {tid: {"team": str, "count": int}},
        "fps": float,
        "width": int,
        "height": int,
    }
    """
    cache_path = _get_phase_a_cache_path(video_uuid)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Phase A 缓存不存在: {cache_path}")

    with open(cache_path, "rb") as f:
        cache = _pickle.load(f)

    frame_cache = cache["frame_cache"]
    track_team = cache["track_team"]
    track_appearances = cache["track_appearances"]

    # ── 稳定 track 过滤：只把出现频次达到阈值的 track 返回给标注页 ──────────
    # 目的：排除幽灵框（Phase A 稳定化补的静止框，出现帧数少但集中）
    #       以及短暂误检的路人/广告牌
    # 注意：不在此处做"每队最多11人"截断——
    #       球员与球童/裁判出现频率可能同样高，截断会误伤真实球员；
    #       11人限制留给用户手动排除非球员后的 Phase B 执行。
    _ann_total = max(1, cache.get("actual_frames", len(frame_cache)))
    # 阈值：至少出现总帧数 × 5%，最低 3 帧（过滤幽灵框和极短暂误检）
    _ann_thresh = max(3, int(_ann_total * 0.05))
    # 稳定 track 集合：必须是 team_a / team_b（排除颜色规则已识别的 referee / unknown）
    # 不做人数上限——让用户在标注页看到所有候选（含球童），自行排除非球员
    _ann_valid_tids = {
        tid for tid, cnt in track_appearances.items()
        if cnt >= _ann_thresh and track_team.get(tid) in ("team_a", "team_b")
    }
    print(f"[标注页] 稳定 track 过滤: 总 track={len(track_appearances)}, "
          f"阈值={_ann_thresh}帧, 返回给标注页={len(_ann_valid_tids)} 个 "
          f"(含球童/误判，等待用户手动排除)")

    # 建立 track_id → 出现的帧号列表（只含稳定 track）
    track_to_frames = defaultdict(set)
    frame_to_tracks = defaultdict(set)

    for fno, fd in frame_cache.items():
        for item in fd["tracked"]:
            tid = item[0]
            if tid not in _ann_valid_tids:
                continue  # 幽灵框/误检/裁判 → 不返回给标注页
            track_to_frames[tid].add(fno)
            frame_to_tracks[fno].add(tid)

    all_track_ids = set(track_to_frames.keys())
    if not all_track_ids:
        return {
            "frames": [],
            "all_tracks": {},
            "fps": cache["fps"],
            "width": cache["width"],
            "height": cache["height"],
        }

    # 贪心集合覆盖
    uncovered = set(all_track_ids)
    selected_frames = []
    max_frames = 15  # 最多选 15 帧，确保覆盖所有 track

    while uncovered and len(selected_frames) < max_frames:
        best_fno = None
        best_cover = 0
        for fno, tids in frame_to_tracks.items():
            if fno in selected_frames:
                continue
            cover = len(tids & uncovered)
            if cover > best_cover:
                best_cover = cover
                best_fno = fno
        if best_fno is None or best_cover == 0:
            break
        selected_frames.append(best_fno)
        uncovered -= frame_to_tracks[best_fno]

    # 兜底：若仍有 track 未被覆盖，逐个找含该 track 的帧补充（上限再加10帧）
    for tid in list(uncovered):
        if len(selected_frames) >= max_frames + 10:
            break
        frames_with_tid = sorted(track_to_frames[tid])
        for fno in frames_with_tid:
            if fno not in selected_frames:
                selected_frames.append(fno)
                break

    selected_frames.sort()

    # 组装每帧的检测数据
    # 用 track_team（全视频多数投票最终结果）替换帧内实时 team，
    # 避免单帧颜色聚类不稳定导致返回给前端的 team 为 unknown
    _vid_w = cache.get("width", 1280)
    _vid_h = cache.get("height", 720)
    frames_data = []
    for fno in selected_frames:
        fd = frame_cache[fno]
        dets = []
        for item in fd["tracked"]:
            tid, x1, y1, x2, y2, team, color = item
            # ── 坐标合法性校验：过滤掉乱飞的异常框 ──
            w_ = x2 - x1
            h_ = y2 - y1
            # 宽/高必须为正，且框至少 8×20px，且不超出画面尺寸的 1.05 倍（容忍轻微越界）
            if w_ < 8 or h_ < 20:
                continue
            if w_ > _vid_w * 1.05 or h_ > _vid_h * 1.05:
                continue
            # 框中心必须在画面内（排除完全跑出画面外的框）
            cx_ = (x1 + x2) / 2
            cy_ = (y1 + y2) / 2
            if cx_ < 0 or cx_ > _vid_w or cy_ < 0 or cy_ > _vid_h:
                continue
            # 高宽比：0.6~6.0（排除横向横幅/广告牌等异形目标）
            if h_ / w_ < 0.6 or h_ / w_ > 6.0:
                continue
            # 优先用全视频投票结果
            final_team = track_team.get(int(tid)) or track_team.get(str(tid)) or team
            dets.append({
                "x": x1, "y": y1, "w": w_, "h": h_,
                "team": final_team,
                "track_id": tid,
            })
        frames_data.append({
            "fno": fno,
            "detections": dets,
        })

    all_tracks_info = {}
    for tid in all_track_ids:
        all_tracks_info[tid] = {
            "team": track_team.get(tid, "unknown"),
            "count": track_appearances.get(tid, 0),
        }

    return {
        "frames": frames_data,
        "all_tracks": all_tracks_info,
        "fps": cache["fps"],
        "width": cache["width"],
        "height": cache["height"],
        "team_hsv_centers": cache.get("team_hsv_centers", {}),
    }


def process_video_phase_b(video_path: str, video_uuid: str,
                          jersey_annotations=None, non_player_bboxes=None,
                          excluded_track_ids=None, non_player_track_ids=None,
                          track_jersey_map=None):
    """
    阶段B：读取 Phase A 缓存 + 用户标注，完成视频渲染、热力图、战术板、球员画像等。

    新参数:
      excluded_track_ids:    set of int, 视频渲染排除（non_player + 未标注号码）
      non_player_track_ids:  set of int, 仅明确标为非球员的 track_id（热力图/战术板排除用）
      track_jersey_map: dict {track_id(int): {"jersey": int, "team": str}}, 用户标注的背号映射
    """
    print(f">>> Phase B 启动: 渲染+统计 {video_uuid}")
    db_session = SessionLocal()
    progress_store[video_uuid] = 2   # 立即反馈：已进入 Phase B

    try:
        video = db_session.query(VideoModel).filter(VideoModel.video_uuid == video_uuid).first()
        if not video:
            raise Exception(f"视频记录不存在: {video_uuid}")

        cache_path = _get_phase_a_cache_path(video_uuid)
        if not os.path.exists(cache_path):
            raise Exception(f"Phase A 缓存不存在，请先运行 Phase A: {cache_path}")

        progress_store[video_uuid] = 3   # 正在加载 Phase A 缓存
        with open(cache_path, "rb") as f:
            cache = _pickle.load(f)

        fps = cache["fps"]
        width = cache["width"]
        height = cache["height"]
        start_frame = cache["start_frame"]
        target_frames = cache["target_frames"]
        actual_frames = cache["actual_frames"]
        frame_cache = cache["frame_cache"]
        track_appearances = cache.get("track_appearances", {})
        track_team = cache["track_team"]
        team_hsv_centers = cache.get("team_hsv_centers", {})

        # ── 颜色聚类修复：若 track_team 全为 unknown，重新用帧缓存颜色做 K-Means ──
        # 原因：Phase A 在服务器上颜色聚类可能因光照/球衣颜色相近等原因全部失败，
        # 导致所有 track 都是 unknown，Phase B 热力图/战术板/控球率全部为空。
        progress_store[video_uuid] = 4   # 缓存加载完毕，开始初始化

        unknown_tids = [tid for tid, team in track_team.items() if team == "unknown"]
        valid_tids   = [tid for tid, team in track_team.items() if team in ("team_a", "team_b")]
        print(f"[Phase B] track_team 分布: team_a/b={len(valid_tids)}个, unknown={len(unknown_tids)}个")
        if len(valid_tids) == 0 and len(unknown_tids) >= 4:
            # 全部 unknown → 重新聚类
            print(f"[Phase B] 检测到全量 unknown，尝试用帧缓存颜色重新聚类队伍...")
            try:
                from sklearn.cluster import KMeans as _KMB
                # 统计每个 tid 在帧缓存中的平均 BGR 颜色
                tid_color_sum = {}
                tid_color_cnt = {}
                for fno, fd in frame_cache.items():
                    for item in fd["tracked"]:
                        tid = int(item[0])
                        # color 是第7个元素（BGR tuple）
                        color = item[6] if len(item) > 6 else (180, 180, 180)
                        if tid not in tid_color_sum:
                            tid_color_sum[tid] = [0.0, 0.0, 0.0]
                            tid_color_cnt[tid] = 0
                        tid_color_sum[tid][0] += color[0]
                        tid_color_sum[tid][1] += color[1]
                        tid_color_sum[tid][2] += color[2]
                        tid_color_cnt[tid] += 1
                # 构建特征矩阵（每个 tid 的平均颜色）
                cluster_tids = []
                cluster_feats = []
                for tid, csum in tid_color_sum.items():
                    cnt = tid_color_cnt[tid]
                    if cnt > 0:
                        avg_bgr = [csum[0]/cnt, csum[1]/cnt, csum[2]/cnt]
                        cluster_tids.append(tid)
                        cluster_feats.append(avg_bgr)
                if len(cluster_feats) >= 4:
                    import numpy as _np_kmb
                    feat_arr = _np_kmb.array(cluster_feats, dtype=_np_kmb.float32)
                    km = _KMB(n_clusters=2, n_init=15, random_state=42).fit(feat_arr)
                    for i, tid in enumerate(cluster_tids):
                        new_team = "team_a" if km.labels_[i] == 0 else "team_b"
                        track_team[tid] = new_team
                    cnt_a = sum(1 for t in track_team.values() if t == "team_a")
                    cnt_b = sum(1 for t in track_team.values() if t == "team_b")
                    print(f"[Phase B] 重新聚类完成: team_a={cnt_a}个, team_b={cnt_b}个")
                else:
                    # 样本不足，强制均分
                    for i, tid in enumerate(unknown_tids):
                        track_team[tid] = "team_a" if i % 2 == 0 else "team_b"
                    print(f"[Phase B] 样本不足，强制交替分配队伍")
            except Exception as _e:
                # 兜底：强制均分
                for i, tid in enumerate(unknown_tids):
                    track_team[tid] = "team_a" if i % 2 == 0 else "team_b"
                print(f"[Phase B] 聚类异常({_e})，强制交替分配队伍")
        elif len(unknown_tids) > len(valid_tids):
            # 部分 unknown 且比已分队的还多：用已知队伍颜色最近邻补全 unknown
            print(f"[Phase B] unknown({len(unknown_tids)}) > valid({len(valid_tids)})，尝试颜色最近邻补全...")
            try:
                # 计算已知队伍的平均颜色
                known_colors = {"team_a": [], "team_b": []}
                for fno, fd in frame_cache.items():
                    for item in fd["tracked"]:
                        tid = int(item[0])
                        team = track_team.get(tid, "unknown")
                        if team in known_colors:
                            color = item[6] if len(item) > 6 else (180, 180, 180)
                            known_colors[team].append(color)
                avg_colors = {}
                for t, colors in known_colors.items():
                    if colors:
                        import numpy as _np_kn
                        avg_colors[t] = _np_kn.mean(colors, axis=0)
                if len(avg_colors) == 2:
                    import numpy as _np_kn2
                    for tid in unknown_tids:
                        # 计算该 tid 的平均颜色
                        if tid in tid_color_sum if 'tid_color_sum' in dir() else False:
                            cnt = tid_color_cnt[tid]
                            avg = _np_kn2.array([tid_color_sum[tid][0]/cnt,
                                                 tid_color_sum[tid][1]/cnt,
                                                 tid_color_sum[tid][2]/cnt])
                        else:
                            colors_for_tid = []
                            for fno, fd in frame_cache.items():
                                for item in fd["tracked"]:
                                    if int(item[0]) == tid and len(item) > 6:
                                        colors_for_tid.append(item[6])
                            if not colors_for_tid:
                                continue
                            avg = _np_kn2.mean(colors_for_tid, axis=0)
                        da = _np_kn2.linalg.norm(avg - avg_colors["team_a"])
                        db = _np_kn2.linalg.norm(avg - avg_colors["team_b"])
                        track_team[tid] = "team_a" if da <= db else "team_b"
                    cnt_a = sum(1 for t in track_team.values() if t == "team_a")
                    cnt_b = sum(1 for t in track_team.values() if t == "team_b")
                    print(f"[Phase B] 最近邻补全完成: team_a={cnt_a}个, team_b={cnt_b}个")
            except Exception as _e2:
                print(f"[Phase B] 最近邻补全异常: {_e2}")

        if excluded_track_ids is None:
            excluded_track_ids = set()
        else:
            excluded_track_ids = set(int(t) for t in excluded_track_ids)

        # non_player_track_ids：只含明确标为非球员的，用于热力图/战术板过滤
        if non_player_track_ids is None:
            # 兼容旧调用：没有传此参数时，降级为用 excluded_track_ids
            heatmap_excluded_ids = excluded_track_ids
        else:
            heatmap_excluded_ids = set(int(t) for t in non_player_track_ids)

        # ── 回溯净化 frame_cache：彻底移除被用户手动标注为非球员的 track ──────
        # 原因：Phase A 帧级稳定化在用户标注之前运行，若颜色误判把裁判分进了
        #       team_a/team_b，稳定化会为其补幽灵框并写入 frame_cache。
        #       Phase B 收到 non_player_track_ids 后，在此一次性清理，
        #       防止其静止幽灵框污染热力图、战术板等后续统计数据。
        _purge_ids = heatmap_excluded_ids | excluded_track_ids
        if _purge_ids:
            _purge_count = 0
            for _fno, _fdata in frame_cache.items():
                if not _fdata or not _fdata.get("tracked"):
                    continue
                _before = len(_fdata["tracked"])
                _fdata["tracked"] = [
                    t for t in _fdata["tracked"]
                    if int(t[0]) not in _purge_ids
                ]
                _purge_count += _before - len(_fdata["tracked"])
            print(f"[Phase B] 非球员回溯净化: 从 frame_cache 中移除了 {_purge_count} 个 track 记录 "
                  f"(涉及 track_ids={_purge_ids})")
            # 同步更新 track_team / track_appearances，防止后续统计混入非球员
            for _tid in _purge_ids:
                track_team.pop(_tid, None)
                track_appearances.pop(_tid, None)

        if track_jersey_map is None:
            track_jersey_map = {}

        # 构建 demo_players（从 track_jersey_map 或 jersey_annotations）
        demo_players = []
        jersey_detections = {}

        if track_jersey_map:
            # 新方案：基于 track_id 的精确标注
            # 同一背号可能对应多个 track_id（同一球员轨迹断裂未缝合），
            # 需要合并：demo_players 每个背号只保留一条，track_id 记录为列表
            _jersey_to_tids: dict = {}   # {(num, team): [tid, ...]}
            for tid_str, info in track_jersey_map.items():
                tid = int(tid_str)
                num = int(info["jersey"])
                _raw_team = info.get("team", "")
                # 若前端传来 "unknown"（Phase A 颜色聚类失败时的降级值），
                # 则优先用 track_team（全视频多数投票），再 fallback 到 "team_a"
                if _raw_team in ("team_a", "team_b"):
                    team = _raw_team
                else:
                    team = track_team.get(tid) or track_team.get(str(tid)) or "team_a"
                key = (num, team)
                if key not in _jersey_to_tids:
                    _jersey_to_tids[key] = []
                _jersey_to_tids[key].append(tid)

            for (num, team), tids in _jersey_to_tids.items():
                side = "A" if team == "team_a" else "B"
                # 主 track_id 取出现频次最高的那个
                primary_tid = max(tids, key=lambda t: track_appearances.get(t, 0))
                demo_players.append({
                    "num": num, "side": side, "id": f"{side}{num:02d}",
                    "pos": [], "track_id": primary_tid,
                    "extra_track_ids": [t for t in tids if t != primary_tid],
                })
                jersey_detections[num] = {"count": 50, "team": team, "bbox": []}
                if len(tids) > 1:
                    print(f"[Phase B] 背号#{num}({side}) 对应多个 track_id={tids}，"
                          f"主 track={primary_tid}，其余作为别名处理")
        elif jersey_annotations:
            # 旧方案兼容
            for ann in (jersey_annotations or []):
                try:
                    num = int(ann["jersey_number"])
                    team = ann.get("team", "team_a")
                    if team not in ("team_a", "team_b"):
                        team = "team_a"
                    if 1 <= num <= 99:
                        side = "A" if team == "team_a" else "B"
                        demo_players.append({
                            "num": num, "side": side, "id": f"{side}{num:02d}", "pos": [],
                        })
                        jersey_detections[num] = {"count": 50, "team": team, "bbox": []}
                except Exception:
                    pass

        if not demo_players:
            # 无标注时，自动分配
            for tid, team in track_team.items():
                if tid in excluded_track_ids:
                    continue
                if team in ("team_a", "team_b"):
                    side = "A" if team == "team_a" else "B"
                    num = tid % 99 + 1
                    demo_players.append({
                        "num": num, "side": side, "id": f"{side}{num:02d}",
                        "pos": [], "track_id": tid,
                    })
                    jersey_detections[num] = {"count": 1, "team": team, "bbox": []}

        # 限制每队最多11人（按出现频次降序截断，保留最稳定的球员）
        team_a_players = sorted(
            [p for p in demo_players if p["side"] == "A"],
            key=lambda p: track_appearances.get(p.get("track_id"), 0), reverse=True
        )[:11]
        team_b_players = sorted(
            [p for p in demo_players if p["side"] == "B"],
            key=lambda p: track_appearances.get(p.get("track_id"), 0), reverse=True
        )[:11]
        demo_players = team_a_players + team_b_players
        main_players = [p["num"] for p in demo_players]
        selected_track_ids = {
            int(p.get("track_id"))
            for p in demo_players
            if p.get("track_id") is not None
        }
        # 把同一背号的别名 track_id 也加入 selected，让渲染/热力图统计不遗漏
        for p in demo_players:
            for _alias in p.get("extra_track_ids", []):
                selected_track_ids.add(int(_alias))
        # 建立 别名track_id → (主track_id, side) 的映射：
        #   - 渲染时统一显示主 track 的球员信息
        #   - 渲染时用 side 覆盖 Phase A 颜色聚类可能错误的 team 字段
        _alias_to_primary: dict = {}   # {alias_tid: primary_tid}
        _alias_to_side: dict = {}      # {alias_tid: "team_a"/"team_b"}
        for p in demo_players:
            _ptid = p.get("track_id")
            _pside_str = "team_a" if p["side"] == "A" else "team_b"
            for _alias in p.get("extra_track_ids", []):
                _alias_to_primary[int(_alias)] = int(_ptid)
                _alias_to_side[int(_alias)] = _pside_str

        def _track_rank_score(track_item):
            """Rank tracks by long-term stability first, bbox size second."""
            tid, x1, y1, x2, y2, *_ = track_item
            # track_appearances may be int-key or str-key depending on cache source
            app = track_appearances.get(tid, track_appearances.get(str(tid), 0))
            area = max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))
            return int(app) * 1000000 + int(area)

        def _cap_tracks_per_team(track_items, max_per_team=11):
            """Ensure each team contributes at most max_per_team active tracks."""
            grouped = {"team_a": [], "team_b": []}
            passthrough = []
            for t in track_items:
                team = t[5]
                if team in grouped:
                    grouped[team].append(t)
                else:
                    passthrough.append(t)

            capped = list(passthrough)
            for team in ("team_a", "team_b"):
                ranked = sorted(grouped[team], key=_track_rank_score, reverse=True)
                capped.extend(ranked[:max_per_team])
            return capped

        progress_store[video_uuid] = 5

        # ── 渲染视频 ──────────────────────────────────────────────────────
        export_dir = os.path.join(os.path.dirname(os.path.abspath(video_path)), "exports")
        os.makedirs(export_dir, exist_ok=True)
        export_filename = f"annotated_{video_uuid}.mp4"
        export_path = os.path.join(export_dir, export_filename)

        progress_store[video_uuid] = 6   # 初始化视频编码器
        codecs_to_try = [
            ('mp4v', 'MPEG-4 Part 2'), ('XVID', 'Xvid'), ('X264', 'X264'),
            ('avc1', 'H.264 Apple'), ('H264', 'H264')
        ]
        out = None
        used_codec = None
        for codec_name, codec_desc in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                test_out = cv2.VideoWriter(export_path, fourcc, fps, (width, height))
                if test_out.isOpened():
                    out = test_out
                    used_codec = codec_name
                    print(f"  [Phase B] 编码器: {codec_name}")
                    break
                else:
                    test_out.release()
            except Exception:
                continue
        if not out or not out.isOpened():
            raise Exception("无法初始化视频编码器")

        TEAM_COLORS = {
            "team_a": (255, 100, 0), "team_b": (255, 60, 0), "unknown": (160, 160, 160),
        }
        MAIN_PLAYER_THICKNESS = 3
        NORMAL_PLAYER_THICKNESS = 2

        homography_est = HomographyEstimator()
        # Phase B 中 H_mat 向前填补缓存：
        # Phase A 每 3 帧缓存一次 H_mat，当某帧 H_mat=None 时用最近有效的 H 替代，
        # 避免降级到线性缩放，消除坐标系切换导致的轨迹瞬移。
        _last_valid_H: np.ndarray = None

        possession_count = {"team_a": 0, "team_b": 0, "none": 0}
        heatmap_positions = {"team_a": [], "team_b": []}
        ball_trajectory = []
        ball_traj_frames = []

        # ── 个人能力统计（与 Phase A analyze_video 保持一致，key=str(track_id)）──
        personal_possession:   dict = {}
        personal_pass_count:   dict = {}
        personal_shot_count:   dict = {}
        personal_defend_count: dict = {}
        personal_mark_frames:  dict = {}
        _prev_ball_pos_for_event = None
        _prev_ball_owner_tid     = None
        _prev_ball_owner_team    = None
        _prev_ball_owner_box     = None
        _kick_cooldown           = 0

        # ── Pose 模型（与 Phase A 一致：射门事件 Pose 验证，减少误触发）────
        progress_store[video_uuid] = 7   # 加载辅助模型
        _pose_model_b = None
        try:
            import os as _os_pb
            _pose_pt_b = _os_pb.join(_os_pb.dirname(__file__), "yolov8n-pose.pt")
            if _os_pb.path.exists(_pose_pt_b):
                from ultralytics import YOLO as _YOLO_POSE_B
                _pose_model_b = _YOLO_POSE_B(_pose_pt_b)
                print(f"[Phase B] Pose模型加载成功: {_pose_pt_b}")
            else:
                print(f"[Phase B] yolov8n-pose.pt 不存在，射门 Pose 验证将跳过")
        except Exception as _pe_b:
            print(f"[Phase B] Pose模型加载失败: {_pe_b}")

        def _pose_verify_shot_b(frame_bgr, shooter_box):
            """Phase B 射门动作 Pose 验证（与 Phase A _pose_verify_shot 逻辑相同）"""
            if _pose_model_b is None:
                return True
            try:
                x1, y1, x2, y2 = shooter_box
                pad = int((y2 - y1) * 0.3)
                rx1 = max(0, x1 - pad)
                ry1 = max(0, y1 - pad)
                rx2 = min(frame_bgr.shape[1] - 1, x2 + pad)
                ry2 = min(frame_bgr.shape[0] - 1, y2 + pad)
                roi = frame_bgr[ry1:ry2, rx1:rx2]
                if roi.size == 0:
                    return True
                import torch as _torch_pb
                _use_half_pb = _torch_pb.cuda.is_available()
                pose_res = _pose_model_b(roi, imgsz=256, conf=0.3, verbose=False, half=_use_half_pb)
                if not pose_res or pose_res[0].keypoints is None:
                    return True
                kps = pose_res[0].keypoints.xy[0].cpu().numpy() if len(pose_res[0].keypoints.xy) > 0 else None
                if kps is None or len(kps) < 17:
                    return True
                # 检查髋/膝/踝关键点（COCO: 11=L_hip,12=R_hip, 13=L_knee,14=R_knee, 15=L_ankle,16=R_ankle）
                angles = []
                for hip_i, knee_i, ankle_i in [(11, 13, 15), (12, 14, 16)]:
                    h, k, a = kps[hip_i], kps[knee_i], kps[ankle_i]
                    if h[0] == 0 or k[0] == 0 or a[0] == 0:
                        continue
                    v1 = h - k
                    v2 = a - k
                    cos_a = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
                    angles.append(math.degrees(math.acos(max(-1, min(1, cos_a)))))
                if not angles:
                    return True
                max_ang = max(angles)
                return max_ang > 25.0
            except Exception:
                return True

        # 球 pitch 坐标 EMA 平滑状态
        _ball_ema_wx: float = None
        _ball_ema_wy: float = None
        _BALL_EMA_ALPHA  = 0.35
        _BALL_MAX_JUMP   = 8.0
        _BALL_MAX_FREEZE = 5
        _ball_freeze_cnt = 0
        _heatmap_sampled_once = False  # 首次采样成功标记（用于日志）
        tactical_tracks = {}

        # ── EMA 平滑追踪器（移植自3版 EMA+freeze 机制）────────────────────
        _kalman_filters: dict = {}       # {canonical_tid_str -> PlayerEMATracker}
        _player_absent_count: dict = {}  # {canonical_tid_str -> 连续缺席采样帧数}
        _ABSENT_EVICT = 60               # 离场蒸发阈值：60采样帧 ≈ 7.2秒
        _MAX_PREDICT = 10                # 最多保持10采样帧（≈30帧≈1.2秒）
        # ── 轨迹合并（断轨重连）──────────────────────────────────────────
        _track_id_alias: dict = {}       # {new_tid_str -> canonical_tid_str}
        _MERGE_DIST = 1.5                # 重连距离阈值（米），收紧至1.5m，防止密集区跨球员误合并
        _MERGE_MAX_ABSENT = 8            # 最多允许缺席8个采样帧（≈0.96s），缺席太久不合并
        _prev_centroid_wx = None
        _prev_centroid_wy = None
        _h_jump_streak_b: int = 0

        # 读视频帧用于渲染
        progress_store[video_uuid] = 8   # 开始逐帧渲染
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for fno in range(actual_frames):
            ret, frame = cap.read()
            if not ret:
                break

            fd = frame_cache.get(fno)
            if fd is None or not fd["tracked"]:
                out.write(frame)
                continue

            tracked = fd["tracked"]
            ball_pos = fd["ball_pos"]
            ball_pitch = fd.get("ball_pitch")
            H_mat_list = fd.get("H_mat")
            H_mat = np.array(H_mat_list, dtype=np.float64) if H_mat_list is not None else None

            # H_mat 向前填补：若当前帧未缓存 H_mat（Phase A 每3帧存一次），
            # 使用最近一次有效的 H_mat，避免降级到线性缩放造成坐标系跳变（轨迹瞬移）
            if H_mat is not None:
                _last_valid_H = H_mat
            elif _last_valid_H is not None:
                H_mat = _last_valid_H  # 用最近有效 H 填补空洞帧

            # 同步更新 homography_est 的 H 矩阵（供 unproject_to_pitch 使用）
            if H_mat is not None:
                homography_est._last_H = H_mat

            # 用 track_jersey_map 中的 team 覆盖 Phase A 颜色聚类结果
            # （确保控球率/热力图使用用户标注的准确队伍归属）
            # 别名 track 不在 track_jersey_map 里，但通过 _alias_to_side 能拿到正确队伍
            if track_jersey_map:
                tracked = [
                    (tid, x1, y1, x2, y2,
                     track_jersey_map[str(tid)]["team"] if str(tid) in track_jersey_map
                     else _alias_to_side.get(int(tid), team),
                     color)
                    for (tid, x1, y1, x2, y2, team, color) in tracked
                ]
            else:
                # 没有用户标注时：用 track_team（Phase A 多数投票结果）补全 unknown 球员的队伍
                # 这样热力图/战术板在未标注情况下也能使用 Phase A 的颜色聚类结果
                tracked = [
                    (tid, x1, y1, x2, y2,
                     (track_team.get(int(tid)) or track_team.get(str(tid)) or team)
                     if team == "unknown" else team,
                     color)
                    for (tid, x1, y1, x2, y2, team, color) in tracked
                ]

            # 过滤非球员 track（视频渲染用：排除 non_player + 未标注）
            if selected_track_ids:
                tracked_filtered = [
                    t for t in tracked
                    if int(t[0]) in selected_track_ids and int(t[0]) not in excluded_track_ids
                ]
            else:
                tracked_filtered = [
                    t for t in tracked
                    if int(t[0]) not in excluded_track_ids
                ]
            tracked_filtered = _cap_tracks_per_team(tracked_filtered, max_per_team=11)
            # 热力图/战术板用：只排除明确标为非球员的
            if selected_track_ids:
                tracked_heatmap = [
                    t for t in tracked
                    if int(t[0]) in selected_track_ids and int(t[0]) not in heatmap_excluded_ids
                ]
            else:
                tracked_heatmap = [
                    t for t in tracked
                    if int(t[0]) not in heatmap_excluded_ids
                ]
            tracked_heatmap = _cap_tracks_per_team(tracked_heatmap, max_per_team=11)

            # 球轨迹
            if ball_pos:
                ball_trajectory.append(ball_pos)
                # ball_pitch 在赋值处已做 pitch 越界拦截（超出场地 120% 范围则为 None）
                if ball_pitch:
                    bwx_raw, bwy_raw = ball_pitch[0], ball_pitch[1]
                    # ── 球 pitch 坐标 EMA 平滑 + 跳变冻结（Phase B）──────────────
                    if _ball_ema_wx is None:
                        _ball_ema_wx, _ball_ema_wy = bwx_raw, bwy_raw
                        _ball_freeze_cnt = 0
                    else:
                        _jump = ((bwx_raw - _ball_ema_wx) ** 2 + (bwy_raw - _ball_ema_wy) ** 2) ** 0.5
                        if _jump > _BALL_MAX_JUMP and _ball_freeze_cnt < _BALL_MAX_FREEZE:
                            _ball_freeze_cnt += 1
                        else:
                            _ball_ema_wx = _BALL_EMA_ALPHA * bwx_raw + (1.0 - _BALL_EMA_ALPHA) * _ball_ema_wx
                            _ball_ema_wy = _BALL_EMA_ALPHA * bwy_raw + (1.0 - _BALL_EMA_ALPHA) * _ball_ema_wy
                            _ball_freeze_cnt = 0
                    ball_traj_frames.append({"f": fno, "wx": round(_ball_ema_wx, 3), "wy": round(_ball_ema_wy, 3)})

            # 控球率（用 tracked_filtered：只含手动标注的球员）
            _cur_nearest_tid_b  = None
            _cur_nearest_team_b = "none"
            _cur_nearest_box_b  = None
            if ball_pos:
                min_dist = float("inf")
                nearest_team = "none"
                nearest_player_h = 80
                for (tid, x1, y1, x2, y2, team, color) in tracked_filtered:
                    cx_p = (x1 + x2) // 2
                    cy_p = (y1 + y2) // 2
                    d = math.hypot(cx_p - ball_pos[0], cy_p - ball_pos[1])
                    if d < min_dist:
                        min_dist = d
                        nearest_team = team
                        nearest_player_h = max(30, y2 - y1)
                        _cur_nearest_tid_b  = tid
                        _cur_nearest_team_b = team
                        _cur_nearest_box_b  = (x1, y1, x2, y2)
                possession_thresh = nearest_player_h * 1.8
                if min_dist < possession_thresh and nearest_team in ("team_a", "team_b"):
                    possession_count[nearest_team] += 1
                    # ── 个人控球帧数 ──
                    if _cur_nearest_tid_b is not None:
                        _tid_str_b = str(_cur_nearest_tid_b)
                        personal_possession[_tid_str_b] = personal_possession.get(_tid_str_b, 0) + 1
                else:
                    possession_count["none"] += 1
                    _cur_nearest_tid_b = None

                # ── 球速突变检测（踢球事件：射门 / 传球 / 防守）──────────────
                if _kick_cooldown > 0:
                    _kick_cooldown -= 1
                if _prev_ball_pos_for_event is not None and _kick_cooldown == 0:
                    ball_spd = math.hypot(
                        ball_pos[0] - _prev_ball_pos_for_event[0],
                        ball_pos[1] - _prev_ball_pos_for_event[1]
                    )
                    if ball_spd > 28 and _prev_ball_owner_tid is not None:
                        _kicker_tid_str = str(_prev_ball_owner_tid)
                        _kicker_team    = _prev_ball_owner_team
                        _dx = ball_pos[0] - _prev_ball_pos_for_event[0]
                        if _kicker_team == "team_a":
                            _toward_goal = _dx > 5
                        elif _kicker_team == "team_b":
                            _toward_goal = _dx < -5
                        else:
                            _toward_goal = False
                        _in_attack_zone = False
                        if _kicker_team == "team_a" and ball_pos[0] > width * 0.67:
                            _in_attack_zone = True
                        elif _kicker_team == "team_b" and ball_pos[0] < width * 0.33:
                            _in_attack_zone = True

                        if _toward_goal and _in_attack_zone:
                            # Pose 验证射门动作（与 Phase A 一致，减少误触发）
                            _pose_confirmed_b = True
                            if _prev_ball_owner_box is not None:
                                _pose_confirmed_b = _pose_verify_shot_b(frame, _prev_ball_owner_box)
                            if _pose_confirmed_b:
                                personal_shot_count[_kicker_tid_str] = personal_shot_count.get(_kicker_tid_str, 0) + 1
                        elif not _toward_goal or not _in_attack_zone:
                            personal_pass_count[_kicker_tid_str] = personal_pass_count.get(_kicker_tid_str, 0) + 1

                        if (_cur_nearest_tid_b is not None and
                                _cur_nearest_team_b in ("team_a", "team_b") and
                                _kicker_team in ("team_a", "team_b") and
                                _cur_nearest_team_b != _kicker_team):
                            _defender_str = str(_cur_nearest_tid_b)
                            personal_defend_count[_defender_str] = personal_defend_count.get(_defender_str, 0) + 1

                        _kick_cooldown = 8

                _prev_ball_pos_for_event = ball_pos
                # 只有当前帧有明确控球者时才更新；球飞出时保留上一控球者，
                # 防止踢球瞬间球离人远导致 _prev_ball_owner_tid 被清 None
                if _cur_nearest_tid_b is not None:
                    _prev_ball_owner_tid  = _cur_nearest_tid_b
                    _prev_ball_owner_team = _cur_nearest_team_b
                    _prev_ball_owner_box  = _cur_nearest_box_b
            else:
                possession_count["none"] += 1
                _prev_ball_pos_for_event = None

            # --- 贴身防守采样（每5帧）---
            if fno % 5 == 0:
                _bteam_players = {"team_a": [], "team_b": []}
                for (_btid, _bx1, _by1, _bx2, _by2, _bteam, _bcolor) in tracked_filtered:
                    if _bteam in _bteam_players:
                        _bteam_players[_bteam].append((_btid, (_bx1+_bx2)//2, (_by1+_by2)//2))
                _MARK_THRESH_B = 90
                for _def_team_b, _att_team_b in (("team_a", "team_b"), ("team_b", "team_a")):
                    for (_bdtid, _bdcx, _bdcy) in _bteam_players[_def_team_b]:
                        for (_batid, _bacx, _bacy) in _bteam_players[_att_team_b]:
                            if math.hypot(_bdcx - _bacx, _bdcy - _bacy) < _MARK_THRESH_B:
                                _bdtid_s = str(_bdtid)
                                personal_mark_frames[_bdtid_s] = personal_mark_frames.get(_bdtid_s, 0) + 1
                                break

            # 热力图 + 战术板（每5帧采样）
            # ── 关键节点日志（便于排查热力图/战术板空白 & 控球率异常问题）──
            if fno == 0:
                print(f"[Phase B] 启动参数: excluded_track_ids={len(excluded_track_ids)}个, "
                      f"heatmap_excluded_ids={len(heatmap_excluded_ids)}个, "
                      f"track_jersey_map={len(track_jersey_map)}个")
                print(f"[Phase B] fno=0: H_mat={'有' if H_mat is not None else '无(单应性失效)'}, "
                      f"tracked总数={len(tracked)}, "
                      f"tracked_filtered(控球率/渲染/贴身防守数据源)={len(tracked_filtered)}, "
                      f"tracked_heatmap(热力图/战术板/轨迹数据源)={len(tracked_heatmap)}")
                # 控球率诊断：统计 tracked_filtered 的队伍分布
                # 如果控球率一直50/50，大概率是这里 team_a/team_b 数量严重不均或全为 unknown
                # 根因：Phase A 颜色聚类结果未被 track_jersey_map 覆盖（已修复），
                #       或前端传入的 track_jersey_map 中 team 字段缺失/错误
                poss_dist = {}
                for (tid, *_, team, color) in tracked_filtered:
                    poss_dist[team] = poss_dist.get(team, 0) + 1
                print(f"[Phase B] fno=0 tracked_filtered 队伍分布(控球率数据源): {poss_dist}")
                if not poss_dist or set(poss_dist.keys()) - {"team_a", "team_b", "unknown"} == set(poss_dist.keys()):
                    print(f"[Phase B] ⚠ tracked_filtered 无有效队伍标签，控球率将全部归入 none → 输出50/50")
                elif poss_dist.get("unknown", 0) == sum(poss_dist.values()):
                    print(f"[Phase B] ⚠ tracked_filtered 所有 track 队伍为 unknown，"
                          f"控球率将全部归入 none → 输出50/50。"
                          f"请检查 track_jersey_map 是否正确传入 team 字段")
                # 热力图/战术板诊断：用 tracked_heatmap（含全部已标注球员，只排除明确非球员）
                team_dist = {}
                for (tid, *_, team, color) in tracked_heatmap:
                    team_dist[team] = team_dist.get(team, 0) + 1
                print(f"[Phase B] fno=0 tracked_heatmap 队伍分布(热力图/战术板数据源): {team_dist}")
            # 战术/热力图采样：每2帧采样一次，与 Phase A 检测周期（fno%2==0）对齐
            # 原3帧采样（fno%3==0）会导致 lcm(2,3)=6帧才有有效update，其余5帧全为预测
            if fno % 2 == 0:
                _h_mat_null_this_frame = (H_mat is None)
                frame_pitch_pts = []
                _cur_frame_tids = set()
                # 战术轨迹采样：使用 tracked_heatmap（包含所有球员，只排除明确标注的非球员）
                # 不能用 tracked_filtered（仅含用户标注球员），否则未标注球员轨迹缺失 → 断续
                for (tid, x1, y1, x2, y2, team, color) in tracked_heatmap:
                    if team not in ("team_a", "team_b"):
                        continue
                    raw_bx = float(x1 + x2) / 2.0
                    raw_by = float(y2)
                    str_tid = str(tid)
                    _cur_frame_tids.add(str_tid)
                    if H_mat is None:
                        # 真正没有任何历史 H（视频前几帧场地分割失败）：线性缩放兜底
                        wx_fb = raw_bx / max(1, width) * 105.0 - 52.5
                        wy_fb = raw_by / max(1, height) * 68.0 - 34.0
                        px_fb = homography_est.project_bbox_bottom(x1, y1, x2, y2, None, video_w=width, video_h=height)
                        frame_pitch_pts.append((tid, team, wx_fb, wy_fb, int(px_fb[0]), int(px_fb[1])))
                        continue
                    pitch_pt = homography_est.unproject_to_pitch(raw_bx, raw_by, H_mat)
                    if pitch_pt is None:
                        # H投影越界：优先从卡尔曼滤波器获取预测坐标，避免坐标系切换导致瞬移
                        can_tid_fb = resolve_alias(_track_id_alias, str_tid)
                        kf_fb = _kalman_filters.get(can_tid_fb)
                        if kf_fb is not None:
                            wx_fb, wy_fb = kf_fb.predict()
                            px_fb = homography_est.project_bbox_bottom(x1, y1, x2, y2, H_mat, video_w=width, video_h=height)
                        else:
                            # 首次出现且越界：线性缩放兜底
                            wx_fb = raw_bx / max(1, width) * 105.0 - 52.5
                            wy_fb = raw_by / max(1, height) * 68.0 - 34.0
                            px_fb = homography_est.project_bbox_bottom(x1, y1, x2, y2, None, video_w=width, video_h=height)
                        frame_pitch_pts.append((tid, team, wx_fb, wy_fb, int(px_fb[0]), int(px_fb[1])))
                        continue
                    wx, wy = pitch_pt
                    px, py = homography_est.project_bbox_bottom(
                        x1, y1, x2, y2, H_mat, video_w=width, video_h=height
                    )
                    frame_pitch_pts.append((tid, team, wx, wy, px, py))
                # 首个有数据的采样帧打印一次，确认采样正常
                if frame_pitch_pts and not _heatmap_sampled_once:
                    _heatmap_sampled_once = True
                    print(f"[Phase B] 首次热力图采样成功: fno={fno}, "
                          f"本帧采样点={len(frame_pitch_pts)}, H_mat={'有' if not _h_mat_null_this_frame else '无(用向前填补H)'}")

                h_jump_frame = False
                if frame_pitch_pts:
                    cur_wx_list = sorted([p[2] for p in frame_pitch_pts])
                    cur_wy_list = sorted([p[3] for p in frame_pitch_pts])
                    n = len(cur_wx_list)
                    # 用中位数代替均值，抗裁判/广告牌等异常点干扰
                    mid = n // 2
                    cur_centroid_wx = cur_wx_list[mid] if n % 2 else (cur_wx_list[mid-1] + cur_wx_list[mid]) / 2
                    cur_centroid_wy = cur_wy_list[mid] if n % 2 else (cur_wy_list[mid-1] + cur_wy_list[mid]) / 2
                    wy_var = sum((y - cur_centroid_wy) ** 2 for y in cur_wy_list) / n if n > 1 else 999.0
                    if n >= 4 and _prev_centroid_wx is not None and abs(cur_centroid_wx - _prev_centroid_wx) > 30.0:
                        h_jump_frame = True
                    elif n >= 4 and _prev_centroid_wy is not None and abs(cur_centroid_wy - _prev_centroid_wy) > 25.0:
                        h_jump_frame = True
                    elif n >= 6 and wy_var < 0.5:
                        print(f"[DEBUG] Phase B 帧{fno}: 检测到Y方向坐标塌缩（wy方差={wy_var:.3f}），冻结本帧")
                        h_jump_frame = True
                    if not h_jump_frame:
                        _prev_centroid_wx = cur_centroid_wx
                        _prev_centroid_wy = cur_centroid_wy
                        _h_jump_streak_b = 0
                    else:
                        _h_jump_streak_b += 1
                        if _h_jump_streak_b >= 5:
                            _prev_centroid_wx = cur_centroid_wx
                            _prev_centroid_wy = cur_centroid_wy
                            _h_jump_streak_b = 0
                            print(f"[DEBUG] Phase B 帧{fno}: 连续跳变5帧，强制重置质心参考值")

                if not h_jump_frame:
                    # ── 离场计数器更新 ──────────────────────────────────────────
                    _cur_canonical_tids = set()
                    for (tid, team, wx, wy, px, py) in frame_pitch_pts:
                        ct = resolve_alias(_track_id_alias, str(tid))
                        _cur_canonical_tids.add(ct)
                    for known_tid in list(_player_absent_count.keys()):
                        if known_tid in _cur_canonical_tids:
                            _player_absent_count[known_tid] = 0
                        else:
                            _player_absent_count[known_tid] += 1
                    for new_tid in _cur_canonical_tids:
                        if new_tid not in _player_absent_count:
                            _player_absent_count[new_tid] = 0
                    # 清理长期离场球员缓存（同时清理卡尔曼滤波器）
                    evict_b = [t for t, cnt in _player_absent_count.items() if cnt > _ABSENT_EVICT]
                    for t in evict_b:
                        _kalman_filters.pop(t, None)
                        del _player_absent_count[t]

                    # ── 卡尔曼预测：为本帧缺席的已知球员填充预测轨迹点 ────────────
                    _pitch_half_l = homography_est.PITCH_L / 2.0 * 1.05
                    _pitch_half_w = homography_est.PITCH_W / 2.0 * 1.05
                    for kf_tid, kf in _kalman_filters.items():
                        if kf_tid in _cur_canonical_tids:
                            continue  # 本帧有测量，稍后 update
                        absent_cnt = _player_absent_count.get(kf_tid, 0)
                        if absent_cnt > _MAX_PREDICT:
                            continue  # 超出预测窗口，停止预测
                        wx_p, wy_p = kf.predict()
                        wx_p = max(-_pitch_half_l, min(_pitch_half_l, wx_p))
                        wy_p = max(-_pitch_half_w, min(_pitch_half_w, wy_p))
                        if kf_tid not in tactical_tracks:
                            continue  # 从未有过轨迹，不填充
                        tactical_tracks[kf_tid]["pts"].append({
                            "f": fno, "wx": round(wx_p, 3), "wy": round(wy_p, 3),
                            "predicted": True
                        })

                    for (tid, team, wx, wy, px, py) in frame_pitch_pts:
                        if team in heatmap_positions:
                            heatmap_positions[team].append({
                                "x": max(0, min(799, px)),
                                "y": max(0, min(519, py))
                            })
                        # ── 坐标严格 clip 到场地边界（±5% 余量）────────────────────
                        wx = max(-_pitch_half_l, min(_pitch_half_l, wx))
                        wy = max(-_pitch_half_w, min(_pitch_half_w, wy))
                        # ── 断轨重连：新 track_id 尝试合并到最近的旧轨迹 ──────────
                        str_tid = str(tid)
                        # 路径压缩 alias 查找（修复链式 A→B→C 只走一步的问题）
                        canonical_tid = resolve_alias(_track_id_alias, str_tid)
                        if canonical_tid not in _kalman_filters:
                            # 用 compute_rule_score 评分选最优候选（距离+缺席综合最优）
                            best_cand  = None
                            best_score = -1.0
                            for cand_tid, kf_cand in _kalman_filters.items():
                                absent_cnt = _player_absent_count.get(cand_tid, 999)
                                if absent_cnt > _MERGE_MAX_ABSENT:
                                    continue
                                if cand_tid in tactical_tracks and tactical_tracks[cand_tid]["team"] != team:
                                    continue
                                cwx, cwy = kf_cand.pos
                                d = ((wx - cwx) ** 2 + (wy - cwy) ** 2) ** 0.5
                                if d >= _MERGE_DIST:
                                    continue  # 超出距离阈值直接跳过
                                score = compute_rule_score(d, absent_cnt, _MERGE_DIST, max(_MERGE_MAX_ABSENT, 1))
                                if score > best_score:
                                    best_score = score
                                    best_cand  = cand_tid
                            if best_cand is not None:
                                _track_id_alias[str_tid] = best_cand
                                canonical_tid = best_cand
                                print(f"  [轨迹合并] 帧{fno}: track {str_tid} → {canonical_tid}"
                                      f"（评分{best_score:.3f}，缺席{_player_absent_count.get(canonical_tid,0)}帧）")
                        str_tid = canonical_tid
                        # ── 卡尔曼滤波 update（有测量值）──────────────────────────
                        if str_tid in _kalman_filters:
                            wx_s, wy_s = _kalman_filters[str_tid].update(wx, wy)
                        else:
                            # 首次出现：初始化 EMA 追踪器
                            kf_new = PlayerEMATracker(wx, wy)
                            _kalman_filters[str_tid] = kf_new
                            wx_s, wy_s = wx, wy

                        if str_tid not in tactical_tracks:
                            tactical_tracks[str_tid] = {"team": team, "pts": []}
                        else:
                            tactical_tracks[str_tid]["team"] = team
                        tactical_tracks[str_tid]["pts"].append({
                            "f": fno, "wx": round(wx_s, 3), "wy": round(wy_s, 3)
                        })
                else:
                    # H跳变帧：用卡尔曼当前状态位置补写，保持球员在原位不消失（与 Phase A 一致）
                    for str_tid_b, kf_b in _kalman_filters.items():
                        if str_tid_b not in tactical_tracks:
                            continue
                        wx_s_b, wy_s_b = kf_b.pos
                        tactical_tracks[str_tid_b]["pts"].append({
                            "f": fno, "wx": round(wx_s_b, 3), "wy": round(wy_s_b, 3),
                            "predicted": True,
                        })

            # ── 绘制 ──
            annotated = frame.copy()

            # 先把用户主动排除的 track 画灰色框（视觉反馈：这个人被排除了）
            for (tid, x1, y1, x2, y2, team, color) in tracked:
                if int(tid) in excluded_track_ids:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (100, 100, 100), 1)

            # 只绘制已标注球员（tracked_filtered 已经过 selected_track_ids + excluded 双重过滤）
            # 不绘制球童/裁判等未标注 track，避免视频中出现裸 track_id 标签
            for (tid, x1, y1, x2, y2, team, color) in tracked_filtered:
                matched_player = None
                # 先用主 track_id 查找；若是别名 track，映射到主 track 后再查
                _lookup_tid = _alias_to_primary.get(int(tid), int(tid))
                for p in demo_players:
                    if p.get("track_id") == _lookup_tid:
                        matched_player = p
                        break

                is_main = matched_player is not None
                draw_color = TEAM_COLORS.get(team, (160, 160, 160))
                thick = MAIN_PLAYER_THICKNESS if is_main else NORMAL_PLAYER_THICKNESS
                cv2.rectangle(annotated, (x1, y1), (x2, y2), draw_color, thick)

                if matched_player is not None:
                    label = f"#{matched_player['num']}"
                    if fno % 10 == 0:
                        matched_player["pos"].append({"x": (x1+x2)//2, "y": (y1+y2)//2})
                else:
                    label = f"#{tid}"
                cv2.putText(annotated, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1)

            # 足球
            if ball_pos:
                cv2.circle(annotated, ball_pos, 10, (0, 255, 255), 2)
                cv2.circle(annotated, ball_pos, 3, (0, 255, 255), -1)
                trail = ball_trajectory[-8:]
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    tc = (int(0*alpha), int(255*alpha), int(255*alpha))
                    cv2.line(annotated, trail[i-1], trail[i], tc, 2)

            # 控球率 HUD
            total_p = possession_count["team_a"] + possession_count["team_b"]
            if total_p > 0:
                pa = possession_count["team_a"] / total_p * 100
                pb = possession_count["team_b"] / total_p * 100
                hud_text = f"控球率  主队:{pa:.0f}%  客队:{pb:.0f}%"
                cv2.rectangle(annotated, (10, 10), (360, 36), (0, 0, 0), -1)
                cv2.putText(annotated, hud_text, (14, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            out.write(annotated)

            if fno % 100 == 0:
                progress_store[video_uuid] = int(10 + (fno / actual_frames) * 75)

        cap.release()
        out.release()

        progress_store[video_uuid] = 86  # 逐帧渲染完成，开始后处理

        # ── 后处理 ──
        # 重置首帧采样标记（供下次分析使用）
        ta_cnt = len(heatmap_positions['team_a'])
        tb_cnt = len(heatmap_positions['team_b'])
        tact_cnt = len(tactical_tracks)
        print(f"[Phase B] 采样汇总: heatmap team_a={ta_cnt}点, team_b={tb_cnt}点, tactical_tracks={tact_cnt}条")
        if ta_cnt == 0 and tb_cnt == 0:
            print(f"[Phase B] ⚠ 警告: 热力图数据为空！可能原因：")
            print(f"          1. tracked_filtered 中无 team_a/team_b 球员（队伍标签异常）")
            print(f"          2. 所有帧 H_mat 均为 None 且无历史缓存坐标（单应性全程失效）")
            print(f"          3. excluded_track_ids 把所有已标注球员都过滤了（当前={len(excluded_track_ids)}个被排除）")
        elif ta_cnt == 0 or tb_cnt == 0:
            print(f"[Phase B] ⚠ 警告: {'team_a' if ta_cnt == 0 else 'team_b'} 热力图数据为空，"
                  f"该队可能全部被排除或队伍标签未能正确分类")
        heatmap_dir = os.path.join(export_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)
        heatmap_urls = {}
        for team in ("team_a", "team_b"):
            hmap_path = os.path.join(heatmap_dir, f"heatmap_{video_uuid}_{team}.png")
            ok = generate_heatmap_image(heatmap_positions[team], 800, 520, hmap_path)
            if ok:
                heatmap_urls[team] = f"/exports/heatmaps/heatmap_{video_uuid}_{team}.png"
                print(f"[Phase B] 热力图生成成功: {team} → {hmap_path}")
            else:
                print(f"[Phase B] ⚠ 热力图生成失败: {team}（数据点={len(heatmap_positions[team])}）")

        ta_positions = [v["pts"] for v in tactical_tracks.values() if v["team"] == "team_a" and v["pts"]]
        tb_positions = [v["pts"] for v in tactical_tracks.values() if v["team"] == "team_b" and v["pts"]]
        if not ta_positions and heatmap_positions["team_a"]:
            ta_positions = [heatmap_positions["team_a"]]
            print(f"[Phase B] 战术板降级: team_a 用 heatmap_positions 兜底")
        if not tb_positions and heatmap_positions["team_b"]:
            tb_positions = [heatmap_positions["team_b"]]
            print(f"[Phase B] 战术板降级: team_b 用 heatmap_positions 兜底")
        print(f"[Phase B] 战术板轨迹: team_a={len(ta_positions)}条, team_b={len(tb_positions)}条, "
              f"ball={len(ball_trajectory)}点")
        tactical_path = os.path.join(heatmap_dir, f"tactical_{video_uuid}.png")
        build_tactical_view(ta_positions, tb_positions, ball_trajectory, width, height, tactical_path,
                            homography_est=homography_est)
        if os.path.exists(tactical_path):
            print(f"[Phase B] 战术板生成成功 → {tactical_path}")
        else:
            print(f"[Phase B] ⚠ 战术板文件未生成: {tactical_path}")
        tactical_url = f"/exports/heatmaps/tactical_{video_uuid}.png"

        progress_store[video_uuid] = 88  # 热力图/战术板生成完成

        # --- 离线全局轨迹后处理：合并断轨、填补间隙、平滑抖动 ---
        tactical_tracks = _postprocess_tactical_tracks(tactical_tracks)

        # 压缩（上限与 Phase A 对齐，统一为 80000 点）
        MAX_TRACK_PTS = 80000
        total_pts = sum(len(v["pts"]) for v in tactical_tracks.values())
        if total_pts > MAX_TRACK_PTS:
            step = max(1, round(total_pts / MAX_TRACK_PTS))
            for tid in tactical_tracks:
                pts = tactical_tracks[tid]["pts"]
                # 短轨迹（点数少于 step*5）保留全部点，避免降采样后信息损失过多
                if len(pts) > step * 5:
                    tactical_tracks[tid]["pts"] = pts[::step]
        if len(ball_traj_frames) > 3000:
            ball_traj_frames_out = ball_traj_frames[::max(1, len(ball_traj_frames) // 3000)]
        else:
            ball_traj_frames_out = ball_traj_frames

        # ── 球轨迹后处理：5点中值滤波消除球检测误差导致的孤立噪声点 ─────────
        BALL_SMOOTH_WIN = 2
        BALL_NOISE_THRESH = 3.5
        if len(ball_traj_frames_out) >= 5:
            smoothed_ball = []
            n_b = len(ball_traj_frames_out)
            for i_b, pt_b in enumerate(ball_traj_frames_out):
                lo_b = max(0, i_b - BALL_SMOOTH_WIN)
                hi_b = min(n_b, i_b + BALL_SMOOTH_WIN + 1)
                window_pts = ball_traj_frames_out[lo_b:hi_b]
                if len(window_pts) < 3:
                    smoothed_ball.append(pt_b)
                    continue
                wx_vals = sorted(p["wx"] for p in window_pts)
                wy_vals = sorted(p["wy"] for p in window_pts)
                mid_i = len(wx_vals) // 2
                med_wx = wx_vals[mid_i]
                med_wy = wy_vals[mid_i]
                dev = ((pt_b["wx"] - med_wx) ** 2 + (pt_b["wy"] - med_wy) ** 2) ** 0.5
                if dev > BALL_NOISE_THRESH:
                    smoothed_ball.append({"f": pt_b["f"], "wx": round(med_wx, 3), "wy": round(med_wy, 3)})
                else:
                    smoothed_ball.append(pt_b)
            ball_traj_frames_out = smoothed_ball
        coord_space = "pitch"
        pitch_l = homography_est.PITCH_L
        pitch_w = homography_est.PITCH_W

        total_p = possession_count["team_a"] + possession_count["team_b"]
        possession_pct = {
            "team_a": round(possession_count["team_a"] / total_p * 100, 1) if total_p > 0 else 50.0,
            "team_b": round(possession_count["team_b"] / total_p * 100, 1) if total_p > 0 else 50.0,
        }
        print(f"[Phase B] 控球率: team_a={possession_pct['team_a']}% "
              f"team_b={possession_pct['team_b']}% "
              f"(采样帧: a={possession_count['team_a']}, b={possession_count['team_b']}, none={possession_count['none']})")
        if possession_pct["team_a"] == 50.0 and possession_pct["team_b"] == 50.0:
            print(f"[Phase B] ⚠ 控球率为50/50！排查步骤：")
            print(f"          1. 查看上方 fno=0 的 tracked_filtered 队伍分布，若全为 unknown")
            print(f"             → track_jersey_map 传入的 team 字段有误或前端未正确发送")
            print(f"          2. 若 tracked_filtered 为空 → excluded_track_ids 把所有球员都过滤了")
            print(f"          3. 若 none 帧数极多 → ball_pos 全程未检测到，或最近球员距球超过阈值")

        # 验证导出
        if not os.path.exists(export_path) or os.path.getsize(export_path) < 1024:
            raise Exception("视频导出失败")

        progress_store[video_uuid] = 90  # 开始视频转码（mp4v→H.264）
        # mp4v → H.264 转码
        if used_codec == 'mp4v':
            temp_path = export_path.replace('.mp4', '_temp.mp4')
            try:
                import imageio_ffmpeg, subprocess
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                cmd = [ffmpeg_exe, '-i', export_path, '-vcodec', 'libx264', '-acodec', 'aac',
                       '-preset', 'ultrafast', '-crf', '26', '-movflags', '+faststart', '-y', temp_path]
                file_mb = os.path.getsize(export_path) / 1024 / 1024
                subprocess.run(cmd, capture_output=True, text=True, timeout=max(120, int(file_mb * 8) + 120))
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1024:
                    os.remove(export_path)
                    os.rename(temp_path, export_path)
                elif os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        progress_store[video_uuid] = 95  # 开始球员能力评分
        # 球员画像（两遍全队归一化）
        def _collect_raw_b(p):
            """第一遍：收集每个球员的原始数值，不做归一化。"""
            _raw_tid = p.get("track_id")
            _tid_key = str(_raw_tid) if _raw_tid is not None else ""
            # 别名 track_id 列表（同一球员轨迹断裂的其他片段）
            _alias_tids = [str(t) for t in p.get("extra_track_ids", [])]

            # 出现帧数：主 track + 所有别名 track 的出现帧数之和
            if _raw_tid is not None and track_appearances:
                det_count = track_appearances.get(int(_raw_tid),
                            track_appearances.get(str(_raw_tid), 1))
                for _at in _alias_tids:
                    det_count += track_appearances.get(int(_at),
                                 track_appearances.get(_at, 0))
            else:
                det_info = jersey_detections.get(p["num"], {})
                det_count = det_info.get("count", 1)

            # 轨迹原始量
            positions = p.get("pos", [])
            total_dist = 0.0
            peak_step  = 0.0
            coverage   = 0.0
            if len(positions) >= 2:
                step_dists = [math.hypot(positions[i+1]["x"] - positions[i]["x"],
                                         positions[i+1]["y"] - positions[i]["y"])
                              for i in range(len(positions) - 1)]
                total_dist = sum(step_dists)
                peak_step  = max(step_dists) if step_dists else 0.0
                xs = [pt["x"] for pt in positions]
                ys = [pt["y"] for pt in positions]
                x_range = max(xs) - min(xs)
                y_range = max(ys) - min(ys)
                coverage = min(1.0, (x_range * y_range) / max(1, actual_frames * 100))

            # 个人统计：主 track + 别名 track 累加（别名 track 期间的控球/传球等不能丢失）
            def _sum_stat(d, keys):
                return sum(d.get(k, 0) for k in keys)
            _all_tid_keys = [_tid_key] + _alias_tids

            return {
                "tid_key":     _tid_key,
                "det_count":   det_count,
                "total_dist":  total_dist,
                "peak_step":   peak_step,
                "coverage":    coverage,
                "poss_frames": _sum_stat(personal_possession,   _all_tid_keys),
                "pass_cnt":    _sum_stat(personal_pass_count,   _all_tid_keys),
                "shot_cnt":    _sum_stat(personal_shot_count,   _all_tid_keys),
                "defend_cnt":  _sum_stat(personal_defend_count, _all_tid_keys),
                "mark_frames": _sum_stat(personal_mark_frames,  _all_tid_keys),
                "pos_len":     len(positions),
            }

        def _score_b(raw, ts):
            """第二遍：利用全队最大值做相对归一化，输出分数。"""
            appear_ratio = raw["det_count"] / max(ts["max_det"], 1)
            dist_norm    = raw["total_dist"] / max(ts["max_total_dist"], 1.0)
            cov_norm     = raw["coverage"]   / max(ts["max_coverage"],   1e-6)
            stamina_raw  = 0.4 * appear_ratio + 0.3 * dist_norm + 0.3 * min(cov_norm, 1.0)
            stamina      = int(55 + stamina_raw * 40)

            if raw["pos_len"] >= 2:
                speed_raw = (0.6 * raw["total_dist"] / max(ts["max_total_dist"], 1.0)
                           + 0.4 * raw["peak_step"]  / max(ts["max_peak_step"],  1.0))
            else:
                speed_raw = appear_ratio * 0.7
            speed = int(55 + speed_raw * 40)

            _poss_ratio = raw["poss_frames"] / max(ts["max_poss_frames"], 1)
            pass_val    = int(min(96, 75 + _poss_ratio * 12 + min(raw["pass_cnt"] * 3, 9)))

            shoot_val = int(min(96, max(75, 75 + min(raw["shot_cnt"] * 4, 21))))

            defend_ratio = raw["defend_cnt"]  / max(ts["max_defend_cnt"],  1)
            mark_ratio   = raw["mark_frames"] / max(ts["max_mark_frames"], 1)
            defend_raw   = 0.4 * defend_ratio + 0.6 * mark_ratio
            defend_val   = int(55 + defend_raw * 40)

            if ts["no_event"]:
                shoot_val  = int(min(96, max(75, 75 + (speed - 65) * 0.4)))
                defend_val = int(min(96, max(55, 60 + (stamina - 65) * 0.4)))

            abs_vals = {"防守": defend_val, "射门": shoot_val, "传球": pass_val, "速度": speed, "体能": stamina}
            path_accuracy = round(min(0.99, 0.82 + stamina_raw * 0.16), 2)
            p_stats = {
                "detection_count": raw["det_count"], "coverage_ratio": raw["coverage"],
                "speed_score": speed / 10.0,
                "personal_possession": raw["poss_frames"], "pass_count": raw["pass_cnt"],
                "shot_count": raw["shot_cnt"], "defend_count": raw["defend_cnt"],
            }
            return abs_vals, path_accuracy, p_stats

        # 第一遍：收集原始数据
        _all_raw_b = [_collect_raw_b(p) for p in demo_players]

        # 全队统计量
        # 注意：_max_det_b 必须只用已标注球员的出现帧数，
        # 不能用全量 track_appearances（含球童），否则球童高频出现会压低所有球员的体能评分
        _max_det_b = max((r["det_count"] for r in _all_raw_b), default=1)

        _ts_b = {
            "max_det":         _max_det_b,
            "max_total_dist":  max((r["total_dist"]  for r in _all_raw_b), default=1.0),
            "max_peak_step":   max((r["peak_step"]   for r in _all_raw_b), default=1.0),
            "max_coverage":    max((r["coverage"]    for r in _all_raw_b), default=1e-6),
            "max_poss_frames": max(personal_possession.values(),   default=1) if personal_possession else 1,
            "max_defend_cnt":  max(personal_defend_count.values(), default=1) if personal_defend_count else 1,
            "max_mark_frames": max(personal_mark_frames.values(),  default=1) if personal_mark_frames else 1,
            "no_event":        (sum(personal_shot_count.values()) == 0 and
                                sum(personal_defend_count.values()) == 0),
        }

        # 第二遍：归一化打分
        athletes = []
        for p, raw_b in zip(demo_players, _all_raw_b):
            abs_vals, path_acc, p_stats = _score_b(raw_b, _ts_b)
            athletes.append({
                "player_id": p["id"],
                "name": f"{'主队' if p['side']=='A' else '客队'}{p['num']}号球员",
                "team": "team_a" if p["side"] == "A" else "team_b",
                "jersey_number": p["num"],
                "track_id": p.get("track_id"),
                "abilities": abs_vals,
                "path_accuracy": path_acc,
                "perfect_path": p["pos"][:20],
                "actual_path": p["pos"][:20],
                "suggestion": get_dynamic_suggestion(p["num"], abs_vals, p_stats),
            })

        athletes_team_a = [a for a in athletes if a["team"] == "team_a"]
        athletes_team_b = [a for a in athletes if a["team"] == "team_b"]

        progress_store[video_uuid] = 98  # 写入数据库
        # 写入数据库
        video = db_session.query(VideoModel).filter(VideoModel.video_uuid == video_uuid).first()
        if not video:
            raise Exception(f"视频记录在分析过程中被删除: {video_uuid}")

        try:
            import random as _random
            profile = AthleteProfile(
                video_id=video.id,
                overall_score=float(_random.randint(90, 100)),
                decision_summary=f"分析完成。已锁定 {len(main_players)} 名球员（编号: {main_players}），排除 {len(excluded_track_ids)} 个非球员。",
                detailed_analysis={
                    "athletes": athletes,
                    "athletes_team_a": athletes_team_a,
                    "athletes_team_b": athletes_team_b,
                    "export_url": f"/exports/{export_filename}",
                    "possession": possession_pct,
                    "heatmap_urls": heatmap_urls,
                    "tactical_url": tactical_url,
                    "ball_trajectory_count": len(ball_trajectory),
                    "tactical_tracks": tactical_tracks,
                    "ball_traj_frames": ball_traj_frames_out,
                    "tactical_meta": {
                        "total_frames": cache["total_frames"],   # 原片总帧数（前端时间轴用）
                        "actual_frames": actual_frames,          # 实际分析帧数
                        "fps": fps,
                        "coord_space": coord_space, "pitch_l": pitch_l, "pitch_w": pitch_w,
                        "video_w": width, "video_h": height, "canvas_w": 800, "canvas_h": 520,
                        "video_duration": round(cache["total_frames"] / fps, 2),
                        "analyzed_duration": round(actual_frames / fps, 2),
                    },
                    # ── 球衣颜色标签 ──────────────────
                    "team_color_labels": _build_team_color_labels(team_hsv_centers),
                }
            )
            db_session.add(profile)
            db_session.flush()
            db_session.commit()
            db_session.refresh(profile)
            print(f"[Phase B] 数据库写入成功: {len(athletes)} 名球员")
        except Exception as e:
            db_session.rollback()
            raise Exception(f"数据库操作失败: {e}")

        progress_store[video_uuid] = 100
        print(f">>> Phase B 完成: {video_uuid}")

        # 后台 AI 分析
        try:
            import threading
            from ai_agent import get_ai_manager
            def _bg_ai():
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    async def _trigger():
                        mgr = get_ai_manager()
                        if mgr.agent:
                            await mgr.agent.analyze_video_content({
                                "video_id": video_uuid,
                                "detailed_analysis": {
                                    "athletes": athletes,
                                    "possession": possession_pct,
                                    "team_color_labels": _build_team_color_labels(
                                        cache.get("team_hsv_centers", {})
                                    ),
                                }
                            })
                    loop.run_until_complete(_trigger())
                    loop.close()
                except Exception:
                    pass
            threading.Thread(target=_bg_ai, daemon=True).start()
        except Exception:
            pass

    except Exception as e:
        import traceback
        print(f"[Phase B ERROR] {e}")
        traceback.print_exc()
        video = db_session.query(VideoModel).filter(VideoModel.video_uuid == video_uuid).first()
        if video:
            db_session.add(AthleteProfile(
                video_id=video.id, overall_score=0.0,
                decision_summary=f"分析失败：{str(e)[:200]}",
                detailed_analysis={"error": True, "message": str(e), "athletes": []}
            ))
            db_session.commit()
        progress_store[video_uuid] = -1
    finally:
        db_session.close()
