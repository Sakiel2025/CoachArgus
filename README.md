# CoachArgus

CoachArgus 是我们团队共同开发的 [GoalScope 足球视频分析系统](https://github.com/EVANDRAUGHT/football-game) 的改版。它沿用了团队原有的视频分析流程、前后端界面和数据管理模块，并调整了部分算法实现。

## 主要改动

- 球场标定结合球场分割和单应矩阵估计，将视频画面中的位置投影到战术坐标。
- 战术坐标处理包含球员落点计算、轨迹平滑和标定失败时的坐标回退。
- 球员检测和重识别采用足球场景模型，跟踪阶段结合位置和外观特征进行关联。

## 署名与发布状态

共享系统由 GoalScope 团队共同开发。原项目及其公开记录见 [EVANDRAUGHT/football-game](https://github.com/EVANDRAUGHT/football-game)。

本仓库发布源码。AI 对话集成、模型文件、历史数据库、视频和缓存均已移除。仓库目前未添加统一许可证。

## 本地运行

使用 Python 3.9 或 3.10。建议在虚拟环境中执行以下命令：

```shell
python -m pip install -r backend/requirements.txt
python -m uvicorn --app-dir backend main:app --host 127.0.0.1 --port 9999
```

打开 http://127.0.0.1:9999，可注册或使用游客入口。需要管理后台时，在首次启动前通过环境变量设置 `COACHARGUS_ADMIN_PASSWORD`（至少 12 个字符）。本项目按本地使用配置。

### 自备模型

- 球员检测需要兼容的 `best.pt`，放在仓库根目录。
- 专用足球检测权重文件名为 `football-ball-detection.pt`。
- 深度重识别需单独安装兼容的 Torchreid 并配置 `osnet_x075_soccernet_best.pth`。
- 球场分割需配置 `soccer_pitch_segmentation.pth`、`mean.npy` 和 `std.npy`。缺少分割模型时使用降级路径。

按各模型提供方的许可自行获取权重。
