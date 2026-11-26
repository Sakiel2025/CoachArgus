# CoachArgus

CoachArgus 是我们团队共同开发的 [GoalScope 足球视频分析系统](https://github.com/EVANDRAUGHT/football-game) 的改版。它沿用了团队原有的视频分析流程、前后端界面和数据管理模块，并调整了部分算法实现。

## 主要改动

- 球场标定结合球场分割和单应矩阵估计，将视频画面中的位置投影到战术坐标。
- 战术坐标处理包含球员落点计算、轨迹平滑和标定失败时的坐标回退。
- 球员检测和重识别采用足球场景模型，跟踪阶段结合位置和外观特征进行关联。

## 署名与发布状态

共享系统由 GoalScope 团队共同开发。原项目及其公开记录见 [EVANDRAUGHT/football-game](https://github.com/EVANDRAUGHT/football-game)。

本仓库目前仅包含项目说明。源码、模型权重以及各自的许可证和来源说明将在确认发布范围后补充。
