# 最新版手部 marker / MANO 代码（2026-10-22）

本包仅交付最新版代码、必要配置、依赖信息和说明。不包含视频、实验 NPZ、原始 C3D/RGB、模型权重、历史候选源码或运行缓存。

采用版本来自冻结目录 `/data2/wangzhengwei/capture1019_palm_thumb_stability_20261022`：左手 `guarded_axes_v3`，右手沿用 baseline。29 个管线 Python 文件与冻结采用版逐字节一致。`tools/verify_code_package.py` 是本次交付新增的只读完整性检查工具，不参与拟合。

## 代码入口与处理流程

当前入口是 `run_stability_pipeline.py`。流程为：基线稳定性测量 → `constrained_thumb.py` 约束优化 → `guard_stability.py` 三维保护回退 → 统一评估 → `gate_no_regression.py` 验收 → `select_stability.py` 选择 → 可选对比渲染与视频解码检查。

该版本固定 root、其他四指姿态、共享 beta、贴面安装、高度及腕部参数；在拇指 marker 约束的弱观测方向内减少掌根形变与拇指轴向补偿。RGB 不进入逐帧 pose/root/安装求解，仅用于离线评估；不使用深度。手部 marker 按直径 10mm 背侧半球处理，允许贴面偏心；沿用的有效中心高度 0mm 是假设，尚未实测认证。

这是已准备观测和基线之上的手部稳定性扩展；完整人体与扫描物体交互管线未在本版本更新，也不是任意新 C3D 的直接冷启动入口。六段开发片段共用算法参数，无逐案例手工补丁。`case_config.json` 中的时间仅用于选取审阅片段。

## 环境

已验证环境为 Linux / Python 3.10.20，原解释器 `/data2/DEV/soma-repair/.venv/bin/python`。依赖版本见 `requirements.txt`；`environment.json` 记录本机版本信息。这是直接依赖版本清单，不是新机器上已经验证的完整环境锁。渲染另需系统 `ffmpeg`。

原环境同时安装了两种 OpenCV 分发；清单仅保留 `opencv-python`，避免要求重复安装同名 `cv2` 提供者。洁净环境安装及跨平台兼容性本次未重新验证。

## 输入和路径要求

仅代码包不能直接完成拟合。需从原冻结目录只读提供下列输入，放在解压后的代码目录中，再创建新的运行副本：

- `baseline/`、`baseline_profile/`、`inputs_v2/`、`wrist_articulation/`、`evaluation_reference/`；
- `evaluation_finger_records.json`、`VISUAL_REVIEW_RECORD.json`：固定评估与候选哈希审阅记录，本包不附实验记录。

还需原始 capture（C3D、相机刚体及时间映射、RGB 分段包与相机内参）、合法取得的 MANO 数值资产和 marker 布局，以及最新米制外参。确切引用见 `external_paths.json`、`capture_config.json`，硬编码模型路径见 `mano_base.py`，外参路径见 `rgb_camera.py`。`source.json` 保留采用版的来源信息。

原实现限制所有运行输出和缓存位于 `/data2/wangzhengwei`；其他机器审阅源码不受此限制，执行前需要在新的代码副本中适配路径和外部资产。不要修改或重跑原冻结实验。

## 校验与运行

在任意位置可以先只读校验交付文件，不需要安装拟合依赖：

```bash
python3 -B tools/verify_code_package.py
```

补齐上述外部资产和输入后，在 `/data2/wangzhengwei` 下的解压目录执行：

```bash
/data2/DEV/soma-repair/.venv/bin/python -B create_research_copy.py --destination /data2/wangzhengwei/palm_thumb_review_run_NEW
cd /data2/wangzhengwei/palm_thumb_review_run_NEW
/data2/DEV/soma-repair/.venv/bin/python -B test_thumb_constraints.py
/data2/DEV/soma-repair/.venv/bin/python -B run_stability_pipeline.py --skip-render
```

省略 `--skip-render` 则追加原 RGB / 最新左手 MANO 两栏视频，含 marker 位置、编号与来源；默认 0.5 倍播放。重放记录只在候选哈希完全相同时允许继承；新生成的视频仍需视觉审阅。源文件中的 `verify_delivery.py` 检查原冻结交付，依赖原 `finalization.json`，本代码包应使用 `tools/verify_code_package.py`。

## 已知效果与限制

原采用版：掌根局部加速度 P95 下降约 12.57%，拇指局部角步变 P95 下降约 7.84%；世界腕根加速度保持不变，最大拇指角步变仍为 134.521°，不能宣称抽动完全解决。367 项非图像硬检查通过；图像代理存在 17 项严格回退，经抽帧视觉审阅接受。原数值测试 4 项通过；独立兄弟副本完成 6 阶段重放，8 个关键 NPZ 字节一致。

本次仅打包与校验代码，没有重跑冻结拟合、渲染或改变采用版本。算法报告、字段说明和验收口径保留在 `docs/`，其中指向实验结果的文件名是原目录引用，不表示实验数据随本包提供。文件校验清单见 `package_manifest.json`，来源对应及冻结哈希见 `provenance.json`。
