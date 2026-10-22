# Hand Marker / MANO Stability

基于手部 marker 观测的 MANO 掌根与拇指稳定性研究代码，整理自 `latest_hand_marker_code_20261022.zip`（2026-10-22）。采用版为左手 `guarded_axes_v3`，右手 baseline。

## 目录导航

```text
hand-marker-stability/
├── README.md                 仓库导航与使用边界
├── .gitignore                数据、模型、实验输出及缓存排除规则
├── .gitattributes            保留原交付文件字节，避免换行转换破坏哈希
└── code/                     原始交付包，44 个文件保持不变
    ├── README.md             原始环境、运行步骤及结果限制
    ├── run_stability_pipeline.py
    ├── requirements.txt      原环境直接依赖版本
    ├── tools/verify_code_package.py
    ├── docs/                 算法报告、注释格式及验证说明
    └── *.py / *.json         管线模块、配置及来源校验清单
```

## 从这里开始

- [原始使用说明](code/README.md)：环境、外部输入、运行命令及已知限制。
- [算法与采用版报告](code/docs/FINAL_REPORT_ZH.md)。
- [验证口径](code/docs/VALIDATION_NOTES.md)。
- [标注格式](code/docs/ANNOTATION_FORMAT_ZH.md)。
- [外部资产与路径清单](code/external_paths.json)。

处理流程：基线测量 → 拇指约束优化 → 三维保护回退 → 统一评估 → 验收与选择 → 可选渲染。

## 先检查完整性

在仓库根目录执行（仅需 Python 3.10 或更新版本，无需安装拟合依赖）：

```bash
python -B code/tools/verify_code_package.py
```

预期 `passed: true`，检查 43 个清单文件和 29 个管线 Python 文件。清单自身不计入这 43 个文件。`code/` 保持原始目录布局和字节内容，以保留包校验与来源对应关系。

## 运行条件与边界

原运行环境为 Linux / Python 3.10.20，直接依赖见 [requirements.txt](code/requirements.txt)，渲染另需 `ffmpeg`。该文件记录原环境版本，未作为可在新机器复现的完整锁文件验证。

仓库仅包含代码和说明。拟合还需要原实验的 baseline、观测输入、评估记录、原始 capture、标定和合法取得的 MANO 数值资产。原实现包含服务器绝对路径，并将运行输出限制在 `/data2/wangzhengwei`。在其他机器运行前，应在新的研究副本中适配路径并补齐资产，具体步骤见原始使用说明。

RGB 仅用于离线评估，不参与逐帧 pose/root/安装求解；不使用深度。这是现有观测与基线之上的稳定性扩展，并非任意新 C3D 的冷启动入口。

本次整理通过文件完整性、Python 静态语法与 JSON 解析检查；未在当前机器重跑拟合、数值测试或视频渲染。原报告中的指标属于历史实验记录。

## 数据与版本管理

原始 ZIP 和周边采集资料不纳入仓库。`.gitignore` 排除视频、C3D、NPZ、模型资产、重放输入和常见运行输出；提交新实验文件前仍需检查 `git status`。原路径元数据为追溯信息，不表示相关数据已上传。

未添加新的开源许可证；MANO 等外部资产须遵守各自授权条款。
