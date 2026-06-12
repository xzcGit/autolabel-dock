# AutoLabel Dock

> 基于 PyQt5 + Ultralytics YOLOv8 的桌面端图像自动标注工具。支持目标检测（detect）、关键点姿态（pose）、图像分类（classify）三类任务，并可选接入 LocateAnything-3B 实现开放词汇（文本）标注。界面为中文，跨 Linux / macOS / Windows 运行。

## 特性

- **三类标注任务**：目标检测、关键点姿态、图像分类
- **模型辅助自动标注**：加载 YOLO 权重对图像批量预标注，人工微调即可
- **训练闭环**：内置数据集准备与训练流程，标注 → 训练 → 用新模型再标注
- **可选文本标注后端**：LocateAnything-3B 开放词汇检测，用自然语言指定要标注的目标（需单独安装）
- **跨平台**：Windows 下对数据集链接做了 symlink → hardlink → copy 三级自动降级
- **中文界面**

## 环境要求

- Python ≥ 3.10
- 操作系统：Linux / macOS / Windows

## 安装

```bash
git clone https://github.com/xzcGit/autolabel-dock.git
cd autolabel-dock
pip install -r requirements.txt
```

可选：安装 LocateAnything-3B 文本标注后端（体积较大，按需安装）：

```bash
pip install -e ".[locateanything]"
```

## 模型权重

为保持仓库轻量，**本仓库不包含任何模型权重（`*.pt`）**：

- **`best.pt`（项目自带的标注模型）**：从本仓库 [Releases](https://github.com/xzcGit/autolabel-dock/releases) 下载，放到项目根目录。
- **YOLOv8 官方预训练模型**（`yolov8n.pt`、`yolov8s.pt` 等）：首次使用相关功能时由 Ultralytics 自动下载，无需手动获取。

## 运行

```bash
python main.py
```

## 平台说明：Windows 数据集链接

训练数据集准备会创建大量"指向原图"的链接。代码（`src/utils/fs.py` 的 `link_or_copy`）按以下优先级自动降级，保证在任何平台都能运行：

```
symlink（符号链接） → hardlink（硬链接） → copy（复制）
```

| 方式 | 触发条件 | 速度 | 额外占用 |
|---|---|---|---|
| symlink | 系统支持且有权限（Linux/macOS 默认；Windows 需开发者模式或管理员） | 最快 | 几乎为零 |
| hardlink | symlink 失败，且源与目标在同一磁盘卷（Windows NTFS 无需特权） | 最快 | 几乎为零 |
| copy | 前两者均失败（典型：跨盘 + 非开发者模式） | 慢 | 与图片总大小相同 |

**Windows 推荐**：启用开发者模式（设置 → 隐私和安全 → 开发者选项），一次设置永久生效，行为与 Linux 一致；或保证项目目录与图片目录在同一磁盘（自动走硬链接路径）。

**其他注意**：建议项目路径使用纯英文；若把图片目录指向另一块磁盘（跨盘），链接会退化为复制，占用额外磁盘空间。

## 许可证

本项目以 **[AGPL-3.0](LICENSE)** 许可证发布。

之所以选择 AGPL-3.0，是因为项目的核心依赖采用强 copyleft 许可：

- **PyQt5** — GPL-3.0
- **Ultralytics (YOLOv8)** — AGPL-3.0

依赖中最严格的是 AGPL-3.0，本项目据此对齐。若需在闭源/商业产品中使用，请自行获取相应依赖的商业授权（PyQt 商业授权、Ultralytics 企业授权）。
