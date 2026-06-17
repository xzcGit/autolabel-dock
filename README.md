# AutoLabel Dock

> 标一点、训一版、再自动标 —— 桌面端图像标注与 YOLO 训练的迭代闭环工具。

![Python](https://badgen.net/badge/Python/%E2%89%A53.10/blue)
![License](https://badgen.net/badge/License/AGPL--3.0/green)
![Qt](https://badgen.net/badge/Qt/PyQt5/41cd52)

[English](README_EN.md) | **简体中文**



AutoLabel Dock 是一款基于 **PyQt5 + Ultralytics YOLOv8** 的桌面端图像标注工具，全中文界面，跨 Linux / macOS / Windows 运行。

它把"标注"与"训练"做成一个迭代闭环：先手动（或用现有模型辅助）标注一批图，人工确认后一键训练自定义 YOLO 模型，再用新模型继续自动标注剩余数据——每迭代一轮，需要手动修正的越来越少。

![AutoLabel Dock](resources/screenshots/overall.png)

---

## 界面预览

| 面板 | 界面展示 |
|:---|:---:|
| 标注（检测/关键点） | ![标注界面](resources/screenshots/labeling.png) |
| 分类 | ![分类界面](resources/screenshots/cls.png) |
| LocateAnything | ![LA界面](resources/screenshots/locateanything.png) |
| 训练面板 | ![训练面板](resources/screenshots/train.png) |
| 模型面板 | ![模型管理](resources/screenshots/models.png) |

---

## 特性

### 🔍 三类标注任务

- **目标检测（detect）**：边界框（bbox）标注
- **关键点姿态（pose）**：bbox + 骨架关键点
- **图像分类（classify）**：整图单标签，网格选择器 + `1`–`9` 数字键快捷标注，标完自动跳下一张

### ✏️ 标注体验

- 类似 LabelImg 的键盘流操作体验
- 画布流畅绘制：bbox 与关键点放置、拖拽移动/缩放、滚轮缩放与平移；大量标注时视口剔除保持帧率
- 每张图独立撤销栈（深度 50），切图自动保存，LRU 图像缓存
- 文件列表按状态着色（已确认/待确认/未标注），支持拖拽导入图片、状态+类别+标签三重过滤、右键批量操作
- Catppuccin Mocha 暗色主题

<details>
<summary><b>快捷键表</b></summary>

**工具与模式**

| 快捷键 | 功能 |
|:---|:---|
| `W` | 画框模式 |
| `K` | 关键点模式 |
| `V` | 选择/移动模式 |

**导航**

| 快捷键 | 功能 |
|:---|:---|
| `A` / `←` | 上一张（自动保存当前） |
| `D` / `→` | 下一张（自动保存当前） |

**标注操作**

| 快捷键 | 功能 |
|:---|:---|
| `Space` | 确认选中标注 |
| `Delete` | 删除选中标注 |
| `Ctrl+Z` / `Ctrl+Y` | 撤销 / 重做 |
| `Ctrl+C` / `Ctrl+V` | 复制 / 粘贴标注 |
| `Ctrl+S` | 保存 |

**视图**

| 快捷键 | 功能 |
|:---|:---|
| `Ctrl++` / `Ctrl+-` | 放大 / 缩小 |
| `Ctrl+0` | 适应窗口 |
| `F5` | 重新扫描图片目录 |

**分类专用**

| 快捷键 | 功能 |
|:---|:---|
| `1`–`9` | 快速选择类别标签 |

</details>

### 🤖 模型辅助自动标注

- 加载任意 YOLO 权重做单图或批量预标注（批量在后台线程运行，逐图落盘）
- 冲突检测：预测结果与已确认的同类标注按 IoU 匹配，避免产生重复框
- 确认生命周期：自动标注默认"待确认"（虚线框），任何手动编辑即视为确认；整图确认后画布锁定防误触
- 可选 LocateAnything-3B 文本标注后端，用自然语言描述要标注的目标（见「[可选：LocateAnything 文本标注](#可选locateanything-文本标注)」）

### 🏷️ 标签（Tag）子系统

- 给图片打自定义标签（独立于分类标签），用于数据集组织
- 三态过滤芯片（无 → 包含 → 排除），多个包含标签时支持 AND/OR 组合
- 选中多张图片后按 `T` 批量打标签
- 训练时可用同一套标签过滤器筛选训练子集

### 🔄 训练闭环

- 一键数据集准备：按主类别分层抽样切分 train/val，符号链接零拷贝（Windows 自动降级，见[平台说明](#平台说明)）
- 训练预设 + 完整超参可调；实时 loss / mAP 曲线；可中途取消
- 训练完成自动注册到模型库并自动加载，立刻可用于推理

### 📦 模型管理

- 模型库：训练产物自动登记，支持导入外部权重、重命名、删除
- 多模型指标对比对话框

### 📥 数据导入 / 导出

| 格式 | 导出 | 导入 | 适用任务 |
|:---|:---:|:---:|:---|
| YOLO (txt) | ✅ | ✅ | 检测 / 姿态 |
| COCO (json) | ✅ | ✅ | 检测 / 姿态 |
| labelme (json) | ✅ | ✅ | 检测 / 姿态 |
| ImageFolder | ✅ | ✅ | 分类（按类别分文件夹） |
| CSV | ✅ | ❌ | 分类 |

### 🛡️ 数据安全

- **自动备份**：导出、类别变更等破坏性操作前自动快照到项目内 `.backups/`，保留最近 20 份；恢复前再做一次安全备份
- **独立存储**：标注按图存 JSON 文件，单个文件损坏不影响其余标注
- **全局配置**：最近项目、窗口几何、阈值等存于 `~/.autolabel/`

---

## 快速开始

```
1. 创建项目  →  2. 导入图片  →  3. 标注  →  4. 确认  →  5. 训练  →  6. 迭代 ↻
```

1. **创建项目**：选择任务类型（检测 / 姿态 / 分类），指定项目目录（项目会自动扫描加载images/labels），或留空后通过下一步拖入
2. **导入图片**：若项目内尚无图片，拖拽图片到文件列表即可（也可后续放入项目 `images/` 目录后按 `F5` 刷新）
3. **标注**：手动绘制；或加载一个 YOLO 权重做自动预标注；也可选用 LocateAnything 文本标注后端，用自然语言描述要标注的目标（见[可选：LocateAnything 文本标注](#可选locateanything-文本标注)）
4. **确认**：逐图检查自动标注结果，编辑即确认
5. **训练**：一键准备数据集并启动训练，实时查看曲线
6. **迭代**：训练完成的新模型自动加载，继续自动标注剩余图片

---

## 环境要求

- Python ≥ 3.10
- 操作系统：Linux / macOS / Windows

## 安装

核心依赖很轻：**PyQt5**（界面）与 **Ultralytics**（YOLO 推理/训练），外加 pyqtgraph（训练曲线）等少量库，完整清单见 `requirements.txt`。

推荐使用 Miniconda 创建独立环境：

```bash
git clone https://github.com/xzcGit/autolabel-dock.git
cd autolabel-dock
conda create -n autolabel python=3.10 -y
conda activate autolabel
pip install -r requirements.txt
```

可选的 LocateAnything-3B 文本标注后端体积较大，按需安装，启用条件见「[可选：LocateAnything 文本标注](#可选locateanything-文本标注)」。

## 运行

```bash
python main.py
```

## 模型权重

本项目**不提供任何模型权重**。YOLO 官方预训练模型（`yolov8n.pt`、`yolov8s.pt`、`yolo26n.pt` 等）首次使用相关功能时由 Ultralytics 自动下载；若自动下载失败，可手动下载对应的 `.pt` 文件放到仓库根目录（`autolabel-dock/`）下。LocateAnything-3B 权重的获取方式见下节。

## 可选：LocateAnything 文本标注

LocateAnything-3B 是可选的开放词汇检测后端，用自然语言描述要标注的目标。不启用它时应用其余功能完全正常。启用需同时满足以下三个条件，任一不满足时界面会给出中文提示，不影响应用本身：

**1. 安装可选依赖**

```bash
pip install -e ".[locateanything]"
```

会额外安装 transformers、accelerate、bitsandbytes、decord（基础安装不包含这些重依赖）。

**2. 提前下载模型权重**

运行时以离线模式（`HF_HUB_OFFLINE=1`）加载，**不会自动下载**，必须提前手动下载 `nvidia/LocateAnything-3B` 到本地 HuggingFace 缓存：

```bash
hf download nvidia/LocateAnything-3B
# 旧版工具：huggingface-cli download nvidia/LocateAnything-3B
```

默认缓存位置为 `~/.cache/huggingface/hub`，也支持通过 `$HF_HOME` 或 `$HUGGINGFACE_HUB_CACHE` 指定。

**3. GPU 显存**

需要 NVIDIA GPU 且 `nvidia-smi` 可用，**不支持 CPU 运行**；总显存 ≥ 6GB，且启用时空闲显存 ≥ 5GB（单卡机器上桌面显示也占用显存，故空闲门槛较高）。

> 另外：LocateAnything 与 YOLO 模型不会同时占用 GPU——启用 LocateAnything 会自动卸载已加载的 YOLO 模型，反向（加载 YOLO 模型或开始训练）会先弹窗确认关闭 LocateAnything。它运行在独立子进程中，与主界面进程隔离。

---

## 平台说明

训练数据集准备会创建大量"指向原图"的链接。代码（`src/utils/fs.py` 的 `link_or_copy`）按以下优先级自动降级，保证在任何平台都能运行：

```
symlink（符号链接） → hardlink（硬链接） → copy（复制）
```

| 方式 | 触发条件 | 速度 | 额外占用 |
|:---|:---|:---:|:---:|
| symlink | 系统支持且有权限（Linux/macOS 默认；Windows 需开发者模式或管理员） | 最快 | 几乎为零 |
| hardlink | symlink 失败，且源与目标在同一磁盘卷（Windows NTFS 无需特权） | 最快 | 几乎为零 |
| copy | 前两者均失败（典型：跨盘 + 非开发者模式） | 慢 | 与图片总大小相同 |

> ⚠️ **Windows 推荐**：启用开发者模式（设置 → 隐私和安全 → 开发者选项），一次设置永久生效，行为与 Linux 一致；或保证项目目录与图片目录在同一磁盘（自动走硬链接路径）。
>
> **其他注意**：建议项目路径使用纯英文；若把图片目录指向另一块磁盘（跨盘），链接会退化为复制，占用额外磁盘空间。

---

## 项目结构

```
autolabel-dock/
├── src/
│   ├── core/         纯数据模型与 IO（标注、项目配置、导入导出格式、备份）
│   ├── engine/       YOLO 训练 / 推理封装与数据集准备
│   ├── controllers/  UI 与核心层之间的桥接与编排
│   ├── ui/           PyQt5 界面组件（画布、面板、对话框）
│   └── utils/        后台线程、撤销栈、图像缓存等工具
├── tests/            pytest 测试
├── resources/        截图等静态资源
├── main.py           应用入口
└── requirements.txt  依赖清单
```

---

## 贡献

欢迎提交 issue 与 PR。开发环境按上文「安装」配置后，运行测试：

```bash
pytest
```

## 许可证

本项目以 **[AGPL-3.0](LICENSE)** 许可证发布。

> 项目的核心依赖采用强 copyleft 许可：
>
> - **PyQt5** — GPL-3.0
> - **Ultralytics (YOLOv8)** — AGPL-3.0
>
> 依赖中最严格的是 AGPL-3.0，本项目据此对齐。若需在闭源/商业产品中使用，请自行获取相应依赖的商业授权（PyQt 商业授权、Ultralytics 企业授权）。

---

## Links

- **[Linux DO](https://linux.do/)**