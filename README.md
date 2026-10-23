<p align="center"><img src="logo/logo.png" alt="IVAS 标志" width="128"></p>

<h1 align="center">Surgical Phase & Tool Annotation · IVAS</h1>
<p align="center">手术视频阶段、动作与器械分割标注工作台</p>
<p align="center">Python · PyQt5 · Segment Anything · 维护者 zack zhao</p>

IVAS 将手术视频的时间标注和单帧器械分割整合到同一个桌面项目：在 **Video** 模式记录 Phase / Task 区间，在 **Frame** 模式精修器械或组织轮廓，再通过视频追踪将已标注对象传播到其他帧。阶段标注与手工分割均可独立使用；启用 AI 分割或追踪时再加载相应模型。

## 主要功能

| 工作流 | 支持的操作 |
| --- | --- |
| 视频阶段与动作 | Phase / Task 双轨时间轴、区间起止、时间点与关键帧、评分、备注、模板、覆盖率检查和审核状态 |
| 器械与组织标注 | 手工多边形、旋转框、轮廓编辑、类别与实例管理、图层、语义 / 实例预览 |
| AI 辅助分割 | SAM 系列模型；按模型能力提供点、框、视觉及文本提示；支持 Original / BF16 / FP16 / FP32 精度设置 |
| 视频对象追踪 | 兼容的 SAM2 / SAM3 / SAM3.1 模型，多帧修正、向前 / 向后 / 双向传播，先暂存审核再提交，支持回滚 |
| 项目与数据交换 | 统一视频项目、帧图像与 JSON 标签分开保存；视频 CSV / TXT 交换；COCO、YOLO、LabelMe、VOC 等图像标注转换 |
| 桌面使用 | 中英文界面、可停靠面板、快捷键、自适应布局、可选 VLC 解码后端、Windows 便携版构建 |

## 关键界面与功能说明

以下截图来自当前项目的实际 Qt 界面。画面、区间和器械掩码均为合成演示数据，用于说明操作位置；掩码由人工预设，不代表模型推理结果。

### 1. 视频阶段与动作标注

![视频阶段与动作标注界面](docs/images/video-phases.png)

- **顶部 Mode / 项目**：切换 Video 与 Frame，管理项目输出目录。“文件”菜单按 **Image / Video / 通用** 分组，以虚线分隔：图像目录与帧标签目录放在 Image，视频、已有视频项目、项目输出目录、媒体重新关联及视频标注导入导出放在 Video；保存当前工作、设置与退出放在通用组。
- **视频标注工具栏**：选择 `Phase` 或 `Task` 和标签，在播放位置记录区间开始与结束。视频控件获得焦点时可用 `I` 开始、`O` 结束。
- **中央预览与底部播放条**：播放 / 暂停、前后逐帧定位、调整速度；时间码用于精确查看标注边界。
- **底部双轨时间轴**：上轨显示 Phase，下轨显示 Task；支持缩放、平移和选择已有区间，便于检查阶段覆盖与动作重叠。
- **右侧视频标注面板**：列出标签、开始 / 结束时间和状态，显示覆盖率及待审核数量；可继续编辑备注、评分和审核状态。

### 2. 当前视频帧的器械分割

![当前视频帧的器械分割界面](docs/images/frame-segmentation.png)

- **进入帧标注**：点击“标注当前视频帧”或选择 `Mode → Frame`，提取播放位置对应帧，进入图像画布；编辑时播放位置固定。
- **常用标注工具栏**：直接点击 **多边形、SAM 点、SAM 框、重绘**；默认快捷键中，多边形为 `C`，SAM 点为 `Q`，SAM 框为 `W`。较宽窗口显示图标和名称，窄窗口显示图标并保留悬停提示；SAM 工具在模型和当前图像编码就绪后可用。
- **中央分割画布**：绘制多边形、调整轮廓和顶点，也可加载模型后使用点 / 框等提示辅助分割。图中浅色轮廓为演示器械的人工标注。
- **右侧类别 / 标注 / 文件列表**：查找并选择器械或组织类别，管理当前帧的对象、实例分组和图层；类别可按任务自定义。面板宽度会根据窗口大小、显示缩放和中英文控件所需空间自动调整，避免按钮被挤窄。
- **保存与返回**：点击对勾 `Finish`（或按 `E`）完成标注后立即保存当前帧 JSON，无需开启全局自动保存；其他修改可按 `Ctrl+S` 保存。点击“返回视频”继续阶段标注。未完成的 Phase / Task 区间会保留，帧标签与时序标注各自保存。

语义和实例显示用于检查结果；编辑轮廓时需要回到标签显示。

不同分辨率下，右侧工具面板会为画布保留空间；窗口很小或字体较大时，可通过面板滚动条访问完整内容。顶部面板标签过多时，用左右箭头切换。拖动画布与面板之间的分隔线可手动调整宽度；窗口尺寸或面板内容变化后会重新适配。浮动面板保留独立尺寸，`视图 → 重置布局` 可恢复默认停靠布局。

### 3. 视频追踪结果审核

![视频追踪暂存结果审核界面](docs/images/tracking-review.png)

通过“事务追踪”设置范围和方向，在兼容模型上运行对象传播；完成后打开暂存结果审核窗口：

- **左侧帧列表**：选择需要检查的结果帧，查看该帧涉及的对象数量。
- **中间原始帧 / 叠加预览**：对照原画面与彩色掩码，检查器械边界、漏分割和对象身份；图例标明类别与追踪 ID。
- **底部操作**：“提交到标签”写入正式帧标注，“丢弃暂存”删除本次待提交结果，“取消”关闭审核窗口。

追踪结果在审核前保持暂存；提交默认保留人工对象，失败会恢复标签，已提交的追踪运行可回滚。截图使用人工构造的暂存结果演示审核流程。

### 4. SAM 模型管理

![SAM 模型管理界面](docs/images/model-manager.png)

通过 `SAM` 菜单打开模型管理，查看模型是否支持 **Image / Video**、显存与磁盘需求，并下载或选择权重。绿色勾号表示支持该类任务；普通图像分割模型不一定支持视频追踪。

模型精度在设置中选择，具体提示方式取决于模型。下载权重放在 `IVAS/checkpoints/`；权重不随 Git 仓库分发。仅做阶段标注时无需加载模型。

### 5. 标注格式转换

![图像标注格式转换界面](docs/images/format-converter.png)

通过“工具”中的格式转换器，在 COCO、YOLO、LabelMe、VOC 等页签中选择输入图像、标签路径及输出目录，执行与 IVAS JSON 的转换。不同目标格式可表达的信息不同，转换后应抽查类别、实例及轮廓。

此窗口处理**图像标注**；视频时序标签的 CSV / TXT 导入导出位于“视频标注”菜单。

## 安装与启动

需要 **Python 3.10 或更高版本**，建议使用 Python 3.11 和独立环境。当前 Windows 开发环境使用 Python 3.11；模型推理还需要与本机驱动匹配的 PyTorch，显存需求以所选模型为准。

```powershell
git clone https://github.com/AIEyeSystem/Surgical_PhaseTool_Annotation.git
cd Surgical_PhaseTool_Annotation
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe main.py
```

也可在激活环境后使用 `ivas` 或 `ivas-sam` 启动。Linux / macOS 的虚拟环境解释器为 `.venv/bin/python`；桌面运行需要图形环境。

源码目录和可写便携目录沿用 `IVAS/` 内的配置与模型目录。安装到 `site-packages` 或程序目录只读时，配置和下载权重使用用户数据目录（Windows 为 `%LOCALAPPDATA%/IVAS`，其他系统为 `$XDG_DATA_HOME/IVAS` 或 `~/.local/share/IVAS`）。可通过 `IVAS_DATA_DIR` 指定位置；首次加载会读取随程序附带的默认配置，保存时再创建用户文件。

视频默认使用 Qt Multimedia。遇到 Qt 无法解码的视频，可安装与 Python 位数一致的 VLC 桌面运行库，程序会尝试 VLC 后端；仅安装 `python-vlc` 并不包含 VLC 原生运行库。

### 第一次标注

1. 在“项目”中设置新视频项目的输出目录，再从“文件”打开视频。
2. 在 Video 模式选择 Phase / Task 标签，播放或逐帧定位，用起止按钮创建区间；按需设置模板、备注、评分和审核状态。
3. 需要标注器械时，在目标时间点点击“标注当前视频帧”，进入 Frame 模式，选择类别并绘制轮廓；AI 辅助功能需先加载模型。
4. 保存帧标注后返回视频。需要跨帧传播时，选择兼容模型，运行追踪并检查暂存结果，再提交。
5. 保存视频项目，并按需导出时序 CSV / TXT 或转换图像标签格式。

只处理独立图像时，通过“文件”打开图像目录，在 Frame 模式完成标注即可。

## 数据保存与快捷键

一个视频病例的输出目录通常包含：

```text
<输出目录>/<视频名>_<标识>/
├── project.ivas.json       # 视频、时序标签、模板、帧引用和追踪记录
├── frames/                 # 提取的视频帧
├── labels/                 # 帧级器械 / 组织标注 JSON
├── tracking/               # 追踪暂存与相关资产
└── exports/                # 保存项目时同步的 CSV / TXT 导出
```

重新打开标注时选择 `project.ivas.json`。迁移项目应保留病例目录内的帧、标签和追踪资产；原视频路径变化后，通过项目菜单重新关联媒体。

| 场景 | 默认快捷键 | 操作 |
| --- | --- | --- |
| 通用 | `Ctrl+S` | 保存当前工作 |
| 视频控件获得焦点 | `Space`、`←` / `→` | 播放 / 暂停、前后逐帧 |
| 视频控件获得焦点 | `I` / `O` | 记录区间开始 / 结束 |
| 视频控件获得焦点 | `Ctrl+Z` / `Ctrl+Shift+Z` | 撤销 / 重做时序标注操作 |
| 图像画布 | `C`、`Q`、`W` | 手工多边形、SAM 点提示、SAM 框提示 |
| 图像画布 | `E` / `Esc` | 完成并保存 / 取消绘制 |
| 图像画布 | `A` / `D`、`F` | 上一张 / 下一张、适应窗口 |

快捷键可在设置中调整；Frame 与 Video 按当前编辑区域分配按键。原版 ISAT 标注 JSON 和 `isat.yaml`、旧版 `project.ivat.json`、`ivat.yaml` 和 `ivat.plugins` 入口仍兼容；新文件使用 IVAS 命名。原版标注无需手动修改 `info.description`。

类别名称以当前类别配置为准，优先读取 `ivas.yaml`。同目录的 `ivas_rename.yaml` 只提供显示别名：类别数量、完整的 ID / 名称映射（无 ID 时按顺序）及双方提供的颜色必须匹配。任何一项不匹配、缺项、重复 ID 或无效内容都会使整份 rename 文件被忽略，所有类别恢复为原名称和类别 ID；不会部分套用别名，也不会修改这两个配置文件。

`ivas.yaml` 保存类别与颜色，`software.yaml` 保存语言、快捷键、模型精度等应用设置。旧版 `software.yaml` 中的 `label` 会被忽略，并在下次保存应用设置时移除。保存标注、修改应用设置和退出程序都不会覆盖已有的类别文件；只有在类别设置中点击“应用”或主动导出时才写入类别，首次保存到没有类别配置的新标签目录时会创建一份副本。在外部编辑类别 YAML 后，重新导入配置或重启程序即可加载修改。

选中画布轮廓或 Annos 条目时，Categories 会同步显示对应类别。要修改已有 mask，先选中轮廓，再在 Categories 选择目标类别，点击与 Segment 同行的 **Change**。轮廓颜色、Annos 名称和标注对象同步更新，几何形状、实例编号与追踪 ID 保留；按 `Ctrl+S` 保存，开启自动保存时会立即保存。支持一次修改多个选中轮廓。

加载 SAM3 / SAM3.1 后，在 Frame 模式选择一个前景类别，Categories 中 **Segment / Change** 的下一行提供文件夹批量分割：

- **SegAll**：用当前类别名作为 prompt，逐张分割当前打开文件夹的全部图片。
- **SegMiss**：只处理已有标注中没有当前类别的图片；没有 JSON、空标注或只有其他类别的图片都会处理，已有该类别的图片跳过。判断使用原始类别名，不受中文显示别名影响。

批量操作会先保存当前图片尚未保存的修改，再将新轮廓追加到各图片在当前标签目录中的 JSON；已有标注保留，新轮廓标记为 SAM 草稿。重复执行 SegAll 会再次追加结果，补漏时可用 SegMiss。进度窗口支持取消：已保存的图片保留，正在推理但尚未保存的结果丢弃；个别图片或标签出错时继续处理后续图片，并在结束后显示错误汇总。无检测结果的图片不会标记为“已有该类别”，以后仍可参与 SegMiss。处理范围是当前文件列表，不递归子目录；视频项目中仅处理已提取到当前帧目录的图片。

## Windows 便携版构建

双击 `build_exe.bat` 即可构建，或在 PowerShell 中运行以下命令。脚本自动识别标准 venv 的 `.venv/Scripts/python.exe` 和 Conda 前缀的 `.venv/python.exe`；也可用 `-PythonExe` 指定已有环境。没有本地环境时会尝试克隆本机 `labeling` Conda 环境。依赖已满足时不再访问网络，否则安装缺失的运行和打包依赖。

```powershell
# 只打包程序及运行库，不内置模型权重
.\build_exe.bat -SkipCheckpoint

# 若要打包 SAM3.1，将 sam3.1_multiplex_bf16.pt 放在项目根目录后执行
.\build_exe.bat

# 指定环境，并跳过依赖安装（需要事先装齐依赖）
.\build_exe.bat -PythonExe .\.venv\Scripts\python.exe -SkipDependencyInstall -SkipCheckpoint
```

构建成功后直接运行 **`build/IVAS/IVAS.exe`**。分发时需要整个 `build/IVAS/` 文件夹及其 `_internal/` 内容。新版本在临时目录完成打包及启动验证后才替换发布目录；上一版保存在 `build/IVAS.previous-时间戳/`。构建日志在 `tmp/portable-build.log`，启动与模型推理验证结果在成品目录的 `SMOKE_TEST.log`。

## 开发与仓库内容

```text
IVAS/               应用入口、界面、标注格式、SAM 适配和视频项目逻辑
docs/images/        README 使用的关键界面截图
example/            原项目随附的示例素材
test/               标注、项目保存、界面与追踪回归测试
tools/              截图与功能验证工具
build_exe.spec      Windows 打包配置
```

在已安装依赖的环境运行现有测试：

```powershell
python -m unittest discover -s test -p "test_*.py"
```

重新生成上述演示截图：

```powershell
python tools/capture_readme_screenshots.py
```

`.gitignore` 排除了虚拟环境、模型权重、构建目录、日志缓存、本地数据 / 标注输出、参考项目和本机凭据；应用源码、模型 YAML 配置、打包脚本和 `docs/images/` 截图纳入版本管理。`IVAS/software.yaml` 是随附默认配置，本地类别文件 `IVAS/ivas.yaml` 由程序管理。

更多界面与保存边界说明见 [工作流审计](docs/annotation-workflow-audit.md)。问题反馈请使用[本仓库 Issues](https://github.com/AIEyeSystem/Surgical_PhaseTool_Annotation/issues)。

## 许可证与致谢

本项目基于 [ISAT with Segment Anything](https://github.com/yatengLG/ISAT_with_segment_anything) 扩展，当前应用名称为 **IVAS**，维护者为 **zack zhao**。保留上游版权及第三方代码声明，项目许可见 [LICENSE](LICENSE)。模型权重和第三方组件遵循各自的许可条款。
