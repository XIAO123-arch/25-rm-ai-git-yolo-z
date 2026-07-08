# 25届 RM 算法组年度考核任务规划

## 当前任务主题

用 AI 工具链 + YOLO 快速做出一个“demo 目标检测与标记”小程序。

当前阶段先使用参考仓库自带的 drone demo 数据集，不加入赛题专用数据要求。这样可以先把 AI 辅助二次开发、Git 规范、YOLO 推理、图片/视频处理和项目表达跑通。

核心考察点：

- AI 工具链辅助二次开发
- Git 基础协作与提交规范
- YOLOv8 基础推理流程理解
- 目标框绘制、类别名显示、置信度显示、结果保存
- 图片、视频、简单多目标过滤
- README、Prompt 记录、个人反思

## Starter 仓库选型

### 推荐基础仓库

仓库：<https://github.com/ynsrc/python-yolov8-examples>

当前使用其中的 `drone-detect/` demo 数据、模型和示例脚本作为参考。

选择理由：

- 代码量小，适合新生阅读和二次开发。
- 已包含 YOLOv8 图片检测脚本：`drone-detect/detect-from-image.py`。
- 已包含 YOLOv8 视频检测脚本：`drone-detect/detect-from-video.py`。
- 已包含示例图片、示例视频和 `model.pt`，便于快速跑通。
- 仓库授权为 The Unlicense，适合作为教学参考和二次裁剪。

不直接选择 `ultralytics/ultralytics` 主仓作为 Starter 的原因：

- 官方仓库功能完整但体量较大。
- 对新生来说 Fork 后改造成本偏高。
- 更适合作为官方 API 和依赖文档来源，而不是考核 Starter。

官方参考：

- Ultralytics GitHub：<https://github.com/ultralytics/ultralytics>
- Ultralytics Python 使用文档：<https://docs.ultralytics.com/usage/python/>

## Starter 目录结构

```text
rm-yolo-demo-starter/
├─ README.md
├─ requirements.txt
├─ detect.py
├─ models/
│  └─ best.pt
├─ assets/
│  ├─ example-images/
│  │  ├─ 01.jpg
│  │  └─ ...
│  └─ example-videos/
│     └─ 01.mp4
├─ outputs/
│  └─ .gitkeep
└─ docs/
   ├─ prompts.md
   └─ reflection.md
```

Starter 默认运行命令：

```bash
pip install -r requirements.txt
python detect.py --source assets/example-images/01.jpg --output outputs/result.jpg
```

## 新生任务要求

学生需要 Fork 本仓库，并在 3-5 天内完成二次开发。

最低功能要求：

- [ ] 能处理单张图片输入。
- [ ] 支持 `--source` 参数指定输入路径。
- [ ] 支持 `--conf` 参数过滤低置信度目标。
- [ ] 在图像上画出目标框。
- [ ] 在目标框旁显示类别名和置信度，例如 `drone 0.87`。
- [ ] 将处理结果保存到 `outputs/`。
- [ ] README 写清楚安装、运行、参数说明和效果展示。
- [ ] `docs/prompts.md` 记录 AI 提问、AI 回答摘要、采纳内容和人工修改。
- [ ] `docs/reflection.md` 写个人反思。

二次开发要求：

- [ ] 支持 `--max-det` 参数，保留置信度最高的前 N 个目标。
- [ ] 支持视频输入和输出：`assets/example-videos/01.mp4` -> `outputs/result.mp4`。
- [ ] 支持批量处理 `assets/example-images/` 目录中的图片。
- [ ] 输出处理统计信息，例如检测到多少个目标、保存到哪里。
- [ ] 整理 1 分钟演示视频。

可选拓展：

- [ ] 增加 `--class-name` 或 `--class-id`，只保留指定类别。
- [ ] 增加 ROI 颜色统计或目标中心点显示。

推荐运行方式：

```bash
python detect.py --source assets/example-images/01.jpg --conf 0.4 --output outputs/01_result.jpg
python detect.py --source assets/example-images --conf 0.4 --output outputs/images
python detect.py --source assets/example-videos/01.mp4 --conf 0.4 --max-det 8 --output outputs/01_result.mp4
```

## 推荐实现思路

当前 Starter 已经完成单张图片检测。学生主要在此基础上补齐过滤、批处理和视频保存。

流程：

```text
读取图片/视频帧
-> YOLOv8 推理得到候选框
-> 根据置信度过滤
-> 按置信度排序并执行 --max-det
-> 画框、写类别名和置信度
-> 保存图片或视频
-> 输出处理统计信息
```

视频处理建议：

- 使用 `cv2.VideoCapture` 读取输入视频。
- 使用 `cv2.VideoWriter` 保存输出视频。
- 每一帧复用图片检测和画框逻辑。
- 注意读取原视频的宽、高、FPS。

批量图片处理建议：

- 如果 `--source` 是目录，则遍历常见图片后缀：`.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`。
- 每张图片单独保存到输出目录。
- 输出文件名保留原始文件名，例如 `outputs/images/01.jpg`。

## Git 规范要求

学生必须保留清晰 Git 历史，不接受最后一次性提交全部代码。

建议提交节奏：

```bash
git commit -m "chore: run starter detection successfully"
git commit -m "feat: add max detection filtering"
git commit -m "feat: support video inference"
git commit -m "feat: support batch image inference"
git commit -m "docs: add prompts and reflection"
```

最低要求：

- [ ] 至少 4 个有效 commit。
- [ ] commit message 能看出每次改动目的。
- [ ] 不提交虚拟环境目录、缓存目录和大量临时输出。
- [ ] `.gitignore` 覆盖 `__pycache__/`、`.venv/`、大体积临时输出等。

## AI 使用要求

允许并鼓励使用 Claude、GPT、Cursor、Codex 等工具，但必须记录过程。

`docs/prompts.md` 至少包含：

- [ ] Prompt 原文或简化后的关键问题。
- [ ] AI 回答摘要。
- [ ] 自己采纳了什么。
- [ ] 自己没有采纳什么，以及原因。
- [ ] 最终代码中哪部分是 AI 辅助完成的。

禁止事项：

- [ ] 只粘贴 AI 生成代码但无法解释。
- [ ] Prompt 文档只写“我问了 AI，它帮我改好了”。
- [ ] README 与实际运行命令不一致。

## 提交材料

最终提交格式：

```text
姓名：
GitHub 仓库链接：
1分钟演示视频链接：
Prompt 文档路径：
自评：通过 / 良好 / 优秀
```

仓库中至少包含：

```text
README.md
requirements.txt
detect.py
docs/prompts.md
docs/reflection.md
outputs/example_result.jpg
```

## 评分标准

基础分 100 分，额外加分最多 20 分。通过线按基础分计算，额外加分主要用于优秀评定和组内排序。

### 基础分：100 分

| 模块 | 分值 | 标准 |
| --- | ---: | --- |
| 程序能跑通 | 25 | 能处理 demo 图片并保存结果；运行命令和 README 一致；结果图中目标框、类别名、置信度可见 |
| YOLO 二次开发 | 20 | 实现 `--max-det`、视频保存、批量图片处理中的至少两项；逻辑清楚，参数可复用 |
| Git 使用 | 20 | 至少 4 个有效 commit；commit message 清楚；没有提交虚拟环境、缓存和大量临时输出 |
| README | 15 | 安装、运行、参数、效果展示、常见问题说明清楚 |
| Prompt 记录 | 10 | 真实记录 AI 提问、AI 回答摘要、采纳内容和人工修改 |
| 个人反思 | 10 | 能说清程序流程、YOLO 输入输出、遇到的问题、AI 帮助和自己修改的部分 |

基础分细则：

- `--source`、`--conf`、`--output` 等参数必须实际可用。
- 视频或批量处理如果只写了代码但没有验证，不计入对应功能分。
- README 中的命令必须能在干净环境中复现。
- Prompt 文档不能只贴完整聊天记录，需要体现自己的判断。
- 答辩时无法解释核心代码的，相关功能分可以酌情扣除。

### 额外加分：最多 20 分

额外加分项不要求所有人完成，鼓励有基础的同学探索推理部署和性能优化。加分必须提供可复现命令、环境说明和前后对比数据。

| 加分项 | 分值 | 标准 |
| --- | ---: | --- |
| TensorRT 推理优化 | 最高 8 | 成功导出并运行 TensorRT engine；提供转换命令、运行命令、FPS/单帧耗时对比 |
| OpenVINO 推理优化 | 最高 8 | 成功导出并运行 OpenVINO 模型；提供转换命令、运行命令、FPS/单帧耗时对比 |
| ONNX Runtime / OpenCV DNN 推理 | 最高 5 | 成功导出 ONNX 并使用非 Ultralytics 原生接口推理；结果可视化正确 |
| Benchmark 脚本 | 最高 4 | 提供 `benchmark.py` 或等价脚本，统计平均耗时、FPS、测试图片/视频数量 |
| 工程化封装 | 最高 4 | 支持后端选择，例如 `--backend torch|onnx|tensorrt|openvino`，并保留普通 PyTorch 兜底路径 |

加分限制：

- 额外加分累计最高 20 分。
- 只提交转换后的文件但不能运行，不加分。
- 只声称“更快”但没有测试数据，不加分。
- 测试数据需说明硬件环境，例如 CPU/GPU 型号、系统、Python 版本。

通过线：60 分。

优秀线：85 分以上，通常需要同时满足：

- 图片、视频、批量图片都能跑通。
- 参数设计清晰。
- Git 历史规范。
- README 有效果图。
- 反思真实，不是模板话。
- 能在答辩中解释核心代码。

## 3-5 天执行节奏

### Day 1：跑通 Starter

- [T] Fork 仓库。
- [T] Clone 到本地。
- [T] 安装依赖。
- [T] 跑通原始图片检测。
- [T] 完成第一条 commit。

### Day 2：理解代码和记录 Prompt

- [T] 用 AI 辅助解释 `detect.py`。
- [T] 标出 YOLO 推理、读取输入、画框、保存结果的位置。
- [T] 创建 `docs/prompts.md`。
- [T] 创建 `docs/reflection.md` 初稿。

### Day 3：实现多目标过滤

- [T] 增加 `--max-det` 参数。
- [T] 按置信度排序目标。
- [T] 只画出保留后的目标。
- [T] 输出检测统计信息。

### Day 4：实现视频和批量图片

- [T] 支持视频读取和保存。
- [T] 支持目录输入，批量处理 demo 图片。
- [T] 测试不同 `--conf` 和 `--max-det` 参数。

### Day 5：整理提交材料

- [ ] 完善 README。
- [ ] 补充 Prompt 记录。
- [ ] 完成个人反思。
- [ ] 录制 1 分钟演示视频。
- [ ] Push 到 GitHub。
- [ ] 提交仓库链接和视频链接。
