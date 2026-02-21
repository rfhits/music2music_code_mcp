# music2music_code_mcp → anticipation 风格 MCP 可行性调研（2026-02-21）

## 1) 调研目标

你的目标是：把 `music2music_code_mcp` 里的编排能力，迁移成类似 `anticipation_mcp/app.py` 的“简单 MCP 工具形态”（函数式、入参清晰、输出 JSON 路径）。

我本次重点回答三个问题：

1. 这个项目是否真的具备可直接调用的“编排推理能力”？
2. 模型大小与部署成本是否可接受？
3. 按 `anticipation_mcp` 的简洁风格改造成 MCP，是否可行、风险在哪？

---

## 2) 调研过程（按执行顺序）

### Step A. 本地代码静态核查（你的 fork）

- 阅读 `Readme.md`，确认该仓库定位、依赖、推理入口、Hugging Face 模型列表。
- 阅读 `api/arranger.py`，确认可直接调用的 3 个核心类：`BandArranger` / `PianoArranger` / `DrumArranger`。
- 阅读 `tutorial.ipynb` 与 `tests/test_infer_api.py`，确认官方/作者给出的最短调用路径。
- 对照 `anticipation_mcp/app.py`，确认可复用的 MCP 封装模式（懒加载模型 + 简单函数 + JSON 返回 + `gradio[mcp]`）。

### Step B. 联网核验 GitHub 仓库状态

- 通过 GitHub API 核验：你的仓库 `rfhits/music2music_code_mcp` 是 fork，父仓库为 `Sonata165/music2music_code`。
- 核验父仓库时间线（UTC）：
  - created_at: **2025-10-02**
  - pushed_at: **2025-11-10**
  - license: **None（未声明）**
- 这说明项目本身不算很老，但**许可证信息缺失**是后续工程化发布的法律风险点。

### Step C. 联网核验 Hugging Face 模型规模

- 通过 HF API（`?blobs=true`）读取模型文件大小、参数量、更新时间。
- 结论：四个主模型都不大，属于可本地部署的小中型（约 87M 参数，单模型权重约 166.7MB）。

| 模型 | 参数量（HF safetensors） | 主权重文件 | 大小 | lastModified (UTC) |
|---|---:|---|---:|---|
| LongshenOu/m2m_pt | 87,388,416 | model.safetensors | ~166.69 MB | 2024-06-21 |
| LongshenOu/m2m_arranger | 87,404,544 | model.safetensors | ~166.73 MB | 2024-08-14 |
| LongshenOu/m2m_pianist_dur | 87,404,544 | model.safetensors | ~166.73 MB | 2024-07-19 |
| LongshenOu/m2m_drummer | 87,404,544 | model.safetensors | ~166.73 MB | 2024-07-15 |
| LongshenOu/m2m_ft（Tokenizer） | - | tokenizer files | ~0.34 MB | 2024-07-11 |

---

## 3) 关键技术发现

### 3.1 已有“可直接推理”的 API，迁移成本不高

`api/arranger.py` 已经是推理 API 雏形：

- `BandArranger(...).arrange(...)`
- `PianoArranger(...).arrange(...)`
- `DrumArranger(...).arrange(...)`

输入是 MIDI 路径，输出是 `remi_z.MultiTrack` 对象，可直接 `.to_midi(...)`。

> 对 MCP 来说，这已经很接近目标，只差“统一参数校验、输出路径管理、JSON 序列化返回”。

### 3.2 推理最小依赖其实很少

`api/arranger.py` 的直接依赖只有：

- `transformers`
- `remi_z`
- `tqdm`

这意味着你不必把训练链路（`lightning/datasets/...`）全搬进 MCP；可以做一个轻量 inference-only 环境。

### 3.3 与 `anticipation_mcp` 的结构映射非常自然

`anticipation_mcp/app.py` 当前模式是：

- 懒加载模型（全局 `_MODEL` 缓存）
- 多个工具函数（continue/inpaint/accompany）
- 参数校验 + 输出 JSON 路径
- `launch(mcp_server=True)` 暴露为 MCP

`music2music` 可以直接平移为：

- `arrange_band_midi(...)`
- `arrange_piano_midi(...)`
- `arrange_drum_midi(...)`

### 3.4 需要修正/注意的问题（很关键）

1. **设备默认值不安全**  
   `api/arranger.py` 默认 `device='cuda'`，无 GPU 环境会直接失败。  
   MCP 版应改成 `auto -> cuda/cpu` 回退（和 anticipation 一样）。

2. **历史条件似乎未更新（代码逻辑风险）**  
   `BandArranger` / `PianoArranger` 里 `prev_bar = None` 后未在循环中更新；  
   `DrumArranger` 里 `prev_seg = None` 也未更新。  
   这会让 `[HIST]` 上下文长期为空，可能影响编排质量。

3. **依赖版本与 Python 版本冲突风险**  
   仓库 README 推荐 Python 3.8，`requirements.txt` 固定了 `numpy==1.23.4`。  
   你当前系统 Python 是 3.12.7，直接按旧 requirements 装可能踩坑。  
   建议单独建 inference venv（3.10/3.11）并只装最小依赖。

4. **许可信息缺失**  
   GitHub 仓库与 HF 模型卡都没有明确 license 字段。  
   研究/个人实验通常可先做，但公开分发前建议补充确认授权范围。

5. **任务边界和 anticipation 不同**  
   `music2music` 是“编排/配器变换”，不是“continue/inpaint”时域编辑。  
   你可以做成“同样简洁”的 MCP，但工具语义会是 arrangement，不是 anticipation 的 1:1 功能复制。

---

## 4) 可行性结论（直接回答你的问题）

**结论：可行，而且工程上是“中低难度可落地”。**

- 从能力角度：已有 3 个可用编排 API，HF 模型可直接加载，模型体量不大。
- 从改造角度：完全可以按 `anticipation_mcp` 的风格做成简洁 MCP。
- 从风险角度：主要在环境版本、默认 device、历史上下文逻辑、license 信息。

我给一个实际评估：

- **MVP（仅推理封装）**：可做，约半天到 1 天。
- **可长期维护版本（含参数健壮性+日志+错误处理）**：约 1~2 天。

---

## 5) 建议的最小 MCP 设计（MVP）

建议先做 3 个工具：

1. `arrange_band_midi(input_midi, output_dir, preset=None, instruments=None, n_samples=1, device='auto', seed=None)`
2. `arrange_piano_midi(input_midi, output_dir, n_samples=1, device='auto', seed=None)`
3. `arrange_drum_midi(input_midi, output_dir, merge_with_input=True, n_samples=1, device='auto', seed=None)`

返回风格与 anticipation 对齐：

```json
[
  {
    "task": "band_arrangement",
    "model": "LongshenOu/m2m_arranger",
    "output_midi": "C:/.../xxx.mid"
  }
]
```

并保留全局缓存（避免重复加载）：

- `_MODELS = { "band": ..., "piano": ..., "drum": ... }`
- `_TOKENIZER = ...`

---

## 6) 我建议你下一步优先做的事

1. 先做 inference-only 环境（不要先装全量训练依赖）。
2. 先跑通 `BandArranger` 一条链路，输出 1 个 MIDI。
3. 再按 anticipation 风格封装为 MCP，并统一输出 JSON。
4. 最后再决定是否修 `prev_bar/prev_seg` 历史逻辑（建议修）。

---

## 7) 证据与来源（本次调研使用）

### 本地代码（你的仓库）

- `music2music_code_mcp/Readme.md`
- `music2music_code_mcp/api/arranger.py`
- `music2music_code_mcp/tests/test_infer_api.py`
- `music2music_code_mcp/tutorial.ipynb`
- `music2music_code_mcp/m2m/hparams/band_arrange/elaborator.yaml`
- `music2music_code_mcp/m2m/hparams/piano_reduction/reduction_dur.yaml`
- `music2music_code_mcp/m2m/hparams/drum_arrange/direct_opd.yaml`
- `anticipation_mcp/app.py`

### 在线来源

- 项目仓库（父仓库）：https://github.com/Sonata165/music2music_code  
- 你的 fork：https://github.com/rfhits/music2music_code_mcp  
- 论文（arXiv）：https://arxiv.org/abs/2408.15176  
- REMI-z 仓库：https://github.com/Sonata165/REMI-z  
- REMI-z PyPI：https://pypi.org/project/REMI-z/  
- 模型页：
  - https://huggingface.co/LongshenOu/m2m_pt
  - https://huggingface.co/LongshenOu/m2m_arranger
  - https://huggingface.co/LongshenOu/m2m_pianist_dur
  - https://huggingface.co/LongshenOu/m2m_drummer
  - https://huggingface.co/LongshenOu/m2m_ft
- GitHub API（仓库元信息）：
  - https://api.github.com/repos/Sonata165/music2music_code
  - https://api.github.com/repos/rfhits/music2music_code_mcp
- Hugging Face API（模型大小/参数）：
  - `https://huggingface.co/api/models/{model_id}?blobs=true`
