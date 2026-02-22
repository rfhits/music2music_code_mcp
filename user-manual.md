# User Manual - Music2Music MCP (GPU)

## 1. 这是什么

本项目提供 3 个符号级（MIDI）编排能力：

- Band Arrangement（乐队编排）
- Piano Arrangement（钢琴编配/还原）
- Drum Arrangement（鼓轨编配）

输入是 **MIDI 文件路径**，输出是新生成的 **MIDI 文件**。不是音频生成器。

---

## 2. 环境准备（仅 GPU）

```powershell
uv python install 3.11
uv venv --python 3.11
uv sync --extra torch-cuda128
```

检查 CUDA：

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 3. 启动页面

```powershell
uv run python app.py
```

默认端口：`7872`  
浏览器打开：`http://127.0.0.1:7872`

改端口：

```powershell
$env:M2M_MCP_PORT=7873
uv run python app.py
```

---

## 4. 页面字段说明

### 4.1 通用字段

- **Input MIDI Path**：本地 `.mid/.midi` 路径（当前是“路径输入”，不是上传按钮）
- **Output Directory**：输出目录（自动创建）
- **Samples**：一次采样生成的候选数量
- **Device**：建议固定 `cuda`
- **Seed**：随机种子（便于复现）
- **Model ID**：可留空，使用默认 HF 模型
- 采样参数：`top_k/top_p/temperature/no_repeat_ngram_size/max_length`

### 4.2 Band 页

- **Preset**：预设 GM 乐器编制
- **Custom Instruments (CSV)**：自定义 GM 乐器号列表（如 `80,26,33`）

规则：

- 选择了 `Preset` 时，优先按 preset；
- 仅当 `Preset` 为空时，`Custom Instruments` 才生效。

当前预设：

- `string_trio` -> `[40, 41, 42]`
- `rock_band` -> `[80, 26, 29, 33]`
- `jazz_band` -> `[64, 40, 61, 26, 0, 44, 33]`

### 4.3 Piano 页

- 使用固定 `piano` 预设（内部仅钢琴目标）。

### 4.4 Drum 页

- 使用固定 `drum` 预设（内部鼓目标，程序号 128）。
- **Merge with input tracks**：是否把生成鼓轨和原始非鼓轨合并输出。

---

## 5. 你关心的限制与边界（重点）

### 5.1 MIDI 长度是否有限制？

- **没有显式“总时长上限”**（代码里未写“最多几秒/几小节”）。
- 但有 **推理 token 预算限制**（`max_length`），所以实际上受“每次生成片段复杂度”影响：
  - Band/Piano：按 **每小节** 逐步生成；
  - Drum：按 **4 小节 segment** 逐步生成。
- 因此：整首歌可以较长，但如果某一小节/某个 4-bar segment 过于密集，可能接近 token 上限，导致质量下降、报错或截断风险。

### 5.2 乐器是否有限制？

- Band 的 `Custom Instruments` 需要逗号分隔整数。
- 实践上应使用 **GM 程序号**（常见 `0..127`），鼓用 `128`（鼓任务内部固定）。
- 代码层对范围校验较宽松（只检查“能否转 int”），但超出训练分布/稀有编号可能效果明显变差。
- 最稳妥做法：优先使用 preset；自定义时用常见 GM 编号。

### 5.3 `max_length` 到底是什么？

- 它是 HuggingFace `generate` 的参数，表示 **生成阶段允许的最大 token 长度上限（输入+输出序列总长上限）**。
- 当前服务默认值来自模型封装，三类任务都设为 `800`。
- 如果你把 `max_length` 调太小，可能“不够生成”；调太大，会更慢、更吃显存，也可能增加不稳定性。
- 现在服务会做保护：如果你传了 `0`/负数，或小于当前输入 token 长度，会自动回退到安全值，避免直接报错中断。
- 建议：
  - 先用默认值；
  - 只在复杂输入报错/输出异常短时，再小步上调（如 960/1024）。

### 5.4 Drum 输入该给什么？Lead sheet 还是随便一段？

- 这个 Drum 任务不是“只吃 lead sheet”的专用接口。
- 根据原项目说明与示例，它是“给一段已有音乐（无鼓或鼓较弱）→ 生成兼容鼓轨”。
- 所以你可以给：
  - 一整首 MIDI；
  - 一个较完整段落（例如 8/16 小节）。
- 也就是“不是必须 lead sheet”；但如果输入只有极简单线条，鼓的上下文会少，生成稳定性和风格匹配可能变差。

### 5.5 其他容易忽略的限制

- 当前 UI 输入是“本地路径字符串”，不是上传控件。
- 仅支持 MIDI，不支持 wav/mp3。
- 数据/配置主要围绕 4/4 量化训练（hparams 里可见 `ts44`），非常规拍号可能效果下降。
- 首次运行会下载模型，第一次慢是正常的。

---

## 6. 最小可用流程（GPU）

1. 启动服务：`uv run python app.py`
2. 打开 `http://127.0.0.1:7872`
3. 任选 Tab（Band/Piano/Drum）
4. 填写 `Input MIDI Path` + `Output Directory`
5. `Device` 选 `cuda`
6. 点击运行，查看返回 JSON 中的 `output_midi` 路径

---

## 7. MCP 工具名（供集成）

- `arrange_band_midi`
- `arrange_piano_midi`
- `arrange_drum_midi`

---

## 8. 常见报错与预防

### `ValueError: ... max_length is set to 0`

原因：`max_length` 被设置成了不合法值（如 `0`）或小于当前输入长度。  
预防建议：

1. 页面里保持默认 `max_length=800`（推荐）。
2. 如需手调，请设成正整数，且不要太小（建议 >= 256）。
3. 如果你之前在同一会话里填过异常参数，改回留空后再试；不放心就重启一次服务。

### 采样参数填错（如 `top_p=0`）怎么办？

- 现在服务端会自动忽略不合法采样参数并回退默认值（不会因 `top_p=0` 直接崩）。
- 默认值为：
  - Band：`top_k=10, top_p=1.0, temperature=1.0, no_repeat_ngram_size=10, max_length=800`
  - Piano：`top_k=30, top_p=1.0, temperature=1.0, no_repeat_ngram_size=10, max_length=800`
  - Drum：`top_k=20, top_p=1.0, temperature=1.0, no_repeat_ngram_size=0, max_length=800`

---

## 9. 信息来源（本次答复依据）

- 本仓库 README：`Readme.md`
- 本仓库代码：
  - `app.py`
  - `mcp_infer.py`
  - `api/arranger.py`
  - `m2m/hparams/band_arrange/elaborator.yaml`
  - `m2m/hparams/drum_arrange/direct_opd.yaml`
- 原项目演示页（作者网站）：
  - https://www.oulongshen.xyz/automatic_arrangement
