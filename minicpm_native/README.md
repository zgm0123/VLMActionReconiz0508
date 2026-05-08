# MiniCPM-V-4.5 原生视频动作定位

## 1. 核心文件

| 文件 | 作用 |
|---|---|
| `infer_video.py` | 主推理脚本。支持长视频自动分段、帧采样、Prompt 构建、JSON 结果提取与整合 |
| `../action_descriptions.json` | 动作详细定义文件（可选加载）。包含每个动作的开始/结束判定标准，通过 `--detailed` 注入 Prompt |

---

## 2. 命令行用法

```bash
python minicpm_native/infer_video.py \
    --video sample_video/ch1_20251120_132545_h264.mp4 \
    --actions 追球,拉球,犬咬 \
    --detailed \
    --segment-duration 5 \
    --output results/output.json
```

### 2.1 参数说明

| 参数 | 是否必填 | 默认值 | 说明 |
|---|---|---|---|
| `--video` | **是** | — | 输入视频文件路径 |
| `--actions` | **是** | — | 要检测的动作列表，逗号分隔。例如：`追球,拉球,犬咬` |
| `--model` | 否 | 自动查找 | 模型本地路径。默认按候选列表自动查找 `MiniCPM-V-4_5` |
| `--fps` | 否 | 3 | 视频采样 fps，决定每秒抽多少帧进入模型 |
| `--detailed` | 否 | false | 加载 `action_descriptions.json` 中的详细定义注入 Prompt |
| `--output` | 否 | 自动保存 | 输出 JSON 文件路径。默认保存到 `results/minicpm_v45/{视频名}.json` |
| `--cpu` | 否 | false | 强制使用 CPU 推理（极慢，仅测试） |
| `--segment-duration` | 否 | 10 | 分段时长（秒）。推荐 **5** 秒，判断更精确 |
| `--start-time` | 否 | — | 手动指定起始时间（秒），与 `--end-time` 配合只跑一段 |
| `--end-time` | 否 | — | 手动指定结束时间（秒），与 `--start-time` 配合只跑一段 |

### 2.2 常用示例

**只跑视频前 5 秒：**
```bash
python minicpm_native/infer_video.py \
    --video sample_video/ch1_20251120_132545_h264.mp4 \
    --actions 追球,拉球,犬咬 \
    --detailed \
    --start-time 0 --end-time 5
```

**跑整个视频，5 秒一段，指定输出路径：**
```bash
python minicpm_native/infer_video.py \
    --video sample_video/ch1_20251120_132545_h264.mp4 \
    --actions 追球,拉球,犬咬 \
    --detailed \
    --segment-duration 5 \
    --output results/my_result.json
```

**使用自定义模型路径：**
```bash
python minicpm_native/infer_video.py \
    --video sample_video/ch1_20251120_132545_h264.mp4 \
    --actions 追球,拉球,犬咬 \
    --model /path/to/MiniCPM-V-4_5 \
    --detailed
```

---

## 3. 视频编码算法 (`encode_video`)

### 2.1 帧采样流程

1. 使用 **decord.VideoReader** 读取视频，获取原始 FPS 和总帧数
2. 根据 `start_time` / `end_time` 参数截取目标时间段
3. **均匀采样**：在截取的时间段内按目标 fps 均匀选取帧
   - 采样公式：`gap = len(frame_range) / n`, `idx = int(i * gap + gap / 2)`
4. **显存保护**：强制限制总帧数不超过 `DEFAULT_MAX_TOTAL_FRAMES=24`
   - 3090 24GB 显存下，MiniCPM-V-4.5 的 visual encoder 最多支持约 24 帧同时进入，超过会 OOM

### 2.2 Temporal IDs 生成

MiniCPM-V-4.5 需要为每帧提供 temporal id，用于内部 3D-Resampler 处理时序信息。

生成逻辑：
- `TIME_SCALE = 0.1`，时间分辨率为 0.1 秒
- 生成时间格点：`scale = np.arange(0, segment_duration, 0.1)`
- 每帧的实际时间戳：`frame_ts = (frame_idx - start_frame) / fps`
- 使用 **scipy.spatial.cKDTree** 将每帧时间戳映射到最近的时间格点
- 最终 temporal id：`frame_ts_id = nearest_scale / TIME_SCALE`（整数）

```python
scale = np.arange(0, segment_duration, TIME_SCALE)      # [0, 0.1, 0.2, ...]
frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
```

---

## 3. Prompt 构建 (`build_prompt`)

### 3.1 三步式结构

```
【步骤1】整体描述视频内容（狗在做什么、场景是什么）
【步骤2】总体判定：视频中狗的行为是否属于以下动作？
【步骤3】只有在确实出现相关动作时，才给出开始和结束时间（单位：秒）
```

### 3.2 动作定义注入

- 默认模式：使用内置简短描述
- `--detailed` 模式：从 `action_descriptions.json` 加载详细判定标准，包含：
  - 【开始判定标准】
  - 【结束判定标准】

### 3.3 关键设计经验

| 经验 | 原因 |
|---|---|
| **不要要求"逐秒分析"** | 模型只收到 24 帧，无法真正逐秒推理，强行要求会导致时间幻觉（编造不存在的时间点） |
| **三步式足够** | 整体描述 → 总体判定 → 时间定位。过多的中间步骤会稀释模型对边界的注意力 |
| **约束时间格式** | 如果视频画面有时间戳水印，模型会误把水印时间填进 JSON，需要 prompt 中明确"输出段内相对时间" |

---

## 4. 分段推理逻辑

### 4.1 自动分段

```
if duration <= segment_duration:
    直接推理整段
else:
    按 segment_duration 切分为多段，循环独立推理
```

- 每段调用 `run_single_segment`，独立提取帧、构建 prompt、调用模型
- 时间偏移修正：将模型输出的**段内相对时间**加上 `seg_start`，转为视频绝对时间

### 4.2 分段时长的选择

| 时长 | 效果 | 适用场景 |
|---|---|---|
| 10秒 | 容易过分类（把非目标动作判为目标动作），也容易漏检 | 不推荐 |
| **5秒** | **判断更准**，模型不容易被"总体判定"带偏，漏检和误检都更少 | **推荐** |

**原因**：片段越短，画面信息越少，模型越难"编"出不存在的行为，只能如实描述看到的画面。

---

## 5. 模型调用

### 5.1 加载方式

```python
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    attn_implementation='sdpa',
    torch_dtype=torch.bfloat16,
)
```

- `trust_remote_code=True`：加载 MiniCPM-V-4.5 自定义模型代码
- `sdpa`：使用 scaled dot-product attention，比 flash_attention 兼容性更好

### 5.2 推理调用

```python
model.chat(
    msgs=[{'role': 'user', 'content': frames + [prompt]}],
    tokenizer=tokenizer,
    use_image_id=False,
    max_slice_nums=1,
    temporal_ids=temporal_ids,
    do_sample=False
)
```

参数说明：
- `use_image_id=False`：视频模式不使用 image id
- `max_slice_nums=1`：限制图像切片数量，降低显存占用
- `temporal_ids`：每帧对应的时间 id 列表
- `do_sample=False`：**贪婪解码**，保证每次运行结果完全一致

### 5.3 为什么用贪婪解码 (`do_sample=False`)

默认采样 (`temperature=0.7, top_p=0.8`) 每次运行结果不同，导致：
- 同一段视频两次运行可能给出完全不同的动作判定
- 时间定位波动大，无法复现

贪婪解码虽然偶尔会出现固定模式的时间编造（如之前出现的 83.3s 超出范围问题），但**在 prompt 正确的前提下**，结果是稳定且可复现的。对于动作定位这种需要精确时间边界的任务，稳定性比多样性更重要。

---

## 6. 结果提取与整合

### 6.1 JSON 提取 (`extract_json`)

1. 优先匹配 ` ```json {...} ``` ` 代码块
2.  fallback 匹配第一个 `{...}` 片段
3. `json.loads` 解析，失败则返回 None

### 6.2 输出结构

```json
{
  "video": "/path/to/video.mp4",
  "duration": 81.03,
  "segment_duration": 5,
  "actions": ["追球", "拉球", "犬咬"],
  "segment_results": [
    {"追球": {"start": null, "end": null}, ...},
    ...
  ],
  "raw_outputs": "每段的完整模型输出文本"
}
```

**不做合并**：每段结果独立保存，由调用方根据业务需求自行合并。

---

## 7. 关键超参数

| 参数 | 值 | 说明 |
|---|---|---|
| `DEFAULT_MAX_TOTAL_FRAMES` | 24 | 3090 24GB 显存安全上限 |
| `MAX_NUM_FRAMES` | 180 | MiniCPM-V-4.5 模型内部限制 |
| `MAX_NUM_PACKING` | 3 | 3D-Resampler 最大 packing 数 |
| `TIME_SCALE` | 0.1 | Temporal id 时间分辨率（秒） |
| `segment_duration` | 5（推荐）/ 10（默认） | 分段时长 |

---

## 8. 环境要求

```
torch==2.4.0+cu121
transformers==4.51.3
decord
scipy
Pillow
numpy
```

**注意**：
- transformers 5.7.0 有 `all_tied_weights_keys` 兼容性问题，需要降级到 4.51.3
- 脚本中有一个 monkey-patch 处理 transformers 高版本的兼容性问题（行 22-31）
