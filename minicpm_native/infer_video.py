#!/usr/bin/env python3
"""
MiniCPM-V-4.5 原生视频推理脚本
支持长视频自动分段（默认10秒一段），模型只加载一次，结果自动整合。
"""

import argparse
import json
import math
import os
import re
import sys

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from scipy.spatial import cKDTree
from transformers import AutoModel, AutoTokenizer

# Patch: transformers 5.7.0 要求 all_tied_weights_keys，但 MiniCPM-V-4.5 旧代码只有 _tied_weights_keys
import transformers.modeling_utils as _modeling_utils
if hasattr(_modeling_utils.PreTrainedModel, "_move_missing_keys_from_meta_to_device"):
    _original_move = _modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device

    def _patched_move(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _original_move(self, *args, **kwargs)

    _modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move

# ========== 视频编码（来自官方 README） ==========
MAX_NUM_FRAMES = 180
MAX_NUM_PACKING = 3

# 3090 24GB 显存安全上限（visual encoder 显存瓶颈）
DEFAULT_MAX_TOTAL_FRAMES = 24
TIME_SCALE = 0.1


def map_to_nearest_scale(values, scale):
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]


def group_array(arr, size):
    return [arr[i:i + size] for i in range(0, len(arr), size)]


def get_video_duration(video_path):
    """获取视频时长（秒）"""
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    return len(vr) / fps


def encode_video(video_path, choose_fps=3, force_packing=None,
                 max_total_frames=DEFAULT_MAX_TOTAL_FRAMES,
                 start_time=0.0, end_time=None):
    """从视频中提取帧，支持指定时间段"""
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    video_duration = len(vr) / fps

    if end_time is None:
        end_time = video_duration

    segment_duration = end_time - start_time
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), len(vr))
    frame_idx_range = list(range(start_frame, end_frame))

    if len(frame_idx_range) == 0:
        return [], [], segment_duration

    if choose_fps * int(segment_duration) <= MAX_NUM_FRAMES:
        packing_nums = 1
        choose_frames = round(min(choose_fps, round(fps)) * min(MAX_NUM_FRAMES, segment_duration))
    else:
        packing_nums = math.ceil(segment_duration * choose_fps / MAX_NUM_FRAMES)
        if packing_nums <= MAX_NUM_PACKING:
            choose_frames = round(segment_duration * choose_fps)
        else:
            choose_frames = round(MAX_NUM_FRAMES * MAX_NUM_PACKING)
            packing_nums = MAX_NUM_PACKING

    # 显存保护：强制限制总帧数
    if choose_frames > max_total_frames:
        choose_frames = max_total_frames
        packing_nums = math.ceil(choose_frames / MAX_NUM_FRAMES)
        if packing_nums < 1:
            packing_nums = 1

    choose_frames = min(choose_frames, len(frame_idx_range))
    if choose_frames <= 0:
        return [], [], segment_duration
    frame_idx = np.array(uniform_sample(frame_idx_range, choose_frames))

    if force_packing:
        packing_nums = min(force_packing, MAX_NUM_PACKING)

    print(f"[*] 段内时长: {segment_duration:.1f}s, 原始FPS: {fps:.1f}")
    print(f"[*] 采样帧数: {len(frame_idx)}, packing_nums: {packing_nums}")

    frames = vr.get_batch(frame_idx).asnumpy()

    # temporal_ids 使用段内相对时间
    frame_idx_ts = (frame_idx - start_frame) / fps
    scale = np.arange(0, segment_duration, TIME_SCALE)
    frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
    frame_ts_id = frame_ts_id.astype(np.int32)

    assert len(frames) == len(frame_ts_id)

    frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]
    frame_ts_id_group = group_array(frame_ts_id, packing_nums)

    return frames, frame_ts_id_group, segment_duration


# ========== Prompt 构建 ==========
def load_action_descriptions():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    desc_path = os.path.join(project_root, "action_descriptions.json")
    if os.path.exists(desc_path):
        with open(desc_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def build_prompt(actions, duration, use_detailed=False):
    action_desc = load_action_descriptions() if use_detailed else {}

    action_lines = []
    for i, action in enumerate(actions, 1):
        if use_detailed and action in action_desc:
            desc = action_desc[action]
            start_desc = desc.get('start', '未定义')
            end_desc = desc.get('end', '未定义')
            action_lines.append(
                f"{i}. {action}\n"
                f"   【开始判定标准】{start_desc}\n"
                f"   【结束判定标准】{end_desc}"
            )
        else:
            default_desc = {
                "追球": "狗奔跑追逐球/物体的动作",
                "拉球": "狗咬住球后，身体向后发力拉扯的动作",
                "咬球": "狗张嘴试图咬住球的动作",
                "犬追物体": "狗奔跑追逐物体的动作",
                "犬拉": "狗咬住物品后身体向后发力拉扯的动作",
                "犬尝试咬物体": "狗张嘴试图咬住物体的动作",
                "犬咬住物体拉动": "狗嘴里含有物体并主动拉拽的动作",
                "犬咬住物体持续咬住": "物体在狗嘴里被闭合包住的持住状态",
                "犬咬": "狗咬住训导员手中/牵引的物体，嘴闭合包住，正在咀嚼或持续含住的状态",
            }
            desc = default_desc.get(action, "")
            if desc:
                action_lines.append(f"{i}. {action}：{desc}")
            else:
                action_lines.append(f"{i}. {action}")

    actions_text = "\n".join(action_lines)

    json_examples = [f'"{action}": {{"start": X, "end": Y}}' for action in actions]
    json_example_str = ", ".join(json_examples)

    prompt = (
        f"以下是一个约{duration:.0f}秒的狗视频片段。\n\n"
        f"你必须严格按照以下三个步骤完成分析，每一步都必须用文字写出来，不能只输出JSON：\n\n"
        f"【步骤1】先观察并描述：视频中的球/物体在哪里？（必须回答）\n"
        f"- 是在狗的嘴里（狗已经叼住）？\n"
        f"- 是在地上（狗尚未接触）？\n"
        f"- 是在空中/被抛出（狗正在追逐）？\n"
        f"- 还是被训导员拿在手里？\n"
        f"- 或者视频中完全没有看到球/物体？\n"
        f"请明确说明球的位置（如果没看到球也要明确说\"没有看到球\"），并描述狗在做什么。这一步必须写文字描述，不能跳过。\n\n"
        f"【步骤2】总体判定：视频中狗的行为是否属于以下动作？（必须回答）\n"
        f"{actions_text}\n"
        f"判断规则（必须严格遵守，违反任何一条都不能判定为\"出现\"）：\n"
        f"- 判定某个动作\"出现\"的前提是：步骤1的描述中必须有直接证据支撑，严禁根据推测或想象判定\n"
        f"- 严禁把\"准备阶段\"、\"试图做\"、\"快要做了\"算作动作出现。只有动作确实发生了才能判定为\"出现\"\n"
        f"- 判定标准中的每条要求都是\"必须同时满足\"，不是\"满足大部分就行\"，严禁放宽标准\n"
        f"- 步骤2的判定必须与步骤1的描述完全一致，严禁引入步骤1中没有提到的新信息\n"
        f"- 如果步骤1的描述中有任何内容与判定标准矛盾（例如步骤1写了\"没有快速奔跑\"或\"狗尚未接触球\"），该动作必须判为\"未出现\"，不能因为其他条件符合就放宽\n"
        f"- 追球、拉球、犬咬是三个完全不同的动作，必须分别独立判定，不能因为狗追了球就认为它也拉了或咬了\n"
        f"- 如果球已经在狗嘴里，则属于\"犬咬\"（狗嘴里叼着球/物品的状态），绝不可能是\"追球\"\n"
        f"- 如果步骤1说球在地上、狗尚未接触、狗没有叼住球，则\"犬咬\"绝对不可能出现\n"
        f"- 追球的必要条件（缺一不可）：球不在嘴里 + 狗在奔跑/快速移动 + 兴奋状态。慢悠悠走向球、只是看着球、试图接近、准备咬，一律不算追球\n"
        f"- 拉球必须有\"狗咬住球后身体向后发力拉扯\"的直接画面证据，仅仅\"狗嘴里有球\"不算拉球\n"
        f"- 如果步骤1没有提到某个动作的关键特征，该动作必须判为\"未出现\"\n"
        f"- 如果狗的行为完全不属于上述动作（例如只是站立、休息、嗅闻地面、缓慢走动等），"
        f"  请明确说明\"不属于上述动作\"，并描述狗实际在做什么。\n"
        f"- 如果上述动作都未出现，请明确说明\"未出现\"。\n"
        f"这一步必须写文字描述，说明你为什么这样判定，不能跳过。\n\n"
        f"【步骤3】只有在确实出现相关动作时，才给出开始和结束时间（单位：秒）。\n"
        f"- 严格限制：时间必须从本片段开头开始算（0秒起），必须在 0 秒到 {duration:.0f} 秒之间，绝对禁止超出这个范围\n"
        f"- 视频画面上可能有时间戳水印（如15:03:38），请完全忽略，不要参考画面上的时间戳\n"
        f"- 你必须根据视频帧的实际内容，判断动作从哪一秒开始、到哪一秒结束。例如：狗从第2秒开始奔跑，第3.5秒停下，就输出2到3.5，绝不能输出0到5\n"
        f"- 在输出JSON之前，先用文字写出每个动作的开始时间和结束时间（如\"追球：2秒到3.5秒\"），然后再输出JSON\n"
        f"- 注意区分：追是奔跑状态（球不在嘴里），拉是咬住后向后拽，咬是球已经在嘴里叼住\n"
        f"- 各动作没有固定先后顺序，请根据画面实际内容独立判断\n"
        f"- 如果某个动作没有出现，明确说明\"未出现\"\n\n"
        f"最后，请输出一个JSON总结（必须包含在回答中）：\n"
        f"{{{json_example_str}}}\n"
        f"未出现的动作，start和end都填null。"
    )
    return prompt


def extract_json(text):
    code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def run_single_segment(model, tokenizer, video_path, actions, seg_start, seg_end,
                       fps, use_detailed, device):
    """运行单段推理，返回修正后的 parsed_result 和 raw_output"""
    frames, temporal_ids, seg_duration = encode_video(
        video_path, choose_fps=fps,
        start_time=seg_start, end_time=seg_end
    )

    if len(frames) == 0:
        print(f"[!] 段 {seg_start:.1f}s-{seg_end:.1f}s 未能提取到帧，跳过")
        return None, ""

    prompt = build_prompt(actions, seg_duration, use_detailed=use_detailed)
    msgs = [{'role': 'user', 'content': frames + [prompt]}]

    answer = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        use_image_id=False,
        max_slice_nums=1,
        temporal_ids=temporal_ids,
        do_sample=False
    )

    result_text = answer if isinstance(answer, str) else str(answer)
    parsed = extract_json(result_text)

    # 修正时间偏移
    if parsed:
        for action in actions:
            if action in parsed and isinstance(parsed[action], dict):
                if parsed[action].get('start') is not None:
                    parsed[action]['start'] = round(parsed[action]['start'] + seg_start, 1)
                if parsed[action].get('end') is not None:
                    parsed[action]['end'] = round(parsed[action]['end'] + seg_start, 1)

    return parsed, result_text




# ========== 主流程 ==========
def main():
    parser = argparse.ArgumentParser(description="MiniCPM-V-4.5 视频动作分析（支持长视频自动分段）")
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument("--actions", required=True, help="动作列表，逗号分隔")
    parser.add_argument("--model", default=None, help="模型本地路径（默认自动查找）")
    parser.add_argument("--fps", type=int, default=3, help="视频采样fps (默认3)")
    parser.add_argument("--detailed", action="store_true", help="加载 action_descriptions.json 详细定义")
    parser.add_argument("--output", default=None, help="输出 JSON 文件路径")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU（极慢，仅测试）")
    parser.add_argument("--segment-duration", type=float, default=10,
                        help="分段时长（秒），超过则自动分段。默认10")
    parser.add_argument("--start-time", type=float, default=None,
                        help="手动指定起始时间（秒），与 --end-time 配合只跑一段")
    parser.add_argument("--end-time", type=float, default=None,
                        help="手动指定结束时间（秒），与 --start-time 配合只跑一段")
    args = parser.parse_args()

    actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    if not actions:
        print("[错误] 请至少指定一个动作")
        sys.exit(1)

    # 自动查找模型路径
    model_path = args.model
    if model_path is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), "MiniCPM-V-4_5"),
            os.path.join(os.path.dirname(__file__), "..", "coarse_locator", "MiniCPM-V-4_5"),
            os.path.join(os.path.dirname(__file__), "..", "MiniCPM-V-4_5"),
        ]
        for c in candidates:
            c = os.path.abspath(c)
            if os.path.exists(os.path.join(c, "config.json")):
                model_path = c
                break

    if not model_path or not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"[错误] 找不到模型配置文件，请通过 --model 指定路径")
        sys.exit(1)

    # 获取视频时长
    duration = get_video_duration(args.video)

    print("=" * 60)
    print("MiniCPM-V-4.5 原生视频推理")
    print("=" * 60)
    print(f"[*] 视频: {args.video}")
    print(f"[*] 时长: {duration:.1f}s")
    print(f"[*] 动作: {actions}")
    print(f"[*] 模型: {model_path}")
    print(f"[*] 采样: {args.fps} fps")
    print(f"[*] 分段: {args.segment_duration}s")
    print(f"[*] 详细定义: {'是' if args.detailed else '否'}")
    print(f"[*] 设备: {'CPU' if args.cpu else 'CUDA (自动)'}")

    # 加载模型（只加载一次）
    print(f"\n[*] 正在加载模型...")
    device = "cpu" if args.cpu else "cuda"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False
    )
    model = model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"[*] 模型加载完成，设备: {device}")

    # 判断是否手动指定时间段
    if args.start_time is not None and args.end_time is not None:
        print(f"\n[*] 手动指定时间段: {args.start_time:.1f}s - {args.end_time:.1f}s")
        parsed, raw = run_single_segment(
            model, tokenizer, args.video, actions,
            args.start_time, args.end_time, args.fps, args.detailed, device
        )
        if raw:
            print(f"[*] 模型输出:\n{raw}\n")
        if parsed:
            print(f"[*] 结果: {json.dumps(parsed, ensure_ascii=False)}")
        all_raw = [raw] if raw else []
        all_parsed = [parsed] if parsed else []
    elif duration <= args.segment_duration:
        print(f"\n[*] 视频时长 {duration:.1f}s <= 分段阈值，直接推理...")
        parsed, raw = run_single_segment(
            model, tokenizer, args.video, actions,
            0, duration, args.fps, args.detailed, device
        )
        all_raw = [raw] if raw else []
        all_parsed = [parsed] if parsed else []
    else:
        num_segments = math.ceil(duration / args.segment_duration)
        print(f"\n[*] 视频时长 {duration:.1f}s > 分段阈值，自动分为 {num_segments} 段推理...")

        all_raw = []
        all_parsed = []
        for i in range(num_segments):
            seg_start = i * args.segment_duration
            seg_end = min((i + 1) * args.segment_duration, duration)
            print(f"\n{'='*60}")
            print(f"[*] 正在处理第 {i+1}/{num_segments} 段: {seg_start:.1f}s - {seg_end:.1f}s")
            print(f"{'='*60}")

            parsed, raw = run_single_segment(
                model, tokenizer, args.video, actions,
                seg_start, seg_end, args.fps, args.detailed, device
            )

            if raw:
                all_raw.append(f"=== 段 {i+1} ({seg_start:.1f}s-{seg_end:.1f}s) ===\n{raw}")
                print(f"[*] 模型输出:\n{raw}\n")
            if parsed:
                all_parsed.append(parsed)
                print(f"[*] 段结果: {json.dumps(parsed, ensure_ascii=False)}")

    # 输出各段结果（不做合并，直接输出每段原始结果）
    print("\n" + "=" * 60)
    print("各段结果:")
    print("=" * 60)
    for i, p in enumerate(all_parsed):
        print(f"段 {i+1}: {json.dumps(p, ensure_ascii=False)}")

    # 保存结果
    output_data = {
        "video": os.path.abspath(args.video),
        "model": model_path,
        "fps": args.fps,
        "duration": round(duration, 2),
        "segment_duration": args.segment_duration,
        "actions": actions,
        "use_detailed": args.detailed,
        "raw_outputs": "\n\n".join(all_raw),
        "segment_results": all_parsed
    }

    if args.output:
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n[*] 结果已保存: {out_path}")
    else:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        default_out = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "minicpm_v45", f"{video_name}_v45.json"
        )
        os.makedirs(os.path.dirname(default_out), exist_ok=True)
        with open(default_out, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n[*] 结果已保存: {default_out}")

    print("\n[*] 完成")


if __name__ == "__main__":
    main()
