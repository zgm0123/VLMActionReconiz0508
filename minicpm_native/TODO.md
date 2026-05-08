# MiniCPM-V-4.5 原生视频接入 - 待办事项

## 1. 模型下载
- [x] 确认磁盘空间足够（约 18GB）
- [x] 运行 `download_model.py` 从 hf-mirror.com 拉取 `openbmb/MiniCPM-V-4_5`
- [ ] 备用：如镜像失败，换 ModelScope 源 `OpenBMB/MiniCPM-V-4_5`

## 2. 环境修复
- [x] 解决 CUDA 兼容问题（torch 2.11+cu130 → 降级为 torch 2.4.0+cu121）
- [x] 验证 `torch.cuda.is_available()` 返回 True
- [x] 解决 transformers 5.7.0 与 MiniCPM-V-4.5 代码的兼容性（降级到 4.51.3）

## 3. 推理脚本开发
- [x] 阅读 MiniCPM-V-4.5 官方推理示例（视频 chat 接口）
- [x] 编写 `infer_video.py`：
  - [x] 输入：视频文件路径 + 动作列表
  - [x] 调用方式：基于 decord 抽帧 + 模型内部 3D-Resampler 处理
  - [x] 输出：JSON 格式的动作时间段
  - [x] 支持自定义 prompt（注入 action_descriptions.json 详细定义）
- [x] 3090 24GB 显存保护：默认限制最大 24 帧，避免 OOM
- [ ] 加入视频分段逻辑（如视频过长，自动按 10s 切片逐段分析）

## 4. 效果验证
- [x] 用 `ch1_20251120_132545_h264.mp4` 跑通第一遍推理
- [ ] 对比 Ollama `coarse_locate.py` 的结果差异
- [ ] 测试 4.5 模型对长视频（>30s）的稳定性（当前 81s 视频被降采样到 24 帧）
- [ ] 调优帧数上限，在显存和精度之间找平衡点

## 5. 项目集成
- [ ] 将原生推理封装为可复用模块（供 example.py 主流程调用）
- [ ] 保留 Ollama 分支作为 fallback（低显存/CPU 环境）
- [ ] 更新顶层 README，说明两种后端的使用方式

---

## 当前状态
- **模型**：`MiniCPM-V-4_5` 已下载（~16.5GB，4 shards）
- **CUDA**：可用（torch 2.4.0+cu121，transformers 4.51.3）
- **推理**：`infer_video.py` 已跑通，81s 视频在 24 帧限制下成功输出 JSON
- **已知限制**：3090 24GB 显存最多支持约 24 帧同时进入 visual encoder

## 下一步行动建议
1. 跑更多视频对比 Ollama 结果，评估 4.5 精度优势
2. 如需处理更长视频，实现分段推理逻辑（每段 24 帧，滑动窗口）
3. 封装为模块，接入 example.py 主流程
