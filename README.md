# VLM Video Action Localization 部署指南

## 快速开始（同一服务器）

### 1. 激活 Conda 环境

```bash
/home/asus/miniconda3/bin/conda activate vlm-action
```

### 2. 下载模型

```bash
cd /home/asus/zhangguangmeng/VLM-Video-Action-Localization-Deploy/minicpm_native
python download_model.py
cd ..
```

模型会自动下载到 `minicpm_native/model_cache/` 目录。

### 3. 准备视频

**注意：** GitHub 仓库不包含视频文件，请自己准备视频放入 `sample_video/` 目录，支持 `.mp4` 格式。

将视频文件放入 `sample_video/` 目录后，服务启动时会自动检测。

### 4. 修改端口为 5003

```bash
sed -i 's/port=5002/port=5003/' app_minicpm.py
```

### 5. 启动服务

```bash
/home/asus/miniconda3/bin/conda run -n vlm-action python app_minicpm.py
```

服务会在 **http://localhost:5003** 启动。

### 6. 访问网页

浏览器打开 `http://localhost:5003` 或 `http://服务器IP:5003`

---

## 文件说明

- `app_minicpm.py` - Flask 网页服务主程序
- `action_descriptions.json` - 动作定义和判定标准
- `minicpm_native/` - MiniCPM 模型推理代码
  - `infer_video.py` - 视频推理核心逻辑
  - `download_model.py` - 模型下载脚本
- `sample_video/` - 示例视频目录
- `results/` - 推理结果输出目录（自动创建）

---

## 常见问题

### Q: 模型下载失败
A: 检查网络连接，或手动从 ModelScope 下载 MiniCPM-V 2.6 模型。

### Q: GPU 内存不足
A: 模型会自动使用 GPU，如显存不足可修改 `minicpm_native/infer_video.py` 使用 CPU。

### Q: 推理速度慢
A: 首次推理需要加载模型，后续会更快。可使用 GPU 加速。

---

## Git 提交

只需提交以下文件到仓库：
- `app_minicpm.py`
- `action_descriptions.json`
- `minicpm_native/` (整个目录)
- `requirements.txt`
- `README.md`

**不要提交：**
- `sample_video/` 中的大视频文件
- `results/` 目录
- `minicpm_native/model_cache/` 模型缓存
