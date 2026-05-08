#!/usr/bin/env python3
"""下载 MiniCPM-V-4.5 模型，使用国内镜像 hf-mirror.com"""

import os

# 使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

model_id = "openbmb/MiniCPM-V-4_5"
local_dir = "./MiniCPM-V-4_5"

print(f"开始下载模型: {model_id}")
print(f"保存到: {os.path.abspath(local_dir)}")
print(f"使用镜像: {os.environ['HF_ENDPOINT']}")

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)

print("模型下载完成！")
