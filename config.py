#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件 - 请根据您的实际环境修改这些路径
"""

import os

# ========== 模型配置 ==========
# QWEN2.5-VL-3B-INSTRUCT模型路径 - 请修改为您的实际路径
#MODEL_PATH = "/home/asus/model/Qwen2.5-VL-3B-Instruct"
MODEL_PATH = "/home/asus/model/Qwen2.5-VL-7B-Instruct"

# ========== 数据配置 ==========
# 数据文件路径（JSON或JSONL格式）
#DATA_PATH = "/home/asus/datasets/MedXpertQA/MM/test.jsonl"
DATA_PATH = "/home/asus/datasets/MedXpertQA/MM/dev.jsonl"


# 图像文件夹路径（包含所有医疗图像）
IMAGE_FOLDER = "/home/asus/datasets/MedXpertQA/images"

# 输出结果路径
OUTPUT_PATH = "/home/asus/results/cursor_medical_vqa_results.jsonl"

# ========== 系统配置 ==========
# 运行设备配置
DEVICE = "auto"  # 可选: "auto", "cuda", "cpu"

# 批处理大小（如果支持批量推理）
BATCH_SIZE = 1

# 最大生成token数
MAX_NEW_TOKENS = 32

# ========== 验证配置 ==========
def validate_config():
    """验证配置是否正确"""
    errors = []
    
    # 检查模型路径
    if not os.path.exists(MODEL_PATH) or MODEL_PATH == "/home/asus/model/Qwen2.5-VL-3B-Instruct":
        errors.append(f"模型路径不存在或未配置: {MODEL_PATH}")
    
    # 检查数据路径（如果已配置）
    if DATA_PATH != "/path/to/your/medical_data.jsonl" and not os.path.exists(DATA_PATH):
        errors.append(f"数据文件不存在: {DATA_PATH}")
    
    # 检查图像文件夹（如果已配置）
    if IMAGE_FOLDER != "/path/to/your/images" and not os.path.exists(IMAGE_FOLDER):
        errors.append(f"图像文件夹不存在: {IMAGE_FOLDER}")
    
    if errors:
        print("❌ 配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        print("\n请修改 config.py 文件中的路径配置")
        return False
    
    print("✅ 配置验证通过")
    return True

# ========== 示例配置 ==========
def get_example_config():
    """获取示例配置（基于您可能的路径）"""
    return {
        "model_path": "/home/asus/model/Qwen2.5-VL-3B-Instruct",
        "data_path": "/home/asus/datasets/medvqa_data.jsonl", 
        "image_folder": "/home/asus/datasets/medvqa_images",
        "output_path": "/home/asus/results/medical_vqa_results.jsonl"
    }

if __name__ == "__main__":
    print("🔧 配置文件检查")
    validate_config()
