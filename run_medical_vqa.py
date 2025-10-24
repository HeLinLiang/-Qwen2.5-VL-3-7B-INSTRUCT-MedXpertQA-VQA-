#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗VQA系统快速启动脚本
整合配置、数据加载和预测功能
"""

import os
import sys
import json
import argparse
from medical_vqa_qwen import MedicalVQA
from config import MODEL_PATH, DATA_PATH, IMAGE_FOLDER, OUTPUT_PATH, DEVICE

def quick_setup():
    """快速设置检查"""
    print("🔧 快速设置检查...")
    
    # 检查必要的包
    try:
        import torch
        import transformers
        from PIL import Image
        print("✅ 必要的Python包已安装")
    except ImportError as e:
        print(f"❌ 缺少必要的包: {e}")
        print("请运行: pip install torch transformers pillow tqdm")
        return False
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"✅ CUDA可用，GPU数量: {torch.cuda.device_count()}")
        print(f"   当前GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  CUDA不可用，将使用CPU（速度较慢）")
    
    return True

def load_and_preview_data(data_path):
    """加载并预览数据"""
    print(f"\n📊 数据预览: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return None
    
    try:
        samples = []
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 3:  # 只预览前3条
                        break
                    if line.strip():
                        samples.append(json.loads(line))
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                samples = data[:3] if isinstance(data, list) else [data]
        
        print(f"✅ 成功加载数据，预览前几条:")
        for i, sample in enumerate(samples):
            print(f"\n样本 {i+1}:")
            print(f"  ID: {sample.get('id', 'N/A')}")
            print(f"  问题: {sample.get('question', 'N/A')[:100]}...")
            print(f"  选项数: {len(sample.get('options', {}))}")
            print(f"  图像: {sample.get('images', [])}")
            print(f"  正确答案: {sample.get('label', 'N/A')}")
        
        return samples
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def run_single_test(model_path, data_path, image_folder, sample_id=None):
    """运行单个测试"""
    print(f"\n🧪 运行单个测试...")
    
    try:
        # 初始化VQA系统
        vqa = MedicalVQA(model_path)
        print("🔹 正在加载模型...")
        vqa.load_model()
        
        # 加载数据
        samples = vqa.load_data(data_path)
        if not samples:
            print("❌ 没有可用的数据样本")
            return
        
        # 选择测试样本
        test_sample = None
        if sample_id:
            for sample in samples:
                if sample.get("id") == sample_id:
                    test_sample = sample
                    break
            if not test_sample:
                print(f"❌ 未找到ID为 {sample_id} 的样本")
                return
        else:
            test_sample = samples[0]  # 使用第一个样本
        
        print(f"🎯 测试样本ID: {test_sample['id']}")
        
        # 执行预测
        result = vqa.predict_single(test_sample, image_folder)
        
        # 显示结果
        print(f"\n📋 预测结果:")
        print(f"样本ID: {result['id']}")
        if result['success']:
            print(f"问题: {result['question'][:150]}...")
            print(f"预测选项: {result.get('pred_label', 'N/A')} - {result.get('pred_answer', 'N/A')}")
            print(f"真实答案: {result.get('true_label', 'N/A')} - {result.get('true_answer', 'N/A')}")
            print(f"预测结果: {'✅ 正确' if result.get('correct') else '❌ 错误'}")
        else:
            print(f"❌ 预测失败: {result.get('error', '未知错误')}")
        
        return result
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_batch_prediction(model_path, data_path, image_folder, output_path):
    """运行批量预测"""
    print(f"\n🚀 开始批量预测...")
    
    try:
        # 初始化VQA系统
        vqa = MedicalVQA(model_path)
        print("🔹 正在加载模型...")
        vqa.load_model()
        
        # 执行批量预测
        stats = vqa.predict_batch(data_path, image_folder, output_path)
        
        # 显示统计结果
        print(f"\n📊 批量预测完成!")
        print(f"总样本数: {stats['total_samples']}")
        print(f"成功预测: {stats['successful_predictions']}")
        print(f"失败预测: {stats['failed_predictions']}")
        print(f"正确预测: {stats['correct_predictions']}")
        print(f"准确率: {stats['accuracy']:.4f} ({stats['correct_predictions']}/{stats['successful_predictions']})")
        print(f"总耗时: {stats['total_time']:.2f}秒")
        print(f"平均耗时: {stats['avg_time_per_sample']:.2f}秒/样本")
        
        if stats['failed_predictions'] > 0:
            print(f"\n⚠️  有 {stats['failed_predictions']} 个样本预测失败，请查看错误日志")
        
        return stats
        
    except Exception as e:
        print(f"❌ 批量预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="医疗VQA系统快速启动")
    parser.add_argument("--mode", choices=["test", "batch", "preview"], default="test", 
                       help="运行模式: test(单个测试), batch(批量预测), preview(数据预览)")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="模型路径")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="数据文件路径")
    parser.add_argument("--image_folder", type=str, default=IMAGE_FOLDER, help="图像文件夹路径")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="输出文件路径")
    parser.add_argument("--sample_id", type=str, help="指定测试样本ID")
    
    args = parser.parse_args()
    
    print("🏥 医疗VQA系统")
    print("="*50)
    
    # 快速设置检查
    if not quick_setup():
        return
    
    # 检查配置
    if not os.path.exists(args.model_path) or args.model_path == "/path/to/Qwen2.5-VL-3B-Instruct":
        print(f"❌ 请先配置正确的模型路径")
        print(f"当前配置: {args.model_path}")
        print("请修改 config.py 文件中的 MODEL_PATH")
        return
    
    if args.mode == "preview":
        # 数据预览模式
        if not os.path.exists(args.data_path) or args.data_path == "/path/to/your/medical_data.jsonl":
            print(f"❌ 请先配置正确的数据路径")
            print(f"当前配置: {args.data_path}")
            return
        load_and_preview_data(args.data_path)
        
    elif args.mode == "test":
        # 单个测试模式
        if not os.path.exists(args.data_path) or args.data_path == "/path/to/your/medical_data.jsonl":
            print(f"❌ 请先配置正确的数据路径")
            return
        if not os.path.exists(args.image_folder) or args.image_folder == "/path/to/your/images":
            print(f"❌ 请先配置正确的图像文件夹路径")
            return
        
        run_single_test(args.model_path, args.data_path, args.image_folder, args.sample_id)
        
    elif args.mode == "batch":
        # 批量预测模式
        if not os.path.exists(args.data_path) or args.data_path == "/path/to/your/medical_data.jsonl":
            print(f"❌ 请先配置正确的数据路径")
            return
        if not os.path.exists(args.image_folder) or args.image_folder == "/path/to/your/images":
            print(f"❌ 请先配置正确的图像文件夹路径")
            return
        
        run_batch_prediction(args.model_path, args.data_path, args.image_folder, args.output_path)

if __name__ == "__main__":
    main()
