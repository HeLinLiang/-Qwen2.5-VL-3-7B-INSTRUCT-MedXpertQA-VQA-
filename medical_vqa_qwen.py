#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于QWEN2.5-VL-3B-INSTRUCT的医疗视觉问答系统
适配医疗数据格式，支持单个问题和批量处理
"""

import os
import json
import time
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import Dict, List, Tuple, Optional


class MedicalVQA:
    def __init__(self, model_path: str, device: str = None):
        """
        初始化医疗VQA系统
        
        Args:
            model_path: QWEN2.5-VL-3B模型路径
            device: 运行设备，None为自动检测
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        
    def load_model(self):
        """加载QWEN2.5-VL模型和处理器"""
        print(f"🔹 正在加载QWEN2.5-VL模型: {self.model_path}")
        print(f"🔹 使用设备: {self.device}")
        
        try:
            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            # 加载模型
            try:
                # 尝试使用量化配置以节省显存
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_path,
                    quantization_config=bnb_config if self.device == "cuda" else None,
                    dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
            except (ImportError, ValueError, Exception) as e:
                print(f"⚠️  量化加载失败，使用标准加载: {e}")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_path,
                    dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
                
            self.model.eval()
            
            # 获取图像标记 - 直接使用processor的属性
            try:
                self.image_token = getattr(self.processor, "image_token", "<|image_pad|>")
                self.image_token_id = getattr(self.processor, "image_token_id", None)
                
                # 如果processor没有直接提供image_token_id，尝试转换
                if self.image_token_id is None:
                    self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.image_token)
                
                if self.image_token_id is None or self.image_token_id == self.processor.tokenizer.unk_token_id:
                    print(f"⚠️  无法获取有效的图像标记ID，将跳过标记验证")
                else:
                    print(f"✅ 图像标记: '{self.image_token}' (ID: {self.image_token_id})")
                    
            except Exception as e:
                print(f"⚠️  图像标记获取失败: {e}")
                self.image_token_id = None
                
            print(f"✅ 模型加载成功!")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    def load_images(self, image_names: List[str], image_folder: str) -> Tuple[List[Image.Image], List[str], List[str]]:
        """
        加载图像文件并进行预处理
        
        Args:
            image_names: 图像文件名列表
            image_folder: 图像文件夹路径
            
        Returns:
            图像对象列表, 路径列表, 错误信息列表
        """
        images = []
        paths = []
        errors = []
        target_size = (512, 512)  # 统一图像尺寸
        
        for name in image_names:
            path = os.path.join(image_folder, name)
            if not os.path.exists(path):
                errors.append(f"图像不存在: {path}")
                continue
                
            try:
                img = Image.open(path).convert("RGB")
                # 图像预处理：统一尺寸
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                new_img = Image.new("RGB", target_size, (255, 255, 255))
                new_img.paste(img, ((target_size[0]-img.size[0])//2, (target_size[1]-img.size[1])//2))
                images.append(new_img)
                paths.append(path)
            except Exception as e:
                errors.append(f"加载失败: {path} - {str(e)}")
                
        return images, paths, errors

    def build_vqa_prompt(self, question: str, options: Dict, images: List, medical_task: str = "", body_system: str = "") -> str:
        """
        构建医疗VQA的提示词
        
        Args:
            question: 医疗问题
            options: 选项字典 {"A": "选项内容", "B": "选项内容", ...}
            images: 图像列表
            medical_task: 医疗任务类型
            body_system: 身体系统
            
        Returns:
            格式化的提示词
        """
        # 生成图像标记（每个图像对应一个标记，与工作代码保持一致）
        image_tokens_str = ""
        if hasattr(self, 'image_token') and self.image_token and images:
            image_tokens_str = "\n".join([self.image_token for _ in range(len(images))])
        
        # 格式化选项
        options_str = "\n".join([f"  {k}. {v}" for k, v in options.items()])
        
        # 构建医疗专业提示词（与工作代码格式保持一致）
        prompt = f"""{image_tokens_str}
# 医疗图像问答任务
医疗任务: {medical_task if medical_task else "医疗诊断"}
涉及系统: {body_system if body_system else "全身各系统"}
问题: {question}
选项：
{options_str}

# 输出要求
1. 仔细分析图像中的关键信息
2. 结合病例描述的临床症状和体征
3. 基于医学知识和经验做出判断
4. 只回答选项字母（如A、B、C、D、E），不要添加其他内容

答案:"""

        return prompt

    def build_multimodal_input(self, images: List[Image.Image], prompt: str) -> List:
        """
        构建多模态输入序列
        
        Args:
            images: 图像列表
            prompt: 文本提示词
            
        Returns:
            多模态输入序列
        """
        input_sequence = []
        
        # 添加图像
        for img in images:
            input_sequence.append(img)
        
        # 添加文本
        input_sequence.append(prompt)
        
        return input_sequence

    def predict_single(self, data_item: Dict, image_folder: str) -> Dict:
        """
        预测单个医疗VQA样本
        
        Args:
            data_item: 数据样本字典
            image_folder: 图像文件夹路径
            
        Returns:
            预测结果字典
        """
        if self.model is None or self.processor is None:
            raise ValueError("模型未加载，请先调用load_model()方法")
            
        sample_id = data_item.get("id", "unknown")
        question = data_item.get("question", "")
        image_names = data_item.get("images", [])
        options = data_item.get("options", {})
        true_label = data_item.get("label", "")
        medical_task = data_item.get("medical_task", "")
        body_system = data_item.get("body_system", "")
        
        # 数据验证
        if not question:
            return {
                "id": sample_id,
                "error": "问题为空",
                "success": False
            }
            
        if not image_names:
            return {
                "id": sample_id,
                "error": "无图像",
                "success": False
            }
            
        # 加载图像
        images, img_paths, img_errors = self.load_images(image_names, image_folder)
        if img_errors:
            return {
                "id": sample_id,
                "error": f"图像加载错误: {'; '.join(img_errors)}",
                "success": False,
                "image_errors": img_errors
            }
            
        # 构建提示词
        prompt = self.build_vqa_prompt(question, options, images, medical_task, body_system)
        
        try:
            # 处理输入 - 使用separate参数方式（已验证可用）
            inputs = self.processor(
                text=prompt,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32)
            
            # 验证输入处理成功
            if hasattr(self, 'image_token_id') and self.image_token_id is not None:
                try:
                    input_ids = inputs["input_ids"][0].cpu().numpy()
                    token_count = (input_ids == self.image_token_id).sum()
                    expected_tokens = len(images)
                    print(f"🔍 输入处理成功: 检测到 {token_count} 个图像标记, {len(images)} 个图像")
                    # 注意：QWEN2.5-VL可能将1个图像分解成多个特征标记，这是正常的
                except Exception as e:
                    print(f"🔍 输入验证失败: {e}")
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    temperature=0.0
                )
            
            # 解码结果 - 使用tokenizer直接解码
            pred_raw = self.processor.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            ).strip()
            
            # 提取预测选项
            pred_label = self.extract_option(pred_raw, options)
            is_correct = pred_label.upper() == true_label.upper()
            
            # 获取选项内容
            pred_answer = options.get(pred_label.upper(), pred_raw)
            true_answer = options.get(true_label.upper(), "")
            
            return {
                "id": sample_id,
                "question": question,
                "images": img_paths,
                "prompt": prompt,
                "pred_raw": pred_raw,
                "pred_label": pred_label.upper(),
                "pred_answer": pred_answer,
                "true_label": true_label.upper(),
                "true_answer": true_answer,
                "correct": is_correct,
                "success": True
            }
            
        except Exception as e:
            return {
                "id": sample_id,
                "error": f"推理失败: {str(e)}",
                "success": False
            }

    def extract_option(self, raw_text: str, options: Dict) -> str:
        """
        从原始输出中提取选项标识符
        
        Args:
            raw_text: 模型原始输出
            options: 选项字典
            
        Returns:
            提取的选项标识符
        """
        raw_text = raw_text.upper().strip()
        
        # 直接匹配选项字母
        for option in options.keys():
            if option in raw_text:
                return option
                
        # 如果没找到，返回第一个找到的选项字母
        import re
        matches = re.findall(r'[A-E]', raw_text)
        if matches:
            return matches[0]
            
        # 默认返回原始文本的最后一个字符（如果是选项字母）
        if raw_text and raw_text[-1] in options:
            return raw_text[-1]
            
        return ""

    def predict_batch(self, data_path: str, image_folder: str, output_path: str = None) -> Dict:
        """
        批量预测医疗VQA数据
        
        Args:
            data_path: 数据文件路径（JSON或JSONL格式）
            image_folder: 图像文件夹路径
            output_path: 结果保存路径
            
        Returns:
            统计结果
        """
        print(f"🔹 开始批量预测: {data_path}")
        
        # 加载数据
        samples = self.load_data(data_path)
        print(f"✅ 加载样本数: {len(samples)}")
        
        results = []
        errors = []
        correct_count = 0
        
        start_time = time.time()
        
        for i, sample in enumerate(tqdm(samples, desc="预测中")):
            result = self.predict_single(sample, image_folder)
            
            if result["success"]:
                results.append(result)
                if result.get("correct", False):
                    correct_count += 1
            else:
                errors.append(result)
                print(f"❌ 样本 {result['id']}: {result.get('error', '未知错误')}")
        
        total_time = time.time() - start_time
        
        # 保存结果
        if output_path:
            self.save_results(results, errors, output_path)
        
        # 统计信息
        stats = {
            "total_samples": len(samples),
            "successful_predictions": len(results),
            "failed_predictions": len(errors),
            "correct_predictions": correct_count,
            "accuracy": correct_count / len(results) if results else 0,
            "total_time": total_time,
            "avg_time_per_sample": total_time / len(samples) if samples else 0
        }
        
        return stats

    def load_data(self, data_path: str) -> List[Dict]:
        """
        加载数据文件
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            数据样本列表
        """
        samples = []
        
        if data_path.endswith('.jsonl'):
            # JSONL格式
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        else:
            # JSON格式
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                else:
                    samples = [data]
        
        return samples

    def save_results(self, results: List[Dict], errors: List[Dict], output_path: str):
        """
        保存预测结果
        
        Args:
            results: 成功预测结果列表
            errors: 错误信息列表
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存主要结果
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # 保存错误日志
        error_path = output_path.replace('.jsonl', '_errors.jsonl')
        with open(error_path, 'w', encoding='utf-8') as f:
            for error in errors:
                f.write(json.dumps(error, ensure_ascii=False) + '\n')
        
        print(f"✅ 结果已保存: {output_path}")
        print(f"✅ 错误日志已保存: {error_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于QWEN2.5-VL的医疗VQA系统")
    parser.add_argument("--model_path", type=str, required=True, help="QWEN2.5-VL模型路径")
    parser.add_argument("--data_path", type=str, help="数据文件路径")
    parser.add_argument("--image_folder", type=str, help="图像文件夹路径")
    parser.add_argument("--output_path", type=str, default="medical_vqa_results.jsonl", help="输出文件路径")
    parser.add_argument("--single_id", type=str, help="处理单个样本ID")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], default="auto", help="运行设备")
    
    args = parser.parse_args()
    
    # 初始化VQA系统
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vqa_system = MedicalVQA(args.model_path, device)
    vqa_system.load_model()
    
    if args.single_id and args.data_path and args.image_folder:
        # 处理单个样本
        samples = vqa_system.load_data(args.data_path)
        target_sample = None
        for sample in samples:
            if sample.get("id") == args.single_id:
                target_sample = sample
                break
        
        if target_sample:
            result = vqa_system.predict_single(target_sample, args.image_folder)
            print("\n=== 单样本预测结果 ===")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"❌ 未找到ID为 {args.single_id} 的样本")
    
    elif args.data_path and args.image_folder:
        # 批量处理
        stats = vqa_system.predict_batch(args.data_path, args.image_folder, args.output_path)
        
        print("\n" + "="*60)
        print("📊 批量预测统计结果:")
        print(f"总样本数: {stats['total_samples']}")
        print(f"成功预测: {stats['successful_predictions']}")
        print(f"失败预测: {stats['failed_predictions']}")
        print(f"正确预测: {stats['correct_predictions']}")
        print(f"准确率: {stats['accuracy']:.4f}")
        print(f"总耗时: {stats['total_time']:.2f}秒")
        print(f"平均耗时: {stats['avg_time_per_sample']:.2f}秒/样本")
        print("="*60)
    
    else:
        print("❌ 请提供数据路径和图像文件夹路径，或指定单个样本ID")


if __name__ == "__main__":
    main()
