# 医疗VQA系统 - 基于QWEN2.5-VL-3B-INSTRUCT

这是一个基于QWEN2.5-VL-3B-INSTRUCT模型的医疗视觉问答系统，专门适配您提供的医疗数据格式。

## 系统特点

- ✅ 支持QWEN2.5-VL-3B-INSTRUCT本地模型
- ✅ 适配您的医疗数据格式（包含id、question、options、label、images等字段）
- ✅ 支持单个样本测试和批量处理
- ✅ 自动处理多图像输入
- ✅ 生成详细的结果统计和错误日志

## 文件说明

- `medical_vqa_qwen.py` - 核心VQA系统类
- `run_medical_vqa.py` - 快速启动脚本
- `config.py` - 配置文件（需要您修改路径）
- `example_usage.py` - 使用示例
- `README_medical_vqa.md` - 本说明文件

## 数据格式要求

您的数据应该包含以下字段（JSON或JSONL格式）：

```json
{
    "id": "MM-2001",
    "question": "医疗问题描述...",
    "options": {
        "A": "选项A内容",
        "B": "选项B内容", 
        "C": "选项C内容",
        "D": "选项D内容",
        "E": "选项E内容"
    },
    "label": "B",
    "images": ["MM-2001-a.jpeg"],
    "medical_task": "Diagnosis",
    "body_system": "Cardiovascular",
    "question_type": "Reasoning"
}
```

## 快速开始

### 1. 配置路径

首先修改 `config.py` 文件中的路径：
文件中路径都按照要求修改

```python
# 修改为您的实际路径
MODEL_PATH = "/path/to/Qwen2.5-VL-3B-Instruct"
DATA_PATH = "/path/to/your/medical_data.jsonl" 
IMAGE_FOLDER = "/path/to/your/images"
```

### 2. 环境检查

```bash
python run_medical_vqa.py --mode preview
```

### 3. 单个测试

```bash
python run_medical_vqa.py --mode test --sample_id MM-2001
```

### 4. 批量处理

```bash
python run_medical_vqa.py --mode batch
```

## 详细使用方法

### 命令行使用

#### 1. 使用默认配置
```bash
# 数据预览
python run_medical_vqa.py --mode preview

# 单个测试（使用第一个样本）
python run_medical_vqa.py --mode test

# 指定样本ID测试
python run_medical_vqa.py --mode test --sample_id MM-2001

# 批量处理
python run_medical_vqa.py --mode batch
```

#### 2. 自定义路径
```bash
python run_medical_vqa.py \
    --mode batch \
    --model_path /your/model/path \
    --data_path /your/data/file.jsonl \
    --image_folder /your/images/folder \
    --output_path /your/results.jsonl
```

### Python代码使用

```python
from medical_vqa_qwen import MedicalVQA
import json

# 初始化系统
vqa = MedicalVQA("/path/to/Qwen2.5-VL-3B-Instruct")
vqa.load_model()

# 准备数据
with open("your_data.jsonl", "r") as f:
    data = json.load(f)

# 单个预测
result = vqa.predict_single(data[0], "/path/to/images")
print(f"预测结果: {result}")

# 批量预测
stats = vqa.predict_batch(
    "your_data.jsonl", 
    "/path/to/images", 
    "results.jsonl"
)
print(f"准确率: {stats['accuracy']:.4f}")
```

## 输出结果格式

### 预测结果
```json
{
    "id": "MM-2001",
    "question": "医疗问题...",
    "images": ["/path/to/MM-2001-a.jpeg"],
    "prompt": "构建的提示词...",
    "pred_raw": "模型原始输出",
    "pred_label": "B",
    "pred_answer": "Chest radiograph", 
    "true_label": "B",
    "true_answer": "Chest radiograph",
    "correct": true,
    "success": true
}
```

### 统计结果
```json
{
    "total_samples": 100,
    "successful_predictions": 95,
    "failed_predictions": 5,
    "correct_predictions": 78,
    "accuracy": 0.8211,
    "total_time": 120.5,
    "avg_time_per_sample": 1.27
}
```

## 系统要求

### 硬件要求
- GPU: 建议使用8GB+显存的GPU（如RTX 3070/4070等）
- RAM: 至少16GB系统内存
- 存储: 至少10GB可用空间

### 软件要求
```bash
pip install torch transformers pillow tqdm
```

Python版本: 3.8+

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型路径是否正确
   - 确认模型是QWEN2.5-VL-3B-Instruct格式
   - 检查是否有足够的显存

2. **图像加载失败**
   - 检查图像文件夹路径
   - 确认图像文件存在且格式正确
   - 检查文件权限

3. **数据格式错误**
   - 确认JSON/JSONL格式正确
   - 检查必需字段是否存在
   - 验证选项和标签的对应关系

4. **CUDA内存不足**
   - 系统会自动切换到CPU模式
   - 可以尝试减少batch_size
   - 或者使用更小的模型

### 性能优化

1. **GPU优化**
   - 使用FP16精度可减少显存使用
   - 适当调整max_new_tokens参数

2. **批量处理**
   - 目前系统逐样本处理，适合大规模数据
   - 可通过多进程加速（需要额外开发）

## 联系方式

如有问题，请检查：
1. 路径配置是否正确
2. 数据格式是否符合要求  
3. 系统环境是否满足要求

## 更新日志

- v1.0: 初始版本，支持基本的医疗VQA功能
- 适配QWEN2.5-VL-3B-Instruct模型
- 支持您提供的数据格式
