#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MedRAG检索增强模块
为医疗VQA系统提供相关医疗知识检索功能
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pickle
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm


class MedicalKnowledgeBase:
    """医疗知识库类"""
    
    def __init__(self, knowledge_path: str = None, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        初始化医疗知识库
        
        Args:
            knowledge_path: 知识库文件路径
            embedding_model: 嵌入模型名称
        """
        self.knowledge_path = knowledge_path
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.knowledge_docs = []
        self.embeddings = None
        self.doc_index = {}
        
    def load_embedding_model(self):
        """加载嵌入模型"""
        print(f"🔹 加载嵌入模型: {self.embedding_model_name}")
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            if torch.cuda.is_available():
                self.embedding_model = self.embedding_model.cuda()
            print("✅ 嵌入模型加载成功")
        except Exception as e:
            print(f"❌ 嵌入模型加载失败: {e}")
            raise
    
    def load_medical_knowledge(self, knowledge_path: str = None):
        """加载医疗知识库"""
        if knowledge_path:
            self.knowledge_path = knowledge_path
            
        if not self.knowledge_path or not os.path.exists(self.knowledge_path):
            print("⚠️  未找到知识库文件，将创建默认知识库")
            self.create_default_knowledge_base()
            return
            
        print(f"🔹 加载医疗知识库: {self.knowledge_path}")
        
        try:
            if self.knowledge_path.endswith('.json'):
                with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.knowledge_docs = data
                    else:
                        self.knowledge_docs = [data]
            elif self.knowledge_path.endswith('.jsonl'):
                with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.knowledge_docs.append(json.loads(line))
            
            print(f"✅ 成功加载 {len(self.knowledge_docs)} 条医疗知识")
            
        except Exception as e:
            print(f"❌ 知识库加载失败: {e}")
            print("创建默认知识库...")
            self.create_default_knowledge_base()
    
    def create_default_knowledge_base(self):
        """创建默认医疗知识库"""
        print("🔹 创建默认医疗知识库...")
        self.knowledge_docs = [
            {
                "id": "cardio_basic_1",
                "title": "心血管系统基础",
                "content": "心血管系统包括心脏、血管和血液。心脏是循环系统的中心，负责泵血到全身各个器官和组织。心血管疾病常见的症状包括胸痛、呼吸困难、心悸、疲劳等。",
                "category": "cardiovascular",
                "keywords": ["心脏", "血管", "血液循环", "胸痛", "呼吸困难"]
            },
            {
                "id": "respiratory_basic_1", 
                "title": "呼吸系统基础",
                "content": "呼吸系统包括鼻腔、咽、喉、气管、支气管和肺。主要功能是进行气体交换，为身体提供氧气并排出二氧化碳。常见呼吸系统疾病包括哮喘、慢性阻塞性肺病(COPD)、肺炎等。",
                "category": "respiratory",
                "keywords": ["呼吸", "肺", "气管", "COPD", "哮喘"]
            },
            {
                "id": "diagnosis_imaging_1",
                "title": "影像诊断检查",
                "content": "医学影像检查包括X光片、CT、MRI、超声等。胸部X光是评估肺部和心脏的基本检查。CT扫描能提供更详细的横截面图像。MRI适用于软组织成像。超声检查无辐射，适合动态观察。",
                "category": "diagnosis",
                "keywords": ["X光", "CT", "MRI", "超声", "影像诊断"]
            },
            {
                "id": "emergency_medicine_1",
                "title": "急诊医学原则", 
                "content": "急诊医学遵循ABC原则：气道(Airway)、呼吸(Breathing)、循环(Circulation)。紧急情况下需要快速评估生命体征，包括血压、心率、呼吸频率、体温和血氧饱和度。",
                "category": "emergency",
                "keywords": ["急诊", "生命体征", "ABC原则", "血压", "心率"]
            },
            {
                "id": "lab_tests_1",
                "title": "实验室检查",
                "content": "常用实验室检查包括血常规(CBC)、生化检查、凝血功能、心肌标志物等。D-二聚体用于评估血栓风险。肌钙蛋白是心肌损伤的特异性标志物。肝功能检查包括ALT、AST、胆红素等。",
                "category": "laboratory", 
                "keywords": ["血常规", "D-二聚体", "肌钙蛋白", "肝功能", "凝血功能"]
            }
        ]
        print(f"✅ 创建了 {len(self.knowledge_docs)} 条默认医疗知识")
    
    def build_embeddings(self):
        """构建知识库的嵌入向量"""
        if not self.embedding_model:
            self.load_embedding_model()
            
        print("🔹 构建知识库嵌入向量...")
        
        # 为每个知识文档创建文本表示
        texts = []
        for i, doc in enumerate(self.knowledge_docs):
            # 组合标题、内容和关键词
            text = f"{doc.get('title', '')}\n{doc.get('content', '')}"
            if 'keywords' in doc:
                text += f"\n关键词: {', '.join(doc['keywords'])}"
            texts.append(text)
            self.doc_index[i] = doc.get('id', f'doc_{i}')
        
        # 计算嵌入向量
        embeddings = []
        batch_size = 32
        for i in tqdm(range(0, len(texts), batch_size), desc="计算嵌入"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts, convert_to_tensor=True)
            embeddings.append(batch_embeddings.cpu().numpy())
        
        self.embeddings = np.vstack(embeddings)
        print(f"✅ 成功构建 {self.embeddings.shape[0]} 个嵌入向量")


class MedRAGRetriever:
    """医疗检索增强生成器"""
    
    def __init__(self, knowledge_base: MedicalKnowledgeBase):
        """
        初始化MedRAG检索器
        
        Args:
            knowledge_base: 医疗知识库实例
        """
        self.kb = knowledge_base
        
    def retrieve_relevant_docs(self, query: str, question: str, options: Dict, 
                             medical_task: str = "", body_system: str = "", 
                             top_k: int = 3) -> List[Dict]:
        """
        检索相关医疗文档
        
        Args:
            query: 查询文本
            question: 医疗问题
            options: 选项字典
            medical_task: 医疗任务类型
            body_system: 身体系统
            top_k: 返回的文档数量
            
        Returns:
            相关文档列表
        """
        if not self.kb.embedding_model or self.kb.embeddings is None:
            print("⚠️  知识库未初始化，跳过检索")
            return []
        
        # 构建查询文本
        query_text = self.build_query_text(query, question, options, medical_task, body_system)
        
        # 计算查询嵌入
        query_embedding = self.kb.embedding_model.encode([query_text], convert_to_tensor=True)
        if torch.cuda.is_available():
            query_embedding = query_embedding.cuda()
        
        # 计算相似度
        query_embedding = query_embedding.cpu().numpy()
        similarities = np.dot(self.kb.embeddings, query_embedding.T).flatten()
        
        # 获取最相关的文档
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        relevant_docs = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # 相似度阈值
                doc = self.kb.knowledge_docs[idx].copy()
                doc['similarity_score'] = float(similarities[idx])
                relevant_docs.append(doc)
        
        return relevant_docs
    
    def build_query_text(self, query: str, question: str, options: Dict, 
                        medical_task: str, body_system: str) -> str:
        """构建查询文本"""
        # 提取关键词
        keywords = []
        
        # 从医疗任务和身体系统提取关键词
        if medical_task:
            keywords.append(medical_task)
        if body_system:
            keywords.append(body_system)
            
        # 从选项内容提取关键词
        option_texts = []
        for key, value in options.items():
            option_texts.append(value)
        
        # 组合查询文本
        query_parts = [query, question]
        if keywords:
            query_parts.append(" ".join(keywords))
        if option_texts:
            query_parts.extend(option_texts)
            
        return " ".join(query_parts)
    
    def format_retrieved_docs(self, docs: List[Dict]) -> str:
        """格式化检索到的文档为文本"""
        if not docs:
            return ""
        
        formatted_text = "\n## 相关医疗知识:\n"
        for i, doc in enumerate(docs, 1):
            formatted_text += f"{i}. **{doc.get('title', '未知标题')}** (相关度: {doc.get('similarity_score', 0):.3f})\n"
            formatted_text += f"   {doc.get('content', '')}\n"
            if 'keywords' in doc:
                formatted_text += f"   关键词: {', '.join(doc['keywords'])}\n"
            formatted_text += "\n"
        
        return formatted_text


def create_medrag_system(knowledge_path: str = None, embedding_model: str = None) -> Tuple[MedicalKnowledgeBase, MedRAGRetriever]:
    """
    创建MedRAG系统的便捷函数
    
    Args:
        knowledge_path: 知识库路径
        embedding_model: 嵌入模型名称
        
    Returns:
        (知识库, 检索器) 元组
    """
    # 初始化知识库
    kb = MedicalKnowledgeBase(knowledge_path, embedding_model or "sentence-transformers/all-MiniLM-L6-v2")
    kb.load_medical_knowledge()
    kb.build_embeddings()
    
    # 初始化检索器
    retriever = MedRAGRetriever(kb)
    
    return kb, retriever
