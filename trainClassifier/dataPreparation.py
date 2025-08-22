import torch
import pandas as pd
import numpy as np
from transformers import BertModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# 加载BERT模型和分词器
BERT_PATH = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
bert_model = BertModel.from_pretrained(BERT_PATH).eval()

# 定义特征提取函数
def extract_feature(text):
    """用BERT提取文本的特征向量"""
    if not text:  # 处理空字符串
        return np.zeros(768)  # BERT-base的隐藏层维度为768
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.pooler_output.numpy()[0]  # 取[CLS]位置的向量

# 构建数据集
class CompanyPairDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        com1 = item["company1_parsed"]  # 解析后的公司1信息（含place/brand/trade/suffix）
        com2 = item["company2_parsed"]  # 解析后的公司2信息
        label = item["is_same"]  # 1表示同一家，0表示不同
        
        # 提取四个维度的特征
        f1_place = extract_feature(com1["place"])
        f1_brand = extract_feature(com1["brand"])
        f1_trade = extract_feature(com1["trade"])
        f1_suffix = extract_feature(com1["suffix"])
        
        f2_place = extract_feature(com2["place"])
        f2_brand = extract_feature(com2["brand"])
        f2_trade = extract_feature(com2["trade"])
        f2_suffix = extract_feature(com2["suffix"])
        
        # 特征融合：计算两家公司的特征差（也可使用拼接或其他方式）
        features = np.concatenate([
            f1_place - f2_place,
            f1_brand - f2_brand,
            f1_trade - f2_trade,
            f1_suffix - f2_suffix
        ])
        
        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32)
        }