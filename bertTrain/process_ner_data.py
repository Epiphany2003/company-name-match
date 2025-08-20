'''将原始标注数据转换为训练格式，处理分词对齐。'''

import os
import json
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer,  BertTokenizerFast

# 标签映射（O→0，B-ORG→1，I-ORG→2）
LABEL2ID = {"O": 0, "B-ORG": 1, "I-ORG": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def load_annotations(file_path):
    """加载原始标注数据"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 分割公司名和标签（按空格，前半部分是公司名，后半部分是标签）
            parts = line.split()
            company_name = parts[0]
            labels = parts[1:]
            # 检查字符数与标签数是否一致
            if len(company_name) != len(labels):
                print(f"跳过无效数据：{line}（字符数与标签数不匹配）")
                continue
            data.append({"text": list(company_name), "labels": labels})
    return data

def align_labels_with_tokens(tokenizer, text, labels):
    """处理分词对齐：子词除第一个外标记为-100（忽略损失）"""
    input_str = "".join(text)
    tokenized = tokenizer(
        input_str,
        return_offsets_mapping=True,
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    offset_mapping = tokenized["offset_mapping"]  # 子词在原始字符串中的位置
    aligned_labels = []
    for start, end in offset_mapping:
        if start == 0 and end == 0:  # 特殊符号[CLS]/[SEP]/[PAD]
            aligned_labels.append(-100)
        else:
            # 找到子词对应的原始字符索引（中文每个字符占1个位置）
            char_idx = start
            if char_idx < len(labels):
                aligned_labels.append(LABEL2ID[labels[char_idx]])
            else:
                aligned_labels.append(-100)
    tokenized["labels"] = aligned_labels
    return tokenized

def process_and_save_data(raw_data, tokenizer, output_dir):
    """处理数据并保存为训练格式"""
    processed = []
    for item in raw_data:
        processed_item = align_labels_with_tokens(
            tokenizer, item["text"], item["labels"]
        )
        # 只保留训练需要的字段
        processed.append({
            "input_ids": processed_item["input_ids"],
            "attention_mask": processed_item["attention_mask"],
            "labels": processed_item["labels"]
        })
    # 划分训练集、验证集、测试集（8:1:1）
    train_val, test = train_test_split(processed, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=1/9, random_state=42)  # 9份里取1份作为验证集
    # 保存为JSON（方便加载）
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False)
    with open(f"{output_dir}/val.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False)
    with open(f"{output_dir}/test.json", "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False)
    print(f"数据处理完成，保存至{output_dir}（训练集{len(train)}，验证集{len(val)}，测试集{len(test)}）")

if __name__ == "__main__":
    # 配置路径
    RAW_DATA_PATH = "data/data_for_train.txt"  # 原始标注数据
    OUTPUT_DIR = "data/processed_ner_data"  # 处理后的数据保存路径
    BERT_PATH = r'C:\Users\22403\Desktop\company-name-match\my-bert-base-chinese'  # 预训练模型

    # 加载分词器
    tokenizer = BertTokenizerFast.from_pretrained(BERT_PATH)
    # 加载并处理数据
    raw_data = load_annotations(RAW_DATA_PATH)
    process_and_save_data(raw_data, tokenizer, OUTPUT_DIR)