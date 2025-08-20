'''定义 NER 模型，加载处理后的数据，微调 BERT'''

import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    BertPreTrainedModel, BertModel,
    TrainingArguments, Trainer,
    BertTokenizer
)
import torch.nn as nn
from seqeval.metrics import f1_score

# 标签映射（与预处理一致）
LABEL2ID = {"O": 0, "B-ORG": 1, "I-ORG": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

class BertForBrandNER(BertPreTrainedModel):
    """BERT+分类头用于品牌识别"""
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 输出标签数
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state)  # (batch_size, seq_len, num_labels)

        loss = None
        if labels is not None:
            # 计算交叉熵损失（忽略-100的标签）
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        return (loss, logits) if loss is not None else logits

def load_processed_data(data_dir):
    """加载处理后的JSON数据为Dataset格式"""
    def load_json(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    train_data = load_json(f"{data_dir}/train.json")
    val_data = load_json(f"{data_dir}/val.json")
    test_data = load_json(f"{data_dir}/test.json")
    return (
        Dataset.from_list(train_data),
        Dataset.from_list(val_data),
        Dataset.from_list(test_data)
    )

def compute_metrics(predictions):
    """计算F1分数（NER任务核心指标）"""
    preds, labels = predictions
    preds = np.argmax(preds, axis=2)  # 取概率最大的标签

    # 转换为标签文本（忽略-100）
    true_preds = [
        [ID2LABEL[p] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(preds, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(preds, labels)
    ]
    return {"f1": f1_score(true_labels, true_preds)}

if __name__ == "__main__":
    # 配置路径
    DATA_DIR = "data/processed_ner_data"  # 预处理后的数据
    BERT_PATH = r'C:\Users\22403\Desktop\company-name-match\my-bert-base-chinese'  # 预训练模型
    OUTPUT_DIR = "models/bert_brand_ner"  # 模型保存路径
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载数据和分词器
    train_dataset, val_dataset, test_dataset = load_processed_data(DATA_DIR)
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    # 加载模型
    model = BertForBrandNER.from_pretrained(
        BERT_PATH,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # 训练参数配置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=32,  # 批次大小（根据GPU显存调整）
        per_device_eval_batch_size=32,
        learning_rate=3e-5,  # BERT微调常用学习率
        num_train_epochs=5,  # 训练轮数（可根据验证集效果调整）
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        eval_strategy="epoch",  # 每轮评估一次
        save_strategy="epoch",  # 每轮保存一次模型
        load_best_model_at_end=True,  # 最后加载验证集效果最好的模型
        metric_for_best_model="f1"  # 以F1分数作为最优模型指标
    )

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # 开始训练
    print("开始训练...")
    trainer.train()

    # 测试集评估
    print("测试集评估...")
    test_results = trainer.evaluate(test_dataset)
    print(f"测试集效果：{test_results}")

    # 保存最终模型和分词器
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"模型保存至{OUTPUT_DIR}/final")