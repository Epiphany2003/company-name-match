'''用训练好的模型提取公司名中的品牌'''

import torch
from transformers import BertTokenizer
from .train_ner import BertForBrandNER

class BrandExtractor:
    def __init__(self, model_path):
        """加载模型和分词器"""
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForBrandNER.from_pretrained(model_path)
        self.model.eval()  # 推理模式
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.id2label = self.model.config.id2label

    def extract(self, company_name):
        """提取公司名中的品牌（B-ORG和I-ORG对应的字符）"""
        # 预处理
        inputs = self.tokenizer(
            company_name,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
            return_token_type_ids=False  # 禁用token_type_ids
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 预测
        with torch.no_grad():
            outputs = self.model(** inputs)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[1]
        predictions = torch.argmax(logits, dim=2).squeeze().tolist()

        # 解析结果
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
        brand_chars = []
        for token, pred in zip(tokens, predictions):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            label = self.id2label[pred]
            if label in ["B-ORG", "I-ORG"]:
                # 去除子词标记（如##）
                brand_chars.append(token.replace("##", ""))
        return "".join(brand_chars)

# 测试
'''if __name__ == "__main__":
    extractor = BrandExtractor("models/bert_brand_ner/final")
    test_names = [
        "阿城众盈汽车销售公司",
        "阿里软件公司",
        "阿盟交通设计研究有限公司"
    ]
    for name in test_names:
        print(f"{name} → 关键词：{extractor.extract(name)}")'''
