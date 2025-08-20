import pandas as pd
from transformers import AutoTokenizer, BertModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_bert_model():
    BERT_PATH = r'bert-base-chinese'
    model = BertModel.from_pretrained(BERT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    model.eval()
    model.to('cuda')  # 将模型加载到 GPU
    return tokenizer, model

def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt')
    # 将输入数据移动到指定设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # 将结果移回 CPU 并转换为 numpy 数组
    return embedding

tokenizers, model = init_bert_model()