from pymilvus import connections, Collection
from transformers import BertModel, AutoTokenizer
import torch

# 连接到 Milvus 数据库
connections.connect("default", host="localhost", port="19530")

# 定义 Collection 名称
collection_name = "place_embedding_collection"

# 检查 Collection 是否存在
collection = Collection(collection_name)

# 加载 Collection
print(f"Loading collection '{collection_name}'...")
collection.load()  # 加载集合

# 准备输入文本
text = "深圳"

# 编码文本为 BERT 向量
def encode_text_to_vector(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        # 使用 pooler_output 作为句子向量
        vector = outputs.pooler_output.squeeze().numpy()
    return vector

# 加载本地 BERT 模型和分词器
BERT_PATH = r'D:\huggingface_model\bert-base-chinese'
model = BertModel.from_pretrained(BERT_PATH)
tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)

query_vector = encode_text_to_vector(text, model, tokenizer)

# 执行搜索
top_k = 10  # 查询前 10 个最近向量
search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}  # 修改为 COSINE

results = collection.search(
    data=[query_vector],  # 查询向量列表
    anns_field="embedding",  # 向量字段名称
    param=search_params,  # 搜索参数
    limit=top_k,  # 返回前 k 个结果
    output_fields=["name"]  # 返回字段
)

# 处理搜索结果
for hits in results:
    for hit in hits:
        print(f"Name: {hit.entity.get('name')}, Distance: {hit.distance}")
