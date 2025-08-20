import pandas as pd
from transformers import AutoTokenizer, BertModel
import torch
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 BERT 模型
def init_bert_model():
    BERT_PATH = r'bert-base-chinese'
    model = BertModel.from_pretrained(BERT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    model.eval()
    model.to('cuda')  # 将模型加载到 GPU
    return tokenizer, model

# 使用 BERT 对文本生成嵌入（批量处理）
def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt')
    # 将输入数据移动到指定设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # 将结果移回 CPU 并转换为 numpy 数组
    return embedding

# 创建 Milvus 集合
def create_collection(collection_name):
    fields = [
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=1024, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    schema = CollectionSchema(fields, description=f"Embeddings for {collection_name}")
    collection = Collection(name=collection_name, schema=schema)
    return collection

# 插入数据到 Milvus
def insert_to_milvus(collection, name, embedding):
    data = [
        [name],
        [embedding.tolist()]  # 将 embedding 转换为列表，并包装成列表的列表
    ]
    collection.insert(data)

# 主函数
import pandas as pd

def main():
    csv_path = "data/company_parsed.csv"
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')  # 尝试使用 UTF-8 编码
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='gbk')  # 如果 UTF-8 失败，尝试 GBK 编码
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin1')  # 如果都失败，尝试 Latin-1 编码

    # 输出前几行数据，检查是否有乱码
    print("Preview of the data (first 5 rows):")
    print(df.head())  # 输出前 5 行数据

    # 截断 name 字段
    df["name"] = df["name"].str.slice(0, 256)

    # 初始化 BERT
    tokenizer, model = init_bert_model()

    # 连接 Milvus
    connections.connect(host="localhost", port="19530")

    # 只处理 place 列
    column = "place"
    collection_name = f"{column}_embedding_collection"
    collection = create_collection(collection_name)

    # 取前 1000 条数据
    texts = df[column].fillna("").astype(str).head(1000).tolist()
    names = df["name"].head(1000).tolist()

    for i in range(len(names)):
        name = names[i]
        text = texts[i]
        # 编码文本
        embedding = encode_text(text, tokenizer, model)
        # 插入到 Milvus
        insert_to_milvus(collection, name, embedding)
        print(f"Inserted {name} into {collection_name}")

    print(f"All data for column {column} inserted into Milvus.")

if __name__ == "__main__":
    main()
