from pymilvus import connections, utility

# 连接到 Milvus
connections.connect(host="localhost", port="19530")

# 获取所有集合的名称
collections = utility.list_collections()
print(f"Existing collections: {collections}")

# 删除所有集合
for collection_name in collections:
    utility.drop_collection(collection_name)
    print(f"Deleted collection: {collection_name}")

print("All collections have been deleted.")
