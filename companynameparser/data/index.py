from pymilvus import connections, utility

# 连接到Milvus
connections.connect("default", host="localhost", port="19530")

# 检查索引状态
status = utility.get_index_building_progress(collection_name="place_embedding_collection")
print(status)
