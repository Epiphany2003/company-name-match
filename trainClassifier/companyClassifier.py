import pandas as pd

class CompanyClassifier(nn.Module):
    def __init__(self, input_dim=768*4):  # 4个维度×每个维度768维
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))  # 输出0-1之间的概率
        return x
    

    # 加载标注数据（假设已解析好公司信息）
# 数据格式示例：
# company1_parsed: {"place": "深圳", "brand": "腾讯", "trade": "科技", "suffix": "有限公司"}
# company2_parsed: {"place": "北京", "brand": "腾讯", "trade": "科技", "suffix": "分公司"}
# is_same: 1（同一家）
df = pd.read_pickle("labeled_company_pairs.pkl")

# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = CompanyPairDataset(train_df)
test_dataset = CompanyPairDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化模型、损失函数和优化器
model = CompanyClassifier()
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# 训练模型
for epoch in range(10):  # 训练10轮
    model.train()
    total_loss = 0
    for batch in train_loader:
        features = batch["features"]
        labels = batch["label"].unsqueeze(1)  # 调整维度
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # 测试集评估
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"]
            labels = batch["label"]
            outputs = model(features)
            preds = (outputs > 0.5).float().squeeze()  # 概率>0.5视为正例
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, "
          f"Test Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# 保存模型
torch.save(model.state_dict(), "company_classifier.pth")