import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

# 检查GPU的可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_excel('/braindat/lab/wuxl/code/shumo/jianmo2023/竞赛发布数据/第三问b_2.xlsx')

# 创建一个空的时间序列数据集
time_series_dataset = []

# 初始化变量以跟踪当前ID和连续计数
current_id = None
count = 0
time_series = []

# 遍历数据
for index, row in data.iterrows():
    # 获取当前行的ID
    current_row_id = row['ID']

    # 如果ID与上一个数据点相同，将其添加到当前时间序列
    if current_row_id == current_id:
        current_data = row.drop(['ID'])  # 去除ID列
        time_series.append(current_data.values)
        count += 1
    else:
        # 如果ID与上一个数据点不同，检查时间序列的长度
        if count >= 1:  # 只添加包含1到9个数据点的时间序列
            time_series_dataset.append(time_series.copy())
        # 重置时间序列和计数
        time_series = [row.drop(['ID']).values]
        count = 1
        current_id = current_row_id

# 添加最后一个时间序列（如果有的话）
if count >= 1:
    time_series_dataset.append(time_series.copy())

time_series_data = np.array(time_series_dataset)    

# 找到最大的时间步长
max_length = max(len(seq) for seq in time_series_data)

# 使用pad_sequences填充数据
padded_data = pad_sequences(time_series_data, maxlen=5, dtype='float32', padding='post', truncating='post')


# 准备输入数据和目标数据
X = padded_data[:, :, :-1]  # 输入数据，排除'90天mRS'字段
y = padded_data[:, 0, -1]   # 目标数据，仅包含'90天mRS'字段

X_2d = X.reshape(X.shape[0], -1)
pca = PCA(n_components=30)  # 选择要保留的主成分数量
X_pca = pca.fit_transform(X_2d)

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 将数据转换为PyTorch张量
X_train = torch.Tensor(X_train).to(device)
y_train = torch.Tensor(y_train).to(device)
X_test = torch.Tensor(X_test).to(device)
y_test = torch.Tensor(y_test).to(device)


# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_prob = 0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 7)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out, _ = self.lstm(x)
        # out = self.fc(out[:,-1. :])  # 获取LSTM的最后一个时间步的输出
        out = self.fc(out[:, :])  # 获取LSTM的最后一个时间步的输出
        return out

# 初始化模型
input_size = X_train.shape[1]
hidden_size = 2048
num_layers = 1

model = LSTMModel(input_size, hidden_size, num_layers).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 设置余弦学习率调度器
num_epochs = 100  # 根据需要调整总的训练周期数
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# 训练模型
for epoch in range(num_epochs):
    total_loss = 0.0  # 定义总损失
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()  # 累积损失
    scheduler.step()  # 更新学习率
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss/(num_epochs*100):.10f}')

from sklearn.metrics import accuracy_score
# 在测试集上进行预测
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    y_pred = test_outputs.argmax(dim=1).cpu().numpy()  # 根据输出的概率值取最大的类别作为预测类别

y_test_cpu = y_test.cpu().numpy()
accuracy = accuracy_score(y_test_cpu, y_pred)

print(f"Accuracy: {accuracy}")
