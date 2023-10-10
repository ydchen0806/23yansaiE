import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
# pca
from sklearn.decomposition import PCA


class mRSDataset(Dataset):
  # 自定义Dataset类加载数据
  
    def __init__(self, prase = 'train', scale_data = False, split_rate = 0.7, use_pca = True):
        # 读入表1,2,3数据
        # 数据连接、填充、one-hot等处理

        self.scale_data = scale_data
        self.split_rate = split_rate
        self.use_pca = use_pca
        table1 = pd.read_excel(r'E:\23研赛\建模2023\建模2023\2023年中国研究生数学建模竞赛赛题\E题\E题\数据\竞赛发布数据\表1-患者列表及临床信息.xlsx') 
        table2 = pd.read_excel(r'E:\23研赛\建模2023\建模2023\2023年中国研究生数学建模竞赛赛题\E题\E题\处理后的表2.xlsx')
        table3_ED = pd.read_excel(r'E:\23研赛\建模2023\建模2023\2023年中国研究生数学建模竞赛赛题\E题\E题\数据\竞赛发布数据\表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx', \
                               sheet_name='ED')
        table3_hemo = pd.read_excel(r'E:\23研赛\建模2023\建模2023\2023年中国研究生数学建模竞赛赛题\E题\E题\数据\竞赛发布数据\表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx', \
                               sheet_name='Hemo')
        self.table1 = table1
        self.table2 = table2
        self.table3_ED = table3_ED
        self.table3_hemo = table3_hemo

        # 2. 数据合并与预处理
        self.data = self.read_data(length = 100)
        self.data_total = self.read_data(length = 160)
        # 3. 特征选择与处理
        # 根据问题描述，选取需要的特征列
        origin_feature = []
        NCCT_feature = []
        for feature in self.data.columns:
            if 'NCCT' in feature:
                NCCT_feature.append(feature)
            elif 'origin' in feature:
                origin_feature.append(feature)
        add_feature = origin_feature + NCCT_feature
        selected_features = ['年龄', '性别', '脑出血前mRS评分', '高血压病史', '卒中病史', '糖尿病史', '房颤史', '冠心病史',
                            '吸烟史', '饮酒史', '发病到首次影像检查时间间隔', '高压', '低压', 'HM_volume', 'ED_volume','HM_ACA_R_Ratio',
                            'HM_MCA_R_Ratio','HM_PCA_R_Ratio','HM_Pons_Medulla_R_Ratio','HM_Cerebellum_R_Ratio',
                            'HM_ACA_L_Ratio','HM_MCA_L_Ratio', 'HM_PCA_L_Ratio','HM_Pons_Medulla_L_Ratio',
                            'HM_Cerebellum_L_Ratio','ED_ACA_R_Ratio','ED_MCA_R_Ratio','ED_PCA_R_Ratio',
                            'ED_Pons_Medulla_R_Ratio','ED_Cerebellum_R_Ratio','ED_ACA_L_Ratio','ED_MCA_L_Ratio',
                            'ED_PCA_L_Ratio','ED_Pons_Medulla_L_Ratio','ED_Cerebellum_L_Ratio']
        selected_features += add_feature
        
        columns_to_normalize = ['年龄', '脑出血前mRS评分', '发病到首次影像检查时间间隔', '高压', '低压', 'HM_volume', 'ED_volume', 
                         'HM_ACA_R_Ratio','HM_MCA_R_Ratio','HM_PCA_R_Ratio','HM_Pons_Medulla_R_Ratio','HM_Cerebellum_R_Ratio',
                            'HM_ACA_L_Ratio','HM_MCA_L_Ratio', 'HM_PCA_L_Ratio','HM_Pons_Medulla_L_Ratio',
                            'HM_Cerebellum_L_Ratio','ED_ACA_R_Ratio','ED_MCA_R_Ratio','ED_PCA_R_Ratio',
                            'ED_Pons_Medulla_R_Ratio','ED_Cerebellum_R_Ratio','ED_ACA_L_Ratio','ED_MCA_L_Ratio',
                            'ED_PCA_L_Ratio','ED_Pons_Medulla_L_Ratio','ED_Cerebellum_L_Ratio']
        columns_to_normalize += add_feature

        if self.use_pca:
            scaler = StandardScaler()
            self.data[columns_to_normalize] = scaler.fit_transform(self.data[columns_to_normalize])
            self.data_total[columns_to_normalize] = scaler.transform(self.data_total[columns_to_normalize])
            # pca
            pca = PCA(n_components=3)
            pca_feature = pca.fit_transform(self.data[add_feature])
            pca_feature_total = pca.transform(self.data_total[add_feature])
            self.data['pca1'] = pca_feature[:,0]
            self.data['pca2'] = pca_feature[:,1]
            self.data['pca3'] = pca_feature[:,2]
            self.data_total['pca1'] = pca_feature_total[:,0]
            self.data_total['pca2'] = pca_feature_total[:,1]
            self.data_total['pca3'] = pca_feature_total[:,2]
            selected_features = ['pca1', 'pca2', 'pca3']

        self.target = '90天mRS'
        self.data = self.data[selected_features + [self.target]]
        self.data_total = self.data_total[selected_features + [self.target]]
        # 处理缺失值（这里简单处理，您可以根据实际情况进行更复杂的处理）
        self.data.fillna(0, inplace=True)
        self.data_total.fillna(0, inplace=True)
        # 对分类特征进行独热编码（One-Hot Encoding）
        if self.scale_data:
            categorical_features = ['性别', '高血压病史', '卒中病史', '糖尿病史', '房颤史', '冠心病史', '吸烟史', '饮酒史']
            self.data = pd.get_dummies(self.data, columns=categorical_features, drop_first=True)      
            #归一化
            scaler = StandardScaler()
            self.data[columns_to_normalize] = scaler.fit_transform(self.data[columns_to_normalize])

    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        features = torch.tensor(self.data.drop(columns=['90天mRS']).iloc[idx].values, dtype=torch.float32)
        mrs_label = int(self.data['90天mRS'][idx])
    
        return features, mrs_label
    
    def read_data(self, length = 100):
        sub100_patients = ['sub{:03d}'.format(i) for i in range(length + 1)]
        table1_sub100 = self.table1[self.table1['Unnamed: 0'].isin(sub100_patients)]
        table2_sub100 = self.table2[self.table2['首次检查流水号'].isin(table1_sub100['入院首次影像检查流水号'])]
        table3_sub100_ED = self.table3_ED[self.table3_ED['流水号'].isin(table1_sub100['入院首次影像检查流水号'])]
        table3_sub100_Hemo = self.table3_hemo[self.table3_hemo['流水号'].isin(table1_sub100['入院首次影像检查流水号'])]
        data = table1_sub100.merge(table2_sub100, left_on='入院首次影像检查流水号', right_on='首次检查流水号', how='inner')
        data = data.merge(table3_sub100_ED, left_on='入院首次影像检查流水号', right_on='流水号', how='inner')
        data = data.merge(table3_sub100_Hemo, left_on='入院首次影像检查流水号', right_on='流水号', how='inner')
        data['高压'] = data['血压'].apply(lambda x: x.split('/')[0])
        data['低压'] = data['血压'].apply(lambda x: x.split('/')[1])
        data.drop(['血压'], axis=1, inplace=True)
        return data


    def get_dataset(self):
        h, _ = self.data.shape
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        trainset = self.data.iloc[:int(h * self.split_rate),:]
        test_set = self.data.iloc[int(h * self.split_rate):,:]
        return trainset, test_set, self.data_total

dataset = mRSDataset(prase='train')  
dataloader = DataLoader(dataset, batch_size=1)

dataset_test = mRSDataset(prase='test')  
dataloader_test = DataLoader(dataset_test, batch_size=1)

# 2. 定义模型


# 定义模型
class mRSModel(nn.Module):
    def __init__(self, input_size):
        super(mRSModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # 输出层有7个节点，对应7个类别
        )

    def forward(self, x):
        return self.layers(x)
    
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(1, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Add more Conv1d layers here as needed
            nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        )
        self.fc = nn.Linear(64, output_size)  # 全连接层
        self.softmax = nn.Softmax(dim=1)  # Softmax 激活函数

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.tcn(x)
        x = x = self.fc(x[:, :, -1])
        x = self.softmax(x) 
        return x

if __name__ == '__main__':
    # 初始化模型
    input_size = 35  # 获取特征的数量
    output_size = 7
    num_channels = 64
    kernel_size = 3
    dropout = 0.2

    model = mRSModel(input_size)
    # model = TCN(input_size, output_size, num_channels, kernel_size, dropout)
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
    learning_rate = 0.001  # 根据需要调整初始学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 设置余弦学习率调度器
    num_epochs = 100  # 根据需要调整总的训练周期数
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0.0  # 定义总损失
        for batch_features, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features.float())
            loss = criterion(outputs, batch_labels.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()  # 累积损失
        scheduler.step()  # 更新学习率
        print(f'Epoch [{epoch + 1}/{num_epochs}], Learning Rate: {scheduler.get_lr()[0]:.6f}, Loss: {total_loss/(num_epochs*100):.4f}')


    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_features, batch_labels in dataloader_test:
            outputs = model(batch_features.float())
            _, predicted = torch.max(outputs.data, 1)  # 获取预测值中的最大值的索引
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy on test data: {accuracy:.4f}')