import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

class mRSDataset(Dataset):
  
    def __init__(self, prase = 'train'):
        # 读入表1,2,3数据
        # 数据连接、填充、one-hot等处理
        table1 = pd.read_excel('/braindat/lab/wuxl/code/shumo/jianmo2023/竞赛发布数据/表1-患者列表及临床信息更新后.xlsx') 
        table2 = pd.read_excel('/braindat/lab/wuxl/code/shumo/jianmo2023/竞赛发布数据/处理后的表216.13.xlsx')
        table3 = pd.read_excel('/braindat/lab/wuxl/code/shumo/jianmo2023/竞赛发布数据/表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx')
        
        # 数据合并与预处理
        if prase == 'train':
            sub100_patients = ['sub{:03d}'.format(i) for i in range(1, 100)]
        else:
            sub100_patients = ['sub{:03d}'.format(i) for i in range(101, 160)]
        table1_sub100 = table1[table1['ID'].isin(sub100_patients)]
        table2_sub100 = table2[table2['首次检查流水号'].isin(table1_sub100['入院首次影像检查流水号'])]
        table3_sub100 = table3[table3['流水号'].isin(table1_sub100['入院首次影像检查流水号'])]
        # 合并数据表
        self.data = table1_sub100.merge(table2_sub100, left_on='入院首次影像检查流水号', right_on='首次检查流水号', how='inner')
        self.data = self.data.merge(table3_sub100, left_on='入院首次影像检查流水号', right_on='流水号', how='inner')
        
        # 特征选择与处理
        selected_features = ['年龄', '性别', '脑出血前mRS评分', '高血压病史', '卒中病史', '糖尿病史', '房颤史', '冠心病史',
                            '吸烟史', '饮酒史', '发病到首次影像检查时间间隔', '高压', '低压', 'HM_volume', 'ED_volume','HM_ACA_R_Ratio',
                            'HM_MCA_R_Ratio','HM_PCA_R_Ratio','HM_Pons_Medulla_R_Ratio','HM_Cerebellum_R_Ratio',
                            'HM_ACA_L_Ratio','HM_MCA_L_Ratio', 'HM_PCA_L_Ratio','HM_Pons_Medulla_L_Ratio',
                            'HM_Cerebellum_L_Ratio','ED_ACA_R_Ratio','ED_MCA_R_Ratio','ED_PCA_R_Ratio',
                            'ED_Pons_Medulla_R_Ratio','ED_Cerebellum_R_Ratio','ED_ACA_L_Ratio','ED_MCA_L_Ratio',
                            'ED_PCA_L_Ratio','ED_Pons_Medulla_L_Ratio','ED_Cerebellum_L_Ratio']
        
        columns_to_normalize = ['年龄', '发病到首次影像检查时间间隔', '高压', '低压', 'HM_volume', 'ED_volume', 
                         'HM_ACA_R_Ratio','HM_MCA_R_Ratio','HM_PCA_R_Ratio','HM_Pons_Medulla_R_Ratio','HM_Cerebellum_R_Ratio',
                            'HM_ACA_L_Ratio','HM_MCA_L_Ratio', 'HM_PCA_L_Ratio','HM_Pons_Medulla_L_Ratio',
                            'HM_Cerebellum_L_Ratio','ED_ACA_R_Ratio','ED_MCA_R_Ratio','ED_PCA_R_Ratio',
                            'ED_Pons_Medulla_R_Ratio','ED_Cerebellum_R_Ratio','ED_ACA_L_Ratio','ED_MCA_L_Ratio',
                            'ED_PCA_L_Ratio','ED_Pons_Medulla_L_Ratio','ED_Cerebellum_L_Ratio']
      
        self.target = '90天mRS'
        self.data = self.data[selected_features + [self.target]]
        # 处理缺失值
        self.data.fillna(0, inplace=True)
        # 对分类特征进行独热编码（One-Hot Encoding）
        categorical_features = ['性别', '高血压病史', '卒中病史', '糖尿病史', '房颤史', '冠心病史', '吸烟史', '饮酒史', '脑出血前mRS评分']
        self.data = pd.get_dummies(self.data, columns=categorical_features, drop_first=True)
        #归一化
        scaler = StandardScaler()
        self.data[columns_to_normalize] = scaler.fit_transform(self.data[columns_to_normalize])
        print(self.data)
        
        



label_encoder = LabelEncoder()
dataset_train = mRSDataset(prase='train') 
train_data, valid_data = train_test_split(dataset_train.data, test_size=0.3)
X_train = train_data.drop(columns=['90天mRS']).values
y_train = label_encoder.fit_transform(train_data['90天mRS'])

X_test = valid_data.drop(columns=['90天mRS']).values
y_test =label_encoder.fit_transform(valid_data['90天mRS'])

#创建随机森林分类器
model = RandomForestClassifier(n_estimators=100, random_state=42)
# model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01, max_depth=3, random_state=42)


#训练模型
model.fit(X_train, y_train)

importances = model.feature_importances_
feature_names = train_data.drop('90天mRS', axis=1).columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df.to_csv('/braindat/lab/wuxl/code/shumo/jianmo2023/代码/feature_importance.csv', index=False)
# print(feature_name)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('准确性：', accuracy)

# # 打印分类报告，包括精确度、召回率和F1分数等
# classification_rep = classification_report(y_test, y_pred, labels=range(7))
# print('分类报告：\n', classification_rep)