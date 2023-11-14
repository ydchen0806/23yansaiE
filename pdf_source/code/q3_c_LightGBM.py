# 导入需要的库
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_excel('/braindat/lab/wuxl/code/shumo/jianmo2023/竞赛发布数据/第三问c_首次.xlsx')

# 分割数据
X = data.drop('90天mRS', axis=1) 
y = data['90天mRS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 创建数据集
train_data = lgb.Dataset(X_train, y_train)

# 训练参数
params = {
    'objective': 'regression',
    'metric': 'rmse', 
    'num_leaves': 10,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbose': -1
}

# 训练模型
model = lgb.train(params, train_data, 400)

# 获取特征重要度
importances = model.feature_importance()

selected_features = []
selected_corrs = []
# 输出特征重要度
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    selected_features.append(X.columns[indices[f]])
    selected_corrs.append(importances[indices[f]])
    
# 将重要度存为字典
feat_imp = dict(zip(selected_features, selected_corrs))

# 保存到CSV文件
import csv

with open('/braindat/lab/wuxl/code/shumo/jianmo2023/代码/LightGBM_feat_importance.csv', 'w') as f:

  writer = csv.writer(f)

  # 写入标题
  writer.writerow(['Feature', 'Importance'])

  # 写入数据
  for key, value in feat_imp.items():
    writer.writerow([key, value])