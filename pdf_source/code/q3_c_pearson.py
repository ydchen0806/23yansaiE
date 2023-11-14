import pandas as pd
from scipy.stats import pearsonr
import numpy as np

# 读取数据
data = pd.read_excel('/braindat/lab/wuxl/code/shumo/jianmo2023/竞赛发布数据/第三问c_首次.xlsx')


# 定义要选择的列
columns = data.columns[1:-1]

corrs = []
for col in columns:

  # 计算特征与mRS的pearson相关系数
  r, p = pearsonr(data[col], data['90天mRS'])
  
  # 记录相关系数
  corrs.append(abs(r)) 

# 获取排序索引  
ranking = np.argsort(corrs)[::-1] 

# 选择前10名  
selected_features = []
selected_corrs = []
for i in ranking[:10]:
  selected_features.append(columns[i])
  selected_corrs.append(corrs[i])

print('Features:', selected_features) 
print('Corrs:', selected_corrs)

# 将特征和系数组成字典
feat_dict = {} 
for f, c in zip(selected_features, selected_corrs):
  feat_dict[f] = c

# 保存为CSV文件
import csv
with open('/braindat/lab/wuxl/code/shumo/jianmo2023/代码/feat_corr_首次.csv', 'w') as f:
  writer = csv.writer(f)

  # 写入标题
  writer.writerow(['Feature', 'Correlation'])  

  # 写入数据
  for k, v in feat_dict.items():
    writer.writerow([k, v])