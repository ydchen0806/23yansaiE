# %%
import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# %%
sell_data = pd.read_excel('附件2.xlsx')

# %%
sell_data

# %%
# 定义起始日期和结束日期
start_date = '2023-06-24'
end_date = '2023-06-30'

# 将起始日期和结束日期转换为日期时间类型
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

sales_within_range = sell_data[(sell_data['销售日期'] >= start_date) & (sell_data['销售日期'] <= end_date)]

# %%
len(sales_within_range['单品编码'].unique())

# %%
class_data = pd.read_excel('附件1.xlsx')
bought_data = pd.read_excel('附件3.xlsx')
loss_data = pd.read_excel('附件4.xlsx',sheet_name='Sheet1')
loss_data_class = pd.read_excel('附件4.xlsx')

# %%
code = class_data['单品编码']
name = class_data['单品名称']
code2name = dict(zip(code,name))
sales_within_range['单品名称'] = sales_within_range['单品编码'].map(code2name)

# %%
len(sales_within_range['单品名称'].unique())

# %%
sales_within_range['单品名称'].unique()

# %%
sell_time = np.unique(sell_data['销售日期'])

# %%
order_num = sell_data.groupby('销售日期').count()['扫码销售时间'].values

# %%
plt.plot(sell_time,order_num)

# %%
items = class_data['单品名称'].values
cols_name = ['time'] + list(items)
table1 = pd.DataFrame(columns=cols_name)

# %%
sell_data

# %%
code = class_data['单品编码'].values
items_name = class_data['单品名称'].values
code2name = dict(zip(code,items_name))
sell_data['单品名称'] = sell_data['单品编码'].map(code2name)

# %%
items = class_data['单品名称'].values
cols_name = ['time'] + list(items)
table1 = pd.DataFrame(columns=cols_name)
for date in tqdm(sell_time):
    temp_data = sell_data[sell_data['销售日期']==date]
    temp_data = temp_data.groupby('单品名称').sum()
    item_name = temp_data.index.values
    item_num = temp_data['销量(千克)'].values
    temp_dict = dict(zip(item_name,item_num))
    temp_dict['time'] = date
    table1 = table1.append(temp_dict,ignore_index=True)

# %%
save_dir = './dealed_data'
os.makedirs(save_dir,exist_ok=True)
table1.to_excel(os.path.join(save_dir,'daily_sell_kg.xlsx'),index=False)

# %%
class_data
item_name = class_data['单品编码'].values
class_name = class_data['分类名称']
item2class = dict(zip(item_name,class_name))
sell_data['分类名称'] = sell_data['单品编码'].map(item2class)

# %%
sell_data['销售额'] = sell_data['销量(千克)'] * sell_data['销售单价(元/千克)']
sell_data

# %%
bought_data

# %%
items = np.unique(class_data['分类名称'])
cols_name = ['time'] + list(items)
table6 = pd.DataFrame(columns=cols_name)
table7 = pd.DataFrame(columns=cols_name)
for date in tqdm(sell_time):
    temp_data = sell_data[sell_data['销售日期']==date]
    temp_data = temp_data.groupby('分类名称').sum()
    item_name = temp_data.index.values
    item_num = temp_data['销售额'].values
    xiaoliang = temp_data['销量(千克)'].values
    ave_price = item_num / xiaoliang
    temp_dict = dict(zip(item_name,ave_price))
    temp_dict['time'] = date
    temp_dict2 = dict(zip(item_name,item_num))
    temp_dict2['time'] = date
    table6 = table6.append(temp_dict,ignore_index=True)
    table7 = table7.append(temp_dict2,ignore_index=True)

save_dir = './dealed_data'
os.makedirs(save_dir,exist_ok=True)
table6.to_excel(os.path.join(save_dir,'daily_mean_price_class.xlsx'),index=False)
table7.to_excel(os.path.join(save_dir,'daily_sell_money_class.xlsx'),index=False)

# %%
items = np.unique(class_data['分类名称'])
cols_name = ['time'] + list(items)
table2 = pd.DataFrame(columns=cols_name)
for date in tqdm(sell_time):
    temp_data = sell_data[sell_data['销售日期']==date]
    temp_data = temp_data.groupby('分类名称').sum()
    item_name = temp_data.index.values
    item_num = temp_data['销量(千克)'].values
    temp_dict = dict(zip(item_name,item_num))
    temp_dict['time'] = date
    table2 = table2.append(temp_dict,ignore_index=True)

# %%
table2.to_excel('./dealed_data/daily_sell_kg_class.xlsx',index=False)

# %%
bought_data[bought_data['日期'] == sell_time[0]]

# %%
items = class_data['单品名称'].values
cols_name = ['time'] + list(items)
table3 = pd.DataFrame(columns=cols_name)
for date in tqdm(sell_time):
    temp_data = bought_data[bought_data['日期']==date]
    temp_data = temp_data.groupby('单品名称').mean()
    item_name = temp_data.index.values
    item_num = temp_data['批发价格(元/千克)'].values
    temp_dict = dict(zip(item_name,item_num))
    temp_dict['time'] = date
    table3 = table3.append(temp_dict,ignore_index=True)

# %%
table3.to_excel('./dealed_data/daily_bought_price.xlsx',index=False)

# %%
items = class_data['单品名称'].values
cols_name = ['time'] + list(items)
table4 = pd.DataFrame(columns=cols_name)
for date in tqdm(sell_time):
    temp_data = sell_data[sell_data['销售日期']==date]
    temp_sheet = temp_data.groupby('单品名称').count()['扫码销售时间']
    # total_data_num = temp_sheet.values
    item_name = temp_sheet.index.values
    discounted_data = temp_data[temp_data['是否打折销售']=='是']
    discounted_data = discounted_data.groupby('单品名称').count()['扫码销售时间']
    discount_rate = discounted_data / temp_sheet
    discount_rate.replace(np.nan,0,inplace=True)
    temp_dict = dict(zip(item_name,discount_rate))
    temp_dict['time'] = date
    table4 = table4.append(temp_dict,ignore_index=True)


table4.to_excel('./dealed_data/daily_discounted_ratio.xlsx',index=False)

# %%
discount_rate.replace(np.nan,0,inplace=True)
discount_rate

# %%
discounted_data / temp_sheet

# %%
sell_data['进货价格'] = 0
for i in tqdm(range(len(sell_data))):
    date = sell_data.iloc[i]['销售日期']
    temp_data = bought_data[bought_data['日期']==date]
    sell_data['进货价格'].iloc[i] = temp_data[temp_data['单品名称']==sell_data.iloc[i]['单品名称']]['批发价格(元/千克)'].values[0]

# %%
sell_data.to_excel('./dealed_data/sell_data_with_bought_price.xlsx',index=False)

# %%
sell_data

# %%
items = class_data['单品名称'].values
cols_name = ['time'] + list(items)
table5 = pd.DataFrame(columns=cols_name)
for date in tqdm(sell_time):
    temp_data = sell_data[sell_data['销售日期']==date]
    temp_data = temp_data.groupby('单品名称').mean()
    item_name = temp_data.index.values
    item_num = temp_data['销售单价(元/千克)'].values
    temp_dict = dict(zip(item_name,item_num))
    temp_dict['time'] = date
    table5 = table5.append(temp_dict,ignore_index=True)

table5.to_excel('./dealed_data/daily_sell_price.xlsx',index=False)

# %%



