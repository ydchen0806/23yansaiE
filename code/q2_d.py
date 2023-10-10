import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from q2_c import extract_feature, causal_analysis
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.stats import pearsonr

if __name__ == '__main__':
    label = pd.read_excel(r'\q2_b_cluster_4_record.xlsx')
    label = label.groupby('ID').first().reset_index()
    id_col = label['ID']
    label_col = label['label']
    id2label = dict(zip(id_col, label_col))
    prob_data = pd.read_excel(r'\q2_total_data_all.xlsx')
    prob_data['label'] = prob_data['ID'].map(id2label)
    treat_info = pd.read_excel(r'\数据\竞赛发布数据\表1-患者列表及临床信息.xlsx')
    _, id2feature = extract_feature(treat_info, '年龄', '营养神经')
    for key in id2feature.keys():
        prob_data[key] = prob_data['ID'].map(id2feature[key])
    prob_data['高压'] = prob_data['血压'].apply(lambda x: int(x.split('/')[0]))
    prob_data['低压'] = prob_data['血压'].apply(lambda x: int(x.split('/')[1]))
    sex_map = {'男': 1, '女': 0}
    prob_data['性别'] = prob_data['性别'].apply(lambda x: int(sex_map[x]))
    prob_data.drop(['Unnamed: 0','血压'], axis=1, inplace=True)
    
    prob_data.to_excel('q2_d.xlsx')
    label = prob_data['label'].unique()
    save_dir = 'q2_d_figure'
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'q2_d.txt'), 'w') as f:
        for i in tqdm(range(4)):
            temp = prob_data[prob_data['label'] == i]
            HM_volume = temp['HM_volume']
            ED_volume = temp['ED_volume']
            grow_time = temp['time_gap']
            # 中文plt
            plt.rcParams['font.sans-serif']=['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.style.use('ggplot')
            plt.figure(figsize=(10, 6))
            plt.scatter(grow_time, HM_volume, label='HM_volume')
            plt.scatter(grow_time, ED_volume, label='ED_volume')
            plt.xlabel('时间')
            plt.ylabel('体积')
            plt.legend()
            plt.title(f'第{i}类病人的血肿，水肿体积变化')
            # plt.savefig(f'q2_d_{i}.png', dpi=480, bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, f'q2_d_{i}.png'), dpi=480, bbox_inches='tight')

            # calculate the correlation matrix of HM and ED
            corr = pearsonr(HM_volume, ED_volume)[0]
            print(f'第{i}类病人的HM_volume和ED_volume的相关系数为{corr}')
            f.write(f'第{i}类病人的HM_volume和ED_volume的相关系数为{corr}\n')
            # if corr > 0.5:
            #     print(f'第{i}类病人的HM_volume和ED_volume的相关系数大于0.5')
            #     f.write(f'第{i}类病人的HM_volume和ED_volume的相关系数大于0.5\n')
            # else:
            #     print(f'第{i}类病人的HM_volume和ED_volume的相关系数小于0.5')
            #     f.write(f'第{i}类病人的HM_volume和ED_volume的相关系数小于0.5\n')
    prob_data_fisrt = prob_data.groupby('ID').first().reset_index()
    prob_data_last = prob_data.groupby('ID').last().reset_index()
    ED_change_rate = (prob_data_last['ED_volume'] - prob_data_fisrt['ED_volume']) / prob_data_fisrt['ED_volume']
    prob_data_fisrt['ED_change_rate'] = ED_change_rate
    HM_change_rate = (prob_data_last['HM_volume'] - prob_data_fisrt['HM_volume']) / prob_data_fisrt['HM_volume']
    prob_data_fisrt['HM_change_rate'] = HM_change_rate
    prob_data_fisrt.to_excel('q2_d_first.xlsx', index=False)
    for i in tqdm(range(4)):
        causal_analysis(prob_data_fisrt[prob_data_fisrt['label'] == i], '脑室引流', '营养神经', save_path=os.path.join(save_dir, f'ED_causal_label_{i}'),y_name='ED_change_rate')
        causal_analysis(prob_data_fisrt[prob_data_fisrt['label'] == i], '脑室引流', '营养神经', save_path=os.path.join(save_dir, f'HM_causal_label_{i}'),y_name='HM_change_rate')


        