
import numpy as np
import matplotlib.pyplot as plt
# from causalnex.match import Match
from causalnex.structure.notears import from_pandas
from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.structure import StructureModel
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler

def extract_feature(data, start_name, end_name):
    start_index = data.columns.tolist().index(start_name)
    end_index = data.columns.tolist().index(end_name)
    feature = data.iloc[:, start_index:end_index+1]
    id2feature = {}
    for i in range(feature.shape[1]):
        id = data['Unnamed: 0'].values
        temp_feature = data.iloc[:, start_index + i].values
        temp_feature_name = data.columns.tolist()[start_index + i]
        id2feature[temp_feature_name] = dict(zip(id, temp_feature))
    return feature, id2feature


def causal_analysis(data, start_name, end_name, save_path=None, y_name='ED_change_rate'):
    feature = data.loc[:, start_name:end_name]
    treat_ment_name = feature.columns.tolist()
    y_name = y_name
    donot_need_name = ['ED_change_diff','首次检查流水号','ID','delta_ED_volume']
    other_name = [i for i in data.columns.tolist() if i not in treat_ment_name and i != y_name and i not in donot_need_name]
    # scaler = StandardScaler()
    # data[other_name] = scaler.fit_transform(data[other_name])
    # # data = data.dropna()
    

    # X = data[treat_ment_name + other_name]
    # y = data[y_name]
    # reg = LinearRegression().fit(X, y)
    # r2 = r2_score(y, reg.predict(X))
    # print(f'r2 is {r2}')
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, 'causal_analysis.txt')
        with open(save_path, 'a') as f:
            for i in range(len(treat_ment_name)):
                control = data[data[treat_ment_name[i]] == 0]
                treatment = data[data[treat_ment_name[i]] == 1]
                if control.shape[0] > 0 and treatment.shape[0] > 0:
                    trt = treat_ment_name[i]
                    t, p = ttest_ind(control[y_name], treatment[y_name])
                    print(f'{treat_ment_name[i]}: {trt}, t is {t}, p is {p}')
                    print(f'mean diff is {np.mean(treatment[y_name]) - np.mean(control[y_name])}') 
                    f.write(f'{treat_ment_name[i]}: {trt}, t is {t}, p is {p}, mean diff is {np.mean(treatment[y_name]) - np.mean(control[y_name])}\n')
                    
                else:
                    print(f'{treat_ment_name[i]}: {trt}, no control or treatment')
                    f.write(f'{treat_ment_name[i]}: {trt}, no control or treatment\n')



if __name__ == '__main__':
    q2_data = pd.read_excel('q2_total_data_all.xlsx')
    feature_data = pd.read_excel(r'\数据\竞赛发布数据\表1-患者列表及临床信息.xlsx')
    start_name = '脑室引流'
    end_name = '营养神经'
    feature, id2feature = extract_feature(feature_data, start_name, end_name)
    for key in id2feature.keys():
        q2_data[key] = q2_data['ID'].map(id2feature[key])
    q3_data_first = q2_data.groupby('ID').first().reset_index()
    q3_data_last = q2_data.groupby('ID').last().reset_index()
    ED_change_diff = q3_data_last['ED_volume'] - q3_data_first['ED_volume']
    q3_data_first['ED_change_diff'] = ED_change_diff
    ED_change_rate = q3_data_last['ED_volume'] / q3_data_first['ED_volume']
    q3_data_first['ED_change_rate'] = ED_change_rate
    feature_data = pd.concat([feature_data['Unnamed: 0'], feature_data.loc[:, '年龄':'血压']], axis=1)
    q3_data_first.drop(['Unnamed: 0'], axis=1, inplace=True)
    q3_data_first = pd.merge(q3_data_first, feature_data, left_on='ID', right_on='Unnamed: 0', how='inner')
    q3_data_first['高压'] = q3_data_first['血压'].apply(lambda x: int(x.split('/')[0]))
    q3_data_first['低压'] = q3_data_first['血压'].apply(lambda x: int(x.split('/')[1]))
    q3_data_first.drop(['血压'], axis=1, inplace=True)

    # table2 = pd.read_excel(r'\处理后的表2.xlsx')
    # table2 = table2.sort_values(by=['ID', '检查时间'])
    # table2_first = table2.groupby('ID').first().reset_index()
    # table2_first = pd.concat([table2_first['ID'], table2_first.loc[:, 'HM_volume':'ED_Cerebellum_L_Ratio']], axis=1)
    # q3_data_first = pd.merge(q3_data_first, table2_first, on='ID', how='inner')
    # q3_data_first.drop_duplicates(subset=['ID'], keep='first', inplace=True)
    q3_data_first.drop(['检查时间','发病到首次影像检查时间间隔_x','time_from_start','Unnamed: 0',\
                        'time_gap'], axis=1, inplace=True)
    sex_map = {'男': 1, '女': 0}
    q3_data_first['性别'] = q3_data_first['性别'].apply(lambda x: int(sex_map[x]))

    # sm.to_excel('q3_causal.xlsx')
    
    q3_data_first.to_excel('q3_data_first.xlsx', index=False)

    causal_analysis(q3_data_first, start_name, end_name, save_path='q3_causal_analysis')
    # q2_data.to_excel('q3_total_data.xlsx', index=False)