import pandas as pd

data = pd.read_excel(r'E:\23研赛\建模2023\建模2023\2023年中国研究生数学建模竞赛赛题\E题\E题\处理后的表2.xlsx')
time_data = pd.read_excel(r'E:\23研赛\建模2023\建模2023\2023年中国研究生数学建模竞赛赛题\E题\E题\数据\竞赛发布数据\表1-患者列表及临床信息.xlsx')
df_result = pd.read_excel(r'E:\23研赛\建模2023\建模2023\2023年中国研究生数学建模竞赛赛题\E题\E题\数据\竞赛发布数据\表4-答案文件.xlsx')
# time_data['发病到首次影像检查时间间隔'] = pd.to_timedelta(time_data['发病到首次影像检查时间间隔'], unit='h')

def query_database(patient_id, data, time_data = time_data):

  volumes_q = data.loc[data['ID'] == patient_id, 'HM_volume']
  times_q = data.loc[data['ID'] == patient_id, '检查时间']
  time_add = time_data.loc[time_data['Unnamed: 0'] == patient_id, '发病到首次影像检查时间间隔']
  
  return volumes_q, times_q, time_add.values

def get_volumes(patient_id, data):
    # 根据patient_id从数据库中查询影像体积数据
    volumes, times, time_add = query_database(patient_id, data) 
    return volumes, times, time_add

def check_expanded(volumes_c, times_c, time_add, first):
    first_vol = volumes_c[first]
    first_time = times_c[first]
    len_ = len(volumes_c)
    
    expanded = 0 
    time = []
    delta_volume = []
    
    for i in range(first + 1,first + len_):
        if ((volumes_c[i] - first_vol)/first_vol >= 0.33 or volumes_c[i] - first_vol >= 6000) and ((times_c[i] - first_time).total_seconds() + time_add * 3600) / 3600 <= 48:
            expanded = 1
            time.append(((times_c[i] - first_time).total_seconds() + time_add * 3600) / 3600)
            delta_volume.append(volumes_c[i] - first_vol)
    if expanded:
        max_volume_index = delta_volume.index(max(delta_volume))
        final_time = time[max_volume_index][0]
    else:
        final_time = 0
    return expanded, final_time, first + len_

expands = []
times_ = []
first = 0
for index, row in df_result.iterrows():
    if index in range(2,102):
        patient_id = row.iloc[0]
        volumes, times, time_add = get_volumes(patient_id, data)
        expand, time, first= check_expanded(volumes, times, time_add, first)
        df_result.iloc[index, 2] = expand
        df_result.iloc[index, 3] = time
        
df_result.to_excel(r'E:\23研赛\建模2023\建模2023\2023年中国研究生数学建模竞赛赛题\E题\E题\数据\竞赛发布数据\表4-答案文件更新后.xlsx', index=False)
        
        