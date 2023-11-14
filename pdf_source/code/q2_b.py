# kmeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf
import statsmodels.tsa.seasonal as seasonal
import matplotlib.pyplot as plt
from q2_a import *
from q1_b_model import train_autoML

def concat_feature(first_data_path, feature_path):
    first_data = pd.read_excel(first_data_path)
    feature = pd.read_excel(feature_path)
    id_col = feature.columns[0]
    concat_data = pd.merge(first_data, feature, right_on=id_col, left_on='ID', how='inner')
    concat_data['高压'] = concat_data['血压'].apply(lambda x: int(x.split('/')[0]))
    concat_data['低压'] = concat_data['血压'].apply(lambda x: int(x.split('/')[1]))
    concat_data.drop(['血压','Unnamed: 0_x','Unnamed: 0_y', '发病到首次影像检查时间间隔_x', 'time_gap',\
                      'time_from_start','数据集划分','入院首次影像检查流水号'], axis=1, inplace=True)
    start_index = concat_data.columns.tolist().index('检查时间')
    print(f'concat_data shape is {concat_data.shape}')
    return concat_data, start_index

def kmeans_cluster(data_in,choose_feature, n_clusters=3):
    data = data_in[choose_feature]
    data = data.dropna()
    data = data.drop(['检查时间'], axis=1)
    # scaler = MinMaxScaler()
    # data = scaler.fit_transform(data)

    # kmeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    labels = kmeans.labels_
    data_in['label'] = labels
    data_in['label'].value_counts()
    # 评价聚类效果
    score = silhouette_score(data, labels)
    print(f'kmeans score is {score}')
    id_col = data_in['ID']
    label_col = data_in['label']
    id2label = dict(zip(id_col, label_col))
    cluster_center = kmeans.cluster_centers_
    cluster_center = pd.DataFrame(cluster_center, columns=data.columns)
    return data_in, id2label, cluster_center, score



if __name__ == '__main__':
    import os
    save_dir = 'q2_b_results'
    os.makedirs(save_dir,exist_ok=True)
    feature_path = r'\数据\竞赛发布数据\表1-患者列表及临床信息.xlsx'
    first_data_path = r'\first_data.xlsx'
    concat_data, start_index = concat_feature(first_data_path, feature_path)
    train_data = concat_data.iloc[:, 2:]
    # train_data = concat_data.iloc[:, start_index:]
    # train_data = pd.concat([concat_data['ED_volume'], train_data], axis=1)
    label = 'ED_volume'
    predictor = train_autoML(train_data, label)
    truth_value = train_data[label]
    pred_value = predictor.predict(train_data)
    rmse = np.sqrt(mean_squared_error(truth_value, pred_value))
    print(f'RMSE of the train set is {rmse}')
    feature_importance = predictor.feature_importance(train_data)
    feature_importance.to_excel(os.path.join(save_dir,'q2_b_feature_importance.xlsx'))
    top_10_feature = feature_importance.iloc[:10, 0].index.tolist()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_mse = int(1e6)
    save_dict = {}
    cluster_num = []
    cluster_score = []
    predict_score = []
    for cluster in range(2,9):
        cluster_num.append(cluster)
        data_in, id2label, cluster_center, score = kmeans_cluster(concat_data, top_10_feature, n_clusters=cluster)
        cluster_score.append(score)
        data_in.to_excel(os.path.join(save_dir,f'q2_b_cluster_{cluster}.xlsx'), index=False)
        cluster_center.to_excel(os.path.join(save_dir,f'q2_b_cluster_{cluster}_center.xlsx'), index=False)
        print(f'cluster {cluster} finished')
        data_train = pd.read_excel(r'\q2_total_data.xlsx')
        data_train['label'] = data_train['ID'].map(id2label)
        mse_error = []
        need_col = ['ID', 'label', 'ED_volume', 'time_from_start', 'time_gap', 'y_pred', 'y_truth', 'diff']
        record_dataframe = pd.DataFrame(columns=need_col)
        for i in range(cluster):
            temp_data = data_train[data_train['label'] == i]
            # x = 
            X = temp_data['time_gap'].values
            y = temp_data['ED_volume'].values
            x_from_start = temp_data['first_ED_volume'].values
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            scaler_x2 = MinMaxScaler()
            X = scaler_X.fit_transform(X.reshape(-1, 1))
            y = scaler_y.fit_transform(y.reshape(-1, 1))
            x_from_start = scaler_x2.fit_transform(x_from_start.reshape(-1, 1))
            X = np.concatenate((X, x_from_start), axis=1)

            X = torch.tensor(X, dtype=torch.float32).reshape(-1, 2)
            y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
            model = train_bilstm_model(X, y, num_epochs=100)
            y_pred, y_truth = evaluate_bilstm_model(model, X, y, scaler_y)
            mse_error.extend(np.sqrt((y_pred - y_truth)**2).tolist())
            temp_dict = {'ID': temp_data['ID'].values, 'label': temp_data['label'].values, 'ED_volume': temp_data['ED_volume'].values, \
                            'time_from_start': temp_data['time_from_start'].values, 'time_gap': temp_data['time_gap'].values, \
                                'y_pred': y_pred.reshape(-1), 'y_truth': y_truth.reshape(-1), 'diff': (y_pred.reshape(-1) - y_truth.reshape(-1)).reshape(-1)}
            temp_data = pd.DataFrame(temp_dict)
            record_dataframe = pd.concat([record_dataframe, temp_data], axis=0)
            
        

        mse_error = np.array(mse_error).mean()
        predict_score.append(mse_error)
        if mse_error < best_mse:
            best_mse = mse_error
            best_cluster = cluster
            print(f'best cluster is {best_cluster}, best mse is {best_mse}')
        

        else:
            print(f'not best cluster, best cluster is {best_cluster}, best mse is {best_mse}, current cluster is {cluster}, current mse is {mse_error}')
        record_dataframe.sort_values(by='ID', inplace=True)
        record_dataframe.to_excel(os.path.join(save_dir,f'q2_b_cluster_{cluster}_record.xlsx'), index=False)
        x = record_dataframe['time_gap'].values
        plt.figure(figsize=(10, 6))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('ggplot')
        # sns.scatterplot(x=x, y=record_dataframe['y_truth'].values, label='真实分布',hue=record_dataframe['label'].values, palette='Set2')
        sns.scatterplot(x=x, y=record_dataframe['y_truth'].values, color='blue', label='真实分布')
        sns.scatterplot(x=x, y=record_dataframe['y_pred'].values, color='red', label='预测分布')
        plt.xlabel('time_gap')
        plt.ylabel('水肿大小')
        plt.title(f'{cluster}聚类下的真实值与预测值散点图')
        plt.savefig(f'q2_b_cluster_{cluster}_scatter.png', dpi=480, bbox_inches='tight')
        plt.show()
        plt.close()
    save_dict = {'cluster_num':cluster_num, 'cluster_score': cluster_score, 'prediction score':predict_score}
    save_df = pd.DataFrame(save_dict)
    save_df.to_excel(os.path.join(save_dir,'聚类灵敏度分析.xlsx'))





    