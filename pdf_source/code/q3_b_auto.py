from q3_a import mRSDataset
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import accuracy_score
import pandas as pd
import os


def train_autoML(train_data, label = 'class'):
    train_data = TabularDataset(train_data)
    predictor = TabularPredictor(label=label).fit(train_data)
    return predictor

def evaluate_model(predictor, raw_train, label = 'class'):
    acc = accuracy_score(predictor.predict(raw_train), raw_train[label])
    return acc

if __name__ == '__main__':
    save_dir = 'q3_b_results'
    os.makedirs(save_dir, exist_ok=True)
    total_data = pd.read_excel(r'E:\外包作业\23研赛\建模2023\建模2023\2023年中国研究生数学建模竞赛赛题\E题\E题\第三问\竞赛发布数据\q3_b2_total.xlsx')
    train_id = ['sub{:03d}'.format(i) for i in range(101)]
    pred_id = ['sub{:03d}'.format(i) for i in range(101, 161)]
    label = '90天mRS'
    train_set = total_data[total_data['ID'].isin(train_id)]
    pred_set = total_data[total_data['ID'].isin(pred_id)]
    train_set.drop(['ID','Unnamed: 0'], axis = 1, inplace = True)
    pred_set.drop(['ID','Unnamed: 0'], axis = 1, inplace = True)
    predictor = train_autoML(train_set, label = label)
    acc = evaluate_model(predictor, train_set, label = label)
    print(f'acc on the test set is {acc}')
    feature_importance = predictor.feature_importance(train_set)
    result = predictor.leaderboard()
    result.to_csv(os.path.join(save_dir,'q3_b_leaderboard.csv'))
    top_10_feature = feature_importance.iloc[:10, :].index
    feature_importance.to_csv(os.path.join(save_dir,'q3_b_feature_importance.csv'))
    print(f'top 10 features are {top_10_feature}')
    top_10_train = train_set[top_10_feature]
    top_10_train = pd.concat([top_10_train, train_set[label]], axis = 1)
    # top_10_train[label] = q3_train[label]
    # top_10_test[label] = q3_test[label]
    predictor_top_10 = train_autoML(top_10_train, label = label)
    acc_top_10 = evaluate_model(predictor_top_10, top_10_train, label = label)
    print(f'acc on the test set with top 10 features is {acc_top_10}')
    top_10_result = predictor_top_10.leaderboard()
    top_10_result.to_csv(os.path.join(save_dir,'q3_b_top_10_leaderboard.csv'))
    total_data['90天mRS_all_feature'] = predictor.predict(total_data)
    total_data['90天mRS_top10'] = predictor_top_10.predict(total_data)
    total_data.to_csv(os.path.join(save_dir,'q3_b_result.csv'))