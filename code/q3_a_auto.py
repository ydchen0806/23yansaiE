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
    save_path = 'q3_a_results'
    os.makedirs(save_path, exist_ok=True)
    q3_data = mRSDataset(prase = 'train', scale_data = False, split_rate=1, use_pca=False)
    q3_train, q3_test, data_total = q3_data.get_dataset()
    label = '90天mRS'
    print(q3_train.shape, q3_test.shape)
    predictor = train_autoML(q3_train, label = label)
    acc = evaluate_model(predictor, q3_train, label = label)
    print(f'acc on the test set is {acc}')
    feature_importance = predictor.feature_importance(q3_train)
    feature_importance.to_csv(os.path.join(save_path,'q3_a_feature_importance.csv'))
    result = predictor.leaderboard()
    result.to_csv(os.path.join(save_path,'q3_a_leaderboard.csv'))
    top_10_feature = feature_importance.iloc[:10, :].index
    print(f'top 10 features are {top_10_feature}')
    top_10_train = q3_train[top_10_feature]
    top_10_test = q3_test[top_10_feature]
    top_10_train = pd.concat([top_10_train, q3_train[label]], axis = 1)
    top_10_test = pd.concat([top_10_test, q3_test[label]], axis = 1)
    # top_10_train[label] = q3_train[label]
    # top_10_test[label] = q3_test[label]
    predictor_top_10 = train_autoML(top_10_train, label = label)
    acc_top_10 = evaluate_model(predictor_top_10, top_10_train, label = label)
    print(f'acc on the test set with top 10 features is {acc_top_10}')
    top_10_result = predictor_top_10.leaderboard()
    top_10_result.to_csv(os.path.join(save_path,'q3_a_top_10_leaderboard.csv'))
    data_total['90天mRS_all_feature'] = predictor.predict(data_total)
    data_total['90天mRS_top10'] = predictor_top_10.predict(data_total)
    data_total.to_csv(os.path.join(save_path,'q3_a_result.csv'))