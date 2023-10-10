import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random

def load_data():
    train_data = pd.read_excel('q1_concat1.xlsx')
    train_label = pd.read_excel(r'数据\竞赛发布数据\表4-答案文件更新后.xlsx', skiprows=2)
    label = train_label['1是，0否']
    label_train = label[:100]
    label_test = label[100:]
    feature = train_data.iloc[:, 4:]
    feature_train = feature[:100]
    feature_test = feature[100:]
    return feature_train, label_train, feature_test, label_test

def preprocess_data(feature_train, label_train):
    train_set = pd.concat([feature_train, label_train], axis=1)
    label_name = train_set.columns[-1]
    raw_train = train_set.rename(columns={label_name: 'class'})
    train_set_1 = raw_train[raw_train['class'] == 1]
    train_set = pd.concat([raw_train, train_set_1, train_set_1], axis=0)
    train_set = train_set.sample(frac=1)
    return train_set

def train_autoML(train_data, label = 'class'):
    train_data = TabularDataset(train_data)
    predictor = TabularPredictor(label=label).fit(train_data)
    return predictor

def evaluate_model(predictor, raw_train, label = 'class'):
    acc = accuracy_score(predictor.predict(raw_train), raw_train[label])
    return acc

def main():
    feature_train, label_train, feature_test, label_test = load_data()
    train_set = preprocess_data(feature_train, label_train)
    
    # AutoML training
    predictor = train_autoML(train_set)
    
    leaderboard = predictor.leaderboard(silent=True)
    leaderboard.to_excel('q1_b_leaderboard.xlsx')

    total_feature = pd.read_excel('q1_concat1.xlsx')
    prob = predictor.predict_proba(total_feature)
    prob.to_excel('q1_b_prob.xlsx')
    
    feature_importance = predictor.feature_importance(train_set)
    feature_importance.to_excel('q1_b_feature_importance.xlsx')
    
    acc = evaluate_model(predictor, train_set)
    print(f'Accuracy of the train set is {acc}')

if __name__ == "__main__":
    main()
