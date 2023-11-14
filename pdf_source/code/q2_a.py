import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf
import statsmodels.tsa.seasonal as seasonal

# Load and preprocess data
def load_and_preprocess_data():
    data_total = pd.read_excel(r'\处理后的表2.xlsx')
    time_data = pd.read_excel(r'\数据\竞赛发布数据\表1-患者列表及临床信息.xlsx')
    time_id_col = time_data.iloc[:, 0].values
    time_get_to_hospital = time_data.loc[:, '发病到首次影像检查时间间隔'].values
    id2time = dict(zip(time_id_col, time_get_to_hospital))
    data_total['发病到首次影像检查时间间隔'] = data_total['ID'].map(id2time)
    unique_id = data_total['ID'].unique()
    unique_id = unique_id[:100]
    data_total = data_total[data_total['ID'].isin(unique_id)]
    time_ED = []
    volumes_ED = []
    
    for id in unique_id:
        temp_data = data_total[data_total['ID'] == id]
        temp_time = temp_data['检查时间'].diff().dt.total_seconds().values
        temp_delta_ed = temp_data['ED_volume'].diff().values / temp_data['ED_volume'].values
        temp_time = np.nan_to_num(temp_time, nan=0)
        temp_time = temp_time / 3600 + temp_data['发病到首次影像检查时间间隔'].values[0]
        temp_time = temp_time.tolist()
        temp_delta_ed = temp_delta_ed.tolist()
        time_ED.extend(temp_time)
        volumes_ED.extend(temp_delta_ed)
    time_ED = np.array(time_ED)
    data_total['time_gap'] = time_ED
    data_total['time_from_start'] = data_total['检查时间'] - data_total['检查时间'].min()
    data_total['delta_ED_volume'] = volumes_ED

    
    first_data = data_total.groupby('ID').first().reset_index()
    first_data_id = first_data['ID'].values
    first_data_volume = first_data['ED_volume'].values
    id2volume = dict(zip(first_data_id, first_data_volume))
    data_total['first_ED_volume'] = data_total['ID'].map(id2volume)
    data_total.to_excel('q2_total_data.xlsx', index=False)
    first_data.to_excel('first_data.xlsx', index=False)
    # first_data = data_total
    first_data['check_time'] = first_data['发病到首次影像检查时间间隔']
    X = data_total['time_gap'].values
    y = data_total['ED_volume'].values
    x_from_start = data_total['first_ED_volume'].values
    season_period = 12
    seasonal_decompose = seasonal.seasonal_decompose(y, period=season_period)
    seasonal_decompose.plot()
    # plt.savefig('seasonal_decompose.png', dpi=480, bbox_inches='tight')
    # plt.show()
    y_trend = seasonal_decompose.trend
    y_seasonal = seasonal_decompose.seasonal
    y_res = seasonal_decompose.resid
    return X, y, first_data, x_from_start, data_total

# Train mixed effects model
def train_mixed_effects_model(X, y, first_data):
    import statsmodels.formula.api as smf

    model = smf.mixedlm("ED_volume ~ time_gap", data=first_data, groups=first_data["ID"])
    result = model.fit()
    pred_y = result.predict(first_data)
    truth_y = y
    rmse = np.sqrt(np.mean((pred_y - truth_y) ** 2))
    pred_y = np.array(pred_y)
    truth_y = np.array(truth_y)

    return result, rmse, pred_y, truth_y

# Train BiLSTM model
def train_bilstm_model(X_train, y_train, num_epochs = 100):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    
    class BiLSTMRegressor(nn.Module):
        def __init__(self, hidden_size, num_layers, input_size=1):
            super(BiLSTMRegressor, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out
        
    input_size = X_train.shape[1]
    hidden_size = 64
    num_layers = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTMRegressor(hidden_size, num_layers,input_size)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, amsgrad=True)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)

    num_epochs = num_epochs
    batch_size = 96

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs = inputs.unsqueeze(1).to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

# Evaluate BiLSTM model
def evaluate_bilstm_model(model, X_test, y_test, scaler_y):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        X_test = X_test.unsqueeze(1).to(device)
        y_pred = model(X_test)

    y_pred = scaler_y.inverse_transform(y_pred.cpu().numpy())
    y_test = scaler_y.inverse_transform(y_test.cpu().numpy())

    return y_pred, y_test

# Main function
def main():
    X, y, first_data, x_from_start, data_total = load_and_preprocess_data()
    result, rmse_mix, y_pred, y_truth = train_mixed_effects_model(X, y, data_total)
    print(result.summary())
    print(f"均方根误差mix (RMSE): {rmse_mix}")

    delta_y = y_pred - y_truth
    # delta_y = np.array(delta_y.values)
    # BiLSTM model training and evaluation
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_x2 = MinMaxScaler()
    X = scaler_X.fit_transform(X.reshape(-1, 1))
    y = scaler_y.fit_transform(y.reshape(-1, 1))
    x_from_start = scaler_x2.fit_transform(x_from_start.reshape(-1, 1))
    X = np.concatenate((X, x_from_start), axis=1)

    X = torch.tensor(X, dtype=torch.float32).reshape(-1, 2)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    # x_from_start = torch.tensor(x_from_start, dtype=torch.float32).reshape(-1, 1)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = train_bilstm_model(X_train, y_train)

    delta_y_pred, delta_y_test = evaluate_bilstm_model(model, X, y, scaler_y)

    print(f"BiLSTM均方根误差 (RMSE): {np.sqrt(mean_squared_error(delta_y_test, delta_y_pred))}")
    # print(f"R平方 (R^2): {r2_score(y_test, y_pred)}")
    # y_pred_agg = y_pred + delta_y_pred.reshape(-1)
    y_pred_agg = delta_y_pred
    x1 = X[:, 0].reshape(-1)
    x2 = X[:, 1].reshape(-1)
    x1 = scaler_X.inverse_transform(x1.reshape(-1, 1))
    x2 = scaler_x2.inverse_transform(x2.reshape(-1, 1))
    print(f'集成模型均方根误差 (RMSE): {np.sqrt(mean_squared_error(y_truth, y_pred_agg))}')
    dict_diff = {'id': data_total['ID'].values ,'time_gap':x1.reshape(-1), 'time_abs':x2.reshape(-1) ,\
                 'y_pred': y_pred.reshape(-1), 'y_pred_agg': y_pred_agg.reshape(-1), 'y_truth': y_truth.reshape(-1),\
                    'diff': (y_pred_agg.reshape(-1) - y_truth.reshape(-1)).reshape(-1)}
    df_diff = pd.DataFrame(dict_diff)
    df_diff = df_diff.groupby('id').mean()
    df_diff.to_excel('q2diff.xlsx')

    # Visualize results
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('ggplot')

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x1.reshape(-1), y=y_truth.reshape(-1), color='blue', label='真实分布')
    # sns.kdeplot(x=X.numpy().reshape(-1), y=y_pred.reshape(-1), cmap='Reds', shade=True, shade_lowest=False, label='预测分布')
    sns.scatterplot(x=x1.reshape(-1), y=y_pred_agg.reshape(-1), color='red', label='拟合趋势线')
    plt.xlabel('时间间隔/h')
    plt.ylabel('水肿大小')
    plt.savefig('q2a拟合趋势.png', dpi=480, bbox_inches='tight')
    plt.show()

    plt.plot(y_truth, label='真实值')
    plt.plot(y_pred_agg, label='集成模型预测值')
    plt.plot(y_pred, label='混合效应模型预测值')
    plt.legend()
    plt.ylabel('水肿大小')
    plt.xlabel('样例')
    plt.savefig('p2a模型预测结果.png', dpi=480, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()

