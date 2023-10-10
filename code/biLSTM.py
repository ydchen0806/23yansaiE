import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


class tableDataset_Val(torch.utils.data.Dataset):
    def __init__(self, dataPath, time_step, Y_len = 7, choose_class = 0, scaler = False, step = 1, random_slicing = True):
        self.path = dataPath
        self.time_step = time_step
        self.Y_len = Y_len
        self.data = pd.read_excel(self.path)
        self.data.fillna(0, inplace=True)
        self.time = self.data.iloc[:,0]
        self.choose_class = choose_class
        self.scaler = scaler
        self.step = step
        self.random = random_slicing
        assert self.choose_class < self.data.shape[1] - 1 and self.choose_class >= 0, "choose_class must be in [0, 6]"
        self.feature = self.data.iloc[:,1 + choose_class] 
        # x, y = sliding_window(self.feature, self.time_step, self.Y_len)
        self.mean = self.feature.mean(axis=0)  # 计算特征的均值
        self.std = self.feature.std(axis=0)    # 计算特征的标准差
    
    def __getitem__(self, index):
        if self.random:
            x, y = self.random_sliding(self.feature, self.time_step, self.Y_len, index)
        else:
            x, y = self.sliding_window(self.feature, self.time_step, self.Y_len, index)
        if self.scaler:
            # 标准化输入特征
            x = (x - self.mean) / self.std
        
        # 将numpy数组转换为PyTorch张量
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        
        return x, y
    
    def __len__(self):
        assert self.data.shape[0] >= self.time_step + self.Y_len, "data length must be larger than time_step + Y_len"
        h = self.data.shape[0]
        return int((h - self.time_step - self.Y_len) / self.step)
    
    def sliding_window(self, data, time_step, Y_len, index):
        assert len(data.shape) == 1, "data must be 1-dim"
        h= data.shape[0]
        step = self.step
        X = np.zeros((time_step))
        Y = np.zeros((Y_len))
        start = index * step
        X = data.iloc[start:start + time_step].values
        # X[time_step] = self.choose_class / 5
        Y = data.iloc[start + time_step:start + time_step + Y_len].values
        X = X.reshape(1, -1)
        Y = Y.reshape(1, -1)
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        return X, Y
    
    def random_sliding(self,data,time_step,Y_len, index):
        assert len(data.shape) == 1, "data must be 1-dim"
        h= data.shape[0]
        start = random.randint(0, h - time_step - Y_len)
        X = np.zeros((time_step))
        Y = np.zeros((Y_len))
        X = data.iloc[start:start + time_step].values
        # X[time_step] = self.choose_class / 5
        Y = data.iloc[start + time_step :start + time_step + Y_len ].values
        X = X.reshape(1, -1)
        Y = Y.reshape(1, -1)
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        return X, Y

# 创建双向LSTM模型
class BiLSTM(nn.Module):
    def __init__(self, input_size=21, hidden_size=64, num_layers=3, output_size=7):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2是因为双向LSTM

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 初始化隐藏状态
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 初始化细胞状态
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -7:, :])  # 预测未来7天销售额
        return out



# 训练模型

def train(model, num_iters, train_loader,choose_class = 0):
    model.train()
    save_dir = os.path.join('/braindat/lab/chenyd/code_230508/guosai23/C题/dealed_data',f'model_{choose_class}')
    os.makedirs(save_dir, exist_ok=True)
    best_MSE = 100000
    best_ites = 0
    visual_bar = tqdm(range(num_iters), desc="iters")
    visual_bar.set_description(f"iters [{0}/{num_iters}], Loss: {0:.4f}")
    for i in range(num_iters):
        for j, (train_X, train_Y) in enumerate(train_loader):
            if torch.cuda.is_available():
                train_X = train_X.cuda()
                train_Y = train_Y.cuda()
            outputs = model(train_X)
            loss = criterion(outputs, train_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        visual_bar.update(1)
        visual_bar.set_description(f"iters [{i + 1}/{num_iters}], Loss: {loss.item():.4f}")
        visual_bar.refresh()

        # if loss.item() < best_loss:
        #     best_loss = loss.item()
        #     torch.save(model.state_dict(), f'/braindat/lab/chenyd/code_230508/guosai23/C题/dealed_data/model/biLSTM_best.pth')
        #     best_ites = i
        # print(i)
        if (i + 1) % 10 == 0 or i == 0:
            mse, pred_y, truth_y = val(model, val_loader)
            if mse < best_MSE:
                best_MSE = mse
                torch.save(model.state_dict(), os.path.join(save_dir, f'biLSTM_best.pth'))
                print(f'iters [{i + 1}/{num_iters}], current MSE: {mse:.4f}, best MSE: {best_MSE:.4f}, model saved!')
                best_ites = i
                plt.figure(figsize=(10, 5))
                plt.plot(np.array(pred_y).reshape(-1), label='pred')
                plt.plot(np.array(truth_y).reshape(-1), label='truth')
                plt.legend()
                plt.savefig(os.path.join(save_dir,'biLSTM_best.png'), dpi=480, bbox_inches='tight')
                plt.close()

            else:
                print(f'iters [{i + 1}/{num_iters}], current MSE: {mse:.4f}, best MSE: {best_MSE:.4f}, model not saved!')
    visual_bar.close()
    print('Finished Training')
    print(f'best MSE is {best_MSE:.4f}, best ites is {best_ites}')

def val(model, val_loader):
    model.eval()
    MSE = 0
    pred_y = []
    truth_y = []
    with torch.no_grad():
        for val_X, val_Y in (val_loader):
            if torch.cuda.is_available():
                val_X = val_X.cuda()
                val_Y = val_Y.cuda()
            outputs = model(val_X)
            # loss = criterion(outputs, val_Y)
            # pred_y.append(items for items in outputs.cpu().numpy())
            # truth_y.append(items for items in val_Y.cpu().numpy())
            pred_y.extend(outputs.cpu().numpy())
            truth_y.extend(val_Y.cpu().numpy())
    MSE = np.mean((np.array(pred_y) - np.array(truth_y)) ** 2)
    model.train()
    return MSE, pred_y, truth_y
     
if __name__ == '__main__':
    import parser
    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--choose_class', type=int, default=0)
    args = parser.parse_args()
    print(args.choose_class)
    y_data = '/braindat/lab/chenyd/code_230508/guosai23/C题/dealed_data/daily_sell_kg_class.xlsx'
    y_len = 7
    time_step = 21
    choose_class = args.choose_class
    dataset = tableDataset_Val(dataPath=y_data, time_step=time_step, Y_len=y_len, choose_class=choose_class,scaler=True, step=1, random_slicing=True)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    val_dataset = tableDataset_Val(dataPath=y_data, time_step=time_step, Y_len=y_len, choose_class=choose_class, scaler=True, step=7,random_slicing=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)

    input_size = 21  # 输入特征维度
    hidden_size = 64  # 隐藏层维度
    num_layers = 3  # LSTM层数
    output_size = 7  # 输出维度
    # 创建模型
    model = BiLSTM(input_size, hidden_size, num_layers, output_size)
    if torch.cuda.is_available():
        model = model.cuda()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    num_iters = 2000
    train(model, num_iters, train_loader)

# # 预测未来7天销售额（假设有新的输入数据）
# future_data = torch.Tensor(new_X)  # 新的输入数据，包括销售额、蔬菜品类和历史时间窗口
# future_pred = model(future_data)
# print("未来7天销售额预测：", future_pred)
