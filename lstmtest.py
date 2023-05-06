import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置随机种子，以便复现结果
torch.manual_seed(1)

# 创建训练集
seq_length = 20
time_steps = np.linspace(0, np.pi, seq_length + 1)
data = np.sin(time_steps)
data.resize((seq_length + 1, 1)) # 将数据转换成列向量
amplitude = np.linspace(0.1, 1, 5)
data = (amplitude.reshape(-1,1) * data.T).T
train_data = torch.tensor(data[:-1]).float().view(seq_length, -1, 1)
train_target = torch.tensor(data[1:]).float().view(seq_length, -1, 1)

# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=10, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.fc(lstm_out.view(len(input_seq), -1))
        return predictions

# 初始化模型和优化器
model = LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# 训练模型
num_epochs = 60
losses = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(train_data)
    loss = criterion(outputs, train_target)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}  Loss: {loss.item()}")

# 使用模型进行预测
test_data = torch.tensor(data).float().view(seq_length, -1, 1)
predictions = []
input_seq = train_data[-1]
model.eval()
with torch.no_grad():
    for i in range(len(test_data)):
        input_seq = model(input_seq)
        predictions.append(input_seq.numpy())
        input_seq = input_seq.reshape(1, len(amplitude), 1)

# 绘制结果
plt.plot(data.flatten(), 'r', label='True data')
plt.plot(np.array(predictions).flatten(), 'b', label='Predictions')
plt.legend(loc='best')
plt.savefig(os.path.join('pic5', 'lstm_prediction.png'))
plt.show()
