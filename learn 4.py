import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optim


# 生成数据集
num_input = 2
num_example = 1000
true_w = [2.4, 3]
true_b = 5.3
feature = torch.tensor(np.random.normal(0, 1, (num_example, num_input)), dtype=torch.float)
label = true_w[0]*feature[:, 0] + true_w[1]*feature[:, 1] + true_b
label += torch.tensor(np.random.normal(0, 0.01, size=label.size()), dtype=torch.float)

# 读取数据 利用PyTorch提供的data包
batch_size = 10
dataset = Data.TensorDataset(feature, label)
dataloader = Data.DataLoader(dataset, batch_size, shuffle=True)


# 定义模型，利用nn.Module
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_input)
print(net)  # 打印网络

# 初始化参数
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

loss = nn.MSELoss()

# 定义优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in dataloader:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()  # 梯度清零
        optimizer.step()
    print('epoch {}/3 loss {:.3f}'.format(epoch+1, l))


# 与真实情况比较
dense = net
print(true_w, dense.linear.weight)
print(true_b, dense.linear.bias)

