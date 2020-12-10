import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import sys
import numpy as np
from collections import OrderedDict


params = {}
params['batch_size'] = 256
params['num_input'] = 784
params['num_output'] = 10
params['lr'] = 0.1
params['epochs'] = 5


def load_data_fashion_mnist(batch_size, resize=None, root=r'E:\Fashion-MNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True,
                                                    download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False,
                                                   download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
    # 沿batch维求了平均了。
    for param in params:
        param.data -= lr * param.grad / batch_size


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        acc_sum += (net(x).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum/n


class LinearNet(nn.Module):
    def __init__(self, num_input, num_output):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_input, num_output)

    def forward(self, x):  # x.shape()=batch_size, 1, 28, 28
        y = self.linear(x.view(x.shape[0], -1))  # 转为batch_size,784
        return y


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


train_iter, test_iter = load_data_fashion_mnist(params['batch_size'])
net = LinearNet(params['num_input'], params['num_output'])
# net = nn.Sequential(
#     OrderedDict([
#         ('flatten', FlattenLayer())
#         ('linear', nn.Linear(params['num_input'], params['num_output']))
#     ])  # 此处出错 typrError: 'tuple' object is not callable. 没找到解决办法，先暂时屏蔽。
# )
# 初始化权重
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), params['lr'])


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for x, y in train_iter:
            y_hat = net(x)
            l = loss(y_hat, y).sum()

            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch {} loss: {:.3f} acc: {:.3f} test_acc:{:.3f}'
              .format(epoch+1, train_l_sum/n, train_acc_sum/n, test_acc))


train_ch3(net, train_iter, test_iter, loss, params['epochs'],
          params['batch_size'], None, None, optimizer)
