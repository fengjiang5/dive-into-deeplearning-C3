import torch
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import random


num_input = 2
num_example = 1000
true_w = [2, -3.4]
true_b = 2.33
feature = torch.randn(num_example, num_input, dtype=torch.float32)
label = true_w[0]*feature[:, 0] + true_w[1]*feature[:, 1] + true_b
label += torch.tensor(np.random.normal(0, 0.01, size=label.size()), dtype=torch.float32)
print(feature[0], label[0])


def use_svg_display():
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


set_figsize()
plt.scatter(feature[:, 1].numpy(), label.numpy(), 1)
plt.show()


def data_iter(batch_size, feature, label):
    num_example = len(feature)
    indice = list(range(num_example))
    random.shuffle(indice)
    for i in range(0, num_example, batch_size):
        j = torch.LongTensor(indice[i:min(i+batch_size, num_example)])
        yield feature.index_select(0, j), label.index_select(0, j)

batch_size = 10
w = torch.tensor(np.random.normal(0, 0.01, (num_input, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


def linreg(X, w, b):
    return torch.mm(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat -y.view(y_hat.size()))**2/2


def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr*param.grad/batch_size


lr = 0.03
epochs = 3
net = linreg
loss = squared_loss

for epoch in range(epochs):
    for X, y in data_iter(batch_size, feature, label):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()

    train_loss = loss(net(feature, w, b), label)
    print('epoch {}/3, loss:{:.3f}'.format(epoch+1, train_loss.mean().item()))


print(true_w, '\n', w)
print(true_b, '\n', b)