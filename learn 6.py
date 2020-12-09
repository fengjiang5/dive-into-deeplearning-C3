import torch
import torchvision
import numpy as np
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l


params = {}
params['batch_size'] = 256
params['num_input'] = 784
params['num_output'] = 10
params['num_epochs'] = 5
params['lr'] = 0.1


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


train_iter, test_iter = load_data_fashion_mnist(params['batch_size'])

w = torch.tensor(np.random.normal(0, 0.01, (params['num_input'], params['num_output']),),
                 dtype=torch.float)
b = torch.zeros(params['num_output'], dtype=torch.float)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


def softmax_f(x):
    x_exp = x.exp()
    partition = x_exp.sum(dim=1, keepdim=True)
    return x_exp/partition


def net(x):
    return softmax_f(torch.mm(x.view(-1, params['num_input']), w) + b)


def cross_entropy(y_hat, y):
    # tensor.gather(dim=0 or 1, index) 按列或者按行索引，index为索引下标
    # 详见: r‘https://blog.csdn.net/hawkcici160/article/details/80771044’
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


def accuracy(y_hat, y):
    # y_hat.argmax(dim=1) 返回行最大下标，后转为浮点型求平均来得到准确率
    return (y_hat.argmax(dim=1) == y).float().mean().item()


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        acc_sum += (net(x).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum/n


def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
    # 沿batch维求了平均了。
    for param in params:
        param.data -= lr * param.grad / batch_size
# print(evaluate_accuracy(test_iter, net))


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
        print('epoch {}/5 loss: {:.3f} acc: {:.3f} test_acc:{:.3f}'
              .format(epoch+1, train_l_sum/n, train_acc_sum/n, test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs=params['num_epochs'],
          batch_size=params['batch_size'], params=[w, b], lr=params['lr'])
