# 多项式函数拟合实验
import torch
import numpy as np
import torch.utils.data
from IPython import display
import matplotlib.pyplot as plt


params = {}
params['n_train'] = 100
params['n_test'] = 100
params['true_w'] = [1.2, -3.4, 5.6]
params['true_b'] = 5
params['num_epochs'] = 100
params['lr'] = 0.01

featrue = torch.randn((params['n_train'] + params['n_test'], 1))
poly_featrue2 = torch.cat((featrue, torch.pow(featrue, 2)), 1)

poly_feature = torch.cat((featrue, torch.pow(featrue, 2), torch.pow(featrue, 3)), 1)  # 按列拼接
labels = (params['true_w'][0]*poly_feature[:, 0] + params['true_w'][1]*poly_feature[:, 1]
          + params['true_w'][2]*poly_feature[:, 2] + params['true_b'])
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

loss = torch.nn.MSELoss()


def semilogy(x_vals, y_vals, x_labels, y_labels, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
    plt.xlabel = x_labels
    plt.ylabel = y_labels
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = torch.nn.Linear(train_features.shape[-1], 1)
    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(), params['lr'])
    train_loss, test_loss = [], []
    for i in range(params['num_epochs']):
        for x, y in train_iter:
            ls = loss(net(x), y.view(-1, 1))
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_loss.append(loss(net(train_features), train_labels).item())
        test_loss.append(loss(net(test_features), test_labels).item())
    print('final epoch: train loss:', train_loss[-1], 'test loss', test_loss[-1])
    semilogy(range(1, params['num_epochs']+1), train_loss, 'epochs', 'loss',
             range(1, params['num_epochs']+1), test_loss, ['train', 'test'])
    print('weight', net.weight.data,
          '\nbias', net.bias.data)


# 正常
# fit_and_plot(poly_feature[:params['n_train'], :], poly_feature[params['n_train']:, :],
#              labels[:params['n_train']], labels[params['n_train']:])

# 欠拟合(2次多项式)
# fit_and_plot(poly_featrue2[:params['n_train'], :], poly_featrue2[params['n_train']:, :],
#              labels[:params['n_train']], labels[params['n_train']:])
# 欠拟合(线性拟合)
# fit_and_plot(featrue[:params['n_train'], :], featrue[params['n_train']:, :],
#              labels[:params['n_train']], labels[params['n_train']:])
# 过拟合
fit_and_plot(poly_feature[:2, :], poly_feature[params['n_train']:, :],
             labels[:2], labels[params['n_train']:])
# print(poly_feature.shape)
