import os
from argparse import Namespace
from torch import optim
from model import cross_entropy_loss, accuracy
from model import deep_mnist
from dataset import mnist_train_loader, mnist_validate_loader
from solver import HW1Solver

root = os.path.join(os.path.expanduser('~'), 'data')

param_list = [[64, 1e-3], [1024, 1e-3], [512, 1e-3], [512, 1e-2]]

config = {'batch_size': 1024, 'epochs': 40, 'resume': '', 'verbosity': 1, 'use_cuda': True,
          'lr': 1e-3, 'save_dir': 'results/1_3_3', 'save_freq': 10, 'save_grad': False,
          'data_dir': root, 'dataset': 'mnist', 'valid': True, 'val_step': 1,
          'visdom': True, 'visdom_iter': True, 'visdom_fit': False}

config = Namespace(**config)

for param in param_list:
    config.batch_size, config.lr = param[0], param[1]
    train_loader = mnist_train_loader(root, config.batch_size)
    val_loader = mnist_validate_loader(root, config.batch_size)
    model = deep_mnist('mnist_{}_{}'.format(config.batch_size, str(config.lr)))
    optimizer = optim.Adam(model.parameters(), config.lr, amsgrad=True)
    loss = cross_entropy_loss
    metrics = [accuracy]
    solver = HW1Solver(model, optimizer, loss, metrics, train_loader, val_loader, config)
    solver.train()
