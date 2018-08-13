import os
from argparse import Namespace
from torch import optim
from model import cross_entropy_loss, accuracy
from model import deep_mnist
from dataset import mnist_train_loader, mnist_validate_loader
from solver import HW1Solver

root = os.path.join(os.path.expanduser('~'), 'data')

config = {'batch_size': 400, 'epochs': 24, 'resume': '', 'verbosity': 1, 'dataset': 'mnist',
          'save_dir': 'results/1_2_1', 'save_freq': 3, 'data_dir': root,
          'use_cuda': True, 'save_grad': False, 'lr': 1e-3, 'valid': True,
          'val_step': 1, 'visdom': True, 'visdom_iter': True, 'visdom_fit': False}

config = Namespace(**config)

for i in range(1, 9):
    save = os.path.join('results/1_2_1', str(i))
    config.save_dir = save
    model = deep_mnist()
    optimizer = optim.Adam(model.parameters(), config.lr)
    loss = cross_entropy_loss
    metrics = [accuracy]
    train_loader = eval(config.dataset + '_train_loader')(root, config.batch_size)
    val_loader = eval(config.dataset + '_validate_loader')(root, config.batch_size)
    solver = HW1Solver(model, optimizer, loss, metrics, train_loader, val_loader, config)
    solver.train()
