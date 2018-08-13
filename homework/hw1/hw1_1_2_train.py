import os
from argparse import Namespace
from torch import optim
from model import cross_entropy_loss, accuracy
from model import deep_mnist, shallow_mnist, middle_mnist
from model import deep_cifar, shallow_cifar, middle_cifar
from dataset import mnist_train_loader, mnist_validate_loader
from dataset import cifar_train_loader, cifar_validate_loader
from solver import HW1Solver

root = os.path.join(os.path.expanduser('~'), 'data')
dataset = ['mnist', 'cifar'][True]  # false means mnist, true means cifar

config = {'batch_size': 400, 'epochs': 100, 'resume': '', 'verbosity': 1, 'use_cuda': True,
          'lr': 1e-3, 'save_dir': 'results/1_1_2', 'save_freq': 20, 'save_grad': False,
          'data_dir': root, 'dataset': dataset, 'valid': True, 'val_step': 1,
          'visdom': True, 'visdom_iter': True, 'visdom_fit': False}

model_list = ['shallow_' + dataset, 'middle_' + dataset, 'deep_' + dataset]

config = Namespace(**config)

for name in model_list:
    model = eval(name)()
    optimizer = optim.Adam(model.parameters(), config.lr)
    loss = cross_entropy_loss
    metrics = [accuracy]
    train_loader = eval(dataset + '_train_loader')(root, config.batch_size)
    val_loader = eval(dataset + '_validate_loader')(root, config.batch_size)
    solver = HW1Solver(model, optimizer, loss, metrics, train_loader, val_loader, config)
    solver.train()
