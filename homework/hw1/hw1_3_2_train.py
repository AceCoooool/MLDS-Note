import os
from argparse import Namespace
from torch import optim
from model import cross_entropy_loss, accuracy
from model import cifar_model, hidden_list
from dataset import cifar_train_loader, cifar_validate_loader
from solver import HW1Solver

root = os.path.join(os.path.expanduser('~'), 'data')

config = {'batch_size': 400, 'epochs': 200, 'resume': '', 'verbosity': 1, 'use_cuda': True,
          'lr': 1e-3, 'save_dir': 'results/1_3_2', 'save_freq': 100, 'save_grad': False,
          'data_dir': root, 'dataset': 'cifar', 'valid': True, 'val_step': 1,
          'visdom': True, 'visdom_iter': False}

config = Namespace(**config)

train_loader = cifar_train_loader(root, config.batch_size)
val_loader = cifar_validate_loader(root, config.batch_size)

for h in hidden_list:
    model = cifar_model(h)
    optimizer = optim.Adam(model.parameters(), config.lr, amsgrad=True)
    loss = cross_entropy_loss
    metrics = [accuracy]
    solver = HW1Solver(model, optimizer, loss, metrics, train_loader, val_loader, config)
    solver.train()
