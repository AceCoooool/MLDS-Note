import os
from argparse import Namespace
from torch import optim
from model import mse_loss, cross_entropy_loss
from model import deep_simulate, deep_mnist
from solver import HW1Solver
from dataset import simulate_dataloader, mnist_train_loader

use_mnist = False
root = os.path.join(os.path.expanduser('~'), 'data')
dataset = ['simulate', 'mnist'][use_mnist]  # false: sin(5pix)/(5pix)  true: mnist
config = {'batch_size': 250, 'resume': '', 'verbosity': 1, 'save_dir': 'results/1_2_2',
          'data_dir': root, 'use_cuda': True, 'dataset': 'simulate',
          'save_grad': True, 'lr': 1e-3, 'valid': False, 'val_step': 200, 'target_func': 'sin',
          'visdom': True, 'visdom_iter': use_mnist, 'visdom_step': 200, 'visdom_fit': False}
config.update({'epochs': [2000, 80][use_mnist], 'save_freq': [500, 20][use_mnist]})
model_name = ['deep_simulate', 'deep_mnist'][use_mnist]

config = Namespace(**config)

model = eval(model_name)()
optimizer = optim.Adam(model.parameters(), config.lr)
loss = [mse_loss, cross_entropy_loss][use_mnist]
metrics = []
if use_mnist:
    train_loader = mnist_train_loader(root, config.batch_size)
else:
    train_loader = simulate_dataloader(config.target_func, config.batch_size)
solver = HW1Solver(model, optimizer, loss, metrics, train_loader, None, config)
solver.train()
