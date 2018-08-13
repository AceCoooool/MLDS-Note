from argparse import Namespace
from torch import optim
from model import mse_loss
from model import deep_simulate, middle_simulate, shallow_simulate
from solver import HW1Solver
from dataset.simulate_data import simulate_dataloader

config = {'batch_size': 250, 'epochs': 2000, 'use_cuda': False, 'lr': 1e-3,
          'resume': '', 'verbosity': 1, 'data_dir': None,
          'save_dir': 'results/1_1_1', 'save_freq': 500, 'save_grad': False,
          'valid': True, 'val_step': 200,
          'visdom': True, 'visdom_iter': False, 'visdom_fit': True}
config.update({'target_func': 'ssin'})  # change to other functions
model_list = ['shallow_simulate', 'middle_simulate', 'deep_simulate']

config = Namespace(**config)

for name in model_list:
    model = eval(name)(name + '_' + config.target_func)
    optimizer = optim.Adam(model.parameters(), config.lr)
    loss = mse_loss
    metrics = []
    train_loader = simulate_dataloader(config.target_func, config.batch_size)
    val_loader = simulate_dataloader(config.target_func, 1, shuffle=False)
    solver = HW1Solver(model, optimizer, loss, metrics, train_loader, val_loader, config)
    solver.train()
