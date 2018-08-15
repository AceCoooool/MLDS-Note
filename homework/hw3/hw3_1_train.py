import os
import random
from torch import optim
from argparse import Namespace
from solver import DCGANSolver
from model import DCGAN, bce_loss, ls_loss
from dataset import anime_loader

root = os.path.join(os.path.expanduser('~'), 'data/mlds/faces')

gan = 'DCGAN'

config = {'batch_size': 256, 'epochs': 500, 'use_cuda': True, 'z_size': 100,
          'save_img_step': 1, 'resume': '', 'verbosity': 1, 'data_dir': root,
          'dis_iter': 1, 'seed': random.randint(1, 10000),
          'gan': gan, 'a': 0, 'b': 1, 'c': 1, 'clip': 0.01, 'valid': False, 'val_step': 200,
          'save_dir': 'results/1_3_1/' + gan, 'save_freq': 50, 'save_grad': False,
          'visdom': False, 'visdom_iter': True}

config = Namespace(**config)
model = DCGAN(name=config.gan, use_sigmoid=config.gan == 'DCGAN')

if config.gan == 'DCGAN':
    optimizer = {'D': optim.Adam(model.D.parameters(), lr=0.0002, betas=(0.5, 0.999)),
                 'G': optim.Adam(model.G.parameters(), lr=0.0002, betas=(0.5, 0.999))}
    criterion = bce_loss
elif config.gan == 'LSGAN':
    config.a, config.b, config.c = 0, 1, 1
    optimizer = {'D': optim.Adam(model.D.parameters(), lr=0.0001, betas=(0.5, 0.999)),
                 'G': optim.Adam(model.G.parameters(), lr=0.0001, betas=(0.5, 0.999))}
    criterion = ls_loss
elif config.gan == 'WGAN':
    config.dis_iter, config.clip = 5, 0.01
    optimizer = {'D': optim.RMSprop(model.D.parameters(), lr=0.00005),
                 'G': optim.RMSprop(model.G.parameters(), lr=0.00005)}
    criterion = None
elif config.gan == 'WGAN-GP':
    config.dis_iter, config.penalty = 5, 10
    optimizer = {'D': optim.Adam(model.D.parameters(), lr=0.0001, betas=(0.5, 0.999)),
                 'G': optim.Adam(model.G.parameters(), lr=0.0001, betas=(0.5, 0.999))}
    criterion = None

train_loader = anime_loader(root, config.batch_size)
solver = DCGANSolver(model, optimizer, criterion, [], train_loader, None, config)
solver.train()