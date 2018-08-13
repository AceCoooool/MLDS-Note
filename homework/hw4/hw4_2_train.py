import gym
import gym_ple
from argparse import Namespace
from torch import optim
from solver import DQNSolver
from model import mse_loss, reward_positive_func, DQN

config = {'env': 'FlappyBird-v0', 'lr': 1e-5, 'gamma': 0.99, 'seed': 1, 'buffer_size': 50000,
          'init_eps': 0.1, 'final_eps': 0.02, 'eps_step': 1000000, 'observate_time': 1000,
          'batch_size': 32, 'save_dir': 'results/dqn', 'update_target': 500, 'use_cuda': True,
          'display_interval': 10, 'save_interval': 1000, 'episode': 150000,
          'visdom': True, 'use_double': False, 'reward_len': 10,
          'in_c': 4, 'num_actions': 2}

config = Namespace(**config)

env = gym.make(config.env)
criterion = mse_loss
reward_func = reward_positive_func
model = DQN(in_c=config.in_c, num_actions=config.num_actions)
optimizer = optim.Adam(model.parameters(), lr=config.lr)
solver = DQNSolver(env, model, optimizer, criterion, reward_func, config)
solver.train_q_learning()
