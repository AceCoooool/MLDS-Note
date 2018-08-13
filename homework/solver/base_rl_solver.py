import os
import torch
from collections import deque
from utils import Logger, ensure_dir


class RLSolverBase(object):
    def __init__(self, env, model, optimizer, criterion, reward_func, config):
        super(RLSolverBase, self).__init__()
        self.config = config
        self.env = env
        self.model = model
        self.optimizer, self.criterion, self.reward_func = optimizer, criterion, reward_func
        ensure_dir(config.save_dir)
        self.logger = Logger()
        self.start_episode, self.step, self.update_step = 1, 0, 0
        self.device = torch.device('cpu')
        if config.use_cuda and torch.cuda.is_available():
            from torch.backends import cudnn
            cudnn.benchmark = True
            self.device = torch.device('cuda:0')
            self.model = self.model.to(self.device)
        self.memory = deque(maxlen=10000)

    def train_q_learning(self):
        """
        training q learning method
        """
        for episode in range(self.start_episode, self.config.episode + 1):
            result = self._train_episode(episode)
            self.logger.add_entry(result)
            if episode % self.config.display_interval == 0:
                print(result)
            if episode % self.config.save_interval == 0:
                self._save_checkpoint(episode)

    def _save_checkpoint(self, episode):
        """
        save results
        """
        arch = self.model.__name__
        state = {
            'episode': episode,
            'logger': self.logger,
            'arch': arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        filename = os.path.join(self.config.save_dir, '{}_episode{}.pth.tar'.format(arch, episode))
        print("save checkpoint: {} ...".format(filename))
        torch.save(state, filename)
