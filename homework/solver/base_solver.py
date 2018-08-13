import os
import math
import shutil
import torch
from utils import Logger, ensure_dir


class SolverBase(object):
    def __init__(self, model, optimizer, loss, metrics, config):
        super(SolverBase, self).__init__()
        self.config = config
        ensure_dir(config.save_dir)
        self.model, self.optimizer, self.loss, self.metrics = model, optimizer, loss, metrics
        self.logger = Logger()
        self.min_loss = math.inf
        self.identity = model.__name__
        self.start_epoch = 1
        self.device = torch.device('cpu')
        if config.use_cuda and torch.cuda.is_available():
            from torch.backends import cudnn
            cudnn.benchmark = True
            self.device = torch.device('cuda:0')
            self.model.to(self.device)
        if config.resume:
            self._resume_checkpoint(config.resume)

    def train(self):
        """
        training phase
        """
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            result = self._train_epoch(epoch)
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    for i, metric in enumerate(self.metrics):
                        log[metric.__name__] = result['metrics'][i]
                elif key == 'val_metrics':
                    for i, metric in enumerate(self.metrics):
                        log['val_' + metric.__name__] = result['metrics'][i]
                else:
                    log[key] = value

            self._save_log(log)
            # TODO: may have bug
            if epoch % self.config.save_freq == 0:
                self._save_checkpoint(epoch, log['loss'])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_log(self, result):
        """
        save result to logger (for draw or print)
        """
        self.logger.add_entry(result)
        if self.config.verbosity >= 1:
            # TODO: fix bug
            print([(k, v) for k, v in result.items() if k not in ['full_loss']])

    def _save_checkpoint(self, epoch, loss):
        """
        save model (include state)
        """
        if loss < self.min_loss:
            self.min_loss = loss
        arch = self.model.__name__
        state = {
            'epoch': epoch,
            'params_num': self.model.params_num(),
            'logger': self.logger,
            'arch': arch,
            'state_dict': self.model.state_dict(),
            'optimizer': {name: optimizer.state_dict() for name, optimizer in self.optimizer.items()}
            if isinstance(self.optimizer, dict) else self.optimizer.state_dict(),
            'min_loss': self.min_loss
        }
        filename = os.path.join(self.config.save_dir,
                                self.identity + '_epoch{:02d}.pth.tar'.format(epoch))
        print("save checkpoint: {} ...".format(filename))
        torch.save(state, filename)
        if loss == self.min_loss:
            shutil.copyfile(filename, os.path.join(self.config.save_dir, self.identity + '_best.pth.tar'))

    def _resume_checkpoint(self, resume_path):
        """
        resume training from resume_path
        """
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.min_loss = checkpoint['min_loss']
        self.model.load_state_dict(checkpoint['state_dict'])
        if isinstance(self.optimizer, dict):
            for name, state in checkpoint['optimizers'].items():
                self.optimizers[name].load_state_dict(state)
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger = checkpoint['logger']
        print("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
