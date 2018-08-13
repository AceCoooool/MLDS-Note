import torch
import numpy as np
from .base_solver import SolverBase
from utils import eval_gradnorm


class HW1Solver(SolverBase):
    """
    homework 1 solver...
    """

    def __init__(self, model, optimizer, loss, metrics, train_loader, val_loader, config):
        super(HW1Solver, self).__init__(model, optimizer, loss, metrics, config)
        self.train_loader, self.val_loader = train_loader, val_loader
        if config.valid:
            assert val_loader is not None
        if config.visdom:
            self._build_visdom()

    def _train_epoch(self, epoch):
        total_loss, total_metrics = 0, np.zeros(len(self.metrics))
        for idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            for i, metric in enumerate(self.metrics):
                total_metrics[i] += self.metrics[i](output, target)
            if self.config.verbosity >= 2 and idx % self.config.log_step == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, idx, len(self.train_loader), 100.0 * idx / len(self.train_loader), loss.item()))
            # visualization by iteration
            if self.config.visdom and self.config.visdom_iter:
                cnt = idx + (epoch - 1) * len(self.train_loader)
                self.visual.update_vis_line(cnt, [loss.item()], 'train', 'append')
        # visualization by epoch --- TODO: may use avg_loss is reasonable
        if self.config.visdom and not self.config.visdom_iter:
            self.visual.update_vis_line(epoch - 1, [loss.item()], 'train', 'append')
        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = (total_metrics / len(self.train_loader.dataset)).tolist()
        log = {'epoch': epoch, 'loss': avg_loss, 'metrics': avg_metrics}
        if self.config.save_grad:
            grad_norm = eval_gradnorm(self.model.parameters())
            log = {**log, 'grad_norm': grad_norm}
        if self.config.valid and epoch % self.config.val_step == 0:
            val_log = self._valid_epoch(epoch - 1, self.config.visdom_fit)
            log = {**log, **val_log}
        return log

    def _valid_epoch(self, epoch, fit=False):
        """
        validate phase
        """
        if fit:
            assert self.val_loader.batch_size == 1
        self.model.eval()
        total_val_loss, total_val_metrics = 0, np.zeros(len(self.metrics))
        with torch.no_grad():
            for idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss(output, target)
                total_val_loss += loss.item()
                for i, metric in enumerate(self.metrics):
                    total_val_metrics[i] += metric(output, target)
                if self.config.visdom and fit:
                    self.visual.update_vis_line(idx, [output.item(), target.item()], 'val', 'append')
        self.model.train()
        avg_val_loss = total_val_loss / len(self.val_loader)
        avg_val_metrics = (total_val_metrics / len(self.val_loader.dataset)).tolist()
        if self.config.visdom and not fit:
            self.visual.update_vis_line(epoch, [avg_val_loss, *avg_val_metrics], 'val', 'append')
        log = {'val_loss': avg_val_loss, 'val_metrics': avg_val_metrics}
        return log

    def _build_visdom(self):
        """
        visualization with visdom
        """
        from utils import Visdom
        self.visual = Visdom(1)
        if self.config.visdom_iter:
            self.visual.create_vis_line('iter', 'loss', 'training loss with iter', ['loss'], 'train')
        else:
            self.visual.create_vis_line('epoch', 'loss', 'training loss with epoch', ['loss'], 'train')
        if self.config.valid:
            if self.config.visdom_fit:
                self.visual.create_vis_line('x', 'predict', 'fitting', ['pred', 'label'], 'val')
            else:
                metric_name = [metric.__name__ for metric in self.metrics]
                self.visual.create_vis_line('epoch', 'loss-metric', 'validation', ['loss', *metric_name], 'val')
