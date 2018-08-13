import torch
from argparse import Namespace
from torch import optim, autograd
from model import deep_simulate
from model import mse_loss
from dataset import simulate_dataloader
from utils import ensure_dir, minimal_ratio, eval_gradnorm, Logger

config = {'batch_size': 250, 'epochs': 4000, 'change_epoch': 2000, 'use_cuda': True,
          'lr': 1e-3, 'save_dir': './results/1_2_3', 'total_num': 100}

config = Namespace(**config)

ensure_dir(config.save_dir)
logger = Logger()
device = torch.device('cuda:0' if torch.cuda.is_available() and config.use_cuda else 'cpu')
train_loader = simulate_dataloader('sin', config.batch_size)
criterion = mse_loss

for i in range(config.total_num):
    net = deep_simulate('sin_{}'.format(i)).to(device)
    opt = optim.Adam(net.parameters(), lr=config.lr, amsgrad=True)
    for epoch in range(1, config.epochs + 1):
        avg_loss, avg_loss_grad = 0.0, 0.0
        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            y_pred = net(x)
            opt.zero_grad()
            loss = criterion(y_pred, y)
            if epoch > config.change_epoch:
                loss_ls, avg_loss = loss.item(), avg_loss + loss.item()
                grads = autograd.grad(loss, net.parameters(), create_graph=True)
                loss = sum([grd.norm() ** 2 for grd in grads])
                loss_grad, avg_loss_grad = loss.item(), avg_loss_grad + loss.item()
            # we only evaluate minimal_ratio once
            if epoch == config.epochs and idx == len(train_loader) - 1:
                ratio = minimal_ratio(grads, net)
            loss.backward()
            opt.step()
        if epoch > config.change_epoch:
            print('epoch (after change): {}, avg_loss: {}, avg_grad: {}'.format(epoch, loss_ls, loss_grad))
        else:
            print('epoch (before change): {}, loss: {}'.format(epoch, loss))
    log = {
        'num': i,
        'avg_loss': avg_loss / (config.epochs - config.change_epochs),
        'avg_loss_grad': avg_loss_grad / (config.epochs - config.change_epochs),
        'loss': loss_ls,
        'loss_grad': loss_grad,
        'grad_norm': eval_gradnorm(net.parameters()),
        'min_ratio': ratio
    }
    print(log)
    logger.add_entry(log)

torch.save(logger, config.save_dir + '/minimum_ratio.pth.tar')
