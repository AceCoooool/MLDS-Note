import os
import torch
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from utils import eval_model, Logger
from dataset import mnist_train_loader, mnist_validate_loader
from model import deep_mnist, cross_entropy_loss, accuracy

root = os.path.join(os.path.expanduser('~'), 'data')
model_list = ['./results/1_3_3/mnist_64_0.001.pth.tar', './results/1_3_3/mnist_1024_0.001.pth.tar']
use_cuda = True
logger = Logger()
param1 = torch.load(model_list[0])['state_dict']
param2 = torch.load(model_list[1])['state_dict']
model = deep_mnist()
loss = cross_entropy_loss
metric = accuracy
train_loader = mnist_train_loader(root, 200)
val_loader = mnist_validate_loader(root, 200)

for alpha in np.arange(-1.0, 2.0, 0.02):
    alpha = float(alpha)
    param = OrderedDict((k, (1 - alpha) * param1[k] + alpha * param2[k]) for k in param1)
    model.load_state_dict(param)

    log = {'alpha': alpha}
    res = eval_model(model, train_loader, loss, metric, use_cuda)
    log.update({'loss': res[0], 'accuracy': res[1]})
    res = eval_model(model, val_loader, loss, metric, use_cuda)
    log.update({'val_loss': res[0], 'val_accuracy': res[1]})
    logger.add_entry(log)

x = [entry['alpha'] for _, entry in logger.entries.items()]
y1_train = [entry['loss'] for _, entry in logger.entries.items()]
y2_train = [entry['accuracy'] for _, entry in logger.entries.items()]
y1_val = [entry['val_loss'] for _, entry in logger.entries.items()]
y2_val = [entry['val_accuracy'] for _, entry in logger.entries.items()]
fig, ax1 = plt.subplots(figsize=(10, 10))
ax1.semilogy(x, y1_train, 'b', label='train')
ax1.semilogy(x, y1_val, 'b--', label='val')
ax1.legend(loc="best")
ax1.set_xlabel('alpha', color='b')
ax1.set_ylabel('cross_entropy', color='b')
ax2 = ax1.twinx()
ax2.plot(x, y2_train, 'r')
ax2.plot(x, y2_val, 'r--')
ax2.set_ylabel('accuracy', color='r')
plt.tight_layout()
plt.show()
