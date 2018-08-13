import torch
import numpy as np
import matplotlib.pyplot as plt
from model import deep_simulate, middle_simulate, shallow_simulate
from dataset import get_target_func

target_fun = 'sin'
arch_list = ['shallow_simulate', 'middle_simulate', 'deep_simulate']
color_list = ['r', 'g', 'b']
epoch_max = 2000
func = get_target_func(target_fun)
x = np.array([i for i in np.linspace(1e-10, 1, 1000)])
y_target = np.array([func(i) for i in x])
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.set_title(target_fun + ' loss')
ax2.set_title(target_fun + ' fit')
ax2.plot(x, y_target, 'k', label='Ground truth')
for arch, color in zip(arch_list, color_list):
    pre_trained = './results/1_1_1/{}_{}_epoch{}.pth.tar'.format(arch, target_fun, epoch_max)
    checkpoint = torch.load(pre_trained)
    logger = checkpoint['logger']
    epoch = [entry['epoch'] for _, entry in logger.entries.items()]
    loss = [entry['loss'] for _, entry in logger.entries.items()]
    ax1.semilogy(epoch, loss, color, label=arch + '_' + target_fun)
    ax1.legend(loc="best")
    model = eval(arch)()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    y_pred = np.array([model(torch.FloatTensor(np.array([[i]]))).data.numpy() for i in x]).squeeze()
    ax2.plot(x, y_pred, color, label=arch + '_' + target_fun)
    ax2.legend(loc="best")
plt.tight_layout()
plt.show()
