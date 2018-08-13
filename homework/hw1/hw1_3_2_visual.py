import torch
import matplotlib.pyplot as plt
from model import hidden_list

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.set_title('cifar loss (params_num - loss)')
ax2.set_title('cifar accuracy (params_num v.s accuracy)')
epoch_max = 200
for i, h in enumerate(hidden_list):
    checkpoint = torch.load('./results/1_3_2/cifar_{}_epoch{}.pth.tar'.format(str(h), str(epoch_max)))
    logger = checkpoint['logger']
    x = checkpoint['params_num']
    y1 = [entry['loss'] for _, entry in logger.entries.items()][-1]
    y2 = [entry['val_loss'] for _, entry in logger.entries.items()][-1]
    ax1.scatter(x, y1, c='b', label='train' if not i else None)
    ax1.scatter(x, y2, c='r', label='validate' if not i else None)
    y1 = [entry['metrics'] for _, entry in logger.entries.items()][-1]
    y2 = [entry['val_metrics'] for _, entry in logger.entries.items()][-1]
    # y1 = [entry['accuracy'] for _, entry in logger.entries.items()][-1]
    # y2 = [entry['val_accuracy'] for _, entry in logger.entries.items()][-1]
    ax2.scatter(x, y1, c='b', label='train' if not i else None)
    ax2.scatter(x, y2, c='r', label='validate' if not i else None)
ax1.legend(loc='best')
ax2.legend(loc='best')
plt.tight_layout()
plt.show()
