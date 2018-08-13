import torch
import matplotlib.pyplot as plt

use_mnist = True
model_name = ['deep_simulate', 'deep_mnist'][use_mnist]
epoch_max = [2000, 80][use_mnist]
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.set_title(model_name + ' loss')
ax2.set_title(model_name + ' log(loss)')
checkpoint = torch.load('./results/1_2_2/{}_epoch{}.pth.tar'.format(model_name, epoch_max))
logger = checkpoint['logger']
epoch = [entry['epoch'] for _, entry in logger.entries.items()]
loss = [entry['loss'] for _, entry in logger.entries.items()]
grad_norm = [entry['grad_norm'] for _, entry in logger.entries.items()]
ax1.plot(epoch, grad_norm, 'b', label='gradnorm')
ax1.plot(epoch, loss, 'r', label='loss')
ax1.legend(loc="best")
ax2.semilogy(epoch, grad_norm, 'b', label='gradnorm')
ax2.semilogy(epoch, loss, 'r', label='loss')
ax2.legend(loc="best")
plt.tight_layout()
plt.show()
