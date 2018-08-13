import torch
import matplotlib.pyplot as plt

epoch_max = 500

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.set_title('mnist loss (train v.s val)')
ax2.set_title('mnist accuracy (train v.s val)')
checkpoint = torch.load('./results/1_3_1/deep_mnist_epoch200.pth.tar')
logger = checkpoint['logger']
x = [entry['epoch'] for _, entry in logger.entries.items()]
loss_train = [entry['loss'] for _, entry in logger.entries.items()]
loss_val = [entry['val_loss'] for _, entry in logger.entries.items()]
acc_train = [entry['accuracy'] for _, entry in logger.entries.items()]
acc_val = [entry['val_accuracy'] for _, entry in logger.entries.items()]
ax1.semilogy(x, loss_train, 'r', label='train')
ax1.semilogy(x, loss_val, 'b', label='val')
ax1.legend(loc='best')
ax2.plot(x, acc_train, 'r', label='train')
ax2.plot(x, acc_val, 'b', label='val')
ax2.legend(loc='best')

plt.tight_layout()
plt.show()
