import torch
import matplotlib.pyplot as plt

dataset = ['mnist', 'cifar'][False]
arch_list = ['shallow_' + dataset, 'middle_' + dataset, 'deep_' + dataset]
epoch_max = 100

color_list = ['r', 'g', 'b']
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.set_title(dataset + ' loss')
ax2.set_title(dataset + ' accuracy')
for arch, color in zip(arch_list, color_list):
    checkpoint = torch.load('./results/1_1_2/{}_epoch{}.pth.tar'.format(arch, epoch_max))
    logger = checkpoint['logger']
    epoch = [entry['epoch'] for _, entry in logger.entries.items()]
    loss = [entry['loss'] for _, entry in logger.entries.items()]
    accuracy = [entry['val_metrics'][0] for _, entry in logger.entries.items()]
    ax1.semilogy(epoch, loss, color, label=arch)
    ax1.legend(loc="best")
    ax2.plot(epoch, accuracy, color, label=arch)
    ax2.legend(loc="best")
plt.tight_layout()
plt.show()
