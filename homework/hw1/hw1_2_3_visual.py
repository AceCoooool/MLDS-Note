import torch
import matplotlib.pyplot as plt

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.set_title('loss - minimum_loss')
ax2.set_title('loss - grad_norm')
logger = torch.load('results/1_2_3/' + 'minimum_ratio.pth.tar')
loss = [entry['loss'] for _, entry in logger.entries.items()]
ratio = [entry['min_ratio'] for _, entry in logger.entries.items()]
grad_norm = [entry['grad_norm'] for _, entry in logger.entries.items()]
ax1.scatter(ratio, loss, c='b', label='ratio-loss')
ax1.legend(loc="best")
ax1.set_xlabel('minimum_ratio')
ax1.set_ylabel('loss')
ax2.scatter(grad_norm, loss, c='r', label='gnorm-loss')
ax2.legend(loc="best")
ax2.set_xlabel('grad norm')
ax2.set_ylabel('loss')
plt.tight_layout()
plt.show()
