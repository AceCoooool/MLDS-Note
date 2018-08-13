import os
import torch
import numpy as np
from model import deep_mnist
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

save_dir = 'results/1_2_1'
step, nums = 3, 8
all_params, one_params, all_accs = [], [], []
cmap = get_cmap('hsv')

for i in range(1, 9):
    checkpoint_base = os.path.join(save_dir, str(i))
    checkpoint_filenames = sorted(os.listdir(checkpoint_base))
    for k in range(1, len(checkpoint_filenames)):
        filename = checkpoint_filenames[k]
        checkpoint = torch.load(os.path.join(checkpoint_base, filename))
        model = eval(checkpoint['arch'])()
        model.load_state_dict(checkpoint['state_dict'])
        params = np.zeros((0,))
        for idx, p in enumerate(model.parameters()):
            params = np.append(params, p.cpu().data.numpy().flatten())
            if idx == 2: one_params.append(params)
        all_params.append(params)
        all_accs.append(checkpoint['logger'].entries[k * step]['val_metrics'][0])
all_params = np.array(all_params)
one_params = np.array(one_params)
all_accs = np.array(all_accs)
pca_all = PCA(n_components=2)
pca_one = PCA(n_components=2)
pca_all.fit(all_params)
pca_one.fit(one_params)
xy_all = pca_all.fit_transform(all_params)
xy_one = pca_one.fit_transform(one_params)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.set_title('all params')
ax2.set_title('two layers')
for i in range(1, 9):
    x_ = xy_all[(i - 1) * nums:i * nums, 0]
    y_ = xy_all[(i - 1) * nums:i * nums, 1]
    x1_ = xy_one[(i - 1) * nums:i * nums, 0]
    y1_ = xy_one[(i - 1) * nums:i * nums, 1]
    a_ = all_accs[(i - 1) * nums:i * nums]
    ax1.plot(x_, y_, 'o', color=cmap((i - 1) / 7))
    ax2.plot(x1_, y1_, 'o', color=cmap((i - 1) / 7))
    for xi, yi, x1i, y1i, ai in zip(x_, y_, x1_, y1_, a_):
        ax1.annotate(str('{:.1f}'.format(ai * 100)), xy=(xi, yi),
                     xytext=(xi, yi), color=cmap((i - 1) / 7.5))
        ax2.annotate(str('{:.1f}'.format(ai * 100)), xy=(x1i, y1i),
                     xytext=(x1i, y1i), color=cmap((i - 1) / 7.5))
plt.show()
