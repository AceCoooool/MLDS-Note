import torch
import matplotlib.pyplot as plt
from torchvision import utils
from model import CondGAN
from utils import create_demo, create_c_demo

pretraiend = '../pre_trained/CondDCGAN_epoch300.pth.tar'
checkpoints = torch.load(pretraiend)
net = CondGAN(use_sigmoid=True)
net.load_state_dict(checkpoints['state_dict'])
z = torch.randn(4, 100, 1, 1)
hair, eyes = range(4), range(4)
c = torch.from_numpy(create_c_demo(hair, eyes)).float()

img = create_demo(net.G, z, c, use_cuda=True, cond=True)
img = utils.make_grid(img.data, normalize=True)
img = img.cpu().numpy().transpose((1, 2, 0))
plt.imshow(img)
plt.show()