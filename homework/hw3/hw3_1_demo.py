import torch
import matplotlib.pyplot as plt
from torchvision import utils
from model import DCGAN
from utils import create_demo

pretraiend = '../pretrained/3_1/DCGAN_epoch300.pth.tar'
checkpoints = torch.load(pretraiend)
net = DCGAN(use_sigmoid=True)
net.load_state_dict(checkpoints['state_dict'])
z = torch.randn(64, 100, 1, 1)

img = create_demo(net.G, z, use_cuda=True)

img = utils.make_grid(img.data, normalize=True)
img = img.cpu().numpy().transpose((1, 2, 0))
plt.imshow(img)
plt.show()