import torch
import torch.nn as nn
from .base_model import BaseModel


class DCGAN_G(BaseModel):
    """ DCGAN Generator
    Args:
        img_size: image size (default=64)
        z_size: random numbers (default=100)
        h_size: Number of hidden nodes (default=128)
        ---for conditional gan---
        cond: whether condtional gan (default=False)
        embed_size: conditional "embedding" (default=64)
    """

    def __init__(self, img_size=64, n_colors=3, z_size=100, h_size=128, use_selu=True, embed_size=64, cond=False):
        super(DCGAN_G, self).__init__()
        main = list()
        mult = img_size // 8
        self.cond = cond
        if cond:
            self.fc = nn.Linear(22, embed_size)
            z_size = z_size + embed_size
        main += [nn.ConvTranspose2d(z_size, h_size * mult, kernel_size=4, stride=1, padding=0, bias=False)]
        if use_selu:
            main += [nn.SELU(inplace=True)]
        else:
            main += [nn.BatchNorm2d(h_size * mult), nn.ReLU(inplace=True)]
        # Middle block (Done until we reach ? x image_size/2 x image_size/2)
        while mult > 1:
            main += [nn.ConvTranspose2d(h_size * mult, h_size * (mult // 2), 4, 2, 1, bias=False)]
            if use_selu:
                main += [nn.SELU(inplace=True)]
            else:
                main += [nn.BatchNorm2d(h_size * (mult // 2)), nn.ReLU(inplace=True)]
            mult = mult // 2
        # End block
        main += [nn.ConvTranspose2d(h_size, n_colors, kernel_size=4, stride=2, padding=1, bias=False), nn.Tanh()]
        self.main = nn.Sequential(*main)

    def forward(self, x, text=None):
        if self.cond:
            emb = self.fc(text)
            emb = emb.view(*emb.size(), 1, 1)
            x = torch.cat((x, emb), 1)
        output = self.main(x)
        return output


class DCGAN_D(BaseModel):
    """DCGAN Discriminator
    Args:
        img_size: image size (default=96)
        n_colors: color channels (default=3)
        h_size: hidden size "base" (default=128)
        use_sigmoid: WGAN not use sigmoid (default=True)
    """

    def __init__(self, img_size=64, n_colors=3, h_size=128, use_sigmoid=True, use_selu=True, embed_size=64, cond=False):
        super(DCGAN_D, self).__init__()
        self.cond = cond
        main = list()
        # Start block
        # Size = n_colors x image_size x image_size
        main += [nn.Conv2d(n_colors, h_size, kernel_size=4, stride=2, padding=1, bias=False)]
        if use_selu:
            main += [nn.SELU(inplace=True)]
        else:
            main += [nn.LeakyReLU(0.2, inplace=True)]
        img_size_new, mult = img_size // 2, 1
        # Middle block (Done until we reach ? x k x k) --- default (?x4x4)
        while img_size_new > 4:
            main += [nn.Conv2d(h_size * mult, h_size * (2 * mult), 4, 2, 1, bias=False)]
            if use_selu:
                main += [nn.SELU(inplace=True)]
            else:
                main += [nn.BatchNorm2d(h_size * (2 * mult)), nn.LeakyReLU(0.2, inplace=True)]
            img_size_new, mult = img_size_new // 2, mult * 2
        # End block
        if cond:
            self.fc = nn.Linear(22, embed_size)
            main += [nn.Conv2d(h_size * mult, 256, 4, stride=1, padding=0, bias=False)]
            tail = [nn.Linear(256 + embed_size, 512), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Linear(512, 1)]
            if use_sigmoid: tail += [nn.Sigmoid()]
            self.tail = nn.Sequential(*tail)
        else:
            main += [nn.Conv2d(h_size * mult, 1, 4, stride=1, padding=0, bias=False)]
            if use_sigmoid: main += [nn.Sigmoid()]
        self.main = nn.Sequential(*main)

    def forward(self, x, text=None):
        if self.cond:
            output = self.main(x).view(x.size(0), -1)
            embed = self.fc(text)
            combined = torch.cat((output, embed), 1)
            output = self.tail(combined)
        else:
            output = self.main(x)
        return output.view(-1)


class DCGAN(BaseModel):
    def __init__(self, name='dcgan', use_sigmoid=True, use_selu=False):
        super(DCGAN, self).__init__()
        self.G = DCGAN_G(use_selu=use_selu)
        self.D = DCGAN_D(use_selu=use_selu, use_sigmoid=use_sigmoid)
        self.__name__ = name


class CondGAN(BaseModel):
    def __init__(self, name='condgan', embed_dim=64, use_sigmoid=True, use_selu=False):
        super(CondGAN, self).__init__()
        self.G = DCGAN_G(embed_size=embed_dim, use_selu=use_selu, cond=True)
        self.D = DCGAN_D(embed_size=embed_dim, use_selu=use_selu, use_sigmoid=use_sigmoid, cond=True)
        self.__name__ = name
