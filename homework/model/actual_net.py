from torch import nn
from .base_model import BaseModel


# actual network "template"
class Model(BaseModel):
    def __init__(self, name, c=[1, 208], k=[5], pool=[0], size=28, num_classes=10, use_bn=False):
        super(Model, self).__init__()
        self.__name__ = name
        layers = list()
        s = 4 if len(c) == 2 else 2
        for i in range(len(c) - 1):
            layers.append(nn.Conv2d(c[i], c[i + 1], kernel_size=k[i], stride=1, padding=k[i] // 2, bias=False))
            if use_bn: layers.append(nn.BatchNorm2d(c[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if i in pool:
                layers.append(nn.MaxPool2d(s, s))
        self.main = nn.Sequential(*layers)
        self.tail = nn.Sequential(nn.Conv2d(c[-1], num_classes, 1, 1),
                                  nn.AvgPool2d(size // max(2 ** len(pool), 4), size // max(2 ** len(pool), 4)))

    def forward(self, x):
        out = self.main(x)
        out = self.tail(out).view(-1, 10)
        return out


# shallow mnist --- one hidden layer
def shallow_mnist(name='shallow_mnist'):
    c, k, pool = [1, 208], [5], [0]
    return Model(name, c, k, pool)


# middle mnist --- two hidden layer
def middle_mnist(name='middle_mnist'):
    c, k, pool = [1, 24, 31], [3, 3], [0, 1]
    return Model(name, c, k, pool)


# deep mnist --- four hidden layer
def deep_mnist(name='deep_mnist'):
    c, k, pool = [1, 16, 16, 16, 16], [3, 3, 3, 3], [0, 3]
    return Model(name, c, k, pool)


# shallow cifar --- two hidden layer
def shallow_cifar(name='shallow_cifar'):
    c, k, pool = [3, 128, 80], [3, 3], [0, 1]
    return Model(name, c, k, pool, size=32, use_bn=True)


# middle cifar --- three hidden layer
def middle_cifar(name='middle_cifar'):
    c, k, pool = [3, 32, 64, 128], [3, 3, 3], [0, 2]
    return Model(name, c, k, pool, size=32, use_bn=True)


# deep cifar --- six hidden layer
def deep_cifar(name='deep_cifar'):
    c, k, pool = [3, 16, 32, 64, 64, 38, 32], [3, 3, 3, 3, 3, 3], [0, 3]
    return Model(name, c, k, pool, size=32, use_bn=True)


# you can add more
hidden_list = [i for i in range(16, 513, 16)]


def cifar_model(hidden):
    c, k, pool = [3, hidden // 4, hidden // 2, hidden, hidden], [3, 3, 3, 3], [0, 1, 2]
    return Model('cifar_' + str(hidden), c, k, pool, size=32, use_bn=True)
