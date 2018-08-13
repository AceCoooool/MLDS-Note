from .base_model import BaseModel
import torch.nn as nn


# simple FNN for simulate data
class SimulateModel(BaseModel):
    def __init__(self, name, c=[1, 5, 10, 10, 10, 10, 10, 5]):
        super(SimulateModel, self).__init__()
        self.__name__ = name
        layers = list()
        for i in range(len(c) - 1):
            layers.append(nn.Linear(c[i], c[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(c[-1], c[0]))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


# shallow fc net: 1 hidden layer
def shallow_simulate(name='shallow_simulate'):
    c = [1, 190]
    return SimulateModel(name, c)


# middle fc net: 4 hidden layer
def middle_simulate(name='middle_simulate'):
    c = [1, 10, 18, 15, 4]
    return SimulateModel(name, c)


# deep fc net: 7 hidden layer
def deep_simulate(name='deep_simulate'):
    c = [1, 5, 10, 10, 10, 10, 10, 5]
    return SimulateModel(name, c)
