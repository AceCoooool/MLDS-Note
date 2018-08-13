from model import BaseModel
from torch import nn


# Deep Q-Learning Network
class DQN(BaseModel):
    def __init__(self, name='DQN', in_c=4, num_actions=3, use_duel=False):
        super(DQN, self).__init__()
        self.__name__ = name
        self.use_duel = use_duel
        self.main = nn.Sequential(nn.Conv2d(in_c, 32, 8, stride=4, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.fc1 = nn.Sequential(nn.Linear(64 * 10 * 10, 512), nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(512, num_actions)
        if use_duel:
            self.fc_value = nn.Linear(512, 1)
            self.fc_act = nn.Linear(512, num_actions)

    def forward(self, x):
        x = self.main(x).view(-1, 64 * 10 * 10)
        x = self.fc1(x)
        if self.use_duel:
            value, act = self.fc_value(x), self.fc_act(x)
            x = value + (act - act.mean(1, keepdim=True))
        q_values = self.fc2(x)
        return q_values
