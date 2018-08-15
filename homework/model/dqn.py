from model import BaseModel
from itertools import chain
from torch import nn
from torch.nn import functional as F, init


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


class A2CNet(BaseModel):
    def __init__(self, name='A2C', num_actions=4):
        super(A2CNet, self).__init__()
        self.__name__ = name
        self.share_conv = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(inplace=True),
                                        nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 32, 3, stride=1), nn.ReLU(inplace=True))
        self.share_fc = nn.Sequential(nn.Linear(32 * 7 * 7, 512), nn.ReLU(inplace=True))
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, num_actions)
        self._weight_init()

    def forward(self, x):
        x = self.share_fc(self.share_conv(x).view(-1, 32 * 7 * 7))
        value = self.critic(x)
        pi = F.softmax(x, dim=1)
        return value, pi

    def _weight_init(self):
        for m in chain(self.share_conv.modules(), self.share_fc.modules()):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.orthogonal_(m.weight.data, gain=init.calculate_gain('relu'))
                init.constant_(m.bias.data, 0)
        for m in self.critic.modules():
            init.orthogonal_(m.weight.data)
            init.constant_(m.bias.data, 0)
        for m in self.actor.modules():
            init.orthogonal_(m.weight.data, gain=0.01)
            init.constant_(m.bias.data, 0)
