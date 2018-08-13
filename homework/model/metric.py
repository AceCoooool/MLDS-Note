import torch


# accuracy metric
def accuracy(y_pred, y):
    _, pred = torch.max(y_pred, 1)
    return (pred == y).sum().item()


# reward for flappy-bird
def reward_positive_func(reward):
    return reward > 0
