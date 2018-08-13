import numpy as np
from torch.utils import data


class SimulateData(data.Dataset):
    def __init__(self, f, x_range=[0, 1], step=0.001):
        super(SimulateData, self).__init__()
        self.left, self.right = x_range[0], x_range[1]
        self.step, self.num = step, round((self.right - self.left) / step)
        self.f = f

    def __getitem__(self, item):
        x = np.expand_dims(self.left + (item + 1) * self.step, 0)
        y = self.f(x)
        return x.astype(np.float32), y.astype(np.float32)

    def __len__(self):
        return self.num


def simulate_dataloader(target_func, batch_size, shuffle=True, x_range=[0, 1], step=0.001, pin=True):
    if target_func == 'sin':
        dataset = SimulateData(lambda x: np.sin(5 * np.pi * x) / (5 * np.pi * x), x_range, step)
    elif target_func == 'ssin':
        dataset = SimulateData(lambda x: np.sign(np.sin(5 * np.pi * x)), x_range, step)
    else:
        assert "illegal target function"
    return data.DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=shuffle, pin_memory=pin)


def get_target_func(target_func):
    if target_func == 'sin':
        return lambda x: np.sin(5 * np.pi * x) / (5 * np.pi * x)
    elif target_func == 'ssin':
        return lambda x: np.sign(np.sin(5 * np.pi * x))
    else:
        assert "illegal target function"
