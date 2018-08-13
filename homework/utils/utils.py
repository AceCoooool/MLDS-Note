import os
from torch import nn
from torch.nn import init


def ensure_dir(path):
    """
    ensure path is a directory, if not: create it.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def weight_init(param, init_type='normal'):
    """
    choose different weight initialization
    """
    if init_type == 'normal':
        if isinstance(param, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            init.normal_(param.weight.data, 0.0, 0.02)
            param.bias.data.zero_() if param.bias is not None else None
        if isinstance(param, nn.BatchNorm2d):
            param.weight.data.normal_(1.0, 0.02)
            param.bias.data.zero_()
    elif init_type == 'uniform':
        if isinstance(param, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            init.uniform_(param.weight.data)
            param.bias.data.zero_() if param.bias is not None else None
    elif init_type == 'xavier_uniform':
        if isinstance(param, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            init.xavier_uniform_(param.weight.data)
            param.bias.data.zero_() if param.bias is not None else None
    elif init_type == 'xavier_normal':
        if isinstance(param, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            init.xavier_normal_(param.weight.data)
            param.bias.data.zero_() if param.bias is not None else None
    elif init_type == 'kaiming_uniform':
        if isinstance(param, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            init.kaiming_uniform_(param.weight.data)
            param.bias.data.zero_() if param.bias is not None else None
    elif init_type == 'kaiming_normal':
        if isinstance(param, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            init.kaiming_normal_(param.weight.data)
            param.bias.data.zero_() if param.bias is not None else None
    else:
        assert "illegal input"
