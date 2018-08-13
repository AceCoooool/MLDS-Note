from .mnist_own import MNIST_OWN
from torch.utils import data
from torchvision import transforms, datasets


# mnist train dataloader
def mnist_train_loader(root, batch_size, target_random=False, pin=True):
    d = MNIST_OWN(root, train=True, transform=transforms.ToTensor(), download=True, target_random=target_random)
    train_loader = data.DataLoader(d, num_workers=4, batch_size=batch_size, shuffle=True, pin_memory=pin)
    return train_loader


# mnist validate dataloader
def mnist_validate_loader(root, batch_size, pin=True):
    d = datasets.MNIST(root, train=False, transform=transforms.ToTensor())
    val_loader = data.DataLoader(d, num_workers=4, batch_size=batch_size, shuffle=False, pin_memory=pin)
    return val_loader


# cifar10 train dataloader
def cifar_train_loader(root, batch_size, size=32, pin=True):
    transform = transforms.Compose([transforms.Pad(4), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(size), transforms.ToTensor()])
    d = datasets.CIFAR10(root, train=True, transform=transform, download=True)
    train_loader = data.DataLoader(d, num_workers=4, batch_size=batch_size, shuffle=True, pin_memory=pin)
    return train_loader


# cifar10 validation dataloader
def cifar_validate_loader(root, batch_size, pin=True):
    d = datasets.CIFAR10(root, train=False, transform=transforms.ToTensor())
    val_loader = data.DataLoader(d, num_workers=4, batch_size=batch_size, shuffle=False, pin_memory=pin)
    return val_loader
