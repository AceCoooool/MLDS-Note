import os
import os.path
from PIL import Image
from torch.utils import data
from torchvision import transforms

default_root = os.path.join(os.path.expanduser('~'), 'data/mlds/faces')

BaseTransform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


# Anime Data : this can also use other image data
class AnimeData(data.Dataset):
    def __init__(self, filename, transform=BaseTransform):
        super(AnimeData, self).__init__()
        self.image_path = list(map(lambda x: os.path.join(filename, x), os.listdir(filename)))
        self.transform = transform

    def __getitem__(self, item):
        image_path = self.image_path[item]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_path)


def anime_loader(root, batch_size, pin=True):
    return data.DataLoader(AnimeData(root), num_workers=4, batch_size=batch_size, shuffle=True, pin_memory=pin)


if __name__ == '__main__':
    loader = anime_loader(default_root, 2)
    loader = iter(loader)
    print(next(loader).size())
