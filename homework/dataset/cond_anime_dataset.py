import os
import os.path
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
from utils import data_pre_process

BaseTransform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


# Conditional Anime Data : this can also use other image data
class CAnimeData(data.Dataset):
    def __init__(self, root, transform=BaseTransform):
        super(CAnimeData, self).__init__()
        filename = os.path.join(root, 'pre_process.pth.tar')
        self.transform = transform
        if not os.path.exists(filename):
            data_pre_process(root, filename)
        logger = torch.load(filename)
        self.img_path = [entry['img_path'] for _, entry in logger.entries.items()]
        self.embdding = [entry['embedding'] for _, entry in logger.entries.items()]

    def __getitem__(self, item):
        image_path = self.img_path[item]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        embedding = torch.from_numpy(self.embdding[item]).float()
        return image, embedding

    def __len__(self):
        return len(self.img_path)


def cond_anime_loader(root, batch_size, pin=True):
    return data.DataLoader(CAnimeData(root), num_workers=4, batch_size=batch_size, shuffle=True, pin_memory=pin)


if __name__ == '__main__':
    root = '/home/ace/data/mlds'
    dataset = CAnimeData(root)
    print(dataset[0])
