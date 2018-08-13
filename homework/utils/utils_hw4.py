from torchvision import transforms

transform = transforms.Compose([transforms.ToPILImage(mode='RGB'),
                                transforms.Grayscale(1),
                                transforms.Resize((84, 84)),
                                transforms.ToTensor()])


# process the image..
def pre_process(x):
    return transform(x)


