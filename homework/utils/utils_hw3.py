import os
import torch
import imageio
import numpy as np
from .utils_logger import Logger

hair_default = ['aqua', 'black', 'blonde', 'blue', 'brown', 'gray',
                'green', 'orange', 'pink', 'purple', 'red', 'white']

eyes_default = ['aqua', 'black', 'blue', 'brown', 'green',
                'orange', 'pink', 'purple', 'red', 'yellow']


class Embedder(object):
    def __init__(self, hair_attr=hair_default, eyes_attr=eyes_default):
        self.hair_attr, self.eyes_attr = hair_attr, eyes_attr
        self.hair_dict, self.eyes_dict = {}, {}
        for i, attr in enumerate(self.hair_attr):
            encode_attr = np.zeros(len(self.hair_attr))
            encode_attr[i] = 1
            self.hair_dict[attr] = encode_attr
        for i, attr in enumerate(self.eyes_attr):
            encode_attr = np.zeros(len(self.eyes_attr))
            encode_attr[i] = 1
            self.eyes_dict[attr] = encode_attr

    def encode_feature(self, feature):
        """
            feature : {'hair': ['color1'], 'eyes': ['color2']}
            :return: np.array shape = (12 + 10,)
                                      hair code first, eyes code follows
        """
        hair_code = self.hair_dict[feature['hair'][0]]
        eyes_code = self.eyes_dict[feature['eyes'][0]]
        return np.concatenate((hair_code, eyes_code), axis=0)


embedding = Embedder()


# data pre-process: delete some "unlabeled images", label embedding
def data_pre_process(root, save_file):
    hair_attr, eyes_attr = [], []
    logger = Logger()
    hair_stoplist = ['damage', 'long', 'short', 'pubic']
    eyes_stoplist = ['bicolored', 'gray']
    tags1_path = os.path.join(root, 'tags_clean.csv')
    img1_path = os.path.join(root, 'faces')
    tags2_path = os.path.join(root, 'extra_data/tags.csv')
    img2_path = os.path.join(root, 'extra_data/images')
    # -----deal with faces-----
    with open(tags1_path) as f:
        for line in f:
            idx, tags = line.split(',')
            tags = tags.split('\t')
            log = {'img_path': img1_path + '/{}.jpg'.format(idx), 'hair': [], 'eyes': []}
            for i, t in enumerate(tags):
                t = t.strip(':0123456789').split()
                if 'hair' in t:
                    if t[0] not in hair_stoplist:
                        log['hair'].append(t[0])
                        if t[0] not in hair_attr:
                            hair_attr.append(t[0])

                if 'eyes' in t and t[0] != 'eyes':
                    if t[0] not in eyes_stoplist:
                        log['eyes'].append(t[0])
                        if t[0] not in eyes_attr:
                            eyes_attr.append(t[0])
            if len(log['hair']) == 1 and len(log['eyes']) == 1:
                logger.add_entry(log)
    # -----deal with extras_image-----
    with open(tags2_path) as f:
        for line in f:
            idx, tags = line.split(',')
            tags = tags.split()
            log = {'img_path': img2_path + '/{}.jpg'.format(idx), 'hair': [], 'eyes': []}
            if tags[0] not in hair_attr:
                hair_attr.append(tags[0])
            if tags[2] not in eyes_attr:
                eyes_attr.append(tags[2])
            log.update({'hair': [tags[0]], 'eyes': [tags[2]]})
            logger.add_entry(log)
    hair_attr.sort()
    eyes_attr.sort()
    embedder = Embedder(hair_attr, eyes_attr)
    [entry.update({'embedding': embedder.encode_feature(entry)}) for _, entry in logger.entries.items()]
    torch.save(logger, open(save_file, 'wb'))


def create_c_demo(h_list, e_list):
    for i, (h, e) in enumerate(zip(h_list, e_list)):
        assert 0 <= h < 12 and 0 <= e < 10
        features = {'hair': [hair_default[h]], 'eyes': [eyes_default[e]]}
        if i == 0:
            c_text = embedding.encode_feature(features)
        else:
            c_text = np.r_[c_text, embedding.encode_feature(features)]
    return c_text.reshape((len(e_list), -1))


def create_c_test(n=8):
    for i in range(n):
        features = {'hair': [hair_default[np.random.randint(0, len(hair_default))]],
                    'eyes': [eyes_default[np.random.randint(0, len(eyes_default))]]}
        if i == 0:
            c_text = embedding.encode_feature(features)
        else:
            c_text = np.r_[c_text, embedding.encode_feature(features)]
    return c_text.reshape((n, -1))


# create git tools
def create_gif(img_path, save_dir):
    names = sorted(list(map(lambda x: os.path.join(img_path, x), os.listdir(img_path))))
    images = []
    for filename in names:
        print(filename)
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(save_dir, 'gan.gif'), images, duration=0.5)


# demo
def create_demo(model, z, c=None, use_cuda=False, cond=False):
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')
        model.to(device)
        z = z.to(device)
        if cond: c = c.to(device)
    model.eval()
    return model(z, c) if cond else model(z)
