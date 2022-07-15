import numpy as np
from joblib import Parallel, delayed
import os
import pandas as pd
from PIL import Image
import random
import torch.utils.data as data


def load(root, filename, resize=(32, 32), jigsaw=False, grid=3):
    im = Image.open(os.path.join(root, filename)).convert("RGB").resize(resize)

    if isinstance(grid, list):
        grid = random.choice(grid)

    if jigsaw:
        s = int(resize[0] / grid)
        tile = [im.crop(np.array([s * (n % grid), s * int(n / grid), s * (n % grid + 1), s * (int(n / grid) + 1)]).astype(int)) for n in range(grid**2)]
        random.shuffle(tile)
        dst = Image.new('RGB', (int(s * grid), int(s * grid)))
        for i, t in enumerate(tile):
            dst.paste(t, (i % grid * s, int(i / grid) * s))
        im = dst

    return im

class LabeledDataset(data.Dataset):
    def __init__(self, root, filenames, labels, domain_label, resize=(32, 32), transform=None, jigsaw=False, grid=3):
        self.root = root
        self.transform = transform
        self.jigsaw = jigsaw
        self.labels = labels
        self.imgs = Parallel(n_jobs=4, verbose=1)([delayed(load)(self.root, filename, resize, jigsaw, grid) for filename in filenames])
        self.domain_labels = np.array([domain_label] * len(self.imgs), dtype=np.int)

    def __getitem__(self, index):
        img = self.imgs[index]

        img1 = self.transform(img)
        img2 = self.transform(img)

        return img1, img2

    def __len__(self):
        return len(self.imgs)

    def concat_dataset(self, dataset):
        assert self.root == dataset.root
        self.imgs.extend(dataset.imgs)
        self.labels = np.concatenate([self.labels, dataset.labels])
        self.domain_labels = np.concatenate([self.domain_labels, dataset.domain_labels])

def get_dataset(parent, dset_taple, domain_label, transform=None, jigsaw=False, grid=3):
    # load train
    dset_name = dset_taple[0]
    dset_labels = np.array(dset_taple[1])
    # text_root = os.path.join("../../Datasets/GDA/data", parent)
    text_root = os.path.join("/nas/data/syamagami/GDA/data/", parent)
    if parent == 'Digit':
        text_train = os.path.join(text_root, "{}_train.txt".format(dset_name))
        resize=(36,36)
    else:
        text_train = os.path.join(text_root, "{}.txt".format(dset_name))
        if parent=='Visda':
            resize=(160,160)
        elif parent == 'PACS':
            resize=(228,228)
        else:
            resize=(255,255)
    df = pd.read_csv(text_train, sep=" ", names=("filename", "label"))
    filenames = df.filename.values
    labels = df.label.values

    use_idx = np.array([i for i,l in enumerate(labels) if l in dset_labels])
    filenames = filenames[use_idx]
    labels = labels[use_idx]

    img_root = os.path.join(text_root, "imgs")
    dataset = LabeledDataset(img_root, filenames, labels, domain_label, resize, transform, jigsaw, grid)

    return dataset

def get_datasets(parent, dset_tables, transform=None, jigsaw=False, grid=3):
    ld = get_dataset(parent, dset_tables[0], 0, transform, jigsaw, grid)

    for i, dset_name in enumerate(dset_tables[1:]):
        ld_t = get_dataset(parent, dset_name, i + 1, transform, jigsaw, grid)

        ld.concat_dataset(ld_t)

    return ld
