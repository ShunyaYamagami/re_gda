from doctest import testfile
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import sys

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.transforms import transforms

from functions import input_const_values, get_jigsaw
from util import to_cuda



def load(config, root, filename, resize):
    original_im = Image.open(os.path.join(root, filename)).convert("RGB").resize(resize)

    im = original_im.copy()
    im = input_const_values(im, resize, const_abs=False, const_pha=True, n_random = resize[0] * resize[1] // 6, const_value=0 )  # 位相・振幅に一定値を入れる．
    im = get_jigsaw(im, resize, config.grid)

    return original_im, im



class LabeledDataset(data.Dataset):
    def __init__(
        self, config, root, filenames, resize, transform, true_labels, true_domain_labels, 
        pseudo_labels=None, edls=None, ed_feats=None
    ):
        self.config = config
        self.root = root
        self.resize = resize
        self.transform = transform

        self.filenames = filenames
        self.imgs_t = Parallel(n_jobs=4, verbose=1)([delayed(load)(self.config, self.root, f_name, resize) for f_name in self.filenames])
        self.original_imgs = [r[0] for r in self.imgs_t]
        self.aug_imgs = [r[1] for r in self.imgs_t]

        self.true_labels = true_labels
        self.true_domain_labels = true_domain_labels
        self.pseudo_labels = pseudo_labels
        self.edls = edls
        self.ed_feats = ed_feats


    def __getitem__(self, index):
        original_img = self.original_imgs[index]
        aug_img = self.aug_imgs[index]

        original_img = self.transform(original_img)
        aug_img1 = self.transform(aug_img)
        aug_img2 = self.transform(aug_img)

        edl = self.edls[index]
        return original_img, aug_img1, aug_img2, edl, index

    def __len__(self):
        return len(self.original_imgs)

    # def update_labels(self, index, results):
    #     self.history[index] = np.hstack((self.history[index, 1:], results.reshape(-1, 1)))

    def update_pseudo(self):
        target, _ = stats.mode(self.history, axis=1)
        self.pseudo_labels = np.squeeze(target)

    def update_estimated_domains(self, idices, edls, ed_feats):
        self.edls[idices] = edls
        self.ed_feats[idices] = ed_feats
        

    def concat_dataset(self, dataset):
        assert self.root == dataset.root
        self.original_imgs.extend(dataset.original_imgs)
        self.aug_imgs.extend(dataset.aug_imgs)
        self.true_labels = np.concatenate([self.true_labels, dataset.true_labels])
        self.true_domain_labels = np.concatenate([self.true_domain_labels, dataset.true_domain_labels])
        if self.pseudo_labels is not None:
            self.pseudo_labels = np.concatenate([self.pseudo_labels, dataset.pseudo_labels])
        if self.edls is not None:
            self.edls = np.concatenate([self.edls, dataset.edls])
        if self.ed_feats is not None:
            self.ed_feats = np.concatenate([self.ed_feats, dataset.ed_feats])
    


def get_transforms(config):
    # transformの内容はドメイン推定の方
    if config.parent == 'Digit':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=config.imsize, scale=(0.08, 1.0)),
            transforms.Grayscale(3),
            # GaussianBlur(kernel_size=int(0.3 * config.imsize), min=0.1, max=2.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5]),
            transforms.Lambda(lambda x: x * (torch.from_numpy(np.random.binomial(1, 0.5, (1)).astype(np.float32) * 2 - 1))),
            transforms.Lambda(lambda x: x * (torch.from_numpy(np.random.uniform(low=0.25, high=1.5, size=(1)).astype(np.float32)))),
            transforms.Lambda(lambda x: x + (torch.from_numpy(np.random.uniform(low=-0.5, high=0.5, size=(1)).astype(np.float32))))
        ])
        test_transform = transforms.Compose([
            transforms.Resize(size=config.imsize),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])]
        )
    elif config.parent == 'Office31':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=config.imsize, scale=(0.08, 1.0)),
            transforms.Grayscale(3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(size=config.imsize),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
    return train_transform, test_transform



def get_dataset(config, logger, dset_taple, true_domain_i: int):
    if config.parent == 'Digit':
        resize=(32,32)
    else:
        if config.parent=='Visda':
            resize=(160,160)
        else:
            resize=(256,256)

    text_root = os.path.join("/nas/data/syamagami/GDA/data/", config.parent)
    text_train = os.path.join(text_root, f"{dset_taple[0]}.txt")

    df = pd.read_csv(text_train, sep=" ", names=("filename", "true_label"))
    filenames = df.filename.values
    true_labels = df.true_label.values

    train_files, test_files = train_test_split(filenames, train_size=0.8, shuffle=True)
    train_indices = sorted([list(filenames).index(x) for x in train_files])
    test_indices = sorted([list(filenames).index(x) for x in test_files])

    train_true_labels = true_labels[train_indices]
    train_true_domain_labels = np.zeros(len(train_indices), dtype=int) + true_domain_i
    train_pseudo_labels = np.zeros(len(train_indices))
    train_edls = np.zeros(len(train_indices), dtype=int)
    train_ed_feats = np.zeros((len(train_indices), config.out_dim))
    
    test_true_labels = true_labels[test_indices]
    test_true_domain_labels = np.zeros(len(test_indices), dtype=int) + true_domain_i

    train_transform, test_transform = get_transforms(config)

    img_root = os.path.join(text_root, "imgs")
    logger.info(f"  [{dset_taple[0]}] {len(train_indices)}")
    train_dataset = LabeledDataset(
        config, img_root, train_files, resize, train_transform, 
        train_true_labels, train_true_domain_labels,
        train_pseudo_labels, train_edls, train_ed_feats
    )
    logger.info(f"  [{dset_taple[0]}] {len(test_indices)}")
    test_dataset = LabeledDataset(
        config, img_root, test_files, resize, test_transform, 
        test_true_labels, test_true_domain_labels,
    )


    return train_dataset, test_dataset


    
def get_datasets(config, logger):

    train_dataset, test_dataset = get_dataset(config, logger, config.dset_taples[0], 0)
    for i, dset_taple in enumerate(config.dset_taples[1:]):
        train_dataset_t, test_dataset_t = get_dataset(config, logger, dset_taple, i+1)
        train_dataset.concat_dataset(train_dataset_t)
        test_dataset.concat_dataset(test_dataset_t)

    return train_dataset, test_dataset
    