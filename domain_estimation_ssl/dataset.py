import numpy as np
from joblib import Parallel, delayed
import os
import pandas as pd
from PIL import Image
import random
import torch
import torch.utils.data as data
from torchvision.transforms import transforms

from data_aug.gaussian_blur import GaussianBlur
from functions import *


def second_load(fi, config, root, filename, resize, mix_filenames=None) -> Image:
    im1 = Image.open(os.path.join(root, filename)).convert("RGB").resize(resize)
    origin_im = im1
    grid = random.choice(grid) if isinstance(config.dataset.grid, list) else config.dataset.grid
    # if config.dataset.fourier:
    #     # im1 = input_const_values(im1, resize, const_abs=False, const_pha=True, n_random = resize[0] * resize[1] // 5, const_value=0)  # 位相・振幅に一定値を入れる．
    #     # im1 = input_random_values(im1, resize, randomize_abs=False, randomize_pha=True, n_random = resize[0] * resize[1] // 10)  # 位相・振幅にランダム値を入れる．
    #     im1 = mix_amp_phase_and_mixup(im1, root, resize, mix_filenames, mix_amp=False, mix_pha=True, mixup=False, LAMB = 0.5)
    #     # im1 = cutmix_spectrums(im1, resize, root, mix_filenames, does_mix_amp=False, does_mix_pha=True, mix_edge_div=4 )
    # if config.dataset.jigsaw:
    #     im1 = get_jigsaw(im1, resize, grid)
    #     # im1 = mask_randomly(im1, resize, square_edge=20, rate=0.3)
    #     # im1 = cutmix_self(im1, resize, grid, n_cutmix=4)
    #     # im1 = cutmix_other(im1, resize, root, mix_filenames, mix_edge_div=5, crop_part='center')

    if config.cuda_dir == 0:
        # im1 = input_const_values(im1, resize, const_abs=False, const_pha=True, n_random = resize[0] * resize[1] // 6, const_value=0 )  # 位相・振幅に一定値を入れる．
        im1 = mix_amp_phase_and_mixup(im1, root, resize, mix_filenames, mix_amp=False, mix_pha=True, mixup=True, LAMB = 0.5)
        im1 = get_jigsaw(im1, resize, grid)
        # im1 = mask_randomly(im1, resize, square_edge=20, rate=0.3)
    elif config.cuda_dir == 1:
        im1 = input_const_values(im1, resize, const_abs=False, const_pha=True, n_random = resize[0] * resize[1] // 6, const_value=0 )  # 位相・振幅に一定値を入れる．
        im1 = get_jigsaw(im1, resize, grid)
    elif config.cuda_dir == 2:
        im1 = mix_amp_phase_and_mixup(im1, root, resize, mix_filenames, mix_amp=False, mix_pha=True, mixup=False, LAMB = 0.5)
        im1 = get_jigsaw(im1, resize, grid)
        # im1 = mask_randomly(im1, resize, square_edge=20, rate=0.3)
    elif config.cuda_dir == 3:
        im1 = get_jigsaw(im1, resize, grid)

    im2 = im1.copy()
    return fi, im1, im2, origin_im


    
def load(fi, config, root, filename, resize) -> Image:
    im1 = Image.open(os.path.join(root, filename)).convert("RGB").resize(resize)
    origin_im = im1
    grid = random.choice(grid) if isinstance(config.dataset.grid, list) else config.dataset.grid

    # if config.dataset.fourier:
    #     im1 = input_const_values(im1, resize, const_abs=False, const_pha=True, n_random = resize[0] * resize[1] // 6, const_value=0 )  # 位相・振幅に一定値を入れる．
    # #     im1 = input_random_values(im1, resize, randomize_abs=False, randomize_pha=True, n_random = resize[0] * resize[1] // 10)  # 位相・振幅にランダム値を入れる．
    # if config.dataset.jigsaw:
    #     im1 = get_jigsaw(im1, resize, grid)
    #     # im1 = mask_randomly(im1, resize, square_edge=20, rate=0.3)
    #     # im1 = cutmix_self(im1, resize, grid, n_cutmix=4)

    if config.cuda_dir == 0:
        im1 = input_const_values(im1, resize, const_abs=False, const_pha=True, n_random = resize[0] * resize[1] // 6, const_value=0 )  # 位相・振幅に一定値を入れる．
        im1 = get_jigsaw(im1, resize, grid)
        im1 = mask_randomly(im1, resize, square_edge=20, rate=0.3)
    elif config.cuda_dir == 1:
        im1 = input_const_values(im1, resize, const_abs=False, const_pha=True, n_random = resize[0] * resize[1] // 6, const_value=0 )  # 位相・振幅に一定値を入れる．
        im1 = get_jigsaw(im1, resize, grid)
    elif config.cuda_dir == 2:
        im1 = get_jigsaw(im1, resize, grid)
        im1 = mask_randomly(im1, resize, square_edge=20, rate=0.3)
    elif config.cuda_dir == 3:
        im1 = get_jigsaw(im1, resize, grid)

    im2 = im1.copy()
    return fi, im1, im2, origin_im


class LabeledDataset(data.Dataset):
    def __init__(self, config, img_root, filenames, labels, domain_label, resize, transform=None, target_all_filenames=None):
        self.config = config
        self.root = img_root
        self.filenames = filenames
        self.labels = labels
        self.resize = resize
        self.transform = transform
        self.edls = list(config.edls) if 'edls' in config.keys() else []  # 引数をget_datasetsから変えるのめんどいから，configにedlsをつけるという暴挙に．

        if config.lap == 1:
            self.processed = Parallel(n_jobs=4, verbose=1)([delayed(load)(fi, self.config, self.root, filename, resize) for fi, filename in enumerate(filenames)])
        else:
            mix_filenames = get_mix_filenames(config, self.edls)
            self.processed = Parallel(n_jobs=4, verbose=1)([delayed(second_load)(fi, self.config, self.root, filename, resize, mix_filenames[edl]) for fi, (filename, edl) in enumerate(zip(filenames, self.edls))])
        self.processed.sort(key=lambda x: x[0])  # 順番を元データ順に (https://qiita.com/kaggle_grandmaster-arai-san/items/4276079bf5e16b7de7a7#%E8%A8%88%E7%AE%97%E9%A0%86%E5%BA%8F%E3%81%AE%E8%A9%B1)
        self.imgs1 = [t[1] for t in self.processed]
        self.imgs2 = [t[2] for t in self.processed]
        self.origin_imgs = [t[3] for t in self.processed]

        self.domain_labels = np.array([domain_label] * len(self.imgs1), dtype=np.int)


    def __getitem__(self, index):
        img1 = self.imgs1[index]
        img2 = self.imgs2[index]
        origin_img = self.origin_imgs[index]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        # img2 = self.transform(origin_img)
        edls = torch.tensor(self.edls[index], dtype=torch.int8) if self.edls else self.edls
        
        return img1, img2, edls

    def __len__(self):
        return len(self.imgs1)

    def concat_dataset(self, dataset):
        assert self.root == dataset.root
        self.imgs1.extend(dataset.imgs1)
        self.imgs2.extend(dataset.imgs2)
        self.origin_imgs.extend(dataset.origin_imgs)
        self.labels = np.concatenate([self.labels, dataset.labels])
        self.domain_labels = np.concatenate([self.domain_labels, dataset.domain_labels])


def get_transforms(config):
    if config.dataset.parent == 'Digit':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=config.model.imsize),
            transforms.Grayscale(),
            GaussianBlur(kernel_size=int(0.3 * config.model.imsize), min=0.1, max=2.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5]),
            transforms.Lambda(lambda x: x * (torch.from_numpy(np.random.binomial(1, 0.5, (1)).astype(np.float32) * 2 - 1))),
            transforms.Lambda(lambda x: x * (torch.from_numpy(np.random.uniform(low=0.25, high=1.5, size=(1)).astype(np.float32)))),
            transforms.Lambda(lambda x: x + (torch.from_numpy(np.random.uniform(low=-0.5, high=0.5, size=(1)).astype(np.float32))))
        ])
        valid_transforms = transforms.Compose([
            transforms.Resize(size=config.model.imsize),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])]
        )

    elif config.dataset.parent == 'Office31':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=config.model.imsize, scale=(0.08, 1.0)),
            transforms.Grayscale(3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        valid_transforms = transforms.Compose([
            transforms.Resize(size=config.model.imsize),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        # train_transforms = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),  # ドメイン情報の学習にFlipは要らない気もするが
        #     transforms.RandomResizedCrop(size=32),
        #     transforms.RandomApply([
        #         transforms.ColorJitter(
        #             brightness=0.5, 
        #             contrast=0.5, 
        #             saturation=0.5, 
        #             hue=0.1)
        #         ], p=0.8),
        #     transforms.RandomGrayscale(p=0.1),
        #     transforms.GaussianBlur(kernel_size=5),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # ])
    return train_transforms, valid_transforms


def get_dataset(config, dset_taple, domain_label, transform=None):
    # load train
    parent = config.dataset.parent
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
    config.resize = resize

    df = pd.read_csv(text_train, sep=" ", names=("filename", "label"))
    filenames = df.filename.values
    labels = df.label.values

    use_idx = np.array([i for i,l in enumerate(labels) if l in dset_labels])
    filenames = filenames[use_idx]
    labels = labels[use_idx]

    img_root = os.path.join(text_root, "imgs")
    dataset = LabeledDataset(config, img_root, filenames, labels, domain_label, resize, transform)

    return dataset
    
    
def get_datasets(config, mode):
    """ get_datasetから取得できるldはLabeledDatasetのインスタンス
        imgs, labels, domain_labelsはクラス管理
        concat_datasetでクラスの値を更新する """
    if mode == 'train':
        transform, _ = get_transforms(config)
    elif mode == 'eval':
        _, transform = get_transforms(config)

    
    ld = get_dataset(config, config.dataset.dset_taples[0], 0, transform)
    for i, dset_name in enumerate(config.dataset.dset_taples[1:]):
        ld_t = get_dataset(config, dset_name, i + 1, transform)

        ld.concat_dataset(ld_t)
    return ld
