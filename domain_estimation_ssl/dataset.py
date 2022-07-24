import numpy as np
from joblib import Parallel, delayed
import os
import pandas as pd
from PIL import Image
import random
import torch
import torch.utils.data as data
from torchvision.transforms import transforms
import warnings
warnings.simplefilter('ignore')

from data_aug.gaussian_blur import GaussianBlur
from functions import *


def second_load(fi, config, root, filename, resize, mix_filenames=None) -> Image:
    im = Image.open(os.path.join(root, filename)).convert("RGB").resize(resize)
    grid = random.choice(grid) if isinstance(config.dataset.grid, list) else config.dataset.grid

    im = input_const_values(im, resize, const_abs=False, const_pha=True, n_random = resize[0] * resize[1] // 6, const_value=0 )  # 位相・振幅に一定値を入れる．
    im = get_jigsaw(im, resize, grid)

    return fi, im

    
def load(fi, config, root, filename, resize) -> Image:
    im = Image.open(os.path.join(root, filename)).convert("RGB").resize(resize)
    grid = random.choice(grid) if isinstance(config.dataset.grid, list) else config.dataset.grid

    if config.cuda_dir == 1:
        im = input_const_values(im, resize, const_abs=False, const_pha=True, n_random = resize[0] * resize[1] // 6, const_value=0 )  # 位相・振幅に一定値を入れる．
        im = get_jigsaw(im, resize, grid)
    elif config.cuda_dir == 2:
        im = get_jigsaw(im, resize, grid)
    elif config.cuda_dir == 3:
        im = input_const_values(im, resize, const_abs=False, const_pha=True, n_random = resize[0] * resize[1] // 6, const_value=0 )  # 位相・振幅に一定値を入れる．
        im = get_jigsaw(im, resize, grid)
        im = mask_randomly(im, resize, square_edge=20, rate=0.3)
    else:
        if fi == 0:
            print("  ------ Augmentation 未選択 -----") 

    return fi, im


class LabeledDataset(data.Dataset):
    def __init__(self, config, img_root, filenames, labels, domain_label, resize, transform=None):
        self.config = config
        self.root = img_root
        self.filenames = filenames
        self.labels = labels
        self.resize = resize
        if self.config.model.ssl != 'random_pseudo':
            self.edls = np.array(config.edls) if 'edls' in config.keys() else np.empty(0)  # 引数をget_datasetsから変えるのめんどいから，configにedlsをつけるという暴挙に．
            self.transform = transform
        else:
            all_filenames = get_all_filenames(config)
            # ランダムなラベルを付与
            self.edls = np.random.randint(0, 2, size=(len(all_filenames), self.config.pseudo_out_dim))
            self.transform = get_transforms(self.config, 'tensor')

        if config.lap == 1:
            self.processed = Parallel(n_jobs=4, verbose=1)([delayed(load)(fi, self.config, self.root, filename, resize) for fi, filename in enumerate(filenames)])
        else:
            mix_filenames = get_mix_filenames(config, self.edls)
            self.processed = Parallel(n_jobs=4, verbose=1)([delayed(second_load)(fi, self.config, self.root, filename, resize, mix_filenames[edl]) for fi, (filename, edl) in enumerate(zip(filenames, self.edls))])

        self.processed.sort(key=lambda x: x[0])  # 順番を元データ順に (https://qiita.com/kaggle_grandmaster-arai-san/items/4276079bf5e16b7de7a7#%E8%A8%88%E7%AE%97%E9%A0%86%E5%BA%8F%E3%81%AE%E8%A9%B1)
        self.imgs = np.array([t[1] for t in self.processed])
        self.domain_labels = np.array([domain_label] * len(self.imgs), dtype=np.int)


    def __getitem__(self, index):
        img = self.imgs[index]
        
        img1 = self.transform(img)
        img2 = self.transform(img)        
        edl = torch.tensor(self.edls[index], dtype=torch.float32) if len(self.edls) > 0 else torch.tensor(self.edls, dtype=torch.float16)

        return img1, img2, edl

    def __len__(self):
        return len(self.imgs)

    def concat_dataset(self, dataset):
        assert self.root == dataset.root
        # self.imgs.extend(dataset.imgs)
        self.imgs = np.concatenate([self.imgs, dataset.imgs])
        self.labels = np.concatenate([self.labels, dataset.labels])
        self.domain_labels = np.concatenate([self.domain_labels, dataset.domain_labels])


def get_transforms(config, mode):
    if config.dataset.parent == 'Digit':
        if mode == 'train':
            transform = transforms.Compose([
                # transforms.RandomResizedCrop(size=config.model.imsize),
                transforms.RandomResizedCrop(size=config.model.imsize, scale=(0.08, 1.0)),
                transforms.Grayscale(3),
                # GaussianBlur(kernel_size=int(0.3 * config.model.imsize), min=0.1, max=2.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5],[0.5]),
                transforms.Lambda(lambda x: x * (torch.from_numpy(np.random.binomial(1, 0.5, (1)).astype(np.float32) * 2 - 1))),
                transforms.Lambda(lambda x: x * (torch.from_numpy(np.random.uniform(low=0.25, high=1.5, size=(1)).astype(np.float32)))),
                transforms.Lambda(lambda x: x + (torch.from_numpy(np.random.uniform(low=-0.5, high=0.5, size=(1)).astype(np.float32))))
            ])
        elif mode == 'eval':
            transform = transforms.Compose([
                transforms.Resize(size=config.model.imsize),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize([0.5],[0.5])]
            )

    elif config.dataset.parent == 'Office31':
        if mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(size=config.model.imsize, scale=(0.08, 1.0)),
                transforms.Grayscale(3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        elif mode == 'eval':
            transform = transforms.Compose([
                transforms.Resize(size=config.model.imsize),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        elif mode == 'tensor':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        # train_transform = transforms.Compose([
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
    return transform



def get_dataset(config, dset_taple, domain_label, transform=None):
    print(f"  ----- get dataset {dset_taple[0]} -----")
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

    if config.dataset.parent == "Digit":
        filenames_seq = list(range(len(filenames)))  # filenameのインデックスの連番作成
        use_idx = sorted(random.sample(filenames_seq, config.dataset.sampling_num))  # Digitはデータ多すぎるのでランダムサンプリング
    else:
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

    transform = get_transforms(config, mode)

    ld = get_dataset(config, config.dataset.dset_taples[0], 0, transform)
    for i, dset_name in enumerate(config.dataset.dset_taples[1:]):
        ld_t = get_dataset(config, dset_name, i + 1, transform)

        ld.concat_dataset(ld_t)
    return ld
