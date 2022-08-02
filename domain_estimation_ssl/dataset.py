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
# augs
from functions import get_jigsaw, input_const_values, input_random_values, mix_amp_phase_and_mixup, mask_randomly, cutmix_self, cutmix_other, cutmix_spectrums, leave_amp_pha_big_small
# get data
from functions import get_all_filenames, get_mix_filenames


# def second_load(fi, config, root, filename, resize, mix_filenames=None) -> Image:
#     im = Image.open(os.path.join(root, filename)).convert("RGB").resize(resize)
#     grid = random.choice(grid) if isinstance(config.dataset.grid, list) else config.dataset.grid

#     im = input_const_values(im, resize, const_abs=False, const_pha=True, n_random = resize[0] * resize[1] // 6, const_value=0 )  # 位相・振幅に一定値を入れる．
#     im = get_jigsaw(im, resize, grid)

#     return fi, im

    
def load(fi, config, root, filename, resize) -> Image:
    im = Image.open(os.path.join(root, filename)).convert("RGB").resize(resize)
    grid = random.choice(grid) if isinstance(config.dataset.grid, list) else config.dataset.grid

    
    if config.cuda_dir == 0:
        # if config.test_counter == 1:
            im = get_jigsaw(im, resize, grid)
            
    if config.cuda_dir == 1:
        # if config.test_counter == 2:
            im = cutmix_self(im, resize, grid=3, n_cutmix=2)
        
    if config.cuda_dir == 2:
        # if config.test_counter == 0:
            im = input_const_values(im, resize, const_abs=False, const_pha=True, n_random=0, const_value=0)
            im = get_jigsaw(im, resize, grid)
    if config.cuda_dir == 3:
        # if config.test_counter == 1:
            im = input_random_values(im, resize, randomize_abs=False, randomize_pha=True, n_random=0)
            im = get_jigsaw(im, resize, grid)

    if config.cuda_dir == 4:
        # if config.test_counter == 0:
            im = input_const_values(im, resize, const_abs=False, const_pha=True, n_random=0, const_value=0)
            im = cutmix_self(im, resize, grid=3, n_cutmix=2)

    if config.cuda_dir == 5:
        # if config.test_counter == 1:
            im = input_random_values(im, resize, randomize_abs=False, randomize_pha=True, n_random=0)
            im = cutmix_self(im, resize, grid=3, n_cutmix=2)


    return fi, im


class LabeledDataset(data.Dataset):
    def __init__(self, config, img_root, filenames, labels, true_domain_label, resize, transform=None):
        self.config = config
        self.root = img_root
        self.filenames = filenames
        self.labels = labels
        self.resize = resize
        self.transform = transform

        if self.config.model.ssl != 'random_pseudo':
            self.edls = config.edls if 'edls' in config.keys() else np.empty(0)  # 引数をget_datasetsから変えるのめんどいから，configにedlsをつけるという暴挙に．
        else:
            all_filenames = get_all_filenames(config)
            if config.lap == 1:
                self.edls = np.random.randint(0, 2, size=(len(all_filenames), self.config.model.out_dim))  # ランダムなラベルを付与
            else:
                self.edls = config.edls

        if config.lap == 1:
            self.processed = Parallel(n_jobs=4, verbose=1)([delayed(load)(fi, self.config, self.root, filename, resize) for fi, filename in enumerate(filenames)])
        else:
            self.processed = Parallel(n_jobs=4, verbose=1)([delayed(load)(fi, self.config, self.root, filename, resize) for fi, filename in enumerate(filenames)])
            # mix_filenames = get_mix_filenames(config, self.edls)
            # self.processed = Parallel(n_jobs=4, verbose=1)([delayed(second_load)(fi, self.config, self.root, filename, resize, mix_filenames[edl]) for fi, (filename, edl) in enumerate(zip(filenames, self.edls))])

        self.processed.sort(key=lambda x: x[0])  # 順番を元データ順に (https://qiita.com/kaggle_grandmaster-arai-san/items/4276079bf5e16b7de7a7#%E8%A8%88%E7%AE%97%E9%A0%86%E5%BA%8F%E3%81%AE%E8%A9%B1)
        self.imgs = np.array([t[1] for t in self.processed])
        self.true_domain_labels = np.array([true_domain_label] * len(self.imgs), dtype=np.int)  # 真のドメインラベル


    def __getitem__(self, index):
        img = self.imgs[index]
        
        img1 = self.transform(img)
        img2 = self.transform(img)        
        edl = torch.tensor(self.edls[index]).float() if len(self.edls) > 0 else torch.tensor(self.edls).float()

        return img1, img2, edl, index

    def __len__(self):
        return len(self.imgs)

    def concat_dataset(self, dataset):
        assert self.root == dataset.root
        # self.imgs.extend(dataset.imgs)
        self.imgs = np.concatenate([self.imgs, dataset.imgs])
        self.labels = np.concatenate([self.labels, dataset.labels])
        self.true_domain_labels = np.concatenate([self.true_domain_labels, dataset.true_domain_labels])



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
    return transform



def get_dataset(config, logger, dset_taple, domain_label, mode):
    parent = config.dataset.parent
    text_root = os.path.join("/nas/data/syamagami/GDA/data/", parent)

    if parent == 'Digit':
        text_train = os.path.join(text_root, "{}_train.txt".format(dset_taple[0]))
        resize=(36,36)
    else:
        text_train = os.path.join(text_root, "{}.txt".format(dset_taple[0]))
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

    if parent == "Digit":
        filenames_seq = list(range(len(filenames)))  # filenameのインデックスの連番作成
        use_idx = sorted(random.sample(filenames_seq, config.dataset.sampling_num))  # Digitはデータ多すぎるのでランダムサンプリング
        filenames = filenames[use_idx]
        labels = labels[use_idx]
    logger.info(f"  read {dset_taple[0]} {len(filenames)}")

    transform = get_transforms(config, mode)
    img_root = os.path.join(text_root, "imgs")
    dataset = LabeledDataset(config, img_root, filenames, labels, domain_label, resize, transform)

    return dataset
    
    

def get_datasets(config, logger, mode):
    ld = get_dataset(config, logger, config.dataset.dset_taples[0], 0, mode)
    for i, dset_name in enumerate(config.dataset.dset_taples[1:]):
        ld_t = get_dataset(config, logger, dset_name, i + 1, mode)

        ld.concat_dataset(ld_t)
    return ld
