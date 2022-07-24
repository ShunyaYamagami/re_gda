import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
import scipy.stats as stats
from sklearn.model_selection import train_test_split

import torch.utils.data as data
from torchvision.transforms import transforms


def load(root, filename, resize):
    return Image.open(os.path.join(root, filename)).convert("RGB").resize(resize)
    

class LabeledDataset(data.Dataset):
    def __init__(self, config, root, filenames, labels, true_domain_label, resize, transform, edls=None):
        self.config = config
        self.root = root
        self.labels = labels
        self.transform = transform
        self.imgs = Parallel(n_jobs=4, verbose=1)([delayed(load)(self.root, filename, resize) for filename in filenames])
        self.true_domain_labels = [true_domain_label] * len(self.imgs)  # 真のドメインラベル
        self.num_history = config.num_history
        self.pseudo_label = np.zeros(len(self.imgs))
        self.history = np.zeros((len(self.imgs), self.num_history))
        self.edls = edls
        

    def __getitem__(self, index):
        img, label = self.imgs[index], self.labels[index]
        # true_domain_label = self.true_domain_labels[index]
        edl = self.edls[index]
        hist = self.pseudo_label[index]
        img = self.transform(img)
        # return img, label, true_domain_label, hist, index
        return img, label, edl, hist, index

    def __len__(self):
        return len(self.imgs)
    
    # def update_domain_label(self, new_domain_labels):
    #     self.domain_labels = new_domain_labels

    def update_labels(self, index, results):
        self.history[index] = np.hstack((self.history[index, 1:], results.reshape(-1, 1)))

    def update_pseudo(self):
        target, _ = stats.mode(self.history, axis=1)
        self.pseudo_label = np.squeeze(target)

    def concat_dataset(self, dataset):
        assert self.root == dataset.root
        self.imgs.extend(dataset.imgs)
        self.labels = np.concatenate([self.labels, dataset.labels])
        self.true_domain_labels = np.concatenate([self.true_domain_labels, dataset.true_domain_labels])
        self.edls = np.concatenate([self.edls, dataset.edls])
        self.history = np.concatenate([self.history, dataset.history])
        self.pseudo_label = np.concatenate([self.pseudo_label, dataset.pseudo_label])


def relabeling_dataset(dataset, union_class): # 複数データセットでクラスラベルを共通化する．(0,1,2), (4,7,9)などで二つ合わせると歯抜けになる時にそれぞれに0,1,2,3,4,5を割り振るための関数．テストセットに対してはout of distributionに-1を振る．
    union_class = sorted(union_class)
    labels = dataset.labels
    unk = len(union_class)
    labels = [union_class.index(l) if l in union_class else unk for l in labels]
    dataset.labels = labels

    return dataset


def get_transforms(config):
    if config.parent == 'Digit':
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    elif config.parent == 'Office31':
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
    return train_transform, test_transform


def get_lut_dataset(config, dset_taple, true_domain_label, edls):
    print("************************************************")
    print("************************************************")
    print("************************************************")
#     """
#     args:
#         dset_taple:
#         true_domain_label: 真のドメインラベル
#         edls: 推定ドメインラベル
#     return:
#         labeled_dataset:  train用クラスラベルありデータ
#         unlabeled_dataset:  train用クラスラベルなしデータ
#         test_dataset: test用データ
#         edls: 推定ドメインラベル(後半だけ残したもの)
#     """
#     ### load train
#     text_root = os.path.join("/nas/data/syamagami/GDA/data/", config.parent)
#     text_train = os.path.join(text_root, f"{dset_taple[0]}_train.txt")
    
#     if config.parent == 'Digit':
#         resize=(32,32)
#     else:
#         if config.parent=='Visda':
#             resize=(160,160)
#         else:
#             resize=(256,256)

#     df = pd.read_csv(text_train, sep=" ", names=("filename", "label"))
#     filenames = df.filename.values
#     labels = df.label.values

#     labeled_index = [i for i, l in enumerate(labels) if l in dset_taple[1]]
#     unlabeled_index = [i for i, l in enumerate(labels) if (l in dset_taple[2]) and (l not in dset_taple[1])]

#     labeled_filenames = filenames[labeled_index]
#     labeled_labels = labels[labeled_index]
#     unlabeled_filenames = filenames[unlabeled_index]
#     unlabeled_labels = labels[unlabeled_index]

#     # 推定ドメインラベル
#     target_edls = edls[:len(filenames)]
#     edls = edls[len(filenames):]  # 次のdset用に前半を削除したedls
#     labeled_dataset_edls = target_edls[labeled_index]
#     unlabeled_dataset_edls = target_edls[unlabeled_index]

#     train_transform, test_transform = get_transforms(config)

#     img_root = os.path.join(text_root, "imgs")
#     labeled_dataset = LabeledDataset(config, img_root, labeled_filenames, labeled_labels, true_domain_label, resize, train_transform, labeled_dataset_edls)
#     unlabeled_dataset = LabeledDataset(config, img_root, unlabeled_filenames, unlabeled_labels, true_domain_label, resize, train_transform, unlabeled_dataset_edls)

#     ### load test
#     text_test = os.path.join(text_root, f"{dset_taple[0]}_test.txt")
#     df = pd.read_csv(text_test, sep=" ", names=("filename", "label"))
#     filenames = df.filename.values
#     labels = df.label.values
#     test_index = [i for i, l in enumerate(labels) if (l in dset_taple[2]) and (l not in dset_taple[1])]
#     test_filenames = filenames[test_index]
#     test_labels = labels[test_index]
#     test_dataset = LabeledDataset(config, img_root, test_filenames, test_labels, true_domain_label, resize, test_transform)

#     return labeled_dataset, unlabeled_dataset, test_dataset, edls



def get_lut_dataset_office31(config, dset_taple, true_domain_label, edls):
    """ 
    Office31用
        trainファイルとtestファイルが分かれていないので,この関数内で分ける.
        testはdset_taple[1]以外のクラスを使うらしいので, dset_taple[1]以外のクラスの画像を7:3に分ける.
        つまり, train時に, ラベル付き画像の方がラベル無し画像より圧倒的に多いので, 教師あり学習に近い形となり精度が出やすくなってそう.
    args:
        dset_taple[1]: labeled (ラベルが付いたクラス)
        dset_taple[2]: unlabeled (ラベルが付いていないクラス)
    """
    ### load train
    text_root = os.path.join("/nas/data/syamagami/GDA/data/", config.parent)
    text_train = os.path.join(text_root, f"{dset_taple[0]}.txt")
    
    if config.parent == 'Digit':
        resize=(32,32)
    else:
        if config.parent=='Visda':
            resize=(160,160)
        else:
            resize=(256,256)

    df = pd.read_csv(text_train, sep=" ", names=("filename", "label"))
    filenames = df.filename.values
    labels = df.label.values

    labeled_train_index = [i for i, l in enumerate(labels) if l in dset_taple[1]]
    unlabeled_train_index_t = [i for i, l in enumerate(labels) if (l in dset_taple[2]) and (l not in dset_taple[1])]

    unlabeled_train_index, unlabeled_test_index = train_test_split(unlabeled_train_index_t, train_size=0.7, shuffle=True)

    labeled_filenames = filenames[labeled_train_index]
    labeled_labels = labels[labeled_train_index]
    unlabeled_filenames = filenames[unlabeled_train_index]
    unlabeled_labels = labels[unlabeled_train_index]

    # 推定ドメインラベル
    target_edls = edls[:len(filenames)]
    edls = edls[len(filenames):]  # 次のdset用に前半を削除したedls
    labeled_train_edls = target_edls[labeled_train_index]
    unlabeled_train_edls = target_edls[unlabeled_train_index]

    train_transform, test_transform = get_transforms(config)

    img_root = os.path.join(text_root, "imgs")
    config.logger.info(f"--- read {dset_taple[0]} labeled ---")
    labeled_dataset = LabeledDataset(config, img_root, labeled_filenames, labeled_labels, true_domain_label, resize, train_transform, labeled_train_edls)
    config.logger.info(f"--- read {dset_taple[0]} unlabeled ---")
    unlabeled_dataset = LabeledDataset(config, img_root, unlabeled_filenames, unlabeled_labels, true_domain_label, resize, train_transform, unlabeled_train_edls)

    ### load test
    test_filenames = filenames[unlabeled_test_index]
    test_labels = labels[unlabeled_test_index]
    unlabeled_test_edls = target_edls[unlabeled_test_index]
    config.logger.info(f"--- read {dset_taple[0]} test ---")
    test_dataset = LabeledDataset(config, img_root, test_filenames, test_labels, true_domain_label, resize, test_transform, unlabeled_test_edls)

    return labeled_dataset, unlabeled_dataset, test_dataset, edls



def get_datasets(config):
    """
    args:
        mode: train or test -> transformを決める
    reutrn:
        ld: labeled_dataset
        ud: unknown_labeled_dataset
        td_list: test_datasetのdsetごとのリスト
        len(union_classes): クラス数(???)
    """
    # csvから推定ドメインラベルを取得(Noneは許さない)
    clustering_filename = "_".join([d[0] for d in config.dset_taples]) + ".csv"
    if config.clustering_method == "simCLR":
        edls = np.loadtxt(os.path.join("clustering", config.parent, clustering_filename), delimiter=",").astype(int)
    elif config.clustering_method == "simCLR_OSDA":
        edls = np.loadtxt(os.path.join("clustering/OSDA", config.parent, clustering_filename), delimiter=",").astype(int)
    else:
        print("edls are not set")
        raise ValueError("edls are not set")

    union_classes = np.unique(sum([t[1] for t in config.dset_taples],[]))
    config.num_class = len(union_classes) + 1  # Out-Of-Distributionのクラス数はconfig.num_class-1

    get_lut_func = get_lut_dataset_office31 if config.parent == 'Office31' else get_lut_dataset

    td_list = []
    ld, ud, td, edls = get_lut_func(config, config.dset_taples[0], 0, edls)
    td_list.append([config.dset_taples[0][0], relabeling_dataset(td, union_classes)])

    for i, dset_taple in enumerate(config.dset_taples[1:]):
        ld_t, ud_t, td_t, edls = get_lut_func(config, dset_taple, i + 1, edls)
        ld.concat_dataset(ld_t)
        ud.concat_dataset(ud_t)
        td_list.append([dset_taple[0], relabeling_dataset(td_t, union_classes)])
    assert edls is None or len(edls) == 0

    ld = relabeling_dataset(ld, union_classes)

    return ld, ud, td_list


