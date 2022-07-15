import argparse
from easydict import EasyDict
import numpy as np
import torch
from torchvision.transforms import transforms
import yaml

from data_aug.gaussian_blur import GaussianBlur
from clustering import run_clustering
from dataset import get_datasets
from simclr import SimCLR



parser = argparse.ArgumentParser(description='choose config')
parser.add_argument('--config', type=str, default="./config/Office31/webcam_amazon.yaml")
parser.add_argument('--log_dir', type=str, default="record")  # デフォルトのlog_dirの後ろに文字や数字指定して保存フォルダの重複を防ぐ．もっと良いlog_dirの指定方法がある気がする．
parser.add_argument('--lap', default=1)  # デフォルトのlog_dirの後ろに文字や数字指定して保存フォルダの重複を防ぐ．もっと良いlog_dirの指定方法がある気がする．
parser.add_argument('--second_flag', default=1)  # デフォルトのlog_dirの後ろに文字や数字指定して保存フォルダの重複を防ぐ．もっと良いlog_dirの指定方法がある気がする．
args = parser.parse_args()

def main(args):
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config = EasyDict(config)
    config.config_path = args.config
    config.log_dir = args.log_dir
    config.lap = int(args.lap)
    config.second_flag = int(args.second_flag)


    if config.dataset.parent == 'Digit':
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(32),
                                        transforms.Grayscale(),
                                        GaussianBlur(kernel_size=int(0.3 * 32), min=0.1, max=2.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5],[0.5]),
                                        transforms.Lambda(lambda x: x * (torch.from_numpy(np.random.binomial(1, 0.5, (1)).astype(np.float32) * 2 - 1))),
                                        transforms.Lambda(lambda x: x * (torch.from_numpy(np.random.uniform(low=0.25, high=1.5, size=(1)).astype(np.float32)))),
                                        transforms.Lambda(lambda x: x + (torch.from_numpy(np.random.uniform(low=-0.5, high=0.5, size=(1)).astype(np.float32))))
                                        ])
    elif config.dataset.parent == 'Office31':
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=config.model.imsize, scale=(0.08, 1.0)),
                                              transforms.Grayscale(3),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                              ])

    print(f"""    ---------------------------------------------------------
            batch_size: {config.batch_size},  epochs: {config.epochs}
            SSL: {config.model.ssl},  base_model: {config.model.base_model}
            jigsaw: {config.dataset.jigsaw},  fourier: {config.dataset.fourier}
            lap: {config.lap},  second_flag: {config.second_flag}
    ---------------------------------------------------------""")
    
    if config.lap == 1:
        print("================= 1周目 =================")
        dataset = get_datasets(config.dataset.parent, config.dataset.dset_taples, data_transforms, config.dataset.jigsaw, config.dataset.grid)
        simclr = SimCLR(dataset, config)
        simclr.train()

        print("================= Clustering 1 =================")
        run_clustering(config)

if __name__ == "__main__":
    main(args)
