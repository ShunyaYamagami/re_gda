import argparse
from easydict import EasyDict
import numpy as np
import os
import pandas as pd
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


    print(f"""    ---------------------------------------------------------
            batch_size: {config.batch_size},  epochs: {config.epochs}
            SSL: {config.model.ssl},  base_model: {config.model.base_model}
            jigsaw: {config.dataset.jigsaw},  fourier: {config.dataset.fourier}
            lap: {config.lap},  second_flag: {config.second_flag}
    ---------------------------------------------------------""")

    # if config.lap == 1:
    #     print("================= 1周目 =================")
    #     # dataset = get_datasets(config.dataset.parent, config.dataset.dset_taples, data_transforms, config.dataset.jigsaw, config.dataset.grid)
    #     dataset = get_datasets(config, 'train')
    #     simclr = SimCLR(dataset, config)
    #     simclr.train()

    #     print("================= Clustering 1 =================")
    #     run_clustering(config)

        
    if config.second_flag:
        config.lap += 1
        print(f"================= {config.lap}周目 =================")
        config.edls = pd.read_csv(os.path.join(config.log_dir, 'cluster_pca_gmm.csv'), names=['domain_label'], dtype=int).domain_label.values
        config.log_dir__old = config.log_dir  # simclrのdoes_load_modelだけのために設けた．
        config.log_dir += '__edls'

        dataset_2 = get_datasets(config, 'train')
        simclr_2 = SimCLR(dataset_2, config)
        simclr_2.train(does_load_model=True)
        
        print(f"================= Clustering {config.lap} =================")
        run_clustering(config)

if __name__ == "__main__":
    main(args)
