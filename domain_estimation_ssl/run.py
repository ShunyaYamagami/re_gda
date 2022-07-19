import argparse
from easydict import EasyDict
import numpy as np
import os
import pandas as pd
import torch
from torchvision.transforms import transforms
import yaml
from time import time

from data_aug.gaussian_blur import GaussianBlur
from clustering import run_clustering
from dataset import get_datasets
from simclr import SimCLR
from log_functions import *


parser = argparse.ArgumentParser(description='choose config')
parser.add_argument('--config', type=str, default="./config/Office31/webcam_amazon.yaml")
parser.add_argument('--log_dir', type=str, default="record")  # デフォルトのlog_dirの後ろに文字や数字指定して保存フォルダの重複を防ぐ．もっと良いlog_dirの指定方法がある気がする．
parser.add_argument('--num_laps', default=1)  # デフォルトのlog_dirの後ろに文字や数字指定して保存フォルダの重複を防ぐ．もっと良いlog_dirの指定方法がある気がする．
parser.add_argument('--spread_message', type=str, default="")  # デフォルトのlog_dirの後ろに文字や数字指定して保存フォルダの重複を防ぐ．もっと良いlog_dirの指定方法がある気がする．
args = parser.parse_args()

def main(args):
    start_time = time()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config = EasyDict(config)
    config.config_path = args.config
    config.log_dir = args.log_dir
    config.num_laps = int(args.num_laps)
    config.spread_message = args.spread_message
    mail_body_texts = []

    print(f"""    ---------------------------------------------------------
        batch_size: {config.batch_size},  epochs: {config.epochs}
        SSL: {config.model.ssl},  base_model: {config.model.base_model}
        jigsaw: {config.dataset.jigsaw},  fourier: {config.dataset.fourier},  grid: {config.dataset.grid}
        num_laps: {config.num_laps}
    ---------------------------------------------------------
    """)

    try:
        # print(f"\n=================  1/{config.num_laps}周目  =================")
        # config.lap = 1
        # dataset = get_datasets(config, 'train')
        # simclr = SimCLR(dataset, config)
        # simclr.train()

        # print(f"=================  Clustering 1/{config.num_laps}  =================")
        # feats, edls_dataset, nmi, nmi_class = run_clustering(config)
        # log_spread_sheet(config, nmi, nmi_class)
        # mail_body_texts.append(get_body_text(config, start_time, nmi, nmi_class))
            
        for ilap in range(2, config.num_laps + 1):  # 何周するか
            config.lap = ilap
            print(f"=================  {config.lap}/{config.num_laps}周目  =================")
            config.edls = pd.read_csv(os.path.join(config.log_dir, 'cluster_pca_gmm.csv'), names=['domain_label'], dtype=int).domain_label.values
            config.log_dir__old = config.log_dir  # simclrのdoes_load_modelだけのために設けた．
            config.log_dir += f'__{config.lap}'

            dataset = get_datasets(config, 'train')
            simclr = SimCLR(dataset, config)
            simclr.train(does_load_model=True)
            
            print(f"=================  Clustering {config.lap}/{config.num_laps}  =================")
            feats, edls_dataset, nmi, nmi_class = run_clustering(config)
            log_spread_sheet(config, nmi, nmi_class)
            mail_body_texts.append(get_body_text(config, start_time, nmi, nmi_class))
    
        send_email(not_error=True, body_texts='\n'.join(mail_body_texts), config=config, nmi=nmi, nmi_class=nmi_class)
    except:
        error_message = traceback.format_exc()
        send_email(not_error=False, error_message=error_message)

if __name__ == "__main__":
    main(args)
