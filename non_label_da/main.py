from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
from easydict import EasyDict
import numpy as np
import os
import shutil
import yaml
import torch
from tensorboardX import SummaryWriter

from dataset import get_datasets
from simclr import SimCLR
from util import get_device, save_config_file
from logging import getLogger, Formatter, StreamHandler, FileHandler, DEBUG

parser = argparse.ArgumentParser(description='choose config')
parser.add_argument('--config', type=str, default="./config/sample.yaml")
parser.add_argument('--log_dir', type=str, default="record")
parser.add_argument('--spread_message', type=str, default="")  # デフォルトのlog_dirの後ろに文字や数字指定して保存フォルダの重複を防ぐ．もっと良いlog_dirの指定方法がある気がする．
parser.add_argument('--cuda_dir', default="-1")
args = parser.parse_args()


def set_logger_writer(config):
    if os.path.exists(config.log_dir):
        shutil.rmtree(config.log_dir)
    os.makedirs(config.checkpoints_dir, exist_ok=True)
    logger = getLogger("show_loss_accuarcy")
    logger.setLevel(DEBUG)
    # formatter = Formatter('%(asctime)s [%(levelname)s] \n%(message)s')
    handlerSh = StreamHandler()
    handlerSh.setLevel(DEBUG)
    # handlerSh.setFormatter(formatter)
    logger.addHandler(handlerSh)
    handlerFile = FileHandler(os.path.join(config.log_dir, "prompts.log"))
    # handlerFile.setFormatter(formatter)
    handlerFile.setLevel(DEBUG)
    logger.addHandler(handlerFile)
    writer = SummaryWriter(log_dir=config.tensorboard_log_dir)
    return logger, writer


def set_config(args):
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config = EasyDict(config)
    # set from args
    config.config_path = args.config
    config.log_dir = args.log_dir
    config.spread_message = args.spread_message
    config.cuda_dir = int(args.cuda_dir)
    # set manually
    config.device = get_device()
    config.target_dsets_name = "_".join(np.array(config.dset_taples, dtype=object)[:,0])  # dset名のリスト
    config.tensorboard_log_dir = os.path.join(config.log_dir, 'logs')
    config.checkpoints_dir = os.path.join(config.log_dir, 'checkpoints')

    save_config_file(config, config.config_path, config.checkpoints_dir)
    return config



def main():
    config = set_config(args)
    logger, writer = set_logger_writer(config)

    logger.info(f"""    
    ===============================================
    ============== {config.target_dsets_name} ==============
    ===============================================
    ---------------------------------------------------------
        cuda_dir: {config.cuda_dir}
        batch_size: {config.batch_size},\t epochs: {config.epochs},\t model: {config.base_model},
        log_dir: {config.log_dir}
        spread_message: {config.spread_message}
    ---------------------------------------------------------
    """)

    train_dataset, test_dataset = get_datasets(config, logger)
    simclr = SimCLR(config, logger, writer, train_dataset, test_dataset)
    simclr.run_train()

if __name__ == '__main__':
    main()
