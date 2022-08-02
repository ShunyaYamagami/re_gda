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
from train.run_source import run_source
from train.run_dann_OS import run_dann_OS
from train.run_mine import run_mine
from util import get_device, save_config_file, set_logger_writer


parser = argparse.ArgumentParser(description='choose config')
parser.add_argument('--config', type=str, default="./config/sample.yaml")
parser.add_argument('--log_dir', type=str, default="record")
parser.add_argument('--spread_message', type=str, default="")  # デフォルトのlog_dirの後ろに文字や数字指定して保存フォルダの重複を防ぐ．もっと良いlog_dirの指定方法がある気がする．
parser.add_argument('--cuda_dir', default="-1")
parser.add_argument('--ed_dir_name', type=str, default="")
args = parser.parse_args()




def set_config(args):
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config = EasyDict(config)
    # set from args
    config.config_path = args.config
    config.log_dir = args.log_dir
    config.spread_message = args.spread_message
    config.cuda_dir = int(args.cuda_dir)
    config.ed_dir_name = args.ed_dir_name
    # set manually
    config.device = get_device()
    config.target_dsets_name = "_".join(np.array(config.dset_taples, dtype=object)[:,0])  # dset名のリスト
    config.tensorboard_log_dir = os.path.join(config.log_dir, 'logs')
    config.checkpoints_dir = os.path.join(config.log_dir, 'checkpoints')

    if not config.training_mode == 'source':
        if config.clustering_method == "simCLR":
            config.est_domains_dir = os.path.join("estimated_domains", config.parent, config.ed_dir_name)
        elif config.clustering_method == "simCLR_OSDA":
            config.est_domains_dir = os.path.join("estimated_domains/OSDA", config.parent, config.ed_dir_name)
    else:
        # sourceの時はどうせ使わないので, エラー回避の為に適当なラベルを与えておく.
        config.est_domains_dir = os.path.join("estimated_domains", config.parent, "true_domain_labels")
    
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
        batch_size: {config.batch_size},  epochs: {config.epochs}
        change_epoch: {config.change_epoch},  change_epoch2: {config.change_epoch2}
        training_mode: {config.training_mode},  model: {config.model}, optim: {config.optim}
        gamma: {config.gamma},  theta: {config.theta},  sigam: {config.sigma},  num_history: {config.num_history}
        log_dir: {config.log_dir}
        est_domains_dir: {config.est_domains_dir}
        spread_message: {config.spread_message}
    ---------------------------------------------------------
    """)

    ld, ud, td_list = get_datasets(config, logger)

    if config.training_mode == "source":
        run_source(config, logger, writer, ld, td_list)
    elif config.training_mode == "dann_OS":
        run_dann_OS(config, logger, writer, ld, ud, td_list)
    elif config.training_mode == "mine":
        run_mine(config, logger, writer, ld, ud, td_list)
    


if __name__ == '__main__':
    main()
