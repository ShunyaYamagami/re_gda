import argparse
from easydict import EasyDict
from time import time
import numpy as np
import pandas as pd
import os
import yaml

from clustering import run_clustering
from dataset import get_datasets
from simclr import SimCLR
from log_functions import log_spread_sheet, send_email, get_body_text
from util import get_device, set_logger_writer, save_config_file


parser = argparse.ArgumentParser(description='choose config')
parser.add_argument('--config', type=str, default="./config/Office31/webcam_amazon.yaml")
parser.add_argument('--log_dir', type=str, default="record")  # デフォルトのlog_dirの後ろに文字や数字指定して保存フォルダの重複を防ぐ．もっと良いlog_dirの指定方法がある気がする．
parser.add_argument('--num_laps', default=1)  # デフォルトのlog_dirの後ろに文字や数字指定して保存フォルダの重複を防ぐ．もっと良いlog_dirの指定方法がある気がする．
parser.add_argument('--spread_message', type=str, default="")  # デフォルトのlog_dirの後ろに文字や数字指定して保存フォルダの重複を防ぐ．もっと良いlog_dirの指定方法がある気がする．
parser.add_argument('--cuda_dir', default=-1)  # デフォルトのlog_dirの後ろに文字や数字指定して保存フォルダの重複を防ぐ．もっと良いlog_dirの指定方法がある気がする．
parser.add_argument('--test_counter', default=0)  # デフォルトのlog_dirの後ろに文字や数字指定して保存フォルダの重複を防ぐ．もっと良いlog_dirの指定方法がある気がする．
args = parser.parse_args()



def set_config(args):
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config = EasyDict(config)
    # from args
    config.config_path = args.config
    config.log_dir = args.log_dir
    config.num_laps = int(args.num_laps)
    config.spread_message = args.spread_message
    config.cuda_dir = int(args.cuda_dir)
    config.test_counter = int(args.test_counter)
    # set manually
    config.device = get_device()
    if 'sampling_num' not in config.dataset.keys():
        config.dataset.sampling_num = -1
    # dirs
    config.log_dir_origin = config.log_dir
    config.target_dsets_name = "_".join(np.array(config.dataset.dset_taples, dtype=object)[:,0])  # dset名のリスト
    config.checkpoints_dir = os.path.join(config.log_dir, 'checkpoints')

    save_config_file(config, config.config_path, config.checkpoints_dir)
    return config


def random_pseudo_laps(config, logger, dataset):
    if config.test_counter == 0:
        conv_edls = "int"
        switch_dataset = True
    elif config.test_counter == 1:
        conv_edls = "int"
        switch_dataset = False
    elif config.test_counter == 2:
        conv_edls = "float"
        switch_dataset = True
    elif config.test_counter == 3:
        conv_edls = "random"
        switch_dataset = False
    elif config.test_counter == 4:
        conv_edls = "random"
        switch_dataset = True

    ### edlsを整数にするか小数のままか, 完全ランダムなedlsにするか否か
    if conv_edls == 'int':
        config.edls = np.round(config.edls)  # 整数に直す
    elif conv_edls == 'float':
        config.edls = config.edls  # 整数に直す
    elif conv_edls == 'random':
        config.edls = np.random.randint(0, 2, size=(len(dataset.edls), config.model.out_dim))  # 完全ランダムにする
    ### datasetを切り替えるか否か
    if switch_dataset:
        dataset = get_datasets(config, logger, 'train')  # datasetを切り替える
    else:
        dataset.edls = config.edls  # datasetを切り替えない
    
    logger.info(f"conv_edls: {conv_edls},  switch_dataset: {switch_dataset}")
    
    return config, dataset


def main():
    start_time = time()
    config = set_config(args)
    logger = set_logger_writer(config)
    mail_body_texts = []

    # if config.cuda_dir == 0:
    #     config.model.out_dim = 4
    # elif config.cuda_dir == 1:
    #     config.model.out_dim = 8
    # elif config.cuda_dir == 2:
    #     config.model.out_dim = 16
    # elif config.cuda_dir == 3:
    #     config.model.out_dim = 32
    # elif config.cuda_dir == 4:
    #     config.model.out_dim = 64
    # elif config.cuda_dir == 5:
    #     config.model.out_dim = 128

    logger.info(f"""    
    ===============================================
    ==============  {config.target_dsets_name}  ==============
    ===============================================
    ---------------------------------------------------------
        cuda_dir: {config.cuda_dir}
        train    batch_size: {config.batch_size},  epochs: {config.epochs}
        model    SSL: {config.model.ssl},  base_model: {config.model.base_model},  out_dim: {config.model.out_dim},
        dataset  grid: {config.dataset.grid},  sampling_num: {config.dataset.sampling_num}
        other    num_laps: {config.num_laps},  test_counter: {config.test_counter}
        log_dir: {config.log_dir}
        spread_message: {config.spread_message}
    ---------------------------------------------------------
    """)

    logger.info(f"\n=================  Training 1/{config.num_laps}周目  =================")
    config.lap = 1
    dataset = get_datasets(config, logger, 'train')
    simclr = SimCLR(dataset, config, logger)
    simclr.train()

    logger.info(f"=================  Clustering 1/{config.num_laps}  =================")
    run_clustering(config, logger)
    run_clustering(config, logger)
    run_clustering(config, logger)
        
    # for ilap in range(2, config.num_laps + 1):  # 何周するか
    #     config.lap = ilap
    #     logger.info(f"\n=================  Training {config.lap}/{config.num_laps}周目  =================")
    #     if config.model.ssl != 'random_pseudo':
    #         load_edls_path = os.path.join(config.log_dir, f'{config.target_dsets_name}_edls.csv')
    #         logger.info(f"  -----  Load EDLs from {load_edls_path}  -----")
    #         config.edls = np.loadtxt(load_edls_path, delimiter=",")
    #     else:
    #         load_edls_path = os.path.join(config.log_dir, f'sigmoid_pseudo.csv')
    #         logger.info(f"  -----  Load EDLs from {load_edls_path}  -----")
    #         config.edls = np.loadtxt(load_edls_path, delimiter=",")
    #     config.log_dir = os.path.join(config.log_dir_origin, f"lap{config.lap}")
    #     config.checkpoints_dir = os.path.join(config.log_dir, 'checkpoints')
    #     os.makedirs(config.checkpoints_dir, exist_ok=True)
       

    #     ## dataset = get_datasets(config, logger, 'train')  # datasetを切り替える
    #     config, dataset = random_pseudo_laps(config, logger, dataset)
    #     simclr = SimCLR(dataset, config, logger)
    #     simclr.train(does_load_model=True)
    #     # simclr.train(does_load_model=False)  ## 多分datasetの内容が変わるからモデルをロードしない方が良いと思う.
        
    #     logger.info(f"=================  Clustering {config.lap}/{config.num_laps}  =================")
    #     run_clustering(config, logger)
    #     run_clustering(config, logger)
    #     run_clustering(config, logger)


    # send_email(not_error=True, body_texts='\n'.join(mail_body_texts), config=config, nmi=config.nmi, nmi_class=config.nmi_class)


if __name__ == "__main__":
    main()
