import argparse
from easydict import EasyDict
import numpy as np
import os
import shutil
from time import time
import torch
import yaml
from tensorboardX import SummaryWriter

from clustering import run_clustering
from dataset import get_datasets
from simclr import SimCLR
from log_functions import log_spread_sheet, send_email, get_body_text
from logging import getLogger, Formatter, StreamHandler, FileHandler, DEBUG


parser = argparse.ArgumentParser(description='choose config')
parser.add_argument('--config', type=str, default="./config/Office31/webcam_amazon.yaml")
parser.add_argument('--log_dir', type=str, default="record")  # デフォルトのlog_dirの後ろに文字や数字指定して保存フォルダの重複を防ぐ．もっと良いlog_dirの指定方法がある気がする．
parser.add_argument('--num_laps', default=1)  # デフォルトのlog_dirの後ろに文字や数字指定して保存フォルダの重複を防ぐ．もっと良いlog_dirの指定方法がある気がする．
parser.add_argument('--spread_message', type=str, default="")  # デフォルトのlog_dirの後ろに文字や数字指定して保存フォルダの重複を防ぐ．もっと良いlog_dirの指定方法がある気がする．
parser.add_argument('--cuda_dir', default=-1)  # デフォルトのlog_dirの後ろに文字や数字指定して保存フォルダの重複を防ぐ．もっと良いlog_dirの指定方法がある気がする．
args = parser.parse_args()


def get_device():
    """ GPU or CPU """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running on:", device)
    return device

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
    return logger

def save_config_file(config, original_path, model_checkpoints_folder):
    os.makedirs(model_checkpoints_folder, exist_ok=True)
    shutil.copy(original_path, os.path.join(model_checkpoints_folder, 'config.yaml'))
    config_list = []
    for k,v in config.items():
        config_list.append(f"{k}: {v}\n")
    with open(os.path.join(model_checkpoints_folder, "config.txt"), 'w') as f:
        f.writelines(config_list)

def set_config(args):
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config = EasyDict(config)
    # from args
    config.config_path = args.config
    config.log_dir = args.log_dir
    config.num_laps = int(args.num_laps)
    config.spread_message = args.spread_message
    config.cuda_dir = int(args.cuda_dir)
    # set manually
    config.dataset.target_dsets = np.array(config.dataset.dset_taples)[:,0]  # dset名のリスト
    config.tensorboard_log_dir = os.path.join(config.log_dir, 'logs')
    config.checkpoints_dir = os.path.join(config.log_dir, 'checkpoints')
    config.model_path = os.path.join(config.log_dir, 'checkpoints', 'model.pth')
    # config.logger, config.writer = set_logger_writer(config)
    config.device = get_device()
    save_config_file(config, config.config_path, config.checkpoints_dir)

    return config


def main():
    start_time = time()
    config = set_config(args)
    logger = set_logger_writer(config)
    mail_body_texts = []
    logger.info(f"""    
    ===============================================
    ========== {"_".join(config.dataset.target_dsets)} ==============
    ===============================================
    ---------------------------------------------------------
        cuda_dir: {config.cuda_dir}
        batch_size: {config.batch_size},  epochs: {config.epochs}
        SSL: {config.model.ssl},  base_model: {config.model.base_model}
        jigsaw: {config.dataset.jigsaw},  fourier: {config.dataset.fourier},  grid: {config.dataset.grid}
        num_laps: {config.num_laps}, sampling_num: {config.dataset.sampling_num}
        log_dir: {config.log_dir}
    ---------------------------------------------------------
    """)

    # try:
    logger.info(f"\n=================  1/{config.num_laps}周目  =================")
    config.lap = 1
    config.pseudo_out_dim = 8
    dataset = get_datasets(config, 'train')
    simclr = SimCLR(dataset, config, logger)
    simclr.train()

    logger.info(f"=================  Clustering 1/{config.num_laps}  =================")
    feats, edls_dataset = run_clustering(config, logger)
    feats, edls_dataset = run_clustering(config, logger)
    feats, edls_dataset = run_clustering(config, logger)
    # log_spread_sheet(config, config.nmi, config.nmi_class)
    # mail_body_texts.append(get_body_text(config, start_time, config.nmi, config.nmi_class))
        
    # for ilap in range(2, config.num_laps + 1):  # 何周するか
    #     config.lap = ilap
    #     logger.info(f"=================  {config.lap}/{config.num_laps}周目  =================")
    #     config.edls = pd.read_csv(os.path.join(config.log_dir, 'cluster_pca_gmm.csv'), names=['domain_label'], dtype=int).domain_label.values
    #     config.log_dir__old = config.log_dir  # simclrのdoes_load_modelだけのために設けた．
    #     config.log_dir += f'__{config.lap}'

    #     dataset = get_datasets(config, 'train')
    #     simclr = SimCLR(dataset, config)
    #     simclr.train(does_load_model=True)
        
    #     logger.info(f"=================  Clustering {config.lap}/{config.num_laps}  =================")
    #     feats, edls_dataset, config.nmi, config.nmi_class = run_clustering(config)
    #     log_spread_sheet(config, config.nmi, config.nmi_class)
    #     mail_body_texts.append(get_body_text(config, start_time, config.nmi, config.nmi_class))

    # send_email(not_error=True, body_texts='\n'.join(mail_body_texts), config=config, nmi=config.nmi, nmi_class=config.nmi_class)
    # except:
    #     error_message = traceback.format_exc()
    #     send_email(not_error=False, error_message=error_message)


if __name__ == "__main__":
    main()
