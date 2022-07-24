from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
from easydict import EasyDict
import os
import shutil
import yaml
import torch
from tensorboardX import SummaryWriter

from dataset import get_datasets
from train.run_source import run_source
from train.run_dann_OS import run_dann_OS
from logging import getLogger, Formatter, StreamHandler, FileHandler, DEBUG



parser = argparse.ArgumentParser(description='choose config')
parser.add_argument('--config', type=str, default="./config/sample.yaml")
parser.add_argument('--log_dir', type=str, default="record")
parser.add_argument('--spread_message', type=str, default="")  # デフォルトのlog_dirの後ろに文字や数字指定して保存フォルダの重複を防ぐ．もっと良いlog_dirの指定方法がある気がする．
parser.add_argument('--cuda_dir', default="-1")
args = parser.parse_args()



def get_device():
    """ GPU or CPU """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running on:", device)

    return device


def save_config_file(original_path, model_checkpoints_folder):
    """ 今回のconfig fileの内容を書きだす（手動で追加した部分も書き出してよいかもな） """
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy(original_path, os.path.join(model_checkpoints_folder, 'config.yaml'))


def set_logger_writer(config):
    logger = getLogger("show_loss_accuarcy")
    # formatter = Formatter('%(asctime)s [%(levelname)s] \n%(message)s')
    handlerSh = StreamHandler()
    # handlerFile = FileHandler(os.path.join(config.log_dir, "prompts.log"))
    # handlerSh.setFormatter(formatter)
    # handlerSh.setLevel(DEBUG)
    # handlerFile.setFormatter(formatter)
    # handlerFile.setLevel(DEBUG)
    logger.setLevel(DEBUG)
    logger.addHandler(handlerSh)
    # logger.addHandler(handlerFile)
    if os.path.exists(config.tensorboard_log_dir):
        shutil.rmtree(config.tensorboard_log_dir)
    os.makedirs(config.tensorboard_log_dir)
    writer = SummaryWriter(config.tensorboard_log_dir)
    return logger, writer


def set_config(args):
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config = EasyDict(config)
    # set from args
    config.config_path = args.config
    config.log_dir = args.log_dir
    config.cuda_dir = int(args.cuda_dir)
    config.spread_message = args.spread_message
    # set manually
    config.checkpoint_dir = os.path.join(config.log_dir, "checkpoints")
    config.tensorboard_log_dir = os.path.join(config.log_dir, "logs")
    logger, writer = set_logger_writer(config)
    config.logger = logger
    config.writer = writer
    config.device = get_device()

    save_config_file(config.config_path, config.log_dir)
    return config



def main():
    config = set_config(args)
    print(f"""    ---------------------------------------------------------
        cuda_dir: {config.cuda_dir}
        batch_size: {config.batch_size},  epochs: {config.epochs}
        training_mode: {config.training_mode},  model: {config.model}, optim: {config.optim}
        gamma: {config.gamma},  theta: {config.theta},  num_history: {config.num_history}
        log_dir: {config.log_dir}
    ---------------------------------------------------------
    """)
    
    ld, ud, td_list = get_datasets(config)
    if config.training_mode == "source":
        run_source(config, ld, td_list)
    elif config.training_mode == "dann_OS":
        run_dann_OS(config, ld, ud, td_list)


if __name__ == '__main__':
    main()
