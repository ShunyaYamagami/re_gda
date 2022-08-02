from easydict import EasyDict
import numpy as np
import os
import shutil
import yaml
import torch
from tensorboardX import SummaryWriter

from logging import getLogger, Formatter, StreamHandler, FileHandler, DEBUG



def get_device():
    """ GPU or CPU """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running on:", device)
    return device


def save_config_file(config, original_path, model_checkpoints_folder):
    os.makedirs(model_checkpoints_folder, exist_ok=True)
    shutil.copy(original_path, os.path.join(model_checkpoints_folder, 'config.yaml'))
    config_list = []
    for k,v in config.items():
        config_list.append(f"{k}: {v}\n")
    with open(os.path.join(model_checkpoints_folder, "config.txt"), 'w') as f:
        f.writelines(config_list)


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