import numpy as np
import os
import shutil
import torch
from tensorboardX import SummaryWriter

from models.baseline_encoder import Encoder
from models.alexnet_simclr import AlexSimCLR
from models.resnet_simclr import ResNetSimCLR
from logging import getLogger, Formatter, StreamHandler, FileHandler, DEBUG


def get_device():
    """ GPU or CPU """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running on:", device)
    return device

    
def get_models_func(config):
    if config.dataset.parent == "Digit":
        if config.model.base_model == 'encoder':
            model = Encoder(config, input_dim=3)
        else:
            print("  ------  モデル未選択 -----")
    elif config.dataset.parent == "Office31":
        if config.model.base_model == 'encoder':
            model = Encoder(config, input_dim=3)
        # elif config.model.base_model == "alexnet":
        #     model = AlexSimCLR(config.model.out_dim)
        # else:
        #     model = ResNetSimCLR(config.model.base_model, config.model.out_dim)
    
    model = model.to(config.device)
    return model


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






