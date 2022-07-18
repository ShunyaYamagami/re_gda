import os
import yaml
from easydict import EasyDict
import numpy as np
from time import time
import sys
sys.dont_write_bytecode = True  # pycacheを作らないようにする

# Spread Sheet用
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd 
from gspread_formatting import *

# メール送信用
import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate
import traceback


LOG_TEXT = 'SimSiam'


def config_to_execute(DATA_DOMAIN, PARENT, N_CUDA, log_dir_opt):
    print("===============================================")
    print(f'==============  {DATA_DOMAIN}  ==============')
    print("===============================================")
    # set config / log directory
    domain_initials = "".join([dname[0] for dname in DATA_DOMAIN.split('_')])
    CONFIG_FILE = os.path.join(os.path.join('config', PARENT, f"{domain_initials}.yaml"))
    config = yaml.load(open(CONFIG_FILE, "r"), Loader=yaml.FullLoader)
    config = EasyDict(config)
    log_dir = os.path.join("record", f"{config.dataset.parent}", f"CUDA{N_CUDA}", f"{domain_initials}_{config.model.base_model}{log_dir_opt}")

    return config, CONFIG_FILE, log_dir

