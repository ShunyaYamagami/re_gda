import os
import yaml
from easydict import EasyDict
import argparse
import numpy as np
from time import time

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


LOG_TEXT = '初期状態'


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


# get nmi score
def get_nmi(log_dir):
  with open(os.path.join(log_dir, "nmi.txt"), 'r') as f:
    nmi = f.read().split(':')[-1].split('\n')[0]
  with open(os.path.join(log_dir, "nmi_class.txt"), 'r') as f:
    nmi_class = f.read().split(':')[-1].split('\n')[0]
  nmi, nmi_class = round(float(nmi), 5), round(float(nmi_class), 5)

  return nmi, nmi_class

  
def log_spread_sheet(config, log_dir, DATA_DOMAIN, spread_message):
  # connect to Spread Sheet
  KEY_JSON_FILE = '../domainestimation-cd0e815895a3.json'
  SPREADSHEET_KEY = '1IHb7CAeFnVsymVtxXy_1IHIdh-_hVeO63eX4txWr-bg'
  scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
  creds = ServiceAccountCredentials.from_json_keyfile_name(KEY_JSON_FILE,scope)
  client = gspread.authorize(creds)
  sheet = client.open_by_key(SPREADSHEET_KEY).sheet1

  # get score
  nmi, nmi_class = get_nmi(log_dir)

  # log on next row of Spread Sheet
  next_row = len(sheet.col_values(1)) + 1  # spread sheetの最終行+1行目
  update_dict = {
    'domain': DATA_DOMAIN,
    'base_model': config.model.base_model,
    'batch_size': config.batch_size,
    'epochs': config.epochs,
    'jigsaw': config.dataset.jigsaw,
    'fourier': config.dataset.fourier,
    'bikou': spread_message,
    'NMI': nmi,
    'NMI_Class': nmi_class
  } 
  for i, v in enumerate(update_dict.values()):
    sheet.update_cell(next_row, i+1, v)
    
  # color the cells with best score in each group
  alphabets = [chr(65 + i) for i in range(len(update_dict))]  # 取得列のアルファベット(大文字)
  ANMI = alphabets[len(update_dict) - 2]  # NMIの列アルファベット
  ANMIC = alphabets[len(update_dict) - 1]  # NMI_Classの列アルファベット
  
  all_values = np.array(sheet.get_all_values())[:, :len(update_dict)]
  white_data_fmt = cellFormat(backgroundColor=color(1, 1, 1))
  emphasis_1st_fmt = cellFormat(backgroundColor=color(1, .45, .45))
  emphasis_2nd_fmt = cellFormat(backgroundColor=color(.9, .7, .7))
  format_cell_range(sheet, f'{ANMI}2:{ANMIC}{len(all_values)}', white_data_fmt)
  
  df = pd.DataFrame(all_values[1:][:], columns=all_values[:1][0])
  df = df.replace('', np.nan).dropna().astype({'NMI': float, 'NMI_Class': float})

  nmi_max_index = df.groupby(['domain', 'base_model'])['NMI'].idxmax().values
  nmi_min_class_index = df.groupby(['domain', 'base_model'])['NMI_Class'].idxmin().values
  nmi_second_max_index = df.drop(index=nmi_max_index).groupby(['domain', 'base_model'])['NMI'].idxmax().values
  nmi_second_min_class_index = df.drop(index=nmi_min_class_index).groupby(['domain', 'base_model'])['NMI_Class'].idxmin().values


  for (max_nmi, min_nmi_class) in zip(nmi_max_index, nmi_min_class_index):
      format_cell_range(sheet, f'{ANMI}{max_nmi + 2}:{ANMI}{max_nmi + 2}', emphasis_1st_fmt)
      format_cell_range(sheet, f'{ANMIC}{min_nmi_class + 2}:{ANMIC}{min_nmi_class + 2}', emphasis_1st_fmt)
  for (second_max_nmi, second_min_nmi_class) in zip(nmi_second_max_index, nmi_second_min_class_index):
      format_cell_range(sheet, f'{ANMI}{second_max_nmi + 2}:{ANMI}{second_max_nmi + 2}', emphasis_2nd_fmt)
      format_cell_range(sheet, f'{ANMIC}{second_min_nmi_class + 2}:{ANMIC}{second_min_nmi_class + 2}', emphasis_2nd_fmt)



def send_email(
    not_error=True,
    subject='実行終了のお知らせ', body_text='実行終了',
    error_subject='エラーのお知らせ', error_text='エラー発生', error_message=None,
    config=None, log_dir=None, DATA_DOMAIN=None,
):
    # SMTPサーバに接続
    sendAddress = 'othello200063@gmail.com'
    password = 'qnsqtqaughestyil'
    smtpobj = smtplib.SMTP('smtp.gmail.com', 587)
    smtpobj.starttls()
    smtpobj.login(sendAddress, password)

    fromAddress = 'imlfajtmmh@gmail.com'
    toAddress = 'imlfajtmmh@gmail.com'

    if not_error:
        # メール作成
        nmi, nmi_class = get_nmi(log_dir)
        body_text += f""" \n {DATA_DOMAIN} \n nmi:{nmi}, nmi_class:{nmi_class} \n 
          epochs:{config.epochs}, jigsaw:{config.dataset.jigsaw}, fourier:{config.dataset.fourier}"""
        msg = MIMEText(body_text)
        msg['Subject'] = subject
    else:
        msg = MIMEText(f"{error_text} : {error_message}")
        msg['Subject'] = error_subject

    msg['From'] = fromAddress
    msg['To'] = toAddress
    msg['Date'] = formatdate()

    # 作成したメールを送信
    smtpobj.send_message(msg)
    smtpobj.close()

