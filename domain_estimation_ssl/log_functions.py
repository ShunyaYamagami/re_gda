import os
import yaml
from easydict import EasyDict
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


def log_spread_sheet(config, nmi, nmi_class):
  spread_message = f'{config.spread_message}__{config.lap}'

  # connect to Spread Sheet
  KEY_JSON_FILE = '../domainestimation-cd0e815895a3.json'
  SPREADSHEET_KEY = '1IHb7CAeFnVsymVtxXy_1IHIdh-_hVeO63eX4txWr-bg'
  scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
  creds = ServiceAccountCredentials.from_json_keyfile_name(KEY_JSON_FILE,scope)
  client = gspread.authorize(creds)
  sheet = client.open_by_key(SPREADSHEET_KEY).sheet1

  # log on next row of Spread Sheet
  next_row = len(sheet.col_values(1)) + 1  # spread sheetの最終行+1行目
  update_dict = {
    'domain': config.target_dsets_name,
    'base_model': config.model.base_model,
    'batch_size': config.batch_size,
    'epochs': config.epochs,
    'bikou': spread_message,
    'NMI': round(nmi, 5),
    'NMI_Class': round(nmi_class, 5),
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
    subject='実行終了のお知らせ', body_texts='実行終了',
    error_subject='エラーのお知らせ', error_text='エラー発生', error_message=None,
    config=None, nmi=None, nmi_class=None,
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
        msg = MIMEText(body_texts)
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



def get_body_text(config, start_time, nmi, nmi_class):
    mail_body_text = f"""
    ==========  {config.target_dsets_name}  {config.lap}/{config.num_laps}周目  ==========
        実行時間: {(time() - start_time)/60:.2f} 分
        nmi: {round(nmi, 5)},  nmi_class: {round(nmi_class, 5)},  
        ------------------------------------------------------------------------
        batch_size: {config.batch_size},  epochs: {config.epochs},  
        SSL: {config.model.ssl},  base_model: {config.model.base_model},  
        grid: {config.dataset.grid},  num_laps: {config.num_laps},  
        ------------------------------------------------------------------------
    """

    return mail_body_text

