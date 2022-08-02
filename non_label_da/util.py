import os
import shutil
import torch



def to_cuda(x, device):
    if isinstance(x, list):
        t = [r.to(device) for r in x]
    else:
        t = x.to(device)
    return t

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

