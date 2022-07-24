import os
import numpy as np
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models.baseline_encoder import Encoder
from models.alexnet_simclr import AlexSimCLR
from models.resnet_simclr import ResNetSimCLR
from loss.nt_xent import NTXentLoss



apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


torch.manual_seed(0)


def get_models_func(config, device):
    if config.dataset.parent == "Digit":
        if config.model.base_model == 'encoder':
            model = Encoder(config.model.ssl, input_dim=3, out_dim=config.model.out_dim, pseudo_out_dim=config.pseudo_out_dim)
        else:
            print("  ------  モデル未選択 -----")
    elif config.dataset.parent == "Office31":
        if config.model.base_model == "alexnet":
            model = AlexSimCLR(config.model.out_dim)
        elif config.model.base_model == 'encoder':
            model = Encoder(config.model.ssl, input_dim=3, out_dim=config.model.out_dim, pseudo_out_dim=config.pseudo_out_dim)
        else:
            model = ResNetSimCLR(config.model.base_model, config.model.out_dim)
    
    model = model.to(device)
    return model



class SimCLR(object):
    def __init__(self, dataset, config, logger):
        self.config = config
        self.logger = logger
        self.device = self._get_device()
        self.writer = SummaryWriter(log_dir=config.tensorboard_log_dir)
        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(config, self.device, config.batch_size, **config.loss)


    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Running on: {device}")
        return device


    def _simclr_step(self, model, xis, xjs, edls):
        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]
        rjs, zjs = model(xjs)  # [N,C]
        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs, edls)
        return loss


    def _simsiam_step(self, model, xis, xjs, criterion, edls):
        # get the representations and the projections
        _, zis, pis = model(xis)  # [N,C]  ミニバッチなので負例も大量に含まれている.
        _, zjs, pjs = model(xjs)  # [N,C]
        zis, zjs = zis.detach(), zjs.detach()  # 勾配の停止
        # normalize projection feature vectors
        zis, pis = F.normalize(zis, dim=1), F.normalize(pis, dim=1)
        zjs, pjs = F.normalize(zjs, dim=1), F.normalize(pjs, dim=1)

        loss1 = criterion(pis, zjs).mean()  # predictopnとprojectionを比較して類似度計算
        loss2 = criterion(pjs, zis).mean()  # 2つの損失関数を用意する
        loss = -(loss1 + loss2) / 2  # それぞれの損失を最適化する
        return loss


    def _random_pseudo_step(self, model, x, criterion, edls):
        pred = model(x)
        pred = F.normalize(pred, dim=1)
        loss = criterion(pred, edls)

        return loss

    def train(self, does_load_model=False):
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=2, drop_last=True)

        model = get_models_func(self.config, self.device)
        # ADD 2回目任意で1回目の重みをロードしてファインチューニングできればと．
        if does_load_model:
            model = self._load_pre_trained_weights(self.config, model)
            # de-fine tuningみたいな
            # for layers in list(model.children())[:-3]:
            #     for param in layers.parameters():
            #         param.requires_grad = False
            
        # set criterion
        if self.config.model.ssl == 'simsiam':
            model.ssl = self.config.model.ssl
            criterion = nn.CosineSimilarity(dim=1).to(self.device)  # コサイン類似度
        if self.config.model.ssl == 'random_pseudo':
            model.ssl = self.config.model.ssl
            criterion = nn.BCEWithLogitsLoss().to(self.device)
        # set optimizer
        optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=eval(self.config.weight_decay))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
        if apex_support and self.config.fp16_precision:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2', keep_batchnorm_fp32=True)

        n_iter = 0
        for epoch_counter in range(self.config.epochs):
            for xis, xjs, edls in train_loader:
                xis, xjs, edls = xis.to(self.device), xjs.to(self.device), edls.to(self.device)

                optimizer.zero_grad()

                if self.config.model.ssl == 'simclr':
                    loss = self._simclr_step(model, xis, xjs, edls)
                elif self.config.model.ssl == 'simsiam':
                    loss = self._simsiam_step(model, xis, xjs, criterion, edls)
                elif self.config.model.ssl == 'random_pseudo':
                    loss = self._random_pseudo_step(model, xis, criterion, edls)

                if n_iter % self.config.log_every_n_steps == 0:
                    self.logger.info(f'Epoch:{epoch_counter}/{self.config.epochs}({n_iter}) loss:{loss}')
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                if apex_support and self.config.fp16_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                scheduler.step()
                n_iter += 1

            torch.save(model.state_dict(), self.config.model_path)


    def _load_pre_trained_weights(self, config, model):
        try:
            if config.lap == 1:
                load_model_dir = config.log_dir
            else:
                load_model_dir = config.log_dir__old
            self.logger.info(f"  ----- Loaded pre-trained model from {load_model_dir} for SSL  -----")
            state_dict = torch.load(os.path.join('./', load_model_dir, 'checkpoints', 'model.pth'))
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            self.logger.info("  ----- Pre-trained weights not found. Training from scratch.  -----")

        return model


    def _validate(self, model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            counter = 0
            for (xis, xjs), _ in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
