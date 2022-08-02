import os
import numpy as np
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from loss.nt_xent import NTXentLoss
from util import get_models_func

torch.manual_seed(0)


apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False



class SimCLR(object):
    def __init__(self, dataset, config, logger):
        self.config = config
        self.logger = logger
        self.writer = SummaryWriter(log_dir = os.path.join(config.log_dir, 'logs'))
        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(config, config.batch_size, **config.loss)

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
        loss = - (loss1 + loss2) / 2  # それぞれの損失を最適化する

        return loss


    def _random_pseudo_step(self, epoch, model, x, criterion, edls, index):
        _, sigmoid_pseudo = model(x)
        loss = criterion(sigmoid_pseudo, edls)

        return loss


    def train(self, does_load_model=False):
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=2, drop_last=True)

        model = get_models_func(self.config)
        # ADD 2回目任意で1回目の重みをロードしてファインチューニングできればと．
        if does_load_model:
            model = self._load_pre_trained_weights(self.config, model)
        if self.config.model.ssl == 'simsiam':
            criterion = nn.CosineSimilarity(dim=1).to(self.config.device)  # コサイン類似度
        if self.config.model.ssl == 'random_pseudo':
            criterion = nn.BCEWithLogitsLoss().to(self.config.device)
        optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=eval(self.config.weight_decay))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
        if apex_support and self.config.fp16_precision:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2', keep_batchnorm_fp32=True)

        n_iter = 0
        for epoch in range(self.config.epochs):
            model.train()
            for xis, xjs, edls, index in train_loader:
                xis, xjs, edls, index = xis.to(self.config.device), xjs.to(self.config.device), edls.to(self.config.device), index.to(self.config.device)

                optimizer.zero_grad()

                if self.config.model.ssl == 'simclr':
                    loss = self._simclr_step(model, xis, xjs, edls)
                elif self.config.model.ssl == 'simsiam':
                    loss = self._simsiam_step(model, xis, xjs, criterion, edls)
                elif self.config.model.ssl == 'random_pseudo':
                    loss = self._random_pseudo_step(epoch, model, xis, criterion, edls, index)

                if n_iter % self.config.log_every_n_steps == 0:
                    self.logger.info(f'Epoch:{epoch}/{self.config.epochs}({n_iter}) loss:{loss}')
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                if apex_support and self.config.fp16_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                scheduler.step()
                n_iter += 1

            torch.save(model.state_dict(), os.path.join(self.config.checkpoints_dir, "model.pth"))


    def _load_pre_trained_weights(self, config, model):
        try:
            if config.lap == 1:
                load_model_dir = config.log_dir
            elif config.lap == 2:
                load_model_dir = config.log_dir_origin
            else:
                load_model_dir = os.path.join(config.log_dir_origin, f"lap{config.lap - 1}")
            self.logger.info(f"  ----- Loaded pre-trained model from {load_model_dir} for SSL  -----")
            state_dict = torch.load(os.path.join('./', load_model_dir, 'checkpoints', 'model.pth'))
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            self.logger.info("  ----- Pre-trained weights not found. Training from scratch.  -----")

        return model


    # def _validate(self, model, valid_loader):
    #     # validation steps
    #     with torch.no_grad():
    #         model.eval()
    #         valid_loss = 0.0
    #         counter = 0
    #         for (xis, xjs), _ in valid_loader:
    #             xis, xjs = xis.to(self.config.device), xjs.to(self.config.device)
    #             

    #             loss = self._step(model, xis, xjs, counter)
    #             valid_loss += loss.item()
    #             counter += 1
    #         valid_loss /= counter
    #     model.train()
    #     return valid_loss
