import os
from sched import scheduler
import numpy as np
import random
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

### Clustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import accuracy_score


from models.baseline_encoder import Encoder
from models.mine_net import MineNet
from loss.nt_xent import NTXentLoss
# from clustering import clustering_exec
from util import to_cuda

# apex_support = False
# try:
#     sys.path.append('./apex')
#     from apex import amp
#     apex_support = True
# except:
#     print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
#     apex_support = False

torch.manual_seed(0)


    
class SimCLR(object):
    def __init__(self, config, logger, writer, train_dataset, test_dataset):
        self.config = config
        self.logger = logger
        self.writer = writer

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=2, drop_last=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=2, drop_last=True)

        parent, base_model, out_dim, device = self.config.parent, self.config.base_model, self.config.out_dim, self.config.device
        batch_size, loss, lr, lr_weight = self.config.batch_size, self.config.loss, self.config.lr, self.config.lr_weight

        if base_model == "encoder":
            self.class_encoder = Encoder(input_dim=3, out_dim=out_dim).to(device)
            self.domain_encoder = Encoder(input_dim=3, out_dim=out_dim).to(device)
            self.mine_net = MineNet(input_dim=out_dim, hidden_dim=128).to(device)
        # set criterion
        self.c_nt_xent_criterion = NTXentLoss(config, batch_size, **loss)
        self.d_nt_xent_criterion = NTXentLoss(config, batch_size, **loss)
        # set optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.class_encoder.parameters(), 'lr': lr},
            {'params': self.domain_encoder.parameters(), 'lr': lr * lr_weight},
            {'params': self.mine_net.parameters(), 'lr': lr}
        ], lr=config.lr)
        # set scheduler
        self.trn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader), eta_min=0, last_epoch=-1)


    
    def _clustering_exec(self, feats, indices, mode):
        true_labels = self.train_loader.dataset.true_labels[indices]
        true_domain_labels = self.train_loader.dataset.true_domain_labels[indices]

        # dim_reduce = TSNE(n_components=2, perplexity=30, verbose=1, n_jobs=3)
        dim_reduce = PCA()
        feats_dim_reduced = dim_reduce.fit_transform(feats)

        domain_clustering = GMM(n_components=self.config.num_domain)
        class_clustering = GMM(n_components=self.config.num_class)

        domain_cluster = np.array(domain_clustering.fit_predict(feats_dim_reduced))
        class_cluster = np.array(class_clustering.fit_predict(feats_dim_reduced))

        nmi_domain = NMI(true_domain_labels, domain_cluster)
        nmi_class = NMI(true_labels, class_cluster)

        # accuracyはドメイン種類数が2と予め設定していた時のみ求める
        if self.config.num_domain == 2 and len(np.unique(true_domain_labels)) == 2:
            domain_accuracy = np.max([accuracy_score(true_domain_labels, domain_cluster), 1 - accuracy_score(true_domain_labels, domain_cluster)])
        else:
            domain_accuracy = -1
        
        if mode == 'class':
            self.c_nmis = [nmi_domain, nmi_class, domain_accuracy]
        elif mode == 'domain':
            self.d_nmis = [nmi_domain, nmi_class, domain_accuracy]

        
        return domain_cluster


    def _class_simclr_step(self, xis, xjs):
        # get the representations and the projections
        ris, zis = self.class_encoder(xis)  # [N,C]
        rjs, zjs = self.class_encoder(xjs)  # [N,C]
        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        feat = F.normalize(rjs, dim=1)

        loss = self.c_nt_xent_criterion(zis, zjs, mode="class")
        return loss, feat
    
    def _domain_simclr_step(self, xis, xjs, edl, epoch):
        # get the representations and the projections
        ris, zis = self.domain_encoder(xis)  # [N,C]
        rjs, zjs = self.domain_encoder(xjs)  # [N,C]
        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        feat = F.normalize(rjs, dim=1)

        loss = self.d_nt_xent_criterion(zis, zjs, mode="domain", edls=edl, epoch=epoch)
        return loss, feat


    def _simclr_step(self, xis, xjs, model, nt_xent_criterion, mode, epoch, edl=None):
        # get the representations and the projections
        ris, zis = self.model(xis)  # [N,C]
        rjs, zjs = self.model(xjs)  # [N,C]
        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        feat = F.normalize(rjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs, mode, edl, epoch)
        return loss, feat


    def _get_mine_loss(self, class_feat, domain_feat, b_edls):
        """
            jointはGfより得た特徴とドメイン特徴を対応させる.
            marginalはドメイン特徴の順番をランダムにシャッフルして関係ないドメイン特徴と対応させる

            サンプルの取り方:  異なる推定ドメインラベルを持ったデータからサンプリングする事にする.
            each_not_edl_indices: ドメインラベルが0,1,2の3種類だったら, each_not_edl_indicesの長さは3. それぞれそのドメインに該当しない要素のインデックスのリストを格納. 
            例)  batch_size:8,  バッチのドメインラベル: [0 1 0 1 1 2 0 2]    =>    each_not_edl_indices: [[1, 3, 4, 5, 7], [0, 2, 5, 6, 7], [0, 1, 2, 3, 4, 6]]
        """
        each_not_edl_indices = [[i for i, edl in enumerate(b_edls) if edl != uni_edl] for uni_edl in sorted(torch.unique(b_edls))]
        # 異なるドメインラベルを持った要素のインデックスを取得.
        marginal_index = [random.choice(each_not_edl_indices[each_edl]) for each_edl in b_edls]  # ドメインラベルは連番であることを想定.

        class_feat = F.normalize(class_feat, dim=1)
        domain_feat = F.normalize(domain_feat, dim=1)
        marginal_domain_feats = domain_feat[marginal_index]

        Tj = self.mine_net(class_feat, domain_feat)
        Tj = F.normalize(Tj, dim=1)
        Tj = torch.mean(Tj)
        
        Tm = self.mine_net(class_feat, marginal_domain_feats)
        Tm = F.normalize(Tm, dim=1)
        expTm = torch.mean(torch.exp(Tm))
        
        self.ma_expTm = ((1-self.config.ma_rate) * self.ma_expTm + self.config.ma_rate * expTm.detach().item())  # Moving Average expTm

        ### Mutual Information
        mutual_information = (Tj - torch.log(expTm) * expTm.detach() / self.ma_expTm)
        mine_loss = -1.0 * mutual_information

        return mine_loss, mutual_information



    def _train_step(self, epoch):
        class_feats = []
        domain_feats = []
        indices = []

        for original_img, aug_img1, aug_img2, edl, index in self.train_loader:  # 必ずdomain_loaderは全て読み込まれないことになってしまうが        
            # MINEをのmarginalのサンプリングを適切に行うために, ある程度学習が進んでドメインラベル精度が出てからMINEを行う．
            # ドメインラベルが全て同じ値だったら, marginalサンプルが取れないので, 逆伝播はしない.
            if epoch <= self.config.mine_start_epoch or len(np.unique(edl)) != 0:
                original_img, aug_img1, aug_img2, edl = to_cuda([original_img, aug_img1, aug_img2, edl], self.config.device)
                indices += index

                self.class_encoder.train()
                self.domain_encoder.train()
                self.mine_net.train()

                class_loss, class_feat = self._simclr_step(original_img, aug_img2, self.class_encoder, self.c_nt_xent_criterion, 'class', epoch)  # 同じaug_img2の特徴量を得る.
                domain_loss, domain_feat = self._simclr_step(aug_img1, aug_img2, self.domain_encoder, self.d_nt_xent_criterion, 'domain', epoch, edl)  # 同じaug_img2の特徴量を得る.
                class_feats.append(class_feat.detach().cpu().numpy())
                domain_feats.append(domain_feat.detach().cpu().numpy())

                if epoch < self.config.mine_start_epoch:
                    loss = class_loss + domain_loss * self.config.domain_weight
                else:
                    mine_loss, mutual_information = self._get_mine_loss(class_feat, domain_feat, edl)  # このdomain_featとedlは1エポックずれてるんよな
                    loss = class_loss + domain_loss * self.config.domain_weight + mine_loss * self.config.mine_weight

                loss.backward()
                self.optimizer.step()
                self.trn_scheduler.step()

        class_feats = np.concatenate(class_feats)
        domain_feats = np.concatenate(domain_feats)

        ### Clustering
        _ = self._clustering_exec(class_feats, indices, 'class')  # クラスタの決め方は慣性的にした方が良いかもな
        domain_cluster = self._clustering_exec(domain_feats, indices, 'domain')
        
        self.train_loader.dataset.update_estimated_domains(indices, edls=domain_cluster, ed_feats=domain_feats)


        self.logger.info(f"\t Loss: {loss.item():.4f}")
        if epoch >= self.config.mine_start_epoch:
            self.logger.info(f"\t Class_Loss: {class_loss.item():.2f} \t Domain_Loss: {domain_loss.item():.2f} \t MINE_Loss: {mine_loss.item():.2f}")
            self.logger.info(f"\t Mutual_Information: {mutual_information}")
        else:
            self.logger.info(f"\t Class_Loss: {class_loss.item():.2f} \t Domain_Loss: {domain_loss.item():.2f}")
        self.logger.info(f"\t [Class]  nmi_domain: {self.c_nmis[0]:.2f},\t nmi_class: {self.c_nmis[1]:.2f},\t domain_acc: {self.c_nmis[2]:.2f}")
        self.logger.info(f"\t [Domain] nmi_domain: {self.d_nmis[0]:.2f},\t nmi_class: {self.d_nmis[1]:.2f},\t domain_acc: {self.d_nmis[2]:.2f}")
        self.logger.info(f"\t domain_weight: {self.config.domain_weight},\t mine_weight: {self.config.mine_weight:.2f},\t ma_expTm: {self.ma_expTm:.2f},\t ma_rate: {self.config.ma_rate}")



    def run_train(self):
        ####################################################################
        self.config.domain_weight = 1
        self.config.ma_rate = 0.5
        self.config.mine_weight = 1000
        self.ma_expTm = 1  # moving_average_expTm
        self.config.mine_start_epoch = 5
        ####################################################################

        for epoch in range(self.config.epochs):
            self.logger.info(f'Epoch: {epoch+1}/{self.config.epochs}')
            self._train_step(epoch)
            # total_acc, accuracy_list, mtx = eval_step(
            #     config, logger, writer, class_encoder, test_loader, epoch
            # )

            # save_models(config, [class_encoder, domain_encoder, mine_net], epoch, total_acc, accuracy_list, mtx)
