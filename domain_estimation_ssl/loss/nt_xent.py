import torch
import torch.nn as nn
import numpy as np
import itertools
import gc

class NTXentLoss(nn.Module):
    def __init__(self, config, device, batch_size, temperature, use_cosine_similarity):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        # self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        if use_cosine_similarity:
            self.similarity_function = self._cosine_simililarity
        else:
            self.similarity_function = self._dot_simililarity
        self.criterion = nn.CrossEntropyLoss(reduction="sum")


    def _get_pos_neg_masks(self, edls, positive=True):
        """ 行列の要素(xis, xis), (xis, xjs), (xjs, xjs) が0,それ以外が1の行列をつくり,類似度行列のフィルタを作る."""
        # if len(edls) == 0:
        pos_pairs = torch.from_numpy(np.eye(2 * self.batch_size, 2 * self.batch_size, k=self.batch_size))
        tril = torch.tril(torch.ones(2 * self.batch_size, 2 * self.batch_size))

        if positive:
            positive_masks = pos_pairs.type(torch.bool).to(self.device)
            return positive_masks
        else:
            negative_masks = (1 - tril - pos_pairs).type(torch.bool).to(self.device)
            return negative_masks


    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v


    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        cosine_similarity = nn.CosineSimilarity(dim=-1)
        v = cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))  # それぞれ1次元に直して類似度計算
        return v


    def forward(self, zis, zjs, edls):
        """ 最初にバッチ内の全ての組合せの類似度行列を作り,正例か負例かでフィルタケ掛けてInfoNCEを計算する. """
        representations = torch.cat([zjs, zis], dim=0)  # 縦に結合
        similarity_matrix = self.similarity_function(representations, representations)  # 類似度行列の作成
        
        # # 類似度行列からpositive/negativeの要素を取り出す.
        positives = similarity_matrix[self._get_pos_neg_masks(edls, positive=True)].view(self.batch_size, -1)  # 1次元化したものをviewで

        if self.config.lap <= 1:
            negatives = similarity_matrix[self._get_pos_neg_masks(edls, positive=False)].view(self.batch_size, -1)
        else:
            # edlsに対応する負例の影響度をさげておく
            similarity_matrix_tmp = similarity_matrix
            edl_indexes = [[i for i, x in enumerate(edls) if x == d] for d in np.unique(edls)][0]
            for edl_index in edl_indexes:
                similarity_matrix_tmp[edl_index] *= 0    # 同推定ドメインの負例に定数掛けて影響度を抑える.
            negatives = similarity_matrix_tmp[self._get_pos_neg_masks(edls, positive=False)].view(self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)  # 横に結合.
        logits /= self.temperature
        labels = torch.zeros(self.batch_size).to(self.device).long()  # 1次元
        loss = self.criterion(logits, labels)  # logitsは重みでもある.重みが大きい(類似度が大きい)　-> lossは小さくなる

        return loss / self.batch_size
