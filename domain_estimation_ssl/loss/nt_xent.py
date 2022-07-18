import torch
import torch.nn as nn
import numpy as np


class NTXentLoss(nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super().__init__()
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


    def _get_pos_neg_masks(self, positive=True):
        """ 行列の要素(xis, xis), (xis, xjs), (xjs, xjs) が0,それ以外が1の行列をつくり,類似度行列のフィルタを作る."""
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        if positive:
            positive_masks = torch.from_numpy(l1 + l2).type(torch.bool).to(self.device)
            return positive_masks
        else:
            negative_masks = (1 - torch.from_numpy(diag + l1 + l2)).type(torch.bool).to(self.device)
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
        

    def forward(self, zis, zjs):
        """ 最初にバッチ内の全ての組合せの類似度行列を作り,正例か負例かでフィルタケ掛けてInfoNCEを計算する. """
        representations = torch.cat([zjs, zis], dim=0)  # 縦に結合.([2*batch_size, channel])
        similarity_matrix = self.similarity_function(representations, representations)  # 類似度行列の作成. ([2*batch_size, 2*batch_size])

        # filter out the scores from the positive/negative pairs
        # 類似度行列からpositive/negativeの要素を取り出す．それぞれ2個ずつ重複しているのがマジで気持ち悪い．
        positives = similarity_matrix[self._get_pos_neg_masks(positive=True)].view(2 * self.batch_size, -1)  # ([2*batch_size, 1])
        negatives = similarity_matrix[self._get_pos_neg_masks(positive=False)].view(2 * self.batch_size, -1)  # ([2*batch_size, 2*batch_size - 2])  -> 同じ画像由来の要素は含まない為

        logits = torch.cat((positives, negatives), dim=1)  # 横に結合.([2*batch_size, 2*batch_size - 1])
        logits /= self.temperature  

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)  # logitsは重みでもある.重みが大きい(類似度が大きい)　-> lossは小さくなる

        return loss / (2 * self.batch_size)  # 平均を取る
