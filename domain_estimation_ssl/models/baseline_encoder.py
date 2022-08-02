import torch
import torch.nn as nn
from models.grl import ReverseLayerF


class Encoder(nn.Module):
    def __init__(self, config, input_dim=1, pred_dim=128):
        super().__init__()
        self.ssl = config.model.ssl
        self.input_dim = input_dim
        self.out_dim = config.model.out_dim
        self.pred_dim = pred_dim

        # encorder f
        self.encorder = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # projection MLP -> z
        self.projector = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, self.out_dim),
        )

        # Predictor MLP -> p
        self.predictor = nn.Sequential(
            nn.Linear(self.out_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, self.out_dim) # output layer
        )

        self.sigmoid_pseudo = nn.Sequential(
            nn.Linear(64, self.out_dim),
            nn.Sigmoid(),
        )


    def forward(self, x):
        x = self.encorder(x)
        h = torch.mean(x, dim=[2, 3])  # embedding

        if self.ssl == 'simclr':
            z = self.projector(h)
            return h, z
        elif self.ssl == 'simsiam':
            z = self.projector(h)
            p = self.predictor(z)
            return h, z, p
        elif self.ssl == 'random_pseudo':
            sigmoid_pseudo = self.sigmoid_pseudo(h)
            return h, sigmoid_pseudo
        else:
            print("ssl設定ミス")
            raise ValueError("ssl設定ミス")