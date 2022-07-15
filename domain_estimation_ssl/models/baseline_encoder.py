import torch
import torch.nn as nn
from models.grl import ReverseLayerF


class Encoder(nn.Module):
    def __init__(self, ssl='simclr', input_dim=1, out_dim=64, pred_dim=128):
        super().__init__()
        self.ssl = ssl
        self.input_dim = input_dim
        self.out_dim = out_dim
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
            nn.Linear(64, out_dim),
        )

        # Predictor MLP -> p
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, out_dim) # output layer
        )


    def forward(self, x):
        x = self.encorder(x)
        h = torch.mean(x, dim=[2, 3])  # embedding
        z = self.projector(h)

        if self.ssl == 'simclr':
            return h, z
        elif self.ssl == 'simsiam':
            p = self.predictor(z)
            return h, z, p
