import torch
import torch.nn as nn
import torch.nn.functional as F


# class MineNet(nn.Module):
#     def __init__(self, input_dim=256, hidden_dim=128):
#         super().__init__()
        
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, 1)

#     def forward(self, X, Y):
#         output = F.elu(self.fc1(input))
#         output = F.elu(self.fc2(output))
#         output = self.fc3(output)
#         return output
        
class MineNet(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()

        self.mine_layer = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.BatchNorm1d(1),  ################# これ付けてた方がよさそう? 後でF.normalizeするのと変わらないのだろうか？　#################
            nn.ReLU(),  ################# これ付けてた方がよさそう?　ReLUつけてた方がTotal_Accuracyが安定した（たまたまかもだが）　#################
        )
    
    def forward(self, X, Y):
        Z = torch.cat((X,Y),1)  # Size(batch_size, 2 * out_dim)
        Z = self.mine_layer(Z)

        return Z

        