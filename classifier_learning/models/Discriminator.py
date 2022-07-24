import torch
import torch.nn as nn

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd, reverse=True):
        ctx.lambd = lambd
        ctx.reverse=reverse
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reverse:
            return (grad_output * -ctx.lambd), None, None
        else:
            return (grad_output * ctx.lambd), None, None

def grad_reverse(x, lambd=1.0, reverse=True):
    return GradReverse.apply(x, lambd, reverse)


class Discriminator(nn.Module):
    def __init__(self, dims, grl=True, reverse=True):
        if len(dims) != 4:
            raise ValueError("Discriminator input dims should be three dim!")
        super().__init__()
        self.grl = grl
        self.reverse = reverse
        self.discriminate_layer = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dims[2], dims[3]),
        )

    def forward(self, x, constant):
        if self.grl:
            x = grad_reverse(x, constant, self.reverse)
        x = self.discriminate_layer(x)
        return x