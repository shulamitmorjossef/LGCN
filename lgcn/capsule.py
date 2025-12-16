import torch
import torch.nn as nn

def squash(v):
    norm = torch.norm(v, dim=-1, keepdim=True)
    return (norm**2 / (1 + norm**2)) * (v / (norm + 1e-8))

class CapsuleLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        u = self.W(x)
        return squash(u)
