import torch
from lgcn.capsule import squash

def dynamic_routing(u_hat, iters=3):
    b = torch.zeros(u_hat.size(0), u_hat.size(1))
    for _ in range(iters):
        c = torch.softmax(b, dim=1)
        s = (c.unsqueeze(-1) * u_hat).sum(dim=0)
        v = squash(s)
        b = b + (u_hat * v).sum(dim=-1)
    return v
