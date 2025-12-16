import torch
import torch.nn as nn

class RoutingLayer(nn.Module):
    def __init__(self, node_caps_dim, num_subroutes, subroute_caps_dim):
        super().__init__()
        self.num_subroutes = num_subroutes
        self.subroute_caps_dim = subroute_caps_dim
        self.transform = nn.Linear(node_caps_dim, num_subroutes * subroute_caps_dim)

    def forward(self, x):
        out = self.transform(x)
        out = out.view(x.size(0), self.num_subroutes, self.subroute_caps_dim)
        out = out / (torch.norm(out, dim=-1, keepdim=True) + 1e-8)
        subroute_caps = out.sum(dim=0)
        return subroute_caps

class LGCN(nn.Module):
    def __init__(self, node_input_dim, node_caps_dim, subroute_caps_dim, num_subroutes):
        super().__init__()
        self.node_fc = nn.Linear(node_input_dim, node_caps_dim)
        self.routing = RoutingLayer(node_caps_dim, num_subroutes, subroute_caps_dim)

    def forward(self, node_features):
        node_caps = self.node_fc(node_features)
        node_caps = node_caps / (torch.norm(node_caps, dim=-1, keepdim=True) + 1e-8)
        subroute_caps = self.routing(node_caps)
        return subroute_caps
