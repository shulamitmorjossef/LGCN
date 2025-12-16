import torch
from lgcn.capsule import CapsuleLayer
from lgcn.routing import dynamic_routing

class LGCN:
    def __init__(self, feature_dim, capsule_dim):
        self.capsule = CapsuleLayer(feature_dim, capsule_dim)

    def infer(self, node_features):
        x = torch.tensor(node_features, dtype=torch.float32)
        caps = self.capsule(x)
        caps = caps.unsqueeze(1)
        route_embedding = dynamic_routing(caps)
        return route_embedding
