import torch.nn as nn
from lgcn.capsule import squash
from lgcn.routing import dynamic_routing

class RoutingLayer(nn.Module):
    """
    Routing Layer for Capsule Networks.

    This layer transforms node capsules into subroute capsules by applying a linear transformation
    and aggregating across nodes to form capsules for each subroute.
    """
    def __init__(self, node_caps_dim, num_subroutes, subroute_caps_dim):
        """
        Initialize the RoutingLayer.

        Args:
            node_caps_dim (int): Dimensionality of input node capsules.
            num_subroutes (int): Number of subroutes (e.g., number of drivers).
            subroute_caps_dim (int): Dimensionality of output subroute capsules.
        """
        super().__init__()
        self.num_subroutes = num_subroutes
        self.subroute_caps_dim = subroute_caps_dim
        self.transform = nn.Linear(node_caps_dim, num_subroutes * subroute_caps_dim)

    def forward(self, x):
        """
        Forward pass of the RoutingLayer.

        Args:
            x (torch.Tensor): Input node capsules of shape (batch_size, node_caps_dim).

        Returns:
            torch.Tensor: Subroute capsules of shape (num_subroutes, subroute_caps_dim).
        """
        u_hat = self.transform(x)
        u_hat = u_hat.view(x.size(0), self.num_subroutes, self.subroute_caps_dim)
        subroute_caps = dynamic_routing(u_hat)
        return subroute_caps

class LGCN(nn.Module):
    """
    LGCN (Location-based Graph Capsule Network) Model.

    This model processes node features to generate subroute capsules for routing problems,
    such as vehicle routing with multiple drivers.
    """
    def __init__(self, node_input_dim, node_caps_dim, subroute_caps_dim, num_subroutes, full_route_caps_dim):
        """
        Initialize the LGCN model.

        Args:
            node_input_dim (int): Dimensionality of input node features (e.g., coordinates).
            node_caps_dim (int): Dimensionality of node capsules after first linear layer.
            subroute_caps_dim (int): Dimensionality of subroute capsules.
            num_subroutes (int): Number of subroutes (e.g., number of drivers).
            full_route_caps_dim (int): Dimensionality of the full route capsule.
        """
        super().__init__()
        self.node_fc = nn.Linear(node_input_dim, node_caps_dim)
        self.routing = RoutingLayer(node_caps_dim, num_subroutes, subroute_caps_dim)
        self.full_route_routing = RoutingLayer(subroute_caps_dim, 1, full_route_caps_dim)

    def forward(self, node_features):
        """
        Forward pass of the LGCN model.

        Args:
            node_features (torch.Tensor): Input node features of shape (num_nodes, node_input_dim).

        Returns:
            tuple: (subroute_caps, full_route_caps)
                - subroute_caps: shape (num_subroutes, subroute_caps_dim)
                - full_route_caps: shape (1, full_route_caps_dim)
        """
        node_caps = self.node_fc(node_features)
        node_caps = squash(node_caps)
        subroute_caps = self.routing(node_caps)
        full_route_caps = self.full_route_routing(subroute_caps)
        return subroute_caps, full_route_caps
