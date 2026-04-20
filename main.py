import torch
import numpy as np
from lgcn.lgcnModel import LGCN


def _distance_matrix(points):
    return np.sqrt(((points[:, None] - points[None, :]) ** 2).sum(axis=2))


def optimize_route(points):
    """
    Given a list of (lat, lon) points for a single driver,
    returns the same points reordered as an optimal route using LGCN.

    Args:
        points (list): List of [lat, lon] coordinate pairs.

    Returns:
        list: Reordered list of [lat, lon] pairs representing the optimal route.
    """
    points = np.array(points, dtype=np.float64)
    n = len(points)

    if n <= 1:
        return points.tolist()

    lgcn = LGCN(
        node_input_dim=2,
        node_caps_dim=8,
        subroute_caps_dim=4,
        num_subroutes=1,
    )
    lgcn.eval()

    node_features = torch.tensor(points, dtype=torch.float32)

    with torch.no_grad():
        node_caps = lgcn.node_fc(node_features)
        node_caps = node_caps / (torch.norm(node_caps, dim=-1, keepdim=True) + 1e-8)
        subroute_caps = lgcn.routing(node_caps)  # [1, subroute_caps_dim]

        proj = torch.nn.Linear(node_caps.size(1), subroute_caps.size(1), bias=False)
        u_hat = proj(node_caps)  # [N, subroute_caps_dim]
        sim = torch.matmul(u_hat, subroute_caps.T).squeeze(1)  # [N]

    dist_mat = _distance_matrix(points)

    # Greedy route: start from node 0, pick next by capsule score - distance penalty
    visited = [False] * n
    route_indices = [0]
    visited[0] = True

    for _ in range(n - 1):
        last = route_indices[-1]
        best_score = -1e9
        best_next = None

        for j in range(n):
            if visited[j]:
                continue
            score = sim[j].item() - 0.01 * dist_mat[last, j]
            if score > best_score:
                best_score = score
                best_next = j

        route_indices.append(best_next)
        visited[best_next] = True

    return points[route_indices].tolist()


if __name__ == "__main__":
    # Example usage
    sample_points = [
        [32.0853, 34.7818],
        [32.0860, 34.7800],
        [32.0820, 34.7900],
        [32.0900, 34.7700],
        [32.0880, 34.7850],
        [32.0830, 34.7750],
        [32.0870, 34.7820],
    ]

    optimal_route = optimize_route(sample_points)
    print("Optimal route:")
    for i, point in enumerate(optimal_route):
        print(f"  Stop {i + 1}: {point}")
