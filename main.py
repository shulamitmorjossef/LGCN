import torch
import numpy as np
from lgcn.lgcnModel import LGCN
from lgcn.capsule import squash


def _distance_matrix(points):
    return np.sqrt(((points[:, None] - points[None, :]) ** 2).sum(axis=2))


def optimize_route(points, traffic_delays=None, priorities=None):
    """
    Given a list of points for a single driver, returns the point indices
    reordered as an optimal route using LGCN.

    Args:
        points (list): List of [lat, lon, tw_start, tw_end, wait_time] per stop.
        traffic_delays (list, optional): N×N matrix of travel-time delays between stops.
                                         Defaults to zeros (no delay).
        priorities (list, optional): Priority value per stop. Defaults to 1 for all stops.

    Returns:
        list: Ordered indices representing the optimal route.
    """
    points = np.array(points, dtype=np.float64)
    n = len(points)

    if n <= 1:
        return list(range(n))

    lgcn = LGCN(
        node_input_dim=5,
        node_caps_dim=8,
        subroute_caps_dim=4,
        num_subroutes=1,
        full_route_caps_dim=4,
    )
    lgcn.eval()

    node_features = torch.tensor(points, dtype=torch.float32)

    with torch.no_grad():
        node_caps = lgcn.node_fc(node_features)
        node_caps = squash(node_caps)
        subroute_caps = lgcn.routing(node_caps)          # [num_subroutes, subroute_caps_dim]
        full_route_caps = lgcn.full_route_routing(subroute_caps)  # [1, full_route_caps_dim]

        proj = torch.nn.Linear(node_caps.size(1), full_route_caps.size(1), bias=False)
        u_hat = proj(node_caps)  # [N, full_route_caps_dim]
        sim = torch.matmul(u_hat, full_route_caps.T).squeeze(1)  # [N]

    dist_mat = _distance_matrix(points[:, :2])  # distance based on lat/lon only

    if traffic_delays is None:
        traffic_delays = np.zeros((n, n), dtype=np.float64)
    else:
        traffic_delays = np.array(traffic_delays, dtype=np.float64)

    if priorities is None:
        priorities = np.ones(n, dtype=np.float64)
    else:
        priorities = np.array(priorities, dtype=np.float64)

    # Greedy route: start from node 0, pick next by capsule score - edge cost
    # edge_cost = dist * (traffic_delay + priority)  per the LGCN paper formula
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
            edge_cost = dist_mat[last, j] * (traffic_delays[last, j] + priorities[j])
            score = sim[j].item() - edge_cost
            if score > best_score:
                best_score = score
                best_next = j

        route_indices.append(best_next)
        visited[best_next] = True

    return route_indices


if __name__ == "__main__":
    # Example usage: [lat, lon, tw_start, tw_end, wait_time]
    sample_points = [
        [32.0853, 34.7818, 8.0, 12.0, 10],
        [32.0860, 34.7800, 9.0, 13.0, 5],
        [32.0820, 34.7900, 7.0, 11.0, 15],
        [32.0900, 34.7700, 10.0, 14.0, 0],
        [32.0880, 34.7850, 8.5, 12.5, 20],
        [32.0830, 34.7750, 9.5, 13.5, 10],
        [32.0870, 34.7820, 11.0, 15.0, 5],
    ]

    optimal_indices = optimize_route(sample_points)
    print("Optimal route indices:", optimal_indices)
    for rank, idx in enumerate(optimal_indices):
        print(f"  Stop {rank + 1}: {sample_points[idx]}")
