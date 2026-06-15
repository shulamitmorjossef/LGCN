import os
import json
import torch
import numpy as np
from lgcn.lgcnModel import LGCN
from lgcn.capsule import squash
from graph.dynamicGraph import DynamicGraph
from graph.node import Node
from graph.edge import Edge

# ── Singleton model (weights persisted to disk) ───────────────────────────────

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "lgcn_weights.pt")
_NODE_CAPS_DIM = 8
_SUBROUTE_CAPS_DIM = 4
_FULL_ROUTE_CAPS_DIM = 4

def _load_or_init_model():
    lgcn = LGCN(
        node_input_dim=5,
        node_caps_dim=_NODE_CAPS_DIM,
        subroute_caps_dim=_SUBROUTE_CAPS_DIM,
        num_subroutes=1,
        full_route_caps_dim=_FULL_ROUTE_CAPS_DIM,
    )
    proj = torch.nn.Linear(_NODE_CAPS_DIM, _FULL_ROUTE_CAPS_DIM, bias=False)
    if os.path.exists(_MODEL_PATH):
        checkpoint = torch.load(_MODEL_PATH, weights_only=True)
        lgcn.load_state_dict(checkpoint["lgcn"])
        proj.load_state_dict(checkpoint["proj"])
    else:
        torch.save({"lgcn": lgcn.state_dict(), "proj": proj.state_dict()}, _MODEL_PATH)
    lgcn.eval()
    proj.eval()
    return lgcn, proj

_lgcn, _proj = _load_or_init_model()

# ── Per-driver graph state (persisted to disk as JSON) ────────────────────────

DRIVER_STATES_DIR = os.path.join(os.path.dirname(__file__), "driver_states")


def _driver_state_path(driver_id: str) -> str:
    return os.path.join(DRIVER_STATES_DIR, f"{driver_id}.json")


def load_driver_graph(driver_id: str) -> DynamicGraph:
    graph = DynamicGraph()
    path = _driver_state_path(driver_id)
    if not os.path.exists(path):
        return graph
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for n in data["nodes"]:
        node = Node(
            node_id=n["id"],
            x=n["x"],
            y=n["y"],
            demand=n.get("demand", 0),
            time_windows=[(tw[0], tw[1]) for tw in n.get("time_windows", [])],
            wait=n.get("wait", 0),
        )
        # Insert directly to avoid re-generating all edges on every add
        graph.nodes[node.id] = node
    # Rebuild edges from stored data
    for key_str, e in data.get("edges", {}).items():
        i_id, j_id = key_str.split(",")
        ni, nj = graph.nodes[int(i_id)], graph.nodes[int(j_id)]
        edge = Edge(ni, nj, traffic=e.get("traffic", 1.0), urgency=e.get("urgency", 1.0))
        graph.edges[(int(i_id), int(j_id))] = edge
    return graph


def save_driver_graph(driver_id: str, graph: DynamicGraph) -> None:
    os.makedirs(DRIVER_STATES_DIR, exist_ok=True)
    data = {
        "nodes": [
            {
                "id": n.id,
                "x": n.x,
                "y": n.y,
                "demand": n.demand,
                "time_windows": list(n.time_windows),
                "wait": n.wait,
            }
            for n in graph.nodes.values()
        ],
        "edges": {
            f"{k[0]},{k[1]}": {"traffic": e.traffic, "urgency": e.urgency}
            for k, e in graph.edges.items()
        },
    }
    with open(_driver_state_path(driver_id), "w", encoding="utf-8") as f:
        json.dump(data, f)


def reset_driver(driver_id: str) -> None:
    path = _driver_state_path(driver_id)
    if os.path.exists(path):
        os.remove(path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _distance_matrix(points):
    return np.sqrt(((points[:, None] - points[None, :]) ** 2).sum(axis=2))


def _node_to_model_input(node: Node):
    """Returns the 5 features the model expects: [x, y, tw_start, tw_end, wait]."""
    tw_start = node.time_windows[0][0] if node.time_windows else 0.0
    tw_end   = node.time_windows[0][1] if node.time_windows else 0.0
    return [node.x, node.y, tw_start, tw_end, node.wait]


# ── Main routing function ─────────────────────────────────────────────────────

def optimize_route(driver_id: str, new_points, traffic_delays=None, priorities=None):
    """
    Incrementally update a driver's graph and return the optimal route over
    all their current stops.

    Args:
        driver_id:      Unique identifier for the driver.
        new_points:     List of NEW stops as [x, y, tw_start, tw_end, wait].
                        Pass all stops on the first call; only new ones on updates.
        traffic_delays: Optional N×N matrix (for the NEW points only, relative
                        to each other). Not stored on existing edges.
        priorities:     Optional priority per NEW stop.

    Returns:
        list: Ordered node-id values representing the optimal route over ALL
              current stops for this driver.
    """
    # Load existing graph state for this driver
    graph = load_driver_graph(driver_id)
    existing_ids = list(graph.nodes.keys())

    # Add new nodes (incremental graph update per LGCN paper §II-A)
    next_id = max(existing_ids, default=-1) + 1
    new_node_ids = []
    for i, pt in enumerate(new_points):
        pt = list(pt)
        tw = [(pt[2], pt[3])] if len(pt) >= 4 else []
        wait = pt[4] if len(pt) >= 5 else 0
        node = Node(node_id=next_id + i, x=pt[0], y=pt[1], time_windows=tw, wait=wait)
        graph.add_node(node)
        new_node_ids.append(node.id)

    # Persist updated graph
    save_driver_graph(driver_id, graph)

    all_nodes = list(graph.nodes.values())
    n = len(all_nodes)

    if n <= 1:
        return [node.id for node in all_nodes]

    # Build model input features for all current nodes
    points_arr = np.array([_node_to_model_input(node) for node in all_nodes], dtype=np.float32)
    node_features = torch.tensor(points_arr, dtype=torch.float32)

    # LGCN inference: capsule similarity scores per node
    with torch.no_grad():
        node_caps = _lgcn.node_fc(node_features)
        node_caps = squash(node_caps)
        subroute_caps = _lgcn.routing(node_caps)
        full_route_caps = _lgcn.full_route_routing(subroute_caps)
        u_hat = _proj(node_caps)
        sim = torch.matmul(u_hat, full_route_caps.T).squeeze(1)

    dist_mat = _distance_matrix(points_arr[:, :2])

    # Build traffic/priority arrays aligned to current node order
    if priorities is None:
        pri = np.ones(n, dtype=np.float64)
    else:
        # priorities apply only to new nodes; existing nodes default to 1
        pri = np.ones(n, dtype=np.float64)
        for local_i, nid in enumerate(new_node_ids):
            pos = next((idx for idx, nd in enumerate(all_nodes) if nd.id == nid), None)
            if pos is not None and local_i < len(priorities):
                pri[pos] = priorities[local_i]

    delay_mat = np.zeros((n, n), dtype=np.float64)
    if traffic_delays is not None:
        td = np.array(traffic_delays, dtype=np.float64)
        offset = len(existing_ids)
        size = min(td.shape[0], n - offset)
        delay_mat[offset:offset + size, offset:offset + size] = td[:size, :size]

    # Greedy route construction (score = capsule similarity − edge cost)
    visited = [False] * n
    route = [0]
    visited[0] = True

    for _ in range(n - 1):
        last = route[-1]
        best_score = -1e9
        best_next = None
        for j in range(n):
            if visited[j]:
                continue
            edge_cost = dist_mat[last, j] * (delay_mat[last, j] + pri[j])
            score = sim[j].item() - edge_cost
            if score > best_score:
                best_score = score
                best_next = j
        route.append(best_next)
        visited[best_next] = True

    # Return actual node IDs (not positional indices)
    return [all_nodes[i].id for i in route]


if __name__ == "__main__":
    # Simulate two calls for the same driver (incremental update)
    initial_stops = [
        [32.0853, 34.7818, 8.0, 12.0, 10],
        [32.0860, 34.7800, 9.0, 13.0, 5],
        [32.0820, 34.7900, 7.0, 11.0, 15],
    ]
    print("First call (initial stops):")
    route1 = optimize_route("driver_42", initial_stops)
    print("Route node IDs:", route1)

    new_stop = [[32.0900, 34.7700, 10.0, 14.0, 0]]
    print("\nSecond call (one new stop added):")
    route2 = optimize_route("driver_42", new_stop)
    print("Route node IDs:", route2)

    # Clean up test state
    reset_driver("driver_42")
