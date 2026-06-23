import os
import json
import torch
import numpy as np
from typing import Optional
from lgcn.lgcnModel import LGCN
from lgcn.capsule import squash
from graph.dynamicGraph import DynamicGraph
from graph.node import Node
from graph.edge import Edge

# ─── Fixed matrices ───────────────────────────────────────────────────────────
# features: [lat, lon, tw_start, tw_end, wait_time]
# Scales: lat~32, lon~35, tw~8-18 (hours), wait~0-20 (minutes)

# W1 [8×5]: features → node capsule
# Each row produces one dimension in the capsule.
# - Rows 0-1: normalize lat/lon to range ~[0,1] (×0.03)
# - Row 2:  time window center = (tw_s+tw_e)/2, scale ×0.06 → value ~0.6-1.0
# - Row 3:  window width = tw_s - tw_e (always negative).
#            narrow window (2h) → -0.3  /  wide window (10h) → -1.5
#            i.e.: urgent = close to 0, flexible = large-negative
# - Row 4:  -tw_end × 0.08: early deadline (11) → -0.88, late (18) → -1.44
#            i.e.: early deadline = less negative value = higher
# - Row 5:  tw_start × 0.10: earliest allowed arrival time
# - Row 6:  -wait × 0.05: wait=0→0, wait=20→-1.0  (less waiting = higher value)
# - Row 7:  combined urgency score: small lat+lon + narrow window + early deadline + low wait
_W1 = np.array([
    [ 0.03,  0.00,  0.00,  0.00,  0.00],
    [ 0.00,  0.03,  0.00,  0.00,  0.00],
    [ 0.00,  0.00,  0.06,  0.06,  0.00],
    [ 0.00,  0.00,  0.15, -0.15,  0.00],
    [ 0.00,  0.00,  0.00, -0.08,  0.00],
    [ 0.00,  0.00,  0.10,  0.00,  0.00],
    [ 0.00,  0.00,  0.00,  0.00, -0.05],
    [ 0.01,  0.01,  0.08, -0.08, -0.04],
], dtype=np.float32)

# W2 [4×8]: node capsule → prediction vector for sub-route (dynamic routing)
# Stops with similar dimensions vote in the same direction → they get grouped together.
# - Row 0: geographic cluster — high weight on dim0(lat) and dim1(lon)
#           geographically close stops = close votes = enter the same sub-route
# - Row 1: time cluster — window center (dim2) + deadline (dim4)
#           stops with similar time window converge together
# - Row 2: urgency cluster — window width (dim3) + deadline (dim4) + wait (dim6)
#           urgent stops = similar votes = enter the "urgent" sub-route
# - Row 3: overall efficiency — combination of all dimensions, special weight for dim7 (combined score)
_W2 = np.array([
    [ 0.90,  0.90,  0.00,  0.00,  0.00,  0.00,  0.00,  0.10],
    [ 0.00,  0.00,  0.85,  0.00,  0.85,  0.00,  0.00,  0.10],
    [ 0.00,  0.00,  0.00,  0.70,  0.60,  0.00,  0.70,  0.20],
    [ 0.25,  0.25,  0.25,  0.10,  0.10,  0.25,  0.20,  0.40],
], dtype=np.float32)

# W3 [4×4]: subroute capsule → prediction vector for full route
# With num_subroutes=1 this is a single pass (no consensus). The matrix performs:
# - Rows 0-1: light blending between geographic and time dimensions (10% cross-mix)
#   → the final capsule is aware of both dimensions, not just one
# - Rows 2-3: same idea for urgency and efficiency dimensions
_W3 = np.array([
    [ 0.90,  0.10,  0.00,  0.00],
    [ 0.10,  0.90,  0.00,  0.00],
    [ 0.00,  0.00,  0.85,  0.15],
    [ 0.00,  0.00,  0.15,  0.85],
], dtype=np.float32)

# W4 [4×8]: node capsule → projection to full_route_caps space (for sim calculation)
# sim[j] = dot(W4 × c_j, full_route_caps) — score of how much stop j "belongs" to the route
#
# Logic: dim3/dim4/dim6 of the capsule are negative.
# An urgent stop = negative values close to 0.
# W4 with positive weights on dim3,dim4,dim6 →
#   urgent: W4×(-0.3) = -0.24  /  flexible: W4×(-1.5) = -1.2
#   full_route_caps[2] (urgency dimension) is also negative →
#   dot: (-0.24)×(negative) > (-1.2)×(negative)  → urgent stop receives higher sim ✓
#
# - Row 0: geographic dimension → maps lat/lon to same dimension as W2/W3
# - Row 1: time dimension → window center + deadline
# - Row 2: urgency dimension → window width + deadline + wait (most important for scoring)
# - Row 3: combined dimension → weighted combination of everything
_W4 = np.array([
    [ 0.80,  0.80,  0.00,  0.00,  0.00,  0.00,  0.00,  0.20],
    [ 0.00,  0.00,  0.80,  0.00,  0.70,  0.50,  0.00,  0.20],
    [ 0.00,  0.00,  0.00,  0.70,  0.60,  0.00,  0.70,  0.30],
    [ 0.20,  0.20,  0.20,  0.30,  0.30,  0.20,  0.30,  0.50],
], dtype=np.float32)
# ──────────────────────────────────────────────────────────────────────────────

# ── Singleton model (weights persisted to disk) ───────────────────────────────

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
    with torch.no_grad():
        lgcn.node_fc.weight.copy_(torch.tensor(_W1, dtype=torch.float32))
        lgcn.node_fc.bias.zero_()
        lgcn.routing.transform.weight.copy_(torch.tensor(_W2, dtype=torch.float32))
        lgcn.routing.transform.bias.zero_()
        lgcn.full_route_routing.transform.weight.copy_(torch.tensor(_W3, dtype=torch.float32))
        lgcn.full_route_routing.transform.bias.zero_()
        proj.weight.copy_(torch.tensor(_W4, dtype=torch.float32))
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


def remove_driver_stop(driver_id: str, x: float, y: float) -> bool:
    graph = load_driver_graph(driver_id)
    node = _find_matching_node(graph, x, y)
    if node is None:
        return False
    graph.remove_node(node.id)
    save_driver_graph(driver_id, graph)
    return True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _distance_matrix(points):
    return np.sqrt(((points[:, None] - points[None, :]) ** 2).sum(axis=2))


def _find_matching_node(graph: DynamicGraph, x: float, y: float, tol: float = 1e-6) -> Optional[Node]:
    """Return an existing node at the same location (x, y), if any.

    Two points are considered "the same point" if their coordinates match
    within `tol`. Used to avoid caching duplicate stops for a driver.
    """
    for node in graph.nodes.values():
        if abs(node.x - x) <= tol and abs(node.y - y) <= tol:
            return node
    return None


def _node_to_model_input(node: Node):
    """Returns the 5 features the model expects: [x, y, tw_start, tw_end, wait]."""
    tw_start = node.time_windows[0][0] if node.time_windows else 0.0
    tw_end   = node.time_windows[0][1] if node.time_windows else 0.0
    return [node.x, node.y, tw_start, tw_end, node.wait]


def _point_to_node(node_id, pt) -> Node:
    pt = list(pt)
    tw = [(pt[2], pt[3])] if len(pt) >= 4 else []
    wait = pt[4] if len(pt) >= 5 else 0
    return Node(node_id=node_id, x=pt[0], y=pt[1], time_windows=tw, wait=wait)


# Sentinel ids for the driver's current location / mandatory end point: they
# are never written to the driver's cache (real cached stop ids are always
# >= 0), so they can't collide with a real stop id.
_CURRENT_LOCATION_ID = -1
_END_LOCATION_ID = -2

# How strongly the route construction "leans" toward the mandatory end point
# while still choosing intermediate stops, so the path doesn't strand a far
# stop for last. Ramps from 0.5x to 1x of this value as the route progresses
# (see optimize_route). Tune if routes feel too end-biased / not enough.
_END_PULL_BASE_WEIGHT = 0.3


# ── Main routing function ─────────────────────────────────────────────────────

def optimize_route(driver_id: str, current_location, stops, end_point=None,
                    traffic_delays=None, priorities=None):
    """
    Incrementally update a driver's graph and return the optimal route over
    all their current stops.

    `current_location` and `end_point` are structural endpoints, not stops:
    they are never written to the driver's cache (GPS readings jitter by a
    few meters between requests, so caching them would create spurious
    duplicate stops on every refresh). They're rebuilt fresh on every call
    from whatever you pass in.

    `stops` are the actual stops to visit — each one is checked against the
    driver's cached stops by coordinates: if it already exists it is NOT
    added again (no duplicate node), but it is still taken into account when
    computing the route. Only genuinely new stops are added to the cache.

    Args:
        driver_id:        Unique identifier for the driver.
        current_location: [x, y, tw_start, tw_end, wait] — the driver's
                           position right now. Always first in the route.
        stops:             List of stops as [x, y, tw_start, tw_end, wait].
                           Pass all stops on the first call; on later calls
                           you may pass the full current set again — stops
                           that already exist for this driver are recognized
                           and skipped, only genuinely new ones get cached.
        end_point:         Optional [x, y, tw_start, tw_end, wait] — a
                           mandatory final destination. If given, it is
                           always last in the route, regardless of its
                           capsule-similarity score; intermediate stops are
                           still chosen with a "pull" toward it so the route
                           doesn't strand a far stop for the very last leg.
                           If omitted, the route ends wherever the greedy
                           construction naturally finishes (today's
                           behavior).
        traffic_delays:    Optional N×N matrix for `stops` only, relative to
                           each other, in that order.
        priorities:        Optional priority per stop in `stops` (same
                           order/length as `stops`).

    Returns:
        list: Ordered node-id values representing the optimal route:
              current_location (id -1) → cached stops → end_point (id -2,
              if given). The endpoints use negative sentinel ids since they
              aren't cached nodes.
    """
    # Load existing graph state for this driver (cached STOPS only — the
    # current location / end point are never persisted here).
    graph = load_driver_graph(driver_id)
    existing_ids = list(graph.nodes.keys())
    next_id = max(existing_ids, default=-1) + 1

    # For each incoming stop, resolve it to a node id: reuse the existing
    # node if this stop is already cached for the driver, otherwise create
    # and cache a new node. `request_node_ids` keeps the same order/length
    # as `stops` so traffic_delays/priorities can be mapped back.
    request_node_ids = []
    graph_changed = False
    for pt in stops:
        match = _find_matching_node(graph, pt[0], pt[1])
        if match is not None:
            request_node_ids.append(match.id)
            continue
        node = _point_to_node(next_id, pt)
        graph.add_node(node)
        request_node_ids.append(node.id)
        next_id += 1
        graph_changed = True

    # Persist updated graph only if a genuinely new stop was actually added
    if graph_changed:
        save_driver_graph(driver_id, graph)

    # current_location / end_point are transient: built fresh every call,
    # used as the route's fixed endpoints, but never added to the cache.
    current_node = _point_to_node(_CURRENT_LOCATION_ID, current_location)
    end_node = _point_to_node(_END_LOCATION_ID, end_point) if end_point is not None else None

    all_nodes = [current_node] + list(graph.nodes.values()) + ([end_node] if end_node else [])
    n = len(all_nodes)

    if n <= 1:
        return [node.id for node in all_nodes]

    id_to_pos = {node.id: idx for idx, node in enumerate(all_nodes)}
    end_pos = id_to_pos[_END_LOCATION_ID] if end_node else None

    # Build model input features for all current nodes
    points_arr = np.array([_node_to_model_input(node) for node in all_nodes], dtype=np.float32)
    node_features = torch.tensor(points_arr, dtype=torch.float32)

    # LGCN inference: capsule similarity scores per node (unaffected by the
    # fixed-endpoint logic below — it just scores every node, including the
    # endpoints, exactly as before).
    with torch.no_grad():
        node_caps = _lgcn.node_fc(node_features)
        node_caps = squash(node_caps)
        subroute_caps = _lgcn.routing(node_caps)
        full_route_caps = _lgcn.full_route_routing(subroute_caps)
        u_hat = _proj(node_caps)
        sim = torch.matmul(u_hat, full_route_caps.T).squeeze(1)

    dist_mat = _distance_matrix(points_arr[:, :2])

    # Build traffic/priority arrays aligned to current node order.
    # `request_node_ids[local_i]` is the node id that `stops[local_i]`
    # resolved to (a new node, or a pre-existing/duplicate one) — used to
    # map the request-ordered priorities/traffic_delays onto graph positions.
    pri = np.ones(n, dtype=np.float64)
    if priorities is not None:
        for local_i, nid in enumerate(request_node_ids):
            if local_i >= len(priorities):
                break
            pos = id_to_pos.get(nid)
            if pos is not None:
                pri[pos] = priorities[local_i]

    delay_mat = np.zeros((n, n), dtype=np.float64)
    if traffic_delays is not None:
        td = np.array(traffic_delays, dtype=np.float64)
        size = min(td.shape[0], len(request_node_ids))
        for a in range(size):
            pos_a = id_to_pos.get(request_node_ids[a])
            if pos_a is None:
                continue
            for b in range(size):
                pos_b = id_to_pos.get(request_node_ids[b])
                if pos_b is not None:
                    delay_mat[pos_a, pos_b] = td[a, b]

    # Greedy route construction (score = capsule similarity − edge cost).
    # current_node is always first (position 0); end_node, if given, is
    # excluded from competition and forced last — but every intermediate
    # choice is nudged ("pulled") toward it so the route doesn't strand a
    # far-away stop for the final leg. The pull strength ramps up as fewer
    # stops remain, since stranding risk grows the closer we get to the end.
    visited = [False] * n
    route = [0]
    visited[0] = True
    if end_pos is not None:
        visited[end_pos] = True  # taken out of the running; appended at the end

    remaining = n - 1 - (1 if end_pos is not None else 0)
    for step in range(remaining):
        last = route[-1]
        end_weight = 0.0
        if end_pos is not None:
            progress = step / remaining if remaining else 0.0
            end_weight = _END_PULL_BASE_WEIGHT * (0.5 + 0.5 * progress)

        best_score = -1e9
        best_next = None
        for j in range(n):
            if visited[j]:
                continue
            edge_cost = dist_mat[last, j] * (delay_mat[last, j] + pri[j])
            score = sim[j].item() - edge_cost
            if end_pos is not None:
                score -= dist_mat[j, end_pos] * end_weight
            if score > best_score:
                best_score = score
                best_next = j
        route.append(best_next)
        visited[best_next] = True

    if end_pos is not None:
        route.append(end_pos)

    # Return actual node IDs (not positional indices)
    return [all_nodes[i].id for i in route]


if __name__ == "__main__":
    # Simulate two calls for the same driver (incremental update).
    # current_location and end_point are now separate arguments — never
    # part of `stops`, never cached.
    current_location_call1 = [32.0840, 34.7810, 0, 24, 0]
    initial_stops = [
        [32.0853, 34.7818, 8.0, 12.0, 10],
        [32.0860, 34.7800, 9.0, 13.0, 5],
        [32.0820, 34.7900, 7.0, 11.0, 15],
    ]
    end_point = [32.0950, 34.7650, 0, 24, 0]  # mandatory final destination
    print("First call (current location + initial stops + end point):")
    route1 = optimize_route("driver_42", current_location_call1, initial_stops, end_point=end_point)
    print("Route node IDs:", route1)  # -1 (start) ... -2 (end, always last)
    print("Cached node count:", len(load_driver_graph("driver_42").nodes))  # expect 3

    # Second call: a slightly-jittered "current location" (simulating GPS
    # noise) + one stop that's already cached + one genuinely new stop.
    # The jittered current location must NOT pollute the cache, the
    # duplicate stop must NOT be re-added, only the new stop gets cached.
    current_location_call2 = [32.0840031, 34.7809972, 0, 24, 0]  # GPS jitter
    repeat_and_new = [
        [32.0853, 34.7818, 8.0, 12.0, 10],   # duplicate of an existing stop
        [32.0900, 34.7700, 10.0, 14.0, 0],   # genuinely new stop
    ]
    print("\nSecond call (jittered location + one duplicate + one new stop + end point):")
    route2 = optimize_route("driver_42", current_location_call2, repeat_and_new, end_point=end_point)
    print("Route node IDs:", route2)
    print("Cached node count:", len(load_driver_graph("driver_42").nodes))  # expect 4, not 5

    # Clean up test state
    reset_driver("driver_42")
