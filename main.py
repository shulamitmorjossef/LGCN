import os
import json
import torch
import numpy as np
from lgcn.lgcnModel import LGCN
from lgcn.capsule import squash
from graph.dynamicGraph import DynamicGraph
from graph.node import Node
from graph.edge import Edge

# ─── מטריצות קבועות ───────────────────────────────────────────────────────────
# features: [lat, lon, tw_start, tw_end, wait_time]
# סקאלות: lat~32, lon~35, tw~8-18 (שעות), wait~0-20 (דקות)

# W1 [8×5]: features → node capsule
# כל שורה מייצרת ממד אחד בקפסולה.
# - שורות 0-1: נרמול lat/lon לטווח ~[0,1] (×0.03)
# - שורה 2:  מרכז חלון הזמן = (tw_s+tw_e)/2, קנה-מידה ×0.06 → ערך ~0.6-1.0
# - שורה 3:  רוחב החלון = tw_s - tw_e (תמיד שלילי).
#            חלון צר (2שע) → -0.3  /  חלון רחב (10שע) → -1.5
#            כלומר: דחוף = קרוב ל-0, גמיש = שלילי-גדול
# - שורה 4:  -tw_end × 0.08: deadline מוקדם (11) → -0.88, מאוחר (18) → -1.44
#            כלומר: deadline מוקדם = ערך פחות שלילי = גבוה יותר
# - שורה 5:  tw_start × 0.10: מתי מותר להגיע לכל היותר מוקדם
# - שורה 6:  -wait × 0.05: המתנה=0→0, המתנה=20→-1.0  (פחות המתנה = ערך גבוה יותר)
# - שורה 7:  ניקוד דחיפות משולב: lat+lon קטנים + חלון צר + deadline מוקדם + wait נמוך
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

# W2 [4×8]: node capsule → prediction vector לתת-מסלול (dynamic routing)
# תחנות עם ממדים דומים יצביעו בכיוון דומה → יקובצו יחד.
# - שורה 0: קלאסטר גיאוגרפי — משקל גבוה על dim0(lat) ו-dim1(lon)
#           תחנות קרובות מרחקית = הצבעות קרובות = יכנסו לאותו תת-מסלול
# - שורה 1: קלאסטר זמן — מרכז חלון (dim2) + deadline (dim4)
#           תחנות עם חלון זמן דומה = יתכנסו יחד
# - שורה 2: קלאסטר דחיפות — רוחב חלון (dim3) + deadline (dim4) + wait (dim6)
#           תחנות דחופות = הצבעות דומות = ייכנסו לתת-מסלול "דחוף"
# - שורה 3: יעילות כוללת — שילוב כל הממדים, משקל מיוחד ל-dim7 (ניקוד משולב)
_W2 = np.array([
    [ 0.90,  0.90,  0.00,  0.00,  0.00,  0.00,  0.00,  0.10],
    [ 0.00,  0.00,  0.85,  0.00,  0.85,  0.00,  0.00,  0.10],
    [ 0.00,  0.00,  0.00,  0.70,  0.60,  0.00,  0.70,  0.20],
    [ 0.25,  0.25,  0.25,  0.10,  0.10,  0.25,  0.20,  0.40],
], dtype=np.float32)

# W3 [4×4]: subroute capsule → prediction vector למסלול שלם
# עם num_subroutes=1 זהו מעבר יחיד (אין הסכמה). המטריצה מבצעת:
# - שורות 0-1: מיזוג קל בין ממד גיאוגרפי לממד זמן (10% cross-mix)
#   → הקפסולה הסופית מודעת לשני הממדים, לא רק לאחד
# - שורות 2-3: אותו רעיון לממדי דחיפות ויעילות
_W3 = np.array([
    [ 0.90,  0.10,  0.00,  0.00],
    [ 0.10,  0.90,  0.00,  0.00],
    [ 0.00,  0.00,  0.85,  0.15],
    [ 0.00,  0.00,  0.15,  0.85],
], dtype=np.float32)

# W4 [4×8]: node capsule → הטלה למרחב full_route_caps (לחישוב sim)
# sim[j] = dot(W4 × c_j, full_route_caps) — ציון כמה תחנה j "שייכת" למסלול
#
# הלוגיקה: dim3/dim4/dim6 של הקפסולה שליליים.
# תחנה דחופה = ערכים שליליים קרובים ל-0.
# W4 עם משקלים חיוביים על dim3,dim4,dim6 →
#   דחוף: W4×(-0.3) = -0.24  /  גמיש: W4×(-1.5) = -1.2
#   full_route_caps[2] (ממד דחיפות) גם שלילי →
#   dot: (-0.24)×(שלילי) > (-1.2)×(שלילי)  → תחנה דחופה מקבלת sim גבוה יותר ✓
#
# - שורה 0: ממד גיאוגרפי → מיפוי lat/lon לאותו ממד כמו W2/W3
# - שורה 1: ממד זמן → מרכז חלון + deadline
# - שורה 2: ממד דחיפות → רוחב חלון + deadline + wait (החשוב ביותר לניקוד)
# - שורה 3: ממד משולב → weighted combination של הכל
_W4 = np.array([
    [ 0.80,  0.80,  0.00,  0.00,  0.00,  0.00,  0.00,  0.20],
    [ 0.00,  0.00,  0.80,  0.00,  0.70,  0.50,  0.00,  0.20],
    [ 0.00,  0.00,  0.00,  0.70,  0.60,  0.00,  0.70,  0.30],
    [ 0.20,  0.20,  0.20,  0.30,  0.30,  0.20,  0.30,  0.50],
], dtype=np.float32)
# ──────────────────────────────────────────────────────────────────────────────

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
