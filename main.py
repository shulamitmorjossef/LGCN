import torch
import numpy as np
from lgcn.lgcnModel import LGCN
from lgcn.capsule import squash

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

    with torch.no_grad():
        lgcn.node_fc.weight.copy_(torch.tensor(_W1))
        lgcn.node_fc.bias.zero_()
        lgcn.routing.transform.weight.copy_(torch.tensor(_W2))
        lgcn.routing.transform.bias.zero_()
        lgcn.full_route_routing.transform.weight.copy_(torch.tensor(_W3))
        lgcn.full_route_routing.transform.bias.zero_()

    lgcn.eval()

    node_features = torch.tensor(points, dtype=torch.float32)

    with torch.no_grad():
        node_caps = lgcn.node_fc(node_features)
        node_caps = squash(node_caps)
        subroute_caps = lgcn.routing(node_caps)
        full_route_caps = lgcn.full_route_routing(subroute_caps)

        proj = torch.nn.Linear(node_caps.size(1), full_route_caps.size(1), bias=False)
        proj.weight.copy_(torch.tensor(_W4))
        u_hat = proj(node_caps)
        sim = torch.matmul(u_hat, full_route_caps.T).squeeze(1)

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
