# ========================= main.py =========================
import torch
import numpy as np
import time
from lgcn.lgcnModel import LGCN


def distance_matrix(points):
    return np.sqrt(((points[:, None] - points[None, :]) ** 2).sum(axis=2))


def run_routing_simulation(nodes, num_drivers, driver_starts, lgcn):
    torch.manual_seed(0)

    while True:
        print("\n================ NEW STEP ================")

        # -------- Build distance matrix --------
        dist_mat = distance_matrix(nodes)

        # -------- Node features --------
        node_features = torch.tensor(nodes, dtype=torch.float32)

        # -------- Node capsules --------
        node_caps = lgcn.node_fc(node_features)  # [N, node_caps_dim]
        node_caps = node_caps / (
            torch.norm(node_caps, dim=-1, keepdim=True) + 1e-8
        )

        # -------- Sub-route capsules --------
        subroute_caps = lgcn.routing(node_caps)  # [num_drivers, subroute_caps_dim]

        # -------- Projection: u_hat_ij --------
        # maps node capsules -> sub-route capsule space
        proj = torch.nn.Linear(
            node_caps.size(1),
            subroute_caps.size(1),
            bias=False
        )

        u_hat = proj(node_caps)  # [N, subroute_caps_dim]

        # -------- Capsule agreement --------
        # agreement(i, j) = u_hat_i · v_j
        sim = torch.matmul(u_hat, subroute_caps.T)  # [N, num_drivers]

        # -------- Assign nodes to drivers --------
        assignments = {d: [] for d in range(num_drivers)}

        for i in range(len(nodes)):
            if i in driver_starts:
                continue
            driver = torch.argmax(sim[i]).item()
            assignments[driver].append(i)

        print("Assignments (capsule-based):", assignments)

        # -------- Build routes (capsule-guided greedy) --------
        driver_routes = {}

        for d in range(num_drivers):
            route = [driver_starts[d]]
            remaining = set(assignments[d])

            while remaining:
                last = route[-1]

                # score combines capsule agreement + locality
                best_score = -1e9
                best_node = None

                for n in remaining:
                    score = (
                        sim[n, d].item()
                        - 0.01 * dist_mat[last, n]
                    )
                    if score > best_score:
                        best_score = score
                        best_node = n

                route.append(best_node)
                remaining.remove(best_node)

            route.append(driver_starts[d])
            driver_routes[d] = route

        print("\nDriver Routes (LGCN-guided):")
        for d, r in driver_routes.items():
            print(f"Driver {d}: {r}")

        # -------- Dynamic event (new node arrives) --------
        new_point = nodes[-1] + np.random.uniform(-0.001, 0.001, size=2)
        nodes = np.vstack([nodes, new_point])

        print("\nAdded new node:", new_point)

        time.sleep(2)


# ========================= INIT =========================
nodes = np.array([
    [32.0853, 34.7818],  # depot 0
    [32.0860, 34.7800],  # depot 1
    [32.0820, 34.7900],
    [32.0900, 34.7700],
    [32.0880, 34.7850],
    [32.0830, 34.7750],
    [32.0870, 34.7820]
])

num_drivers = 2
driver_starts = [0, 1]

lgcn = LGCN(
    node_input_dim=2,
    node_caps_dim=8,
    subroute_caps_dim=4,
    num_subroutes=num_drivers
)

run_routing_simulation(nodes, num_drivers, driver_starts, lgcn)
# TODO : Add: time windows in main input,  and implement callback




