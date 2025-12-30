import torch
import torch.nn as nn
import numpy as np
import time
from lgcn.lgcnModel import LGCN

def assign_nodes_to_drivers(nodes, num_drivers, driver_starts, dist_mat):
    assignments = {i: [] for i in range(num_drivers)}

    for i in range(len(nodes)):
        dists = [dist_mat[i, start] for start in driver_starts]
        driver = np.argmin(dists)
        assignments[driver].append(i)

    return assignments

def distance_matrix(points, points2):
    return np.sqrt(((points[:, np.newaxis] - points2[np.newaxis, :]) ** 2).sum(axis=2))

def run_routing_simulation(nodes, num_drivers, driver_starts, lgcn):   # TODO make it lgcn


    while True:   #TODO callback - by event not busy wait

        dist_mat = distance_matrix(nodes, nodes)
        assignments = assign_nodes_to_drivers(nodes,num_drivers, driver_starts, dist_mat)
        print("Assignments:", assignments)

        node_features = torch.tensor(nodes, dtype=torch.float32)
        subroute_caps = lgcn(node_features)
        print("Subroute Capsules:", subroute_caps)

        driver_routes = {}
        for driver, node_ids in assignments.items():
            route = [driver_starts[driver]]
            visited = set(route)
            remaining_nodes = set(node_ids)
            while remaining_nodes:
                last_node = route[-1]
                candidates = [n for n in remaining_nodes if n not in visited]
                if not candidates:
                    break
                next_node = min(candidates, key=lambda n: dist_mat[last_node, n])
                route.append(next_node)
                visited.add(next_node)
                remaining_nodes.remove(next_node)
            route.append(driver_starts[driver])
            driver_routes[driver] = route

        print("\nDriver Routes (final, no repeats):")
        for driver, route in driver_routes.items():
            print(f"Driver {driver} route: {route}")
  
        new_point = nodes[-1] + np.random.uniform(-0.001, 0.001, size=2)    #TODO get a new stop from?
        nodes = np.vstack([nodes, new_point])
        print("\nAdded new point:", new_point)

        time.sleep(2)


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

lgcn = LGCN(node_input_dim=2, node_caps_dim=8, subroute_caps_dim=2, num_subroutes=num_drivers)

run_routing_simulation(nodes, num_drivers, driver_starts, lgcn)
