import math

def greedy_route(nodes):
    route = []
    current = nodes[0]
    remaining = nodes[1:]

    while remaining:
        nxt = min(
            remaining,
            key=lambda n: math.dist((current.x, current.y), (n.x, n.y))
        )
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt

    return route
