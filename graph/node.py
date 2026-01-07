class Node:
    def __init__(self, node_id, x, y, demand=0, time_windows=None, wait=0):
        self.id = node_id  
        self.x = x
        self.y = y
        self.demand = demand
        # time_windows: list of (start, end) tuples
        self.time_windows = time_windows if time_windows is not None else []
        self.wait = wait

    def features(self, max_windows=3):
        # Flatten all time windows, pad with zeros if less than max_windows
        tw_flat = []
        for tw in self.time_windows[:max_windows]:
            tw_flat.extend([tw[0], tw[1]])
        
        while len(tw_flat) < 2 * max_windows:
            tw_flat.append(0)
        return [self.x, self.y, self.demand, self.wait] + tw_flat
