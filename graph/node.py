class Node:
    def __init__(self, node_id, x, y, demand=0, time_window=None, wait=0):
        self.id = node_id  
        self.x = x
        self.y = y
        self.demand = demand
        self.time_window = time_window
        self.wait = wait

    def features(self):
        return [self.x, self.y, self.demand, self.wait] # TODO time window?
