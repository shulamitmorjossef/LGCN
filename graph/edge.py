import math

class Edge:
    def __init__(self, i, j, traffic=1.0, urgency=1.0):
        self.dist = math.dist((i.x, i.y), (j.x, j.y))
        self.traffic = traffic
        self.urgency = urgency

    def weight(self):
        return self.dist * (self.traffic + self.urgency)  
