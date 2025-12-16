import math

class Edge:
    def __init__(self, i, j, traffic=1.0, urgency=1.0):
        self.i = i
        self.j = j
        self.traffic = traffic
        self.urgency = urgency

    def weight(self):
        dist = math.dist((self.i.x, self.i.y), (self.j.x, self.j.y))
        return dist * self.traffic + self.urgency
