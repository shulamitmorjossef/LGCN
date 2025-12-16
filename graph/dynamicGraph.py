from graph.edge import Edge

class DynamicGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node):
        self.nodes[node.id] = node
        for other in self.nodes.values():
            if other.id != node.id:
                self.edges[(node.id, other.id)] = Edge(node, other)

    def remove_node(self, node_id):
        self.nodes.pop(node_id, None)
        self.edges = {k: v for k, v in self.edges.items()
                      if node_id not in k}

    def get_features(self):
        return [n.features() for n in self.nodes.values()]
