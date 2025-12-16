from graph.node import Node
from graph.dynamicGraph import DynamicGraph
from lgcn.lgcnModel import LGCN
from baseLine.greedy import greedy_route

graph = DynamicGraph()

graph.add_node(Node(0, 0, 0))
graph.add_node(Node(1, 2, 1, demand=3))
graph.add_node(Node(2, 1, 4, demand=2))

baseline = greedy_route(list(graph.nodes.values()))
print("Greedy:", [n.id for n in baseline])

lgcn = LGCN(feature_dim=4, capsule_dim=8)
embedding = lgcn.infer(graph.get_features())
print("LGCN route embedding:", embedding)
