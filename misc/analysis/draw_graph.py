from sys import stdin, argv
from os import system
from tempfile import mkstemp

from collections import defaultdict

import pydot

graph = pydot.Dot('my_graph', graph_type='graph')

edges = defaultdict(list)

record = False
for line in stdin:
    print(line, end='')
    if "Edges:" in line:
        record = True
        continue
    
    if not record:
        continue

    if not line[0].isdigit():
        break

    colon_split = line.split(':')

    node_id = int(colon_split[0])

    edges_str = ':'.join(colon_split[1:])

    for edge_desc in edges_str.split(';'):
        if edge_desc.strip():
            other_node_id = int(edge_desc.split()[0])
            edges[node_id].append(other_node_id)

for node in edges.keys():
    graph.add_node(pydot.Node(str(node)))

for node, other_nodes in edges.items():
    for other_node in other_nodes:
        if other_node > node: # do not add edges twice
            graph.add_edge(pydot.Edge(str(node), str(other_node)))

print(edges)

if len(argv) > 1:
    out_file = argv[1]
else:
    out_file, out_filename = mkstemp()

graph.write_png(out_file)
print(f"Output written to {out_filename}")

if len(argv) == 1:
    system(f"xdg-open {out_filename}")
