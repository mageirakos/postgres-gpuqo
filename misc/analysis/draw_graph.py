from sys import stdin, argv
from os import system
from tempfile import mkstemp

from collections import defaultdict
from pprint import pprint

import pydot

COLORSCHEME='dark28'
COLORSCHEME_CYCLE = 8

nodes = defaultdict(lambda: {
    'visited': False, 
    'depth': -1,
    'low': 1 << 30,
    'parent': None,
    'articulation': False, 
    'blocks': []
})
edges = defaultdict(list)

block_idx = 0


def output_comp(u, v, stack):
    global block_idx

    print(f"output_comp({u}, {v}, {stack})")

    comp = set()
    while True:
        eu, ev = stack.pop()
        comp.add(eu)
        comp.add(ev)

        if eu == u and ev == v:
            break

    block_idx += 1
    print(block_idx, comp)

    for n in comp:
        nodes[n]['blocks'].append(block_idx)


def get_articulation_points(i, d, stack):
    nodes[i]['visited'] = True
    nodes[i]['depth'] = d
    nodes[i]['low'] = d

    child_count = 0
    is_articulation = False

    for ni in edges[i]:
        if not nodes[ni]['visited']:
            stack.append((i, ni))
            nodes[ni]['parent'] = i
            get_articulation_points(ni, d+1, stack)
            child_count += 1
            if nodes[ni]['low'] >= nodes[i]['depth']:
                is_articulation = True
                output_comp(i, ni, stack)
            nodes[i]['low'] = min(nodes[i]['low'], nodes[ni]['low'])
        elif ni != nodes[i]['parent'] and nodes[ni]['depth'] < nodes[i]['depth']:
            stack.append((i, ni))
            nodes[i]['low'] = min(nodes[i]['low'], nodes[ni]['depth'])
    if ((nodes[i]['parent'] is not None and is_articulation) 
        or (nodes[i]['parent'] is None and child_count > 1)
    ):
        print(f"{i} is an articulation")
        nodes[i]['articulation'] = True
    else:
        print(f"{i} is not an articulation")

def color_blocks():
    block_id = 0
    for node in nodes:
        if nodes[node]['articulation']:
            continue
        if nodes[node]['blocks']:
            continue

        block_id += 1

        stack = [node]
        visited = {n:False for n in nodes}
        while stack:
            n = stack.pop() 

            if visited[n]:
                continue

            visited[n] = True
            nodes[n]['blocks'].append(block_id)

            if not nodes[n]['articulation']:
                stack += edges[n]

    for node in nodes:
        if not nodes[node]['articulation']:
            continue
        for n in edges[node]:
            if nodes[n]['articulation']:
                if not any(
                    b in nodes[n]['blocks'] for b in nodes[node]['blocks']
                ):
                    block_id += 1
                    nodes[node]['blocks'].append(block_id)
                    nodes[n]['blocks'].append(block_id)

    # TODO improve...
    for n1 in nodes:
        for n2 in nodes:
            if n2 > n1:
                s1 = set(nodes[n1]['blocks'])
                s2 = set(nodes[n2]['blocks'])
                inters = s1.intersection(s2)
                if len(inters) > 1:
                    b = min(inters)
                    for n3 in nodes:
                        s3 = set(nodes[n3]['blocks'])
                        if s3.intersection(inters):
                            nodes[n3]['blocks'] = list((s3-inters)|{b})


def block2col(block_id):
    return str((block_id-1) % COLORSCHEME_CYCLE + 1)

graph = pydot.Dot('my_graph', graph_type='graph')

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

if edges:
    get_articulation_points(1, 0, [])

    pprint(nodes)

    for node in edges.keys():
        if len(nodes[node]["blocks"]) > 1:
            style = 'wedged'
            fillcolor = ('"' 
                + ':'.join(block2col(i) for i in nodes[node]["blocks"])
                + '"'
            )
        elif len(nodes[node]["blocks"]) == 1:
            style = 'filled'
            fillcolor = block2col(nodes[node]["blocks"][0])
        else:
            style = 'filled'
            fillcolor = "black"

        if nodes[node]['articulation']:
            penwidth=2
        else:
            penwidth=1

        style_kwargs = {
            'style': style, 
            'colorscheme': COLORSCHEME,
            'fillcolor': fillcolor,
            'fontcolor': 'white',
            'penwidth': penwidth,
        }
        print(node, style_kwargs)
            
        graph.add_node(pydot.Node(
            str(node),
            **style_kwargs
        ))

    for node, other_nodes in edges.items():
        for other_node in other_nodes:
            if other_node > node: # do not add edges twice
                common_blocks = [
                    b 
                    for b in nodes[node]['blocks'] 
                    if b in nodes[other_node]['blocks']
                ]
                col = block2col(common_blocks[0]) if common_blocks else 'black'
                graph.add_edge(pydot.Edge(
                    str(node), 
                    str(other_node),
                    colorscheme=COLORSCHEME,
                    color=col
                ))

    if len(argv) > 1:
        out_filename = argv[1]
        out_file = out_filename
    else:
        out_file, out_filename = mkstemp()

    graph.write_png(out_file)
    print(f"Output written to {out_filename}")

    if len(argv) == 1:
        system(f"xdg-open {out_filename}")
