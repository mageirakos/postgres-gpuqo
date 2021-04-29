import sys
import os
import argparse
import sqlparse
import re
from collections import defaultdict
from collections import deque
from tempfile import mkstemp
from sqlparse import sql

from sqlparse.sql import Identifier

sys.path.append(os.path.dirname(__file__))

from draw_graph import draw_graph

def perror(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)

def sanitize(s):
    return re.sub(r'[\'"]', '', s.strip())

def alias_or_name(identifier):
    return identifier.get_alias() or identifier.get_real_name()

def filter_tokens(func, token):
    if func(token):
        yield token
    
    if issubclass(type(token), sqlparse.sql.TokenList):
        for t in token.tokens:
            yield from filter_tokens(func, t)


class Rel:
    def __init__(self, id, name, alias = None):
        self.id = id
        if "AS" in name:
            split_name = name.split("AS")
            self.name = sanitize(split_name[0])
            self.alias = sanitize(split_name[1])
        elif alias is not None:
            self.name = sanitize(name)
            self.alias = sanitize(alias)
        else:
            self.name = sanitize(name)
            self.alias = sanitize(name)
        self.eq_classes = {}
        self.attrs = {}

    def __str__(self):
        out = ""
        out += self.name
        out += ": "
        for attr in self.attrs:
            out += f"{attr} ({self.attrs[attr]}), "
        
        return out[:-2]

    def __repr__(self):
        return self.__str__()

    def stmt(self):
        if self.alias != self.name:
            return f"\"{self.name}\" AS {self.alias}"
        else:
            return f"\"{self.name}\""


def parse(stmt):
    # SELECT * FROM "release_country", "release", "country_area", "medium", "release_packaging" WHERE "release_country"."release" = "release"."id" AND "release"."packaging" = "release_packaging"."id" AND "release_country"."country" = "country_area"."area" AND "release"."id" = "medium"."release"; 

    relations = {}
    identifier_list = list(filter(
        lambda x: type(x) is sqlparse.sql.IdentifierList, 
        stmt.tokens)
    )[-1]
    # perror(identifier_list)
    identifiers = filter(
        lambda x: type(x) is sqlparse.sql.Identifier, 
        identifier_list.tokens
    )

    for i, identifier in enumerate(identifiers):
        # perror(i, identifier)
        relations[alias_or_name(identifier)] = Rel(
            i+1,
            identifier.get_real_name(), 
            identifier.get_alias()
        )
    
    # perror(relations)

    where_clause = next(filter(
        lambda x: type(x) is sqlparse.sql.Where, 
        stmt.tokens)
    ) 
    comparison_list = list(filter(
        lambda x: type(x) is sqlparse.sql.Comparison, 
        where_clause.tokens
    ))
    binary_condition_list = filter(
        lambda x: (
            sum(map(lambda y: type(y) is sqlparse.sql.Identifier, x.tokens))==2
            and sum(map(lambda y: y.normalized == '=', x.tokens)) == 1
        ), 
        comparison_list
    )

    eq_class_count = 0
    for condition in binary_condition_list:
        identifiers = list(filter(lambda y: type(y) is sqlparse.sql.Identifier, 
                                condition.tokens))
        
        # perror(identifiers[0], identifiers[1])
        left_rel_name = identifiers[0].get_parent_name()
        left_attr_name = identifiers[0].get_real_name()
        right_rel_name = identifiers[1].get_parent_name()
        right_attr_name = identifiers[1].get_real_name()

        left_rel = relations[left_rel_name]
        right_rel = relations[right_rel_name]

        if left_attr_name in left_rel.attrs:
            eq_class_id = left_rel.attrs[left_attr_name]
            right_rel.attrs[right_attr_name] = eq_class_id
            right_rel.eq_classes[eq_class_id] = right_attr_name
        elif right_attr_name in right_rel.attrs:
            eq_class_id = right_rel.attrs[right_attr_name]
            left_rel.attrs[left_attr_name] = eq_class_id
            left_rel.eq_classes[eq_class_id] = left_attr_name
        else:
            eq_class_id = eq_class_count
            right_rel.attrs[right_attr_name] = eq_class_id
            right_rel.eq_classes[eq_class_id] = right_attr_name
            left_rel.attrs[left_attr_name] = eq_class_id
            left_rel.eq_classes[eq_class_id] = left_attr_name

            eq_class_count += 1

    unary_condition_list = filter(
        lambda x: (
            sum(map(lambda y: type(y) is sqlparse.sql.Identifier, x.tokens))==1
            and sum(map(lambda y: y.normalized == '=', x.tokens)) == 1
        ), 
        comparison_list
    )

    literals = defaultdict(list)
    for condition in unary_condition_list:
        identifier = list(filter(lambda y: type(y) is sqlparse.sql.Identifier, 
                                condition.tokens))[0]
        literal = list(filter(lambda y: "Token.Literal" in str(y.ttype), 
                                condition.tokens))[0]

        # perror(identifier, literal)
        literals[literal.normalized].append(identifier)
    
    for literal, l in literals.items():
        if len(l) <= 1:
            continue

        # perror(literal, l)
        
        for identifier in l:
            rel_name = identifier.get_parent_name()
            attr_name = identifier.get_real_name()

            rel = relations[rel_name]

            eq_class_id = eq_class_count
            rel.attrs[attr_name] = eq_class_id
            rel.eq_classes[eq_class_id] = attr_name

        eq_class_count += 1

    return relations

def build_adj_lists(relations):
    edges = defaultdict(list)
    eq_class_map = defaultdict(list)

    for rel in relations:
        for eq_class in relations[rel].eq_classes:
            eq_class_map[eq_class].append(rel)
    
    for rel in relations:
        for eq_class in relations[rel].eq_classes:
            for other_rel in eq_class_map[eq_class]:
                if rel != other_rel:
                    edges[relations[rel].id].append(relations[other_rel].id)
    
    for v in edges:
        edges[v].sort()

    return edges

def bfs_index(edges):
    queue = deque([min(edges.keys())])
    seen = defaultdict(lambda: False)
    remap_idxs = {}
    count = 0

    while queue:
        v = queue.pop()

        seen[v] = True
        count += 1
        remap_idxs[v] = count

        for w in edges[v]:
            if not seen[w]:
                queue.appendleft(w)
                seen[w] = True

    return remap_idxs

def remap_edges(edges, remap_idxs):
    new_edges = defaultdict(list)

    for v in edges:
        for w in edges[v]:
            new_edges[remap_idxs[v]].append(remap_idxs[w])
    
    return new_edges

def reorder_relations(relations, remap_idxs):
    id_rel = {remap_idxs[relations[rel].id]:relations[rel] for rel in relations}
    out = []
    for id in sorted(id_rel.keys()):
        out.append(id_rel[id])
    
    return out

def reorder_comparison(comparison, rel_order):
    aliases = [rel.alias for rel in rel_order]
    i = 0
    while i < len(comparison.tokens):
        if type(comparison.tokens[i]) is sqlparse.sql.Identifier:
            break
        i += 1
    j = i+1
    while j < len(comparison.tokens):
        if type(comparison.tokens[j]) is sqlparse.sql.Identifier:
            break
        j += 1
    
    idx1 = aliases.index(comparison.tokens[i].get_parent_name())
    idx2 = aliases.index(comparison.tokens[j].get_parent_name())

    if idx2 < idx1:
        comparison.tokens[i], comparison.tokens[j] = \
            comparison.tokens[j], comparison.tokens[i]


def order_val(token, rel_order):
    if not issubclass(type(token), sqlparse.sql.TokenList):
        return -1
    
    identifiers = list(filter_tokens(
        lambda x: type(x) is sqlparse.sql.Identifier, 
        token
    ))
    aliases = [rel.alias for rel in rel_order]

    # perror(token, identifiers)

    if len(identifiers) == 1:
        i = aliases.index(identifiers[0].get_parent_name())
        return (i+1)*(len(rel_order)+1)
    elif len(identifiers) >= 2:
        parent_names = [i.get_parent_name() for i in identifiers]
        parent_names.sort(key=lambda i: aliases.index(i))

        i = aliases.index(parent_names[0])
        j = aliases.index(parent_names[1])
        
        return (i+1)*(len(rel_order)+1) + (j+1)
    else:
        return -1

def reorder_stmt(stmt, rel_order):
    # perror(rel_order)
    
    aliases = [rel.alias for rel in rel_order]

    identifier_list = list(filter(
        lambda x: type(x) is sqlparse.sql.IdentifierList, 
        stmt.tokens)
    )[-1] 
    
    change = True
    while change:
        change = False
        for i in range(len(identifier_list.tokens)):
            for j in range(i+1, len(identifier_list.tokens)):
                if (type(identifier_list.tokens[i]) is sqlparse.sql.Identifier
                    and type(identifier_list.tokens[j]) is sqlparse.sql.Identifier
                ):
                    v1 = aliases.index(alias_or_name(identifier_list.tokens[i]))
                    v2 = aliases.index(alias_or_name(identifier_list.tokens[j]))

                    if v2 < v1:
                        identifier_list.tokens[i], identifier_list.tokens[j] = \
                            identifier_list.tokens[j], identifier_list.tokens[i]
                        change = True

    where_clause = next(filter(
        lambda x: type(x) is sqlparse.sql.Where, 
        stmt.tokens)
    ) 

    for i in range(len(where_clause.tokens)):
        t = where_clause.tokens[i]
        if type(t) is sqlparse.sql.Comparison:
            if (sum(map(
                    lambda y: type(y) is sqlparse.sql.Identifier, t.tokens
                    )) == 2 
                and sum(map(lambda y: y.normalized == '=', t.tokens)) == 1
            ):
                reorder_comparison(t, rel_order)

    
    change = True
    while change:
        change = False
        for i in range(len(where_clause.tokens)):
            for j in range(i+1, len(where_clause.tokens)):
                v1 = order_val(where_clause.tokens[i], rel_order)
                v2 = order_val(where_clause.tokens[j], rel_order)

                if v1 != -1 and v2 != -1 and v2 < v1:
                    where_clause.tokens[i], where_clause.tokens[j] = \
                            where_clause.tokens[j], where_clause.tokens[i]
                    change = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='do some operations on SQL queries.')
    parser.add_argument('operation', type=str, nargs=1,
                        help='Operation to do on input SQL query',
                        choices=["draw", "remap-draw", "reorder"])
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin)
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'))

    args = parser.parse_args()

    query = args.infile.read()
    stmt = sqlparse.parse(query)[0]
    relations = parse(stmt)
    edges = build_adj_lists(relations)

    if "draw" in args.operation[0]:
        if not args.outfile:
            out_file, out_filename = mkstemp()
        else:
            out_filename = args.outfile.name
            out_file = out_filename

        if "remap" in args.operation[0]:
            remap_idxs = bfs_index(edges)
            edges = remap_edges(edges, remap_idxs)

        graph = draw_graph(edges)
        graph.write_png(out_file)

        if not args.outfile:
            os.system(f"xdg-open {out_filename}")

    elif args.operation[0] == "reorder":
        remap_idxs = bfs_index(edges)
        order = reorder_relations(relations, remap_idxs)
        reorder_stmt(stmt, order)
        s = str(stmt)

        if args.outfile:
            args.outfile.write(s)
        else:
            print(s)

    else:
        print("Operation not recognized: ", args.operation)

