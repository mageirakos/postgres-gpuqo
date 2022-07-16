#!/bin/env python3 

import string
import os
from random import randint, choice

from tqdm import tqdm


TABLE_PATTERN="""CREATE TABLE T_{0:s} (
    pk SERIAL PRIMARY KEY{1:s}
);
"""

COLUMN_PATTERN="t_{0:s} INT"

FK_PATTERN="""ALTER TABLE T_{0:s}
ADD FOREIGN KEY (t_{1:s}) REFERENCES T_{1:s}(pk);
"""

DROP_FK_PATTERN="""ALTER TABLE T_{0:s}
DROP CONSTRAINT t_{0:s}_t_{1:s}_fkey;
"""

def id_to_str(id):
    return '_'.join(map(str, id))

class SnowflakeSchema:
    def __init__(self, n_dims, n_leaves):
        self.n_dims = n_dims
        self.n_leaves = n_leaves

    def __make_create_table(self, id, leaf=False):
        columns = ",\n    ".join([
            COLUMN_PATTERN.format(id_to_str((*id, i)))
            for i in range(1,self.n_dims[len(id)]+1)
        ]) if not leaf else ""
        if columns:
            columns = ",\n    " + columns
        return TABLE_PATTERN.format(id_to_str(id), columns)

    def __make_create_tables_rec(self, id):
        if len(id) >= len(self.n_dims):
            return []

        out = []
        for i in range(1,self.n_dims[len(id)]+1):
            new_id = (*id,i)
            leaf = i <= self.n_leaves[len(id)] 
            out.append(self.__make_create_table(new_id, leaf))
            if not leaf:
                out += self.__make_create_tables_rec(new_id)
        
        return out

    def make_create_tables(self):
        out = self.__make_create_tables_rec(tuple())
        return '\n\n'.join(out)

    def __make_foreign_keys(self, id):
        out = [FK_PATTERN.format(id_to_str(id), id_to_str((*id,i))) 
            for i in range(1,self.n_dims[len(id)]+1)
        ]
        return '\n'.join(out)

    def __make_foreign_keys_rec(self, id):
        if len(id) >= len(self.n_dims):
            return []

        out = []
        for i in range(1,self.n_dims[len(id)]+1):
            new_id = (*id,i)
            leaf = i <= self.n_leaves[len(id)] 
            if not leaf:
                out.append(self.__make_foreign_keys(new_id))
                out += self.__make_foreign_keys_rec(new_id)
        
        return out

    def make_foreign_keys(self):
        out = self.__make_foreign_keys_rec(tuple())
        return '\n\n'.join(out)

    def __make_drop_foreign_keys(self, id):
        out = [DROP_FK_PATTERN.format(id_to_str(id), id_to_str((*id,i))) 
            for i in range(1,self.n_dims[len(id)]+1)
        ]
        return '\n'.join(out)

    def __make_drop_foreign_keys_rec(self, id):
        if len(id) >= len(self.n_dims):
            return []

        out = []
        for i in range(1,self.n_dims[len(id)]+1):
            new_id = (*id,i)
            leaf = i <= self.n_leaves[len(id)] 
            if not leaf:
                out.append(self.__make_drop_foreign_keys(new_id))
                out += self.__make_drop_foreign_keys_rec(new_id)
        
        return out

    def make_drop_foreign_keys(self):
        out = self.__make_drop_foreign_keys_rec(tuple())
        return '\n\n'.join(out)

    def __make_insert_into(self, id, size, child_sizes):
        columns = ', '.join([
            "t_%s" % id_to_str((*id,i)) 
            for i in range(1,len(child_sizes)+1)
        ])
        if columns:
            columns = ",\n    " + columns

        out = []
        out.append(f"INSERT INTO T_{id_to_str(id)} (pk{columns})")
        out.append("VALUES")
        for i in range(size):
            values = [i]
            for child_size in child_sizes:
                values.append(randint(0,child_size-1))
            sep = ',' if i < size - 1 else ';'
            out.append("    (%s)%s" % (', '.join(map(str, values)), sep))

        return '\n'.join(out)

    def __make_insert_into_rec(self, id, sizes, min_size, max_size):
        print(f"__make_insert_into_rec{id}")
        if len(id)+1 >= len(self.n_dims):
            return

        for i, size in zip(range(1,self.n_dims[len(id)]+1), sizes):
            new_id = (*id,i)
            leaf = i <= self.n_leaves[len(id)] 
            if not leaf:
                child_sizes = [] 
                total_children = self.n_dims[len(id)+1]
                half = int(0.50*total_children+0.5) 
                # esentially step distribution [10,10k] 50% and [10k,1M] 50%
                for _ in range(half):
                    # manual 10, 10k for now...
                    child_sizes.append(randint(10, 10_000)) 
                for _ in range(total_children - half):
                    child_sizes.append(randint(min_size,max_size)) # 10k, 1M
                print(child_sizes, "\nlen = ", len(child_sizes))
            else:
                child_sizes = []
            yield self.__make_insert_into(new_id, size, child_sizes)
            yield from self.__make_insert_into_rec(new_id, child_sizes, 
                                                min(size//10, min_size), size)


    def write_insert_into(self, f, min_size=10_000, max_size=1_000_000):
        for out in self.__make_insert_into_rec(tuple(), [max_size], min_size, max_size):
            f.write(out)
            f.write('\n\n')

    def make_query(self, query_size):
        qs = []

        neigs = [(1,)]
        while neigs and len(qs) < query_size:
            id = choice(neigs)
            neigs.remove(id)

            leaf = id[-1] <= self.n_leaves[len(id)-1] 
            if not leaf:
                neigs += [(*id, i) for i in range(1, self.n_dims[len(id)]+1)]
            qs.append(id)
        
            
        from_clause = ", ".join(map(lambda id: f"T_{id_to_str(id)}", qs))
        where_clauses = []
        for t in qs:
            if len(t) == 1:
                continue

            where_clauses.append("T_{0}.t_{1} = T_{1}.pk".format(
                                id_to_str(t[:-1]), id_to_str(t)))

        where_clause = " AND ".join(where_clauses)
        return f"SELECT * FROM {from_clause} WHERE {where_clause}; -- {query_size}"


if __name__ == "__main__":
    # STAR 2

    # labels for query graph
    labels = [f"{a}{b}" for a in string.ascii_lowercase for b in string.ascii_lowercase]
    # here we can change to create "star" schema

    # param1 = number of nodes at each level -> 1 = 1 node at level 1, 16 nodes connected to nodes of previous level
    # 8 of the 16 do not have children, each of level 2 nodes have 4

    # at Level 1 (1 node, 0 children) : Level 2 (16 nodes connected to level1, 8 do not have children)
    # (Level 4 connectedto Level 2, 4 do not have children), level 5 is empty because level 4 has no children
    # schema = SnowflakeSchema((1, 16, 8, 4, 0),
    #                          (0,  8, 4, 4, 0))

    # you need to 0,0 in the last level to know when to stop
    # star schema with 1000 dimension tables and 1 fact table


    # POSTGRES UPPER LIMIT TO 1600 COLUMNS PER TABLE
    # SO FACT TABLE CAN ONLY HAVE 1 pk and 1599 fk for the dimension tables
    schema = SnowflakeSchema((1, 1599, 0),
                             (0, 1599, 0))
    
    with open("create_tables.sql", 'w') as f:
        f.write(schema.make_create_tables())
        f.write('\n')

    with open("add_foreign_keys.sql", 'w') as f:
        f.write(schema.make_foreign_keys())
        f.write('\n')

    with open("drop_foreign_keys.sql", 'w') as f:
        f.write(schema.make_drop_foreign_keys())
        f.write('\n')

    # fill tables, number of rows : (min_range, max_range) (fact table always maximum)
    with open("fill_tables.sql", 'w') as f:
        schema.write_insert_into(f, 10_000, 1_000_000)
    
    try:
        os.mkdir("queries")
    except FileExistsError:
        # directory already exists
        pass

    # create queries 
    # 10 to 100 step of 10
    for n in tqdm(range(10,101,10)):
        for i in range(104): # 104 because multiple of 26 (letters to name the queries)
            with open(f"queries/{n:04d}{labels[i]}.sql", 'w') as f:
                f.write(schema.make_query(n))
                f.write("\n")

    # 100 to 1000 step of 100
    for n in tqdm(range(100,1001,100)):
        for i in range(104):
            with open(f"queries/{n:04d}{labels[i]}.sql", 'w') as f:
                f.write(schema.make_query(n))
                f.write("\n")
