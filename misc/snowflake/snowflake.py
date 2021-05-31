#!/bin/env python3 

import string
import os
from random import randint, sample, choice, choices

from tqdm import tqdm


FACT_TABLE_PATTERN="""CREATE TABLE FACT (
    pk INT PRIMARY KEY,
{0:s}
);
"""

DIMENSION_TABLE_PATTERN="""CREATE TABLE DIM_{0:02d} (
    pk INT PRIMARY KEY,
{1:s}
);
"""

LEAF_TABLE_PATTERN="""CREATE TABLE DIM_{0:02d}_{1:02d} (
    pk INT PRIMARY KEY
);
"""

FACT_DIM_FK_PATTERN="""ALTER TABLE FACT
ADD FOREIGN KEY (dim_{0:02d}) REFERENCES DIM_{0:02d}(pk);
"""

DIM_LEAF_FK_PATTERN="""ALTER TABLE DIM_{0:02d}
ADD FOREIGN KEY (dim_{0:02d}_{1:02d}) REFERENCES DIM_{0:02d}_{1:02d}(pk);
"""

FACT_DIM_FK_DROP_PATTERN="""ALTER TABLE FACT
DROP CONSTRAINT fact_dim_{0:02d}_fkey;
"""

DIM_LEAF_FK_PATTERN="""ALTER TABLE DIM_{0:02d}
ADD FOREIGN KEY (dim_{0:02d}_{1:02d}) REFERENCES DIM_{0:02d}_{1:02d}(pk);
"""

DIM_LEAF_FK_DROP_PATTERN="""ALTER TABLE DIM_{0:02d}
DROP CONSTRAINT dim_{0:02d}_dim_{0:02d}_{1:02d}_fkey;
"""


class SnowflakeSchema:
    def __init__(self, n_dims, n_leaves_per_dim):
        self.n_dims = n_dims
        self.n_leaves_per_dim = n_leaves_per_dim

    def __make_create_fact_table(self):
        columns = ",\n".join(["dim_%02d INT" % i for i in range(1,self.n_dims+1)])
        return FACT_TABLE_PATTERN.format(columns)

    def __make_create_dim_table(self, dim_id):
        columns = ",\n".join(["dim_%02d_%02d INT" % (dim_id, i) for i in range(1,self.n_leaves_per_dim+1)])
        return DIMENSION_TABLE_PATTERN.format(dim_id, columns)

    def __make_create_leaf_table(self, dim_id, leaf_id):
        return LEAF_TABLE_PATTERN.format(dim_id, leaf_id)

    def make_create_tables(self):
        out = []
        out.append(self.__make_create_fact_table())
        for dim_id in range(1, self.n_dims+1):
            out.append(self.__make_create_dim_table(dim_id))
            for leaf_id in range(1, self.n_leaves_per_dim+1):
                out.append(self.__make_create_leaf_table(dim_id, leaf_id))
        return '\n\n'.join(out)

    def __make_foreign_keys_fact(self):
        out = [FACT_DIM_FK_PATTERN.format(i) for i in range(1, self.n_dims+1)]
        return '\n'.join(out)

    def __make_foreign_keys_dim(self, dim_id):
        out = [DIM_LEAF_FK_PATTERN.format(dim_id, i) for i in range(1, self.n_leaves_per_dim+1)]
        return '\n'.join(out)

    def make_foreign_keys(self):
        out = []
        out.append(self.__make_foreign_keys_fact())
        for dim_id in range(1, self.n_dims+1):
            out.append(self.__make_foreign_keys_dim(dim_id))
        return '\n\n'.join(out)

    def __make_drop_foreign_keys_fact(self):
        out = [FACT_DIM_FK_DROP_PATTERN.format(i) for i in range(1, self.n_dims+1)]
        return '\n'.join(out)

    def __make_drop_foreign_keys_dim(self, dim_id):
        out = [DIM_LEAF_FK_DROP_PATTERN.format(dim_id, i) for i in range(1, self.n_leaves_per_dim+1)]
        return '\n'.join(out)

    def make_drop_foreign_keys(self):
        out = []
        out.append(self.__make_drop_foreign_keys_fact())
        for dim_id in range(1, self.n_dims+1):
            out.append(self.__make_drop_foreign_keys_dim(dim_id))
        return '\n\n'.join(out)

    def __make_insert_into_fact(self, fact_size, dims_sizes):
        columns = ', '.join(["dim_%02d" % i for i in range(1,self.n_dims+1)])
        out = []
        out.append(f"INSERT INTO FACT (pk, {columns})")
        out.append("VALUES")
        for i in range(fact_size):
            values = [i]
            for size in dims_sizes:
                values.append(randint(0,size-1))
            sep = ',' if i < fact_size - 1 else ';'
            out.append("    (%s)%s" % (', '.join(map(str, values)), sep))

        return '\n'.join(out)

    def __make_insert_into_dim(self, dim_id, dim_size, leaves_sizes):
        columns = ', '.join(["dim_%02d_%02d" % (dim_id, i) for i in range(1,self.n_leaves_per_dim+1)])
        out = []
        out.append(f"INSERT INTO DIM_{dim_id:02d} (pk, {columns})")
        out.append("VALUES")
        for i in range(dim_size):
            values = [i]
            for size in leaves_sizes:
                values.append(randint(0,size-1))
            
            sep = ',' if i < dim_size - 1 else ';'
            out.append("    (%s)%s" % (', '.join(map(str, values)), sep))

        return '\n'.join(out)

    def __make_insert_into_leaf(self, dim_id, leaf_id, leaf_size):
        out = []
        out.append(f"INSERT INTO DIM_{dim_id:02d}_{leaf_id:02d} (pk)")
        out.append("VALUES")
        for i in range(leaf_size):
            sep = ',' if i < leaf_size - 1 else ';'
            out.append(f"    ({i}){sep}")
        return '\n'.join(out)

    def make_insert_into(self, min_size=1000, max_size=100000):
        out = []

        dims_sizes = [randint(min_size,max_size) for i in range(self.n_dims)]
        out.append(self.__make_insert_into_fact(max_size, dims_sizes))

        for dim_id, dim_size in zip(range(1,self.n_dims+1), tqdm(dims_sizes)):
            leaves_sizes = [randint(min_size,max_size) for i in range(self.n_leaves_per_dim)]
            out.append(self.__make_insert_into_dim(dim_id, dim_size, leaves_sizes))
            for leaf_id, leaf_size in zip(range(1,self.n_leaves_per_dim+1), leaves_sizes):
                out.append(self.__make_insert_into_leaf(dim_id, leaf_id, leaf_size))

        return '\n\n'.join(out[::-1])

    def make_query(self, query_size):
        dim_tables = list(range(1, self.n_dims+1))
        leaf_tables = list(range(1, self.n_leaves_per_dim+1))

        size_choices = [i for i in (range(0, self.n_leaves_per_dim+1))]
        n_div_4 = len(size_choices)//4
        n_rem = len(size_choices) - 2*n_div_4 - 1
        size_choice_weights = [1] + [8/n_div_4] * n_div_4 + [4/n_div_4] * n_div_4 + [2/n_rem] * n_rem

        qs = {"FACT"}
        while len(qs) < query_size:
            if dim_tables:
                dim_table = choice(dim_tables)
                dim_tables.remove(dim_table)

                dim_table_name = f"DIM_{dim_table:02d}"
                n_leaves = choices(size_choices, weights=size_choice_weights)[0]
            else:
                dim_table = choice(list(range(1, self.n_dims+1)))
                dim_table_name = f"DIM_{dim_table:02d}"
                remove_qs = [q for q in qs if dim_table_name in q]

                for q in remove_qs:
                    qs.remove(q)

                n_leaves = self.n_leaves_per_dim
            
            n_leaves = min(n_leaves, query_size - len(qs) - 1)
            
            qs.add(dim_table_name)
            for leaf in sample(leaf_tables, n_leaves):
                qs.add(f"DIM_{dim_table:02d}_{leaf:02d}")
                
            
        from_clause = ", ".join(qs)
        where_clauses = []
        for t in qs:
            if len(t) == 4: # FACT
                continue
            elif len(t) == 6: # DIM_XX
                left_table = "FACT"
                left_attr = t.lower()
                right_table = t
                right_attr = "pk"
            elif len(t) == 9: # DIM_XX_XX
                left_table = t[:6]
                left_attr = t.lower()
                right_table = t
                right_attr = "pk"
            else:
                print("Unrecognized table ", t)
                continue
            where_clauses.append(f"{left_table}.{left_attr} = {right_table}.{right_attr}")

        where_clause = " AND ".join(where_clauses)
        return f"SELECT * FROM {from_clause} WHERE {where_clause}; -- {query_size}"


if __name__ == "__main__":
    labels = [f"{a}{b}" for a in string.ascii_lowercase for b in string.ascii_lowercase]

    schema = SnowflakeSchema(32, 16)

    # with open("create_tables.sql", 'w') as f:
    #     f.write(schema.make_create_tables())
    #     f.write('\n')

    # with open("add_foreign_keys.sql", 'w') as f:
    #     f.write(schema.make_foreign_keys())
    #     f.write('\n')

    with open("drop_foreign_keys.sql", 'w') as f:
        f.write(schema.make_drop_foreign_keys())
        f.write('\n')

    # with open("fill_tables.sql", 'w') as f:
    #     f.write(schema.make_insert_into(1000,100000))
    #     f.write('\n')

    # try:
    #     os.mkdir("queries")
    # except FileExistsError:
    #     # directory already exists
    #     pass

    # for n in tqdm(range(2,128)):
    #     for i in range(128):
    #         with open(f"queries/{n:02d}{labels[i]}.sql", 'w') as f:
    #             f.write(schema.make_query(n))
    #             f.write("\n")
