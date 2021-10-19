
import string
import os
from random import randint, sample, choice
import numpy as np
from tqdm import tqdm


class SnowflakeSchema:
    def __init__(self, n_dims, n_leaves):
        self.n_dims = n_dims
        self.n_leaves = n_leaves

    def make_query(self, query_size):
        qs = [] # set of relations

        neigs = [(1,)]
        while neigs and len(qs) < query_size:
            id = choice(neigs)
            neigs.remove(id)

            leaf = id[-1] <= self.n_leaves[len(id)-1] 
            if not leaf:
                neigs += [(*id, i) for i in range(1, self.n_dims[len(id)]+1)]
            qs.append(id)
        # here I have all tables in qs        
        # from qs I generate from clause
        from_clause = ", ".join(map(lambda id: f"T_{id_to_str(id)}", qs))
        # generate workloads (pk,fk)
        where_clauses = []
        # for t in qs, ara select sugkekrimena t
        # select 25% of tables from qs


        for t in qs:
            if len(t) == 1:
                continue

            # T {0} is the parent
            where_clauses.append("T_{0}.t_{1} = T_{1}.pk".format(
                                id_to_str(t[:-1]), id_to_str(t)))
            # ADD HERE PREDICATE
        # TEST 1: 25% of tables have predicate, randing from 10 to 90% selective
        for t in sample(qs, int(0.25*len(qs))):
            if len(t) == 1:
                continue
            sel = randint(20,80)
            rows = card[f"t_{id_to_str(t)}"]
            fk = int(np.percentile(np.array(range(rows)), sel))
            print(f"{rows}\t{sel}\tt_{id_to_str(t)}")
            # print(f"{t} {sel} {rows} T_{id_to_str(t)}.pk > {fk}")
            # random choice 250 tables?
            where_clauses.append(f"T_{id_to_str(t)}.pk > {fk}") # fk_random is some random selectivity

        where_clause = " AND ".join(where_clauses)
        return f"SELECT * FROM {from_clause} WHERE {where_clause}; -- {query_size}"


def id_to_str(id):
    return '_'.join(map(str, id))

card = {}
def readCardinalities():
    global card
    with open(f"cardinalities.txt", 'r') as f:
        for l in f.readlines():
            # print(l)
            # print(" NEXT ")
            temp = [x.strip() for i,x in enumerate(l.split("|")) if i in (0,2)]
            for table_name, cardinality  in zip(temp, temp[1:]):
                if cardinality != "reltuples":
                    card[table_name] = int(cardinality)
    return 

if __name__ == "__main__":
    # to create cardinalities file run :
    # query_cardinalities.sql 
        # SELECT relname, relkind, reltuples, relpages
        # FROM pg_class
        # WHERE relname LIKE 't\_%' AND relkind = 'r';
    # with $psql star3 > ./cardinalities.txt < ./query_cardinalities.sql 
    readCardinalities() #  create card dict
    # for k,v in card.items():
    #     if v == 0:
    #         print(k,v)
    #TODO: FIX PROBLEM WHERE THESE BELOW APPEAR AS 0 IN query_cardinalities.sql
    card["t_1_231"] = 47
    card["t_1_527"] = 37
    card["t_1_795"] = 22


    schema = SnowflakeSchema((1, 1599, 0),
                             (0, 1599, 0))

    labels = [f"{a}{b}" for a in string.ascii_lowercase for b in string.ascii_lowercase]
    try:
        os.mkdir("queries_2")
    except FileExistsError:
        # directory already exists
        pass
        
    # create queries 
    # 10 to 100 step of 10
    for n in tqdm(range(10,101,10)):
        for i in range(104): # 104 because multiple of 26 (letters to name the queries)
            with open(f"queries_2/{n:04d}{labels[i]}.sql", 'w') as f:
                f.write(schema.make_query(n))
                f.write("\n")

    # 100 to 1000 step of 100
    for n in tqdm(range(100,1001,100)):
        for i in range(104):
            with open(f"queries_2/{n:04d}{labels[i]}.sql", 'w') as f:
                f.write(schema.make_query(n))
                f.write("\n")