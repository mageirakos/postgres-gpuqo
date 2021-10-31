from random import randint, sample
import string
import os
from tqdm import tqdm

TABLE_PATTERN="""CREATE TABLE T%d (
    pk INT PRIMARY KEY,
%s
);
"""

FK_PATTERN="""ALTER TABLE T%d
ADD FOREIGN KEY (t%d) REFERENCES T%d(pk);
"""

def make_create_tables(n):
    out = ""
    for i in range(1,n+1):
        columns = ",\n".join(["t%d INT" % j for j in range(1,n+1) if i != j])
        out += TABLE_PATTERN % (i,columns)
    return out

def make_foreign_keys(n):
    out = ""
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i != j:
                out += FK_PATTERN % (i,j,j)
    return out

def make_insert_into(n, size=10000):
    #TODO: change this so that the cardinalities in each table are random

    # we need to know cardinality for each table 
    # 1. Decide cardinalities of each table
    card = {} # "table":int(cardinality)
    for i in range(1,n+1):
        card[f"T{i}"] = randint(10_000, 1_000_000) # min_size=10_000, max_size=1_000_000
        # card[f"T{i}"] = randint(10, 1_000) # min_size=10_000, max_size=1_000_000
        print(f"T{i} : ", card[f"T{i}"])
    
    # 2. Populate tables and make sure foreign key constraint is kept based on above cardinalities
    out = ""
    for i in tqdm(range(1,n+1)):
        values = []
        columns = ', '.join([f"t{j}" for j in range(1,n+1) if i != j])
        out += f"INSERT INTO T{i} (pk, {columns})\nVALUES\n"
        # for each row based on this table's cardinality
        # print("here: ", i, card[f"T{i}"])
        for row in range(card[f"T{i}"]):
            temp = []
            for col in range(1,n+1):
                if i != col:
                    temp.append(str(randint(0,card[f"T{col}"]-1)))
            values.append(f"    ({row}, {', '.join(temp)})")
        # print(row)
        # values.append(f"    ({j}, {', '.join([str(randint(0,card[f"T{t}"]-1)) for t in range(1,n+1) if j != t])})")
        # values = [f"    ({j}, {', '.join([str(randint(0,size-1)) for _ in range(1,n)])})" for j in range(size)]
        out += ",\n".join(values)
        out += ";\n\n"
    return out

def make_query(N, n):
    # qs chooses n number of neighbours from N total tables
    qs = sample(list(range(1,N)), n)
    from_clause = ", ".join(["T%d" % j for j in qs])
    #todo: look into if joins should change, seems very deterministic no?
    where_clause = " AND ".join([f"T{i}.t{j} = T{j}.pk" for i in qs for j in qs if i < j])
    return f"SELECT * FROM {from_clause} WHERE {where_clause}; -- {n}"


if __name__ == "__main__":
    labels = [f"{a}{b}" for a in string.ascii_lowercase for b in string.ascii_lowercase]

    N = 200

    with open("create_tables.sql", 'w') as f:
        f.write(make_create_tables(N))
        f.write('\n')

    with open("add_foreign_keys.sql", 'w') as f:
        f.write(make_foreign_keys(N))
        f.write('\n')

    print("Make Inserts...")
    with open("fill_tables.sql", 'w') as f:
        f.write(make_insert_into(N))
        f.write('\n')

    # # for n in range(2,N):
    # #     for i in range(10):
    # #         with open(f"queries/{n:02d}{labels[i]}.sql", 'w') as f:
    # #             f.write(make_query(N, n))
    # #             f.write("\n")
    try:
        os.mkdir("queries")
    except FileExistsError:
        # directory already exists
        pass
    # # 2 to 10 step 1
    # for n in tqdm(range(2,10)):
    #     for i in range(104):
    #         with open(f"queries/{n:04d}{labels[i]}.sql", 'w') as f:
    #             f.write(make_query(N, n))
    #             f.write("\n")


    # 10 to 200 step 10
    for n in tqdm(range(10,201,10)):
        for i in range(104):
            with open(f"queries/{n:04d}{labels[i]}.sql", 'w') as f:
                f.write(make_query(N, n))
                f.write("\n")