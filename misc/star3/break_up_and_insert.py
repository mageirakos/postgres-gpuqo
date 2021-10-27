from tqdm import tqdm
import os
import subprocess


if __name__ == "__main__":
    # STAR 3
    num_dimension_tables = 1599 # star3 has 2k dimension tables

    # star table
    offset = 20_000 # Assumming 5k is breaking it down enough at 1.5k relations
    iters = (1_000_000) // offset # start and end done alone thats why -2

    # print("Reading fill_tables.sql...")
    # with open("fill_tables.sql", 'r') as f:
    #     data = f.read()

    # d = ";"
    # inserts =  [e+d for e in data.split(d) if e]
    # # fact table
    # fact_insert = inserts[0].split("\n")
    # fact_insert_into = "\n".join(fact_insert[:3])
    # values = fact_insert[3:]

    # if not os.path.exists("./inserts"):
    #     print("Creating /inserts directory...")
    #     os.makedirs("./inserts")
    
    # --- BREAK INSERTS
    
    # # # star table
    # print("Breaking INSERT INTO statements...")
    # for i in tqdm(range(iters)):
    #     with open(f"./inserts/insert_0_{i}.sql", 'w') as f:
    #         temp = fact_insert_into + "\n" + str("".join(values[offset*i:offset*(i+1)])).rstrip(",")  + ";"
    #         f.write(temp)

    # # dimension tables
    # for i in tqdm(range(1, len(inserts))):
    #     with open(f"./inserts/insert_{i}.sql", 'w') as f:
    #         f.write(inserts[i])
     
    # --- POPULATE


    # # star table
    print("Running INSERT INTO statements...")
    # star
    for i in tqdm(range(iters)):
        subprocess.run(
            f"psql -f inserts/insert_0_{i}.sql star3",
            shell=True,
        )

    # dimensions
    for i in tqdm(range(1,num_dimension_tables+1)):
        subprocess.run(
            f"psql -f inserts/insert_{i}.sql star3",
            shell=True,
        )
    
    
    print("Altering Tables statements...")
    subprocess.run(
        f"psql -f add_foreign_keys.sql star3",
        shell=True,
    )