from tqdm import tqdm
import os
import subprocess


if __name__ == "__main__":
    # STAR 3
    num_dimension_tables = 200 # star3 has 2k dimension tables

    # # dimensions
    for i in tqdm(range(1,num_dimension_tables+1)):
        subprocess.run(
            f"psql -f inserts/insert_T_{i}.sql clique",
            shell=True,
        )
    
    
    # print("Altering Tables statements...")
    # subprocess.run(
    #     f"psql -f add_foreign_keys.sql star3",
    #     shell=True,
    # )