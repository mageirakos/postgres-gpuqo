from tqdm import tqdm
import os
import subprocess


if __name__ == "__main__":

    # # star table
    print("Running INSERT INTO statements...")
    # star
    for i in tqdm(range(iters)):
        subprocess.run(
            f""
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