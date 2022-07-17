from tqdm import tqdm

if __name__ == "__main__":
    # STAR 2
    with open("fill_tables.sql", 'r') as f:
        data = f.read()

    d = ";"
    inserts =  [e+d for e in data.split(d) if e]

    # fact table
    fact_insert = inserts[0].split("\n")
    fact_insert_into = "\n".join(fact_insert[:3])
    values = fact_insert[3:]

    with open(f"./inserts/insert_0_1.sql", 'w') as f:
        temp = fact_insert_into + "\n" + str("".join(values[:250_000])).rstrip(",") + ";"
        f.write(temp)
    
    with open(f"./inserts/insert_0_2.sql", 'w') as f:
        temp = fact_insert_into + "\n" + str("".join(values[250_000:500_000])).rstrip(",")  + ";"
        f.write(temp)
        
    with open(f"./inserts/insert_0_3.sql", 'w') as f:
        temp = fact_insert_into + "\n" + str("".join(values[500_000:750_000])).rstrip(",") + ";"
        f.write(temp)

    with open(f"./inserts/insert_0_4.sql", 'w') as f:
        temp = fact_insert_into + "\n" + "".join(values[750_000:]) 
        f.write(temp)


    # dimension tables
    for i in tqdm(range(1, len(inserts))):
        with open(f"./inserts/insert_{i}.sql", 'w') as f:
            f.write(inserts[i])
    