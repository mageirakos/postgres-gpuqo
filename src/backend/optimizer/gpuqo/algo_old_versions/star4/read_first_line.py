with open(f"fill_tables.sql", 'r') as f:
    data = f.read()

d = ";"
inserts =  [e+d for e in data.split(d) if e]
# fact table
fact_insert = inserts[0].split("\n")
fact_insert_into = "\n".join(fact_insert[:3])
values = fact_insert[3:]

temp = fact_insert_into
with open(f"blahFACT.txt", "w") as f:
    f.write(temp)