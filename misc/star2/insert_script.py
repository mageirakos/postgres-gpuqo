import subprocess

for i in range(1,402):
    subprocess.run(
        f"psql -f inserts/insert_{i}.sql star2",
        shell=True,
    )
