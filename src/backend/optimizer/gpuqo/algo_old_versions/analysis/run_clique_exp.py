from tqdm import tqdm
import os
import subprocess


# IDP 15
subprocess.run(
    f"idp_type=IDP2 idp_n_iters=15 ./run_all_generic.sh gpuqo_bicc_dpsub summary-full clique mageirak 60 /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0010aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0030**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0040**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0050**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0060**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0070**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0080**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0090**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100**.sql | tee /scratch2/mageirak/clique/IDP2_15/results.txt",
    shell=True,
)

# IDP 25
subprocess.run(
    f"idp_type=IDP2 idp_n_iters=25 ./run_all_generic.sh gpuqo_bicc_dpsub summary-full clique mageirak 60 /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0010aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0030**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0040**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0050**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0060**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0070**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0080**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0090**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100**.sql | tee /scratch2/mageirak/clique/IDP2_25/results.txt",
    shell=True,
)

# UNIONDP 15
subprocess.run(
    f"idp_type=UNIONDP idp_n_iters=15 ./run_all_generic.sh gpuqo_bicc_dpsub summary-full clique mageirak 60 /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0010aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0030**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0040**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0050**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0060**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0070**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0080**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0090**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100**.sql | tee /scratch2/mageirak/clique/UnionDP_15/results.txt",
    shell=True,
)

# GEQO
subprocess.run(
    f"./run_all_generic.sh geqo summary-full clique mageirak 60 /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0010aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0030**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0040**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0050**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0060**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0070**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0080**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0090**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100**.sql | tee /scratch2/mageirak/clique/GEQO/results.txt",
    shell=True,
)

# GOO
subprocess.run(
    f"./run_all_generic.sh gpuqo_cpu_goo summary-full clique mageirak 60 /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0010aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0030**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0040**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0050**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0060**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0070**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0080**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0090**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100**.sql | tee /scratch2/mageirak/clique/GOO/results.txt",
    shell=True,
)

# IKKBZ
subprocess.run(
    f"./run_all_generic.sh gpuqo_cpu_ikkbz summary-full clique mageirak 60 /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0010aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0030**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0040**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0050**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0060**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0070**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0080**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0090**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100**.sql | tee /scratch2/mageirak/clique/IKKBZ/results.txt",
    shell=True,
)


# LINDP
subprocess.run(
    f"./run_all_generic.sh gpuqo_cpu_dplin summary-full clique mageirak 60 /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0010aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0030**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0040**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0050**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0060**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0070**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0080**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0090**.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100**.sql | tee /scratch2/mageirak/clique/LinDP/results.txt",
    shell=True,
)