from tqdm import tqdm
import os
import subprocess


# # IDP 5
# subprocess.run(
#     f"idp_type=IDP2 idp_n_iters=5 ./run_all_generic.sh gpuqo_bicc_dpsub summary-full clique mageirak 300 /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0050aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ab.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ac.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ad.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ae.sql | tee /scratch2/mageirak/clique/IDP2_5_5min/results_5min_timeout.txt",
#     shell=True,
# )

# IDP 10
subprocess.run(
    f"idp_type=IDP2 idp_n_iters=10 ./run_all_generic.sh gpuqo_bicc_dpsub summary-full clique mageirak 300 /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0050aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ab.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ac.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ad.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ae.sql | tee /scratch2/mageirak/clique/IDP2_10_5min/results_5min_timeout.txt",
    shell=True,
)

# IDP 15
subprocess.run(
    f"idp_type=IDP2 idp_n_iters=15 ./run_all_generic.sh gpuqo_bicc_dpsub summary-full clique mageirak 300 /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0050aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ab.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ac.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ad.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ae.sql | tee /scratch2/mageirak/clique/IDP2_15_5min/results_5min_timeout.txt",
    shell=True,
)


# UNION 10
subprocess.run(
    f"idp_type=UNIONDP idp_n_iters=10 ./run_all_generic.sh gpuqo_bicc_dpsub summary-full clique mageirak 300 /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0050aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ab.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ac.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ad.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ae.sql | tee /scratch2/mageirak/clique/UnionDP_10_5min/results_5min_timeout.txt",
    shell=True,
)

# GOO
subprocess.run(
    f"./run_all_generic.sh gpuqo_cpu_goo summary-full clique mageirak 300 /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0050aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100ab.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100ac.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100ad.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100ae.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ab.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ac.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ad.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ae.sql | tee /scratch2/mageirak/clique/GOO_5min/results_5min_timeout.txt",
    shell=True,
)

# IKKBZ
subprocess.run(
    f"./run_all_generic.sh gpuqo_cpu_ikkbz summary-full clique mageirak 300 /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0050aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100ab.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100ac.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100ad.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100ae.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ab.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ac.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ad.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ae.sql | tee /scratch2/mageirak/clique/IKKBZ_5min/results_5min_timeout.txt",
    shell=True,
)

# LINDP
subprocess.run(
    f"./run_all_generic.sh gpuqo_cpu_dplin summary-full clique mageirak 300 /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0050aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100ab.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100ac.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100ad.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100ae.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ab.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ac.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ad.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ae.sql | tee /scratch2/mageirak/clique/LinDP_5min/results_5min_timeout.txt",
    shell=True,
)

# GEQO
subprocess.run(
    f"./run_all_generic.sh geqo summary-full clique mageirak 300 /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0050aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100ab.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100ac.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100ad.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100ae.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ab.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ac.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ad.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0200ae.sql | tee /scratch2/mageirak/clique/GEQO/results_5min_timeout.txt",
    shell=True,
)