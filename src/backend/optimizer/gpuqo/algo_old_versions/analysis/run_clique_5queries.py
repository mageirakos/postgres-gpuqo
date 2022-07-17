from tqdm import tqdm
import os
import subprocess


# UNIONDP 15
subprocess.run(
    "idp_type=UNIONDP idp_n_iters=15 ./run_all_generic.sh gpuqo_bicc_dpsub summary-full clique mageirak 60 /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0010aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0{03,04,05,06,07,08,10}0a{a..o}.sql | tee /scratch2/mageirak/clique/UnionDP_15_card_fix_2/results.txt",
    shell=True,
    executable='/bin/bash'
)


# subprocess.run(
#     "idp_type=UNIONDP idp_n_iters=10 ./run_all_generic.sh gpuqo_bicc_dpsub summary-full clique mageirak 60 /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0010aa.sql /scratch2/mageirak/postgres-gpuqo/misc/clique/queries/0100a{a..o}.sql | tee /scratch2/mageirak/clique/UnionDP_15_card_fix_2/results_large.txt",
#     shell=True,
#     executable='/bin/bash'
# )