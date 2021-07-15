#!/bin/bash

tag=$1
qt=snowflake2
qtdir=../postgres/src/misc/$qt/queries

run(){
	alg=$1
	label=$2
	outdir="../$qt/benchmark/${label}_postgres_$tag"
	mkdir -p "$outdir"
	bash run_all_generic.sh \
			$alg \
			summary-full \
			$qt rmancini \
			60 \
			$qtdir/002aa.sql \
			`ls $qtdir/{010,015,020,025,030,040,050,060,070,080,090,100,125,150,175,200,250}a{a,b,c,d,e,f,g,h,i,j}.sql` \
		| tee $outdir/results0.txt
}

for alg in dp geqo gpuqo_cpu_dplin gpuqo_cpu_goo gpuqo_cpu_ikkbz; do 
	run $alg $alg 
done

for alg in gpuqo_cpu_dpccp gpuqo_bicc_dpsub; do 
	for n in 5 10 15 20 25; do
		export idp_n_iters=$n
		run $alg "${alg}_${n}idp2"	
		unset idp_n_iters
	done
done
