#!/bin/bash

tag=$1
qt=snowflake2
db=snowflake3
qtdir=../postgres/src/misc/$qt/queries

run(){
	alg=$1
	label=$2
	outdir="../$qt/benchmark/${label}_postgres_$tag"
	mkdir -p "$outdir"
	bash run_all_generic.sh \
			$alg \
			summary-full \
			$db rmancini \
			60 \
			$qtdir/002aa.sql \
			`ls $qtdir/{00{1,2,3,4,5,6,7,8,9}0,0{1,2,3,4}00}{a,b,c,d}{a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y}.sql` \
		| tee $outdir/results0.txt
}

#for alg in dp geqo gpuqo_cpu_dplin gpuqo_cpu_goo gpuqo_cpu_ikkbz gpuqo_cpu_dpccp gpuqo_bicc_dpsub; do 
for alg in geqo gpuqo_cpu_dplin gpuqo_cpu_goo gpuqo_cpu_ikkbz gpuqo_bicc_dpsub; do 
	run $alg $alg 
done

for alg in gpuqo_cpu_dpccp gpuqo_bicc_dpsub; do 
	for n in 5 10 15 20 25 30; do
		export idp_n_iters=$n
		run $alg "${alg}_${n}idp2"	
		unset idp_n_iters
	done
done
