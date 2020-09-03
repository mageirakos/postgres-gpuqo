#!/bin/sh

# Usage: <base|db|geqo|gpuqo> <analyze|summmary|run> <db> <timeout> <query1> <query2> ...

# run
#   runs the given guery in the DB with the given commands
#   Before execution the planning is run to warm up
#  Parameters:
#   1) database
#   2) Query file
#   3) Configuration commands
#   4) Execution mode (EXPLAIN ANALYZE, ....)
#   5) Warmup file
#   6) Timeout in seconds
function run(){
	cat <(echo "$3; EXPLAIN ") \
	    $5 \
	    <(echo "; $4 ") \
	    $2 \
	    <(echo ";") \
	| timeout -s SIGINT "$6" psql $1
}

dpsub_opt_pattern='SET gpuqo_dpsub_n_parallel TO $n_parallel; SET gpuqo_dpsub_filter TO $enable_filter; SET gpuqo_dpsub_filter_threshold TO $filter_threshold; SET gpuqo_dpsub_csg TO $enable_csg; SET gpuqo_dpsub_csg_threshold TO $csg_threshold; SET gpuqo_dpsub_filter_cpu_enum_threshold TO $filter_cpu_enum_threshold; SET gpuqo_dpsub_filter_keys_overprovisioning TO $filter_keys_overprovisioning'

n_parallel=${n_parallel:-10240}
enable_filter=${enable_filter:-on}
filter_threshold=${filter_threshold:-16}
enable_csg=${enable_csg:-on}
csg_threshold=${csg_threshold:-128}
filter_cpu_enum_threshold=${filter_cpu_enum_threshold:-1024}
filter_keys_overprovisioning=${filter_keys_overprovisioning:-128}

case $1 in
base)
	SETUP=""
	;;
dp)
	SETUP="SET gpuqo TO off; SET geqo TO off;"
	;;
gpuqo_dpsize) 	
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO dpsize;"
	;;
gpuqo_dpsub) 	
	dpsub_opt=$(eval "echo \"$dpsub_opt_pattern\"")
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO dpsub; $dpsub_opt;"
	;;
gpuqo_unfiltered_dpsub) 	
	enable_filter=off
	enable_csg=off
	dpsub_opt=$(eval "echo \"$dpsub_opt_pattern\"")
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO dpsub; $dpsub_opt;"
	;;
gpuqo_filtered_dpsub) 	
	enable_filter=on
	filter_threshold=0
	enable_csg=off
	dpsub_opt=$(eval "echo \"$dpsub_opt_pattern\"")
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO dpsub; $dpsub_opt;"
	;;
gpuqo_csg_dpsub) 
	enable_filter=off
	enable_csg=on
	csg_threshold=0
	dpsub_opt=$(eval "echo \"$dpsub_opt_pattern\"")
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO dpsub; $dpsub_opt;"
	;;
gpuqo_cpu_dpsize)
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO cpu_dpsize;"
	;;
gpuqo_cpu_dpsub)
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO cpu_dpsub;"
	;;
gpuqo_cpu_dpccp)
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO cpu_dpccp;"
	;;
gpuqo_dpe_dpsize)
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO dpe_dpsize;"
	;;
gpuqo_dpe_dpsub)
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO dpe_dpsub;"
	;;
gpuqo_dpe_dpccp)
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO dpe_dpccp; SET gpuqo_dpe_pairs_per_depbuf TO 1024; SET gpuqo_dpe_n_threads TO 8;"
	;;
geqo)	
	SETUP="SET gpuqo TO off; SET geqo_threshold TO 2;"
	;;
*)	
	echo "Unrecognized option: $1"
	exit 1
	;;
esac

case $2 in
analyze) 
	CMD="EXPLAIN ANALYZE"
	GREP="Planning Time\|Execution Time"
	;;
summary) 
	CMD="EXPLAIN (SUMMARY)"
	GREP="Planning Time"
	;;
summary-cost) 
	CMD="EXPLAIN (SUMMARY)"
	GREP="cost"
	;;
analyze-raw) 
	CMD="EXPLAIN ANALYZE"
	GREP=""
	;;
summary-raw) 
	CMD="EXPLAIN (SUMMARY)"
	GREP=""
	;;
run)	 
	CMD=""
	GREP=""
	;;
*)	echo "Unrecognized option: $2"
	exit 1
	;;
esac

DB="$3"
TIMEOUT="$4"

shift
shift
shift
shift

echo "$SETUP"

WARMUP=$(echo $@ | cut -d" " -f1)
for f in $@; do 
	echo $f
	date 1>&2
	run "$DB" "$f" "$SETUP" "$CMD" "$WARMUP" "$TIMEOUT" | grep "$GREP"
	if [ $? -ne 0 ]; then # error
		cnt=$(($cnt+1))
		# if returns error twice in a row, stop
		if [ $cnt -ge 2 ]; then
			exit 124
		fi
	else # ok, reset exit counter
		cnt=0
	fi
done
