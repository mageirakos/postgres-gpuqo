#!/bin/bash

# Usage: <base|db|geqo|gpuqo_*> <analyze|summmary|...> <db> <role> <timeout> <warmup_query> <query1> <query2> ...

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
	(
		echo "$3"; \
		cat <(echo "EXPLAIN ") \
			$5 \
			<(echo ";") \
		| tr '\n' ' '; \
		echo; \
		cat <(echo "$4 ") \
			$2 \
			<(echo ";") \
		| tr '\n' ' '; \
		echo; \
	) | tee /dev/stderr | timeout -s SIGINT -k 10 "$6" postgres --single $1
}

# common
scratchpad_size=${scratchpad_size:-100}
min_memo_size=${min_memo_size:-1}
max_memo_size=${max_memo_size:-7000}
n_parallel=${n_parallel:-40960}
enable_spanning=${enable_spanning:-off}

# dpsize
dpsize_opt_pattern='SET gpuqo_scratchpad_size_mb TO $scratchpad_size; SET gpuqo_max_memo_size_mb TO $max_memo_size; SET gpuqo_n_parallel TO $n_parallel'

# dpsub
enable_filter=${enable_filter:-on}
filter_threshold=${filter_threshold:-0}
enable_csg=${enable_csg:-on}
enable_tree=${enable_tree:-off}
enable_bicc=${enable_bicc:-off}
csg_threshold=${csg_threshold:-32}
filter_cpu_enum_threshold=${filter_cpu_enum_threshold:-1024}
filter_keys_overprovisioning=${filter_keys_overprovisioning:-128}

dpsub_opt_pattern='SET gpuqo_n_parallel TO $n_parallel; SET gpuqo_dpsub_filter TO $enable_filter; SET gpuqo_dpsub_filter_threshold TO $filter_threshold; SET gpuqo_dpsub_csg TO $enable_csg; SET gpuqo_dpsub_tree TO $enable_tree; SET gpuqo_dpsub_bicc TO $enable_bicc; SET gpuqo_dpsub_csg_threshold TO $csg_threshold; SET gpuqo_dpsub_filter_cpu_enum_threshold TO $filter_cpu_enum_threshold; SET gpuqo_dpsub_filter_keys_overprovisioning TO $filter_keys_overprovisioning; SET gpuqo_max_memo_size_mb TO $max_memo_size; SET gpuqo_min_memo_size_mb TO $min_memo_size'

#dpe
pairs_per_depbuf=${pairs_per_depbuf:-16384}
n_threads=${n_threads:-8}
chunk_size=${chunk_size:-256}

case $1 in
base)
	SETUP=""
	;;
dp)
	SETUP="SET gpuqo TO off; SET geqo TO off;"
	;;
gpuqo_dpsize) 	
	dpsize_opt=$(eval "echo \"$dpsize_opt_pattern\"")
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO dpsize; $dpsize_opt;"
	;;
gpuqo_dpsub) 	
	dpsub_opt=$(eval "echo \"$dpsub_opt_pattern\"")
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO dpsub; $dpsub_opt;"
	;;
gpuqo_unfiltered_dpsub) 	
	enable_filter=off
	enable_csg=off
	enable_tree=off
	dpsub_opt=$(eval "echo \"$dpsub_opt_pattern\"")
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO dpsub; $dpsub_opt;"
	;;
gpuqo_filtered_dpsub) 	
	enable_filter=on
	enable_csg=off
	enable_tree=off
	enable_bicc=off
	dpsub_opt=$(eval "echo \"$dpsub_opt_pattern\"")
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO dpsub; $dpsub_opt;"
	;;
gpuqo_csg_dpsub) 
	enable_filter=on
	enable_csg=on
	enable_tree=off
	enable_bicc=off
	dpsub_opt=$(eval "echo \"$dpsub_opt_pattern\"")
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO dpsub; $dpsub_opt;"
	;;
gpuqo_tree_dpsub) 
	enable_filter=on
	enable_csg=on
	enable_tree=on
	enable_bicc=off
	dpsub_opt=$(eval "echo \"$dpsub_opt_pattern\"")
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO dpsub; $dpsub_opt;"
	;;
gpuqo_bicc_dpsub) 
	enable_filter=on
	enable_csg=on
	enable_tree=off
	enable_bicc=on
	dpsub_opt=$(eval "echo \"$dpsub_opt_pattern\"")
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO dpsub; $dpsub_opt;"
	;;
gpuqo_cpu_dpsize)
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO cpu_dpsize;"
	;;
gpuqo_cpu_dpsub)
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO cpu_dpsub;"
	;;
gpuqo_cpu_dpsub_bicc)
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO cpu_dpsub_bicc;"
	;;
gpuqo_cpu_dpccp)
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO cpu_dpccp;"
	;;
gpuqo_cpu_dpsub_parallel)
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO parallel_cpu_dpsub; SET gpuqo_dpe_n_threads TO $n_threads; SET gpuqo_cpu_dpsub_parallel_chunk_size TO $chunk_size;"
	;;
gpuqo_cpu_dpsub_bicc_parallel)
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO parallel_cpu_dpsub_bicc; SET gpuqo_dpe_n_threads TO $n_threads; SET gpuqo_cpu_dpsub_parallel_chunk_size TO $chunk_size;"
	;;
gpuqo_dpe_dpsize)
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO dpe_dpsize;"
	;;
gpuqo_dpe_dpsub)
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO dpe_dpsub;"
	;;
gpuqo_dpe_dpccp)
	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2; SET gpuqo_algorithm TO dpe_dpccp; SET gpuqo_dpe_pairs_per_depbuf TO $pairs_per_depbuf; SET gpuqo_dpe_n_threads TO $n_threads;"
	;;
geqo)	
	SETUP="SET gpuqo TO off; SET geqo_threshold TO 2;"
	;;
*)	
	echo "Unrecognized option: $1"
	exit 1
	;;
esac

if [ $enable_spanning = 'on' ]; then
	SETUP="$SETUP SET gpuqo_spanning_tree TO on;";
fi

case $2 in
analyze) 
	CMD="EXPLAIN ANALYZE"
	GREP="grep 'Planning Time\|Execution Time'"
	;;
summary) 
	CMD="EXPLAIN (SUMMARY)"
	GREP="grep 'Planning Time'"
	;;
summary-profile) 
	CMD="EXPLAIN (SUMMARY)"
	GREP="grep 'took\|iteration\|Planning Time'"
	;;
summary-cost) 
	CMD="EXPLAIN (SUMMARY)"
	GREP="grep 'cost'"
	;;
summary-total-cost) 
	CMD="EXPLAIN (SUMMARY)"
	GREP="grep -E '^[A-Z].*cost' | grep -oP '(?<=\.\.)[0-9\.]+' | tail -n 1"
	;;
gpuqo-cost) 
	CMD="EXPLAIN (SUMMARY)"
	GREP="grep 'gpuqo cost is'"
	;;
analyze-raw) 
	CMD="EXPLAIN ANALYZE"
	GREP="grep ''"
	;;
summary-raw) 
	CMD="EXPLAIN (SUMMARY)"
	GREP="grep ''"
	;;
run)	 
	CMD=""
	GREP="grep ''"
	;;
*)	echo "Unrecognized option: $2"
	exit 1
	;;
esac

DB="$3"
ROLE="$4"
TIMEOUT="$5"
WARMUP="$6"

shift
shift
shift
shift
shift
shift

SETUP="SET ROLE TO $ROLE; $SETUP"

echo "$SETUP"

for f in $@; do 
	echo $f
	date 1>&2
	run "$DB" "$f" "$SETUP" "$CMD" "$WARMUP" "$TIMEOUT" | sed 's/.* QUERY PLAN = "\(.*\)".*/\1/g' | bash -c "$GREP"
	if [ ${PIPESTATUS[0]} -ne 0 ]; then # error
		cnt=$(($cnt+1))
		# if returns error twice in a row, stop
		if [ $cnt -ge 2 ]; then
			exit 124
		fi

		echo "Sleeping 5s hoping things will get better"
		sleep 5
	else # ok, reset exit counter
		cnt=0
	fi
done
