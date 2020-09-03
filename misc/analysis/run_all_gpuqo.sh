#!/bin/sh

function run(){
	cat <(echo "SET geqo TO off; SET gpuqo_threshold TO 2; EXPLAIN ") \
	    $1 \
	    <(echo " EXPLAIN (SUMMARY) ") \
	    $1 \
	| psql imdbload
}

for f in {?,??}?.sql; do 
	echo $f
	run $f > /dev/null
	run $f | grep "Planning Time\|Execution Time"
done
