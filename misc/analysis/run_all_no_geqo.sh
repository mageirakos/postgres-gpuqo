#!/bin/sh

for f in {?,??}?.sql; do 
	echo $f
	psql imdbload -f $f > /dev/null
	echo "SET geqo TO off; EXPLAIN ANALYZE" | cat - $f | psql imdbload | grep "Planning Time\|Execution Time"
done
