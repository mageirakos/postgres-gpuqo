#!/bin/sh

for f in {?,??}?.sql; do 
	echo $f
	echo EXPLAIN ANALYZE | cat - $f | psql imdbload | grep "Planning Time\|Execution Time"
done
