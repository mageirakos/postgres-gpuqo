#!/bin/sh

# Usage: <base|db|geqo|gpuqo> <analyze|summmary|run> <db> <query1> <query2> ...

case $1 in
base)	SETUP=""
	;;
dp)	SETUP="SET gpuqo TO off; SET geqo TO off;"
	;;
gpuqo)	SETUP="SET geqo TO off; SET gpuqo_threshold TO 2;"
	;;
geqo)	SETUP="SET gpuqo TO off; SET geqo_threshold TO 2;"
	;;
*)	echo "Unrecognized option: $1"
	exit 1
	;;
esac

export SETUP="$SETUP"

case $2 in
analyze) CMD="EXPLAIN ANALYZE"
	 ;;
summary) CMD="EXPLAIN (SUMMARY)"
	 ;;
run)	 CMD=""
	;;
*)	echo "Unrecognized option: $2"
	exit 1
	;;
esac

export CMD="$CMD"

export DB="$3"

shift
shift
shift

export QUERIES="$@"

(
echo "$SETUP"
echo EXPLAIN
cat $(echo "$QUERIES" | cut -d" " -f1)
for f in $QUERIES; do	
	echo "-- $f"
	echo "$CMD"
	cat $f
done
) | psql "$DB" -a | grep "\-\- \|Planning\|Execution"
