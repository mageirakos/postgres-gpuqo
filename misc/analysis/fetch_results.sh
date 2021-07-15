#!/bin/bash

FOLDER=$(dirname $(realpath $0))

DBS="chain star snowflake snowflake2 clique musicbrainz"

for DB in $DBS
do
    DB_FOLDER="$FOLDER/$DB/"
    COST_FOLDER="$FOLDER/total_costs/$DB/"
    echo $DB_FOLDER
    mkdir -p "$DB_FOLDER" 
    mkdir -p "$COST_FOLDER" 
    rsync -av diascld30:/scratch/rmancini/$DB/benchmark/ "$DB_FOLDER" &
    rsync -av diascld30:/scratch/rmancini/$DB/total_cost/ "$COST_FOLDER" &
done
mkdir -p "$FOLDER/total_costs/join-order-benchmark/" "$FOLDER/analyze/join-order-benchmark/"
rsync -av diascld30:/scratch/rmancini/join-order-benchmark/total_cost/ "$FOLDER/total_costs/join-order-benchmark/" &
rsync -av diascld30:/scratch/rmancini/join-order-benchmark/analyze/ "$FOLDER/analyze/join-order-benchmark/" &

wait
