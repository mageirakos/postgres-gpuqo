#!/bin/sh

find . -name "results?.txt" | \
    while read file; do 
        dir=$(dirname "$file")
        series=$(basename "$dir")
        qt=$(basename "$(dirname "$dir")")
        out_dir="csv/$qt"
        out_file="$out_dir/$series.csv"

        echo $file
        echo $dir
        echo $series
        echo $qt
        echo $out_dir
        echo $out_file
        echo -------------------------
        echo

        mkdir -p "$out_dir"
        python analyze.py "$dir" -d "../$qt/queries" -m plan -t none --csv "$out_file"
    done 
