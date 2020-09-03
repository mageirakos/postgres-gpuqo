#!/bin/sh

for f in {?,??}?.sql; do 
	echo $f
	diff <(echo "$1" | cat - $f | timeout 60 psql imdbload) <(echo "$2" | cat - $f | timeout 60 psql imdbload)
	if [ $? = "0" ]; then
		echo "OK"
	fi
done
