#!/usr/bash

rm benchmark.csv
echo "n_samples,eps,n_partitions,time" > benchmark.csv

for i in `seq 0 4`;
    ipython benchmark.py $i
done