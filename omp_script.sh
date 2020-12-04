#!/bin/bash

for N in {1..32}
do
    export OMP_NUM_THREADS=${N} 
    ./main >> omp_timing_nadd_5h300.txt
done
