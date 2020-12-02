#!/bin/bash

for N in {1..32}
do
    export OMP_NUM_THREADS=${N} 
    ./main >> omp_timing.txt
done
