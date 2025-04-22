#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 48
#SBATCH --time=01:00:00
#SBATCH --partition=plgrid
#SBATCH --account=plgmpr25-cpu

module load jq/1.6-gcccore-11.2.0

gcc bucket_sort_zad3.c -o bucket_sort -fopenmp
./test_script_sort.sh zad3
gcc bucket_sort_zad4.c -o bucket_sort -fopenmp
./test_script_sort.sh zad4
