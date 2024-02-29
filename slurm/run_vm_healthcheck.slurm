#!/bin/bash
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --ntasks-per-node=1

/opt/azurehpc/test/run-tests.sh --mofed-lts true