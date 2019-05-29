#!/bin/bash
#SBATCH --job-name=smallgrid
#SBATCH --output=smallgrid.out
#SBATCH --error=smallgrid.err
#SBATCH --time=1:30:00
#SBATCH --partition=broadwl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=14
#SBATCH --mem-per-cpu=2000

module load python/3.7.0
python3 ./main.py data/projects_2012_2013.csv small
