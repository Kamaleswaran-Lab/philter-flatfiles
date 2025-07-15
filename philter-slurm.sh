#!/bin/bash
#SBATCH --job-name=philter
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=32G
#SBATCH --time=250:00:00
#SBATCH --output=./philter_output_July_15.out
#SBATCH --error=./philter_error_July_15.err

python process_chunks.py