#!/bin/bash
#SBATCH --account=sadow
#SBATCH --partition=sadow
#SBATCH --nodes=1
##SBATCH --J=pinns
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time 3-00:00:00
#SBATCH --mem=12gb
#SBATCH --mail-user=linneamw@hawaii.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=./slurm/output_%j.out
##SBATCH --error=./error_%j.err

## CHECK FILE NAME!
export PATH="/home/linneamw/sadow_lts/personal/linneamw/anaconda3/bin:$PATH"
source /home/linneamw/profiles/auto.profile
conda activate pinns
module load --ignore-cache lib/slurm-drmaa/1.1.3
python src/run_pinn.py