#!/bin/bash
## CHECK FILE NAME

#SBATCH --account=sadow
#SBATCH --partition=sadow
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --error=./error_%j.err
#SBATCH --output=./output_%j.out
#SBATCH --mem=12gb
#SBATCH --mail-user=linneamw@hawaii.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#echo hello

export PATH="/home/linneamw/sadow_lts/personal/linneamw/anaconda3/bin:$PATH"
source activate pinns
python sherpa_pinn_force_field_equation.py