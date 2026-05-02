#!/bin/bash
#SBATCH --account=def-anonymous            
#SBATCH --job-name=merge_kaden
#SBATCH --time=10:00:00                  
#SBATCH --nodes=1                        
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1               
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/metrics/%x_%j.out      
#SBATCH --error=logs/metrics/%x_%j.err

module load StdEnv/2023
module load gcc/12.3 python/3.12.4
module load arrow/17.0.0

source ./env/bin/activate
python -u ./src/preprocessing/merge_kaden_chunks.py

echo "Merge process complete."