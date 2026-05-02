#!/bin/bash
#SBATCH --account=def-anonymous            
#SBATCH --job-name=vcm_aggregate
#SBATCH --time=06:00:00
#SBATCH --nodes=1                        
#SBATCH --gpus-per-node=1                
#SBATCH --cpus-per-task=16               
#SBATCH --output=logs/%x_%j.out      
#SBATCH --error=logs/%x_%j.err       

module load StdEnv/2023
module load gcc/12.3 python/3.12.4
module load arrow/17.0.0

source ./env/bin/activate
echo "Starting Global Metric Aggregation using $SLURM_CPUS_PER_TASK CPU cores..."
python ./src/analyze/aggregate_per_metric.py

echo "Aggregation Complete!"