#!/bin/bash
#SBATCH --account=def-anonymous            
#SBATCH --job-name=vcm_deg
#SBATCH --time=12:00:00=
#SBATCH --nodes=1                        
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0                  # 0-14 for 15 parallel jobs
#SBATCH --output=logs/%x_%A_%a.out      
#SBATCH --error=logs/%x_%A_%a.err       

module load StdEnv/2023
module load gcc/12.3 python/3.12.4
module load arrow/17.0.0

source ./env/bin/activate

DATASETS=(
    "genespider2"
)

CURRENT_DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Starting DEG calculation for dataset: $CURRENT_DATASET"
python ./src/preprocessing/get_deg_weights_compute_canada.py \
    --dataset $CURRENT_DATASET \
    --base_path ./datasets/synthetic