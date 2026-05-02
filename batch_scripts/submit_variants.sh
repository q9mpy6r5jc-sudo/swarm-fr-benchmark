#!/bin/bash
#SBATCH --account=def-anonymous
#SBATCH --job-name=vcm_generate
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2                     # 0-13 for 14 parallel jobs   
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err  

module load StdEnv/2023
module load gcc/12.3 python/3.12.4
module load arrow/17.0.0

source ./env/bin/activate

DATASETS=(
    "kaden25fibroblast"
    "kaden25rpe1"
    "replogle22k562gwps"
)

CURRENT_DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
echo "Starting job for dataset: $CURRENT_DATASET"

python ./src/preprocessing/generate_variants_compute_canada.py \
    --dataset $CURRENT_DATASET \
    --base_path ./datasets/real