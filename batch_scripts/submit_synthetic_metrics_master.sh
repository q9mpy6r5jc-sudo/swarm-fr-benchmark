#!/bin/bash

BASE_PATH="./datasets/synthetic"
RESULTS_PATH="./results/synthetic"

RUN_ID=$(date +%s)
TASK_FILE="logs/metrics/metrics_task_list_${RUN_ID}.txt"

echo "Scanning for incomplete variants..."
mkdir -p logs/metrics
touch $TASK_FILE

for TARGET_DIR in $(find $BASE_PATH -mindepth 3 -maxdepth 3 -type d | grep "/variants/"); do
    
    DATASET_NAME=$(echo $TARGET_DIR | awk -F'/' '{print $(NF-2)}')
    VARIANT_NAME=$(echo $TARGET_DIR | awk -F'/' '{print $NF}')
    
    if [[ "$VARIANT_NAME" == "GT_Reference" ]]; then
        EXPECTED_FILE="$RESULTS_PATH/$DATASET_NAME/performance_metrics_gt_baseline.csv"
    else
        LOWER_VARIANT=$(echo "$VARIANT_NAME" | tr '[:upper:]' '[:lower:]')
        EXPECTED_FILE="$RESULTS_PATH/$DATASET_NAME/performance_metrics_${LOWER_VARIANT}.csv"
    fi
    
    if [[ ! -f "$EXPECTED_FILE" ]]; then
        echo "$TARGET_DIR" >> $TASK_FILE
    fi
done

NUM_TASKS=$(wc -l < $TASK_FILE)

if [[ $NUM_TASKS -eq 0 ]]; then
    echo "All datasets and variants are fully complete! Nothing to run."
    rm $TASK_FILE
    exit 0
fi

echo "Found $NUM_TASKS variants that need processing."

MAX_TASKS=300
if [[ $NUM_TASKS -gt $MAX_TASKS ]]; then
    echo "Limiting submission to $MAX_TASKS to prevent SLURM QOS limit errors."
    NUM_TASKS=$MAX_TASKS
fi

sbatch <<EOT
#!/bin/bash
#SBATCH --account=def-anonymous            
#SBATCH --job-name=vcm_metrics
#SBATCH --time=10:00:00                  
#SBATCH --nodes=1                        
#SBATCH --gpus-per-node=1                
#SBATCH --cpus-per-task=16               
#SBATCH --array=1-${NUM_TASKS}%50                 
#SBATCH --output=logs/metrics/%x_%A_%a.out      
#SBATCH --error=logs/metrics/%x_%A_%a.err       

module load StdEnv/2023
module load gcc/12.3 python/3.12.4
module load arrow/17.0.0

source \./env/bin/activate

TARGET_DIR=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" $TASK_FILE)

if [[ -z "\$TARGET_DIR" ]]; then
    exit 0
fi

DATASET_NAME=\$(echo \$TARGET_DIR | awk -F'/' '{print \$(NF-2)}')
VARIANT_NAME=\$(echo \$TARGET_DIR | awk -F'/' '{print \$NF}')

echo "Executing Task \$SLURM_ARRAY_TASK_ID: Dataset=\$DATASET_NAME | Variant=\$VARIANT_NAME"

python \./src/metrics/compute_metrics_compute_canada.py \
    --dataset \$DATASET_NAME \
    --variant \$VARIANT_NAME \
    --base_path $BASE_PATH \
    --results_path $RESULTS_PATH
EOT

echo "Array job submitted successfully!"