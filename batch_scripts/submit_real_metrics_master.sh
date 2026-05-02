#!/bin/bash
BASE_PATH="./datasets/real"
RESULTS_PATH="./results/real"

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
        BASE_FILE_NO_EXT="$RESULTS_PATH/$DATASET_NAME/performance_metrics_gt_baseline"
    else
        LOWER_VARIANT=$(echo "$VARIANT_NAME" | tr '[:upper:]' '[:lower:]')
        EXPECTED_FILE="$RESULTS_PATH/$DATASET_NAME/performance_metrics_${LOWER_VARIANT}.csv"
        BASE_FILE_NO_EXT="$RESULTS_PATH/$DATASET_NAME/performance_metrics_${LOWER_VARIANT}"
    fi

    if [[ ! -f "$EXPECTED_FILE" ]]; then
        # Check for individual chunks - massive datasets
        if [[ "$DATASET_NAME" == *"kaden25"* ]] || [[ "$DATASET_NAME" == *"replogle22k562gwps"* ]]; then
            NUM_CHUNKS=20
            for (( c=0; c<$NUM_CHUNKS; c++ )); do
                CHUNK_FILE="${BASE_FILE_NO_EXT}_chunk_${c}.csv"
                # Only add this specific chunk if it hasn't been completed yet
                if [[ ! -f "$CHUNK_FILE" ]]; then
                    echo "$TARGET_DIR $c $NUM_CHUNKS" >> $TASK_FILE
                fi
            done
        else
            echo "$TARGET_DIR 0 1" >> $TASK_FILE
        fi
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

ARRAY_JOB_ID=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --account=def-anonymous            
#SBATCH --job-name=vcm_metrics
#SBATCH --time=24:00:00                  
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


LINE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" $TASK_FILE)

if [[ -z "\$LINE" ]]; then
    exit 0
fi

TARGET_DIR=\$(echo \$LINE | awk '{print \$1}')
CHUNK_IDX=\$(echo \$LINE | awk '{print \$2}')
NUM_CHUNKS=\$(echo \$LINE | awk '{print \$3}')

DATASET_NAME=\$(echo \$TARGET_DIR | awk -F'/' '{print \$(NF-2)}')
VARIANT_NAME=\$(echo \$TARGET_DIR | awk -F'/' '{print \$NF}')

echo "Executing Task \$SLURM_ARRAY_TASK_ID: Dataset=\$DATASET_NAME | Variant=\$VARIANT_NAME | Chunk=\$CHUNK_IDX/\$NUM_CHUNKS"

python \./src/metrics/compute_metrics_compute_canada.py \
    --dataset \$DATASET_NAME \
    --variant \$VARIANT_NAME \
    --base_path $BASE_PATH \
    --results_path $RESULTS_PATH \
    --chunk_idx \$CHUNK_IDX \
    --num_chunks \$NUM_CHUNKS
EOT
)

echo "Array job submitted successfully with ID: $ARRAY_JOB_ID"

sbatch <<EOT
#!/bin/bash
#SBATCH --account=def-anonymous
#SBATCH --job-name=vcm_merge
#SBATCH --dependency=afterok:$ARRAY_JOB_ID
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1 
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/metrics/merge_%j.out
#SBATCH --error=logs/metrics/merge_%j.err

module load StdEnv/2023
module load gcc/12.3 python/3.12.4
module load arrow/17.0.0

source \./env/bin/activate

echo "Starting automated chunk merging..."

python -c "
import os, glob
import pandas as pd

results_dir = '$RESULTS_PATH'
chunk_zero_files = glob.glob(os.path.join(results_dir, '*', '*_chunk_0.csv'))

for chunk_zero in chunk_zero_files:
    base_path = chunk_zero.replace('_chunk_0.csv', '')
    base_name = os.path.basename(base_path)

    chunk_files = glob.glob(f'{base_path}_chunk_*.csv')
    print(f'Merging {len(chunk_files)} chunks for {base_name}...')
    
    try:
        df = pd.concat([pd.read_csv(f) for f in chunk_files])
        final_path = f'{base_path}.csv'
        df.to_csv(final_path, index=False)
        print(f'Successfully saved: {final_path}')
        
        for f in chunk_files:
            os.remove(f)
        print(f'Cleaned up chunk files for {base_name}')
    except Exception as e:
        print(f'Error merging {base_name}: {e}')
"
echo "Merge and cleanup complete!"
EOT

echo "Merge job submitted! It will run automatically once job $ARRAY_JOB_ID finishes."