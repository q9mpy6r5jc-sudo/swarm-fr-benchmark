import pandas as pd
import numpy as np
import os, glob, re, sys, gc, argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def process_single_file_chunked(file_path):
    try:
        filename = os.path.basename(file_path).lower()
        dataset_name = os.path.basename(os.path.dirname(file_path))
        
        variant = "Unknown"
        level = 0
        snr = "Infinity"
        
        if "gt_baseline" in filename:
            variant = "GT_Baseline"
        elif "noise" in filename:
            variant = "Noise"
            match = re.search(r"level_(\d+)", filename)
            if match: level = int(match.group(1))
        elif "structure" in filename:
            variant = "Structure"
            match = re.search(r"level_(\d+)", filename)
            if match: level = int(match.group(1))
        elif "magnitude" in filename:
            variant = "Magnitude"
            match = re.search(r"level_(\d+)", filename)
            if match: level = int(match.group(1))
            
        snr_match = re.search(r"snr_([0-9\.]+)(?=\.csv)", filename)
        if snr_match:
            snr = float(snr_match.group(1))
            
        use_cols = ['Metric', 'Weighting_Strategy', 'Reference_Strategy', 'Aggregation_Strategy', 'Attack_Type', 'Value', 'Feature_Space']
        
        chunk_aggregations = []
        local_group_cols = ['Metric', 'Weighting_Strategy', 'Reference_Strategy', 'Aggregation_Strategy', 'Feature_Space', 'Attack_Type']
        
        chunk_size = 500_000 
        
        try:
            for chunk in pd.read_csv(file_path, usecols=use_cols, low_memory=False, chunksize=chunk_size):
                if chunk.empty:
                    continue
                    
                chunk['Weighting_Strategy'] = chunk['Weighting_Strategy'].fillna('None')
                chunk['Reference_Strategy'] = chunk['Reference_Strategy'].fillna('None')
                chunk['Aggregation_Strategy'] = chunk['Aggregation_Strategy'].fillna('None')
                
                if 'Feature_Space' not in chunk.columns:
                    chunk['Feature_Space'] = 'Raw_Genes'

                chunk['Value'] = pd.to_numeric(chunk['Value'], errors='coerce')
                chunk = chunk.dropna(subset=['Value'])

                chunk_agg = chunk.groupby(local_group_cols)['Value'].agg(
                    File_Sum='sum', 
                    File_SqSum=lambda x: np.sum(x**2),
                    File_Count='count'
                ).reset_index()
                
                chunk_aggregations.append(chunk_agg)
                del chunk, chunk_agg
                gc.collect()
                
        except ValueError:
            use_cols.remove('Feature_Space')
            for chunk in pd.read_csv(file_path, usecols=use_cols, low_memory=False, chunksize=chunk_size):
                if chunk.empty:
                    continue
                
                chunk['Weighting_Strategy'] = chunk['Weighting_Strategy'].fillna('None')
                chunk['Reference_Strategy'] = chunk['Reference_Strategy'].fillna('None')
                chunk['Aggregation_Strategy'] = chunk['Aggregation_Strategy'].fillna('None')
                chunk['Feature_Space'] = 'Raw_Genes'

                chunk['Value'] = pd.to_numeric(chunk['Value'], errors='coerce')
                chunk = chunk.dropna(subset=['Value'])

                chunk_agg = chunk.groupby(local_group_cols)['Value'].agg(
                    File_Sum='sum', 
                    File_SqSum=lambda x: np.sum(x**2),
                    File_Count='count'
                ).reset_index()
                
                chunk_aggregations.append(chunk_agg)
                del chunk, chunk_agg
                gc.collect()

        if not chunk_aggregations:
            return None

        file_master = pd.concat(chunk_aggregations, ignore_index=True)
        final_file_agg = file_master.groupby(local_group_cols).agg(
            File_Sum=('File_Sum', 'sum'),
            File_SqSum=('File_SqSum', 'sum'),
            File_Count=('File_Count', 'sum')
        ).reset_index()

        del chunk_aggregations, file_master
        gc.collect()
        
        final_file_agg['Dataset'] = dataset_name
        final_file_agg['Variant'] = variant
        final_file_agg['Level'] = level
        final_file_agg['SNR'] = snr
        
        return final_file_agg

    except Exception as e:
        print(f"FAILED ON FILE: {file_path}. Error: {e}", file=sys.stderr)
        return None

def aggregate_all_datasets(results_root, save_dir, exclude_list=None):
    print(f"--- 1. Gathering all CSVs from {results_root} ---")
    
    search_pattern = os.path.join(results_root, "**", "performance_metrics_*.csv")
    all_files = glob.glob(search_pattern, recursive=True)
    
    if not all_files:
        print("No performance metric files found.")
        return

    if exclude_list:
        original_count = len(all_files)
        all_files = [f for f in all_files if not any(excl in f.split(os.sep) for excl in exclude_list)]
        print(f"Excluded {original_count - len(all_files)} files belonging to: {exclude_list}")

    print(f"Found {len(all_files)} CSV files to process. Loading and Pre-Aggregating...")
    collected_data = []
    
    num_cores = 16 
    print(f"Utilizing {num_cores} CPU cores...")

    try:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(tqdm(executor.map(process_single_file_chunked, all_files, chunksize=1), total=len(all_files), desc="Processing Files"))
            
            for res in results:
                if res is not None:
                    collected_data.append(res)
    except Exception as e:
        print(f"CRITICAL MULTIPROCESSING FAILURE: {e}")
        return

    print("\n--- 2. Building Master Dataframe ---")
    if not collected_data:
        print("No data was successfully loaded from any file.")
        return
        
    full_df = pd.concat(collected_data, ignore_index=True)

    print(f"\n--- 3. Calculating True Global Means and Saving Unique Files ---")
    os.makedirs(save_dir, exist_ok=True)
    
    metric_identifiers = ['Metric', 'Weighting_Strategy', 'Reference_Strategy', 'Aggregation_Strategy', 'Feature_Space']
    group_cols = ['Dataset', 'Variant', 'Level', 'SNR', 'Attack_Type']
    
    grouped_by_metric = full_df.groupby(metric_identifiers)
    
    file_count = 0
    for (metric, weight, ref, agg_strat, feat_space), subset_df in tqdm(grouped_by_metric, desc="Aggregating by Metric"):
        
        agg_df = subset_df.groupby(group_cols).agg(
            Total_Sum=('File_Sum', 'sum'),
            Total_SqSum=('File_SqSum', 'sum'),
            Total_Count=('File_Count', 'sum')
        ).reset_index()
        
        agg_df['Value'] = np.where(
            agg_df['Total_Count'] > 0,
            agg_df['Total_Sum'] / agg_df['Total_Count'],
            np.nan
        )

        variance = np.where(
            agg_df['Total_Count'] > 0,
            (agg_df['Total_SqSum'] / agg_df['Total_Count']) - (agg_df['Value'] ** 2),
            np.nan
        )
        agg_df['std'] = np.sqrt(np.maximum(variance, 0))
        
        agg_df = agg_df.drop(columns=['Total_Sum', 'Total_SqSum'])
        agg_df = agg_df.rename(columns={'Total_Count': 'count'})
        agg_df = agg_df.sort_values(by=['Dataset', 'Variant', 'Level', 'SNR', 'Attack_Type'])
        
        clean_metric = re.sub(r'[^\w\-]', '_', str(metric))
        clean_weight = re.sub(r'[^\w\-]', '_', str(weight))
        clean_ref = re.sub(r'[^\w\-]', '_', str(ref))
        clean_agg = re.sub(r'[^\w\-]', '_', str(agg_strat))
        clean_feat = re.sub(r'[^\w\-]', '_', str(feat_space))
        
        filename = f"{clean_metric}_{clean_weight}_{clean_ref}_{clean_agg}_{clean_feat}.csv"
        save_path = os.path.join(save_dir, filename)
        agg_df.to_csv(save_path, index=False)
        file_count += 1

    print(f"\nSuccess! Generated {file_count} unique aggregated CSV files in: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate virtual cell metrics.")
    parser.add_argument('--exclude', nargs='+', default=[], help="List of dataset folder names to exclude")
    args = parser.parse_args()

    results_dir = ("../../results/real")
    output_dir = ("../../analysis/overall_real")
    
    aggregate_all_datasets(results_dir, output_dir, exclude_list=args.exclude)