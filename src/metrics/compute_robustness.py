import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

HIB_METRICS = ['Pearson', 'R2', 'PDS', 'NIR', 'Centroid_Accuracy', 'DES_Robust', 'DES_VCC', 'Cosine', 'Rank_Cosine', 'Rank_Pearson']
LIB_METRICS = ['MSE', 'MAE', 'Wasserstein', 'Sym_KL_Divergence', 'E_Distance', 'MMD', 'RMSE']
HACK_THRESHOLD = 0.01

def compute_robustness(input_dir, output_dir):
    search_pattern = os.path.join(input_dir, "*.csv")
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        print("No CSV files found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(all_files)} files. Computing Robustness scores...")
    processed_count = 0
    
    for file_path in tqdm(all_files):
        filename = os.path.basename(file_path)
        
        is_hib = None
        for m in HIB_METRICS:
            if filename.startswith(m):
                is_hib = True
                break
        if is_hib is None:
            for m in LIB_METRICS:
                if filename.startswith(m):
                    is_hib = False
                    break
                    
        if is_hib is None:
            continue
            
        try:
            df = pd.read_csv(file_path, low_memory=False, dtype={'SNR': str})
            
            if 'SNR' in df.columns:
                df['SNR'] = df['SNR'].astype(str).str.strip().str.lower()
                df['SNR'] = df['SNR'].replace({'inf': 'infinity', '1': '1.0', '1.00': '1.0'})
            else:
                df['SNR'] = 'infinity'
            
            dataset_records = []
            
            for dataset, d_data in df.groupby('Dataset'):
                
                struct_data = d_data[d_data['Variant'] == 'Structure']
                mag_data = d_data[d_data['Variant'] == 'Magnitude']
                
                struct_envs = struct_data[['Level', 'SNR']].drop_duplicates()
                mag_envs = mag_data[['Level', 'SNR']].drop_duplicates()
                
                if len(struct_envs) < 20 or len(mag_envs) < 20:
                    continue
                
                gt_row = d_data[(d_data['Variant'] == 'GT_Baseline') & (d_data['Attack_Type'] == 'No_Attack')]
                if gt_row.empty:
                    continue
                gt_score = gt_row['Value'].values[0]

                baseline_df = d_data[d_data['Attack_Type'] == 'No_Attack'].copy()
                baseline_map = baseline_df.set_index(['Variant', 'Level', 'SNR'])['Value'].to_dict()

                attack_df = d_data[(d_data['Attack_Type'] != 'No_Attack') & (d_data['Variant'] != 'GT_Baseline')].copy()
                
                if attack_df.empty:
                    continue

                attack_df['Score_GT'] = gt_score
                attack_df['Score_Baseline'] = attack_df.apply(
                    lambda row: baseline_map.get((row['Variant'], row['Level'], row['SNR']), np.nan), axis=1
                )
                
                attack_df = attack_df.dropna(subset=['Score_Baseline'])

                if is_hib:
                    attack_df['Deficit'] = attack_df['Score_GT'] - attack_df['Score_Baseline']
                    attack_df['Gain'] = attack_df['Value'] - attack_df['Score_Baseline']
                else:
                    attack_df['Deficit'] = attack_df['Score_Baseline'] - attack_df['Score_GT']
                    attack_df['Gain'] = attack_df['Score_Baseline'] - attack_df['Value']

                attack_df = attack_df[attack_df['Deficit'] > 1e-6].copy()
                
                if attack_df.empty:
                    continue

                # Calculate absolute False Gain
                attack_df['False_Gain'] = np.maximum(0, attack_df['Gain'])
                
                # Calculate Relative Gain for every single row immediately.
                # This normalizes the gain by the total deficit, so that it's scale-invariant.
                attack_df['Relative_Gain'] = attack_df['False_Gain'] / (attack_df['Deficit'] + 1e-9)
                attack_df['Relative_Gain'] = np.clip(attack_df['Relative_Gain'], 0.0, 1.0)

                # Minimax
                unique_attacks = attack_df['Attack_Type'].unique()
                total_attacks = len(unique_attacks)
                gamed_attacks_count = 0
                
                attack_scores = {}

                for attack in unique_attacks:
                    attack_subset = attack_df[attack_df['Attack_Type'] == attack]
                    
                    if (attack_subset['Relative_Gain'] > HACK_THRESHOLD).any():
                        gamed_attacks_count += 1

                        successful_hacks = attack_subset[attack_subset['Relative_Gain'] > HACK_THRESHOLD]
                        mean_severity = successful_hacks['Relative_Gain'].mean()
                        
                        attack_scores[f"Rob_{attack}"] = 1.0 - mean_severity
                    else:
                        attack_scores[f"Rob_{attack}"] = 1.0

                # OVERALL ROBUSTNESS: 1.0 - max(severity)
                if attack_scores:
                    final_robustness = min(attack_scores.values())
                else:
                    final_robustness = 1.0

                final_robustness = max(0.0, min(1.0, final_robustness))

                record = {
                    'Dataset': dataset,
                    'Overall_Robustness': final_robustness,
                    'Gamed_Count': gamed_attacks_count,
                    'Total_Attacks': total_attacks
                }
                record.update(attack_scores)
                
                dataset_records.append(record)
            
            if dataset_records:
                res_df = pd.DataFrame(dataset_records)
                save_path = os.path.join(output_dir, filename)
                res_df.to_csv(save_path, index=False)
                processed_count += 1
            
        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    print(f"\nSuccess! Computed Robustness for {processed_count} metric files.")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    overall_dir = ("../../analysis/overall_synthetic")
    robustness_dir = ("../../analysis/robustness_synthetic")
    
    compute_robustness(overall_dir, robustness_dir)