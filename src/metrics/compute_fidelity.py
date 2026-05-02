import pandas as pd
import numpy as np
import os, glob
from scipy.stats import kendalltau
from tqdm import tqdm

HIB_METRICS = ['Pearson', 'R2', 'PDS', 'NIR', 'Centroid_Accuracy', 'DES_Robust', 'DES_VCC', 'Cosine', 'Rank_Cosine', 'Rank_Pearson']
LIB_METRICS = ['MSE', 'MAE', 'Wasserstein', 'Sym_KL_Divergence', 'E_Distance', 'MMD', 'RMSE']

def compute_fidelity(input_dir, output_dir):
    print(f"--- Scanning for aggregated CSVs in {input_dir} ---")
    
    search_pattern = os.path.join(input_dir, "*.csv")
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        print("No CSV files found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(all_files)} files. Computing Fidelity scores...")
    
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
            df = pd.read_csv(file_path, dtype={'SNR': str})
            df = df[df['Attack_Type'] == 'No_Attack'].copy()
            if df.empty:
                continue

            df['SNR'] = df['SNR'].astype(str).str.strip().str.lower()
            df['SNR'] = df['SNR'].replace({'inf': 'infinity', '1': '1.0', '1.00': '1.0'})
            
            dataset_records = []
            
            for dataset, d_data in df.groupby('Dataset'):
                
                # Helper function to extract aligned Levels and Scores for a specific sweep
                def get_sweep_data(variant_type, target_snr):
                    subset = d_data[(d_data['Variant'] == variant_type) & (d_data['SNR'] == target_snr)]
                    if subset.empty: return [], []
                    
                    # Sort strictly by Level (1 through 5)
                    subset = subset.sort_values('Level')
                    return subset['Level'].tolist(), subset['Value'].tolist()

                s_lvl_inf, s_val_inf = get_sweep_data('Structure', 'infinity')
                s_lvl_1, s_val_1 = get_sweep_data('Structure', '1.0')
                s_lvl_01, s_val_01 = get_sweep_data('Structure', '0.1')
                s_lvl_001, s_val_001 = get_sweep_data('Structure', '0.01')
                
                m_lvl_inf, m_val_inf = get_sweep_data('Magnitude', 'infinity')
                m_lvl_1, m_val_1 = get_sweep_data('Magnitude', '1.0')
                m_lvl_01, m_val_01 = get_sweep_data('Magnitude', '0.1')
                m_lvl_001, m_val_001 = get_sweep_data('Magnitude', '0.01')

                # A dataset is only complete if ALL 8 vectors have exactly 5 points
                if (len(s_val_inf) != 5 or len(s_val_1) != 5 or 
                    len(s_val_01) != 5 or len(s_val_001) != 5 or
                    len(m_val_inf) != 5 or len(m_val_1) != 5 or 
                    len(m_val_01) != 5 or len(m_val_001) != 5):
                    continue
                
                def calculate_correctness_kendall(levels, scores):
                    tau, _ = kendalltau(levels, scores)
                    if np.isnan(tau): tau = 0.0
                    
                    monotonicity = -tau if is_hib else tau
                    start_val = scores[0]
                    epsilon = 1e-5
                    
                    smapes = []
                    # Compare baseline (Level 1) against all degraded levels
                    for i in range(1, len(scores)):
                        current_val = scores[i]
                        step_smape = abs(current_val - start_val) / (abs(start_val) + abs(current_val) + epsilon)
                        smapes.append(step_smape)
                    
                    magnitude_factor = np.mean(smapes) if smapes else 0.0
                    combined_score = monotonicity * magnitude_factor
                    
                    return max(0.0, combined_score)
                
                s_rho_inf = calculate_correctness_kendall(s_lvl_inf, s_val_inf)
                s_rho_1 = calculate_correctness_kendall(s_lvl_1, s_val_1)
                s_rho_01 = calculate_correctness_kendall(s_lvl_01, s_val_01)
                s_rho_001 = calculate_correctness_kendall(s_lvl_001, s_val_001)
                
                m_rho_inf = calculate_correctness_kendall(m_lvl_inf, m_val_inf)
                m_rho_1 = calculate_correctness_kendall(m_lvl_1, m_val_1)
                m_rho_01 = calculate_correctness_kendall(m_lvl_01, m_val_01)
                m_rho_001 = calculate_correctness_kendall(m_lvl_001, m_val_001)
                
                struct_rhos = [s_rho_inf, s_rho_1, s_rho_01, s_rho_001]
                mag_rhos = [m_rho_inf, m_rho_1, m_rho_01, m_rho_001]
                
                struct_average_rho = np.average(struct_rhos)
                mag_average_rho = np.average(mag_rhos)
                
                struct_std = np.std(struct_rhos)
                mag_std = np.std(mag_rhos)
                
                if struct_average_rho == 0.0 or mag_average_rho == 0.0:
                    fidelity = 0.0
                    fidelity_std = 0.0
                else:
                    fidelity = (2.0 * struct_average_rho * mag_average_rho) / (struct_average_rho + mag_average_rho)
                    fidelity_std = np.sqrt(0.5 * (struct_std**2 + mag_std**2))
                
                dataset_records.append({
                    'Dataset': dataset,
                    'Struct_Rho_Inf': s_rho_inf,
                    'Struct_Rho_SNR1': s_rho_1,
                    'Struct_Rho_SNR01': s_rho_01,
                    'Struct_Rho_SNR001': s_rho_001,
                    'Mag_Rho_Inf': m_rho_inf,
                    'Mag_Rho_SNR1': m_rho_1,
                    'Mag_Rho_SNR01': m_rho_01,
                    'Mag_Rho_SNR001': m_rho_001,
                    'Fidelity_Score': fidelity,
                    'Fidelity_Std': fidelity_std
                })
            
            if dataset_records:
                res_df = pd.DataFrame(dataset_records)
                save_path = os.path.join(output_dir, filename)
                res_df.to_csv(save_path, index=False)
                processed_count += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nSuccess! Computed Fidelity for {processed_count} metric files.")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    overall_dir = ("../../analysis/overall_synthetic")
    fidelity_dir = ("../../analysis/fidelity_synthetic")
    
    compute_fidelity(overall_dir, fidelity_dir)