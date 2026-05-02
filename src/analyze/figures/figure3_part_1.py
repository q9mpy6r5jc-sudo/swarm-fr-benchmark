import os, itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

LITERATURE_METRICS = {
    'MSE_Static_Specific_Perturbation_None_Raw_Genes': 'MSE',
    'MSE_DEG_Continuous_Specific_Perturbation_None_Raw_Genes': 'Weighted MSE',
    'RMSE_Static_Specific_Perturbation_None_Raw_Genes': 'RMSE',
    'RMSE_Top20_DEG_Binary_Specific_Perturbation_None_Raw_Genes': 'RMSE 20 DEGs',
    'Pearson_Static_Specific_Perturbation_None_Raw_Genes': 'Pearson (Naive)',
    'Pearson_Static_Control_Mean_Shift_None_Raw_Genes': 'Pearson ∆ Ctrl',
    'Pearson_Static_All_Perturbations_Mean_Shift_None_Raw_Genes': 'Pearson ∆ Pert',
    'Pearson_DEG_Binary_Control_Mean_Shift_None_Raw_Genes': 'Pearson ∆ Ctrl DEGs',
    'Pearson_DEG_Binary_All_Perturbations_Mean_Shift_None_Raw_Genes': 'Pearson ∆ Pert DEGs',
    'Pearson_Top20_DEG_Binary_Control_Mean_Shift_None_Raw_Genes': 'Pearson ∆ Ctrl 20 DEGs',
    'Pearson_Top20_DEG_Binary_All_Perturbations_Mean_Shift_None_Raw_Genes': 'Pearson ∆ Pert 20 DEGs',
    'R2_Static_Control_Mean_Shift_None_Raw_Genes': 'R² ∆ Ctrl',
    'R2_Static_All_Perturbations_Mean_Shift_None_Raw_Genes': 'R² ∆ Pert',
    'R2_DEG_Binary_Control_Mean_Shift_None_Raw_Genes': 'R² ∆ Ctrl DEGs',
    'R2_DEG_Binary_All_Perturbations_Mean_Shift_None_Raw_Genes': 'R² ∆ Pert DEGs',
    'R2_DEG_Continuous_Control_Mean_Shift_None_Raw_Genes': 'Weighted R² ∆ Ctrl',
    'R2_DEG_Continuous_All_Perturbations_Mean_Shift_None_Raw_Genes': 'Weighted R² ∆ Pert',
    'Cosine_Static_Control_Mean_Shift_None_Raw_Genes': 'Cosine LogFC',
    'Rank_Cosine_Static_Control_Mean_Shift_Distribution_Raw_Genes': 'Cosine LogFC Rank',
    'NIR_Static_Control_Mean_Shift_Distribution_Raw_Genes': 'NIR',
    'PDS_Static_Control_Mean_Shift_Distribution_Raw_Genes': 'PDS ∆ Ctrl',
    'Centroid_Accuracy_Static_Control_Mean_Shift_Distribution_Raw_Genes': 'Centroid Accuracy',
    'E_Distance_Static_Specific_Perturbation_Distribution_Raw_Genes': 'E-distance',
    'Wasserstein_Static_Specific_Perturbation_Distribution_Raw_Genes': 'Wasserstein Distance',
    'Sym_KL_Divergence_DEG_Binary_Specific_Perturbation_Distribution_Raw_Genes': 'KL Divergence',
    'MMD_None_Specific_Perturbation_Distribution_PCA_256': 'MMD PCA256',
    'MAE_Static_Specific_Perturbation_None_Raw_Genes': 'MAE',
    'DES_VCC_None_None_Distribution_Raw_Genes': 'DES'
}

def format_novel_metric_name(name):
    replacements = {
        "All_Perturbations_Mean_Shift": "Δ Pert",
        "Control_Mean_Shift": "Δ Ctrl",
        "Specific_Perturbation": "",
        "Distribution": "",
        "Raw_Genes": "",
        "Static": "",
        "None": ""
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    name = name.replace('_', ' ')
    return ' '.join(name.split())

def load_data(r_real_base, f_real_base, r_synth_base, f_synth_base):
    print("--- 1. Loading Base Data (Real + Synthetic) ---")
    
    r_real_inc = os.path.join(r_real_base, 'included') if os.path.exists(os.path.join(r_real_base, 'included')) else r_real_base
    f_real_inc = os.path.join(f_real_base, 'included') if os.path.exists(os.path.join(f_real_base, 'included')) else f_real_base
    
    r_synth_inc = os.path.join(r_synth_base, 'included') if os.path.exists(os.path.join(r_synth_base, 'included')) else r_synth_base
    f_synth_inc = os.path.join(f_synth_base, 'included') if os.path.exists(os.path.join(f_synth_base, 'included')) else f_synth_base
    
    swarm_files = [f for f in os.listdir(r_real_inc) if f.endswith('.csv')]
    swarm_data, lit_data = [], []
    
    def process_file(filename, is_literature=False, display_name=None):
        try:
            r_real_path = os.path.join(r_real_inc, filename) if os.path.exists(os.path.join(r_real_inc, filename)) else os.path.join(r_real_base, filename)
            f_real_path = os.path.join(f_real_inc, filename) if os.path.exists(os.path.join(f_real_inc, filename)) else os.path.join(f_real_base, filename)
            
            r_synth_path = os.path.join(r_synth_inc, filename) if os.path.exists(os.path.join(r_synth_inc, filename)) else os.path.join(r_synth_base, filename)
            f_synth_path = os.path.join(f_synth_inc, filename) if os.path.exists(os.path.join(f_synth_inc, filename)) else os.path.join(f_synth_base, filename)

            if not os.path.exists(r_real_path) or not os.path.exists(f_real_path):
                return None
                
            r_real_df = pd.read_csv(r_real_path)
            f_real_df = pd.read_csv(f_real_path)
            
            r_synth_df = pd.read_csv(r_synth_path) if os.path.exists(r_synth_path) else pd.DataFrame()
            f_synth_df = pd.read_csv(f_synth_path) if os.path.exists(f_synth_path) else pd.DataFrame()
            
            r_df = pd.concat([r_real_df, r_synth_df], ignore_index=True)
            f_df = pd.concat([f_real_df, f_synth_df], ignore_index=True)

            metric_raw = filename.replace('.csv', '')
            metric_name = display_name if is_literature else format_novel_metric_name(metric_raw)
            
            # Extract independent Structure and Magnitude dictionaries
            struct_cols = [c for c in f_df.columns if 'Struct_Rho' in c]
            mag_cols = [c for c in f_df.columns if 'Mag_Rho' in c]
            
            f_struct_scores = {}
            f_mag_scores = {}
            for _, row in f_df.iterrows():
                ds = row['Dataset']
                f_struct_scores[ds] = np.mean([row[c] for c in struct_cols]) if struct_cols else 0.0
                f_mag_scores[ds] = np.mean([row[c] for c in mag_cols]) if mag_cols else 0.0
            
            r_scores = {}
            for ac in [c for c in r_df.columns if c.startswith('Rob_')]:
                r_scores[ac] = r_df.set_index('Dataset')[ac].to_dict()
                
            return {
                'Metric': metric_name, 'Metric_Raw': metric_raw, 
                'F_Struct_Scores': f_struct_scores, 'F_Mag_Scores': f_mag_scores, 
                'R_Scores': r_scores
            }
        except Exception:
            return None

    for f in swarm_files:
        record = process_file(f, is_literature=False)
        if record: swarm_data.append(record)
            
    for raw_name, display_name in LITERATURE_METRICS.items():
        record = process_file(f"{raw_name}.csv", is_literature=True, display_name=display_name)
        if record: lit_data.append(record)

    all_records = swarm_data + lit_data
    global_datasets = sorted(list(set().union(*(r['F_Struct_Scores'].keys() for r in all_records))))

    def finalize_scores(r):
        struct_vals = [r['F_Struct_Scores'].get(ds, 0.0) for ds in global_datasets]
        mag_vals = [r['F_Mag_Scores'].get(ds, 0.0) for ds in global_datasets]
        
        mean_struct = np.mean(struct_vals) if struct_vals else 0.0
        mean_mag = np.mean(mag_vals) if mag_vals else 0.0
        
        r['Mean_F'] = (mean_struct + mean_mag) / 2.0

        attack_means = []
        for ac, ac_dict in r['R_Scores'].items():
            r_vals = [ac_dict.get(ds, 0.0) for ds in global_datasets]
            attack_means.append(np.mean(r_vals))
            
        r['Mean_R'] = np.min(attack_means) if attack_means else 0.0
        return r

    swarm_data = [finalize_scores(r) for r in swarm_data]
    lit_data = [finalize_scores(r) for r in lit_data]

    def get_diverse_pool(data_list, pool_size=20):
        if not data_list: return pd.DataFrame()
        df = pd.DataFrame(data_list)
        df['Harmonic'] = (2 * df['Mean_F'] * df['Mean_R']) / (df['Mean_F'] + df['Mean_R'] + 1e-9)
        
        k = pool_size // 3
        top_f = df.sort_values('Mean_F', ascending=False).head(k)
        top_r = df.sort_values('Mean_R', ascending=False).head(k)
        top_h = df.sort_values('Harmonic', ascending=False).head(k)
        
        pool = pd.concat([top_f, top_r, top_h]).drop_duplicates(subset=['Metric_Raw'])
        
        if len(pool) < pool_size:
            rem = df[~df['Metric_Raw'].isin(pool['Metric_Raw'])].sort_values('Harmonic', ascending=False)
            pool = pd.concat([pool, rem.head(pool_size - len(pool))])
            
        return pool.reset_index(drop=True)

    df_swarm = get_diverse_pool(swarm_data, 20)
    df_lit = get_diverse_pool(lit_data, 20)
    
    return df_swarm, df_lit

def evaluate_vectorized_ensembles(df, sizes=[1, 2, 3, 4, 5], top_n=10):
    all_datasets = sorted(list(set().union(*(d.keys() for d in df['F_Struct_Scores']))))
    attack_cols = sorted(list(set().union(*(d.keys() for d in df['R_Scores']))))
    
    metrics_list = df['Metric'].tolist()
    metrics_raw_list = df['Metric_Raw'].tolist()
    
    F_struct_matrix = np.zeros((len(df), len(all_datasets)))
    F_mag_matrix = np.zeros((len(df), len(all_datasets)))
    R_tensor = np.zeros((len(df), len(all_datasets), len(attack_cols)))
    
    for i, row in df.iterrows():
        for j, ds in enumerate(all_datasets):
            F_struct_matrix[i, j] = row['F_Struct_Scores'].get(ds, 0.0)
            F_mag_matrix[i, j] = row['F_Mag_Scores'].get(ds, 0.0)
            for a_idx, ac in enumerate(attack_cols):
                R_tensor[i, j, a_idx] = row['R_Scores'].get(ac, {}).get(ds, 0.0)
            
    results = []
    for k in sizes:
        indices = np.array(list(itertools.combinations(range(len(df)), k)))
        
        if k == 1:
            F_combined = (F_struct_matrix + F_mag_matrix) / 2.0
            F_port = np.mean(F_combined, axis=1)
            R_port = np.min(np.mean(R_tensor, axis=1), axis=1)
        else:
            # Symmetrical optimization: Maximize Structure and Magnitude independently
            F_struct_port = np.max(F_struct_matrix[indices], axis=1)
            F_mag_port = np.max(F_mag_matrix[indices], axis=1)
            
            F_combined = (F_struct_port + F_mag_port) / 2.0
            F_port = np.mean(F_combined, axis=1)
            
            R_port = np.min(np.mean(np.max(R_tensor[indices], axis=1), axis=1), axis=1)
            
        combined = (2 * F_port * R_port) / (F_port + R_port + 1e-9)
        
        top_indices = np.argsort(combined)[::-1][:top_n]
        
        for rank, idx in enumerate(top_indices, 1):
            combo_metrics = [metrics_list[i] for i in indices[idx]] if k > 1 else [metrics_list[indices[idx][0]]]
            combo_metrics_raw = [metrics_raw_list[i] for i in indices[idx]] if k > 1 else [metrics_raw_list[indices[idx][0]]]
            results.append({
                'Size': k,
                'Rank': rank,
                'Combined_Score': combined[idx],
                'F_Score': F_port[idx],
                'R_Score': R_port[idx],
                'Metrics': " + ".join(combo_metrics),
                'Metrics_Raw': " + ".join(combo_metrics_raw)
            })
            
    return pd.DataFrame(results)

def plot_plume_trajectories(df, save_path, parsimony_threshold=0.01):
    print("\nGenerating Plume Plot...")
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    sizes = sorted(df['Size'].unique())
    max_scores = {}
    
    for setup, color, marker in zip(['SWARM', 'Literature'], ['#1f77b4', '#ff7f0e'], ['o', 's']):
        setup_df = df[df['Setup'] == setup]
        rank1, min3, min5, min10 = [], [], [], []
        for s in sizes:
            size_df = setup_df[setup_df['Size'] == s]
            rank1.append(size_df[size_df['Rank'] == 1]['Combined_Score'].values[0] if not size_df[size_df['Rank'] == 1].empty else 0)
            min3.append(size_df[size_df['Rank'] <= 3]['Combined_Score'].min() if not size_df[size_df['Rank'] <= 3].empty else 0)
            min5.append(size_df[size_df['Rank'] <= 5]['Combined_Score'].min() if not size_df[size_df['Rank'] <= 5].empty else 0)
            min10.append(size_df[size_df['Rank'] <= 10]['Combined_Score'].min() if not size_df[size_df['Rank'] <= 10].empty else 0)
            
        ax.fill_between(sizes, min5, min10, color=color, alpha=0.15, linewidth=0)
        ax.fill_between(sizes, min3, min5, color=color, alpha=0.30, linewidth=0)
        ax.fill_between(sizes, rank1, min3, color=color, alpha=0.45, linewidth=0)
        
        line_style = '-' if setup == 'SWARM' else '--'
        ax.plot(sizes, rank1, color=color, linewidth=3, marker=marker, markersize=8, linestyle=line_style)

        absolute_max = max(rank1)
        optimal_size = min([s for s, r in zip(sizes, rank1) if r >= absolute_max - parsimony_threshold])
        optimal_score = rank1[sizes.index(optimal_size)]
        max_scores[setup] = {'size': optimal_size, 'score': optimal_score}
        ax.plot(optimal_size, optimal_score, marker='*', color=color, markeredgecolor='black', markeredgewidth=1.2, markersize=18, linestyle='None', zorder=5)

    ax.set_xticks(sizes)
    ax.set_xlabel('M-Set Size (Number of Metrics Combined)', fontweight='normal')
    ax.set_ylabel('M-Set Score', fontweight='normal')
    ax.set_ylim(0, 1.15) 
    
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', lw=3, marker='o', markersize=8, label='SWARM Rank 1'),
        Line2D([0], [0], color='#ff7f0e', lw=3, marker='s', markersize=8, linestyle='--', label='Literature Rank 1'),
        Patch(facecolor='#1f77b4', alpha=0.35, label='SWARM Top 10 Density'),
        Patch(facecolor='#ff7f0e', alpha=0.35, label='Literature Top 10 Density'),
        Line2D([0], [0], color='w', marker='*', markerfacecolor='#1f77b4', markeredgecolor='black', markersize=14, label=f"SWARM Optimal (t={max_scores['SWARM']['size']})"),
        Line2D([0], [0], color='w', marker='*', markerfacecolor='#ff7f0e', markeredgecolor='black', markersize=14, label=f"Literature Optimal (t={max_scores['Literature']['size']})")
    ]
    ax.legend(handles=legend_elements, loc='upper center', frameon=True, fontsize=10, ncol=2)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved Plume Plot to: {save_path}")

def export_detailed_top_sets(base_df, res_df, setup_name, peak_size, output_dir):
    print(f"\n--- Exporting Detailed Report for {setup_name} (Optimal Size: {peak_size}) ---")
    top_5 = res_df[(res_df['Size'] == peak_size) & (res_df['Rank'] <= 5)].copy()
    
    results = []
    for idx, row in top_5.iterrows():
        metrics_in_combo = row['Metrics'].split(" + ")
        raw_metrics_in_combo = row['Metrics_Raw'].split(" + ")
        combo_df = base_df[base_df['Metric'].isin(metrics_in_combo)]
        combo_dicts = combo_df.to_dict('records')
        
        all_datasets = sorted(list(set().union(*(d['F_Struct_Scores'].keys() for d in combo_dicts))))
        attack_cols = sorted(list(set().union(*(d['R_Scores'].keys() for d in combo_dicts))))
        
        contributions = {m: [] for m in metrics_in_combo}
        port_rob_data = {}
        
        # --- Robustness Attribution ---
        for ac in attack_cols:
            ds_defenses = [max(m['R_Scores'].get(ac, {}).get(ds, 0) for m in combo_dicts) for ds in all_datasets]
            best_defense = np.mean(ds_defenses)
            
            attack_scores = {m['Metric']: np.mean([m['R_Scores'].get(ac, {}).get(ds, 0) for ds in all_datasets]) for m in combo_dicts}
            port_rob_data[ac] = {'best_defense': best_defense, 'individual_scores': attack_scores}
            
            best_indiv_score = max(attack_scores.values()) if attack_scores else 0
            for m_name, score in attack_scores.items():
                if abs(score - best_indiv_score) < 1e-6:
                    clean_attack_name = ac.replace('Rob_', '').replace('_', ' ')
                    contributions[m_name].append(clean_attack_name)
                    
        # --- Fidelity Attribution ---
        struct_scores = {m['Metric']: np.mean([m['F_Struct_Scores'].get(ds, 0) for ds in all_datasets]) for m in combo_dicts}
        mag_scores = {m['Metric']: np.mean([m['F_Mag_Scores'].get(ds, 0) for ds in all_datasets]) for m in combo_dicts}
        
        best_struct_score = max(struct_scores.values()) if struct_scores else 0
        best_mag_score = max(mag_scores.values()) if mag_scores else 0
        
        for m_name in metrics_in_combo:
            if abs(struct_scores[m_name] - best_struct_score) < 1e-6:
                contributions[m_name].append("High Fidelity to Structure")
            if abs(mag_scores[m_name] - best_mag_score) < 1e-6:
                contributions[m_name].append("High Fidelity to Magnitude")
                    
        worst_case_rob = row['R_Score']
        worst_attacks = [ac for ac, data in port_rob_data.items() if abs(data['best_defense'] - worst_case_rob) < 1e-6]
        bottleneck_attack = worst_attacks[0] if worst_attacks else "Unknown"
        clean_bottleneck = bottleneck_attack.replace('Rob_', '').replace('_', ' ')
        
        bottleneck_scores = [f"{m['Metric']}: {port_rob_data.get(bottleneck_attack, {}).get('individual_scores', {}).get(m['Metric'], 0):.2f}" for m in combo_dicts]
        bottleneck_str = f"{clean_bottleneck} (" + ", ".join(bottleneck_scores) + ")"
        
        record = {
            'Metrics': row['Metrics'],
            'Metrics_Raw': row['Metrics_Raw'],
            'Portfolio_F': row['F_Score'],
            'Worst_Case_R': row['R_Score'],
            'Combined_Score': row['Combined_Score'],
            'Worst_Case_Breakdown': bottleneck_str
        }
        
        for i, (m_name, m_raw_name) in enumerate(zip(metrics_in_combo, raw_metrics_in_combo)):
            record[f'Metric_{i+1}'] = m_name
            record[f'Metric_{i+1}_Raw'] = m_raw_name
            covered_attacks = contributions[m_name]
            if covered_attacks:
                record[f'Metric_{i+1}_Reason'] = "Selected for: " + ", ".join(list(set(covered_attacks)))
            else:
                record[f'Metric_{i+1}_Reason'] = "Selected for minor orthogonal dataset coverage"
                
        results.append(record)
        
    df_results = pd.DataFrame(results)
    
    print("=========================================================")
    print(f"      TOP 5 {setup_name.upper()} METRIC SETS (SIZE = {peak_size})")
    print("=========================================================")
    for idx, row in df_results.iterrows():
        print(f"\nRANK {idx+1} | Score: {row['Combined_Score']:.3f} (F: {row['Portfolio_F']:.3f}, R: {row['Worst_Case_R']:.3f})")
        for i in range(1, peak_size + 1):
            print(f"  {i}. {row[f'Metric_{i}']} -> {row[f'Metric_{i}_Reason']}")

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'metric_sets.csv' if setup_name == 'SWARM' else 'metric_sets_literature.csv')
    df_results.to_csv(save_path, index=False)

def generate_latex_tables(df_all, save_dir):
    tex_path = os.path.join(save_dir, 'supp_table_metric_sets.tex')
    with open(tex_path, 'w') as f:
        f.write("% --- GENERATED LATEX TABLE ---\n")
        f.write("\\begin{table*}[h!]\n\\centering\n")
        f.write("\\caption{Top 5 Metric Set Combinations across ensemble sizes.}\n")
        f.write("\\label{tab:metric_sets_trajectories}\n\\small\n")
        f.write("\\begin{tabularx}{\\textwidth}{@{} l c c c c >{\\raggedright\\arraybackslash}X @{}}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Setup} & \\textbf{Size} & \\textbf{Rank} & \\textbf{Score} & \\textbf{F / R} & \\textbf{Included Metrics} \\\\\n")
        f.write("\\midrule\n")
        for setup in ['SWARM', 'Literature']:
            setup_df = df_all[(df_all['Setup'] == setup) & (df_all['Rank'] <= 5)]
            for size in sorted(setup_df['Size'].unique()):
                size_df = setup_df[setup_df['Size'] == size].sort_values(by='Rank')
                for i, (_, row) in enumerate(size_df.iterrows()):
                    setup_str = f"\\textbf{{{setup}}}" if i == 0 else ""
                    size_str = str(size) if i == 0 else ""
                    f.write(f"{setup_str} & {size_str} & {row['Rank']} & {row['Combined_Score']:.3f} & {row['F_Score']:.2f} / {row['R_Score']:.2f} & {row['Metrics'].replace('&', '\\&').replace('%', '\\%')} \\\\\n")
                f.write("\\addlinespace\n")
            f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabularx}\n\\end{table*}\n")

if __name__ == "__main__":
    r_real = ("../../analysis/robustness_real")
    f_real = ("../../analysis/fidelity_real")
    r_synth = ("../../analysis/robustness_synthetic")
    f_synth = ("../../analysis/fidelity_synthetic")
    
    plots_directory = ("../../analysis/plots")
    tables_directory = ("../../analysis/tables")
    os.makedirs(plots_directory, exist_ok=True); os.makedirs(tables_directory, exist_ok=True)
    
    df_swarm_base, df_lit_base = load_data(r_real, f_real, r_synth, f_synth)
    
    print("\n--- Evaluating Combinations ---")
    res_swarm = evaluate_vectorized_ensembles(df_swarm_base, sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], top_n=10)
    res_swarm['Setup'] = 'SWARM'
    
    res_lit = evaluate_vectorized_ensembles(df_lit_base, sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], top_n=10)
    res_lit['Setup'] = 'Literature'
    
    df_combined = pd.concat([res_swarm, res_lit], ignore_index=True)
    df_combined.to_csv(os.path.join(tables_directory, 'metric_sets_overall_trajectories.csv'), index=False)
    
    PARSIMONY_THRESHOLD = 0.01
    def get_optimal_size(df_setup):
        r1 = df_setup[df_setup['Rank'] == 1]
        valid_sizes = r1[r1['Combined_Score'] >= (r1['Combined_Score'].max() - PARSIMONY_THRESHOLD)]
        if valid_sizes.empty: return r1['Size'].min()
        return valid_sizes['Size'].min()
    
    peak_swarm = get_optimal_size(res_swarm)
    export_detailed_top_sets(df_swarm_base, res_swarm, 'SWARM', peak_swarm, tables_directory)
    
    peak_lit = get_optimal_size(res_lit)
    export_detailed_top_sets(df_lit_base, res_lit, 'Literature', peak_lit, tables_directory)

    plot_plume_trajectories(df_combined, os.path.join(plots_directory, 'figure3a_metric_sets_plume.pdf'))
    generate_latex_tables(df_combined, tables_directory)