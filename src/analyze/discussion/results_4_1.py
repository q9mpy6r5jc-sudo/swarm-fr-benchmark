import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CORE_TO_CLASS = {
    'MSE': 'Spatial', 'MAE': 'Spatial', 'RMSE': 'Spatial',
    'Pearson': 'Spatial', 'Cosine': 'Spatial', 'R2': 'Spatial',
    'Wasserstein': 'Distributional', 'E_Distance': 'Distributional', 
    'Sym_KL_Divergence': 'Distributional', 'MMD': 'Distributional',
    'PDS': 'Retrieval/Rank', 'NIR': 'Retrieval/Rank', 'Centroid_Accuracy': 'Retrieval/Rank',
    'Rank_Cosine': 'Retrieval/Rank', 'Rank_Pearson': 'Retrieval/Rank', 'DES_VCC': 'Diff. Expression',
    'DES_Robust': 'Diff. Expression'
}

CLASS_COLORS = {
    'Retrieval/Rank': '#9467bd',   # Purple
    'Distributional': '#2ca02c',   # Green
    'Spatial': '#d62728',          # Red
    'Diff. Expression': '#ff7f0e', # Orange
    'Unknown': '#333333'           # Dark Gray
}

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

def calculate_swarm_improvements(robust_base, faith_base):
    print("\n--- 3. Calculating SWARM vs Literature Improvements ---")
    robust_included = os.path.join(robust_base, 'included')
    faith_included = os.path.join(faith_base, 'included')
    
    all_metrics_data = []
    
    all_files = glob.glob(os.path.join(robust_included, "*.csv")) + glob.glob(os.path.join(robust_base, "*.csv"))
    unique_files = set([os.path.basename(f) for f in all_files])
    
    for filename in unique_files:
        r_path = os.path.join(robust_included, filename)
        if not os.path.exists(r_path): r_path = os.path.join(robust_base, filename)
            
        f_path = os.path.join(faith_included, filename)
        if not os.path.exists(f_path): f_path = os.path.join(faith_base, filename)
            
        if not os.path.exists(r_path) or not os.path.exists(f_path):
            continue
            
        try:
            rob_df = pd.read_csv(r_path)
            faith_df = pd.read_csv(f_path)
            
            f_score = faith_df['Fidelity_Score'].mean()
            r_score = rob_df['Overall_Robustness'].mean()
            
            h_score = (2 * f_score * r_score) / (f_score + r_score + 1e-9)
            raw_metric_name = filename.replace('.csv', '')
            
            all_metrics_data.append({
                'Metric': raw_metric_name,
                'Fidelity': f_score,
                'Robustness': r_score,
                'Harmonic': h_score,
                'Is_Literature': raw_metric_name in LITERATURE_METRICS
            })
        except Exception:
            continue

    df_all = pd.DataFrame(all_metrics_data)
    
    if df_all.empty:
        print("No data found for comparison.")
        return

    df_lit = df_all[df_all['Is_Literature'] == True]
    lit_f_mean = df_lit['Fidelity'].mean()
    lit_r_mean = df_lit['Robustness'].mean()
    
    n_samples = len(LITERATURE_METRICS)
    df_swarm = df_all.sort_values(by='Harmonic', ascending=False).head(n_samples)
    swarm_f_mean = df_swarm['Fidelity'].mean()
    swarm_r_mean = df_swarm['Robustness'].mean()
    
    f_improvement = ((swarm_f_mean - lit_f_mean) / lit_f_mean) * 100
    r_improvement = ((swarm_r_mean - lit_r_mean) / lit_r_mean) * 100
    
    print(f"Literature (n={len(df_lit)}): Mean Fidelity = {lit_f_mean:.3f}, Mean Robustness = {lit_r_mean:.3f}")
    print(f"Top SWARM  (n={len(df_swarm)}): Mean Fidelity = {swarm_f_mean:.3f}, Mean Robustness = {swarm_r_mean:.3f}")
    print("=======================================================")
    print(f"Fidelity Improvement [X%]:   {f_improvement:.1f}%")
    print(f"Robustness Improvement [Y%]: {r_improvement:.1f}%")
    print("=======================================================\n")

def generate_vulnerability_matrix(robust_base, faith_base, tables_dir, plots_dir):
    print("--- 1. Extracting Literature Vulnerability Data ---")
    
    robust_included = os.path.join(robust_base, 'included')
    faith_included = os.path.join(faith_base, 'included')
    
    records = []
    
    for raw_name, display_name in LITERATURE_METRICS.items():
        filename = f"{raw_name}.csv"
        
        r_path = os.path.join(robust_included, filename)
        if not os.path.exists(r_path): r_path = os.path.join(robust_base, filename)
            
        f_path = os.path.join(faith_included, filename)
        if not os.path.exists(f_path): f_path = os.path.join(faith_base, filename)
            
        if not os.path.exists(r_path) or not os.path.exists(f_path):
            continue
            
        try:
            rob_df = pd.read_csv(r_path)
            faith_df = pd.read_csv(f_path)
            
            rec = {
                'Metric': display_name,
                'Fidelity Score (F)': faith_df['Fidelity_Score'].mean(),
                'Robustness Score (R)': rob_df['Overall_Robustness'].mean()
            }
            
            attack_cols = [c for c in rob_df.columns if c.startswith('Rob_')]
            for ac in attack_cols:
                clean_attack_name = ac.replace('Rob_', '').replace('_', ' ')
                rec[clean_attack_name] = rob_df[ac].mean()
                
            records.append(rec)
        except Exception as e:
            print(f"Error processing {display_name}: {e}")

    df = pd.DataFrame(records)
    
    if df.empty:
        print("No data found!")
        return

    print("\n=======")
    print("   STATS")
    print("=======")
    print(f"Robustness [A, B]:  [{df['Robustness Score (R)'].min():.3f}, {df['Robustness Score (R)'].max():.3f}]")
    print(f"Robustness Median:  {df['Robustness Score (R)'].median():.3f}")
    print(f"Fidelity [C, D]:    [{df['Fidelity Score (F)'].min():.3f}, {df['Fidelity Score (F)'].max():.3f}]")
    print(f"Fidelity Median:    {df['Fidelity Score (F)'].median():.3f}")
    
    os.makedirs(tables_dir, exist_ok=True)
    csv_path = os.path.join(tables_dir, 'supp_literature_vulnerability_matrix.csv')
    df = df.sort_values(by='Metric', ascending=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved Data Table: {csv_path}")

    print("\n--- 2. Generating Dual-Axis Heatmap Visualization ---")
    os.makedirs(plots_dir, exist_ok=True)
    
    attack_cols = [c for c in df.columns if c not in ['Metric', 'Fidelity Score (F)', 'Robustness Score (R)']]
    
    fig, (ax_f, ax_r) = plt.subplots(
        ncols=2, 
        figsize=(12, 12), 
        gridspec_kw={'width_ratios': [1, len(attack_cols)], 'wspace': 0.08}
    )
    sns.set_theme(style="white", font_scale=1.1)
    
    sns.heatmap(
        df.set_index('Metric')[['Fidelity Score (F)']], 
        annot=True, fmt=".2f", cmap="Blues", 
        vmin=0.0, vmax=0.5, 
        cbar=False,
        ax=ax_f, linewidths=.5
    )
    ax_f.set_ylabel('')
    ax_f.xaxis.tick_top()
    plt.setp(ax_f.get_xticklabels(), rotation=45, ha='left', fontweight='normal')
    
    sns.heatmap(
        df.set_index('Metric')[attack_cols], 
        annot=True, fmt=".2f", cmap="RdYlGn", 
        vmin=0.0, vmax=1.0, 
        cbar_kws={"shrink": .6, "label": "Robustness Score (R)"}, 
        ax=ax_r, linewidths=.5
    )
    ax_r.set_ylabel('')
    ax_r.set_yticks([])
    ax_r.xaxis.tick_top()
    plt.setp(ax_r.get_xticklabels(), rotation=45, ha='left', fontweight='normal')
    
    ax_r.set_xlabel('Mathematical Exploits', fontweight='normal', labelpad=20, fontsize=12)
    ax_r.xaxis.set_label_position('bottom')
    
    save_pdf = os.path.join(plots_dir, 'supp_literature_robustness_heatmap.pdf')
    save_png = os.path.join(plots_dir, 'supp_literature_robustness_heatmap.png')
    
    plt.savefig(save_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Heatmap Figure: {save_pdf}")

def generate_swarm_vulnerability_matrix(robust_base, faith_base, tables_dir, plots_dir):
    print("\n--- 4. Extracting SWARM-Exclusive Vulnerability Data ---")
    
    robust_included = os.path.join(robust_base, 'included')
    faith_included = os.path.join(faith_base, 'included')
    
    all_files = glob.glob(os.path.join(robust_included, "*.csv")) + glob.glob(os.path.join(robust_base, "*.csv"))
    unique_files = set([os.path.basename(f) for f in all_files])
    
    records = []
    
    for filename in unique_files:
        raw_name = filename.replace('.csv', '')
        
        if raw_name in LITERATURE_METRICS:
            continue
            
        r_path = os.path.join(robust_included, filename)
        if not os.path.exists(r_path): r_path = os.path.join(robust_base, filename)
            
        f_path = os.path.join(faith_included, filename)
        if not os.path.exists(f_path): f_path = os.path.join(faith_base, filename)
            
        if not os.path.exists(r_path) or not os.path.exists(f_path):
            continue
            
        try:
            rob_df = pd.read_csv(r_path)
            faith_df = pd.read_csv(f_path)
            
            display_name = format_novel_metric_name(raw_name)
            
            rec = {
                'Metric': display_name,
                'Fidelity Score (F)': faith_df['Fidelity_Score'].mean(),
                'Robustness Score (R)': rob_df['Overall_Robustness'].mean()
            }
            
            attack_cols = [c for c in rob_df.columns if c.startswith('Rob_')]
            for ac in attack_cols:
                clean_attack_name = ac.replace('Rob_', '').replace('_', ' ')
                rec[clean_attack_name] = rob_df[ac].mean()
                
            records.append(rec)
        except Exception:
            continue

    df = pd.DataFrame(records)
    
    if df.empty:
        print("No SWARM-exclusive data found!")
        return

    df = df.sort_values(by='Metric', ascending=True)
    
    os.makedirs(tables_dir, exist_ok=True)
    csv_path = os.path.join(tables_dir, 'supp_swarm_vulnerability_matrix.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved SWARM Data Table: {csv_path}")

    print("\n--- 5. Generating SPLIT Dual-Axis Heatmap Visualizations ---")
    os.makedirs(plots_dir, exist_ok=True)
    
    attack_cols = [c for c in df.columns if c not in ['Metric', 'Fidelity Score (F)', 'Robustness Score (R)']]
    
    max_rows_per_plot = 55
    num_chunks = int(np.ceil(len(df) / max_rows_per_plot))
    
    for i in range(num_chunks):
        chunk_df = df.iloc[i * max_rows_per_plot : (i + 1) * max_rows_per_plot]
        fig_height = (len(chunk_df) * 0.25) + 2
        
        fig, (ax_f, ax_r) = plt.subplots(
            ncols=2, 
            figsize=(16, fig_height), 
            gridspec_kw={'width_ratios': [1, len(attack_cols)], 'wspace': 0.05}
        )
        sns.set_theme(style="white", font_scale=1.0)
        
        sns.heatmap(
            chunk_df.set_index('Metric')[['Fidelity Score (F)']], 
            annot=True, fmt=".2f", cmap="Blues", 
            vmin=0.0, vmax=0.5, 
            cbar=False,
            annot_kws={"size": 9}, 
            ax=ax_f, linewidths=.5
        )
        ax_f.set_ylabel('')
        ax_f.xaxis.tick_top()
        plt.setp(ax_f.get_xticklabels(), rotation=45, ha='left', fontweight='normal')
        
        sns.heatmap(
            chunk_df.set_index('Metric')[attack_cols], 
            annot=True, fmt=".2f", cmap="RdYlGn", 
            vmin=0.0, vmax=1.0, 
            cbar_kws={"shrink": 0.5, "label": "Robustness Score (R)"},
            annot_kws={"size": 9},
            ax=ax_r, linewidths=.5
        )
        ax_r.set_ylabel('')
        ax_r.set_yticks([])
        ax_r.xaxis.tick_top()
        plt.setp(ax_r.get_xticklabels(), rotation=45, ha='left', fontweight='normal')
        
        ax_r.set_xlabel('Mathematical Exploits', fontweight='normal', labelpad=20, fontsize=12)
        ax_r.xaxis.set_label_position('bottom')
        
        save_pdf = os.path.join(plots_dir, f'supp_swarm_robustness_heatmap_part{i+1}.pdf')
        save_png = os.path.join(plots_dir, f'supp_swarm_robustness_heatmap_part{i+1}.png')
        
        plt.savefig(save_pdf, format='pdf', bbox_inches='tight')
        plt.savefig(save_png, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved Split Heatmap Figure ({i+1}/{num_chunks}): {save_pdf}")

def generate_dichotomy_plot(robust_base, plots_dir, vuln_threshold=0.01):
    print("\n--- 6. Generating Exploit Dichotomy Scatter Plot ---")
    robust_included = os.path.join(robust_base, 'included')
    all_files = glob.glob(os.path.join(robust_included, "*.csv")) + glob.glob(os.path.join(robust_base, "*.csv"))
    unique_files = set([os.path.basename(f) for f in all_files])

    records = []
    gamed_by_all = []
    gamed_by_mc_not_norm = []

    for filename in unique_files:
        r_path = os.path.join(robust_included, filename)
        if not os.path.exists(r_path): r_path = os.path.join(robust_base, filename)

        try:
            rob_df = pd.read_csv(r_path)
            
            mc_cols = [c for c in rob_df.columns if 'Mode_Collapse' in c]
            norm_cols = [c for c in rob_df.columns if 'Normalization' in c]
            mc_norm_cols = mc_cols + norm_cols
            scale_cols = [c for c in rob_df.columns if 'Scaling' in c]
            
            if not mc_norm_cols or not scale_cols: continue

            val_mc_norm = rob_df[mc_norm_cols].mean().min()
            val_scale = rob_df[scale_cols].mean().min()

            max_vuln_mc = 1.0 - rob_df[mc_cols].mean().min() if mc_cols else 0
            max_vuln_norm = 1.0 - rob_df[norm_cols].mean().min() if norm_cols else 0
            max_vuln_scale = 1.0 - rob_df[scale_cols].mean().min() if scale_cols else 0

            raw_name = filename.replace('.csv', '')
            
            if max_vuln_mc > vuln_threshold and max_vuln_norm > vuln_threshold and max_vuln_scale > vuln_threshold:
                gamed_by_all.append(raw_name)
                
            if max_vuln_mc > vuln_threshold and max_vuln_norm <= vuln_threshold:
                gamed_by_mc_not_norm.append(raw_name)

            is_rank = any(x in raw_name for x in ['PDS', 'NIR', 'Centroid', 'Rank_'])
            m_type = 'Retrieval/Rank-Based' if is_rank else 'Spatial, Dist., &\n Diff. Expression'

            records.append({
                'Metric': raw_name,
                'Robustness Score (R) for Mode Collapse & Norm. Mismatch': val_mc_norm,
                'Robustness Score (R) for Scaling Exploit': val_scale,
                'Metric Family': m_type
            })
        except Exception:
            continue

    df = pd.DataFrame(records)
    if df.empty: return
    
    print("\n--- Vulnerability Insights ---")
    print(f"Metrics vulnerable to ALL exploits (MC, Norm, and Scaling):")
    if gamed_by_all:
        for m in sorted(gamed_by_all): print(f"  - {m}")
    else:
        print("  (None found)")
        
    print(f"\nMetrics vulnerable to Mode Collapse but NOT Normalization Mismatch:")
    if gamed_by_mc_not_norm:
        for m in sorted(gamed_by_mc_not_norm): print(f"  - {m}")
    else:
        print("  (None found)")
    print("------------------------------\n")

    os.makedirs(plots_dir, exist_ok=True)
    
    sns.set_theme(style="ticks", context="paper", font_scale=1.6)

    color_map = {
        'Spatial, Dist., &\n Diff. Expression': '#1f77b4',
        'Retrieval/Rank-Based': '#ff7f0e'
    }

    g = sns.jointplot(
        data=df,
        x='Robustness Score (R) for Mode Collapse & Norm. Mismatch',
        y='Robustness Score (R) for Scaling Exploit',
        hue='Metric Family',
        palette=color_map,
        alpha=0.7,
        height=4,
        s=70,
        marginal_kws={'fill': True, 'common_norm': False}
    )

    g.ax_joint.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    g.ax_joint.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

    g.ax_joint.set_xlim(-0.05, 1.05)
    g.ax_joint.set_ylim(-0.05, 1.05)

    g.ax_joint.set_xticks([0.0, 0.5, 1.0])
    g.ax_joint.set_yticks([0.0, 0.5, 1.0])

    g.ax_joint.set_xlabel('R Score for Mode Collapse\n & Norm. Mismatch', fontsize=16)
    g.ax_joint.set_ylabel('R Score for \nScaling Exploit', fontsize=16)
    
    g.ax_joint.tick_params(axis='both', which='major', labelsize=14)

    g.ax_joint.legend(title='', title_fontsize=15, fontsize=16, loc='lower center', 
                      bbox_to_anchor=(0.5, 1.25), frameon=False, ncol=1)

    save_pdf = os.path.join(plots_dir, 'figure4_vulnerability_dichotomy.pdf')
    save_png = os.path.join(plots_dir, 'figure4_vulnerability_dichotomy.png')
    plt.savefig(save_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Dichotomy Plot: {save_pdf}")

def generate_fidelity_dichotomy(faith_base, plots_dir):
    print("\n--- 9. Generating Fidelity Dichotomy ---")
    faith_included = os.path.join(faith_base, 'included')
    all_files = glob.glob(os.path.join(faith_included, "*.csv")) + glob.glob(os.path.join(faith_base, "*.csv"))
    unique_files = set([os.path.basename(f) for f in all_files])

    records = []

    def get_class(metric_name):
        for cm in sorted(CORE_TO_CLASS.keys(), key=len, reverse=True):
            if metric_name.startswith(cm):
                if 'DES_Robust' in metric_name:
                    return 'Diff. Expression'
                return CORE_TO_CLASS[cm]
        return 'Unknown'

    for filename in unique_files:
        f_path = os.path.join(faith_included, filename)
        if not os.path.exists(f_path): f_path = os.path.join(faith_base, filename)

        try:
            faith_df = pd.read_csv(f_path)
            
            mag_cols = [c for c in faith_df.columns if 'Mag_Rho' in c]
            str_cols = [c for c in faith_df.columns if 'Struct_Rho' in c]
            
            if not mag_cols or not str_cols:
                continue

            val_mag = faith_df[mag_cols].mean().mean() + np.random.uniform(-0.015, 0.015)
            val_str = faith_df[str_cols].mean().mean() + np.random.uniform(-0.015, 0.015)

            raw_name = filename.replace('.csv', '')
            m_class = get_class(raw_name)

            records.append({
                'Metric': raw_name,
                'Fidelity to Signal Magnitude': val_mag,
                'Fidelity to Signal Structure': val_str,
                'Metric Class': m_class
            })
        except Exception:
            continue

    df = pd.DataFrame(records)
    if df.empty: 
        print("Could not find data to plot Temporary Fidelity Dichotomy.")
        return

    os.makedirs(plots_dir, exist_ok=True)
    
    sns.set_theme(style="ticks", context="paper", font_scale=1.6)

    color_map = {k: v for k, v in CLASS_COLORS.items() if k != 'Unknown'}

    g = sns.jointplot(
        data=df,
        x='Fidelity to Signal Magnitude',
        y='Fidelity to Signal Structure',
        hue='Metric Class',
        palette=color_map,
        alpha=0.7,
        height=5,
        s=50,
        marginal_kws={'fill': True, 'common_norm': False}
    )

    g.ax_joint.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    g.ax_joint.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

    max_x = df['Fidelity to Signal Magnitude'].max()
    max_y = df['Fidelity to Signal Structure'].max()
    
    g.ax_joint.set_xlim(-0.05, max(1.05, max_x + 0.05))
    g.ax_joint.set_ylim(-0.05, max(1.05, max_y + 0.05))

    g.ax_joint.set_xlabel('Fidelity to Signal Magnitude', fontsize=14)
    g.ax_joint.set_ylabel('Fidelity to Signal Structure', fontsize=14)
    
    g.ax_joint.tick_params(axis='both', which='major', labelsize=12)

    g.ax_joint.legend(title='Metric Class', title_fontsize=12, fontsize=11, 
                      bbox_to_anchor=(1.25, 1.0), loc='upper left', frameon=True)

    save_pdf = os.path.join(plots_dir, 'supp_temporary_fidelity_classes.pdf')
    save_png = os.path.join(plots_dir, 'supp_temporary_fidelity_classes.png')
    plt.savefig(save_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Temporary Fidelity Classes Plot: {save_pdf}")

def generate_full_leaderboard_latex(robust_base, faith_base, tables_dir):
    print("\n--- 7. Generating Full Leaderboard LaTeX Tables ---")
    robust_included = os.path.join(robust_base, 'included')
    faith_included = os.path.join(faith_base, 'included')
    
    all_files = glob.glob(os.path.join(robust_included, "*.csv")) + glob.glob(os.path.join(robust_base, "*.csv"))
    unique_files = set([os.path.basename(f) for f in all_files])
    
    records = []
    for filename in unique_files:
        r_path = os.path.join(robust_included, filename)
        if not os.path.exists(r_path): r_path = os.path.join(robust_base, filename)
            
        f_path = os.path.join(faith_included, filename)
        if not os.path.exists(f_path): f_path = os.path.join(faith_base, filename)
            
        if not os.path.exists(r_path) or not os.path.exists(f_path):
            continue
            
        try:
            rob_df = pd.read_csv(r_path)
            faith_df = pd.read_csv(f_path)
            
            f_score = faith_df['Fidelity_Score'].mean()
            r_score = rob_df['Overall_Robustness'].mean()
            h_score = (2 * f_score * r_score) / (f_score + r_score + 1e-9)
            
            raw_name = filename.replace('.csv', '')
            if raw_name in LITERATURE_METRICS:
                display_name = f"{raw_name} (Literature)"
            else:
                display_name = raw_name
                
            records.append({
                'Metric': display_name,
                'Fidelity Score': f_score,
                'Robustness Score': r_score,
                'Harmonic': h_score
            })
        except Exception:
            continue

    df = pd.DataFrame(records)
    if df.empty:
        print("No data found for leaderboard.")
        return

    # Sort by Harmonic score first to establish the definitive global rank
    df = df.sort_values(by='Harmonic', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1

    os.makedirs(tables_dir, exist_ok=True)

    def write_longtable(dataframe, filepath, caption, label):
        with open(filepath, 'w') as f:
            f.write("% --- GENERATED LATEX LONGTABLE ---\n")
            f.write("\\begin{longtable}{@{} c >{\\raggedright\\arraybackslash}p{0.55\\textwidth} c c @{}}\n")
            f.write(f"\\caption{{{caption}}}\\\\\n")
            f.write(f"\\label{{{label}}}\\\\\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Rank} & \\textbf{Metric Name} & \\textbf{Fidelity} & \\textbf{Robustness} \\\\\n")
            f.write("\\midrule\n")
            f.write("\\endfirsthead\n")
            f.write("\\multicolumn{4}{c}%\n")
            f.write("{{\\bfseries \\tablename\\ \\thetable{} -- continued from previous page}} \\\\\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Rank} & \\textbf{Metric Name} & \\textbf{Fidelity} & \\textbf{Robustness} \\\\\n")
            f.write("\\midrule\n")
            f.write("\\endhead\n")
            f.write("\\midrule\n")
            f.write("\\multicolumn{4}{r}{{Continued on next page}} \\\\\n")
            f.write("\\endfoot\n")
            f.write("\\bottomrule\n")
            f.write("\\endlastfoot\n")
            
            for _, row in dataframe.iterrows():
                metric_safe = str(row['Metric']).replace('&', '\\&').replace('%', '\\%').replace('_', '\\_\\allowbreak{}')
                f.write(f"{row['Rank']} & {metric_safe} & {row['Fidelity Score']:.3f} & {row['Robustness Score']:.3f} \\\\\n")
            
            f.write("\\end{longtable}\n")

    df_harmonic = df.copy()
    p1 = os.path.join(tables_dir, 'supp_leaderboard_harmonic.tex')
    write_longtable(df_harmonic, p1, 
                "Full Evaluation Leaderboard of SWARM Metrics (Sorted by Harmonic Mean of Fidelity and Robustness).", 
                "tab:full_leaderboard_harmonic")
    print(f"Saved Harmonic Leaderboard to: {p1}")

    df_alpha = df.sort_values(by='Metric', ascending=True)
    p2 = os.path.join(tables_dir, 'supp_leaderboard_alphabetical.tex')
    write_longtable(df_alpha, p2, 
                "Full Evaluation Leaderboard of SWARM Metrics (Sorted Alphabetically).", 
                "tab:full_leaderboard_alphabetical")
    print(f"Saved Alphabetical Leaderboard to: {p2}")


if __name__ == "__main__":
    robustness_base = ("../../analysis/robustness_real")
    fidelity_base = ("../../analysis/fidelity_real")
    tables_directory = ("../../analysis/tables")
    plots_directory = ("../../analysis/plots")
    
    generate_vulnerability_matrix(robustness_base, fidelity_base, tables_directory, plots_directory)
    calculate_swarm_improvements(robustness_base, fidelity_base)
    generate_swarm_vulnerability_matrix(robustness_base, fidelity_base, tables_directory, plots_directory)
    generate_dichotomy_plot(robustness_base, plots_directory)
    generate_full_leaderboard_latex(robustness_base, fidelity_base, tables_directory)
    generate_fidelity_dichotomy(fidelity_base, plots_directory)