import os, glob, ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DEGS = { 
    'frangieh21': 0, 'genespider2': 1, 'kaden25rpe1': 1, 'kaden25fibroblast': 1, 'tian21crispri': 1,
    'replogle22k562gwps': 4, 'sunshine23': 6, 'tian21crispra': 7, 'nadig25jurkat': 10,
    'nadig25hepg2': 13, 'wessels23': 18, 'replogle22rpe1': 34, 'replogle22k562': 37,
    'norman19': 104, 'adamson16': 132, 'vcc2025': 827
}

def format_novel_metric_name(name):
    for old, new in {"All_Perturbations_Mean_Shift": "Δ Pert", "Control_Mean_Shift": "Δ Ctrl", "Specific_Perturbation": "", "Distribution": "", "Raw_Genes": "", "Static": "", "None": ""}.items():
        name = name.replace(old, new)
    return ' '.join(name.replace('_', ' ').split())

def get_portfolio_from_trajectories(csv_path, setup, size=None, get_peak=False, threshold=0.01):
    df = pd.read_csv(csv_path)
    setup_df = df[(df['Setup'] == setup) & (df['Rank'] == 1)]
    if get_peak:
        max_score = setup_df['Combined_Score'].max()
        valid = setup_df[setup_df['Combined_Score'] >= max_score - threshold]
        target_row = valid.loc[valid['Size'].idxmin()]
    else:
        target_row = setup_df[setup_df['Size'] == size].iloc[0]
        
    m_str = str(target_row['Metrics_Raw'])
    metrics = ast.literal_eval(m_str) if m_str.startswith('[') else [m.strip() for m in m_str.split('+')]
    return metrics, target_row['Size'], target_row['F_Score'], target_row['R_Score']

def load_portfolio_data(portfolio_files, robust_dir_real, faith_dir_real, robust_dir_synth, faith_dir_synth):
    dataframes = []
    def load_df(base_dir, metric_name):
        path_inc = os.path.join(base_dir, 'included', f"{metric_name}.csv")
        path_base = os.path.join(base_dir, f"{metric_name}.csv")
        if os.path.exists(path_inc): return pd.read_csv(path_inc)
        if os.path.exists(path_base): return pd.read_csv(path_base)
        return pd.DataFrame()

    for m in portfolio_files:
        r_real = load_df(robust_dir_real, m)
        f_real = load_df(faith_dir_real, m)
        r_synth = load_df(robust_dir_synth, m)
        f_synth = load_df(faith_dir_synth, m)
        r_df = pd.concat([r_real, r_synth], ignore_index=True)
        f_df = pd.concat([f_real, f_synth], ignore_index=True)
        if r_df.empty or f_df.empty: continue
        
        # Extract Structure and Magnitude columns alongside Fidelity_Score
        f_cols = ['Dataset', 'Fidelity_Score'] + [c for c in f_df.columns if 'Struct_Rho' in c or 'Mag_Rho' in c]
        f_cols = list(set(f_cols).intersection(f_df.columns))
        dataframes.append(pd.merge(r_df, f_df[f_cols], on='Dataset'))
        
    return dataframes

def calculate_portfolio_score_per_dataset(dataframes):
    if not dataframes: return pd.DataFrame()
        
    datasets = sorted(list(set().union(*(df['Dataset'].unique() for df in dataframes))))
    attack_cols = sorted(list(set().union(*(df.columns[df.columns.str.startswith('Rob_')] for df in dataframes))))
    
    # Identify Structure and Magnitude columns
    struct_cols = [c for c in dataframes[0].columns if 'Struct_Rho' in c]
    mag_cols = [c for c in dataframes[0].columns if 'Mag_Rho' in c]
    
    results = []
    for d in datasets:
        d_rows = [df[df['Dataset'] == d].iloc[0] for df in dataframes if not df[df['Dataset'] == d].empty]
        if not d_rows: continue
        
        # Fidelity optimization
        if struct_cols and mag_cols:
            best_struct = max([np.mean([row.get(c, 0) for c in struct_cols]) for row in d_rows])
            best_mag = max([np.mean([row.get(c, 0) for c in mag_cols]) for row in d_rows])
            port_f = (best_struct + best_mag) / 2.0
        else:
            port_f = np.max([row['Fidelity_Score'] for row in d_rows])
            
        port_rob_attacks = [max([row.get(ac, 0) for row in d_rows]) for ac in attack_cols]
        worst_case_rob = min(port_rob_attacks) if port_rob_attacks else 0
        
        combined = (2 * port_f * worst_case_rob) / (port_f + worst_case_rob + 1e-9)
        results.append({'Dataset': d, 'Score': combined})
        
    return pd.DataFrame(results)

def get_leaderboard_baselines(r_real, f_real):
    robust_included = os.path.join(r_real, 'included')
    faith_included = os.path.join(f_real, 'included')
    
    all_files = glob.glob(os.path.join(robust_included, "*.csv")) + glob.glob(os.path.join(r_real, "*.csv"))
    unique_files = set([os.path.basename(f) for f in all_files])
    
    records = []
    for filename in unique_files:
        r_path = os.path.join(robust_included, filename)
        if not os.path.exists(r_path): r_path = os.path.join(r_real, filename)
        f_path = os.path.join(faith_included, filename)
        if not os.path.exists(f_path): f_path = os.path.join(f_real, filename)
            
        if not os.path.exists(r_path) or not os.path.exists(f_path): continue
            
        try:
            r_df = pd.read_csv(r_path)
            f_df = pd.read_csv(f_path)
            f_val = f_df['Fidelity_Score'].mean()
            r_val = r_df['Overall_Robustness'].mean()
            h_val = (2 * f_val * r_val) / (f_val + r_val + 1e-9)
            
            records.append({'Metric': filename.replace('.csv',''), 'F': f_val, 'R': r_val, 'H': h_val})
        except Exception:
            continue
            
    df = pd.DataFrame(records)
    top_overall = df.loc[df['H'].idxmax()]
    top_f = df.sort_values(by=['F', 'R'], ascending=[False, False]).iloc[0]
    top_r = df.sort_values(by=['R', 'F'], ascending=[False, False]).iloc[0]
    
    return (
        [top_overall['Metric']], top_overall['Metric'], top_overall['F'], top_overall['R'],
        [top_f['Metric']], top_f['Metric'], top_f['F'], top_f['R'],
        [top_r['Metric']], top_r['Metric'], top_r['F'], top_r['R']
    )

def generate_figure3(r_real, f_real, r_synth, f_synth, tables_dir, save_dir):
    print("--- 1. Reading Target Portfolios ---")
    traj_csv = os.path.join(tables_dir, 'metric_sets_overall_trajectories.csv')
    
    lit_metrics, lit_size, lit_f, lit_r = get_portfolio_from_trajectories(traj_csv, 'Literature', get_peak=True)
    swarm_metrics, swarm_size, swarm_f, swarm_r = get_portfolio_from_trajectories(traj_csv, 'SWARM', get_peak=True)
    swarm_match_metrics, _, swarm_match_f, swarm_match_r = get_portfolio_from_trajectories(traj_csv, 'SWARM', size=lit_size)
    
    (s_metrics, s_name, s_f, s_r,
     f_metrics, f_name, f_f, f_r,
     r_metrics, r_name, r_f, r_r) = get_leaderboard_baselines(r_real, f_real)

    print("\n--- 2. Calculating Dataset-Level Scores ---")
    plot_df = pd.DataFrame(list(DATASET_DEGS.items()), columns=['Dataset', 'Mean_DEGs']).sort_values('Mean_DEGs').reset_index(drop=True)
    
    for metrics, col_name in [
        (swarm_metrics, 'SWARM_Peak'), (swarm_match_metrics, 'SWARM_Matched'), (lit_metrics, 'Lit_Peak'),
        (s_metrics, 'Single_Metric'), (f_metrics, 'Max_F_Metric'), (r_metrics, 'Max_R_Metric')
    ]:
        dfs = load_portfolio_data(metrics, r_real, f_real, r_synth, f_synth)
        scores = calculate_portfolio_score_per_dataset(dfs) 
        plot_df = plot_df.merge(scores.rename(columns={'Score': col_name}), on='Dataset', how='left')

    print("\n--- 3. Generating Plot ---")
    sns.set_theme(style="ticks", context="paper", font_scale=1.6)
    fig, ax1 = plt.subplots(figsize=(10, 4)) 
    
    ax2 = ax1.twinx()
    x_pos = np.arange(len(plot_df))
    ax2.bar(x_pos, plot_df['Mean_DEGs'], color='lightgray', alpha=0.5, label='Mean DEGs (n)')
    ax2.set_ylabel('Mean DEGs (n)', fontweight='normal', color='gray', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=14)

    def prep_name(n):
        clean_name = format_novel_metric_name(n)
        words = clean_name.split(' ')
        if len(words) > 1:
            return ' '.join(words[:2]) + '\n' + ' '.join(words[2:])
        return clean_name
    
    label_swarm_peak = f'Top SWARM Set (k={int(swarm_size)})\nMin R: {swarm_r:.2f}, Max F: {swarm_f:.2f}'
    label_swarm_match = f'Top SWARM Set (k={int(lit_size)})\nMin R: {swarm_match_r:.2f}, Max F: {swarm_match_f:.2f}'
    label_lit_peak = f'Top Literature Set (k={int(lit_size)})\nMin R: {lit_r:.2f}, Max F: {lit_f:.2f}'
    
    label_single = f'Leaderboard #1 ({prep_name(s_name)})\nR: {s_r:.2f}, F: {s_f:.2f}'
    label_max_f = f'Max Fidelity ({prep_name(f_name)})\nR: {f_r:.2f}, F: {f_f:.2f}'
    label_max_r = f'Max Robustness ({prep_name(r_name)})\nR: {r_r:.2f}, F: {r_f:.2f}'

    ax1.plot(x_pos, plot_df['SWARM_Peak'], marker='o', markersize=9, linestyle='none', color='#ff7f0e', zorder=10, label=label_swarm_peak) 
    ax1.plot(x_pos, plot_df['SWARM_Matched'], marker='^', markersize=8, linestyle='none', color='#1f77b4', zorder=9, label=label_swarm_match)
    ax1.plot(x_pos, plot_df['Lit_Peak'], marker='s', markersize=7, linestyle='none', color='#555555', alpha=0.8, zorder=5, label=label_lit_peak)
    ax1.plot(x_pos, plot_df['Single_Metric'], marker='D', markersize=6, linestyle='none', color='#777777', alpha=0.8, zorder=4, label=label_single)
    ax1.plot(x_pos, plot_df['Max_F_Metric'], marker='v', markersize=6, linestyle='none', color='#999999', alpha=0.8, zorder=3, label=label_max_f)
    ax1.plot(x_pos, plot_df['Max_R_Metric'], marker='P', markersize=6, linestyle='none', color='#bbbbbb', alpha=0.8, zorder=2, label=label_max_r)

    ax1.set_ylabel('M-Set Score', fontweight='normal', fontsize=16)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{row['Dataset']}" for _, row in plot_df.iterrows()], rotation=45, ha='right', fontsize=11)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.set_xlabel('Datasets ranked by mean DEGs (n)', fontweight='normal', fontsize=16)
    
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc='lower center', bbox_to_anchor=(0.5, 1.04), frameon=False, fontsize=16, ncol=2)
    sns.despine(right=False)
    
    os.makedirs(save_dir, exist_ok=True)
    save_pdf = os.path.join(save_dir, 'figure3b_metric_sets_across_datasets.pdf')
    save_png = os.path.join(save_dir, 'figure3b_metric_sets_across_datasets.png')
    plt.savefig(save_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2b to: {save_pdf}")

if __name__ == "__main__":
    r_real = ("../../analysis/robustness_real")
    f_real = ("../../analysis/fidelity_real")
    r_synth = ("../../analysis/robustness_synthetic")
    f_synth = ("../../analysis/fidelity_synthetic")
    tables_directory = ("../../analysis/tables")
    output_directory = ("../../analysis/plots")
    generate_figure3(r_real, f_real, r_synth, f_synth, tables_directory, output_directory)