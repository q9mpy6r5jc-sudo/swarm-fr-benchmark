import os, textwrap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

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

def generate_overview_figure(robust_base_dir, faith_base_dir, save_dir):
    print("--- 1. Loading Aggregated Metric Data ---")
    
    robust_included_dir = os.path.join(robust_base_dir, '')
    faith_included_dir = os.path.join(faith_base_dir, '')
    print(f"Robust Included Dir = {robust_included_dir}")
    
    all_files = set()
    for d in [robust_base_dir, robust_included_dir]:
        if os.path.exists(d):
            all_files.update([f for f in os.listdir(d) if f.endswith('.csv')])
            
    master_data = []
    
    for filename in all_files:
        metric_name = filename.replace('.csv', '')
        is_literature = metric_name in LITERATURE_METRICS
        
        if is_literature:
            display_name = LITERATURE_METRICS[metric_name]
        else:
            display_name = format_novel_metric_name(metric_name)
        
        r_path = os.path.join(robust_included_dir, filename) if os.path.exists(os.path.join(robust_included_dir, filename)) else os.path.join(robust_base_dir, filename)
        f_path = os.path.join(faith_included_dir, filename) if os.path.exists(os.path.join(faith_included_dir, filename)) else os.path.join(faith_base_dir, filename)
        
        if not os.path.exists(f_path):
            continue
            
        try:
            rob_df = pd.read_csv(r_path)
            faith_df = pd.read_csv(f_path)
            
            mean_r = rob_df['Overall_Robustness'].mean()
            mean_f = faith_df['Fidelity_Score'].mean()
            
            master_data.append({
                'Metric_Name': metric_name,
                'Display_Name': display_name,
                'Mean_R': mean_r,
                'Mean_F': mean_f,
                'Is_Literature': is_literature
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    df_master = pd.DataFrame(master_data).fillna(0)
    print(f"Loaded data for {len(df_master)} metrics.")

    print("\n--- 2. Ranking the Top 10 Metrics ---")
    eps = 1e-9
    df_master['Combined_Score'] = (2.0 * df_master['Mean_R'] * df_master['Mean_F']) / (df_master['Mean_R'] + df_master['Mean_F'] + eps)
    df_master.loc[(df_master['Mean_R'] <= 0) | (df_master['Mean_F'] <= 0), 'Combined_Score'] = 0.0
    
    df_master = df_master.sort_values(by='Combined_Score', ascending=False).reset_index(drop=True)
    
    top_10 = df_master.head(5).copy()
    top_10_names = top_10['Metric_Name'].tolist()
    print(top_10_names)
    
    the_rest = df_master[~df_master['Metric_Name'].isin(top_10_names)].copy()
    the_rest_lit = the_rest[the_rest['Is_Literature']]
    the_rest_novel = the_rest[~the_rest['Is_Literature']]

    print("\n--- 3. Generating Overview Scatter Plot ---")
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    
    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    LIT_MARKER = '^'
    NOVEL_MARKER = 'o'

    if not the_rest_novel.empty:
        ax.scatter(the_rest_novel['Mean_R'], the_rest_novel['Mean_F'], 
                   marker=NOVEL_MARKER, color='lightgray', s=35, alpha=0.4, edgecolors='none', zorder=1)

    if not the_rest_lit.empty:
        ax.scatter(the_rest_lit['Mean_R'], the_rest_lit['Mean_F'], 
                   marker=LIT_MARKER, color='gray', s=45, alpha=0.7, edgecolors='white', linewidths=0.5, zorder=2)

    colors = sns.color_palette("tab10", len(top_10))
    legend_elements = []

    for idx, row in top_10.iterrows():
        marker = LIT_MARKER if row['Is_Literature'] else NOVEL_MARKER
        size = 80 if row['Is_Literature'] else 60
        
        ax.scatter(row['Mean_R'], row['Mean_F'], 
                   marker=marker, color=colors[idx], s=size, alpha=1.0, edgecolors='white', linewidths=0.8, zorder=3)
        
        wrapped_name = textwrap.fill(row['Display_Name'], width=35)
        label_text = f"{wrapped_name} (R:{row['Mean_R']:.2f}, F:{row['Mean_F']:.2f})"
        
        legend_elements.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor=colors[idx], 
                                      markersize=10, label=label_text))

    legend_elements.append(Line2D([0], [0], color='w', label=' '))
    legend_elements.append(Line2D([0], [0], marker=NOVEL_MARKER, color='w', markerfacecolor='black', markersize=10, label='SWARM Novel Metric Configuration'))
    legend_elements.append(Line2D([0], [0], marker=LIT_MARKER, color='w', markerfacecolor='black', markersize=10, label='Existing Literature Metric Configuration'))

    ax.set_xlabel('Robustness Score (R)', fontweight='normal', fontsize=13)
    ax.set_ylabel('Fidelity Score (F)', fontweight='normal', fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.03, 1), 
              frameon=False, fontsize=10.5, labelspacing=1.0)

    sns.despine()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path_pdf = os.path.join(save_dir, 'figure2_overview.pdf')
    save_path_png = os.path.join(save_dir, 'figure2_overview.png')
    
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Figure 2 to: {save_path_pdf}")

if __name__ == "__main__":
    robustness_directory = ("../../analysis/robustness_real")
    fidelity_directory = ("../../analysis/fidelity_real")
    output_directory = ("../../analysis/plots")
    
    generate_overview_figure(robustness_directory, fidelity_directory, output_directory)