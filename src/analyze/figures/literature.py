import os, itertools, textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    'PDS_Static_Control_Mean_Shift_Distribution_Raw_Genes': 'PDS',
    'Centroid_Accuracy_Static_Control_Mean_Shift_Distribution_Raw_Genes': 'Centroid Accuracy',
    'E_Distance_Static_Specific_Perturbation_Distribution_Raw_Genes': 'E-distance',
    'Wasserstein_Static_Specific_Perturbation_Distribution_Raw_Genes': 'Wasserstein Distance',
    'Sym_KL_Divergence_DEG_Binary_Specific_Perturbation_Distribution_Raw_Genes': 'KL Divergence',
    'MMD_None_Specific_Perturbation_Distribution_PCA_256': 'MMD PCA256',
    'MAE_Static_Specific_Perturbation_None_Raw_Genes': 'MAE',
    'DES_VCC_None_None_Distribution_Raw_Genes': 'DES'
}

def generate_figure1_original(robust_base_dir, faith_base_dir, save_dir):
    print(f"--- 1. Loading Aggregated Metric Data ---")
    
    robust_included_dir = os.path.join(robust_base_dir, 'included')
    faith_included_dir = os.path.join(faith_base_dir, 'included')
    
    all_robust_files = [f for f in os.listdir(robust_base_dir) if f.endswith('.csv')]
    master_data = []
    
    for filename in all_robust_files:
        metric_filename = filename.replace('.csv', '')
        is_literature = metric_filename in LITERATURE_METRICS
        
        if not is_literature and not os.path.exists(os.path.join(robust_included_dir, filename)):
            continue
            
        if os.path.exists(os.path.join(robust_included_dir, filename)):
            robust_path = os.path.join(robust_included_dir, filename)
            faith_path = os.path.join(faith_included_dir, filename)
        else:
            robust_path = os.path.join(robust_base_dir, filename)
            faith_path = os.path.join(faith_base_dir, filename)
            
        if not os.path.exists(faith_path):
            continue
            
        try:
            display_name = LITERATURE_METRICS.get(metric_filename, metric_filename.replace('_', ' '))

            rob_df = pd.read_csv(robust_path)
            faith_df = pd.read_csv(faith_path)
            
            mean_r = rob_df['Overall_Robustness'].mean()
            std_r = rob_df['Overall_Robustness'].std()
            cons_r = np.maximum(0, mean_r - std_r)
            
            mean_f = faith_df['Fidelity_Score'].mean()
            std_f = faith_df['Fidelity_Score'].std()
            cons_f = np.maximum(0, mean_f - std_f)
            
            master_data.append({
                'Metric_Name': metric_filename,
                'Display_Name': display_name,
                'Mean_R': mean_r,
                'Std_R': std_r,
                'Mean_F': mean_f,
                'Std_F': std_f,
                'Conservative_R': cons_r,
                'Conservative_F': cons_f,
                'Is_Literature': is_literature,
                'Robust_Path': robust_path, # Track specific paths for later loading
                'Faith_Path': faith_path
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    df_master = pd.DataFrame(master_data).fillna(0)
    
    if len(df_master) > 1:
        df_master['Z_R'] = (df_master['Conservative_R'] - df_master['Conservative_R'].mean()) / df_master['Conservative_R'].std()
        df_master['Z_F'] = (df_master['Conservative_F'] - df_master['Conservative_F'].mean()) / df_master['Conservative_F'].std()
        df_master['Combined_Score'] = df_master['Z_R'] + df_master['Z_F']
        df_master = df_master.sort_values(by='Combined_Score', ascending=False).reset_index(drop=True)
    
    literature_df = df_master[df_master['Is_Literature'] == True].reset_index(drop=True)
    the_rest = df_master[df_master['Is_Literature'] == False].reset_index(drop=True)

    print(f"Total metrics parsed: {len(df_master)} | Literature Standard Metrics Found: {len(literature_df)}")

    print("\n--- 2. Generating Scatter Plot ---")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(16, 12)) 
    
    if len(the_rest) > 0:
        plt.errorbar(
            x=the_rest['Mean_R'], 
            y=the_rest['Mean_F'], 
            xerr=the_rest['Std_R'], 
            yerr=the_rest['Std_F'], 
            fmt='o', 
            color='lightgray', 
            ecolor='gainsboro', 
            elinewidth=1.5, 
            capsize=3, 
            markersize=6, 
            alpha=0.4,
            label='_nolegend_'
        )

    if len(literature_df) > 0:
        base_colors = sns.color_palette("tab20", 20) 
        base_markers = ['o', 's', '^', 'D', '*', 'X', 'p', 'h', 'v', '<']
        style_combos = list(itertools.product(base_markers, base_colors))
        
        for idx, row in literature_df.iterrows():
            current_marker, current_color = style_combos[idx % len(style_combos)]
            
            wrapped_name = textwrap.fill(row['Display_Name'], width=40)
            legend_label = f"{wrapped_name}\n(R: {row['Mean_R']:.2f}, F: {row['Mean_F']:.2f})"
            
            plt.errorbar(
                x=row['Mean_R'], 
                y=row['Mean_F'], 
                xerr=row['Std_R'], 
                yerr=row['Std_F'], 
                fmt=current_marker,
                color=current_color, 
                ecolor=current_color, 
                elinewidth=2, 
                capsize=4, 
                markersize=10, 
                alpha=1.0,
                label=legend_label
            )

    # plt.title('Performance Metric Landscape: Evaluating Existing Standards', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Robustness Scores (Higher = More resistant to mathematical exploits)', fontsize=12, fontweight='bold')
    plt.ylabel('Fidelity Scores (Higher = More faithful to biological signal)', fontsize=12, fontweight='bold')
    
    plt.legend(
        title='Existing Metrics (Literature)', 
        title_fontsize='13', 
        fontsize='9', # Slightly smaller to fit 28 items
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        borderaxespad=0.,
        labelspacing=1.0 
    )

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'figure1_existing_literature.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("\n--- 3. Generating Per-Dataset Bar Plots for Literature Metrics ---")
    
    faith_records = []
    rob_records = []
    
    for _, row in literature_df.iterrows():
        short_name = textwrap.fill(row['Display_Name'], width=25)
        
        try:
            r_df = pd.read_csv(row['Robust_Path'])
            f_df = pd.read_csv(row['Faith_Path'])
            
            for _, r_row in r_df.iterrows():
                rob_records.append({
                    'Metric': short_name,
                    'Dataset': r_row['Dataset'],
                    'Score': r_row['Overall_Robustness'],
                    'Std': r_row.get('Robustness_Std', 0.0) 
                })
                
            for _, f_row in f_df.iterrows():
                faith_records.append({
                    'Metric': short_name,
                    'Dataset': f_row['Dataset'],
                    'Score': f_row['Fidelity_Score'],
                    'Std': f_row.get('Fidelity_Std', 0.0) 
                })
        except Exception as e:
            print(f"Error extracting per-dataset data for {row['Display_Name']}: {e}")

    df_faith_bar = pd.DataFrame(faith_records)
    df_rob_bar = pd.DataFrame(rob_records)

    fig, axes = plt.subplots(2, 1, figsize=(32, 22))
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    palette = sns.color_palette("husl", len(literature_df))

    MASTER_ORDER = [
        'frangieh21', 'kaden25fibroblast', 'tian21crispri', 'kaden25rpe1', 
        'replogle22k562gwps', 'sunshine23', 'tian21crispra', 'nadig25jurkat', 
        'nadig25hepg2', 'wessels23', 'replogle22rpe1', 'replogle22k562', 
        'norman19', 'adamson16'
    ]

    present_datasets = df_faith_bar['Dataset'].unique()
    datasets = sorted(present_datasets, key=lambda x: MASTER_ORDER.index(x) if x in MASTER_ORDER else 999)
    metrics = df_faith_bar['Metric'].unique()
    
    x = np.arange(len(datasets))
    width = 0.85 / len(metrics) 
    
    axes[0].set_title('Fidelity Scores Across Datasets (Literature Metrics)', fontsize=22, fontweight='bold', pad=15)
    
    for i, metric in enumerate(metrics):
        metric_data = df_faith_bar[df_faith_bar['Metric'] == metric]
        scores = []
        stds = []
        for d in datasets:
            match = metric_data[metric_data['Dataset'] == d]
            scores.append(match['Score'].values[0] if not match.empty else 0)
            stds.append(match['Std'].values[0] if not match.empty else 0)
            
        offset = (i - len(metrics)/2) * width + width/2
        axes[0].bar(x + offset, scores, width, yerr=stds, label=metric, color=palette[i], alpha=0.9, capsize=2)

    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel('Fidelity Score', fontsize=18, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets, rotation=45, ha='right', fontsize=16)
    axes[0].legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Literature Metrics', title_fontsize='14', fontsize='9', ncol=2)

    # axes[1].set_title('Robustness Scores Across Datasets (Literature Metrics)', fontsize=22, fontweight='bold', pad=15)
    
    for i, metric in enumerate(metrics):
        metric_data = df_rob_bar[df_rob_bar['Metric'] == metric]
        scores = []
        stds = []
        for d in datasets:
            match = metric_data[metric_data['Dataset'] == d]
            scores.append(match['Score'].values[0] if not match.empty else 0)
            stds.append(match['Std'].values[0] if not match.empty else 0)
            
        offset = (i - len(metrics)/2) * width + width/2
        axes[1].bar(x + offset, scores, width, yerr=stds, color=palette[i], alpha=0.9, capsize=2)

    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel('Robustness Score', fontsize=18, fontweight='bold')
    axes[1].set_xlabel('Single-Cell Dataset', fontsize=18, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets, rotation=45, ha='right', fontsize=16)

    plt.tight_layout()
    save_path_bars = os.path.join(save_dir, 'figure1_literature_per_dataset_bars.png')
    plt.savefig(save_path_bars, dpi=300, bbox_inches='tight')
    plt.close()

    print("\n--- 4. Generating Per-Dataset Bar Plots for Literature Metrics (Grouped by Cell Type) ---")
    
    faith_records = []
    rob_records = []
    
    for _, row in literature_df.iterrows():
        short_name = textwrap.fill(row['Display_Name'], width=25)
        
        try:
            r_df = pd.read_csv(row['Robust_Path'])
            f_df = pd.read_csv(row['Faith_Path'])
            
            for _, r_row in r_df.iterrows():
                rob_records.append({
                    'Metric': short_name,
                    'Dataset': r_row['Dataset'],
                    'Score': r_row['Overall_Robustness'],
                    'Std': r_row.get('Robustness_Std', 0.0) 
                })
                
            for _, f_row in f_df.iterrows():
                faith_records.append({
                    'Metric': short_name,
                    'Dataset': f_row['Dataset'],
                    'Score': f_row['Fidelity_Score'],
                    'Std': f_row.get('Fidelity_Std', 0.0) 
                })
        except Exception as e:
            print(f"Error extracting per-dataset data for {row['Display_Name']}: {e}")

    df_faith_bar = pd.DataFrame(faith_records)
    df_rob_bar = pd.DataFrame(rob_records)

    # --- CELL TYPE MAPPING ---
    CELL_TYPE_MAP = {
        'adamson16': 'K562',
        'norman19': 'K562',
        'replogle22k562': 'K562',
        'replogle22k562gwps': 'K562',
        'frangieh21': 'Melanoma',
        'kaden25fibroblast': 'Fibroblast',
        'kaden25rpe1': 'RPE1',
        'replogle22rpe1': 'RPE1',
        'nadig25hepg2': 'HepG2',
        'nadig25jurkat': 'Jurkat',
        'sunshine23': 'Calu3',
        'tian21crispra': 'iPSC Neurons',
        'tian21crispri': 'iPSC Neurons',
        'vcc2025': 'H1 hESC',
        'wessels23': 'Monocytes'
    }

    # Map the cell types and sort datasets within those groups
    df_faith_bar['Cell_Type'] = df_faith_bar['Dataset'].map(CELL_TYPE_MAP)
    df_rob_bar['Cell_Type'] = df_rob_bar['Dataset'].map(CELL_TYPE_MAP)

    # Define the order of cell types to group similar ones together
    CELL_TYPE_ORDER = [
        'K562', 'RPE1', 'iPSC Neurons', 'Melanoma', 'Fibroblast', 
        'HepG2', 'Jurkat', 'Calu3', 'H1 hESC', 'Monocytes'
    ]

    # Get unique datasets present and sort them strictly by Cell Type, then alphabetically
    present_datasets = df_faith_bar['Dataset'].unique()
    
    datasets = sorted(
        present_datasets, 
        key=lambda x: (
            CELL_TYPE_ORDER.index(CELL_TYPE_MAP.get(x, 'Z')) if CELL_TYPE_MAP.get(x) in CELL_TYPE_ORDER else 999,
            x
        )
    )

    metrics = df_faith_bar['Metric'].unique()

    fig, axes = plt.subplots(2, 1, figsize=(34, 24)) # Slightly wider to accommodate group labels
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    
    palette = sns.color_palette("husl", len(literature_df))

    # --- PLOTTING LOGIC WITH GAPS FOR GROUPS ---
    # Instead of a simple arange, we calculate custom X positions to create physical gaps between cell types
    x_positions = []
    current_x = 0
    current_cell_type = None
    group_boundaries = []
    group_labels = []
    group_label_x = []

    for d in datasets:
        ct = CELL_TYPE_MAP.get(d, 'Unknown')
        if current_cell_type is not None and ct != current_cell_type:
            # Add a gap when the cell type changes
            current_x += 1.5 
            group_boundaries.append(current_x - 0.75) # Midpoint of the gap for drawing dividing lines
            
            # Record the center position of the previous group for the label
            group_label_x.append(np.mean(x_positions[-len([ds for ds in datasets if CELL_TYPE_MAP.get(ds) == current_cell_type]):]))
            group_labels.append(current_cell_type)

        x_positions.append(current_x)
        current_cell_type = ct
        current_x += 1

    # Record the label for the final group
    group_label_x.append(np.mean(x_positions[-len([ds for ds in datasets if CELL_TYPE_MAP.get(ds) == current_cell_type]):]))
    group_labels.append(current_cell_type)

    x = np.array(x_positions)
    width = 0.85 / len(metrics) 
    
    def plot_grouped_bars(ax, df_data, title, y_label):
        ax.set_title(title, fontsize=24, fontweight='bold', pad=50)
        
        for i, metric in enumerate(metrics):
            metric_data = df_data[df_data['Metric'] == metric]
            scores = []
            stds = []
            for d in datasets:
                match = metric_data[metric_data['Dataset'] == d]
                scores.append(match['Score'].values[0] if not match.empty else 0)
                stds.append(match['Std'].values[0] if not match.empty else 0)
                
            offset = (i - len(metrics)/2) * width + width/2
            ax.bar(x + offset, scores, width, yerr=stds, label=metric if ax == axes[0] else "", color=palette[i], alpha=0.9, capsize=2)

        ax.set_ylim(0, 1.05)
        ax.set_ylabel(y_label, fontsize=18, fontweight='bold')
        
        # Dataset Ticks
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=14)
        
        # Add Vertical Lines to separate cell types
        for boundary in group_boundaries:
            ax.axvline(x=boundary, color='black', linestyle='--', alpha=0.3, linewidth=2)
            
        # Add Cell Type Labels at the top of the plot
        for gx, label in zip(group_label_x, group_labels):
            ax.text(gx, 1.07, label, ha='center', va='bottom', fontsize=16, fontweight='bold', color='darkblue')

    # Plot Top Row (Fidelity)
    plot_grouped_bars(axes[0], df_faith_bar, 'Fidelity Scores Grouped by Cell Type (Literature Metrics)', 'Fidelity Score')
    axes[0].legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Literature Metrics', title_fontsize='14', fontsize='9', ncol=2)

    # Plot Bottom Row (Robustness)
    plot_grouped_bars(axes[1], df_rob_bar, 'Robustness Scores Grouped by Cell Type (Literature Metrics)', 'Robustness Score')
    axes[1].set_xlabel('Single-Cell Dataset', fontsize=18, fontweight='bold', labelpad=15)

    plt.tight_layout()
    # Add extra space at the top of the subplots so the cell type labels don't get cut off
    plt.subplots_adjust(top=0.88, hspace=0.3) 
    
    save_path_bars = os.path.join(save_dir, 'figure1_literature_per_dataset_bars_grouped.png')
    plt.savefig(save_path_bars, dpi=300, bbox_inches='tight')
    plt.close()


    print("\n--- 5. Generating Ridgeline Distribution Plots for Literature Metrics ---")
    
    def plot_ridgeline(df, title, x_label, save_filename, colormap='coolwarm'):
        np.random.seed(42) 
        df['Score'] = df['Score'] + np.random.normal(0, 0.01, size=len(df))
        
        metric_means = df.groupby('Metric')['Score'].mean().sort_values(ascending=False)
        df['Metric'] = pd.Categorical(df['Metric'], categories=metric_means.index, ordered=True)
        df = df.sort_values('Metric')
        df['Mean_Score'] = df['Metric'].map(metric_means)

        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        g = sns.FacetGrid(
            df, row='Metric', hue='Mean_Score', 
            aspect=12, height=0.6, palette=colormap
        )
        
        g.map(sns.kdeplot, 'Score', bw_adjust=0.8, clip_on=False, fill=True, alpha=0.85, linewidth=1.5, warn_singular=False)
        g.map(sns.kdeplot, 'Score', bw_adjust=0.8, clip_on=False, color="w", lw=2, warn_singular=False)
        g.map(plt.axhline, y=0, lw=1.5, clip_on=False, color='black', alpha=0.5)

        for i, ax in enumerate(g.axes.flat):
            metric_clean = str(metric_means.index[i]).replace('\n', ' ') 
            ax.text(
                -0.05, 0.05, metric_clean, 
                fontweight='normal', fontsize=12, color=ax.lines[-1].get_color(),
                ha='right', va='bottom'
            )

        g.figure.subplots_adjust(hspace=-0.4, left=0.4, top=0.95)

        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        g.set(xlim=(-0.05, 1.10))
        
        plt.setp(ax.get_xticklabels(), fontsize=12, fontweight='normal')
        plt.xlabel(x_label, fontweight='normal', fontsize=14, labelpad=15)
        
        g.figure.suptitle(title, y=0.99, fontsize=18, fontweight='normal')

        save_path = os.path.join(save_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    plot_ridgeline(
        df=df_faith_bar.copy(), 
        title='', 
        x_label='Fidelity Score', 
        save_filename='supp_literature_ridgeline_fidelity.png',
        colormap='mako' 
    )

    plot_ridgeline(
        df=df_rob_bar.copy(), 
        title='', 
        x_label='Robustness Score', 
        save_filename='supp_literature_ridgeline_robustness.png',
        colormap='rocket' 
    )
    
    print("\nSuccess! All Literature plots generated successfully.")

if __name__ == "__main__":
    robustness_base = ("../../analysis/robustness_real")
    fidelity_base = ("../../analysis/fidelity_real")
    output_directory = ("../../analysis/plots")
    
    generate_figure1_original(
        robust_base_dir=robustness_base, 
        faith_base_dir=fidelity_base, 
        save_dir=output_directory
    )