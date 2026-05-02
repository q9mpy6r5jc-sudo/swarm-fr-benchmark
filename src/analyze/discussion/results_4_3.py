import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator

DATASET_DEGS = { 
    'frangieh21': 0, 'kaden25fibroblast': 1, 'kaden25rpe1': 1, 'tian21crispri': 1,
    'genespider2': 1,
    'replogle22k562gwps': 4, 'sunshine23': 6, 'tian21crispra': 7, 'nadig25jurkat': 10,
    'nadig25hepg2': 13, 'wessels23': 18, 'replogle22rpe1': 34, 'replogle22k562': 37,
    'norman19': 104, 'adamson16': 132, 'vcc2025': 827
}

HIB_METRICS = ['Pearson', 'R2', 'PDS', 'NIR', 'Centroid_Accuracy', 'DES_Robust', 'DES_VCC', 'Cosine', 'Rank_Cosine', 'Rank_Pearson']
LIB_METRICS = ['MSE', 'MAE', 'Wasserstein', 'Sym_KL_Divergence', 'E_Distance', 'MMD', 'RMSE']
HACK_THRESHOLD = 0.01

def load_and_calculate_gaming_ratio(real_dir, synth_dir):
    print("--- 1. Loading Aggregated Metric Data ---")
    all_files = []
    if os.path.exists(real_dir):
        all_files.extend([(f, 'Real') for f in glob.glob(os.path.join(real_dir, "*.csv"))])
    if os.path.exists(synth_dir):
        all_files.extend([(f, 'Synthetic') for f in glob.glob(os.path.join(synth_dir, "*.csv"))])
        
    if not all_files:
        return pd.DataFrame()

    master_records = []
    for file_path, source in tqdm(all_files, desc="Processing Metrics"):
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
            df = pd.read_csv(file_path)
            pivot_df = df.pivot_table(
                index=['Dataset', 'Variant', 'Level', 'SNR'], 
                columns='Attack_Type', 
                values='Value'
            ).reset_index()
            
            if 'No_Attack' not in pivot_df.columns:
                continue
                
            attacks = [c for c in pivot_df.columns if c not in ['Dataset', 'Variant', 'Level', 'SNR', 'No_Attack']]
            
            for _, row in pivot_df.iterrows():
                baseline = row['No_Attack']
                for att in attacks:
                    att_val = row[att]
                    if pd.isna(att_val) or pd.isna(baseline):
                        continue
                        
                    if is_hib:
                        gain = att_val - baseline
                    else:
                        gain = baseline - att_val
                        
                    is_gamed = 1 if gain > HACK_THRESHOLD else 0
                    
                    master_records.append({
                        'Source': source,
                        'Dataset': row['Dataset'],
                        'Variant': row['Variant'],
                        'Level': int(row['Level']),
                        'SNR': row['SNR'],
                        'Attack_Type': att.replace('_', ' '),
                        'Metric_Filename': filename,
                        'Is_Gamed': is_gamed
                    })
        except Exception:
            continue
            
    return pd.DataFrame(master_records)

def format_attack_labels(cols, width):
    clean_cols = []
    for c in cols:
        new_c = c.replace('Mode Collapse ', 'MC ')\
                 .replace('Normalization Mismatch', 'Norm. Mism.')\
                 .replace('Perturbed', 'Pert.')\
                 .replace('Control', 'Ctrl.')\
                 .replace('Scaling Exploit', 'Scaling Exp.')
        clean_cols.append(textwrap.fill(new_c, width=width))
    return clean_cols

def format_snr(s):
    s_str = str(s).lower()
    if s_str in ['inf', 'infinity']: return 'Inf'
    try:
        f = float(s)
        if f.is_integer(): return str(int(f))
        return str(f)
    except:
        return str(s)

def snr_sort_key(s):
    s_str = str(s).lower()
    if s_str in ['inf', 'infinity']: return float('inf')
    try: return float(s)
    except: return -1.0

def generate_vulnerability_distribution_plot(df, plots_dir, half_width=False):
    print(f"\n--- 2. Generating Vulnerability Distribution Plot ({'Half Width' if half_width else 'Full Width'}) ---")
    
    var_df = df[df['Variant'].isin(['Magnitude', 'Structure'])].copy()
    if var_df.empty: return
        
    var_df['nDEGs'] = var_df['Dataset'].map(DATASET_DEGS)
    
    metric_gamed_df = var_df.groupby(['Dataset', 'nDEGs', 'Variant', 'Level', 'SNR', 'Attack_Type', 'Metric_Filename'])['Is_Gamed'].max().reset_index()
    suite_vuln = metric_gamed_df.groupby(['Dataset', 'nDEGs', 'Variant', 'Level', 'SNR', 'Attack_Type'])['Is_Gamed'].mean().reset_index()
    
    suite_vuln['Dataset_Label'] = suite_vuln['Dataset'] + '\n(n=' + suite_vuln['nDEGs'].astype(int).astype(str) + ')'
    
    unique_snrs = sorted(suite_vuln['SNR'].unique(), key=snr_sort_key, reverse=True)
    clean_snr_order = [format_snr(s) for s in unique_snrs]
    
    suite_vuln['Clean_SNR'] = suite_vuln['SNR'].apply(format_snr)
    suite_vuln['Clean_SNR'] = pd.Categorical(suite_vuln['Clean_SNR'], categories=clean_snr_order, ordered=True)
    
    suite_vuln = suite_vuln.sort_values(by=['nDEGs', 'Clean_SNR'])
    os.makedirs(plots_dir, exist_ok=True)
    
    palette = ['#fcae91', '#fb6a4a', '#de2d26', '#a50f15']
    
    if half_width:
        # Group and average across all datasets
        plot_df = suite_vuln.groupby(['Clean_SNR', 'Variant', 'Level', 'Attack_Type'], observed=True)['Is_Gamed'].mean().reset_index()
        
        # Reverse categorical order so Y-axis progresses Inf (bottom) -> 0.01 (top)
        plot_df['Clean_SNR'] = pd.Categorical(plot_df['Clean_SNR'], categories=clean_snr_order[::-1], ordered=True)
        # Reverse the palette to match the reversed categories and maintain the exact color gradient
        half_palette = palette[::-1]
        
        sns.set_theme(style="ticks", context="paper", font_scale=1.6)
        
        # Create a broken axis layout: Width ratio of 4:1
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6.5, 6.5), gridspec_kw={'width_ratios': [4, 1]})
        fig.subplots_adjust(bottom=0.2, wspace=0.1)
        
        for ax in [ax1, ax2]:
            sns.boxplot(
                data=plot_df,
                y='Clean_SNR',
                x='Is_Gamed',
                hue='Clean_SNR',
                palette=half_palette,
                ax=ax,
                fliersize=0, 
                boxprops={'alpha': 0.6},
                zorder=1,
                legend=False
            )
            
            sns.stripplot(
                data=plot_df,
                y='Clean_SNR',
                x='Is_Gamed',
                hue='Clean_SNR',
                palette=half_palette,
                dodge=False,
                alpha=0.7,
                size=6,
                ax=ax,
                zorder=2,
                jitter=True,
                legend=False
            )
        
        # Calculate dynamic x-limit for the primary (left) plot
        max_val = plot_df['Is_Gamed'].max()
        left_max = max(0.35, max_val + 0.05)
        
        ax1.set_xlim(-0.02, left_max)
        ax2.set_xlim(0.92, 1.02)
        ax2.set_xticks([1.0])
        ax1.xaxis.set_major_locator(MaxNLocator(4))
        
        # Style the spines to create the "break" illusion
        sns.despine(ax=ax1, right=True, top=True)
        sns.despine(ax=ax2, left=True, right=True, top=True)
        ax2.tick_params(left=False)
        
        # Draw the diagonal cut lines on the bottom spines
        d = .025  
        kwargs = dict(transform=ax1.transAxes, color='#333333', clip_on=False, lw=1.5)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

        kwargs.update(transform=ax2.transAxes)  
        d_x = d * 4  # Adjust angle to account for the 4:1 width ratio
        ax2.plot((-d_x, +d_x), (-d, +d), **kwargs)
        
        # Bind the axis labels
        ax1.set_ylabel('Signal-to-Noise Ratio (SNR)', fontsize=18)
        ax2.set_ylabel('')
        
        ax1.set_xlabel('Fraction of Exploited Metrics\n(Average across Datasets)', fontsize=18)
        ax1.xaxis.set_label_coords(0.62, -0.15) # Spans perfectly across the center of both axes
        ax2.set_xlabel('')
        
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        
        save_pdf = os.path.join(plots_dir, 'figure3_vulnerability_distribution_halfwidth.pdf')
        save_png = os.path.join(plots_dir, 'figure3_vulnerability_distribution_halfwidth.png')
        
    else:
        sns.set_theme(style="ticks", font_scale=1.0)
        fig, ax = plt.subplots(figsize=(15, 7))
        
        sns.boxplot(
            data=suite_vuln,
            x='Dataset_Label',
            y='Is_Gamed',
            hue='Clean_SNR',
            palette=palette,
            ax=ax,
            fliersize=0, 
            boxprops={'alpha': 0.6},
            zorder=1
        )
        
        sns.stripplot(
            data=suite_vuln,
            x='Dataset_Label',
            y='Is_Gamed',
            hue='Clean_SNR',
            palette=palette,
            dodge=True,
            alpha=0.7,
            size=4,
            ax=ax,
            zorder=2
        )
        
        handles, labels = ax.get_legend_handles_labels()
        n_snrs = len(clean_snr_order)
        ax.legend(handles[:n_snrs], labels[:n_snrs], title='Signal-to-Noise Ratio (SNR)', bbox_to_anchor=(0.98, 0.98), loc='upper right', frameon=True, borderaxespad=0.0)
        
        ax.set_ylabel('Fraction of Exploited Metrics', fontsize=18)
        ax.set_xlabel('Datasets sorted by mean DEGs (n)', fontsize=18)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        
        for i in range(1, len(suite_vuln['Dataset_Label'].unique())):
            ax.axvline(i - 0.5, color='gray', linestyle=':', alpha=0.5)
            
        sns.despine()
        plt.tight_layout()
        
        save_pdf = os.path.join(plots_dir, 'figure3_vulnerability_distribution.pdf')
        save_png = os.path.join(plots_dir, 'figure3_vulnerability_distribution.png')

    plt.savefig(save_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Distribution Plot: {save_pdf}")

def generate_attack_specific_distribution_plots(df, plots_dir):
    print("\n--- Generating Attack-Specific Distribution Plots (Supplementary) ---")
    
    var_df = df[df['Variant'].isin(['Magnitude', 'Structure'])].copy()
    if var_df.empty: return
        
    var_df['nDEGs'] = var_df['Dataset'].map(DATASET_DEGS)
    
    metric_gamed_df = var_df.groupby(['Dataset', 'nDEGs', 'Variant', 'Level', 'SNR', 'Attack_Type', 'Metric_Filename'])['Is_Gamed'].max().reset_index()
    suite_vuln = metric_gamed_df.groupby(['Dataset', 'nDEGs', 'Variant', 'Level', 'SNR', 'Attack_Type'])['Is_Gamed'].mean().reset_index()
    
    suite_vuln['Dataset_Label'] = suite_vuln['Dataset'] + '\n(n=' + suite_vuln['nDEGs'].astype(int).astype(str) + ')'
    
    unique_snrs = sorted(suite_vuln['SNR'].unique(), key=snr_sort_key, reverse=True)
    clean_snr_order = [format_snr(s) for s in unique_snrs]
    
    suite_vuln['Clean_SNR'] = suite_vuln['SNR'].apply(format_snr)
    suite_vuln['Clean_SNR'] = pd.Categorical(suite_vuln['Clean_SNR'], categories=clean_snr_order, ordered=True)
    
    attacks = sorted(suite_vuln['Attack_Type'].unique())
    palette = ['#fcae91', '#fb6a4a', '#de2d26', '#a50f15']
    
    for attack in attacks:
        attack_df = suite_vuln[suite_vuln['Attack_Type'] == attack].sort_values(by=['nDEGs', 'Clean_SNR'])
        
        os.makedirs(plots_dir, exist_ok=True)
        sns.set_theme(style="ticks", font_scale=1.0)
        fig, ax = plt.subplots(figsize=(15, 7))
        
        sns.boxplot(
            data=attack_df,
            x='Dataset_Label',
            y='Is_Gamed',
            hue='Clean_SNR',
            palette=palette,
            ax=ax,
            fliersize=0, 
            boxprops={'alpha': 0.6},
            zorder=1
        )
        
        sns.stripplot(
            data=attack_df,
            x='Dataset_Label',
            y='Is_Gamed',
            hue='Clean_SNR',
            palette=palette,
            dodge=True,
            alpha=0.7,
            size=4,
            ax=ax,
            zorder=2
        )
        
        handles, labels = ax.get_legend_handles_labels()
        n_snrs = len(clean_snr_order)
        ax.legend(handles[:n_snrs], labels[:n_snrs], title='Signal-to-Noise Ratio (SNR)', bbox_to_anchor=(0.98, 0.98), loc='upper right', frameon=True, borderaxespad=0.0)
        
        # ax.set_title(f'Vulnerability Distribution: {attack}', fontweight='bold', pad=15)
        ax.set_ylabel('Fraction of Exploited Metrics')
        ax.set_xlabel('Datasets sorted by mean DEGs (n)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim(-0.05, 1.05)
        
        for i in range(1, len(attack_df['Dataset_Label'].unique())):
            ax.axvline(i - 0.5, color='gray', linestyle=':', alpha=0.5)
            
        sns.despine()
        plt.tight_layout()
        
        safe_att_name = attack.replace(' ', '_').replace('.', '').lower()
        save_pdf = os.path.join(plots_dir, f'supp_figure3_distribution_{safe_att_name}.pdf')
        save_png = os.path.join(plots_dir, f'supp_figure3_distribution_{safe_att_name}.png')
        
        plt.savefig(save_pdf, format='pdf', bbox_inches='tight')
        plt.savefig(save_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Attack-Specific Plot ({attack}): {save_pdf}")

def generate_level_variant_distribution_plot(df, plots_dir):
    print("\n--- 3. Generating Level/Variant Distribution Plot (Supplementary) ---")
    
    var_df = df[df['Variant'].isin(['Magnitude', 'Structure'])].copy()
    if var_df.empty: return
        
    var_df['nDEGs'] = var_df['Dataset'].map(DATASET_DEGS)
    
    metric_gamed_df = var_df.groupby(['Dataset', 'nDEGs', 'Variant', 'Level', 'SNR', 'Attack_Type', 'Metric_Filename'])['Is_Gamed'].max().reset_index()
    suite_vuln = metric_gamed_df.groupby(['Dataset', 'nDEGs', 'Variant', 'Level', 'SNR', 'Attack_Type'])['Is_Gamed'].mean().reset_index()
    
    suite_vuln['Dataset_Label'] = suite_vuln['Dataset'] + '\n(n=' + suite_vuln['nDEGs'].astype(int).astype(str) + ')'
    
    # Combine Variant and Level (e.g., "Mag L1", "Str L5")
    suite_vuln['Variant_Level'] = suite_vuln['Variant'].str[:3] + " L" + suite_vuln['Level'].astype(str)
    
    # Force the specific ordering: 5 Magnitudes, then 5 Structures
    variant_level_order = [f"Mag L{i}" for i in range(1, 6)] + [f"Str L{i}" for i in range(1, 6)]
    suite_vuln['Variant_Level'] = pd.Categorical(suite_vuln['Variant_Level'], categories=variant_level_order, ordered=True)
    
    suite_vuln = suite_vuln.sort_values(by=['nDEGs', 'Variant_Level'])
    
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="ticks", font_scale=1.0)
    
    fig, ax = plt.subplots(figsize=(20, 7))
    palette = sns.color_palette("Blues", 6)[1:] + sns.color_palette("Greens", 6)[1:]
    
    sns.boxplot(
        data=suite_vuln,
        x='Dataset_Label',
        y='Is_Gamed',
        hue='Variant_Level',
        palette=palette,
        ax=ax,
        fliersize=0, 
        boxprops={'alpha': 0.6},
        zorder=1
    )
    
    sns.stripplot(
        data=suite_vuln,
        x='Dataset_Label',
        y='Is_Gamed',
        hue='Variant_Level',
        palette=palette,
        dodge=True,
        alpha=0.7,
        size=3, 
        ax=ax,
        zorder=2
    )
    
    handles, labels = ax.get_legend_handles_labels()
    n_cats = len(variant_level_order)
    ax.legend(handles[:n_cats], labels[:n_cats], title='Degradation Type & Level', bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True)
    
    ax.set_ylabel('Fraction of Exploited Metrics')
    ax.set_xlabel('Datasets sorted by mean DEGs (n)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylim(-0.05, 1.05)
    
    for i in range(1, len(suite_vuln['Dataset_Label'].unique())):
        ax.axvline(i - 0.5, color='gray', linestyle=':', alpha=0.5)
        
    sns.despine()
    plt.tight_layout()
    
    save_pdf = os.path.join(plots_dir, 'supp_figure3_vulnerability_by_level.pdf')
    save_png = os.path.join(plots_dir, 'supp_figure3_vulnerability_by_level.png')
    
    plt.savefig(save_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Level/Variant Plot: {save_pdf}")

def generate_attack_specific_level_variant_plots(df, plots_dir):
    print("\n--- Generating Attack-Specific Level/Variant Plots (Supplementary) ---")
    
    var_df = df[df['Variant'].isin(['Magnitude', 'Structure'])].copy()
    if var_df.empty: return
        
    var_df['nDEGs'] = var_df['Dataset'].map(DATASET_DEGS)
    
    metric_gamed_df = var_df.groupby(['Dataset', 'nDEGs', 'Variant', 'Level', 'SNR', 'Attack_Type', 'Metric_Filename'])['Is_Gamed'].max().reset_index()
    suite_vuln = metric_gamed_df.groupby(['Dataset', 'nDEGs', 'Variant', 'Level', 'SNR', 'Attack_Type'])['Is_Gamed'].mean().reset_index()
    
    suite_vuln['Dataset_Label'] = suite_vuln['Dataset'] + '\n(n=' + suite_vuln['nDEGs'].astype(int).astype(str) + ')'
    suite_vuln['Variant_Level'] = suite_vuln['Variant'].str[:3] + " L" + suite_vuln['Level'].astype(str)
    
    variant_level_order = [f"Mag L{i}" for i in range(1, 6)] + [f"Str L{i}" for i in range(1, 6)]
    suite_vuln['Variant_Level'] = pd.Categorical(suite_vuln['Variant_Level'], categories=variant_level_order, ordered=True)
    
    attacks = sorted(suite_vuln['Attack_Type'].unique())
    palette = sns.color_palette("Blues", 6)[1:] + sns.color_palette("Greens", 6)[1:]
    
    for attack in attacks:
        attack_df = suite_vuln[suite_vuln['Attack_Type'] == attack].sort_values(by=['nDEGs', 'Variant_Level'])
        
        os.makedirs(plots_dir, exist_ok=True)
        sns.set_theme(style="ticks", font_scale=1.0)
        
        fig, ax = plt.subplots(figsize=(20, 7))
        
        sns.boxplot(
            data=attack_df,
            x='Dataset_Label',
            y='Is_Gamed',
            hue='Variant_Level',
            palette=palette,
            ax=ax,
            fliersize=0, 
            boxprops={'alpha': 0.6},
            zorder=1
        )
        
        sns.stripplot(
            data=attack_df,
            x='Dataset_Label',
            y='Is_Gamed',
            hue='Variant_Level',
            palette=palette,
            dodge=True,
            alpha=0.7,
            size=3, 
            ax=ax,
            zorder=2
        )
        
        handles, labels = ax.get_legend_handles_labels()
        n_cats = len(variant_level_order)
        ax.legend(handles[:n_cats], labels[:n_cats], title='Degradation Type & Level', bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True)
        
        # ax.set_title(f'Vulnerability by Degradation Level: {attack}', fontweight='bold', pad=15)
        ax.set_ylabel('Fraction of Exploited Metrics')
        ax.set_xlabel('Datasets sorted by mean DEGs (n)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim(-0.05, 1.05)
        
        for i in range(1, len(attack_df['Dataset_Label'].unique())):
            ax.axvline(i - 0.5, color='gray', linestyle=':', alpha=0.5)
            
        sns.despine()
        plt.tight_layout()
        
        safe_att_name = attack.replace(' ', '_').replace('.', '').lower()
        save_pdf = os.path.join(plots_dir, f'supp_figure3_vulnerability_by_level_{safe_att_name}.pdf')
        save_png = os.path.join(plots_dir, f'supp_figure3_vulnerability_by_level_{safe_att_name}.png')
        
        plt.savefig(save_pdf, format='pdf', bbox_inches='tight')
        plt.savefig(save_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Attack-Specific Level/Variant Plot ({attack}): {save_pdf}")

def generate_extended_vulnerability_heatmap(df, plots_dir):
    print("\n--- 4. Generating Extended Vulnerability Heatmap (Supplementary) ---")
    
    var_df = df[df['Variant'].isin(['Magnitude', 'Structure'])].copy()
    if var_df.empty: return
        
    var_df['nDEGs'] = var_df['Dataset'].map(DATASET_DEGS)
    
    metric_gamed_df = var_df.groupby(['Dataset', 'nDEGs', 'Variant', 'Level', 'SNR', 'Attack_Type', 'Metric_Filename'])['Is_Gamed'].max().reset_index()
    suite_vulnerability = metric_gamed_df.groupby(['Dataset', 'nDEGs', 'Variant', 'Level', 'SNR', 'Attack_Type'])['Is_Gamed'].mean().reset_index()
    
    suite_vulnerability['Clean_SNR'] = suite_vulnerability['SNR'].apply(format_snr)
    suite_vulnerability['Att_Var_Lvl_SNR'] = suite_vulnerability['Attack_Type'] + " [" + suite_vulnerability['Variant'].str[:3] + " L" + suite_vulnerability['Level'].astype(str) + " S:" + suite_vulnerability['Clean_SNR'] + "]"
    
    pivot_df = suite_vulnerability.pivot_table(
        index=['Dataset', 'nDEGs'], 
        columns='Att_Var_Lvl_SNR', 
        values='Is_Gamed'
    ).fillna(0)
    
    pivot_df = pivot_df.sort_index(level='nDEGs', ascending=True)
    ylabels = [f"{idx[0]} (n={int(idx[1])})" for idx in pivot_df.index]
    
    attacks = sorted(suite_vulnerability['Attack_Type'].unique())
    variants = ['Magnitude', 'Structure']
    levels = sorted(suite_vulnerability['Level'].unique())
    snrs = sorted(suite_vulnerability['SNR'].unique(), key=snr_sort_key, reverse=True)
    
    sorted_cols = []
    for att in attacks:
        for var in variants:
            for lvl in levels:
                for snr in snrs:
                    clean_snr = format_snr(snr)
                    col_name = f"{att} [{var[:3]} L{lvl} S:{clean_snr}]"
                    if col_name in pivot_df.columns:
                        sorted_cols.append(col_name)

    chunk_size = 1 
    attack_chunks = [attacks[i:i + chunk_size] for i in range(0, len(attacks), chunk_size)]
    
    for part_num, attack_chunk in enumerate(attack_chunks, 1):
        chunk_cols = [c for c in sorted_cols if any(c.startswith(att) for att in attack_chunk)]
        if not chunk_cols:
            continue
            
        plot_data = pivot_df[chunk_cols].T
        clean_cols = format_attack_labels(chunk_cols, width=18)
        
        os.makedirs(plots_dir, exist_ok=True)
        sns.set_theme(style="white", font_scale=1.0)
        
        n_cols = plot_data.shape[1] 
        n_rows = plot_data.shape[0] 
        fig_width = max(14, n_cols * 1.0)
        fig_height = max(6, n_rows * 0.45)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        sns.heatmap(
            plot_data, 
            cmap=sns.light_palette("#c97f5a", as_cmap=True), 
            annot=True, 
            fmt=".2f", 
            vmin=0.0,
            vmax=1.1,
            linewidths=.5, 
            linecolor='white',
            cbar_kws={
                'label': 'Fraction of Exploited Metrics', 
                'fraction': 0.03,
                'pad': 0.02,
                'aspect': 40
            },
            annot_kws={"size": 11}, 
            ax=ax
        )
        
        ax.set_xticklabels(ylabels, rotation=45, ha='right')
        ax.set_yticklabels(clean_cols, rotation=0)
        ax.set_xlabel('Datasets sorted by mean DEGs (n)', fontweight='normal', labelpad=15)
        ax.set_ylabel('')
        
        current_idx = 0
        for att in attack_chunk:
            att_cols = [c for c in chunk_cols if c.startswith(att)]
            current_idx += len(att_cols)
            if current_idx < len(chunk_cols):
                ax.axhline(current_idx, color='black', lw=1.5)
                
        plt.tight_layout()
        
        save_pdf = os.path.join(plots_dir, f'supp_figure_vulnerability_heatmap_extended_part{part_num}.pdf')
        save_png = os.path.join(plots_dir, f'supp_figure_vulnerability_heatmap_extended_part{part_num}.png')
        
        plt.savefig(save_pdf, format='pdf', bbox_inches='tight')
        plt.savefig(save_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Extended Heatmap (Part {part_num}): {save_pdf}")

if __name__ == "__main__":
    real_data_dir = ("../../analysis/overall_real")
    synth_data_dir = ("../../analysis/overall_synthetic")
    plots_directory = ("../../analysis/plots")
    
    df_damage = load_and_calculate_gaming_ratio(real_data_dir, synth_data_dir)
    
    if not df_damage.empty:
        generate_vulnerability_distribution_plot(df_damage, plots_directory, half_width=False)
        generate_vulnerability_distribution_plot(df_damage, plots_directory, half_width=True)
        
        generate_attack_specific_distribution_plots(df_damage, plots_directory)
        generate_level_variant_distribution_plot(df_damage, plots_directory)
        generate_attack_specific_level_variant_plots(df_damage, plots_directory)
        generate_extended_vulnerability_heatmap(df_damage, plots_directory)