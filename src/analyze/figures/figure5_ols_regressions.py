import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
from matplotlib.ticker import MaxNLocator
import statsmodels.formula.api as smf

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
    'Retrieval/Rank': '#9467bd',   
    'Distributional': '#2ca02c',   
    'Spatial': '#d62728',          
    'Diff. Expression': '#ff7f0e', 
    'Unknown': '#333333'           
}

SWARM_AXES = {
    'Weighting_Strategy': ['Static', 'DEG_Binary', 'DEG_Continuous', 'Hybrid_Continuous'],
    'Reference_Strategy': ['Specific_Perturbation', 'Control_Mean_Shift', 'All_Perturbations_Mean_Shift', 'None'],
    'Aggregation_Strategy': ['None', 'Pseudobulk', 'Distribution'],
    'Feature_Space': ['Raw_Genes', 'PCA_256']
}

def parse_swarm_attributes(filename):
    name = filename.replace('.csv', '')
    attrs = {}
    
    if name == 'DES_VCC_None_None_Distribution_Raw_Genes':
        attrs['Core_Metric'] = 'DES_VCC'
        attrs['Metric_Class'] = 'Diff. Expression'
        attrs['Weighting_Strategy'] = 'Static'
        attrs['Reference_Strategy'] = 'None'
        attrs['Aggregation_Strategy'] = 'Distribution'
        attrs['Feature_Space'] = 'Raw_Genes'
        return attrs
        
    attrs['Core_Metric'] = 'Unknown'
    attrs['Metric_Class'] = 'Unknown'
    for cm in sorted(CORE_TO_CLASS.keys(), key=len, reverse=True):
        if name.startswith(cm):
            if 'DES_Robust' in name:
                attrs['Core_Metric'] = 'DES_Custom'
            else:
                attrs['Core_Metric'] = cm
            attrs['Metric_Class'] = CORE_TO_CLASS[cm]
            name = name[len(cm)+1:]
            break
            
    attrs['Feature_Space'] = 'Unknown'
    for s in sorted(SWARM_AXES['Feature_Space'], key=len, reverse=True):
        if name.endswith(s):
            attrs['Feature_Space'] = s
            name = name[:-(len(s)+1)]
            break
            
    attrs['Aggregation_Strategy'] = 'Unknown'
    for a in sorted(SWARM_AXES['Aggregation_Strategy'], key=len, reverse=True):
        if name.endswith(a):
            attrs['Aggregation_Strategy'] = a
            name = name[:-(len(a)+1)]
            break
            
    attrs['Reference_Strategy'] = 'Unknown'
    for r in sorted(SWARM_AXES['Reference_Strategy'], key=len, reverse=True):
        if name.endswith(r):
            attrs['Reference_Strategy'] = r
            name = name[:-(len(r)+1)]
            break
            
    weighting = name if name else 'Static'
    if weighting.startswith('Top20_'):
        weighting = weighting.replace('Top20_', '')
    elif weighting == 'Top20':
        weighting = 'Static'
        
    if weighting == 'None':
        weighting = 'Static'
        
    attrs['Weighting_Strategy'] = weighting
    return attrs

def get_ols_results(df, formula):
    model = smf.ols(formula=formula, data=df).fit()
    
    res = pd.DataFrame({
        'Coefficient': model.params,
        'Lower_CI': model.conf_int()[0],
        'Upper_CI': model.conf_int()[1],
        'P_Value': model.pvalues
    }).reset_index()
    
    res = res[res['index'] != 'Intercept'].copy()
    res = res.dropna(subset=['Coefficient'])
    
    def parse_idx(idx):
        cat = idx.split(",")[0].replace("C(", "")
        lvl = idx.split("[")[1].replace("T.", "").replace("S.", "").replace("]", "")
        
        cat_map = {
            'Core_Metric': 'Core Metric',
            'Weighting_Strategy': 'Gene Weighting',
            'Reference_Strategy': 'Reference Shift',
            'Aggregation_Strategy': 'Aggregation',
            'Feature_Space': 'Input Space'
        }
        
        cat = cat_map.get(cat, cat)
        lvl = lvl.replace('_', ' ')
        
        if cat == 'Reference Shift':
            if lvl == 'All Perturbations Mean Shift':
                lvl = 'Δ Pert'
            elif lvl == 'Control Mean Shift':
                lvl = 'Δ Ctrl'
            elif lvl == 'Specific Perturbation':
                lvl = 'None'
                
        if cat == 'Input Space':
            if lvl == 'Raw Genes':
                lvl = 'Ambient'

        # Apply Abbreviations to fit text inside the compact rectangle
        abbr_map = {
            'DES VCC': 'DES',
            'Sym KL Divergence': 'Sym KL Div.',
            'Centroid Accuracy': 'Centroid Acc.',
            'Hybrid Continuous': 'Hybrid Cont.',
            'DEG Continuous': 'DEG Cont.'
        }
        for old, new in abbr_map.items():
            lvl = lvl.replace(old, new)

        return lvl, cat

    res[['Level', 'Category']] = res['index'].apply(parse_idx).apply(pd.Series)
    return res

def create_forest_figure(df, formula_F, formula_R, save_path, title_prefix="", compact=False, half_width=False):
    print(f"\nFitting OLS Models for: {save_path.split('/')[-1]}")
    res_F = get_ols_results(df, formula_F)
    res_R = get_ols_results(df, formula_R)
    
    def get_class(row):
        if row['Category'] == 'Core Metric':
            lookup_key = row['Level'].replace(' ', '_')
            if lookup_key == 'R²': lookup_key = 'R2'
            if lookup_key == 'DES_Custom': lookup_key = 'DES_Robust'
            
            # Reverse abbreviations so we can look up their color/sorting class
            if lookup_key == 'Sym_KL_Div.': lookup_key = 'Sym_KL_Divergence'
            if lookup_key == 'Centroid_Acc.': lookup_key = 'Centroid_Accuracy'
            
            return CORE_TO_CLASS.get(lookup_key, 'Unknown')
        return 'Unknown'

    res_F['Metric_Class'] = res_F.apply(get_class, axis=1)
    
    class_sort_map = {list(CLASS_COLORS.keys())[i]: i for i in range(len(CLASS_COLORS))}
    res_F['Class_Order'] = res_F['Metric_Class'].map(class_sort_map)
    
    res_F['Unique_ID'] = res_F['Category'] + "||" + res_F['Level']
    res_R['Unique_ID'] = res_R['Category'] + "||" + res_R['Level']
    
    res_F = res_F.sort_values(by=['Category', 'Class_Order', 'Coefficient'], ascending=[True, True, False])
    master_order = res_F['Unique_ID'].tolist()
    
    res_R['Unique_ID'] = pd.Categorical(res_R['Unique_ID'], categories=master_order, ordered=True)
    res_R = res_R.sort_values(by='Unique_ID')
    
    categories = res_F['Category'].unique()
    
    if half_width:
        num_items = len(res_F)
        row_height = 0.22 
        cat_padding = 1.0
        visual_gap = 1.6
        rect_pad_y = 0.8
        fig_width = 9.5
        fig_height = max(5.0, (num_items * row_height) + (len(categories) * cat_padding))
        title_font, label_font, star_font, marker_size = 15, 14, 14, 50
        left_adj, wspace_adj = 0.45, 0.15
        x_label = 'Effect Size\n(Dev. from Mean)'
        rect_left_ax1 = -0.75  
        cat_x_pos = -0.80      
        rect_right_ax2 = 1.10
    else:
        num_items = len(res_F)
        row_height = 0.28 if compact else 0.45
        cat_padding = 1.0 if compact else 1.5
        visual_gap = 1.6 if compact else 2.0
        rect_pad_y = 0.8
        fig_width = 16
        fig_height = max(5.5 if compact else 6, (num_items * row_height) + (len(categories) * cat_padding))
        title_font, label_font, star_font = 18, 16, 16
        marker_size = 80
        left_adj, wspace_adj = 0.30, 0.08
        x_label = 'Effect Size (Deviation from Global Mean)'
        rect_left_ax1 = -0.35  
        cat_x_pos = -0.38      
        rect_right_ax2 = 1.05

    sns.set_theme(style="ticks", context="paper", font_scale=1.6)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharey=True)
    fig.subplots_adjust(left=left_adj, right=0.95, wspace=wspace_adj) 
    
    # Make the axes backgrounds transparent so they don't cover the bounding rectangle
    for ax in axes:
        ax.set_facecolor('none')
    
    # Compute dynamic rectangle width across both axes
    pt = axes[1].transAxes.transform((rect_right_ax2, 0))
    ax2_right_in_ax1 = axes[0].transAxes.inverted().transform(pt)[0]
    trans_rect = mtransforms.blended_transform_factory(axes[0].transAxes, axes[0].transData)
    
    for ax_idx, (ax, res_df, metric_name) in enumerate(zip(axes, [res_F, res_R], ["Fidelity Score", "Robustness Score"])):
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, zorder=1)
        
        # Solid Blue for Fidelity, Solid Orange for Robustness
        plot_color = '#1f77b4' if ax_idx == 0 else '#ff7f0e'
        
        current_y = 0
        for cat in categories:
            cat_df = res_df[res_df['Category'] == cat]
            start_y = current_y
            
            for _, row in cat_df.iterrows():
                # Plot Lines and Scatter
                ax.plot([row['Lower_CI'], row['Upper_CI']], [current_y, current_y], color=plot_color, linewidth=2.5 if not half_width else 1.8, zorder=3)
                ax.scatter(row['Coefficient'], current_y, color=plot_color, s=marker_size, edgecolor='white', linewidth=1, zorder=4)
                
                # Significance Stars
                p = row['P_Value']
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                if stars:
                    star_x = max(row['Upper_CI'], row['Coefficient']) + 0.01
                    ax.text(star_x, current_y - 0.06, stars, va='center', ha='left', color=plot_color, fontweight='bold', fontsize=star_font, zorder=5)
                
                # Y-Axis Label (Metric Levels)
                if ax_idx == 0:
                    label_color = '#333333'
                    font_weight = 'normal'
                        
                    ax.text(-0.05, current_y, row['Level'], ha='right', va='center', 
                            transform=ax.get_yaxis_transform(), fontsize=label_font, 
                            color=label_color, fontweight=font_weight)
                
                current_y -= 1
            
            end_y = current_y + 1
            
            # Draw the Master Rectangle and stacked category names once (on ax 0)
            if ax_idx == 0:
                rect = plt.Rectangle((rect_left_ax1, end_y - rect_pad_y), ax2_right_in_ax1 - rect_left_ax1, start_y - end_y + (2 * rect_pad_y),
                                     transform=trans_rect, fill=False, edgecolor='#cccccc', linewidth=1.5, zorder=0, clip_on=False)
                ax.add_patch(rect)
                
                mid_y = (start_y + end_y) / 2
                cat_label = cat.replace(' ', '\n')
                ax.text(cat_x_pos, mid_y, cat_label, ha='right', va='center', 
                        transform=ax.get_yaxis_transform(), fontweight='normal', fontsize=label_font)
            
            current_y -= visual_gap

        ax.set_title(f'{metric_name}', fontweight='normal', fontsize=title_font, pad=10 if half_width else 15)
        ax.set_xlabel(x_label, fontweight='normal', fontsize=label_font)
        
        if half_width:
            ax.xaxis.set_major_locator(MaxNLocator(4))
            
        ax.xaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
        ax.yaxis.grid(False)
        ax.set_yticks([]) 
        sns.despine(ax=ax, left=True)

    # 2. Compile Master Legend (Significance Stars) and place at the center of the full figure
    legend_elements = [
        Line2D([0], [0], marker='none', linestyle='none', label='* p < 0.05'),
        Line2D([0], [0], marker='none', linestyle='none', label='** p < 0.01'),
        Line2D([0], [0], marker='none', linestyle='none', label='*** p < 0.001')
    ]
    
    fig.legend(handles=legend_elements, title="", loc='lower center', 
               bbox_to_anchor=(0.5, 0.94), frameon=False, 
               fontsize=16, 
               ncol=3)

    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def generate_all_forest_plots(robust_base_dir, faith_base_dir, save_dir):
    print("--- Loading Metric Data ---")
    robust_included_dir = os.path.join(robust_base_dir, 'included')
    faith_included_dir = os.path.join(faith_base_dir, 'included')
    
    all_files = set()
    for d in [robust_base_dir, robust_included_dir]:
        if os.path.exists(d):
            all_files.update([f for f in os.listdir(d) if f.endswith('.csv')])
            
    master_data = []
    for filename in all_files:
        r_path = os.path.join(robust_included_dir, filename) if os.path.exists(os.path.join(robust_included_dir, filename)) else os.path.join(robust_base_dir, filename)
        f_path = os.path.join(faith_included_dir, filename) if os.path.exists(os.path.join(faith_included_dir, filename)) else os.path.join(faith_base_dir, filename)
        
        if not os.path.exists(f_path):
            continue
            
        try:
            rob_df = pd.read_csv(r_path)
            faith_df = pd.read_csv(f_path)
            attrs = parse_swarm_attributes(filename)
            record = {
                'Metric_Filename': filename,
                'Mean_R': rob_df['Overall_Robustness'].mean(),
                'Mean_F': faith_df['Fidelity_Score'].mean()
            }
            record.update(attrs)
            master_data.append(record)
        except Exception:
            continue

    df_master = pd.DataFrame(master_data).fillna('None')
    df_filtered = df_master[(df_master['Mean_R'] > 0) & (df_master['Mean_F'] > 0)].copy()

    eps = 1e-9
    df_filtered['Combined_Score'] = (2.0 * df_filtered['Mean_R'] * df_filtered['Mean_F']) / (df_filtered['Mean_R'] + df_filtered['Mean_F'] + eps)

    print("\n--- Dynamically Dropping Baseline Categories ---")
    categories_to_relevel = ['Core_Metric', 'Weighting_Strategy', 'Reference_Strategy', 'Aggregation_Strategy', 'Feature_Space']
    
    for col in categories_to_relevel:
        if col in df_filtered.columns:
            if col == 'Weighting_Strategy':
                omitted_level = 'Static'
            else:
                omitted_level = df_filtered.groupby(col)['Combined_Score'].mean().idxmin()
            
            if omitted_level in df_filtered[col].values:
                other_levels = [l for l in df_filtered[col].unique() if l != omitted_level]
                df_filtered[col] = pd.Categorical(df_filtered[col], categories=other_levels + [omitted_level])

    os.makedirs(save_dir, exist_ok=True)

    f_main = "Mean_F ~ C(Core_Metric, Sum) + C(Weighting_Strategy, Sum) + C(Reference_Strategy, Sum)"
    r_main = "Mean_R ~ C(Core_Metric, Sum) + C(Weighting_Strategy, Sum) + C(Reference_Strategy, Sum)"
    
    # 1. Generate Full Width (For Supplement)
    create_forest_figure(df_filtered, f_main, r_main, os.path.join(save_dir, 'figure5_ols_main.pdf'), title_prefix="", compact=True, half_width=False)
    
    # 2. Generate Byte-Sized Half Width (For Main Text 1x2)
    create_forest_figure(df_filtered, f_main, r_main, os.path.join(save_dir, 'figure5_ols_main_halfwidth.pdf'), title_prefix="", compact=True, half_width=True)

    f_agg = "Mean_F ~ C(Aggregation_Strategy, Sum)"
    r_agg = "Mean_R ~ C(Aggregation_Strategy, Sum)"
    create_forest_figure(df_filtered, f_agg, r_agg, os.path.join(save_dir, 'supp_aggregation_ols.pdf'), title_prefix="Aggregation Analysis.", compact=False)

    df_dist = df_filtered[df_filtered['Metric_Class'] == 'Distributional'].copy()
    f_feat = "Mean_F ~ C(Feature_Space, Sum)"
    r_feat = "Mean_R ~ C(Feature_Space, Sum)"
    create_forest_figure(df_dist, f_feat, r_feat, os.path.join(save_dir, 'supp_input_space_ols.pdf'), title_prefix="Input Space Analysis (Distributional Metrics).", compact=False)

    print("\nSuccess! All Forest Plots generated.")

if __name__ == "__main__":
    robustness_directory = ("../../analysis/robustness_real")
    fidelity_directory = ("../../analysis/fidelity_real")
    output_directory = ("../../analysis/plots")
    
    generate_all_forest_plots(robustness_directory, fidelity_directory, output_directory)