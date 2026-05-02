import os, sys, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from utils.utils import load_sparse_from_csv

def deg_analysis(p_path, y_control_path, y_path, save_dir):
    """
    Performs DEG analysis using the Diversity by Design methodology.
    Generates Binary, Continuous, and Hybrid (33/33/33) weights.
    """
    print(f"--- Processing {os.path.basename(save_dir)} ---")
    print("Loading Data...")
    try:
        P_sparse = load_sparse_from_csv(p_path)
        X_pert_raw = load_sparse_from_csv(y_path)
        X_ctrl_raw = load_sparse_from_csv(y_control_path)
        
    except FileNotFoundError as e:
        print(f"Skipping: {e}")
        return

    dim0_match = (X_pert_raw.shape[0] == X_ctrl_raw.shape[0])
    dim1_match = (X_pert_raw.shape[1] == X_ctrl_raw.shape[1])
    
    if dim1_match and not dim0_match:
        print("   > Detected Orientation: Cells x Genes")
        X_pert_sparse = X_pert_raw
        X_ctrl_sparse = X_ctrl_raw
        P_cells_x_genes = P_sparse 
        n_genes = X_pert_raw.shape[1]
    elif dim0_match and not dim1_match:
        print("   > Detected Orientation: Genes x Cells")
        X_pert_sparse = X_pert_raw.transpose().tocsr() 
        X_ctrl_sparse = X_ctrl_raw.transpose().tocsr()
        P_cells_x_genes = P_sparse.transpose().tocsr()
        n_genes = X_pert_raw.shape[0]
    else:
        raise ValueError("> Orientation completely mismatched. Cannot resolve.")

    print("Constructing AnnData object...")
    X_combined = sp.vstack([X_pert_sparse, X_ctrl_sparse])
    gene_names = [f"G{i+1}" for i in range(n_genes)]
    
    n_pert_cells = X_pert_sparse.shape[0]
    n_ctrl_cells = X_ctrl_sparse.shape[0]
    
    obs_names = [f"Pert_Cell_{i}" for i in range(n_pert_cells)] + \
                [f"Ctrl_Cell_{i}" for i in range(n_ctrl_cells)]
                
    adata = sc.AnnData(X=X_combined)
    adata.var_names = gene_names
    adata.obs_names = obs_names
    
    conditions = []
    for i in range(n_pert_cells):
        start_idx = P_cells_x_genes.indptr[i]
        end_idx = P_cells_x_genes.indptr[i+1]
        
        if end_idx > start_idx:
            pert_indices = P_cells_x_genes.indices[start_idx:end_idx]
            name = "+".join([f"G{idx+1}" for idx in sorted(pert_indices)])
            conditions.append(f"Pert_{name}")
        else:
            conditions.append("Unknown")
            
    conditions.extend(["Control"] * n_ctrl_cells)
    adata.obs["condition"] = conditions
    
    adata_pert = adata[(adata.obs["condition"] != "Unknown") & (adata.obs["condition"] != "Control")].copy()
    adata_all = adata[adata.obs["condition"] != "Unknown"].copy()
    unique_perts = sorted([c for c in adata_pert.obs["condition"].unique()])
    unique_perts.sort(key=lambda x: int(x.split("_G")[1].split('+')[0])) 
    
    results_bin_df = pd.DataFrame(index=unique_perts, columns=gene_names)
    results_cont_df = pd.DataFrame(index=unique_perts, columns=gene_names)
    results_hybrid_df = pd.DataFrame(index=unique_perts, columns=gene_names)

    print("Running Scanpy t-tests...")
    sc.pp.normalize_total(adata_pert, target_sum=1e4)
    sc.pp.log1p(adata_pert)
    
    sc.pp.normalize_total(adata_all, target_sum=1e4)
    sc.pp.log1p(adata_all)

    print("   > Running Perturbation vs Rest...")
    sc.tl.rank_genes_groups(
        adata_pert, groupby='condition', reference='rest', 
        method='t-test_overestim_var', key_added='de_vs_rest'
    )
    
    # Pert vs Control (Used for 1/3 of Hybrid)
    print("   > Running Perturbation vs Control...")
    sc.tl.rank_genes_groups(
        adata_all, groupby='condition', reference='Control', 
        method='t-test_overestim_var', key_added='de_vs_ctrl'
    )
    
    print(f"Calculating Weights for {len(unique_perts)} perturbations...")
    
    for pert_group in unique_perts:
        genes_ordered_rest = adata_pert.uns['de_vs_rest']['names'][pert_group]
        
        # Binary Weight Calc
        pvals_adj = adata_pert.uns['de_vs_rest']['pvals_adj'][pert_group]
        p_series = pd.Series(pvals_adj, index=genes_ordered_rest).reindex(gene_names).fillna(1.0)
        w_bin = (p_series < 0.05).astype(int).values
        results_bin_df.loc[pert_group] = w_bin
        
        # Continuous Weight Calc
        t_scores_rest = adata_pert.uns['de_vs_rest']['scores'][pert_group]
        t_series_rest = pd.Series(t_scores_rest, index=genes_ordered_rest).reindex(gene_names).fillna(0)
        
        w_cont = np.abs(t_series_rest.values)
        w_min, w_max = w_cont.min(), w_cont.max()
        if w_max > w_min: w_cont = (w_cont - w_min) / (w_max - w_min)
        else: w_cont = np.zeros_like(w_cont)
        
        w_cont = w_cont ** 2
        w_sum = w_cont.sum()
        if w_sum > 0: w_cont = w_cont / w_sum
        results_cont_df.loc[pert_group] = w_cont

        # Hybrid Weight Calc (33/33/33)
        w_rest_hybrid = w_cont 
        
        genes_ordered_ctrl = adata_all.uns['de_vs_ctrl']['names'][pert_group]
        t_scores_ctrl = adata_all.uns['de_vs_ctrl']['scores'][pert_group]
        t_series_ctrl = pd.Series(t_scores_ctrl, index=genes_ordered_ctrl).reindex(gene_names).fillna(0)
        
        w_ctrl_raw = np.abs(t_series_ctrl.values)
        c_max = np.max(w_ctrl_raw)
        w_ctrl_norm = (w_ctrl_raw / c_max)**2 if c_max > 0 else np.zeros_like(w_ctrl_raw)
        c_sum = np.sum(w_ctrl_norm)
        w_ctrl_hybrid = w_ctrl_norm / c_sum if c_sum > 0 else np.zeros_like(w_ctrl_norm)
        w_uniform = np.ones(n_genes) / n_genes
        
        w_hybrid = (1/3) * w_uniform + (1/3) * w_ctrl_hybrid + (1/3) * w_rest_hybrid
        results_hybrid_df.loc[pert_group] = w_hybrid

    clean_index = [idx.replace("Pert_", "") for idx in results_bin_df.index]
    results_bin_df.index = clean_index
    results_cont_df.index = clean_index
    results_hybrid_df.index = clean_index
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    bin_path = os.path.join(save_dir, "DEG_Adjacency_Matrix_binary.csv")
    cont_path = os.path.join(save_dir, "DEG_Adjacency_Matrix_cont.csv")
    hybrid_path = os.path.join(save_dir, "DEG_Adjacency_Matrix_hybrid.csv")
    
    results_bin_df.to_csv(bin_path)
    results_cont_df.to_csv(cont_path)
    results_hybrid_df.to_csv(hybrid_path)
    print(f"Done! Results saved to: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate DEG weights for a specific GT_Reference.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        help="The folder name of the dataset (e.g., 'adamson16')"
    )
    parser.add_argument(
        "--base_path", 
        type=str, 
        default=("../../datasets/real"),
        help="The root directory containing the dataset folders."
    )
    
    args = parser.parse_args()
    dataset_dir = os.path.join(args.base_path, args.dataset)
    gt_reference_dir = os.path.join(dataset_dir, "variants", "GT_Reference")
    
    if not os.path.exists(gt_reference_dir):
        print(f"ERROR: GT_Reference folder not found at {gt_reference_dir}")
        print("Please ensure generate_variants.py has finished running for this dataset.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"RUNNING DEG ANALYSIS FOR: {args.dataset.upper()} (GT_Reference)")
    print(f"{'='*60}")
    
    binary_path = os.path.join(gt_reference_dir, "DEG_Adjacency_Matrix_binary.csv")
    cont_path = os.path.join(gt_reference_dir, "DEG_Adjacency_Matrix_cont.csv")
    hybrid_path = os.path.join(gt_reference_dir, "DEG_Adjacency_Matrix_hybrid.csv")
    
    if os.path.exists(binary_path) and os.path.exists(cont_path) and os.path.exists(hybrid_path):
        print(f"Skipping {args.dataset}: All 3 Adjacency Matrices already exist.")
        sys.exit(0)

    p_path = os.path.join(gt_reference_dir, "P_perturbation.csv")
    y_control_path = os.path.join(gt_reference_dir, "Y_control_counts.csv")
    y_path = os.path.join(gt_reference_dir, "Y_counts.csv")

    if all(os.path.exists(p) for p in [p_path, y_control_path, y_path]):
        deg_analysis(
            p_path=p_path,
            y_control_path=y_control_path,
            y_path=y_path,
            save_dir=gt_reference_dir
        )
    else:
        print(f"ERROR: Missing one or more required CSVs (Y_counts, Y_control_counts, P_perturbation) in {gt_reference_dir}.")
        sys.exit(1)