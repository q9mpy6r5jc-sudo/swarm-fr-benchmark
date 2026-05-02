import pandas as pd
import numpy as np
import scanpy as sc
import os, pathlib, csv, argparse, sys
import scipy.sparse as sp

def get_sparse_counts(ad):
    """Extracts sparse raw counts from an AnnData object."""
    if 'counts' in ad.layers:
        mtx = ad.layers['counts']
    else:
        mtx = ad.X
    return sp.csr_matrix(mtx) if not sp.issparse(mtx) else mtx.tocsr()

def is_variant_complete(folder):
    expected_files = ["Y_counts.csv", "Y_control_counts.csv", "P_perturbation.csv", "A_network.npz", "info.txt"]
    return all(os.path.exists(os.path.join(folder, f)) for f in expected_files)

def save_variant(folder, Y_sparse, Y_ctrl_sparse, P, A, info):
    os.makedirs(folder, exist_ok=True)
    print(f"   > Saving to {folder} ...")
    
    y_out_path = os.path.join(folder, "Y_counts.csv")
    chunk_size = 10000  
    
    with open(y_out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for start_idx in range(0, Y_sparse.shape[0], chunk_size):
            end_idx = min(start_idx + chunk_size, Y_sparse.shape[0])
            chunk_dense = np.round(Y_sparse[start_idx:end_idx].toarray().astype(np.float32))
            writer.writerows(chunk_dense)
            
    y_ctrl_out_path = os.path.join(folder, "Y_control_counts.csv")
    with open(y_ctrl_out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for start_idx in range(0, Y_ctrl_sparse.shape[0], chunk_size):
            end_idx = min(start_idx + chunk_size, Y_ctrl_sparse.shape[0])
            chunk_dense = np.round(Y_ctrl_sparse[start_idx:end_idx].toarray().astype(np.float32))
            writer.writerows(chunk_dense)
    
    pd.DataFrame(P).to_csv(os.path.join(folder, "P_perturbation.csv"), header=False, index=False)
    sp.save_npz(os.path.join(folder, "A_network.npz"), A.tocsr())
    
    with open(os.path.join(folder, "info.txt"), "w") as f:
        f.write(info)

def apply_genespider_snr_noise(Y_clean_sparse, target_snr, dispersion=1.0, chunk_size=10000):
    """
    Adaptation of GeneSpider2's dynamic SNR noise model.
    """
    if target_snr is None or target_snr <= 0:
        return Y_clean_sparse.copy()

    gene_means = Y_clean_sparse.mean(axis=0).A1
    inv_disp = 1.0 / dispersion
    dropout_probs = (inv_disp / (gene_means + 1e-9 + inv_disp)) ** inv_disp

    Y_log = Y_clean_sparse.copy()
    Y_log.data = np.log1p(Y_log.data)
    mean_log = float(Y_log.mean())
    Y_log.data = Y_log.data ** 2
    mean_log_sq = float(Y_log.mean())
    global_signal_var = max(0.0, mean_log_sq - (mean_log ** 2))
    log_std = np.sqrt(global_signal_var / target_snr)

    n_cells = Y_clean_sparse.shape[0]
    if n_cells == 0:
        return Y_clean_sparse.copy()
    noisy_blocks = []

    for start_idx in range(0, n_cells, chunk_size):
        end_idx = min(start_idx + chunk_size, n_cells)
        chunk_dense = Y_clean_sparse[start_idx:end_idx].toarray()
        
        # Apply Dropout Mask
        keep_mask = np.random.binomial(n=1, p=(1 - dropout_probs), size=chunk_dense.shape)
        chunk_dropped = chunk_dense * keep_mask
        
        # Apply Lognormal Noise
        multiplicative_noise = np.random.lognormal(mean=0.0, sigma=log_std, size=chunk_dropped.shape)
        chunk_noisy = np.round(chunk_dropped * multiplicative_noise).astype(np.float32)
        
        noisy_blocks.append(sp.csr_matrix(chunk_noisy))

    return sp.vstack(noisy_blocks, format='csr')

def sampled_mean_abs_corr(Y_csc, gene_indices, n_pairs=4000, rng=None):
    if len(gene_indices) < 2:
        return 0.0
    if rng is None:
        rng = np.random.default_rng(42)
    idx = np.asarray(list(gene_indices), dtype=int)
    pair_i = rng.choice(idx, size=n_pairs, replace=True)
    pair_j = rng.choice(idx, size=n_pairs, replace=True)
    valid = pair_i != pair_j
    pair_i = pair_i[valid]
    pair_j = pair_j[valid]
    vals = []
    for gi, gj in zip(pair_i, pair_j):
        xi = Y_csc[:, gi].toarray().ravel()
        xj = Y_csc[:, gj].toarray().ravel()
        si = xi.std()
        sj = xj.std()
        if si < 1e-12 or sj < 1e-12:
            continue
        c = np.corrcoef(xi, xj)[0, 1]
        if np.isfinite(c):
            vals.append(abs(c))
    return float(np.mean(vals)) if vals else 0.0

def process_dataset(
        dataset_name, h5ad_path, output_root, pert_col='perturbation', 
        ctrl_label='control', n_struct_levels=5, total_genes=5000
        ):
    print(f"\n{'='*60}")
    print(f"PROCESSING DATASET: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    dataset_out_dir = os.path.join(output_root, 'variants')
    os.makedirs(dataset_out_dir, exist_ok=True)

    print("Loading Data...")
    np.random.seed(42)
    try:
        adata = sc.read_h5ad(h5ad_path)
        if sp.issparse(adata.X) and not sp.isspmatrix_csr(adata.X):
            adata.X = adata.X.tocsr()
    except Exception as e:
        print(f"Failed to load {h5ad_path}: {e}")
        return

    if 'counts' in adata.layers:
        print("   > Found 'counts' layer. Overwriting .X with raw counts to prevent double-logging.")
        if not sp.issparse(adata.layers['counts']):
            adata.X = sp.csr_matrix(adata.layers['counts'])
        else:
            adata.X = adata.layers['counts'].copy()
    else:
        print("   > No 'counts' layer found. Assuming .X is raw and creating a backup.")
        adata.layers['counts'] = adata.X.copy()

    data_array = adata.X.data if sp.issparse(adata.X) else adata.X
    is_raw = np.allclose(data_array, np.round(data_array), atol=1e-4)

    if not is_raw:
        print("   > WARNING: Data contains non-integers. Reversing log1p (np.expm1)...")
        if sp.issparse(adata.X):
            adata.X.data = np.round(np.expm1(adata.X.data))
        else:
            adata.X = np.round(np.expm1(adata.X))
    else:
        print("   > Verified data is in standard linear/integer scale.")

    adata.layers['counts'] = adata.X.copy()

    unique_perts = adata.obs[pert_col].unique()
    if ctrl_label not in unique_perts:
        print(f"ERROR: Control label '{ctrl_label}' not found.")
        return
        
    print("   > Identifying Target Genes...")
    all_targets_raw = set(unique_perts)
    all_targets_raw.remove(ctrl_label)
    all_individual_targets = set()
    for t_raw in all_targets_raw:
        for t_split in str(t_raw).split('+'): 
            all_individual_targets.add(t_split.strip())

    valid_targets = [g for g in all_individual_targets if g in adata.var_names]
    if len(valid_targets) == 0:
        print("No target genes found in transcriptome. Skipping.")
        return

    print("   > Normalizing and Log-transforming...")
    print(f"Max value = {adata.X.max()}")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    print("   > Computing HVGs...")
    sc.pp.highly_variable_genes(adata, flavor='seurat', subset=False)

    actual_total = min(total_genes, adata.n_vars)
    genes_to_keep = set(valid_targets)
    hvg_ranked = adata.var.sort_values('dispersions_norm', ascending=False).index.tolist()
    for gene in hvg_ranked:
        if len(genes_to_keep) >= actual_total:
            break
        genes_to_keep.add(gene)

    ordered_genes_to_keep = [g for g in adata.var_names if g in genes_to_keep]
    adata = adata[:, ordered_genes_to_keep].copy()

    genes = adata.var_names.tolist()
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    n_genes = len(genes)

    ctrl_mask = adata.obs[pert_col] == ctrl_label
    adata_ctrl = adata[ctrl_mask]
    adata_pert = adata[~ctrl_mask]

    Y_ctrl_base = get_sparse_counts(adata_ctrl)
    Y_pert_base = get_sparse_counts(adata_pert)

    targets = adata_pert.obs[pert_col].values
    P_matrix = np.zeros((len(targets), n_genes))

    count_found = 0
    for i, target_str in enumerate(targets):
        found_for_cell = False
        for t in str(target_str).split('+'):
            t = t.strip()
            if t in gene_to_idx:
                P_matrix[i, gene_to_idx[t]] = -1
                found_for_cell = True
        if found_for_cell:
            count_found += 1

    max_cells_for_corr = 20000
    if adata_ctrl.n_obs > max_cells_for_corr:
        print(f"   > Subsampling {adata_ctrl.n_obs} cells down to {max_cells_for_corr} for memory-safe network inference.")
        adata_ctrl_for_corr = adata_ctrl.copy()
        sc.pp.subsample(adata_ctrl_for_corr, n_obs=max_cells_for_corr, random_state=42)
        Y_dense_ctrl_norm = adata_ctrl_for_corr.X.toarray() if sp.issparse(adata_ctrl_for_corr.X) else np.asarray(adata_ctrl_for_corr.X)
    else:
        Y_dense_ctrl_norm = adata_ctrl.X.toarray() if sp.issparse(adata_ctrl.X) else np.asarray(adata_ctrl.X)
    
    corr_matrix = np.corrcoef(Y_dense_ctrl_norm, rowvar=False) 
    np.fill_diagonal(corr_matrix, 0) 
    corr_matrix = np.nan_to_num(corr_matrix, 0) 

    gene_degrees = np.sum(np.abs(corr_matrix), axis=1)
    ranked_gene_indices = np.argsort(gene_degrees)

    corr_matrix[np.abs(corr_matrix) < 0.05] = 0
    A_baseline = sp.csr_matrix(corr_matrix)

    # --- SAVE GT REFERENCE ---
    gt_folder = os.path.join(dataset_out_dir, "GT_Reference")
    if not is_variant_complete(gt_folder):
        save_variant(gt_folder, Y_pert_base, Y_ctrl_base, P_matrix, A_baseline, f"Real {dataset_name} Data (GT)")

    # --- DEFINE NOISE ENVIRONMENTS ---
    snr_levels = [None, 1, 0.1, 0.01] # No noise, low noise, medium noise, high noise

    # --- STRUCTURE SWEEP ---
    print("\nGenerating Structure x Noise Variants...")
    total_genes_to_target = len(ranked_gene_indices)

    for i in range(1, n_struct_levels + 1):
        frac = i * 0.2
        n_target = int(total_genes_to_target * frac)
        target_gene_indices = list(ranked_gene_indices[:n_target])

        Y_struct = Y_pert_base.tocsc().copy()
        Y_ctrl_struct = Y_ctrl_base.tocsc().copy()
        A_mod = A_baseline.copy().tolil()
        
        n_pert_cells = Y_struct.shape[0]
        n_ctrl_cells = Y_ctrl_struct.shape[0]
        agg_pre_damage = sampled_mean_abs_corr(Y_struct, target_gene_indices, n_pairs=4000)

        for gene_idx in target_gene_indices:
            start_p, end_p = Y_struct.indptr[gene_idx], Y_struct.indptr[gene_idx+1]
            nnz_p = end_p - start_p
            if nnz_p > 0:
                new_rows_p = np.random.choice(n_pert_cells, nnz_p, replace=False)
                new_rows_p.sort() 
                Y_struct.indices[start_p:end_p] = new_rows_p
                np.random.shuffle(Y_struct.data[start_p:end_p])

            start_c, end_c = Y_ctrl_struct.indptr[gene_idx], Y_ctrl_struct.indptr[gene_idx+1]
            nnz_c = end_c - start_c
            if nnz_c > 0:
                new_rows_c = np.random.choice(n_ctrl_cells, nnz_c, replace=False)
                new_rows_c.sort()
                Y_ctrl_struct.indices[start_c:end_c] = new_rows_c
                np.random.shuffle(Y_ctrl_struct.data[start_c:end_c])

            A_mod[gene_idx, :] = 0
            A_mod[:, gene_idx] = 0

        agg_post_damage = sampled_mean_abs_corr(Y_struct, target_gene_indices, n_pairs=4000)
            
        print(f"     > Level {i} Shuffling Complete (Targeted {n_target} genes).")
        print(f"       - Mean Abs Correlation Pre-Shuffle:  {agg_pre_damage:.4f}")
        print(f"       - Mean Abs Correlation Post-Shuffle: {agg_post_damage:.4f}")

        Y_struct_csr = Y_struct.tocsr()
        Y_ctrl_struct_csr = Y_ctrl_struct.tocsr()

        for j, target_snr in enumerate(snr_levels, 1):
            folder_name = f"Structure_Level_{i}_SNR_{target_snr if target_snr else 'Infinity'}"
            folder = os.path.join(dataset_out_dir, folder_name)
            
            if not is_variant_complete(folder):
                Y_struct_noisy = apply_genespider_snr_noise(Y_clean_sparse=Y_struct_csr, target_snr=target_snr)
                Y_ctrl_struct_noisy = apply_genespider_snr_noise(Y_clean_sparse=Y_ctrl_struct_csr, target_snr=target_snr)
                info_string = f"Structure Level {i} (Frac: {frac}) | Noise Level {j} (SNR: {target_snr})"
                
                save_variant(folder, Y_struct_noisy, Y_ctrl_struct_noisy, P_matrix, A_mod.tocsr(), info_string)
            else:
                print(f"     > {folder_name} already exists. Skipping.")

    # --- MAGNITUDE SWEEP ---
    print("\nGenerating Magnitude x Noise Variants...")
    mu_ctrl = Y_ctrl_base.mean(axis=0).A1 

    shrink_factors = [0.8, 0.6, 0.4, 0.2, 0.0]
    for i, shrink_factor in enumerate(shrink_factors, 1):
        
        Y_mag_sparse_blocks = []
        chunk_size = 10000
        for start_idx in range(0, Y_pert_base.shape[0], chunk_size):
            end_idx = min(start_idx + chunk_size, Y_pert_base.shape[0])
            chunk_dense = Y_pert_base[start_idx:end_idx].toarray()
            
            effect = chunk_dense - mu_ctrl
            chunk_mag = np.maximum(mu_ctrl + (effect * shrink_factor), 0)
            chunk_mag_rounded = np.round(chunk_mag).astype(np.float32)
            
            Y_mag_sparse_blocks.append(sp.csr_matrix(chunk_mag_rounded))
            
        Y_mag_sparse_base = sp.vstack(Y_mag_sparse_blocks, format='csr')
        
        for j, target_snr in enumerate(snr_levels, 1):
            folder_name = f"Magnitude_Level_{i}_SNR_{target_snr if target_snr else 'Infinity'}"
            folder = os.path.join(dataset_out_dir, folder_name)
            
            if not is_variant_complete(folder):
                Y_mag_noisy = apply_genespider_snr_noise(Y_clean_sparse=Y_mag_sparse_base, target_snr=target_snr)
                Y_ctrl_noisy = apply_genespider_snr_noise(Y_clean_sparse=Y_ctrl_base, target_snr=target_snr)
                info_string = f"Magnitude Level {i} (Shrink: {shrink_factor}) | Noise Level {j} (SNR: {target_snr})"
                save_variant(folder, Y_mag_noisy, Y_ctrl_noisy, P_matrix, A_baseline, info_string)
            else:
                print(f"     > {folder_name} already exists. Skipping.")

    print(f"Finished {dataset_name}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate combinatorial variants for a specific dataset.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        help="The folder name of the dataset to process (e.g., 'vcc2025' or 'replogle2022_k562_gwps')"
    )

    parser.add_argument(
        "--base_path", 
        type=str, 
        default=os.path.expanduser("~/virtual-cell/virtual-cell-metrics/datasets/real"),
        help="The root directory containing the dataset folders."
    )
    
    args = parser.parse_args()

    dataset_folder = os.path.join(args.base_path, args.dataset)
    if not os.path.isdir(dataset_folder):
        print(f"ERROR: Dataset folder not found at {dataset_folder}")
        sys.exit(1)

    h5ad_files = list(pathlib.Path(dataset_folder).rglob("*.h5ad"))
    if not h5ad_files:
        print(f"ERROR: No .h5ad files found in {dataset_folder}")
        sys.exit(1)
        
    dataset_path = h5ad_files[0]
    output_root_folder = dataset_folder

    print(f"\n{'='*60}")
    print(f"PARALLEL JOB: {args.dataset.upper()}")
    print(f"File Size: {os.path.getsize(dataset_path) / (1024**3):.2f} GB")
    print(f"{'='*60}\n")

    if args.dataset == 'vcc2025':
        process_dataset(
            dataset_name=args.dataset,
            h5ad_path=dataset_path,
            output_root=output_root_folder,
            pert_col='target_gene',
            ctrl_label='non-targeting'
        )
    else:
        process_dataset(
            dataset_name=args.dataset,
            h5ad_path=dataset_path,
            output_root=output_root_folder
        )