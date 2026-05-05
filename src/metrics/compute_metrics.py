import os, sys, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import ranksums
from sklearn.decomposition import PCA
import concurrent.futures
from functools import partial
import anndata as ad
import pertpy as pt
from statsmodels.stats.multitest import multipletests
from utils.utils import load_sparse_from_csv
import warnings
warnings.filterwarnings('ignore')

global_X_pred = None
global_X_gt = None
global_precomputed_ctrl_pred = None
global_precomputed_ctrl_gt = None

class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._null_stdout = open(os.devnull, 'w')
        self._null_stderr = open(os.devnull, 'w')
        sys.stdout = self._null_stdout
        sys.stderr = self._null_stderr

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        # Kinda bandaid fix sorry: Close the file descriptors to prevent OS limits from crashing the job
        self._null_stdout.close()
        self._null_stderr.close()

def weighted_mse(Y_true, Y_pred, w):
    return np.average((Y_true - Y_pred) ** 2, weights=w, axis=1)

def weighted_rmse(Y_true, Y_pred, w):
    return np.sqrt(weighted_mse(Y_true, Y_pred, w))

def weighted_mae(Y_true, Y_pred, w):
    return np.average(np.abs(Y_true - Y_pred), weights=w, axis=1)

def weighted_r2(Y_true, Y_pred, w):
    mean_true = np.average(Y_true, weights=w, axis=1, keepdims=True)
    ss_tot = np.sum(w * (Y_true - mean_true)**2, axis=1)
    ss_res = np.sum(w * (Y_true - Y_pred)**2, axis=1)
    safe_division = np.divide(ss_res, ss_tot, out=np.full_like(ss_res, np.nan), where=ss_tot!=0)
    return 1 - safe_division

def weighted_pearson(Y_true, Y_pred, w):
    if np.sum(w) == 0:
        return np.full(Y_true.shape[0], np.nan)
        
    mean_t = np.average(Y_true, weights=w, axis=1, keepdims=True)
    mean_p = np.average(Y_pred, weights=w, axis=1, keepdims=True)
    
    cov = np.sum(w * (Y_true - mean_t) * (Y_pred - mean_p), axis=1).astype(float)
    var_t = np.sum(w * (Y_true - mean_t)**2, axis=1)
    var_p = np.sum(w * (Y_pred - mean_p)**2, axis=1)
    
    denominator = np.sqrt(var_t * var_p)
    return np.divide(cov, denominator, out=np.full_like(cov, np.nan, dtype=float), where=denominator!=0)

def weighted_cosine(Y_true, Y_pred, w):
    if np.sum(w) == 0:
        return np.full(Y_true.shape[0], np.nan)
        
    dot_product = np.sum(w * Y_true * Y_pred, axis=1).astype(float)
    norm_true = np.sqrt(np.sum(w * (Y_true ** 2), axis=1))
    norm_pred = np.sqrt(np.sum(w * (Y_pred ** 2), axis=1))
    
    denominator = norm_true * norm_pred
    return np.divide(dot_product, denominator, out=np.full_like(dot_product, np.nan, dtype=float), where=denominator!=0)

def calculate_pertpy_metrics(Y_true, Y_pred, mask=None, weights=None, allow_kl=True):
    if mask is not None:
        Y_t = Y_true[:, mask].copy()
        Y_p = Y_pred[:, mask].copy()
    elif weights is not None:
        sqrt_w = np.sqrt(weights)
        Y_t = Y_true * sqrt_w
        Y_p = Y_pred * sqrt_w
    else:
        Y_t = Y_true.copy()
        Y_p = Y_pred.copy()

    if Y_t.shape[1] == 0:
        return {}

    X_combined = np.vstack([Y_t, Y_p])
    obs = pd.DataFrame({'Expcategory': ['stimulated'] * len(Y_t) + ['imputed'] * len(Y_p)})
    
    adata = ad.AnnData(X=X_combined, obs=obs)
    adata.layers['X'] = adata.X 

    metrics_to_run = ['wasserstein', 'edistance', 'mmd']
    if allow_kl:
        metrics_to_run.append('sym_kldiv')

    results = {}
    for m in metrics_to_run:
        try:
            with SuppressOutput():
                Distance = pt.tools.Distance(metric=m, layer_key='X')
                pairwise_df = Distance.onesided_distances(
                    adata, groupby="Expcategory", selected_group='imputed', groups=["stimulated"]
                )
                perf = float(pairwise_df['stimulated'])
                if m == 'sym_kldiv': perf = np.log2(perf + 1)
                results[m] = perf
        except Exception:
            pass
    return results

def calculate_retrieval_metrics(target_pred_vec, target_pert_name, gt_universe_dict, weights):
    dists_l1, dists_l2 = [], []
    cosines, pearsons = [], []
    target_l1, target_l2, target_cos, target_pearson = None, None, None, None
    
    print(f"BEFORE Target shape inside calculate_retrieval_metrics: {np.shape(target_pred_vec)} | Weights shape inside calculate_retrieval_metrics: {np.shape(weights)}")
    target_pred_vec = np.asarray(target_pred_vec).flatten()
    weights = np.asarray(weights).flatten()
    print(f"AFTER Target shape inside calculate_retrieval_metrics: {np.shape(target_pred_vec)} | Weights shape inside calculate_retrieval_metrics: {np.shape(weights)}")

    w_sum = np.sum(weights)
    if w_sum == 0: return {}

    # Pre-calculate prediction norms/means for Cosine and Pearson
    norm_p = np.sqrt(np.sum(weights * (target_pred_vec ** 2)))
    mean_p = np.average(target_pred_vec, weights=weights)
    var_p = np.sum(weights * (target_pred_vec - mean_p)**2)
    
    for gt_name, gt_vec in gt_universe_dict.items():
        # L1 / L2 (Distances - Lower is better)
        diff = target_pred_vec - gt_vec
        d1 = np.sum(weights * np.abs(diff))
        d2 = np.sum(weights * (diff ** 2))
        dists_l1.append(d1)
        dists_l2.append(d2)
        
        # Cosine Similarity (Higher is better)
        dot = np.sum(weights * target_pred_vec * gt_vec)
        norm_g = np.sqrt(np.sum(weights * (gt_vec ** 2)))
        cos_sim = dot / (norm_p * norm_g) if (norm_p * norm_g) != 0 else 0.0
        cosines.append(cos_sim)

        # Pearson Correlation (Higher is better)
        mean_g = np.average(gt_vec, weights=weights)
        cov = np.sum(weights * (target_pred_vec - mean_p) * (gt_vec - mean_g))
        var_g = np.sum(weights * (gt_vec - mean_g)**2)
        pear_sim = cov / np.sqrt(var_p * var_g) if (var_p * var_g) != 0 else 0.0
        pearsons.append(pear_sim)
        
        if gt_name == target_pert_name:
            target_l1, target_l2 = d1, d2
            target_cos, target_pearson = cos_sim, pear_sim
            
    if target_l1 is None: return {} 

    dists_l1, dists_l2 = np.array(dists_l1), np.array(dists_l2)
    cosines, pearsons = np.array(cosines), np.array(pearsons)
    n_total = len(dists_l1)

    # Rank calculations (Note the flipped operators for similarities)
    rank_l1 = np.sum(dists_l1 < target_l1)
    rank_l2 = np.sum(dists_l2 < target_l2)
    rank_cos = np.sum(cosines > target_cos)
    rank_pearson = np.sum(pearsons > target_pearson)

    # Convert to 1.0 (Best) to -1.0 (Worst) scoring scale
    return {
        'PDS': 1.0 - (2.0 * (rank_l1 / n_total)),
        'NIR': 1.0 - (2.0 * (rank_l2 / n_total)),
        'Rank_Cosine': 1.0 - (2.0 * (rank_cos / n_total)),
        'Rank_Pearson': 1.0 - (2.0 * (rank_pearson / n_total)),
        'Centroid_Accuracy': 1.0 if rank_l2 == 0 else 0.0
    }

def calculate_des_vcc(pred_group, pred_ctrl, gt_group, gt_ctrl, max_k=None):
    eps = 1e-9
    p_pred, p_gt = [], []
    for i in range(pred_group.shape[1]):
        try: _, pp = ranksums(pred_group[:, i], pred_ctrl[:, i])
        except ValueError: pp = 1.0 
        p_pred.append(pp)
        
        try: _, pg = ranksums(gt_group[:, i], gt_ctrl[:, i])
        except ValueError: pg = 1.0
        p_gt.append(pg)

    reject_pred, _, _, _ = multipletests(np.nan_to_num(p_pred, nan=1.0), alpha=0.05, method='fdr_bh')
    reject_gt, _, _, _ = multipletests(np.nan_to_num(p_gt, nan=1.0), alpha=0.05, method='fdr_bh')

    g_pred_sig = np.where(reject_pred)[0]
    g_gt_sig = np.where(reject_gt)[0]

    n_gt_sig = len(g_gt_sig)
    if n_gt_sig == 0: return 1.0 if len(g_pred_sig) == 0 else 0.0

    k_eff = n_gt_sig if max_k is None else min(n_gt_sig, max_k)

    pred_mean_group = np.maximum(pred_group.mean(axis=0), 0)
    pred_mean_ctrl = np.maximum(pred_ctrl.mean(axis=0), 0) + eps
    lfc_pred = np.nan_to_num(np.log2((pred_mean_group + eps) / pred_mean_ctrl), nan=0.0, posinf=0.0, neginf=0.0)
    
    gt_mean_group = np.maximum(gt_group.mean(axis=0), 0)
    gt_mean_ctrl = np.maximum(gt_ctrl.mean(axis=0), 0) + eps
    lfc_gt = np.nan_to_num(np.log2((gt_mean_group + eps) / gt_mean_ctrl), nan=0.0, posinf=0.0, neginf=0.0)

    gt_sig_lfc = np.abs(lfc_gt[g_gt_sig])
    gt_sorted_idx = g_gt_sig[np.argsort(gt_sig_lfc)[::-1]] 
    gt_topk = set(gt_sorted_idx[:k_eff])

    if len(g_pred_sig) > 0:
        pred_sig_lfc = np.abs(lfc_pred[g_pred_sig])
        pred_sorted_idx = g_pred_sig[np.argsort(pred_sig_lfc)[::-1]]
        pred_topk = set(pred_sorted_idx[:k_eff])
    else:
        pred_topk = set()

    intersection = len(pred_topk & gt_topk)
    return intersection / k_eff

def calculate_des_robust(pred_group, pred_ctrl, gt_group, gt_ctrl, max_k=None):
    eps = 1e-9
    p_pred, p_gt = [], []
    for i in range(pred_group.shape[1]):
        try: _, pp = ranksums(pred_group[:, i], pred_ctrl[:, i])
        except ValueError: pp = 1.0 
        p_pred.append(pp)
        
        try: _, pg = ranksums(gt_group[:, i], gt_ctrl[:, i])
        except ValueError: pg = 1.0
        p_gt.append(pg)

    reject_pred, _, _, _ = multipletests(np.nan_to_num(p_pred, nan=1.0), alpha=0.05, method='fdr_bh')
    reject_gt, _, _, _ = multipletests(np.nan_to_num(p_gt, nan=1.0), alpha=0.05, method='fdr_bh')

    g_pred_sig = np.where(reject_pred)[0]
    g_gt_sig = np.where(reject_gt)[0]

    n_gt_sig = len(g_gt_sig)
    if n_gt_sig == 0: return 1.0 if len(g_pred_sig) == 0 else 0.0

    k_eff = n_gt_sig if max_k is None else min(n_gt_sig, max_k)

    pred_mean_group = np.maximum(pred_group.mean(axis=0), 0)
    pred_mean_ctrl = np.maximum(pred_ctrl.mean(axis=0), 0) + eps
    lfc_pred = np.nan_to_num(np.log2((pred_mean_group + eps) / pred_mean_ctrl), nan=0.0, posinf=0.0, neginf=0.0)
    
    gt_mean_group = np.maximum(gt_group.mean(axis=0), 0)
    gt_mean_ctrl = np.maximum(gt_ctrl.mean(axis=0), 0) + eps
    lfc_gt = np.nan_to_num(np.log2((gt_mean_group + eps) / gt_mean_ctrl), nan=0.0, posinf=0.0, neginf=0.0)

    gt_sig_lfc = np.abs(lfc_gt[g_gt_sig])
    gt_sorted_idx = g_gt_sig[np.argsort(gt_sig_lfc)[::-1]]
    gt_topk = set(gt_sorted_idx[:k_eff])

    if len(g_pred_sig) > 0:
        pred_sig_lfc = np.abs(lfc_pred[g_pred_sig])
        pred_sorted_idx = g_pred_sig[np.argsort(pred_sig_lfc)[::-1]]
        pred_topk = set(pred_sorted_idx[:k_eff])
    else:
        pred_topk = set()

    intersection = pred_topk & gt_topk
    union = pred_topk | gt_topk
    
    weights = np.maximum(np.abs(lfc_pred), np.abs(lfc_gt))
    union_weight = weights[list(union)].sum()
    if union_weight == 0: return 1.0 
    
    return weights[list(intersection)].sum() / union_weight

def evaluate_single_perturbation(pert_gene, pert_to_indices, deg_cont, deg_bin, deg_hybrid, gene_names, 
                                 mu_ctrl_pred, mu_ctrl_gt, mu_pert_pred, mu_pert_gt, 
                                 median_ctrl_pred, median_pert_pred, gt_raw_means):
    
    local_results = []
    clean_pert_name = pert_gene.replace("Pert_", "")
    if clean_pert_name not in deg_cont.index: 
        return []
        
    w_cont = np.abs(deg_cont.loc[clean_pert_name, gene_names].fillna(0).values)
    w_bin = deg_bin.loc[clean_pert_name, gene_names].fillna(0).values
    w_static = np.ones_like(w_cont)
    
    if clean_pert_name in deg_hybrid.index:
        w_hybrid = deg_hybrid.loc[clean_pert_name, gene_names].fillna(0).values
    else:
        w_hybrid = np.ones_like(w_cont) / len(w_cont)

    top_20_indices = np.argsort(w_cont)[-20:]
    mask_top20 = np.zeros_like(w_cont, dtype=bool)
    mask_top20[top_20_indices] = True
    mask_top20 = mask_top20 & (w_bin > 0)

    all_weights = {
        'DEG_Continuous': w_cont, 'DEG_Binary': w_bin, 'Static': w_static, 'Hybrid_Continuous': w_hybrid,
        'Top20_DEG_Continuous': w_cont * mask_top20, 'Top20_DEG_Binary': w_bin * mask_top20, 
        'Top20_Static': w_static * mask_top20, 'Top20_Hybrid_Continuous': w_hybrid * mask_top20
    }

    group_indices = pert_to_indices[pert_gene]
    block_p_raw = global_X_pred[group_indices].toarray()
    block_gt_raw = global_X_gt[group_indices].toarray()

    grp_mean_pred = block_p_raw.mean(axis=0)
    grp_mean_effect = grp_mean_pred - mu_ctrl_pred

    attack_scenarios = {
        'No_Attack': block_p_raw,
        'Mode_Collapse_Control': np.round(np.tile(mu_ctrl_pred, (len(group_indices), 1))),
        'Mode_Collapse_Mean_Perturbed': np.round(np.tile(mu_pert_pred, (len(group_indices), 1))),
        'Mode_Collapse_Median_Control': np.round(np.tile(median_ctrl_pred, (len(group_indices), 1))),
        'Mode_Collapse_Median_Perturbed': np.round(np.tile(median_pert_pred, (len(group_indices), 1))),
        'Normalization_Mismatch': np.log1p(block_p_raw),
        'Scaling_Exploit': np.round(np.tile(mu_ctrl_pred + (grp_mean_effect * 1e6), (len(group_indices), 1)))
    }

    # Convert precomputed 1D control lists back to 2D arrays for the Ranksums test
    ctrl_pred_mat = np.array(global_precomputed_ctrl_pred).T 
    ctrl_gt_mat = np.array(global_precomputed_ctrl_gt).T

    for attack_name, block_p_attacked in attack_scenarios.items():
        des_vcc = calculate_des_vcc(
            pred_group=block_p_attacked, 
            pred_ctrl=ctrl_pred_mat, 
            gt_group=block_gt_raw, 
            gt_ctrl=ctrl_gt_mat, 
            max_k=None
        )

        des_robust = calculate_des_robust(
            pred_group=block_p_attacked, pred_ctrl=ctrl_pred_mat, 
            gt_group=block_gt_raw, gt_ctrl=ctrl_gt_mat, max_k=None
        )
        
        local_results.extend([
            {'Sample_ID': clean_pert_name, 'Perturbation': clean_pert_name, 'Attack_Type': attack_name, 'Weighting_Strategy': 'None', 'Reference_Strategy': 'None', 'Aggregation_Strategy': 'Distribution', 'Feature_Space': 'Raw_Genes', 'Metric': 'DES_VCC', 'Value': des_vcc},
            {'Sample_ID': clean_pert_name, 'Perturbation': clean_pert_name, 'Attack_Type': attack_name, 'Weighting_Strategy': 'None', 'Reference_Strategy': 'None', 'Aggregation_Strategy': 'Distribution', 'Feature_Space': 'Raw_Genes', 'Metric': 'DES_Robust', 'Value': des_robust}
        ])

        strategies = {
            'Specific_Perturbation': (block_p_attacked, block_gt_raw, 0, 0),
            'Control_Mean_Shift': (block_p_attacked - mu_ctrl_pred, block_gt_raw - mu_ctrl_gt, mu_ctrl_pred, mu_ctrl_gt),
            'All_Perturbations_Mean_Shift': (block_p_attacked - mu_pert_pred, block_gt_raw - mu_pert_gt, mu_pert_pred, mu_pert_gt)
        }

        for ref_name, (mat_p, mat_t, shift_p, shift_t) in strategies.items():

            n_components = min(mat_t.shape[0], mat_t.shape[1], 256)
            if n_components > 5:
                pca = PCA(n_components=n_components, random_state=42)
                mat_t_pca = pca.fit_transform(mat_t)
                mat_p_pca = pca.transform(mat_p)
                
                pertpy_results_pca = calculate_pertpy_metrics(mat_t_pca, mat_p_pca, mask=None, weights=None, allow_kl=False)
                for m_name, m_val in pertpy_results_pca.items():
                    clean_m_name = m_name.capitalize() if m_name != 'mmd' else 'MMD'
                    if m_name == 'edistance': clean_m_name = 'E_Distance'
                        
                    local_results.append({
                        'Sample_ID': clean_pert_name, 'Perturbation': clean_pert_name, 'Attack_Type': attack_name,
                        'Weighting_Strategy': 'None', 'Reference_Strategy': ref_name, 'Aggregation_Strategy': 'Distribution', 
                        'Feature_Space': 'PCA_256', 'Metric': clean_m_name, 'Value': m_val
                    })

            for w_name, weights in all_weights.items():
                if np.sum(weights) == 0: continue
                is_top20 = w_name.startswith('Top20_')
                is_bin = 'Binary' in w_name

                if not is_top20:
                    universe = {p: (m - shift_t) for p, m in gt_raw_means.items()}
                    if attack_name == 'No_Attack': target_vec = grp_mean_pred
                    elif attack_name == 'Mode_Collapse_Control': target_vec = mu_ctrl_pred
                    elif attack_name == 'Mode_Collapse_Mean_Perturbed': target_vec = mu_pert_pred
                    elif attack_name == 'Mode_Collapse_Median_Control': target_vec = median_ctrl_pred
                    elif attack_name == 'Mode_Collapse_Median_Perturbed': target_vec = median_pert_pred
                    elif attack_name == 'Normalization_Mismatch': target_vec = np.mean(np.log1p(block_p_raw), axis=0)
                    elif attack_name == 'Scaling_Exploit': target_vec = np.round(np.tile(mu_ctrl_pred + (grp_mean_effect * 1e6), (len(group_indices), 1)))
                    else: target_vec = grp_mean_pred 
                    
                    target_vec = np.mean(mat_p, axis=0) # Converting 2D attack result matrix to a 1D centroid
                    r_metrics = calculate_retrieval_metrics(target_vec - shift_p, pert_gene, universe, weights)
                    for m_name, m_val in r_metrics.items():
                        local_results.append({
                            'Sample_ID': clean_pert_name, 'Perturbation': clean_pert_name, 'Attack_Type': attack_name,
                            'Weighting_Strategy': w_name, 'Reference_Strategy': ref_name, 'Aggregation_Strategy': 'Distribution', 
                            'Feature_Space': 'Raw_Genes', 'Metric': m_name, 'Value': m_val
                        })

                mask = weights > 0 if is_bin else None
                curr_p_mat = mat_p[:, mask] if mask is not None else mat_p
                curr_t_mat = mat_t[:, mask] if mask is not None else mat_t
                
                # If it's a binary metric, the weight of every remaining column is exactly 1
                # If it's continuous, use the actual weight values
                curr_w = np.ones(np.sum(mask)) if is_bin else weights

                if curr_p_mat.shape[1] < 1: continue
                
                mse_arr = weighted_mse(curr_t_mat, curr_p_mat, curr_w)
                rmse_arr = weighted_rmse(curr_t_mat, curr_p_mat, curr_w)
                mae_arr = weighted_mae(curr_t_mat, curr_p_mat, curr_w)
                pearson_arr = weighted_pearson(curr_t_mat, curr_p_mat, curr_w)
                r2_arr = weighted_r2(curr_t_mat, curr_p_mat, curr_w)
                cos_arr = weighted_cosine(curr_t_mat, curr_p_mat, curr_w)

                for local_idx, global_idx in enumerate(group_indices):
                    # FIX 3: Added Feature_Space to Spatial (None)
                    local_results.extend([
                        {'Sample_ID': global_idx, 'Perturbation': clean_pert_name, 'Attack_Type': attack_name, 'Weighting_Strategy': w_name, 'Reference_Strategy': ref_name, 'Aggregation_Strategy': 'None', 'Feature_Space': 'Raw_Genes', 'Metric': 'MSE', 'Value': mse_arr[local_idx]},
                        {'Sample_ID': global_idx, 'Perturbation': clean_pert_name, 'Attack_Type': attack_name, 'Weighting_Strategy': w_name, 'Reference_Strategy': ref_name, 'Aggregation_Strategy': 'None', 'Feature_Space': 'Raw_Genes', 'Metric': 'RMSE', 'Value': rmse_arr[local_idx]},
                        {'Sample_ID': global_idx, 'Perturbation': clean_pert_name, 'Attack_Type': attack_name, 'Weighting_Strategy': w_name, 'Reference_Strategy': ref_name, 'Aggregation_Strategy': 'None', 'Feature_Space': 'Raw_Genes', 'Metric': 'MAE', 'Value': mae_arr[local_idx]},
                        {'Sample_ID': global_idx, 'Perturbation': clean_pert_name, 'Attack_Type': attack_name, 'Weighting_Strategy': w_name, 'Reference_Strategy': ref_name, 'Aggregation_Strategy': 'None', 'Feature_Space': 'Raw_Genes', 'Metric': 'Pearson', 'Value': pearson_arr[local_idx]},
                        {'Sample_ID': global_idx, 'Perturbation': clean_pert_name, 'Attack_Type': attack_name, 'Weighting_Strategy': w_name, 'Reference_Strategy': ref_name, 'Aggregation_Strategy': 'None', 'Feature_Space': 'Raw_Genes', 'Metric': 'R2', 'Value': r2_arr[local_idx]},
                        {'Sample_ID': global_idx, 'Perturbation': clean_pert_name, 'Attack_Type': attack_name, 'Weighting_Strategy': w_name, 'Reference_Strategy': ref_name, 'Aggregation_Strategy': 'None', 'Feature_Space': 'Raw_Genes', 'Metric': 'Cosine', 'Value': cos_arr[local_idx]}
                    ])
                
                pb_t_mat = np.mean(curr_t_mat, axis=0, keepdims=True)
                pb_p_mat = np.mean(curr_p_mat, axis=0, keepdims=True)
                
                # Added Feature_Space to Spatial (Pseudobulk)
                local_results.extend([
                    {'Sample_ID': clean_pert_name, 'Perturbation': clean_pert_name, 'Attack_Type': attack_name, 'Weighting_Strategy': w_name, 'Reference_Strategy': ref_name, 'Aggregation_Strategy': 'Pseudobulk', 'Feature_Space': 'Raw_Genes', 'Metric': 'MSE', 'Value': weighted_mse(pb_t_mat, pb_p_mat, curr_w)[0]},
                    {'Sample_ID': clean_pert_name, 'Perturbation': clean_pert_name, 'Attack_Type': attack_name, 'Weighting_Strategy': w_name, 'Reference_Strategy': ref_name, 'Aggregation_Strategy': 'Pseudobulk', 'Feature_Space': 'Raw_Genes', 'Metric': 'RMSE', 'Value': weighted_rmse(pb_t_mat, pb_p_mat, curr_w)[0]},
                    {'Sample_ID': clean_pert_name, 'Perturbation': clean_pert_name, 'Attack_Type': attack_name, 'Weighting_Strategy': w_name, 'Reference_Strategy': ref_name, 'Aggregation_Strategy': 'Pseudobulk', 'Feature_Space': 'Raw_Genes', 'Metric': 'MAE', 'Value': weighted_mae(pb_t_mat, pb_p_mat, curr_w)[0]},
                    {'Sample_ID': clean_pert_name, 'Perturbation': clean_pert_name, 'Attack_Type': attack_name, 'Weighting_Strategy': w_name, 'Reference_Strategy': ref_name, 'Aggregation_Strategy': 'Pseudobulk', 'Feature_Space': 'Raw_Genes', 'Metric': 'Pearson', 'Value': weighted_pearson(pb_t_mat, pb_p_mat, curr_w)[0]},
                    {'Sample_ID': clean_pert_name, 'Perturbation': clean_pert_name, 'Attack_Type': attack_name, 'Weighting_Strategy': w_name, 'Reference_Strategy': ref_name, 'Aggregation_Strategy': 'Pseudobulk', 'Feature_Space': 'Raw_Genes', 'Metric': 'R2', 'Value': weighted_r2(pb_t_mat, pb_p_mat, curr_w)[0]},
                    {'Sample_ID': clean_pert_name, 'Perturbation': clean_pert_name, 'Attack_Type': attack_name, 'Weighting_Strategy': w_name, 'Reference_Strategy': ref_name, 'Aggregation_Strategy': 'Pseudobulk', 'Feature_Space': 'Raw_Genes', 'Metric': 'Cosine', 'Value': weighted_cosine(pb_t_mat, pb_p_mat, curr_w)[0]}
                ])

                if not is_top20:
                    allow_kl = (ref_name == 'Specific_Perturbation') and is_bin
                    
                    # We pass the raw (unmasked) matrices to Pertpy, 
                    # but we pass the cont_weights so Pertpy can handle the masking internally
                    cont_weights = None if is_bin else weights
                    pertpy_results_raw = calculate_pertpy_metrics(mat_t, mat_p, mask=mask, weights=cont_weights, allow_kl=allow_kl)
                    
                    for m_name, m_val in pertpy_results_raw.items():
                        clean_m_name = m_name.capitalize() if m_name != 'mmd' else 'MMD'
                        if m_name == 'sym_kldiv': clean_m_name = 'Sym_KL_Divergence'
                        if m_name == 'edistance': clean_m_name = 'E_Distance'
                            
                        local_results.append({
                            'Sample_ID': clean_pert_name, 'Perturbation': clean_pert_name, 'Attack_Type': attack_name,
                            'Weighting_Strategy': w_name, 'Reference_Strategy': ref_name, 'Aggregation_Strategy': 'Distribution', 
                            'Feature_Space': 'Raw_Genes', 'Metric': clean_m_name, 'Value': m_val
                        })
                        
    return local_results

def compute_performance(y_pred_path, y_ctrl_pred_path, y_gt_path, y_ctrl_gt_path, p_gt_path,
                        deg_binary_path, deg_cont_path, deg_hybrid_path, save_path,
                        chunk_idx=0, num_chunks=1):
    print(f"Processing target into: {save_path}")

    try:
        X_pred_raw = load_sparse_from_csv(y_pred_path)
        X_ctrl_pred_raw = load_sparse_from_csv(y_ctrl_pred_path)
        X_gt_raw = load_sparse_from_csv(y_gt_path)
        X_ctrl_gt_raw = load_sparse_from_csv(y_ctrl_gt_path)
        P_gt_sparse = load_sparse_from_csv(p_gt_path)
        
        deg_bin = pd.read_csv(deg_binary_path, index_col=0)
        deg_cont = pd.read_csv(deg_cont_path, index_col=0)
        deg_hybrid = pd.read_csv(deg_hybrid_path, index_col=0)

        dim0_match = (X_gt_raw.shape[0] == X_ctrl_gt_raw.shape[0])
        dim1_match = (X_gt_raw.shape[1] == X_ctrl_gt_raw.shape[1])

        if dim1_match and not dim0_match:
            X_pred, X_ctrl_pred, X_gt, X_ctrl_gt, P_gt = X_pred_raw, X_ctrl_pred_raw, X_gt_raw, X_ctrl_gt_raw, P_gt_sparse
        elif dim0_match and not dim1_match:
            X_pred = X_pred_raw.tocsr().T.tocsr()
            X_ctrl_pred = X_ctrl_pred_raw.tocsr().T.tocsr()
            X_gt = X_gt_raw.tocsr().T.tocsr()
            X_ctrl_gt = X_ctrl_gt_raw.tocsr().T.tocsr()
            P_gt = P_gt_sparse.tocsr().T.tocsr()
        elif dim0_match and dim1_match:
            # C_pert == C_ctrl (Synthetic Data Edge Case)
            if X_pred_raw.shape[0] > X_pred_raw.shape[1]: # Assumes that the total number of cells in the synthetic dataset is higher than the number of genes
                X_pred, X_ctrl_pred, X_gt, X_ctrl_gt, P_gt = X_pred_raw, X_ctrl_pred_raw, X_gt_raw, X_ctrl_gt_raw, P_gt_sparse
            else:
                X_pred = X_pred_raw.tocsr().T.tocsr()
                X_ctrl_pred = X_ctrl_pred_raw.tocsr().T.tocsr()
                X_gt = X_gt_raw.tocsr().T.tocsr()
                X_ctrl_gt = X_ctrl_gt_raw.tocsr().T.tocsr()
                P_gt = P_gt_sparse.tocsr().T.tocsr()
        else:
            raise ValueError("Orientation completely mismatched.")

        n_genes = X_pred.shape[1]
        gene_names = [f"G{i+1}" for i in range(n_genes)]
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        return

    mu_ctrl_pred = X_ctrl_pred.mean(axis=0).A1
    mu_ctrl_gt = X_ctrl_gt.mean(axis=0).A1
    
    pert_gt = []
    for i in range(X_gt.shape[0]):
        start_idx = P_gt.indptr[i]
        end_idx = P_gt.indptr[i+1]
        if end_idx > start_idx:
            pert_indices = P_gt.indices[start_idx:end_idx]
            name = "+".join([f"G{idx+1}" for idx in sorted(pert_indices)])
            pert_gt.append(f"Pert_{name}")
        else:
            pert_gt.append("None")

    pert_to_indices = {}
    for idx, p_name in enumerate(pert_gt):
        if p_name not in pert_to_indices: pert_to_indices[p_name] = []
        pert_to_indices[p_name].append(idx)

    gt_raw_means = {p: X_gt[idxs].toarray().mean(axis=0) for p, idxs in pert_to_indices.items()}

    all_pert_indices = [idx for p, idxs in pert_to_indices.items() if p != "None" for idx in idxs]
    mu_pert_pred = X_pred[all_pert_indices].mean(axis=0).A1
    mu_pert_gt = X_gt[all_pert_indices].mean(axis=0).A1
    
    sparse_pred_pert = X_pred[all_pert_indices].tocsc()
    sparse_ctrl_pred = X_ctrl_pred.tocsc()
    sparse_ctrl_gt = X_ctrl_gt.tocsc()
    
    median_pert_pred = np.zeros(n_genes)
    median_ctrl_pred = np.zeros(n_genes)
    
    for i in tqdm(range(n_genes), desc="Calculating Medians"):
        median_pert_pred[i] = np.median(sparse_pred_pert[:, i].toarray().flatten())
        median_ctrl_pred[i] = np.median(sparse_ctrl_pred[:, i].toarray().flatten())

    global global_X_pred, global_X_gt, global_precomputed_ctrl_pred, global_precomputed_ctrl_gt
    global_X_pred = X_pred
    global_X_gt = X_gt
    global_precomputed_ctrl_pred = [sparse_ctrl_pred[:, i].toarray().flatten() for i in range(n_genes)]
    global_precomputed_ctrl_gt = [sparse_ctrl_gt[:, i].toarray().flatten() for i in range(n_genes)]
    
    unique_perts = [p for p in list(pert_to_indices.keys()) if p != "None"]
    unique_perts = sorted(unique_perts)

    if num_chunks > 1:
        chunk_size = int(np.ceil(len(unique_perts) / num_chunks))
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(unique_perts))
        unique_perts = unique_perts[start_idx:end_idx]
        
        # Appended chunk identifier to the save path
        base, ext = os.path.splitext(save_path)
        save_path = f"{base}_chunk_{chunk_idx}{ext}"

    worker_func = partial(
        evaluate_single_perturbation,
        pert_to_indices=pert_to_indices, deg_cont=deg_cont, deg_bin=deg_bin, deg_hybrid=deg_hybrid,
        gene_names=gene_names, mu_ctrl_pred=mu_ctrl_pred, mu_ctrl_gt=mu_ctrl_gt,
        mu_pert_pred=mu_pert_pred, mu_pert_gt=mu_pert_gt,
        median_ctrl_pred=median_ctrl_pred, median_pert_pred=median_pert_pred, gt_raw_means=gt_raw_means
    )

    results_list = []
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    max_workers = max(1, slurm_cpus)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for local_results in tqdm(executor.map(worker_func, unique_perts), total=len(unique_perts), desc="Computing Metrics"):
            results_list.extend(local_results)

    if results_list:
        df_res = pd.DataFrame(results_list)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_res.to_csv(save_path, index=False)
        print(f"Done. Saved to {save_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics for a dataset variant.")
    parser.add_argument("--dataset", type=str, required=True, help="Folder name of the dataset")
    parser.add_argument("--variant", type=str, required=True, help="Folder name of the variant (e.g., 'GT_Reference' or 'Structure_Level_3_SNR_0.01')")
    parser.add_argument("--base_path", type=str, default=("../../datasets/real"), help="Root directory containing dataset folders")
    parser.add_argument("--results_path", type=str, default=("../../results/real"), help="Root directory to save results")
    parser.add_argument("--chunk_idx", type=int, default=0, help="The index of the chunk to process")
    parser.add_argument("--num_chunks", type=int, default=1, help="Total number of chunks to split the job into")
    args = parser.parse_args()

    dataset_variant_root = os.path.join(args.base_path, args.dataset, "variants")
    dataset_results_root = os.path.join(args.results_path, args.dataset)
    gt_dir = os.path.join(dataset_variant_root, "GT_Reference")
    
    deg_binary_path = os.path.join(gt_dir, "DEG_Adjacency_Matrix_binary.csv")
    deg_cont_path = os.path.join(gt_dir, "DEG_Adjacency_Matrix_cont.csv")
    deg_hybrid_path = os.path.join(gt_dir, "DEG_Adjacency_Matrix_hybrid.csv")

    if not (os.path.exists(deg_binary_path) and os.path.exists(deg_cont_path) and os.path.exists(deg_hybrid_path)):
        print(f"ERROR: Missing DEG weights in {gt_dir}")
        sys.exit(1)

    print(f"\n{'='*60}\nEVALUATING METRICS FOR: {args.dataset.upper()} | {args.variant.upper()}\n{'='*60}")

    variant_dir = os.path.join(dataset_variant_root, args.variant)
    
    if args.variant == "GT_Reference":
        save_filename = "performance_metrics_gt_baseline.csv"
    else:
        save_filename = f"performance_metrics_{args.variant.lower()}.csv"
        
    variant_save_path = os.path.join(dataset_results_root, save_filename)

    if not os.path.exists(os.path.join(variant_dir, "Y_counts.csv")):
        print(f"ERROR: {args.variant} is missing Y_counts.csv. Skipping.")
        sys.exit(1)
        
    if os.path.exists(variant_save_path):
        print(f"SUCCESS: Results already exist at {variant_save_path}. Skipping.")
        sys.exit(0)

    compute_performance(
        y_pred_path=os.path.join(variant_dir, "Y_counts.csv"),
        y_ctrl_pred_path=os.path.join(variant_dir, "Y_control_counts.csv"),
        y_gt_path=os.path.join(gt_dir, "Y_counts.csv"),
        y_ctrl_gt_path=os.path.join(gt_dir, "Y_control_counts.csv"),
        p_gt_path=os.path.join(gt_dir, "P_perturbation.csv"),
        deg_binary_path=deg_binary_path, deg_cont_path=deg_cont_path, deg_hybrid_path=deg_hybrid_path,
        save_path=variant_save_path,
        chunk_idx=args.chunk_idx,
        num_chunks=args.num_chunks
    )
