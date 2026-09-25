"""Scoring-rule sensitivity analysis across metric formulations and exploit subsets.

Demonstrates that the SWARM > literature ranking, reported F/R improvements, and the
best-standalone metric superiority survive perturbations of the scoring rule across four axes:

    sMAPE       -> MAPE                      (degradation-sensitivity formulation)
    harmonic    -> min                       (combining F_str and F_mag)
    anchor l=1  -> anchor l=0 (ground truth) (degradation-sensitivity reference anchor)
    full exploit set -> leave-one-exploit-out (robustness worst-case set)

Operates on aggregated per-metric overall CSVs without requiring full grid recomputation.

Usage:
    python -m src.rebuttals.scoring_rule_sensitivity \
        --overall-dir $SCRATCH/virtual-cell/virtual-cell-metrics/analysis/overall_real \
        --output-dir  $SCRATCH/virtual-cell/virtual-cell-metrics/analysis/rebuttal/scoring_sensitivity_real
"""

import os
import glob
import argparse
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

# Metric directionality
HIB_METRICS = ['Pearson', 'R2', 'PDS', 'NIR', 'Centroid_Accuracy', 'DES_Robust',
               'DES_VCC', 'Cosine', 'Rank_Cosine', 'Rank_Pearson']
LIB_METRICS = ['MSE', 'MAE', 'Wasserstein', 'Sym_KL_Divergence', 'E_Distance', 'MMD', 'RMSE']
HACK_THRESHOLD = 0.01
FIDELITY_EPSILON = 1e-5
DEFICIT_EPSILON = 1e-6

# Standard literature metric configurations (n=28)
LITERATURE_METRICS = {
    'MSE_Static_Specific_Perturbation_None_Raw_Genes',
    'MSE_DEG_Continuous_Specific_Perturbation_None_Raw_Genes',
    'RMSE_Static_Specific_Perturbation_None_Raw_Genes',
    'RMSE_Top20_DEG_Binary_Specific_Perturbation_None_Raw_Genes',
    'Pearson_Static_Specific_Perturbation_None_Raw_Genes',
    'Pearson_Static_Control_Mean_Shift_None_Raw_Genes',
    'Pearson_Static_All_Perturbations_Mean_Shift_None_Raw_Genes',
    'Pearson_DEG_Binary_Control_Mean_Shift_None_Raw_Genes',
    'Pearson_DEG_Binary_All_Perturbations_Mean_Shift_None_Raw_Genes',
    'Pearson_Top20_DEG_Binary_Control_Mean_Shift_None_Raw_Genes',
    'Pearson_Top20_DEG_Binary_All_Perturbations_Mean_Shift_None_Raw_Genes',
    'R2_Static_Control_Mean_Shift_None_Raw_Genes',
    'R2_Static_All_Perturbations_Mean_Shift_None_Raw_Genes',
    'R2_DEG_Binary_Control_Mean_Shift_None_Raw_Genes',
    'R2_DEG_Binary_All_Perturbations_Mean_Shift_None_Raw_Genes',
    'R2_DEG_Continuous_Control_Mean_Shift_None_Raw_Genes',
    'R2_DEG_Continuous_All_Perturbations_Mean_Shift_None_Raw_Genes',
    'Cosine_Static_Control_Mean_Shift_None_Raw_Genes',
    'Rank_Cosine_Static_Control_Mean_Shift_Distribution_Raw_Genes',
    'NIR_Static_Control_Mean_Shift_Distribution_Raw_Genes',
    'PDS_Static_Control_Mean_Shift_Distribution_Raw_Genes',
    'Centroid_Accuracy_Static_Control_Mean_Shift_Distribution_Raw_Genes',
    'E_Distance_Static_Specific_Perturbation_Distribution_Raw_Genes',
    'Wasserstein_Static_Specific_Perturbation_Distribution_Raw_Genes',
    'Sym_KL_Divergence_DEG_Binary_Specific_Perturbation_Distribution_Raw_Genes',
    'MMD_None_Specific_Perturbation_Distribution_PCA_256',
    'MAE_Static_Specific_Perturbation_None_Raw_Genes',
    'DES_VCC_None_None_Distribution_Raw_Genes',
}


@dataclass
class ScoringRule:
    """A single configuration point in scoring-rule space. Defaults match the primary rule."""
    name: str
    magnitude_error: str = 'smape'        # 'smape' | 'mape'
    fidelity_combine: str = 'harmonic'    # 'harmonic' | 'min' | 'product'
    anchor: str = 'l1'                    # 'l1' | 'l0'
    excluded_exploit: Optional[str] = None  # None or specific Attack_Type to drop


def default_rules() -> List[ScoringRule]:
    """Primary rule plus candidate scoring perturbations."""
    return [
        ScoringRule('baseline'),
        ScoringRule('mape', magnitude_error='mape'),
        ScoringRule('combine_min', fidelity_combine='min'),
        ScoringRule('anchor_l0', anchor='l0'),
    ]


def metric_is_hib(filename: str) -> Optional[bool]:
    """True = higher-is-better, False = lower-is-better, None = unknown."""
    for m in HIB_METRICS:
        if filename.startswith(m):
            return True
    for m in LIB_METRICS:
        if filename.startswith(m):
            return False
    return None


def load_overall(path: str) -> pd.DataFrame:
    """Read one aggregated per-metric overall file and normalize SNR values."""
    df = pd.read_csv(path, low_memory=False, dtype={'SNR': str}, compression='infer')
    if 'SNR' in df.columns:
        df['SNR'] = df['SNR'].astype(str).str.strip().str.lower()
        df['SNR'] = df['SNR'].replace({'inf': 'infinity', '1': '1.0', '1.00': '1.0'})
    else:
        df['SNR'] = 'infinity'
    return df


def _correctness(levels, scores, is_hib, magnitude_error, anchor_val):
    """Directional correctness x degradation sensitivity."""
    tau, _ = kendalltau(levels, scores)
    if np.isnan(tau):
        tau = 0.0
    monotonicity = -tau if is_hib else tau

    eps = FIDELITY_EPSILON
    errs = []
    if anchor_val is None:
        start_val = scores[0]
        compare = scores[1:]
    else:
        start_val = anchor_val
        compare = scores

    for current_val in compare:
        num = abs(current_val - start_val)
        if magnitude_error == 'mape':
            denom = abs(start_val) + eps
        else:  # smape
            denom = abs(start_val) + abs(current_val) + eps
        errs.append(num / denom)

    magnitude_factor = np.mean(errs) if errs else 0.0
    return max(0.0, monotonicity * magnitude_factor)


def _combine_fidelity(struct_avg, mag_avg, mode):
    """Combine Structure and Magnitude fidelity into overall Fidelity."""
    if struct_avg == 0.0 or mag_avg == 0.0:
        return 0.0
    if mode == 'min':
        return min(struct_avg, mag_avg)
    if mode == 'product':
        return struct_avg * mag_avg
    return (2.0 * struct_avg * mag_avg) / (struct_avg + mag_avg)


def fidelity_for_df(df, is_hib, rule: ScoringRule) -> float:
    """Mean Fidelity across datasets for one metric configuration under `rule`."""
    na = df[df['Attack_Type'] == 'No_Attack'].copy()
    if na.empty:
        return np.nan

    per_dataset = []
    for dataset, d_data in na.groupby('Dataset'):

        anchor_val = None
        if rule.anchor == 'l0':
            gt = df[(df['Variant'] == 'GT_Baseline') & (df['Attack_Type'] == 'No_Attack')
                    & (df['Dataset'] == dataset)]
            if gt.empty:
                continue
            anchor_val = gt['Value'].values[0]

        def sweep(variant_type, target_snr):
            subset = d_data[(d_data['Variant'] == variant_type) & (d_data['SNR'] == target_snr)]
            if subset.empty:
                return [], []
            subset = subset.sort_values('Level')
            return subset['Level'].tolist(), subset['Value'].tolist()

        snrs = ['infinity', '1.0', '0.1', '0.01']
        struct = [sweep('Structure', s) for s in snrs]
        mag = [sweep('Magnitude', s) for s in snrs]

        if any(len(v) != 5 for _, v in struct) or any(len(v) != 5 for _, v in mag):
            continue

        struct_rhos = [_correctness(lv, val, is_hib, rule.magnitude_error, anchor_val)
                       for lv, val in struct]
        mag_rhos = [_correctness(lv, val, is_hib, rule.magnitude_error, anchor_val)
                    for lv, val in mag]

        fidelity = _combine_fidelity(np.average(struct_rhos), np.average(mag_rhos),
                                     rule.fidelity_combine)
        per_dataset.append(fidelity)

    return float(np.mean(per_dataset)) if per_dataset else np.nan


def robustness_for_df(df, is_hib, rule: ScoringRule) -> float:
    """Mean Overall_Robustness across datasets for one metric configuration under `rule`."""
    per_dataset = []

    for dataset, d_data in df.groupby('Dataset'):
        struct_envs = d_data[d_data['Variant'] == 'Structure'][['Level', 'SNR']].drop_duplicates()
        mag_envs = d_data[d_data['Variant'] == 'Magnitude'][['Level', 'SNR']].drop_duplicates()
        if len(struct_envs) < 20 or len(mag_envs) < 20:
            continue

        gt_row = d_data[(d_data['Variant'] == 'GT_Baseline') & (d_data['Attack_Type'] == 'No_Attack')]
        if gt_row.empty:
            continue
        gt_score = gt_row['Value'].values[0]

        baseline_df = d_data[d_data['Attack_Type'] == 'No_Attack'].copy()
        baseline_map = baseline_df.set_index(['Variant', 'Level', 'SNR'])['Value'].to_dict()

        attack_df = d_data[(d_data['Attack_Type'] != 'No_Attack')
                           & (d_data['Variant'] != 'GT_Baseline')].copy()

        # Leave-one-exploit-out
        if rule.excluded_exploit is not None:
            attack_df = attack_df[attack_df['Attack_Type'] != rule.excluded_exploit].copy()

        if attack_df.empty:
            continue

        attack_df['Score_GT'] = gt_score
        attack_df['Score_Baseline'] = attack_df.apply(
            lambda row: baseline_map.get((row['Variant'], row['Level'], row['SNR']), np.nan), axis=1)
        attack_df = attack_df.dropna(subset=['Score_Baseline'])

        if is_hib:
            attack_df['Deficit'] = attack_df['Score_GT'] - attack_df['Score_Baseline']
            attack_df['Gain'] = attack_df['Value'] - attack_df['Score_Baseline']
        else:
            attack_df['Deficit'] = attack_df['Score_Baseline'] - attack_df['Score_GT']
            attack_df['Gain'] = attack_df['Score_Baseline'] - attack_df['Value']

        attack_df = attack_df[attack_df['Deficit'] > DEFICIT_EPSILON].copy()
        if attack_df.empty:
            continue

        attack_df['False_Gain'] = np.maximum(0, attack_df['Gain'])
        attack_df['Relative_Gain'] = attack_df['False_Gain'] / (attack_df['Deficit'] + 1e-9)
        attack_df['Relative_Gain'] = np.clip(attack_df['Relative_Gain'], 0.0, 1.0)

        attack_scores = {}
        for attack in attack_df['Attack_Type'].unique():
            subset = attack_df[attack_df['Attack_Type'] == attack]
            if (subset['Relative_Gain'] > HACK_THRESHOLD).any():
                successful = subset[subset['Relative_Gain'] > HACK_THRESHOLD]
                attack_scores[attack] = 1.0 - successful['Relative_Gain'].mean()
            else:
                attack_scores[attack] = 1.0

        final = min(attack_scores.values()) if attack_scores else 1.0
        per_dataset.append(max(0.0, min(1.0, final)))

    return float(np.mean(per_dataset)) if per_dataset else np.nan


def rank_metrics(overall_dir: str, rule: ScoringRule) -> pd.DataFrame:
    """Compute F, R, and harmonic mean across all configs under `rule`."""
    files = sorted(glob.glob(os.path.join(overall_dir, "*.csv"))
                   + glob.glob(os.path.join(overall_dir, "*.csv.zst")))
    records = []
    for path in files:
        filename = os.path.basename(path)
        raw_name = filename.replace('.csv.zst', '').replace('.csv', '')
        is_hib = metric_is_hib(filename)
        if is_hib is None:
            continue
        try:
            df = load_overall(path)
            f_score = fidelity_for_df(df, is_hib, rule)
            r_score = robustness_for_df(df, is_hib, rule)
            if np.isnan(f_score) or np.isnan(r_score):
                continue
            h_score = (2 * f_score * r_score) / (f_score + r_score + 1e-9)
            records.append({
                'Metric': raw_name,
                'Fidelity': f_score,
                'Robustness': r_score,
                'Harmonic': h_score,
                'Is_Literature': raw_name in LITERATURE_METRICS,
            })
        except Exception as e:
            print(f"  [warn] {filename}: {e}")

    df_all = pd.DataFrame(records)
    if df_all.empty:
        return df_all
    df_all = df_all.sort_values('Harmonic', ascending=False).reset_index(drop=True)
    df_all['Rank'] = df_all.index + 1
    return df_all


def summarize_rule(df_all: pd.DataFrame) -> dict:
    """Compute summary statistics comparing the top 28 SWARM configurations to the 28 literature metrics."""
    df_lit = df_all[df_all['Is_Literature']]
    df_swarm = df_all[~df_all['Is_Literature']]

    lit_f, lit_r = df_lit['Fidelity'].mean(), df_lit['Robustness'].mean()
    n = len(LITERATURE_METRICS)
    top_swarm = df_swarm.sort_values('Harmonic', ascending=False).head(n)
    sw_f, sw_r = top_swarm['Fidelity'].mean(), top_swarm['Robustness'].mean()

    best = df_all.iloc[0]
    return {
        'Best_Standalone_Metric': best['Metric'],
        'Best_Standalone_F': best['Fidelity'],
        'Best_Standalone_R': best['Robustness'],
        'Max_F': df_all['Fidelity'].max(),
        'Max_R': df_all['Robustness'].max(),
        'Lit_F_mean': lit_f,
        'Lit_R_mean': lit_r,
        'SWARM_topN_F_mean': sw_f,
        'SWARM_topN_R_mean': sw_r,
        'F_improvement_%': ((sw_f - lit_f) / lit_f) * 100 if lit_f else np.nan,
        'R_improvement_%': ((sw_r - lit_r) / lit_r) * 100 if lit_r else np.nan,
        'Top_Is_SWARM': not bool(best['Is_Literature']),
        'N_configs': len(df_all),
    }


def run(overall_dir: str, output_dir: str, rules: Optional[List[ScoringRule]] = None):
    os.makedirs(output_dir, exist_ok=True)
    rankings_dir = os.path.join(output_dir, 'rankings')
    os.makedirs(rankings_dir, exist_ok=True)

    if rules is None:
        rules = default_rules()

    files = sorted(glob.glob(os.path.join(overall_dir, "*.csv"))
                   + glob.glob(os.path.join(overall_dir, "*.csv.zst")))
    if not files:
        print(f"No aggregated CSVs found in {overall_dir}")
        return
    probe = load_overall(files[0])
    exploits = sorted(a for a in probe['Attack_Type'].unique()
                      if a != 'No_Attack')
    print(f"Discovered {len(exploits)} exploit(s) for leave-one-out: {exploits}")
    for atk in exploits:
        rules.append(ScoringRule(f'drop_{atk}', excluded_exploit=atk))

    baseline_ranking = None
    summary_rows = []

    for rule in rules:
        print(f"\n=== Scoring rule: {rule.name} ===")
        df_all = rank_metrics(overall_dir, rule)
        if df_all.empty:
            print("  no rankable metrics; skipping.")
            continue
        df_all.to_csv(os.path.join(rankings_dir, f'{rule.name}.csv'), index=False)

        summary = summarize_rule(df_all)
        summary['Rule'] = rule.name

        if rule.name == 'baseline':
            baseline_ranking = df_all.set_index('Metric')['Rank']
            summary['Spearman_vs_baseline'] = 1.0
        elif baseline_ranking is not None:
            this_rank = df_all.set_index('Metric')['Rank']
            shared = baseline_ranking.index.intersection(this_rank.index)
            if len(shared) > 2:
                rho, _ = spearmanr(baseline_ranking.loc[shared], this_rank.loc[shared])
                summary['Spearman_vs_baseline'] = rho
            else:
                summary['Spearman_vs_baseline'] = np.nan
        else:
            summary['Spearman_vs_baseline'] = np.nan

        summary_rows.append(summary)
        print(f"  best standalone: {summary['Best_Standalone_Metric']} "
              f"(F={summary['Best_Standalone_F']:.3f}, R={summary['Best_Standalone_R']:.3f})")
        print(f"  SWARM vs Lit: +{summary['F_improvement_%']:.1f}% F, "
              f"+{summary['R_improvement_%']:.1f}% R | top is SWARM: {summary['Top_Is_SWARM']} "
              f"| Spearman vs baseline: {summary['Spearman_vs_baseline']:.3f}")

    if not summary_rows:
        print("No summaries produced.")
        return

    cols = ['Rule', 'Best_Standalone_Metric', 'Best_Standalone_F', 'Best_Standalone_R',
            'Max_F', 'Max_R', 'Lit_F_mean', 'Lit_R_mean', 'SWARM_topN_F_mean',
            'SWARM_topN_R_mean', 'F_improvement_%', 'R_improvement_%', 'Top_Is_SWARM',
            'Spearman_vs_baseline', 'N_configs']
    summary_df = pd.DataFrame(summary_rows)[cols]
    out_csv = os.path.join(output_dir, 'scoring_rule_sensitivity.csv')
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved sensitivity summary: {out_csv}")
    print(f"Per-rule full rankings in: {rankings_dir}")
    print("\n--- Summary ---")
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(summary_df.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="Scoring-rule sensitivity analysis.")
    p.add_argument('--overall-dir', default=None,
                   help="Directory of aggregated per-metric `overall_*` CSVs.")
    p.add_argument('--output-dir', default=None,
                   help="Where to write the summary + per-rule rankings.")
    p.add_argument('--datatype', choices=['real', 'synthetic'], default='real',
                   help="Used to build default paths (default: 'real').")
    args = p.parse_args()

    overall_dir = args.overall_dir or os.path.expandvars(
        f"$SCRATCH/virtual-cell/virtual-cell-metrics/analysis/overall_{args.datatype}")
    output_dir = args.output_dir or os.path.expandvars(
        f"$SCRATCH/virtual-cell/virtual-cell-metrics/analysis/rebuttal/scoring_sensitivity_{args.datatype}")

    print(f"Overall dir: {overall_dir}")
    print(f"Output dir:  {output_dir}")
    run(overall_dir, output_dir)


if __name__ == "__main__":
    main()
