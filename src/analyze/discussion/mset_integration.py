"""Sensitivity analysis: M-Set gain reflects genuine multi-metric integration.

Evaluates two key properties of the reference M-Set at the selected ensemble size:

(a) Integration vs. single-metric ceiling:
    - Single-metric ceiling: the best F / R / combined score any single constituent achieves
      as a standalone metric, contrasted against the M-Set's joint F / R / combined score.
    - Per-dataset / per-axis winner diversity: how many distinct members achieve the top score
      on at least one dataset-axis. If the winner differs across datasets and axes, no
      single metric replicates the M-Set coverage.

(b) Aggregation rule sensitivity:
    - Re-scores the reference M-Set under a 2x2 grid of aggregation rules
      (member-combination in {max (primary), mean}, exploit-combination in {min (primary), mean})
      to test whether the ensemble consistently beats the best standalone metric.

Operates directly on pre-computed per-dataset fidelity and robustness files.

Usage:
    python -m src.rebuttals.mset_integration \
        --r-real ".../analysis/robustness_real" \
        --f-real ".../analysis/fidelity_real" \
        --r-synth ".../analysis/robustness_synthetic" \
        --f-synth ".../analysis/fidelity_synthetic" \
        --output-dir ".../analysis/rebuttal/mset_integration" \
        --set-size 6
"""

import os
import argparse

import numpy as np
import pandas as pd

# Support both module execution and direct script execution
try:
    from .mset_stability import (
        load_metric_records, all_datasets_in, build_pool, optimal_mset,
        core_metric_of, class_of,
    )
except ImportError:
    from mset_stability import (
        load_metric_records, all_datasets_in, build_pool, optimal_mset,
        core_metric_of, class_of,
    )


def harmonic(f, r):
    return (2 * f * r) / (f + r + 1e-9)


def attack_cols_of(records):
    return sorted(set().union(*(r['R_Scores'].keys() for r in records)))


def standalone_scores(record, datasets, attack_cols):
    """Score one metric config as a standalone metric, exactly matching the leaderboard:
    F = mean over datasets of harmonic(struct, mag); R = min over attacks of mean-over-datasets."""
    struct = np.array([record['F_Struct_Scores'].get(ds, 0.0) for ds in datasets])
    mag = np.array([record['F_Mag_Scores'].get(ds, 0.0) for ds in datasets])
    f = float(np.mean(harmonic(struct, mag)))
    attack_means = [np.mean([record['R_Scores'].get(ac, {}).get(ds, 0.0) for ds in datasets])
                    for ac in attack_cols]
    r = min(attack_means) if attack_means else 0.0
    return {'F': f, 'R': r, 'Combined': harmonic(f, r)}


def ensemble_scores(members, datasets, attack_cols, member_combine, attack_combine):
    """Score an M-Set under a chosen aggregation rule.
      member_combine: 'max' (primary) or 'mean' -- how members are combined per dataset/axis
      attack_combine: 'min' (primary, worst-case) or 'mean' -- how attacks are combined for R
    """
    n = len(members)
    struct = np.zeros((n, len(datasets)))
    mag = np.zeros((n, len(datasets)))
    R = np.zeros((n, len(datasets), len(attack_cols)))
    for i, m in enumerate(members):
        for j, ds in enumerate(datasets):
            struct[i, j] = m['F_Struct_Scores'].get(ds, 0.0)
            mag[i, j] = m['F_Mag_Scores'].get(ds, 0.0)
            for a, ac in enumerate(attack_cols):
                R[i, j, a] = m['R_Scores'].get(ac, {}).get(ds, 0.0)

    combine = np.max if member_combine == 'max' else np.mean
    c_struct = combine(struct, axis=0)          # [n_ds]
    c_mag = combine(mag, axis=0)                 # [n_ds]
    c_R = combine(R, axis=0)                     # [n_ds, n_attacks]

    f = float(np.mean(harmonic(c_struct, c_mag)))
    per_attack = np.mean(c_R, axis=0)            # [n_attacks]
    r = float(np.min(per_attack) if attack_combine == 'min' else np.mean(per_attack))
    return {'F': f, 'R': r, 'Combined': harmonic(f, r)}


def winner_diversity(members, datasets):
    """For each dataset, identifies which member maximizes structure-fidelity and which maximizes
    magnitude-fidelity. Returns (per-dataset winner table, per-member win counts, n_distinct_winners)."""
    rows = []
    win_counts = {m['Metric_Raw']: 0 for m in members}
    for ds in datasets:
        struct_vals = {m['Metric_Raw']: m['F_Struct_Scores'].get(ds, 0.0) for m in members}
        mag_vals = {m['Metric_Raw']: m['F_Mag_Scores'].get(ds, 0.0) for m in members}
        s_winner = max(struct_vals, key=struct_vals.get)
        m_winner = max(mag_vals, key=mag_vals.get)
        win_counts[s_winner] += 1
        win_counts[m_winner] += 1
        rows.append({'Dataset': ds,
                     'Struct_Winner': s_winner, 'Struct_Best': struct_vals[s_winner],
                     'Mag_Winner': m_winner, 'Mag_Best': mag_vals[m_winner],
                     'Winners_Differ': s_winner != m_winner})
    n_distinct = sum(1 for c in win_counts.values() if c > 0)
    return pd.DataFrame(rows), win_counts, n_distinct


def run(r_real, f_real, r_synth, f_synth, output_dir, set_size=6):
    os.makedirs(output_dir, exist_ok=True)

    all_records = load_metric_records(r_real, f_real, r_synth, f_synth)
    swarm_records = all_records
    datasets = all_datasets_in(swarm_records)
    attack_cols = attack_cols_of(swarm_records)
    print(f"Loaded {len(swarm_records)} SWARM configs, {len(datasets)} datasets, "
          f"{len(attack_cols)} attacks.")

    # --- Reference M-Set (paper's optimal SWARM set at this size) --------------
    pool = build_pool(swarm_records, datasets)
    reference = optimal_mset(pool, datasets, set_size)
    if reference is None:
        print("Could not select a reference M-Set.")
        return
    by_raw = {r['Metric_Raw']: r for r in swarm_records}
    members = [by_raw[m] for m in reference['members']]

    print(f"\nReference M-Set (size {set_size}): "
          f"F={reference['F']:.3f}, R={reference['R']:.3f}, Combined={reference['Combined']:.3f}")

    # --- Global best standalone metric (reference line) ------------------------
    global_standalone = []
    for rec in swarm_records:
        s = standalone_scores(rec, datasets, attack_cols)
        global_standalone.append({'Metric_Raw': rec['Metric_Raw'], **s})
    gs_df = pd.DataFrame(global_standalone).sort_values('Combined', ascending=False).reset_index(drop=True)
    best_global = gs_df.iloc[0]
    print(f"Best standalone metric (of all {len(gs_df)}): {best_global['Metric_Raw']} "
          f"(F={best_global['F']:.3f}, R={best_global['R']:.3f}, Combined={best_global['Combined']:.3f})")

    # --- (a) Single-metric ceiling among the M-Set's own members ---------------
    member_standalone = []
    for m in members:
        s = standalone_scores(m, datasets, attack_cols)
        member_standalone.append({'Metric_Raw': m['Metric_Raw'],
                                  'Core_Metric': core_metric_of(m['Metric_Raw']),
                                  'Metric_Class': class_of(m['Metric_Raw']), **s})
    ms_df = pd.DataFrame(member_standalone)
    ms_df.to_csv(os.path.join(output_dir, 'member_standalone_scores.csv'), index=False)

    ceil_f = ms_df['F'].max()
    ceil_r = ms_df['R'].max()
    ceil_c = ms_df['Combined'].max()
    best_f_member = ms_df.loc[ms_df['F'].idxmax(), 'Metric_Raw']
    best_r_member = ms_df.loc[ms_df['R'].idxmax(), 'Metric_Raw']

    print("\n--- (a) Integration vs. best single member ---")
    print(f"  M-Set:                        F={reference['F']:.3f}, R={reference['R']:.3f}, "
          f"Combined={reference['Combined']:.3f}")
    print(f"  Best single member (ceiling): F={ceil_f:.3f}, R={ceil_r:.3f}, Combined={ceil_c:.3f}")
    print(f"  M-Set combined exceeds best single member by "
          f"{(reference['Combined'] - ceil_c):.3f} ({(reference['Combined']/ceil_c - 1)*100:.1f}%)")
    print(f"  Best-F member ({best_f_member}) != Best-R member ({best_r_member}): "
          f"{best_f_member != best_r_member}  <- complementarity (no single member is best at both)")

    # Per-dataset winner diversity
    win_df, win_counts, n_distinct = winner_diversity(members, datasets)
    win_df.to_csv(os.path.join(output_dir, 'per_dataset_fidelity_winners.csv'), index=False)
    n_split = int(win_df['Winners_Differ'].sum())
    print(f"  Distinct members that are sole best on >=1 dataset-axis: {n_distinct} of {set_size}")
    print(f"  Datasets where struct-winner != mag-winner: {n_split} of {len(datasets)}")
    print("  Per-member fidelity-axis win counts (across datasets):")
    for raw, c in sorted(win_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {c:>3}  {raw}")

    # --- (b) Aggregation-rule grid --------------------------------------------
    grid = []
    for mc in ['max', 'mean']:
        for ac in ['min', 'mean']:
            s = ensemble_scores(members, datasets, attack_cols, mc, ac)
            grid.append({
                'Member_Combine': mc, 'Attack_Combine': ac,
                'Is_Paper_Rule': (mc == 'max' and ac == 'min'),
                'F': s['F'], 'R': s['R'], 'Combined': s['Combined'],
                'Beats_Best_Standalone': s['Combined'] > best_global['Combined'],
                'Gain_vs_Best_Standalone_%': (s['Combined'] / best_global['Combined'] - 1) * 100,
            })
    grid_df = pd.DataFrame(grid)
    grid_df.to_csv(os.path.join(output_dir, 'aggregation_rule_grid.csv'), index=False)

    print("\n--- (b) Ensemble gain under different aggregation rules ---")
    print(f"  (best standalone combined = {best_global['Combined']:.3f})")
    with pd.option_context('display.width', 200):
        print(grid_df.to_string(index=False))
    print(f"\n  Ensemble beats best standalone under "
          f"{int(grid_df['Beats_Best_Standalone'].sum())} of {len(grid_df)} aggregation rules.")

    # Save a compact summary for write-up / appendix tables
    pd.DataFrame([{
        'MSet_F': reference['F'], 'MSet_R': reference['R'], 'MSet_Combined': reference['Combined'],
        'Ceiling_F': ceil_f, 'Ceiling_R': ceil_r, 'Ceiling_Combined': ceil_c,
        'Best_Standalone_Combined': best_global['Combined'],
        'Best_Standalone_Metric': best_global['Metric_Raw'],
        'Distinct_Fidelity_Winners': n_distinct, 'Set_Size': set_size,
        'Datasets_Split_Winner': n_split, 'N_Datasets': len(datasets),
        'BestF_ne_BestR_member': best_f_member != best_r_member,
    }]).to_csv(os.path.join(output_dir, 'integration_summary.csv'), index=False)

    print(f"\nAll outputs written to: {output_dir}")


def main():
    p = argparse.ArgumentParser(description="M-Set integration / coverage-aggregation sensitivity analysis.")
    p.add_argument('--r-real', default=None)
    p.add_argument('--f-real', default=None)
    p.add_argument('--r-synth', default=None)
    p.add_argument('--f-synth', default=None)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--set-size', type=int, default=6,
                   help="Ensemble size to analyze (default: 6, matching the paper's SWARM M-Set).")
    args = p.parse_args()

    def dflt(v, name):
        return v or os.path.expandvars(f"$SCRATCH/virtual-cell/virtual-cell-metrics/analysis/{name}")

    r_real = dflt(args.r_real, 'robustness_real')
    f_real = dflt(args.f_real, 'fidelity_real')
    r_synth = dflt(args.r_synth, 'robustness_synthetic')
    f_synth = dflt(args.f_synth, 'fidelity_synthetic')
    output_dir = args.output_dir or os.path.expandvars(
        "$SCRATCH/virtual-cell/virtual-cell-metrics/analysis/rebuttal/mset_integration")

    print(f"Robustness (real):  {r_real}")
    print(f"Fidelity  (real):   {f_real}")
    print(f"Robustness (synth): {r_synth}")
    print(f"Fidelity  (synth):  {f_synth}")
    print(f"Output dir:         {output_dir}")

    run(r_real, f_real, r_synth, f_synth, output_dir, set_size=args.set_size)


if __name__ == "__main__":
    main()
