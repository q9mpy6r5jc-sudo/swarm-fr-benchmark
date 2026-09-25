"""M-Set stability and composition sensitivity analysis.

Evaluates how stable the selected M-Sets are across datasets under two resamplings:
    leave-one-dataset-out  -- hold out each dataset in turn, reselect
    bootstrap              -- resample datasets with replacement (1,000 refits), reselect

Measures stability at three granularities:
    - exact configuration (specific SWARM parameterization)
    - core metric (e.g., PDS, Pearson, Cosine)
    - metric class (Spatial, Distributional, Diff. Expression, Retrieval/Rank)
Also measures retention of the vulnerability dichotomy pairing across the class divide.

Operates directly on pre-computed per-dataset fidelity and robustness files.

Usage:
    python -m src.rebuttals.mset_stability \
        --r-real   ".../analysis/robustness_real" \
        --f-real   ".../analysis/fidelity_real" \
        --r-synth  ".../analysis/robustness_synthetic" \
        --f-synth  ".../analysis/fidelity_synthetic" \
        --output-dir ".../analysis/rebuttal/mset_stability" \
        --set-size 6
"""

import os
import glob
import argparse
import itertools

import numpy as np
import pandas as pd

# Core-metric -> metric class mapping
CORE_TO_CLASS = {
    'MSE': 'Spatial', 'MAE': 'Spatial', 'RMSE': 'Spatial',
    'Pearson': 'Spatial', 'Cosine': 'Spatial', 'R2': 'Spatial',
    'Wasserstein': 'Distributional', 'E_Distance': 'Distributional',
    'Sym_KL_Divergence': 'Distributional', 'MMD': 'Distributional',
    'PDS': 'Retrieval/Rank', 'NIR': 'Retrieval/Rank', 'Centroid_Accuracy': 'Retrieval/Rank',
    'Rank_Cosine': 'Retrieval/Rank', 'Rank_Pearson': 'Retrieval/Rank',
    'DES_VCC': 'Diff. Expression', 'DES_Robust': 'Diff. Expression',
}


def core_metric_of(raw_name):
    """Return the core metric prefix of a config name.
    Matches longest-prefix-first so 'Rank_Cosine' precedes 'Cosine'."""
    for core in sorted(CORE_TO_CLASS.keys(), key=len, reverse=True):
        if raw_name.startswith(core):
            return core
    return 'Unknown'


def class_of(raw_name):
    """Metric class of a config name via its core metric."""
    return CORE_TO_CLASS.get(core_metric_of(raw_name), 'Unknown')


# Vulnerability-dichotomy groups (Sec 4.2):
# Retrieval/Rank on one side; Spatial + Distributional + Diff. Expression on the other.
def dichotomy_group_of(raw_name):
    """'Retrieval/Rank' -> 'rank'; everything else -> 'complement'."""
    return 'rank' if class_of(raw_name) == 'Retrieval/Rank' else 'complement'


def spans_dichotomy(members):
    """True if an M-Set contains >=1 Retrieval/Rank metric AND >=1 complementary metric."""
    groups = set(dichotomy_group_of(m) for m in members)
    return 'rank' in groups and 'complement' in groups


# Standard literature configurations
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

POOL_SIZE = 20


def _resolve(base):
    inc = os.path.join(base, 'included')
    return inc if os.path.isdir(inc) else base


def _read(path):
    return pd.read_csv(path, compression='infer')


def load_metric_records(r_real, f_real, r_synth, f_synth):
    """Load per-metric records with per-dataset F and R scores without cross-dataset aggregation."""
    r_real_i, f_real_i = _resolve(r_real), _resolve(f_real)
    r_synth_i, f_synth_i = _resolve(r_synth), _resolve(f_synth)

    filenames = sorted({os.path.basename(f).replace('.csv.zst', '.csv')
                        for f in glob.glob(os.path.join(r_real_i, '*.csv'))
                        + glob.glob(os.path.join(r_real_i, '*.csv.zst'))})

    def path_for(directory, filename):
        for cand in (os.path.join(directory, filename),
                     os.path.join(directory, filename + '.zst')):
            if os.path.exists(cand):
                return cand
        return None

    records = []
    for filename in filenames:
        r_real_p = path_for(r_real_i, filename)
        f_real_p = path_for(f_real_i, filename)
        if r_real_p is None or f_real_p is None:
            continue
        try:
            r_df = _read(r_real_p)
            f_df = _read(f_real_p)
            r_synth_p = path_for(r_synth_i, filename)
            f_synth_p = path_for(f_synth_i, filename)
            if r_synth_p:
                r_df = pd.concat([r_df, _read(r_synth_p)], ignore_index=True)
            if f_synth_p:
                f_df = pd.concat([f_df, _read(f_synth_p)], ignore_index=True)

            struct_cols = [c for c in f_df.columns if 'Struct_Rho' in c]
            mag_cols = [c for c in f_df.columns if 'Mag_Rho' in c]

            f_struct, f_mag = {}, {}
            for _, row in f_df.iterrows():
                ds = row['Dataset']
                f_struct[ds] = np.mean([row[c] for c in struct_cols]) if struct_cols else 0.0
                f_mag[ds] = np.mean([row[c] for c in mag_cols]) if mag_cols else 0.0

            r_scores = {}
            for ac in [c for c in r_df.columns if c.startswith('Rob_')]:
                r_scores[ac] = r_df.set_index('Dataset')[ac].to_dict()

            raw = filename.replace('.csv', '')
            records.append({
                'Metric_Raw': raw,
                'F_Struct_Scores': f_struct,
                'F_Mag_Scores': f_mag,
                'R_Scores': r_scores,
                'Is_Literature': raw in LITERATURE_METRICS,
            })
        except Exception as e:
            print(f"  [warn] {filename}: {e}")

    return records


def all_datasets_in(records):
    return sorted(set().union(*(r['F_Struct_Scores'].keys() for r in records)))


def build_pool(records, datasets, pool_size=POOL_SIZE):
    """Construct diverse candidate pool over a given dataset subset."""
    rows = []
    for r in records:
        struct_vals = np.array([r['F_Struct_Scores'].get(ds, 0.0) for ds in datasets])
        mag_vals = np.array([r['F_Mag_Scores'].get(ds, 0.0) for ds in datasets])
        dataset_f = (2 * struct_vals * mag_vals) / (struct_vals + mag_vals + 1e-9)
        mean_f = np.mean(dataset_f) if dataset_f.size else 0.0

        attack_means = [np.mean([ac_dict.get(ds, 0.0) for ds in datasets])
                        for ac_dict in r['R_Scores'].values()]
        mean_r = np.min(attack_means) if attack_means else 0.0

        rows.append({**r, 'Mean_F': mean_f, 'Mean_R': mean_r,
                     'Harmonic': (2 * mean_f * mean_r) / (mean_f + mean_r + 1e-9)})

    df = pd.DataFrame(rows)
    if df.empty:
        return []

    k = pool_size // 3
    top_f = df.sort_values('Mean_F', ascending=False).head(k)
    top_r = df.sort_values('Mean_R', ascending=False).head(k)
    top_h = df.sort_values('Harmonic', ascending=False).head(k)
    pool = pd.concat([top_f, top_r, top_h]).drop_duplicates(subset=['Metric_Raw'])
    if len(pool) < pool_size:
        rem = df[~df['Metric_Raw'].isin(pool['Metric_Raw'])].sort_values('Harmonic', ascending=False)
        pool = pd.concat([pool, rem.head(pool_size - len(pool))])
    return pool.to_dict('records')


def optimal_mset(pool, datasets, size):
    """Vectorized portfolio selection for an ensemble of size `size`.
    Portfolio F = mean over datasets of harmonic(max struct, max mag).
    Portfolio R = min over attacks of mean-over-datasets of max member robustness.
    Combined = harmonic mean of portfolio F and R.
    """
    n = len(pool)
    if n < size:
        return None

    attack_cols = sorted(set().union(*(p['R_Scores'].keys() for p in pool)))

    F_struct = np.zeros((n, len(datasets)))
    F_mag = np.zeros((n, len(datasets)))
    R = np.zeros((n, len(datasets), len(attack_cols)))
    for i, p in enumerate(pool):
        for j, ds in enumerate(datasets):
            F_struct[i, j] = p['F_Struct_Scores'].get(ds, 0.0)
            F_mag[i, j] = p['F_Mag_Scores'].get(ds, 0.0)
            for a, ac in enumerate(attack_cols):
                R[i, j, a] = p['R_Scores'].get(ac, {}).get(ds, 0.0)

    indices = np.array(list(itertools.combinations(range(n), size)))

    if size == 1:
        F_combined = (2 * F_struct * F_mag) / (F_struct + F_mag + 1e-9)
        F_port = np.mean(F_combined, axis=1)
        R_port = np.min(np.mean(R, axis=1), axis=1)
    else:
        F_struct_port = np.max(F_struct[indices], axis=1)
        F_mag_port = np.max(F_mag[indices], axis=1)
        F_combined = (2 * F_struct_port * F_mag_port) / (F_struct_port + F_mag_port + 1e-9)
        F_port = np.mean(F_combined, axis=1)
        R_port = np.min(np.mean(np.max(R[indices], axis=1), axis=1), axis=1)

    combined = (2 * F_port * R_port) / (F_port + R_port + 1e-9)
    best = int(np.argmax(combined))
    members = [pool[i]['Metric_Raw'] for i in indices[best]]
    return {'members': members, 'F': float(F_port[best]),
            'R': float(R_port[best]), 'Combined': float(combined[best])}


def _select(records, datasets, size):
    """Full selection pipeline over a dataset subset: pool -> optimal M-Set."""
    pool = build_pool(records, datasets)
    if not pool:
        return None
    return optimal_mset(pool, datasets, size)


def run(r_real, f_real, r_synth, f_synth, output_dir, set_size=6,
        n_bootstrap=1000, seed=0):
    os.makedirs(output_dir, exist_ok=True)

    all_records = load_metric_records(r_real, f_real, r_synth, f_synth)
    swarm_records = all_records
    datasets = all_datasets_in(swarm_records)
    print(f"Loaded {len(swarm_records)} SWARM metric configs over {len(datasets)} datasets.")
    print(f"Datasets: {datasets}")

    # --- Full-data reference M-Set ---
    reference = _select(swarm_records, datasets, set_size)
    if reference is None:
        print("Could not select a reference M-Set (too few metrics/datasets).")
        return
    ref_set = set(reference['members'])
    ref_cores = set(core_metric_of(m) for m in reference['members'])
    ref_classes = set(class_of(m) for m in reference['members'])
    print(f"\nFull-data reference M-Set (size {set_size}, "
          f"F={reference['F']:.3f}, R={reference['R']:.3f}, "
          f"Combined={reference['Combined']:.3f}):")
    for m in reference['members']:
        print(f"  - {m}  [{core_metric_of(m)} | {class_of(m)}]")
    print(f"  Core metrics: {sorted(ref_cores)}")
    print(f"  Metric classes: {sorted(ref_classes)}")

    def jaccard(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if (a | b) else 1.0

    LEVELS = [
        ('config', lambda m: m, ref_set, 'Metric_Raw'),
        ('core',   core_metric_of, ref_cores, 'Core_Metric'),
        ('class',  class_of,       ref_classes, 'Metric_Class'),
    ]

    def summarize(samples, mode_name):
        valid = [(label, ms) for label, ms in samples if ms is not None]
        n = len(valid)

        rows = []
        n_rank_present = 0
        n_complement_present = 0
        n_spans = 0
        for label, ms in valid:
            members = ms['members']
            groups = set(dichotomy_group_of(m) for m in members)
            has_rank = 'rank' in groups
            has_complement = 'complement' in groups
            n_rank_present += int(has_rank)
            n_complement_present += int(has_complement)
            n_spans += int(has_rank and has_complement)
            rows.append({
                'Sample': label, 'F': ms['F'], 'R': ms['R'], 'Combined': ms['Combined'],
                'Jaccard_config': jaccard(members, reference['members']),
                'Jaccard_core': jaccard([core_metric_of(m) for m in members], ref_cores),
                'Jaccard_class': jaccard([class_of(m) for m in members], ref_classes),
                'Has_Rank': has_rank, 'Has_Complement': has_complement,
                'Spans_Dichotomy': has_rank and has_complement,
                'Members': " + ".join(members),
            })
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, f'mset_{mode_name}_samples.csv'), index=False)

        print(f"\n=== {mode_name} ({n} resamples) ===")
        if n:
            print(f"  [dichotomy] Retrieval/Rank present: {n_rank_present}/{n} "
                  f"({n_rank_present/n:.3f}) | complementary group present: "
                  f"{n_complement_present}/{n} ({n_complement_present/n:.3f}) | "
                  f"spans both groups: {n_spans}/{n} ({n_spans/n:.3f})")
        mean_jacs = {}
        results = {}
        for level_name, key_fn, ref_keys, col in LEVELS:
            recurrence = {}
            jaccards = []
            for _, ms in valid:
                keys = set(key_fn(m) for m in ms['members'])
                jaccards.append(len(keys & ref_keys) / len(keys | ref_keys) if (keys | ref_keys) else 1.0)
                for k in keys:
                    recurrence[k] = recurrence.get(k, 0) + 1

            rec_df = pd.DataFrame([
                {col: k, 'Recurrence_Count': c, 'Recurrence_Freq': c / n if n else 0.0,
                 'In_Reference': k in ref_keys}
                for k, c in recurrence.items()
            ]).sort_values(['Recurrence_Freq', 'In_Reference'], ascending=False).reset_index(drop=True)
            rec_df.to_csv(
                os.path.join(output_dir, f'mset_{mode_name}_recurrence_{level_name}.csv'), index=False)

            mean_jac = float(np.mean(jaccards)) if jaccards else float('nan')
            mean_jacs[level_name] = mean_jac
            results[level_name] = rec_df

            stable = rec_df[rec_df['Recurrence_Freq'] >= 0.9]
            print(f"\n  [{level_name}] mean Jaccard vs reference: {mean_jac:.3f} "
                  f"| {level_name}s recurring in >=90% of resamples: {len(stable)}")
            with pd.option_context('display.max_colwidth', 80, 'display.width', 200):
                head = 15 if level_name == 'config' else len(rec_df)
                print(rec_df.head(head).to_string(index=False))

        return mean_jacs, results

    # --- Leave-one-dataset-out ---
    lodo_samples = []
    for held_out in datasets:
        subset = [d for d in datasets if d != held_out]
        lodo_samples.append((f'drop_{held_out}', _select(swarm_records, subset, set_size)))
    summarize(lodo_samples, 'leave_one_dataset_out')

    # --- Bootstrap over datasets (with replacement) ---
    rng = np.random.default_rng(seed)
    n_ds = len(datasets)
    boot_samples = []
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_ds, size=n_ds)
        subset = [datasets[i] for i in idx]
        boot_samples.append((f'boot_{b}', _select(swarm_records, subset, set_size)))
    summarize(boot_samples, 'bootstrap')

    # Save the reference set for documentation
    pd.DataFrame([{'Rank': i + 1, 'Metric_Raw': m,
                   'Core_Metric': core_metric_of(m), 'Metric_Class': class_of(m)}
                  for i, m in enumerate(reference['members'])]).to_csv(
        os.path.join(output_dir, 'mset_reference_full_data.csv'), index=False)

    print(f"\nAll outputs written to: {output_dir}")


def main():
    p = argparse.ArgumentParser(description="M-Set selection stability analysis.")
    p.add_argument('--r-real', default=None)
    p.add_argument('--f-real', default=None)
    p.add_argument('--r-synth', default=None)
    p.add_argument('--f-synth', default=None)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--set-size', type=int, default=6,
                   help="Ensemble size to analyze (default: 6, matching the paper's SWARM M-Set).")
    p.add_argument('--n-bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    def dflt(v, name):
        return v or os.path.expandvars(f"$SCRATCH/virtual-cell/virtual-cell-metrics/analysis/{name}")

    r_real = dflt(args.r_real, 'robustness_real')
    f_real = dflt(args.f_real, 'fidelity_real')
    r_synth = dflt(args.r_synth, 'robustness_synthetic')
    f_synth = dflt(args.f_synth, 'fidelity_synthetic')
    output_dir = args.output_dir or os.path.expandvars(
        "$SCRATCH/virtual-cell/virtual-cell-metrics/analysis/rebuttal/mset_stability")

    print(f"Robustness (real):  {r_real}")
    print(f"Fidelity  (real):   {f_real}")
    print(f"Robustness (synth): {r_synth}")
    print(f"Fidelity  (synth):  {f_synth}")
    print(f"Output dir:         {output_dir}")

    run(r_real, f_real, r_synth, f_synth, output_dir,
        set_size=args.set_size, n_bootstrap=args.n_bootstrap, seed=args.seed)


if __name__ == "__main__":
    main()
