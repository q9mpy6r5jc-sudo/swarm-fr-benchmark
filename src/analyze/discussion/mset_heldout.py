"""Select M-Sets on training datasets, then score the omitted dataset.

Run from the repository root:
    python -m src.rebuttals.mset_heldout

Defaults use analysis/{fidelity,robustness}_{real,synthetic}. No SLURM,
raw expression data, or model inference is required. Fixed sizes test
transfer of selection conditional on size; they do not validate the size-selection rule.
Missing scores are never imputed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Support both module execution from root and direct script execution
try:
    from .mset_stability import LITERATURE_METRICS, build_pool, optimal_mset
except ImportError:
    from mset_stability import LITERATURE_METRICS, build_pool, optimal_mset


ATTACKS = tuple('Rob_' + name for name in (
    'Mode_Collapse_Control', 'Mode_Collapse_Mean_Perturbed',
    'Mode_Collapse_Median_Control', 'Mode_Collapse_Median_Perturbed',
    'Normalization_Mismatch', 'Scaling_Exploit',
))
SNR_SUFFIXES = ('Inf', 'SNR1', 'SNR01', 'SNR001')
METHODS = ('SWARM_6', 'SWARM_5', 'Literature_5', 'Standalone')
SCORES = ('F', 'R', 'Combined')


def harmonic(a, b):
    return 2 * a * b / (a + b + 1e-9)


def load_inputs(analysis_dir, empirical_only=False):
    """Strict schema/duplicate checks; retain missing values for fold-level audit.

    Prefer included/ as in Figure 3. Real/synthetic labels come from the input
    directories, not from name heuristics. File hashes identify the exact inputs.
    """
    records, dataset_types, manifest = {}, {}, []
    for domain in ('real',) if empirical_only else ('real', 'synthetic'):
        for kind in ('fidelity', 'robustness'):
            base = Path(analysis_dir) / f'{kind}_{domain}'
            directory = base / 'included' if (base / 'included').is_dir() else base
            paths = sorted([*directory.glob('*.csv'), *directory.glob('*.csv.zst')])
            if not paths:
                raise ValueError(f'No score CSVs in {directory}')
            seen = set()
            for path in paths:
                name = path.name.removesuffix('.zst').removesuffix('.csv')
                if name in seen:
                    raise ValueError(f'Duplicate compressed/uncompressed metric: {path}')
                seen.add(name)
                frame = pd.read_csv(path, compression='infer')
                cols = ([f'{axis}_Rho_{snr}' for axis in ('Struct', 'Mag')
                         for snr in SNR_SUFFIXES] if kind == 'fidelity' else list(ATTACKS))
                missing = set(['Dataset', *cols]) - set(frame.columns)
                if missing:
                    raise ValueError(f'{path}: missing columns {sorted(missing)}')
                if frame['Dataset'].isna().any() or frame['Dataset'].duplicated().any():
                    raise ValueError(f'{path}: missing or duplicate dataset IDs')
                values = frame[cols].apply(pd.to_numeric, errors='raise').to_numpy(float)
                # Missing/nonfinite values are audited below, not coerced to zero.
                finite = values[np.isfinite(values)]
                if np.any((finite < -1e-8) | (finite > 1 + 1e-8)):
                    raise ValueError(f'{path}: primary scores outside [0,1]')
                rec = records.setdefault(name, {
                    'Metric_Raw': name, 'Is_Literature': name in LITERATURE_METRICS,
                    'F_Struct_Scores': {}, 'F_Mag_Scores': {},
                    'R_Scores': {a: {} for a in ATTACKS},
                })
                for idx, ds in enumerate(frame['Dataset'].astype(str)):
                    domain_label = 'empirical' if domain == 'real' else 'synthetic'
                    if ds in dataset_types and dataset_types[ds] != domain_label:
                        raise ValueError(f'Dataset {ds} occurs in real and synthetic inputs')
                    dataset_types[ds] = domain_label
                    if kind == 'fidelity':
                        rec['F_Struct_Scores'][ds] = float(np.mean(values[idx, :4]))
                        rec['F_Mag_Scores'][ds] = float(np.mean(values[idx, 4:]))
                    else:
                        for j, attack in enumerate(ATTACKS):
                            rec['R_Scores'][attack][ds] = float(values[idx, j])
                manifest.append({'path': str(path).replace('\\', '/'), 'name': path.name,
                                 'rows': len(frame),
                                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if len(dataset_types) < 2:
        raise ValueError('At least two datasets are required')
    return [records[n] for n in sorted(records)], dataset_types, manifest


def missing_components(rec, dataset):
    values = {'F_str': rec['F_Struct_Scores'].get(dataset, np.nan),
              'F_mag': rec['F_Mag_Scores'].get(dataset, np.nan),
              **{a: rec['R_Scores'].get(a, {}).get(dataset, np.nan) for a in ATTACKS}}
    return [key for key, value in values.items() if not np.isfinite(value)]


def select_on_training(records, training, pool_size=20, sizes=(6, 5, 5)):
    """No held-out scores or availability enter eligibility, pools, or selection.

    All candidates must have every score on every training dataset. Reuses the
    manuscript's diverse-pool construction and harmonic optimization, separately
    for the expanded SWARM space (including literature) and literature subset.
    Standalone is optimized over ALL eligible configurations, without truncation.
    """
    eligible, excluded = [], []
    for rec in sorted(records, key=lambda r: r['Metric_Raw']):
        reasons = {ds: missing_components(rec, ds) for ds in training}
        reasons = {ds: fields for ds, fields in reasons.items() if fields}
        if reasons:
            excluded.append({'Metric_Raw': rec['Metric_Raw'], 'Missing': reasons})
        else:
            eligible.append(rec)
    pools = {'SWARM': build_pool(eligible, training, pool_size),
             'Literature': build_pool([r for r in eligible if r['Is_Literature']],
                                      training, pool_size)}
    specs = [('SWARM_6', pools['SWARM'], sizes[0]),
             ('SWARM_5', pools['SWARM'], sizes[1]),
             ('Literature_5', pools['Literature'], sizes[2]),
             ('Standalone', eligible, 1)]
    selections = {}
    for method, pool, size in specs:
        if len(pool) < size:
            raise ValueError(f'{method}: need {size} complete training candidates; got {len(pool)}')
        selections[method] = optimal_mset(pool, training, size)
    return selections, pools, excluded, len(eligible)


def score_heldout(members, records_by_name, dataset):
    """Complete selected-member data required; do not reselect after seeing test data."""
    recs = [records_by_name[m] for m in members]
    missing = {r['Metric_Raw']: missing_components(r, dataset) for r in recs}
    missing = {name: fields for name, fields in missing.items() if fields}
    if missing:
        return {**dict.fromkeys(SCORES, np.nan), 'Missing': missing}
    f_str = max(r['F_Struct_Scores'][dataset] for r in recs)
    f_mag = max(r['F_Mag_Scores'][dataset] for r in recs)
    attack_scores = {a: max(r['R_Scores'][a][dataset] for r in recs) for a in ATTACKS}
    f, r = harmonic(f_str, f_mag), min(attack_scores.values())
    return {'F': f, 'R': r, 'Combined': harmonic(f, r),
            'F_str_max': f_str, 'F_mag_max': f_mag, **attack_scores, 'Missing': {}}


def summarize(scores, tolerance=1e-9):
    """All summaries use the SAME complete folds for all four methods per domain."""
    rows, pairs = [], []
    for domain, group in scores.groupby('Domain', sort=True):
        valid = group[group['Common_Valid']]
        n_total, n_valid = group['Dataset'].nunique(), valid['Dataset'].nunique()
        for method in METHODS:
            sub = valid[valid['Method'] == method]
            rows.append({'Domain': domain, 'Method': method, 'N_Total': n_total,
                         'N_Common': n_valid,
                         **{f'Mean_{s}': sub[s].mean() for s in SCORES}})
        for left, right in [('SWARM_6', 'Literature_5'), ('SWARM_5', 'Literature_5'),
                            ('SWARM_6', 'Standalone'), ('SWARM_5', 'Standalone')]:
            for score in SCORES:
                pivot = valid.pivot(index='Dataset', columns='Method', values=score)
                differences = (pivot[left] - pivot[right] if n_valid else pd.Series(dtype=float))
                pairs.append({'Domain': domain, 'Method': left, 'Comparator': right,
                               'Score': score, 'N_Total': n_total, 'N_Common': n_valid,
                               'Mean_Difference': differences.mean(),
                               'Median_Difference': differences.median(),
                               'Wins': int((differences > tolerance).sum()),
                               'Ties': int((differences.abs() <= tolerance).sum()),
                               'Losses': int((differences < -tolerance).sum())})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def plot_scores(scores, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for domain, group in scores.groupby('Domain', sort=True):
        datasets = sorted(group['Dataset'].unique())
        fig, axes = plt.subplots(3, 1, figsize=(max(8, len(datasets) * .65), 9), sharex=True)
        x = np.arange(len(datasets))
        for axis, score in zip(axes, SCORES):
            for idx, method in enumerate(METHODS):
                values = group[group['Method'] == method].set_index('Dataset').reindex(datasets)
                # Match the population used in the tabular comparisons.
                y = values[score].where(values['Common_Valid'])
                axis.plot(x + (idx - 1.5) * .10, y, 'o', label=method, markersize=5)
            axis.set_ylabel('H(F, R)' if score == 'Combined' else score)
            axis.set_ylim(-.03, 1.03)
            axis.grid(axis='y', alpha=.25)
        axes[0].legend(ncol=4, fontsize=8)
        axes[0].set_title(f'Omitted-dataset evaluation: {domain} (common complete folds)')
        axes[-1].set_xticks(x, datasets, rotation=55, ha='right')
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(output_dir / f'heldout_{domain}.{extension}', dpi=180)
        plt.close(fig)


def run(analysis_dir, output_dir, empirical_only=False, make_plots=True, tolerance=1e-9):
    output_dir = Path(output_dir)
    # Avoid silently replacing a prior experiment's outputs.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f'Output directory is not empty; choose a new --output-dir: {output_dir}')
    records, dataset_types, manifest = load_inputs(analysis_dir, empirical_only)
    by_name = {r['Metric_Raw']: r for r in records}
    datasets = sorted(dataset_types)
    rows, selection_rows, pool_rows, exclusions = [], [], [], []
    print(f'Loaded {len(records)} configurations across {len(datasets)} datasets.', flush=True)
    for index, heldout in enumerate(datasets, 1):
        training = [d for d in datasets if d != heldout]
        choices, pools, excluded, n_eligible = select_on_training(records, training)
        for rec in excluded:
            exclusions.append({'Heldout': heldout, 'Metric_Raw': rec['Metric_Raw'],
                               'Missing_Training': json.dumps(rec['Missing'], sort_keys=True)})
        for label, pool in pools.items():
            for rank, rec in enumerate(pool, 1):
                pool_rows.append({'Heldout': heldout, 'Pool': label, 'Order': rank,
                                  'Metric_Raw': rec['Metric_Raw'],
                                  **{key: rec[key] for key in ('Mean_F', 'Mean_R', 'Harmonic')}})
        fold = []
        for method, choice in choices.items():
            result = score_heldout(choice['members'], by_name, heldout)
            selection_rows.append({'Heldout': heldout, 'Method': method,
                                   'Training_Datasets': json.dumps(training),
                                   'Members': json.dumps(choice['members']),
                                   'N_Eligible': n_eligible, 'Size': len(choice['members']),
                                   **{f'Train_{s}': choice[s] for s in SCORES}})
            fold.append({'Dataset': heldout, 'Domain': dataset_types[heldout], 'Method': method,
                         'Valid': not bool(result['Missing']),
                         'Missing_Test': json.dumps(result.pop('Missing'), sort_keys=True),
                         **result})
        common = all(row['Valid'] for row in fold)
        rows.extend([{**row, 'Common_Valid': common} for row in fold])
        print(f'[{index}/{len(datasets)}] {heldout}: {n_eligible} training candidates; '
              f'common held-out scores {"available" if common else "unavailable"}', flush=True)
    scores = pd.DataFrame(rows)
    summary, comparisons = summarize(scores, tolerance)
    # Paired differences for every fold (including explicit missing test values).
    differences = []
    for dataset, group in scores.groupby('Dataset', sort=True):
        values = group.set_index('Method')
        for left, right in [('SWARM_6', 'Literature_5'), ('SWARM_5', 'Literature_5'),
                            ('SWARM_6', 'Standalone'), ('SWARM_5', 'Standalone')]:
            differences.append({'Dataset': dataset, 'Domain': dataset_types[dataset],
                                'Method': left, 'Comparator': right,
                                'Common_Valid': bool(group['Common_Valid'].all()),
                                **{f'Delta_{s}': values.loc[left, s] - values.loc[right, s]
                                   for s in SCORES}})
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in [('heldout_scores.csv', scores), ('summary.csv', summary),
                            ('paired_comparisons.csv', comparisons),
                            ('paired_differences.csv', pd.DataFrame(differences)),
                            ('selections.csv', pd.DataFrame(selection_rows)),
                            ('candidate_pools.csv', pd.DataFrame(pool_rows)),
                            ('training_exclusions.csv', pd.DataFrame(exclusions, columns=[
                                'Heldout', 'Metric_Raw', 'Missing_Training']))]:
        frame.to_csv(output_dir / filename, index=False)
    metadata = {
        'schema_version': 1, 'analysis_dir': str(analysis_dir).replace('\\', '/'),
        'empirical_only': empirical_only, 'datasets': dataset_types,
        'n_configs': len(records), 'attacks': ATTACKS, 'pool_size': 20,
        'set_sizes': {'SWARM_6': 6, 'SWARM_5': 5, 'Literature_5': 5, 'Standalone': 1},
        'missing_policy': 'Training complete cases only; no held-out filtering of candidates. '
                          'Any missing selected-member test score invalidates that method; '
                          'all summaries share folds valid for all four methods.',
        'selection': 'Diverse pools rebuilt on training datasets only. '
                     'F=mean_d H(max_m F_str,max_m F_mag); '
                     'R=min_a mean_d max_m R; maximize H(F,R). '
                     'Standalone optimized over all eligible training configurations.',
        'test_scoring': 'F=H(max_m F_str,max_m F_mag); R=min_a max_m R; Combined=H(F,R).',
        'summary': 'Mean of held-out F, R, and H(F,R), separately by domain; '
                   'Mean_Combined is not H(Mean_F,Mean_R).',
        'ties': 'Sorted metric names, existing pool ordering, first maximizing combination; '
                'paired win/tie/loss uses absolute tolerance.',
        'tie_tolerance': tolerance, 'input_files': manifest,
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__},
        'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (Path(__file__), Path(__file__).with_name('mset_stability.py'))},
        'limitations': ['Fixed sizes were motivated by full-data exploration; this is not '
                        'validation of size selection or an untouched external benchmark.',
                        'Dataset folds may share studies/cellular contexts.',
                        'Strict missing-data eligibility differs from earlier zero-filled selection.'],
    }
    (output_dir / 'protocol.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if make_plots:
        plot_scores(scores, output_dir)
    print(summary.to_string(index=False), flush=True)
    print(f'Outputs written to: {output_dir}', flush=True)
    return scores, summary, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', type=Path, default=Path('analysis'))
    parser.add_argument('--output-dir', type=Path, default=Path('analysis/rebuttal/mset_heldout'))
    parser.add_argument('--empirical-only', action='store_true',
                        help='Use only empirical datasets for training and holdout folds.')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--tie-tolerance', type=float, default=1e-9)
    args = parser.parse_args()
    if not np.isfinite(args.tie_tolerance) or args.tie_tolerance < 0:
        parser.error('--tie-tolerance must be finite and nonnegative')
    run(args.analysis_dir, args.output_dir, args.empirical_only,
        not args.no_plots, args.tie_tolerance)


if __name__ == '__main__':
    main()
