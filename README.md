# SWARM-FR: Benchmarking Virtual Cell Metrics

This repository contains the evaluation framework, analysis scripts, and batch pipelines for **SWARM-FR: Benchmarking Virtual Cell Metrics**.

## Scope & Execution Overview

Full dataset generation and metric computation use the supplied SLURM batch scripts. The scoring-sensitivity, M-Set selection-stability, and coverage-aggregation analyses operate on saved metric outputs and can be run as Python modules. The commands below specify the inputs and settings used for the reported results. The selected SWARM M-Set uses six metrics and harmonic combination of structural and magnitude fidelity.

---

## Repository Structure

- `batch_scripts/`: Batch submission scripts to distribute computation runs across compute clusters.
- `datasets/`: Real and synthetic perturbation datasets (accessible via Harvard Dataverse).
- `results/`: Extracted metrics and summary tables.
- `src/`: Python source code, including:
  - `metrics/`: Core metrics computations (fidelity, robustness).
  - `preprocessing/`: Code to generate dataset variants and inject technical noise.
  - `analyze/`: Aggregation, regression analysis, and figure generation (`figure3_part_1.py`, `figure3_part_2.py`).
  - `analyze/discussion/`: M-Set stability, coverage aggregation, scoring-rule sensitivity, and held-out transfer.
  - `auxillary/`: Scripts for downloading empirical datasets and generating synthetic datasets.
- `environment.yml`: Conda environment record exported from experimental runs.

---

## Data Availability & Setup

The raw single-cell matrices, generated perturbation variants, and pre-computed metric outputs are hosted on **Harvard Dataverse**:  
[https://dataverse.harvard.edu/previewurl.xhtml?token=035dc52a-7a56-4f2f-bd7d-9c70cb7c8f8e](https://dataverse.harvard.edu/previewurl.xhtml?token=035dc52a-7a56-4f2f-bd7d-9c70cb7c8f8e)

To reproduce the manuscript tables and figures without running the multi-day cluster grid, download the pre-computed `analysis/` archives and place them in the repository root:
```text
analysis/
├── fidelity_real/
├── robustness_real/
├── fidelity_synthetic/
├── robustness_synthetic/
└── overall_real/
```

---

## Reproducing Manuscript Analyses (No SLURM Required)

These analysis commands operate on saved metric outputs and can be executed locally in minutes. Run all commands from the repository root:

### 1. M-Set Selection Stability
Evaluates 16 leave-one-dataset-out folds and 1,000 bootstrap refits for the six-metric SWARM M-Set ($t=6$):
```bash
python -m src.analyze.discussion.mset_stability \
    --r-real analysis/robustness_real \
    --f-real analysis/fidelity_real \
    --r-synth analysis/robustness_synthetic \
    --f-synth analysis/fidelity_synthetic \
    --set-size 6 \
    --n-bootstrap 1000 \
    --seed 0 \
    --output-dir analysis/rebuttal/mset_stability
```

### 2. Coverage Aggregation Sensitivity
Evaluates the fixed six-member SWARM M-Set ($t=6$) across the 2×2 member/exploit aggregation grid:
```bash
python -m src.analyze.discussion.mset_integration \
    --r-real analysis/robustness_real \
    --f-real analysis/fidelity_real \
    --r-synth analysis/robustness_synthetic \
    --f-synth analysis/fidelity_synthetic \
    --set-size 6 \
    --output-dir analysis/rebuttal/mset_integration
```

### 3. Scoring-Rule Sensitivity
Evaluates ranking robustness across 10 scoring formulations:
```bash
python -m src.analyze.discussion.scoring_rule_sensitivity \
    --overall-dir analysis/overall_real \
    --output-dir analysis/rebuttal/scoring_sensitivity_real
```

### 4. Omitted-Dataset Generalization (Held-out Cross-Validation)
Performs strict leave-one-dataset-out cross-validation:
```bash
python -m src.analyze.discussion.mset_heldout \
    --analysis-dir analysis \
    --output-dir analysis/rebuttal/mset_heldout
```

### 5. Leaderboard & M-Set Trajectories
Computes the primary harmonic leaderboard and selects optimal M-Sets across pool sizes $t \in [1, 15]$:
```bash
python -m src.analyze.figures.figure3_part_1
python -m src.analyze.figures.figure3_part_2
```

---

## Cluster-Scale Metric Evaluation (SLURM Required)

Generating perturbation variants and evaluating all 396 metric configurations across gigabyte-scale count matrices requires high-performance computing resources:

```bash
cd batch_scripts/

# 1. Synthesize knowledge-variant degradations
sbatch submit_variants.sh

# 2. Compute 396-metric evaluation grid across empirical datasets
sbatch submit_real_metrics_master.sh

# 3. Compute 396-metric evaluation grid across synthetic GeneSpider2 datasets
sbatch submit_synthetic_metrics_master.sh

# 4. Aggregate per-perturbation scores into overall metric summaries
sbatch submit_aggregate_metrics.sh
```

---

## Environment & Dependencies

See `environment.yml` for an export of the Conda environment used for reported runs. To create the environment:
```bash
conda env create -f environment.yml
conda activate swarm-fr
```

## License

This software is distributed under the MIT License. See `LICENSE` for details.
```
