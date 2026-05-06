# SWARM-FR: Benchmarking Virtual Cell Metrics

This repository contains the codebase and scripts to reproduce the experiments, evaluations, and datasets presented in the "SWARM-FR: Benchmarking Virtual Cell Metrics" paper for the NeurIPS 2026 Evaluations & Datasets Track.

## Repository Structure

- `datasets/`: Contains the real and synthetic datasets used for evaluations (access via the Harvard Dataverse URL).
- `results/`: Evaluation results and extracted metrics  (access via the Harvard Dataverse URL).
- `src/`: Python source code, including:
  - `metrics/`: Core metrics computations (fidelity, robustness).
  - `preprocessing/`: Code to generate dataset variants (downsampling, merging chunks).
  - `analyze/`: Code for aggregating results and generating paper figures and tables (e.g., OLS regressions, metric aggregation).
  - `auxillary/`: Scripts for downloading empirical datasets and generating synthetic datasets.
- `batch_scripts/`: Batch submission scripts to distribute computation runs locally or on compute clusters.

## Reproducing the Results

To recalculate the evaluations from scratch, execute the provided bash scripts (update paths and cluster environments as needed).

1. Generate dataset variants:
   ```bash
   sbatch submit_variants.sh
   # or
   sbatch submit_deg.sh
   ```

2. Compute evaluation metrics across variants:
   ```bash
   sbatch submit_real_metrics_master.sh
   sbatch submit_synthetic_metrics_master.sh
   ```

3. Aggregate metrics and produce final plots/tables:
   ```bash
   sbatch submit_aggregate_metrics.sh
   ```

## License

This software is distributed under the MIT License. See `LICENSE` for details.
