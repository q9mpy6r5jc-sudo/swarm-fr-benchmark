import os
import pandas as pd
from pathlib import Path

source_base = "../../datasets"
dest_base = "../../swarm_rf_datasets/sample_data"
datasets_to_sample = ["real/frangieh21", "synthetic/genespider2"]

print("Starting compression...")

for ds in datasets_to_sample:
    src_dir = os.path.join(source_base, ds)
    if not os.path.exists(src_dir):
        print(f"Skipping {ds} - not found.")
        continue
        
    for src_path in Path(src_dir).rglob('*.csv'):
        rel_path = src_path.relative_to(source_base)
        dest_path = Path(dest_base) / rel_path.parent
        dest_path.mkdir(parents=True, exist_ok=True)
        
        dest_file = dest_path / (src_path.stem + '.parquet')
        
        if dest_file.exists():
            continue
            
        print(f"Compressing: {rel_path} -> .parquet")
        try:
            df = pd.read_csv(src_path)
            df.to_parquet(dest_file, index=False)
        except Exception as e:
            print(f"Error compressing {src_path}: {e}")

print("Compression complete!")