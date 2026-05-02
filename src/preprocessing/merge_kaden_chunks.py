import os
import glob
import pandas as pd

def merge_complete_chunks():
    target_dir = "../../results/real/kaden25rpe1"
    expected_chunks = 20

    print(f"Scanning {target_dir} for completed chunks...\n")
    chunk_zero_files = glob.glob(os.path.join(target_dir, '*_chunk_0.csv'))

    if not chunk_zero_files:
        print("No chunk_0.csv files found. Nothing to merge.")
        return

    for chunk_zero in chunk_zero_files:
        base_path = chunk_zero.replace('_chunk_0.csv', '')
        base_name = os.path.basename(base_path)
        chunk_files = glob.glob(f'{base_path}_chunk_*.csv')
        
        if len(chunk_files) == expected_chunks:
            print(f"Merging {expected_chunks} chunks for {base_name}...")
            try:
                chunk_files.sort(key=lambda x: int(x.split('_chunk_')[-1].split('.csv')[0]))
                final_path = f'{base_path}.csv'

                for i, f in enumerate(chunk_files):
                    print(f"  -> Appending chunk {i}/{expected_chunks-1}...", end='\r')
                    df = pd.read_csv(f)
                    df.to_csv(final_path, mode='a', header=(i == 0), index=False)
                
                print(f"\nSUCCESS: Saved {final_path} and cleaning up chunks...")
            
                for f in chunk_files:
                    os.remove(f)
                print("Cleanup complete.\n")
                
            except Exception as e:
                print(f"\nERROR merging {base_name}: {e}\n")
        else:
            print(f"SKIPPING: {base_name} only has {len(chunk_files)}/{expected_chunks} chunks.")

if __name__ == "__main__":
    merge_complete_chunks()