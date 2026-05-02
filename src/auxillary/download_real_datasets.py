import os
import subprocess
import sys

S3_BUCKET = "s3://public-access-bucket-i9sdj34j"
S3_PREFIX = "cellsimbench-data"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)

print("Pulling datasets from S3...")
print(f"Bucket: {S3_BUCKET}/{S3_PREFIX}/\n")

DATASETS = [
    "adamson16",
    "frangieh21",
    "kaden25fibroblast",
    "kaden25rpe1",
    "nadig25hepg2",
    "nadig25jurkat",
    "norman19",
    "replogle22k562",
    "replogle22k562gwps",
    "replogle22rpe1",
    "sunshine23",
    "tian21crispra",
    "tian21crispri",
    "wessels23"
]

print("=== Downloading processed dataset files ===")

for dataset in DATASETS:
    local_file = f"data/{dataset}/{dataset}_processed_complete.h5ad"
    s3_uri = f"{S3_BUCKET}/{S3_PREFIX}/{local_file}"
    
    print(f"Downloading: {local_file}")
    command = [
        "aws", "s3", "cp", "--no-sign-request",
        s3_uri,
        local_file
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError downloading {dataset}. Ensure AWS CLI is installed.")
        sys.exit(1)

print("\nAll datasets downloaded successfully!")
