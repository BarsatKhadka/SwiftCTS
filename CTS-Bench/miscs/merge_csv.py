import pandas as pd
import os
DATASET_ROOT = "dataset_root"
csv_files = [
    os.path.join(DATASET_ROOT, "picorv32_batch1.csv"),
    os.path.join(DATASET_ROOT, "picorv32_batch2.csv"),
    os.path.join(DATASET_ROOT, "aes_batch1.csv"),
    os.path.join(DATASET_ROOT, "aes_batch2.csv"),
    os.path.join(DATASET_ROOT, "aes_batch3.csv"),
    os.path.join(DATASET_ROOT, "sha256_batch1.csv"),
    os.path.join(DATASET_ROOT, "sha256_batch2.csv"),
    os.path.join(DATASET_ROOT, "sha256_batch3.csv"),
    os.path.join(DATASET_ROOT, "sha256_batch4.csv"),
    os.path.join(DATASET_ROOT, "ethmac_batch1.csv"),
    os.path.join(DATASET_ROOT, "ethmac_batch2.csv"),
    os.path.join(DATASET_ROOT, "ethmac_batch3.csv"),
    os.path.join(DATASET_ROOT, "ethmac_batch4.csv"),
]

dfs = []
global_run_id = 0

for f in csv_files:
    df = pd.read_csv(f)

    # remove old local run_id
    if "run_id" in df.columns:
        df = df.drop(columns=["run_id"])

    # assign new global run_id
    df.insert(0, "run_id", range(global_run_id, global_run_id + len(df)))
    global_run_id += len(df)

    dfs.append(df)

main_df = pd.concat(dfs, ignore_index=True)

main_df.to_csv(os.path.join(DATASET_ROOT, "main_dataset.csv"), index=False)
