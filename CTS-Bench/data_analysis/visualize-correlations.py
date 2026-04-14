import os
# --- CRITICAL FIX: Set Backend to 'Agg' FIRST ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
import seaborn as sns
# ------------------------------------------------

import pandas as pd
import numpy as np

# --- CONFIGURATION ---
DATASET_ROOT = "./dataset_root"
OUTPUT_ROOT = "correlation_outputs"

CSV_FILES = [
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
    os.path.join(DATASET_ROOT, "ethmac_batch4.csv")
]

# INPUTS (The Knobs you change)
INPUT_KNOBS = [
    'aspect_ratio', 
    'core_util', 
    'density', 
    'cts_max_wire', 
    'cts_buf_dist', 
    'cts_cluster_size', 
    'cts_cluster_dia'
]

# OUTPUTS (The Metrics you measure)
OUTPUT_METRICS = [
    # Timing - Setup
    'skew_setup', 
    'setup_slack', 
    'setup_tns',
    'setup_vio_count',
    
    # Timing - Hold
    'skew_hold', 
    'hold_slack', 
    'hold_tns',
    'hold_vio_count',
    
    # Physical / Power
    'clock_buffers', 
    'clock_inverters',
    'wirelength', 
    'power_total', 
    'utilization'
]

def generate_correlation_heatmap(df, output_folder, run_name):
    """Generates a correlation heatmap (Inputs vs Outputs) for a specific subset."""
    if df.empty: return

    os.makedirs(output_folder, exist_ok=True)
    
    # Filter valid columns
    valid_inputs = [c for c in INPUT_KNOBS if c in df.columns]
    valid_outputs = [c for c in OUTPUT_METRICS if c in df.columns]
    
    if not valid_inputs or not valid_outputs:
        print(f"   ⚠️ Skipping {run_name}: Missing columns.")
        return

    # Select only the relevant columns
    subset = df[valid_inputs + valid_outputs]
    
    # Check for zero variance (if a column is constant, correlation is NaN)
    # We fill NaNs with 0 to make the plot renderable, though they technically mean "undefined".
    corr_matrix = subset.corr(method='pearson').fillna(0)
    
    # Slice the matrix: Rows = Inputs, Cols = Outputs
    heatmap_data = corr_matrix.loc[valid_inputs, valid_outputs]

    # Plot
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        center=0,
        vmin=-1, vmax=1,  # Fix scale from -1 to 1 for consistency
        linewidths=0.5, 
        linecolor='gray',
        cbar_kws={'label': 'Correlation Coefficient (Pearson)'}
    )
    
    plt.title(f"{run_name}: Input Knobs vs. Metrics Correlation")
    plt.xlabel("Output Metrics (Performance)")
    plt.ylabel("Input Knobs (Constraints)")
    plt.xticks(rotation=45, ha='right') 
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(output_folder, f"{run_name}_correlation_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   ✅ Heatmap saved to {save_path}")

def main():
    print("🔹 1. Loading Data...")
    df_list = []
    
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    for f in CSV_FILES:
        base_name = os.path.basename(f)
        design_name = base_name.split('_')[0]
        
        if os.path.exists(f):
            try:
                temp_df = pd.read_csv(f)
                temp_df['Design'] = design_name
                df_list.append(temp_df)
            except: pass
    
    if not df_list: 
        print("❌ No data found.")
        return

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"   Loaded {len(full_df)} total data points.")

    # --- 2. Process ALL DESIGNS (Global) ---
    print("\n🔹 2. Generating Global Heatmap...")
    global_dir = os.path.join(OUTPUT_ROOT, "00_ALL_DESIGNS")
    generate_correlation_heatmap(full_df, global_dir, "All_Designs")

    # --- 3. Process Per Design ---
    print("\n🔹 3. Generating Per-Design Heatmaps...")
    unique_designs = full_df['Design'].unique()
    
    for design in unique_designs:
        design_subset = full_df[full_df['Design'] == design]
        design_dir = os.path.join(OUTPUT_ROOT, design)
        
        print(f"   Processing: {design} -> {design_dir}")
        generate_correlation_heatmap(design_subset, design_dir, design)

    print(f"\n✅ All Done! Check '{OUTPUT_ROOT}' directory.")

if __name__ == "__main__":
    main()