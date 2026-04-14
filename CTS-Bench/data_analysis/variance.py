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
OUTPUT_ROOT = "stats_outputs"

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

# The full list of metrics to analyze
TARGET_METRICS = [
    'utilization', 
    'wirelength', 
    'power_total',
    
    # Setup Timing
    'setup_slack', 
    'setup_tns',       # Total Negative Slack
    'setup_vio_count', 
    'skew_setup',      
    
    # Hold Timing
    'hold_slack', 
    'hold_tns',
    'hold_vio_count', 
    'skew_hold',      
    
    # Resources
    'clock_buffers', 
    'clock_inverters', 
    'timing_repair_buffers'
]

def generate_stats_and_plots(df, output_folder, run_name):
    """Generates statistics table and histogram plots for a specific subset."""
    if df.empty: return

    os.makedirs(output_folder, exist_ok=True)
    
    # Filter valid metrics for this specific subset
    # (Some designs might have 0 variance in some columns, we still keep them to show that)
    valid_metrics = [c for c in TARGET_METRICS if c in df.columns]

    # --- 1. Statistics Report ---
    stats = df[valid_metrics].describe().T
    
    # Coefficient of Variation (CV) = StdDev / Mean
    # Handle division by zero if mean is 0
    stats['CV (%)'] = stats.apply(
        lambda row: (row['std'] / abs(row['mean']) * 100) if row['mean'] != 0 else 0.0, axis=1
    )
    
    report_df = stats[['min', 'max', 'mean', 'std', 'CV (%)']]
    
    report_path = os.path.join(output_folder, f"{run_name}_stats_report.txt")
    with open(report_path, "w") as f:
        f.write(f"--- STATISTICS REPORT: {run_name} ---\n")
        f.write(f"Total Samples: {len(df)}\n\n")
        f.write(report_df.round(2).to_markdown())
    print(f"   ✅ Stats Report saved to {report_path}")

    # --- 2. Distribution Plots ---
    num_plots = len(valid_metrics)
    rows = (num_plots // 4) + 1  
    
    plt.figure(figsize=(20, 4 * rows))
    plt.suptitle(f"{run_name}: Metric Distributions", fontsize=20)
    
    for i, col in enumerate(valid_metrics):
        plt.subplot(rows, 4, i + 1)
        
        # Color logic
        if "vio" in col or "tns" in col: color = "firebrick"
        elif "slack" in col: color = "darkorange"
        else: color = "steelblue"
        
        # Plot
        try:
            sns.histplot(df[col], kde=True, color=color, bins=30)
        except Exception: 
            # Fallback if singular matrix (all values same)
            plt.hist(df[col], color=color, bins=30)

        plt.title(f"{col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        
        # Add Range Text
        r_min, r_max = df[col].min(), df[col].max()
        plt.text(0.95, 0.95, f"Min: {r_min:.2f}\nMax: {r_max:.2f}", 
                 transform=plt.gca().transAxes, ha='right', va='top', 
                 bbox=dict(boxstyle="round", fc="white", alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    plot_path = os.path.join(output_folder, f"{run_name}_distributions.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"   ✅ Plots saved to {plot_path}")

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
    print("\n🔹 2. Generating Global Report...")
    global_dir = os.path.join(OUTPUT_ROOT, "00_ALL_DESIGNS")
    generate_stats_and_plots(full_df, global_dir, "All_Designs")

    # --- 3. Process Per Design ---
    print("\n🔹 3. Generating Per-Design Reports...")
    unique_designs = full_df['Design'].unique()
    
    for design in unique_designs:
        design_subset = full_df[full_df['Design'] == design]
        design_dir = os.path.join(OUTPUT_ROOT, design)
        
        print(f"   Processing: {design} -> {design_dir}")
        generate_stats_and_plots(design_subset, design_dir, design)

    print(f"\n✅ All Done! Check '{OUTPUT_ROOT}' directory.")

if __name__ == "__main__":
    main()