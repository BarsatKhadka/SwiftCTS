import os
# --- CRITICAL FIX: Set Backend to 'Agg' FIRST ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
import seaborn as sns
# ------------------------------------------------

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

# --- CONFIGURATION ---
DATASET_ROOT = "./dataset_root"
OUTPUT_ROOT = "fidelity_outputs"

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

def get_graph_stats(path):
    """Loads graph and returns basic topology stats."""
    try:
        # weights_only=False allows loading complex PyG Data objects
        data = torch.load(path, weights_only=False)
        
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
        
        # Calculate Center of Mass (Mean of X, Y coords)
        # Assuming pos is in the first 2 columns of x (standard for many GNN datasets)
        if hasattr(data, 'x') and data.x is not None:
            pos_mean = data.x[:, :2].mean(dim=0).numpy()
            cx, cy = pos_mean[0], pos_mean[1]
        else:
            cx, cy = 0, 0
        
        return {
            "nodes": num_nodes,
            "edges": num_edges,
            "degree": avg_degree,
            "cx": cx,
            "cy": cy
        }
    except Exception as e:
        # print(f"Error loading {path}: {e}") # Optional: Uncomment for debug
        return None

def generate_fidelity_plots(results_df, output_folder, title_prefix):
    """Generates the comparison plots for a specific subset of data."""
    if results_df.empty:
        print(f"   ⚠️ No data for {title_prefix}")
        return

    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, f"{title_prefix}_fidelity_report.png")

    plt.figure(figsize=(14, 6))

    # Plot A: Compression Consistency
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=results_df, x="raw_nodes", y="cluster_nodes", alpha=0.7, hue="Design" if "Design" in results_df.columns else None)
    
    # Trend line
    if len(results_df) > 1:
        try:
            m, b = np.polyfit(results_df["raw_nodes"], results_df["cluster_nodes"], 1)
            plt.plot(results_df["raw_nodes"], m*results_df["raw_nodes"] + b, color="red", linestyle="--", 
                     label=f"Trend (Ratio ~ {results_df['compression_ratio'].mean():.1f}x)")
        except: pass

    plt.title(f"{title_prefix}: Compression Consistency\n(Raw vs Clustered Node Count)")
    plt.xlabel("Raw Nodes (Ground Truth)")
    plt.ylabel("Clustered Nodes (Proxy)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot B: Topology Preservation (Degree)
    plt.subplot(1, 2, 2)
    sns.kdeplot(results_df["cluster_degree"], fill=True, color="green", label="Clustered", warn_singular=False)
    sns.kdeplot(results_df["raw_degree"], fill=True, color="blue", label="Raw", warn_singular=False)
    
    plt.title(f"{title_prefix}: Topology Preservation\n(Node Degree Distribution)")
    plt.xlabel("Avg Degree (Connections per Node)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"   ✅ Saved Report: {save_path}")
    print(f"      Avg Compression Ratio: {results_df['compression_ratio'].mean():.2f}x")
    print(f"      Avg Center Shift: {results_df['center_shift'].mean():.4f}")

def main():
    print("🔹 1. Loading Logs...")
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
    
    if not df_list: return
    df = pd.concat(df_list, ignore_index=True)
    
    # Unique placements only
    unique_df = df.drop_duplicates(subset=['placement_id'])
    print(f"   Found {len(unique_df)} unique graph pairs to analyze.")

    print("🔹 2. Computing Graph Statistics (This may take a moment)...")
    stats = []
    
    # Limit for testing speed if needed, remove [:X] for full run
    for idx, row in unique_df.iterrows():
        raw_path = row['raw_graph_path']
        cluster_path = row['cluster_graph_path']
        
        # Ensure paths exist
        if not os.path.exists(raw_path) or not os.path.exists(cluster_path):
            continue
        
        r_stats = get_graph_stats(raw_path)
        c_stats = get_graph_stats(cluster_path)
        
        if r_stats and c_stats:
            stats.append({
                "Design": row['Design'],
                "placement_id": row['placement_id'],
                "raw_nodes": r_stats['nodes'],
                "cluster_nodes": c_stats['nodes'],
                "compression_ratio": r_stats['nodes'] / c_stats['nodes'] if c_stats['nodes'] > 0 else 0,
                "raw_degree": r_stats['degree'],
                "cluster_degree": c_stats['degree'],
                "center_shift": ((r_stats['cx'] - c_stats['cx'])**2 + (r_stats['cy'] - c_stats['cy'])**2)**0.5
            })
            
    if not stats:
        print("❌ No valid graph files found or loaded.")
        return

    results = pd.DataFrame(stats)
    
    print("\n🔹 3. Generating Fidelity Reports...")

    # A. Global Report (All Designs Combined)
    all_dir = os.path.join(OUTPUT_ROOT, "00_ALL_DESIGNS")
    print(f"\n   Processing: ALL DESIGNS -> {all_dir}")
    generate_fidelity_plots(results, all_dir, "All_Designs")
    results.to_csv(os.path.join(all_dir, "fidelity_metrics.csv"), index=False)

    # B. Per-Design Reports
    unique_designs = results['Design'].unique()
    for design in unique_designs:
        design_subset = results[results['Design'] == design]
        design_dir = os.path.join(OUTPUT_ROOT, design)
        
        print(f"   Processing: {design} -> {design_dir}")
        generate_fidelity_plots(design_subset, design_dir, design)

    print(f"\n✅ All Done! Check '{OUTPUT_ROOT}' directory.")

if __name__ == "__main__":
    main()