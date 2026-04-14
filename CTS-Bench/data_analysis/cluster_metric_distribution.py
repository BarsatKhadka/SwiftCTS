import os
# --- CRITICAL FIX: Set Backend to 'Agg' FIRST ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
import seaborn as sns
# ------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.mixture import GaussianMixture  # <--- NEW ALGORITHM

# --- CONFIGURATION ---
DATASET_ROOT = "./dataset_root"
OUTPUT_ROOT = "clustering_outputs_GMM" # New output folder for clarity

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

FEATURES = [
    'setup_tns', 'hold_tns', 'setup_vio_count',  
    'wirelength', 'power_total', 'clock_buffers'     
]

def make_radar_chart(df, categories, title, save_path):
    from math import pi
    if df.empty: return

    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=7)
    plt.ylim(0, 1)
    
    colors = {"Golden (Clean)": "green", "Marginal": "orange", "Broken": "red"}
    
    for idx, row in df.iterrows():
        label = row['Label']
        values = row[categories].values.flatten().tolist()
        values += values[:1]
        
        color = colors.get(label, "blue")
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=label, color=color)
        ax.fill(angles, values, color=color, alpha=0.1)
        
    plt.title(title, size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig(save_path, dpi=300)
    plt.close() 

def process_single_design(design_name, df, output_root):
    print(f"\n🔹 Processing Design: {design_name}")
    
    design_dir = os.path.join(output_root, design_name)
    os.makedirs(design_dir, exist_ok=True)
    
    X_raw = df[FEATURES].dropna()
    
    if len(X_raw) < 3:
        print(f"   ⚠️ Skipping {design_name}: Not enough data")
        return None

    # --- 1. BETTER PRE-PROCESSING ---
    # We use PowerTransformer (Yeo-Johnson) instead of StandardScaler
    # This automatically fixes SKEW (makes the data look more bell-shaped)
    # This helps ENORMOUSLY with TNS data.
    scaler = PowerTransformer()
    X_scaled = scaler.fit_transform(X_raw)
    
    # --- 2. SWITCH TO GMM (Gaussian Mixture Model) ---
    # n_init=5 means it tries 5 times to find the best fit
    # covariance_type='full' allows clusters to be different shapes (stretched)
    gmm = GaussianMixture(n_components=3, n_init=10, random_state=42, covariance_type='full')
    clusters = gmm.fit_predict(X_scaled)
    
    # --- 3. Labeling ---
    df.loc[X_raw.index, 'Cluster'] = clusters
    valid_df = df.dropna(subset=['Cluster']).copy()
    
    profile = valid_df.groupby('Cluster')[FEATURES].mean()
    profile['abs_tns'] = profile['setup_tns'].abs()
    sorted_clusters = profile.sort_values(by='abs_tns', ascending=True)
    
    cluster_names = {
        sorted_clusters.index[0]: "Golden (Clean)",
        sorted_clusters.index[1]: "Marginal",
        sorted_clusters.index[2]: "Broken"
    }
    valid_df['Label'] = valid_df['Cluster'].map(cluster_names)
    
    # --- 4. Plotting ---
    # Box Plots
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"{design_name}: GMM Distribution", fontsize=16)
    for i, col in enumerate(FEATURES):
        plt.subplot(2, 3, i+1)
        sns.boxplot(data=valid_df, x="Label", y=col, 
                    order=["Golden (Clean)", "Marginal", "Broken"],
                    palette={"Golden (Clean)": "green", "Marginal": "orange", "Broken": "red"})
        plt.title(col)
        plt.xlabel("")
        plt.xticks(rotation=15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(design_dir, f"{design_name}_box_metrics.png"), dpi=300)
    plt.close()
    
    # Radar Chart
    radar_df = valid_df.copy()
    min_max = MinMaxScaler()
    radar_df['Setup Severity'] = -1 * radar_df['setup_tns']
    radar_df['Hold Severity']  = -1 * radar_df['hold_tns']
    
    plot_feats_map = {
        'Setup Severity': 'Setup Severity',
        'Hold Severity': 'Hold Severity',
        'setup_vio_count': 'Vio Count',
        'wirelength': 'Wirelength',
        'power_total': 'Power',
        'clock_buffers': 'Buffers'
    }
    cols_to_norm = list(plot_feats_map.keys())
    radar_df[cols_to_norm] = min_max.fit_transform(radar_df[cols_to_norm])
    radar_df = radar_df.rename(columns=plot_feats_map)
    final_cats = list(plot_feats_map.values())
    
    radar_centers = radar_df.groupby('Label')[final_cats].mean().reset_index()
    make_radar_chart(radar_centers, final_cats, f"{design_name} Profile", os.path.join(design_dir, f"{design_name}_radar.png"))
    
    # Print individual counts
    counts = valid_df['Label'].value_counts()
    print(f"   📊 {design_name} (GMM) Counts: {counts.to_dict()}")

    return valid_df[['Design', 'Label']]

def generate_total_counts_chart(all_labels_df, output_root):
    print(f"\n🔹 Generating Total Count Summary...")
    agg_dir = os.path.join(output_root, "00_AGGREGATED_COUNTS")
    os.makedirs(agg_dir, exist_ok=True)
    
    # Truth Table
    print("\n" + "="*40)
    print("      DATA COUNT VERIFICATION TABLE")
    print("="*40)
    count_table = pd.crosstab(all_labels_df['Design'], all_labels_df['Label'])
    cols = [c for c in ["Golden (Clean)", "Marginal", "Broken"] if c in count_table.columns]
    count_table = count_table[cols]
    count_table['TOTAL_RUNS'] = count_table.sum(axis=1)
    print(count_table)
    print("="*40 + "\n")
    
    count_table.to_csv(os.path.join(agg_dir, "verification_count_table.csv"))
    
    # 1. Total Bar Chart
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=all_labels_df, x="Label", 
                  order=["Golden (Clean)", "Marginal", "Broken"],
                  palette={"Golden (Clean)": "green", "Marginal": "orange", "Broken": "red"})
    
    plt.title(f"Total Count (GMM Adjusted)", fontsize=16)
    plt.ylabel("Count")
    plt.xlabel("")
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(agg_dir, "total_counts_summary.png"), dpi=300)
    plt.close()

    # 2. Stacked Bar Chart
    ct = pd.crosstab(all_labels_df['Label'], all_labels_df['Design'])
    ct = ct.reindex(["Golden (Clean)", "Marginal", "Broken"]) 
    
    ct.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='viridis')
    plt.title("Breakdown of Categories by Design", fontsize=16)
    plt.ylabel("Number of Runs")
    plt.xticks(rotation=0)
    plt.legend(title="Design", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(agg_dir, "stacked_counts_by_design.png"), dpi=300)
    plt.close()

def main():
    print("🔹 1. Loading Data...")
    df_list = []
    
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    for file_name in CSV_FILES:
        base_name = os.path.basename(file_name)
        design_name = base_name.split('_')[0] 
        if os.path.exists(file_name):
            try:
                temp_df = pd.read_csv(file_name)
                temp_df['Design'] = design_name
                df_list.append(temp_df)
            except Exception: pass

    if not df_list:
        print("❌ No data found.")
        return

    full_df = pd.concat(df_list, ignore_index=True)
    all_labeled_runs = []
    unique_designs = full_df['Design'].unique()
    
    for design in unique_designs:
        design_df = full_df[full_df['Design'] == design].copy()
        labeled_subset = process_single_design(design, design_df, OUTPUT_ROOT)
        if labeled_subset is not None:
            all_labeled_runs.append(labeled_subset)

    if all_labeled_runs:
        total_df = pd.concat(all_labeled_runs, ignore_index=True)
        generate_total_counts_chart(total_df, OUTPUT_ROOT)

    print(f"\n✅ All Done! Results are in '{OUTPUT_ROOT}/'")

if __name__ == "__main__":
    main()