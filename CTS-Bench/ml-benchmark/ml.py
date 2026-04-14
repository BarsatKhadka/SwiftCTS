import pandas as pd
import torch
import torch.nn as nn
import os
import time
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, global_mean_pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

DATA_DIR = "dataset_root" # <--- VERIFY PATH
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> Device: {DEVICE}")

# --- 1. DATASET ---
class FusionDataset(Dataset):
    def __init__(self, df, cache, scaler, is_train=True):
        self.df = df
        self.cache = cache
        
        # Scalar Knobs
        p_cols = ['aspect_ratio', 'core_util', 'density', 'synth_strategy', 'io_mode', 'time_driven', 'routability_driven']
        c_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
        
        # TARGETS: Individual physical metrics
        self.target_cols = ['gap_skew', 'gap_power', 'gap_wl'] 
        
        knobs = df[p_cols + c_cols].values
        if is_train:
            self.scaler = StandardScaler()
            self.norm_knobs = self.scaler.fit_transform(knobs)
        else:
            self.scaler = scaler
            self.norm_knobs = self.scaler.transform(knobs)

    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = self.cache[row['placement_id']].clone()
        
        all_k = torch.tensor(self.norm_knobs[idx], dtype=torch.float)
        data.place_knobs = all_k[:7].unsqueeze(0)
        data.cts_knobs   = all_k[7:].unsqueeze(0)
        
        # TARGET: Multi-variate y (Shape: [1, 3])
        targets = row[self.target_cols].values.astype(np.float32)
        data.y = torch.tensor(targets, dtype=torch.float).unsqueeze(0)
        
        return data

# --- 2. CACHE ---
def build_cache(df, col_name):
    print(f"    Caching unique graphs from {col_name}...")
    cache = {}
    unique = df.drop_duplicates(subset='placement_id')
    for i, row in unique.iterrows():
        path = row[col_name]
        if os.path.exists(path):
            cache[row['placement_id']] = torch.load(path, weights_only=False)
    print(f"    ✅ Loaded {len(cache)} unique graphs.")
    return cache

# --- 3. MODEL ---
class FusionModel(nn.Module):
    def __init__(self, backbone, in_channels, hidden_dim=64, dropout=0.5, place_dim=32, cts_dim=16):
        super().__init__()
        self.dropout_rate = dropout
        
        if backbone == 'GCN': self.gnn = GCNConv(in_channels, hidden_dim)
        elif backbone == 'SAGE': self.gnn = SAGEConv(in_channels, hidden_dim)
        elif backbone == 'GATv2': self.gnn = GATv2Conv(in_channels, hidden_dim, heads=1, edge_dim=1)
        
        self.place_mlp = nn.Sequential(nn.Linear(7, place_dim), nn.ReLU())
        self.cts_mlp = nn.Sequential(nn.Linear(4, cts_dim), nn.ReLU())
        
        fusion_in = hidden_dim + place_dim + cts_dim
        # Change output to 3
        self.head = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, 3) 
        )  

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        if isinstance(self.gnn, GATv2Conv):
            out = self.gnn(x, edge_index, edge_attr=edge_attr)
        else:
            out = self.gnn(x, edge_index)
            
        out = out.relu()
        out = nn.functional.dropout(out, p=self.dropout_rate, training=self.training)
        g = global_mean_pool(out, data.batch)
        
        p = self.place_mlp(data.place_knobs)
        p = nn.functional.dropout(p, p=self.dropout_rate, training=self.training)
        
        c = self.cts_mlp(data.cts_knobs)
        c = nn.functional.dropout(c, p=self.dropout_rate, training=self.training)
        
        return self.head(torch.cat([g, p, c], dim=1))

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def benchmark(col_name, feat_dim, label, hidden_dim=64, place_dim=32, cts_dim=16, lr=0.001):
    print(f"\n>>> STARTING WORKLOAD: {label}")
    full_train_df = pd.read_csv(os.path.join(DATA_DIR, "clocknet_unified_manifest.csv"))
    zero_shot_df = pd.read_csv(os.path.join(DATA_DIR, "clocknet_unified_manifest_test.csv"))
    train_df, val_df = train_test_split(full_train_df, test_size=0.2, random_state=42)

    for d in [train_df, val_df, zero_shot_df]:
        d['synth_strategy'] = d['synth_strategy'].astype('category').cat.codes

    cache = build_cache(full_train_df, col_name)
    cache.update(build_cache(zero_shot_df, col_name))
    
    train_ds = FusionDataset(train_df, cache, None, True)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(FusionDataset(val_df, cache, train_ds.scaler, False), batch_size=32)
    zs_loader = DataLoader(FusionDataset(zero_shot_df, cache, train_ds.scaler, False), batch_size=32)
    
    target_names = ['Skew', 'Power', 'Wire']
    workload_summary = []

    for net in ['GCN', 'SAGE', 'GATv2']:
        print(f"  Profiling {net}...")
        log_data = []
        if DEVICE.type == 'cuda': torch.cuda.reset_peak_memory_stats()
        
        model = FusionModel(net, feat_dim, hidden_dim, 0.5, place_dim, cts_dim).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        crit = nn.MSELoss()
        
        start_time = time.time()
        for epoch in range(1, 101):
            model.train()
            for batch in train_loader:
                batch = batch.to(DEVICE); opt.zero_grad()
                out = model(batch)
                loss = crit(out, batch.y); loss.backward(); opt.step()
            
            def get_component_metrics(loader):
                model.eval(); p, t = [], []
                with torch.no_grad(): 
                    for b in loader:
                        b = b.to(DEVICE)
                        p.append(model(b).cpu().numpy())
                        t.append(b.y.cpu().numpy().reshape(-1, 3))
                p, t = np.concatenate(p), np.concatenate(t)
                return r2_score(t, p, multioutput='raw_values'), mean_absolute_error(t, p, multioutput='raw_values')

            s_r2s, s_maes = get_component_metrics(val_loader)
            u_r2s, u_maes = get_component_metrics(zs_loader)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Ep {epoch:3d} | Seen R2: {np.mean(s_r2s):.3f} [Sk:{s_r2s[0]:.2f}/Po:{s_r2s[1]:.2f}/Wi:{s_r2s[2]:.2f}]")
                print(f"         | Unseen R2: {np.mean(u_r2s):.3f} [Sk:{u_r2s[0]:.2f}/Po:{u_r2s[1]:.2f}/Wi:{u_r2s[2]:.2f}]")
                print(f"         | Seen MAE (S/P/W): {s_maes[0]:.4f}, {s_maes[1]:.4f}, {s_maes[2]:.4f} | Unseen MAE: {u_maes[0]:.4f}, {u_maes[1]:.4f}, {u_maes[2]:.4f}")
                print("-" * 80)
            
            epoch_log = {"Epoch": epoch, "Time": time.time()-start_time, "U_Avg_R2": np.mean(u_r2s)}
            for i, name in enumerate(target_names):
                epoch_log[f"S_MAE_{name}"] = s_maes[i]
                epoch_log[f"U_R2_{name}"] = u_r2s[i]
                epoch_log[f"U_MAE_{name}"] = u_maes[i]
            log_data.append(epoch_log)

        # --- STATISTICAL SUMMARY ---
        log_df = pd.DataFrame(log_data)
        mem_mb = torch.cuda.max_memory_allocated() / 1024**2 if DEVICE.type == 'cuda' else 0
        
        workload_summary.append({
            "Graph": label, "Net": net, "VRAM_MB": f"{mem_mb:.1f}",
            "Max_U_R2_Wire": f"{log_df['U_R2_Wire'].max():.3f}",  # Best Trend
            "Min_U_MAE_Wire": f"{log_df['U_MAE_Wire'].min():.4f}", # Best Physical Accuracy
            "Min_U_MAE_Power": f"{log_df['U_MAE_Power'].min():.4f}",
            "Max_S_R2": f"{log_df['U_Avg_R2'].max():.3f}",
            "Time_Total": f"{time.time() - start_time:.1f}s"
        })
        log_df.to_csv(f"full_log_{label}_{net}.csv", index=False)
            
    return workload_summary

if __name__ == "__main__":
    all_results = []
    # Efficiency Run
    all_results.extend(benchmark('cluster_graph_path', 10, "Clustered", 16, 8, 4, 0.0005))
    # Accuracy Baseline
    all_results.extend(benchmark('raw_graph_path', 4, "Raw", 64, 32, 16, 0.001))
    
    final_df = pd.DataFrame(all_results)
    final_df.to_csv("mlbench_workload_summary.csv", index=False)
    print("\n" + "="*95 + "\nFINAL MLBENCH CHARACTERIZATION (MIN/MAX OVER 100 EPOCHS)\n" + "="*95)
    print(final_df.to_string(index=False))