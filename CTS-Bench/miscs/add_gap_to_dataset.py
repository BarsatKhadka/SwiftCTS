import pandas as pd

# Load the main dataset containing all 4,360 data points
df = pd.read_csv('dataset_root/test_zipdiv.csv')

# 1. Establish design-specific baselines (The 'Theoretical Best' per architecture)
design_baselines = df.groupby('design_name').agg({
    'skew_setup': 'min',
    'power_total': 'min',
    'wirelength': 'min'
}).rename(columns={
    'skew_setup': 'min_skew',
    'power_total': 'min_power',
    'wirelength': 'min_wl'
})

# 2. Merge baselines to calculate relative performance
df = df.merge(design_baselines, on='design_name')

# 3. Calculate Gap Ratios (How far from the best?)
df['gap_skew'] = df['skew_setup'] / df['min_skew']
df['gap_power'] = df['power_total'] / df['min_power']
df['gap_wl'] = df['wirelength'] / df['min_wl']

# 4. Bottleneck Analysis
# Find the largest deviation to identify which constraint is 'failing' most
df['max_gap'] = df[['gap_skew', 'gap_power', 'gap_wl']].max(axis=1)

# Normalize gaps by the row-maximum to highlight the bottleneck (Scaling to 0.0-1.0)
df['norm_gap_skew'] = df['gap_skew'] / df['max_gap']
df['norm_gap_power'] = df['gap_power'] / df['max_gap']
df['norm_gap_wl'] = df['gap_wl'] / df['max_gap']

# 5. Efficiency Score: Pareto distance from the ideal [1, 1, 1]
# Represents the Euclidean distance to the 'perfect' design for that architecture
df['pareto_dist'] = ((df['gap_skew']-1)**2 + (df['gap_power']-1)**2 + (df['gap_wl']-1)**2)**0.5

# Save the unified dataset
# This now contains: Original features + CTS metrics + Pareto Benchmarks
df.to_csv('dataset_root/clocknet_unified_manifest_test.csv', index=False)

print(f"Unified dataset created with {len(df.columns)} columns.")