import torch
import torch.nn.functional as F

def get_density_map(graph, grid_size=10):
    """
    Converts a graph's node positions into a 2D density heatmap.
    """
    # 1. Robust Coordinate Extraction
    # Check if 'pos' exists AND is not None
    if hasattr(graph, 'pos') and graph.pos is not None:
        pos = graph.pos
    else:
        # Fallback: Extract (x, y) from the first 2 columns of node features
        # Make sure we detach gradients just in case
        pos = graph.x[:, :2].detach()

    # 2. Normalize positions to 0-1 range
    min_xy = pos.min(dim=0)[0]
    max_xy = pos.max(dim=0)[0]
    
    # Safety: Avoid divide by zero if the chip is flat or single point
    width = max_xy - min_xy
    width[width == 0] = 1.0 
    
    norm_pos = (pos - min_xy) / width

    # 3. Create Histogram (Grid)
    # Clamp values to ensure they don't go out of bounds (0 to 9)
    grid_indices = (norm_pos * (grid_size - 0.001)).long()
    
    # Create the empty grid
    density_map = torch.zeros((grid_size, grid_size))
    
    # Fill the grid
    for x, y in grid_indices:
        density_map[x, y] += 1
        
    return density_map

def compare_mismatched_graphs(path_a, path_b):
    print(f"--- robust Comparison ---")
    
    # 1. Load
    g1 = torch.load(path_a, weights_only=False)
    g2 = torch.load(path_b, weights_only=False)

    print(f"Graph A Nodes: {g1.num_nodes}")
    print(f"Graph B Nodes: {g2.num_nodes}")
    
    if g1.num_nodes != g2.num_nodes:
        print(">> Netlists are DIFFERENT (Buffer insertion detected).")
        print(">> Switching to Density Map Comparison...")
    else:
        print(">> Netlists might be same size, but checking Density anyway.")

    # 2. Get Heatmaps (10x10 grid)
    map1 = get_density_map(g1, grid_size=10)
    map2 = get_density_map(g2, grid_size=10)

    # 3. Compare Heatmaps
    # We normalize them so total count doesn't bias (e.g. if G2 has more buffers)
    map1_norm = map1 / map1.sum()
    map2_norm = map2 / map2.sum()
    
    # Calculate difference (L1 Error: Sum of absolute differences)
    # 0.0 = Identical Distribution
    # 2.0 = Completely Disjoint (All logic in left for A, all logic in right for B)
    diff_score = (map1_norm - map2_norm).abs().sum().item()
    
    print(f"\nDistribution Diff Score: {diff_score:.4f}")
    
    # Thresholds for 'Differentness'
    if diff_score < 0.1:
        print(" Too Similar: The logic 'clouds' are in the same spots.")
    elif diff_score > 0.4:
        print(" Very Different: The logic has moved significantly.")
    else:
        print("Distinct: Good variation.")

# Usage
compare_mismatched_graphs("./aes_run_20260118_193324.pt", "./aes_run_20260118_161602.pt")