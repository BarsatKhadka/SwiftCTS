import os
import random
import sys

from sympy import group
from extract_placement_def_to_dict import process_design
from collections import defaultdict, deque
import numpy as np
import networkx as nx
import torch

# Change this to the specific run folder you are processing
if len(sys.argv) < 2:
    print(" Error: You must provide the Run Tag as an argument!")
    print("Usage: python 2-gen-saif.py <RUN_TAG>")
    sys.exit(1)

FILENAME = sys.argv[1]  # Takes the argument passed from Bash

design_name = FILENAME.split("_")[0]

if design_name in ["picorv32", "aes" , "sha256"] :
    design_data = process_design(FILENAME, clock_port="clk")
elif design_name == "ethmac":
    design_data  = process_design(FILENAME, clock_port="wb_clk_i")
elif design_name == "zipdiv":
    design_data  = process_design(FILENAME, clock_port="i_clk")


#aggregate flops and their one hop neighbors
def form_atomic_clusters(design_data):

    claimed_gates = {}
    atomic_clusters = []

    raw_edges = set()

    flops = [k for k, v in design_data.items() if v['type'] == 'flip_flop']

    # Shuffle strictly to remove systematic bias
    random.shuffle(flops)

    flop_name_to_id = {name: idx for idx, name in enumerate(flops)}

    for ff_name , i in flop_name_to_id.items():
        ff_data = design_data[ff_name]

        #keep track of current cluster members , initialize with the flop itself
        cluster_members = [ff_name]

        queue = deque(ff_data.get('fan_out', []))
        while queue: 
            node_name = queue.popleft()

            if node_name not in design_data: continue
            node_data = design_data[node_name]

            #if the node is a flip-flop , then current cluster is going to have an edge to that flop because they are related
            if node_data['type'] == 'flip_flop':
                neighbor_id = flop_name_to_id[node_name]
                if i != neighbor_id:
                    edge = tuple(sorted((i, neighbor_id)))
                    raw_edges.add(edge)
                continue
            

            #If a gate is already claimed by another flop , skip it , we will make a link with that flop through an edge
            if node_name in claimed_gates:
                owner_id = claimed_gates[node_name]

                if owner_id != i:
                    edge = tuple(sorted((i, owner_id)))
                    raw_edges.add(edge)
                continue
                

            if node_data['type'] == 'logic':
                claimed_gates[node_name] = i
                cluster_members.append(node_name)

                continue

        #build cluster features from cluster members
        valid_coords = []
        member_tcs = []
        for m in cluster_members:
            m_data = design_data[m]
            if 'coords' in m_data:
                valid_coords.append(m_data['coords'])
            
            tc = design_data[m]['toggle_count']
            member_tcs.append(tc)

        if valid_coords:
            arr = np.array(valid_coords)
            centroid = np.mean(arr, axis=0)
            spread = np.std(arr, axis=0)
        else:
            RuntimeError(f"No valid coordinates found for cluster rooted at {ff_name}")
        
        # Toggle Vector m = {max, sum, nonzero}
        arr_tc = np.array(member_tcs)
        tc_max = np.max(arr_tc) if len(arr_tc) > 0 else 0.0
        tc_sum = np.sum(arr_tc)
        tc_nz  = np.count_nonzero(arr_tc)
        
        gravity_center = design_data[ff_name].get('gravity_center', np.array([0.0, 0.0]))
        gravity_vector = design_data[ff_name].get('gravity_vector', np.array([0.0, 0.0]))
        control_net = ff_data.get('control_net', 'NO_RESET')
        atomic_clusters.append({
            'id': i,
            'flop_name': ff_name,
            'members': cluster_members,
            'centroid': centroid,
            'spread': spread,
            'gravity_center': gravity_center,
            'gravity_vector': gravity_vector, 
            'size': len(cluster_members),
            'control_net': control_net, 
            'toggle_feats': [tc_max, tc_sum, tc_nz]
        })



    return atomic_clusters , len(atomic_clusters) , raw_edges


atomic_clusters , num_clusters , raw_edges = form_atomic_clusters(design_data)


def merge_atomic_clusters(atomic_clusters , raw_edges , dist_limit=0.1 , gravity_alignment_threshold=0.86):
    final_clusters = []
    merge_candidates = defaultdict(list)
    
    # 1. TRAVERSE & FILTER
    for c in atomic_clusters:
        # Check Spread 
        is_high_spread = np.max(c['spread']) > 0.05
        
        if is_high_spread:
            # High spread? It goes directly to final 
            final_clusters.append(create_macro_cluster([c]))
        else:
            # Low spread? Add to the bin for its Reset Net
            # This drastically reduces search space , later we will only merge within same reset net
            net = c['control_net']
            merge_candidates[net].append(c)

    print(f"Filtered {len(final_clusters)} High-Spread Clusters.")
    print(f"Processing {len(merge_candidates)} Reset Groups...")

    merged_count = 0
    for net_name, cluster_list in merge_candidates.items():
        # Skip if only 1 cluster in this bin
        if len(cluster_list) < 2:
            final_clusters.append(create_macro_cluster(cluster_list))
            continue

        used = [False] * len(cluster_list)
        for i in range(len(cluster_list)):
                if used[i]: continue
                
                # Start a new merged group with 'i'
                current_group = [cluster_list[i]]
                used[i] = True
                
                # Look for partners for 'i'
                base_centroid = cluster_list[i]['centroid']
                base_vector = cluster_list[i]['gravity_vector']
                
                # Normalize base vector for dot product (avoid div by zero)
                norm_base = np.linalg.norm(base_vector)
                if norm_base > 0:
                    #unit vector
                    base_vector_u = base_vector / norm_base
                else:
                    base_vector_u = np.zeros(2)

                for j in range(i + 1, len(cluster_list)):
                    if used[j]: continue

                    target = cluster_list[j]
                    # --- CHECK 1: MANHATTAN DISTANCE ---
                    dist = np.sum(np.abs(base_centroid - target['centroid']))

                    if dist > dist_limit:
                        continue

                    # --- CHECK 2: GRAVITY ALIGNMENT (Cosine Similarity) ---
                    target_vector = target['gravity_vector']
                    norm_target = np.linalg.norm(target_vector)
                
                    # If either has no gravity (0,0), we assume they are compatible (neutral)
                    if norm_base > 0 and norm_target > 0:
                        target_vector_u = target_vector / norm_target
                    # Dot product of unit vectors = Cosine Similarity (-1 to 1)
                        alignment = np.dot(base_vector_u, target_vector_u)
                    
                    # If alignment is low, they are pulling apart.
                        if alignment < gravity_alignment_threshold:
                            continue

                # PASSED ALL CHECKS: MERGE
                    current_group.append(target)
                    used[j] = True
                    merged_count += 1

                # Create the final merged object from 'current_group'
                # (Logic to combine centroids, members, etc.)
                final_clusters.append(create_macro_cluster(current_group))


    print(f"Physics Merge Complete. Total Merges: {merged_count}")
    print(f"Final Macro Clusters: {len(final_clusters)}")
    return final_clusters          



def create_macro_cluster(group):
    """ Helper to combine a list of atomic clusters into one Macro Cluster """
        
    all_members = []
    all_centroids = []

    batch_sums = []
    batch_maxs = []
    batch_nzs = []
    
    # We pick the ID of the first one as the new ID (or generate new one)
    leader_id = group[0]['id']
    reset_net = group[0]['control_net']
    
    for c in group:
        all_members.extend(c['members'])
        all_centroids.append(c['centroid'])

        t_feats = c['toggle_feats']
        batch_maxs.append(t_feats[0])
        batch_sums.append(t_feats[1])
        batch_nzs.append(t_feats[2])
        
    # Recalculate Centroid
    arr = np.array(all_centroids)
    new_centroid = np.mean(arr, axis=0)
    
    # Recalculate Spread (Radius)
    # Note: simple std dev of centroids is an approximation, but fast
    if len(group) > 1:
        new_spread = np.std(arr, axis=0)
    else:
        # If it's a singleton, preserve its original spread
        new_spread = group[0]['spread']

    # Re-calc Toggle Vector m
    new_tc_max = np.max(batch_maxs)
    new_tc_sum = np.sum(batch_sums)
    new_tc_nz  = np.sum(batch_nzs) 

    atomic_ids = [c['id'] for c in group]  
    
    return {
        'id': leader_id,
        'atomic_ids': atomic_ids,
        'members': all_members,
        'centroid': new_centroid,
        'spread': new_spread,
        'size': len(all_members),
        'control_net': reset_net,
        'num_of_ff': sum(1 for m in all_members if design_data[m]['type'] == 'flip_flop'),
        'num_of_logic': sum(1 for m in all_members if design_data[m]['type'] == 'logic'),
        'toggle_feats': [new_tc_max, new_tc_sum, new_tc_nz],
        'type': 'cluster'
    }

final_clusters = merge_atomic_clusters(atomic_clusters , raw_edges , dist_limit=0.05 , gravity_alignment_threshold=0.9)
# print(final_clusters[:5])

def build_macro_edges(final_clusters, raw_edges):
    atom_to_macro = {}
    
    # 1. Create a Lookup Map: Atomic ID -> Macro Index
    for macro_idx, cluster in enumerate(final_clusters):
        if 'atomic_ids' in cluster:
            for atom_id in cluster['atomic_ids']:
                atom_to_macro[atom_id] = macro_idx
        else:
            print("!!!!!'atomic_ids' missing in cluster. Cannot map edges.")
            return None
        
    # Logic: If Atom A (in Macro 1) was connected to Atom B (in Macro 2),
    # Then Macro 1 is connected to Macro 2.
    unique_macro_edges = set()
    
    for u, v in raw_edges:
        # Check if both atoms map to a valid macro
        if u in atom_to_macro and v in atom_to_macro:
            macro_u = atom_to_macro[u]
            macro_v = atom_to_macro[v]
            
            # We only care about connections BETWEEN clusters.
            # (Internal connections are absorbed into the node features)
            if macro_u != macro_v:
                # Add edge (undirected)
                # Using a set handles duplicates automatically.
                edge = tuple(sorted((macro_u, macro_v)))
                unique_macro_edges.add(edge)
    
    print(f"  - Collapsed {len(raw_edges)} Atomic Edges -> {len(unique_macro_edges)} Macro Edges")

    return unique_macro_edges

unique_macro_edges = build_macro_edges(final_clusters , raw_edges)

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def make_final_graph(final_clusters, unique_macro_edges, output_name=f"{FILENAME}.pt"):
    
    # --- 1. Nodes (Features) ---
    node_feats = []
    centroid_lookup = {} 

    for idx, c in enumerate(final_clusters):
        cx, cy = c['centroid']
        sx, sy = c['spread']
        
        # Log-scale features
        size_log = np.log1p(c['size']) 
        n_ff = c['num_of_ff']
        n_logic = c['num_of_logic']
        tc_feats = [np.log1p(val) for val in c['toggle_feats']]

        # Feature Vector
        node_feats.append([cx, cy, sx, sy, size_log, n_ff, n_logic] + tc_feats)
        centroid_lookup[idx] = np.array([cx, cy])
        
    x = torch.tensor(node_feats, dtype=torch.float)

    # --- 2. Edges (Manhattan Distance) ---
    if len(unique_macro_edges) > 0:
        src_list, dst_list = zip(*unique_macro_edges)
        
        # Calculate Manhattan Distance for the base edges
        dists = []
        for u, v in unique_macro_edges:
            p1 = centroid_lookup[u]
            p2 = centroid_lookup[v]
            manhattan_dist = np.sum(np.abs(p1 - p2)) 
            dists.append([manhattan_dist])

        # Create Tensors
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(dists, dtype=torch.float)

        # MAGIC STEP: Make Undirected
        # 1. Double the indices (u->v AND v->u)
        edge_index = to_undirected(edge_index)
        
        # 2. Double the attributes (Distance A->B is same as B->A)
        # We simply stack the distance list on top of itself
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 1), dtype=torch.float)

    # --- 3. Save ---
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    save_dir = "dataset_root/graphs/clustered_graphs"
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{FILENAME}_clustered.pt")
    torch.save(data, output_path)

    
        
    print(f"✅ Saved Graph: {output_path}")
    print(f"   - Nodes: {data.num_nodes}")
    print(f"   - Edges: {data.num_edges}")


make_final_graph(final_clusters, unique_macro_edges, output_name=f"{FILENAME}.pt")



# import torch
# import networkx as nx
# import matplotlib.pyplot as plt
# from torch_geometric.utils import to_networkx

# def view_graph_pt(filename):
#     print(f"--- Loading {filename} ---")
#     # 1. Load Data (Safety flag added)
#     data = torch.load(filename, weights_only=False)
    
#     # 2. Convert to NetworkX
#     # We force undirected to simplify the visual
#     G = to_networkx(data, to_undirected=True, remove_self_loops=True)
    
#     print(f"NetworkX Graph Stats:")
#     print(f"  - Nodes: {G.number_of_nodes()}")
#     print(f"  - Edges: {G.number_of_edges()}") 
    
#     if G.number_of_edges() == 0:
#         print("⚠️ WARNING: The graph object has 0 edges. Check your build_and_save_tensor function!")
#         return

#     # 3. Extract Positions
#     pos = {}
#     node_sizes = []
    
#     # We color nodes by type (FF vs Logic) to make it look cooler
#     node_colors = []
    
#     for i in range(data.num_nodes):
#         x = data.x[i, 0].item()
#         y = data.x[i, 1].item()
#         pos[i] = (x, y)
        
#         # Size based on mass
#         size_val = data.x[i, 4].item()
#         node_sizes.append(20 + (size_val * 30))
        
#         # Color: if num_ff > 0 (index 5) -> Red, else Blue
#         if data.x[i, 5].item() > 0:
#             node_colors.append('#e74c3c') # Red for FF-heavy clusters
#         else:
#             node_colors.append('#3498db') # Blue for Logic clusters

#     # 4. Draw High-Contrast Plot
#     fig, ax = plt.subplots(figsize=(12, 12))
    
#     # --- CHANGE 1: Darker, Thicker Edges ---
#     nx.draw_networkx_edges(G, pos, 
#                            alpha=0.6,          # Increased from 0.1 to 0.6 (Much darker)
#                            edge_color='#555555', # Dark Grey instead of light grey
#                            width=1.0)          # Thicker lines
    
#     # Draw Nodes
#     nx.draw_networkx_nodes(G, pos, 
#                            node_size=node_sizes, 
#                            node_color=node_colors, 
#                            edgecolors='black', # Add black border to dots
#                            linewidths=0.5,
#                            alpha=0.9)

#     ax.set_title(f"Graph Visualization: {filename}\n{G.number_of_edges()} Connections Visible")
#     ax.set_xlim(-0.02, 1.02)
#     ax.set_ylim(-0.02, 1.02)
#     plt.grid(True, linestyle=':', alpha=0.3)
#     plt.show()

# # Run it
# view_graph_pt("picorv32_run_20260107_145745.pt")