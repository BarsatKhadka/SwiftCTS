import sys
from extract_placement_def_to_dict import process_design , extract_die_area
import random , torch

if len(sys.argv) < 2:
    print("❌ Error: Run Tag argument missing.")
    sys.exit(1)

FILENAME = sys.argv[1]
design_name = FILENAME.split("_")[0]

if design_name in ["picorv32", "aes" , "sha256"] :
    design_data = process_design(FILENAME, clock_port="clk")
elif design_name == "ethmac":
    design_data  = process_design(FILENAME, clock_port="wb_clk_i")
elif design_name == "zipdiv":
    design_data  = process_design(FILENAME, clock_port="i_clk")
sample = random.sample(list(design_data.keys()), 10)
for key in sample:
    print(key, design_data[key])
print(len(design_data))

# # Assuming your dictionary is named design_data
# target_key = "_66290_"

# if target_key in design_data:
#     print(f"{target_key}: {design_data[target_key]}")
# else:
#     print(f"Key {target_key} not found in design data.")


def build_gnn_tensors(design_data):
    # 1. Map gate names to integer IDs (0 to N-1)
    name_to_id = {name: i for i, name in enumerate(design_data.keys())}
    num_nodes = len(design_data)
    
    node_features = []
    edge_index = []
    edge_attr = []
    
    for name, data in design_data.items():
        u = name_to_id[name]
        
        # building Node Features 
        x_coord, y_coord = data['coords']
        is_ff = 1.0 if data['type'] == 'flip_flop' else 0.0
        tc = float(data['toggle_count'])
        node_features.append([x_coord, y_coord, is_ff, tc])
        
        # building Edges & Edge Features ---
        # We look at fan_out to build directed edges: current -> neighbor
        if 'fan_out' in data:
            for neighbor in data['fan_out']:
                if neighbor in name_to_id:
                    v = name_to_id[neighbor]
                    v_data = design_data[neighbor]
                    
                    # Calculate Manhattan Distance for edge attribute
                    v_coords = v_data['coords']
                    dist = abs(x_coord - v_coords[0]) + abs(y_coord - v_coords[1])
                    
                    edge_index.append([u, v])
                    edge_attr.append([dist])
                    
    # 2. Convert to Tensors
    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    
    return x, edge_index, edge_attr

x, edge_index, edge_attr = build_gnn_tensors(design_data)

from torch_geometric.data import Data
import os

# Bundling the tensors into a PyG Graph Object
graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

save_dir = "dataset_root/graphs/raw_graphs"
os.makedirs(save_dir, exist_ok=True)
save_path = f"{FILENAME}.pt"

# 3. Save the graph object
save_path = os.path.join(save_dir, f"{FILENAME}_raw.pt")  
torch.save(graph, save_path)  

def get_graph():
    return graph    


# Output will look like: Data(x=[num_nodes, 4], edge_index=[2, num_edges], edge_attr=[num_edges, 1])

# import networkx as nx       
# import matplotlib.pyplot as plt
# from torch_geometric.utils import to_networkx

# def visualize_chip_graph(graph):
#     # 1. Convert PyG graph to NetworkX for easier plotting
#     G = to_networkx(graph, to_undirected=False)
    
#     # 2. Extract positions from node features (x, y are indices 0 and 1)
#     pos = {i: (graph.x[i, 0].item(), graph.x[i, 1].item()) for i in range(graph.num_nodes)}
    
#     # 3. Identify Flip-Flops for distinct coloring
#     # Convert tensor to list of booleans for standard Python loop
#     is_ff_list = graph.x[:, 2].bool().tolist()
    
#     # FIXED LOGIC:
#     node_colors = ['red' if ff else 'skyblue' for ff in is_ff_list]
#     node_sizes = [20 if ff else 5 for ff in is_ff_list]

#     plt.figure(figsize=(12, 10))
    
#     # 4. Draw Edges (low alpha to see the nodes better)
#     nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='gray', arrows=False)
    
#     # 5. Draw Nodes
#     nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)

#     plt.title(f"Spatial Graph Visualization\n(Red = Flip-Flops, Blue = Combinational)")
#     plt.xlabel("X Coordinate")
#     plt.ylabel("Y Coordinate")
#     plt.show()
# # Run the visualization
# visualize_chip_graph(graph)

# print(f"Node Feature Matrix Shape: {x.shape}")     
# print(f"Edge Index Shape: {edge_index.shape}")      
# print(f"Edge Attribute Shape: {edge_attr.shape}")   


            

            
# sample_keys = random.sample(list(design_data.keys()), 10)


# for k in sample_keys:
#     print(k, design_data[k])