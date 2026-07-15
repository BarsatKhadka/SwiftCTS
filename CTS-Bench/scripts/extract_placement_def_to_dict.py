import re
import numpy as np


def load_file_content(filepath):
    try:
        with open(filepath, "r") as f:
            return f.read()
    except FileNotFoundError:
        return None
    
def extract_die_area(def_text):
    for line in def_text.splitlines():
        if line.startswith("DIEAREA"):
            # Extract numbers inside parentheses
            # This finds all sequences of digits
            nums = re.findall(r'\d+', line)
            if len(nums) == 4:
                die_x_min = int(nums[0])
                die_y_min = int(nums[1])
                die_x_max = int(nums[2])
                die_y_max = int(nums[3])
                return die_x_min, die_y_min, die_x_max, die_y_max
    return RuntimeError("DIEAREA not found in DEF file")


def build_connectivity_graph(def_text, all_flops):
    """
    Categorizes Fan-in and Fan-out using strict Regex.
    Maintains self-loops for future BFS traversal.
    """
    design_data = {}
    
    # 1. Initialize all known Flops
    for flop in all_flops:
        design_data[flop] = {
            "type": "flip_flop",
            "coords": None,
            "toggle_count": 0,
            "fan_in": [],
            "fan_out": []
        }

    # Regex for instance-pin pairs: ( _12345_ D )
    conn_pattern = re.compile(r'\(\s+(\S+)\s+(\S+)\s+\)')
    # Regex for target logic gates: _09295_
    logic_gate_pattern = re.compile(r'^_\d+_$')
    
    nets_start = def_text.find("NETS")
    nets_end = def_text.find("END NETS")
    nets_section = def_text[nets_start:nets_end]
    
    # 2. Process each NET block
    for net_block in nets_section.split(";"):
        if not net_block.strip(): continue
        
        connections = conn_pattern.findall(net_block)
        
        net_logic_gates = []
        net_d_flops = []
        net_q_flops = []
        
        for inst, pin in connections:
            # Case A: It's a Flip-Flop we are tracking
            if inst in all_flops:
                if pin == "D": net_d_flops.append(inst)
                elif pin == "Q": net_q_flops.append(inst)
            
            # Case B: It's a Logic Gate (matches _digits_ and isn't a PIN/Port)
            elif logic_gate_pattern.match(inst):
                net_logic_gates.append(inst)
                if inst not in design_data:
                    design_data[inst] = {"type": "logic", "coords": None, "toggle_count": 0}

        # 3. Map Connectivity
        # Logic gates on a 'D' net are Fan-In (Predecessors)
        for flop in net_d_flops:
            design_data[flop]["fan_in"].extend(net_logic_gates)
            
        # Logic gates on a 'Q' net are Fan-Out (Successors)
        for flop in net_q_flops:
            design_data[flop]["fan_out"].extend(net_logic_gates)

   
    return design_data


#saif parsing
def extract_saif_data(saif_text, design_data):
    targets = set(design_data.keys())
    
    # preferred Regex patterns
    start_pattern = re.compile(r'\(INSTANCE\s+([a-zA-Z0-9_]+)')
    tc_pattern = re.compile(r'\(TC\s+(\d+)\)')

    for match in start_pattern.finditer(saif_text):
        gate_name = match.group(1)
        if gate_name in targets:
            start_index = match.start()
            current_index = start_index
            balance = 0
            
            # Parenthesis tracking
            while current_index < len(saif_text):
                char = saif_text[current_index]
                if char == '(': balance += 1
                elif char == ')': balance -= 1
                if balance == 0: break
                current_index += 1
            
            full_block = saif_text[start_index : current_index + 1]
            max_tc = 0
            
            for line in full_block.splitlines():
                tc_match = tc_pattern.search(line)
                if tc_match:
                    # Ignore CLK
                    if "CLK" in line.upper() or "CLOCK" in line.upper(): continue
                    val = int(tc_match.group(1))
                    if val > max_tc: max_tc = val
            
            design_data[gate_name]["toggle_count"] = np.log1p(max_tc)
    return design_data

def calculate_gravity_vectors(design_data):
    """
    Calculates the 'Gravitational Pull' for each Flip-Flop.
    
    Physics:
    1. Finds all immediate neighbors (Fan-In + Fan-Out).
    2. Calculates the Average Coordinate (Centroid) of those neighbors.
    3. Calculates the Vector (dx, dy) from the FF to that Centroid.
    """
    gravity_data = {}
    
    for name, data in design_data.items():
        # Only process Flip-Flops
        if data.get('type') != 'flip_flop':
            continue

        # 1. Gather all connected logic (Inputs AND Outputs)
        neighbors = data.get('fan_in', []) + data.get('fan_out', [])
        
        neighbor_coords = []
        for n_name in neighbors:
            # We must verify the neighbor exists in our dict
            if n_name in design_data:
                neighbor_coords.append(design_data[n_name]['coords'])
        
        # 2. Calculate Average (The Center of Gravity)
        if not neighbor_coords:
            # If floating/disconnected, no gravity
            gravity_data[name] = {'vector': (0.0, 0.0), 'magnitude': 0.0}
            continue

        # Convert to numpy for easy mean calculation
        coords_array = np.array(neighbor_coords)
        avg_x = np.mean(coords_array[:, 0])
        avg_y = np.mean(coords_array[:, 1])
        
        # 3. Calculate Vector (Where is it being pulled?)
        current_x, current_y = data['coords']
        
        dx = avg_x - current_x
        dy = avg_y - current_y
        
        # Store it directly in the dictionary if you want, or a new dict
        design_data[name]['gravity_vector'] = (dx, dy)
        design_data[name]['gravity_center'] = (avg_x, avg_y)


    return design_data

def add_reset_net_to_flops(design_data, def_text):
    """
    Scans the NETS section to find which net is connected to the RESET_B pin of each Flip-Flop.
    Assigns 'NO_RESET' if no connection is found.
    This creates 'Electrical Domains' for safe clustering.
    """

    # 1. Initialize Default for ALL Flip-Flops
    for name, data in design_data.items():
        if data['type'] == 'flip_flop':
            data['control_net'] = "NO_RESET"

    # 2. Extract NETS section
    nets_start = def_text.find("NETS")
    nets_end = def_text.find("END NETS")
    if nets_start == -1:
        RuntimeError("NETS section not found in DEF file")
        return design_data

    nets_section = def_text[nets_start:nets_end]
    
    # 3. Parse NETS
    # Regex to capture: "- net_name"
    net_split_pattern = re.compile(r'-\s+(\S+)')
    
    # Split by ";" to handle each net definition separately
    # This is much faster than line-by-line for massive files
    blocks = nets_section.split(";")
    
    match_count = 0
    
    for block in blocks:
        if not block.strip(): continue

        # A. Identify the Net Name
        header_match = net_split_pattern.search(block)
        if not header_match: continue
        current_net = header_match.group(1)
        
        # B. Find instances connected to RESET_B on this net
        # We explicitly look for the string "RESET_B" in the pin position
        # Pattern captures: ( instance_name RESET_B )
        # \s+ handles spaces/newlines robustly
        connections = re.findall(r'\(\s+(\S+)\s+RESET_B\s+\)', block)
        
        for inst in connections:
            # Check if this instance is actually a Flop in our data
            # (Avoids capturing other cells that might have a RESET_B pin)
            if inst in design_data and design_data[inst]['type'] == 'flip_flop':
                design_data[inst]['control_net'] = current_net
                match_count += 1

    return design_data

#main wrapper
def process_design(filename, clock_port="clk"):
    design_name = filename.split("_")[0]
    base_path = f"./runs/{filename}"
    
    def_text = load_file_content(f"{base_path}/11-openroad-detailedplacement/{design_name}.def")
    saif_text = load_file_content(f"{base_path}/{design_name}.saif")

    x_min, y_min, x_max, y_max = extract_die_area(def_text)
    x_range = x_max - x_min
    y_range = y_max - y_min

    
    # 1. Identify Flops (Using your clock block logic)
    clock_pattern = rf'-\s+{re.escape(clock_port)}\s+\(\s+PIN\s+{re.escape(clock_port)}\s+\).*?;'
    clock_match = re.search(clock_pattern, def_text, re.DOTALL)
    all_flops = set(re.findall(r'\(\s+((?!PIN)\S+)\s+CLK\s+\)', clock_match.group(0)))
    
    # 2. Connectivity 
    design_data = build_connectivity_graph(def_text, all_flops)
    
    # 3. Coordinates
    coord_pattern = re.compile(r'-\s+(\S+)\s+.*?\(\s+(\d+)\s+(\d+)\s+\)')
    for line in def_text.splitlines():
        if "PLACED" in line:
            m = coord_pattern.search(line)
            if m and m.group(1) in design_data:
                raw_x = int(m.group(2))
                raw_y = int(m.group(3))
                
                # Normalize coordinates to [0, 1] range
                # We subtract x_min/y_min in case the die doesn't start at (0,0)
                norm_x = (raw_x - x_min) / x_range if x_range > 0 else 0
                norm_y = (raw_y - y_min) / y_range if y_range > 0 else 0
                
                design_data[m.group(1)]["coords"] = (norm_x, norm_y)
                 
    # 4. Toggles
    design_data = extract_saif_data(saif_text, design_data)

    # 5. Gravitational Pull Calculation
    design_data = calculate_gravity_vectors(design_data)

    #6 adding reset nets to flops
    design_data = add_reset_net_to_flops(design_data, def_text)
    
    return design_data 

