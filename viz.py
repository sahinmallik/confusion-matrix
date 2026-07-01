import os
import math
import matplotlib.pyplot as plt
import networkx as nx
import torch

# 1. Import graph parameters from your main network script
# (Replace 'wavelet_gnn_file' with the actual filename of your main script minus the .py)
from wavelet_gnn_file import EEG_COORDS, DIST_THRESH, build_edge_index

# 2. Import the dataset loader function from your dataloader script
from dataloader import load_bci_data_cv

def run_dataset_visualization(data_directory, n_channels=64, save_image=True):
    """
    Loads real names from the dataset and visualizes the exact 
    coordinate routing graph parsed by the GAT model blocks.
    """
    # Define absolute paths matching your environment setup
    train_mat = os.path.join(data_directory, 'Train', 'Data_Sample01.mat')
    val_mat = os.path.join(data_directory, 'Validation', 'Data_Sample01.mat')
    
    if not os.path.exists(train_mat):
        raise FileNotFoundError(f"Could not locate dataset matrix file at: {train_mat}")
        
    # Extract the live channel configuration array directly from the .mat headers
    print("Extracting live channel configurations from data repository...")
    _, _, _, _, _, dataset_channels = load_bci_data_cv(train_mat, val_mat)
    
    # Generate edge indices using the network layout code logic
    edge_index = build_edge_index(dataset_channels, n_channels)
    
    # Isolate unique coordinate linkages (filtering out directional pairs & self-loops for clean plots)
    edges = edge_index.t().tolist()
    clean_edges = [(src, dst) for src, dst in edges if src != dst and src < dst]
    
    # Initialize NetworkX graph structure
    G = nx.Graph()
    active_names = [ch.strip() for ch in dataset_channels[:n_channels]]
    
    # Populate structural positions from 10-20 circle priors
    pos = {}
    valid_nodes = []
    for i, name in enumerate(active_names):
        if name in EEG_COORDS:
            G.add_node(i, label=name)
            pos[i] = EEG_COORDS[name]
            valid_nodes.append(i)
            
    G.add_edges_from(clean_edges)
    
    # Configure Matplotlib canvas window layout
    plt.figure(figsize=(10, 10))
    
    # Overlay standard scalp hemisphere bounding ring
    scalp_contour = plt.Circle((0, 0), 1.0, color='#cbd5e1', fill=False, linewidth=2, linestyle='--')
    plt.gca().add_patch(scalp_contour)
    
    # Render network structural connections
    nx.draw_networkx_edges(G, pos, edgelist=clean_edges, edge_color='#94a3b8', alpha=0.5, width=1.0)
    
    # Render blue electrode sensor markers
    nx.draw_networkx_nodes(G, pos, nodelist=valid_nodes, node_color='#2563eb', node_size=280, alpha=0.95)
    
    # Add high-contrast white textual electrode tags inside node spaces
    node_labels = {i: active_names[i] for i in valid_nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color='white', font_weight='bold')
    
    plt.title(f"Neuro-Anatomical Coordinate Graph Routing\n(Extracted from Dataset Matrix | Threshold: {DIST_THRESH})", 
              fontsize=13, fontweight='bold', pad=25)
    
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.axis('off')
    plt.tight_layout()
    
    if save_image:
        figure_output = "live_anatomical_graph.png"
        plt.savefig(figure_output, dpi=300, bbox_inches='tight')
        print(f"✅ Real-time dataset graph plot saved successfully to: {os.path.abspath(figure_output)}")
        
    plt.show()

if __name__ == '__main__':
    # Feed your actual system directory path directly here
    BASE_PATH = '/media/csedept/cse2018/Project/Codes/2nd approach/C1/'
    
    # Run pipeline rendering execution
    run_dataset_visualization(BASE_PATH, n_channels=64)
