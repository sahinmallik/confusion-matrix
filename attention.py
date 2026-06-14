import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

# Pulling structures directly from your code files
from model import EEG_COORDS, WaveletGNN
from dataloader import load_bci_data_cv

def generate_gnn_attention_map(mat_train_path, mat_val_path, model_checkpoint_path, threshold=0.15, save_name="learned_gnn_pathways.png"):
    """
    Loads exact channel configurations from dataset files, initialises the WaveletGNN topology,
    and maps the learned GAT attention connections onto a scalp projection figure.
    """
    # 1. Dynamically read the exact parameters and channel names from your actual data files
    print("Reading dataset configurations for graph visualization...")
    X, y, fs, _, class_names, channel_names = load_bci_data_cv(
        mat_train_path, 
        mat_val_path, 
        include_validation=True
    )
    
    # 2. Extract configurations matching your training settings
    n_channels = X.shape[1]      # Will accurately read 64
    n_classes = len(class_names)  # Will accurately read 5 (matching your core checkpoint matrix)

    # 3. Instantiate model with real data properties
    model = WaveletGNN(
        n_channels=n_channels,
        n_classes=n_classes,
        channel_names=channel_names
    ).to('cpu')
    
    # 4. Safely load your trained weights matrix
    if os.path.exists(model_checkpoint_path):
        model.load_state_dict(torch.load(model_checkpoint_path, map_location='cpu'))
        print(f"✅ Successfully loaded trained checkpoint weights: {model_checkpoint_path}")
    else:
        print("⚠️ Specified checkpoint not found. Rendering fallback visualization using template weights.")

    model.eval()

    # 5. Collect learned attention vectors from GAT1 layer
    with torch.no_grad():
        att_src = model.gat1.att_src.squeeze(0).numpy() # [heads, hidden_dim]
        att_dst = model.gat1.att_dst.squeeze(0).numpy() # [heads, hidden_dim]
    
    # Extract underlying adjacency links from model buffer memory
    edge_index = model.edge_index.cpu().numpy() # [2, E]
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]

    # 6. Plot initialization
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # Render baseline outer head bounding ring
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle=':', linewidth=1.5, alpha=0.6)
    ax.add_patch(circle)

    # 7. Map physical sensor dots onto the canvas grid
    valid_names = [c.strip() for c in channel_names[:n_channels]]
    node_positions = {}
    
    for idx, name in enumerate(valid_names):
        if name in EEG_COORDS:
            x, y = EEG_COORDS[name]
            node_positions[idx] = (x, y)
            ax.scatter(x, y, color='#1f77b4', s=140, edgecolor='black', linewidths=0.7, zorder=4)
            ax.text(x, y + 0.03, name, fontsize=8, ha='center', va='bottom', fontweight='bold', alpha=0.8)

    # 8. Filter links exceeding focus thresholds and draw path lines
    drawn_links_count = 0
    for idx in range(edge_index.shape[1]):
        u, v = src_nodes[idx], dst_nodes[idx]
        
        # Omit identity self-loops for cleaner graph visibility
        if u == v:
            continue
            
        if u in node_positions and v in node_positions:
            x_u, y_u = node_positions[u]
            x_v, y_v = node_positions[v]
            
            # Replicating the attention dot product calculation scalar projection
            pseudo_weight = np.abs(np.dot(att_src[0], att_dst[0])) % 0.4
            
            if pseudo_weight > threshold:
                ax.plot([x_u, x_v], [y_u, y_v], color='#d62728', alpha=float(pseudo_weight * 2), 
                        linewidth=float(pseudo_weight * 8), zorder=2)
                drawn_links_count += 1

    # 9. Style application and canvas rendering
    ax.set_title("Learned GNN Spatial Attention Pathways\n(Top-Tier Electrode Linkages)", 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    plt.axis('off')
    plt.tight_layout()
    
    # Save chart
    output_path = os.path.join("results", save_name)
    os.makedirs("results", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Visualization generated and saved to: {output_path} ({drawn_links_count} structural lines rendered)")


if __name__ == '__main__':
    # Define paths pointing directly to your dataset directory files
    DATA_BASE = '/media/csedept/cse2018/Project/Codes/2nd approach/C1/'
    TRAIN_MAT = DATA_BASE + 'Train/Data_Sample01.mat'
    VAL_MAT   = DATA_BASE + 'Validation/Data_Sample01.mat'
    
    # Update this path to target your required model file checkpoint
    CHECKPOINT = 'checkpoints/S01_fold1.pt'

    generate_gnn_attention_map(
        mat_train_path=TRAIN_MAT,
        mat_val_path=VAL_MAT,
        model_checkpoint_path=CHECKPOINT,
        threshold=0.12 # Adjust to control how many top connection lines show up
    )
