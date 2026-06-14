import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import EEG_COORDS, WaveletGNN
from dataloader import load_bci_data_cv

def generate_true_gnn_attention(mat_train_path, mat_val_path, model_checkpoint_path, save_name="learned_gnn_pathways.png"):
    # 1. Load data settings
    X, y, fs, _, class_names, channel_names = load_bci_data_cv(
        mat_train_path, mat_val_path, include_validation=True
    )
    
    n_channels = X.shape[1]
    n_classes = len(class_names)

    # 2. Instantiate and load model weights
    model = WaveletGNN(n_channels=n_channels, n_classes=n_classes, channel_names=channel_names)
    if os.path.exists(model_checkpoint_path):
        model.load_state_dict(torch.load(model_checkpoint_path, map_location='cpu'))
        print(f"✅ Loaded weights from: {model_checkpoint_path}")
    else:
        print("⚠️ Checkpoint missing. Plotting template parameters.")

    model.eval()

    # 3. CRITICAL FIX: Capture real attention coefficients dynamically from the forward hook
    # PyG's GATConv stores the latest forward multi-head attention matrix during evaluation execution paths.
    # We pass a single real data trial from your X block through the node feature pipeline.
    with torch.no_grad():
        dummy_x = torch.tensor(X[:1], dtype=torch.float32) # Take 1 trial batch
        node_feats = model._node_features(dummy_x)        # [64, node_feat_dim]
        
        # Call GATConv1 layer manually with return_attention_weights=True flag activated
        edge_index_out, alpha = model.gat1(node_feats, model.edge_index, return_attention_weights=True)
        
        # Average attention weights across your 4 heads
        alpha_mean = alpha.mean(dim=1).cpu().numpy() 
        edge_index_np = edge_index_out.cpu().numpy()

    # 4. Canvas Setup
    fig, ax = plt.subplots(figsize=(9, 9))
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle=':', linewidth=1.5, alpha=0.5)
    ax.add_patch(circle)

    # 5. Plot Electrode Nodes
    valid_names = [c.strip() for c in channel_names[:64]]
    node_positions = {}
    for idx, name in enumerate(valid_names):
        if name in EEG_COORDS:
            x, y = EEG_COORDS[name]
            node_positions[idx] = (x, y)
            ax.scatter(x, y, color='#1f77b4', s=130, edgecolor='black', linewidths=0.7, zorder=4)
            ax.text(x, y + 0.03, name, fontsize=8, ha='center', va='bottom', fontweight='bold', alpha=0.7)

    # 6. Normalize attention weights for visible line scale contrast
    alpha_normalized = (alpha_mean - alpha_mean.min()) / (alpha_mean.max() - alpha_mean.min() + 1e-8)
    
    # 7. Dynamically filter and render top 15% highest attention connections across the scalp
    top_threshold = np.percentile(alpha_normalized, 85) 
    drawn_links = 0

    for idx in range(edge_index_np.shape[1]):
        u, v = edge_index_np[0, idx], edge_index_np[1, idx]
        weight = alpha_normalized[idx]
        
        if u == v: continue # Omit self loops for plot readability
        
        if weight > top_threshold and u in node_positions and v in node_positions:
            x_u, y_u = node_positions[u]
            x_v, y_v = node_positions[v]
            
            # Line thickness and alpha driven by actual model attention weights
            ax.plot([x_u, x_v], [y_u, y_v], color='#d62728', 
                    alpha=float(weight * 0.8), 
                    linewidth=float(weight * 3.5), zorder=2)
            drawn_links += 1

    ax.set_title("Learned GNN Spatial Attention Pathways\n(Dynamic Multi-Head Connection Strengths)", 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    plt.axis('off')
    plt.tight_layout()
    
    output_path = os.path.join("results", save_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 True performance graph generated. Saved to: {output_path} ({drawn_links} pathways mapped)")

if __name__ == '__main__':
    DATA_BASE = '/media/csedept/cse2018/Project/Codes/2nd approach/C1/'
    generate_true_gnn_attention(
        mat_train_path=DATA_BASE + 'Train/Data_Sample01.mat',
        mat_val_path=DATA_BASE + 'Validation/Data_Sample01.mat',
        model_checkpoint_path='checkpoints/S01_fold1.pt'
    )
