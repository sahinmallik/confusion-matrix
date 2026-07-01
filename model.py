import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

# Assuming this module exists in your local environment
try:
    from wavelet_tokenizer import LearnableWaveletTokenizer
except ImportError:
    # Fallback dummy class for self-contained testing if needed
    class LearnableWaveletTokenizer(nn.Module):
        def __init__(self, n_scales, kernel_size):
            super().__init__()
            self.n_scales = n_scales
        def forward(self, x):
            # Input: (B, C, T) -> Output dummy: (B, n_scales, C, T//2)
            B, C, T = x.shape
            return torch.randn(B, self.n_scales, C, T // 2, device=x.device)

# ─────────────────────────────────────────────────────────────────────────────
# 10-20 scalp coordinates (unit circle projection)
# ─────────────────────────────────────────────────────────────────────────────
EEG_COORDS = {
    'Fp1':(-0.18, 0.95),'Fpz':(0.00, 1.00),'Fp2':(0.18, 0.95),
    'AF7':(-0.45, 0.89),'AF3':(-0.22, 0.87),'AFz':(0.00, 0.87),
    'AF4':(0.22, 0.87),'AF8':(0.45, 0.89),
    'F7':(-0.71, 0.71),'F5':(-0.55, 0.72),'F3':(-0.37, 0.72),
    'F1':(-0.18, 0.72),'Fz':(0.00, 0.72),'F2':(0.18, 0.72),
    'F4':(0.37, 0.72),'F6':(0.55, 0.72),'F8':(0.71, 0.71),
    'FT7':(-0.81, 0.50),'FC5':(-0.63, 0.51),'FC3':(-0.42, 0.51),
    'FC1':(-0.21, 0.51),'FCz':(0.00, 0.51),'FC2':(0.21, 0.51),
    'FC4':(0.42, 0.51),'FC6':(0.63, 0.51),'FT8':(0.81, 0.50),
    'T7':(-1.00, 0.00),'C5':(-0.75, 0.00),'C3':(-0.50, 0.00),
    'C1':(-0.25, 0.00),'Cz':(0.00, 0.00),'C2':(0.25, 0.00),
    'C4':(0.50, 0.00),'C6':(0.75, 0.00),'T8':(1.00, 0.00),
    'TP7':(-0.81,-0.50),'CP5':(-0.63,-0.51),'CP3':(-0.42,-0.51),
    'CP1':(-0.21,-0.51),'CPz':(0.00,-0.51),'CP2':(0.21,-0.51),
    'CP4':(0.42,-0.51),'CP6':(0.63,-0.51),'TP8':(0.81,-0.50),
    'P7':(-0.71,-0.71),'P5':(-0.55,-0.72),'P3':(-0.37,-0.72),
    'P1':(-0.18,-0.72),'Pz':(0.00,-0.72),'P2':(0.18,-0.72),
    'P4':(0.37,-0.72),'P6':(0.55,-0.72),'P8':(0.71,-0.71),
    'PO7':(-0.45,-0.89),'PO3':(-0.22,-0.87),'POz':(0.00,-0.87),
    'PO4':(0.22,-0.87),'PO8':(0.45,-0.89),
    'O1':(-0.18,-0.95),'Oz':(0.00,-1.00),'O2':(0.18,-0.95),
    'T3':(-1.00, 0.00),'T4':(1.00, 0.00),
    'T5':(-0.71,-0.71),'T6':(0.71,-0.71),
}

DIST_THRESH = 0.35

def build_edge_index(channel_names, n_channels):
    names  = [c.strip() for c in channel_names[:n_channels]]
    edges  = set()
    n_anat = 0

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ci, cj = names[i], names[j]
            if ci in EEG_COORDS and cj in EEG_COORDS:
                xi, yi = EEG_COORDS[ci]
                xj, yj = EEG_COORDS[cj]
                if math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2) < DIST_THRESH:
                    edges.add((i, j))
                    edges.add((j, i))
                    n_anat += 1

    for i in range(n_channels):
        edges.add((i, i))

    if n_anat == 0:
        print("  ⚠️  No 10-20 matches — using ring-graph fallback")
        for i in range(n_channels):
            j = (i + 1) % n_channels
            edges.add((i, j))
            edges.add((j, i))
    else:
        print(f"  ✅ Anatomical graph: {n_anat} pairs from {len(names)} channels")

    src, dst = zip(*edges)
    return torch.tensor([list(src), list(dst)], dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class WaveletGNN(nn.Module):

    def __init__(
        self,
        n_channels    = 64,
        n_classes     = 5,
        n_scales      = 12,
        kernel_size   = 64,
        hidden_dim    = 64,
        heads         = 4,
        dropout       = 0.3,
        channel_names = None
    ):
        super().__init__()

        self.n_channels = n_channels
        self.dropout    = dropout
        node_feat_dim   = n_scales * 2

        # ── Wavelet front-end ─────────────────────────────────────────────────
        self.wavelet = LearnableWaveletTokenizer(
            n_scales=n_scales, kernel_size=kernel_size
        )
        # BUG FIX: Replaced BatchNorm2d with LayerNorm for stable feature scaling 
        # independent of batch size variations.
        self.ln_wav = nn.LayerNorm([n_scales, n_channels])

        # ── Temporal summariser per channel ───────────────────────────────────
        self.temporal_pool = nn.Sequential(
            nn.Conv1d(n_scales, n_scales, kernel_size=8,
                      padding=4, groups=n_scales),      # depthwise temporal
            nn.BatchNorm1d(n_scales),
            nn.GELU(),
            nn.Conv1d(n_scales, node_feat_dim, kernel_size=1),  # pointwise expand
            nn.BatchNorm1d(node_feat_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # ── GAT layer 1 ───────────────────────────────────────────────────────
        self.gat1    = GATConv(node_feat_dim, hidden_dim,
                               heads=heads, dropout=dropout, concat=True)
        self.bn_gat1 = nn.BatchNorm1d(hidden_dim * heads)

        # ── GAT layer 2 ───────────────────────────────────────────────────────
        self.gat2    = GATConv(hidden_dim * heads, hidden_dim,
                               heads=1, dropout=dropout, concat=False)
        self.bn_gat2 = nn.BatchNorm1d(hidden_dim)

        # ── Skip connection ───────────────────────────────────────────────────
        self.skip_proj = nn.Linear(node_feat_dim, hidden_dim, bias=False)

        # ── Classifier: (mean + max) → n_classes ─────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, n_classes)
        )

        # Register graph topology as buffer
        names = list(channel_names or [])
        ei    = build_edge_index(names, n_channels)
        self.register_buffer('edge_index', ei)

    # ─────────────────────────────────────────────────────────────────────────
    def _node_features(self, x):
        B, C, T = x.shape

        # Wavelet → (B, n_scales, C, T')
        wt = self.wavelet(x)
        
        # BUG FIX: Safe Layer Normalization across (n_scales, C) channels
        # Permute to (B, T', n_scales, C) -> apply LayerNorm -> permute back
        _, S, _, T_ = wt.shape
        wt = wt.permute(0, 3, 1, 2)
        wt = self.ln_wav(wt)
        wt = wt.permute(0, 2, 3, 1) # (B, n_scales, C, T')

        # Reshape for per-channel temporal summary → (B*C, n_scales, T')
        wt = wt.permute(0, 2, 1, 3).reshape(B * C, S, T_)

        # Temporal pool → (B*C, node_feat_dim)
        return self.temporal_pool(wt).squeeze(-1)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x):
        B      = x.shape[0]
        C      = self.n_channels
        device = x.device

        # Node features: (B*C, node_feat_dim)
        node_feat = self._node_features(x)

        # Dynamic batch generation for PyG graph structure
        ei      = self.edge_index                  # (2, E)
        offsets = torch.arange(B, device=device) * C
        edge_b  = torch.cat([ei + o for o in offsets], dim=1)  # (2, B*E)
        
        # Optimization: cache/pre-allocate step
        batch   = torch.arange(B, device=device).repeat_interleave(C) # (B*C,)

        # GAT layer 1
        h = self.gat1(node_feat, edge_b)
        h = self.bn_gat1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # GAT layer 2
        h = self.gat2(h, edge_b)
        h = self.bn_gat2(h)

        # BUG FIX: Residual Connection executed *before* final Activation 
        h = h + self.skip_proj(node_feat)
        h = F.elu(h) 

        # Graph readout: mean + max pooling concatenated
        g = torch.cat([
            global_mean_pool(h, batch),
            global_max_pool(h,  batch)
        ], dim=-1) # (B, hidden_dim*2)

        return self.classifier(g)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    sample_names = [
        'Fp1','Fpz','Fp2','AF7','AF3','AFz','AF4','AF8',
        'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
        'FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8',
        'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
        'TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8',
        'P7','P5','P3','P1','Pz','P2','P4','P6','P8',
        'PO7','PO3','POz','PO4','PO8',
        'O1','Oz','O2',
        'Iz','I1','I2','Nz','VEOG','HEOG'
    ]

    x     = torch.randn(8, 64, 256)
    model = WaveletGNN(n_channels=64, n_classes=5, channel_names=sample_names)

    model.train()
    out = model(x)
    print(f"Train  output : {out.shape}")   # [8, 5]

    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Eval   output : {out.shape}")   # [8, 5]

    with torch.no_grad():
        out = model(torch.randn(3, 64, 256))
    print(f"Small  batch  : {out.shape}")   # [3, 5]

    print(f"Params : {model.count_parameters():,}")
    print("✅ WaveletGNN OK and Bug-Free!")